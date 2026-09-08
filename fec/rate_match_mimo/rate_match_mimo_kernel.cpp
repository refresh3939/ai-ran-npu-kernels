







#include "kernel_operator.h"
#include "rate_match_mimo.h"

using namespace AscendC;
namespace rmm = airan::rate_match_mimo;

namespace {
constexpr uint32_t I8_ALIGN = 32;
constexpr uint32_t I16_ALIGN = 16;

static_assert(rmm::Q_M == 8 && rmm::BLOCK_DIM == 4,
              "rate_match_mimo expects two bit planes per AIV");
static_assert(rmm::N_DATA_RE % I16_ALIGN == 0,
              "valid per-layer extent must be 32-byte aligned");
static_assert(rmm::N_DATA_PAD % I16_ALIGN == 0,
              "padded per-layer extent must be 32-byte aligned");
static_assert(rmm::N_CB_BUF % I8_ALIGN == 0,
              "encoded CB stride must be an int8 datablock multiple");
static_assert(rmm::COPY_PAD >= rmm::COPY_ELEMS + I8_ALIGN - 1,
              "aligned input staging buffer is too small");

__aicore__ inline uint32_t MinU32(uint32_t left, uint32_t right)
{
    return left < right ? left : right;
}
}

extern "C" __global__ __aicore__ void rate_match_mimo_kernel(
    GM_ADDR code_blocks_gm, GM_ADDR descriptors_gm, GM_ADDR bits_nr_gm,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm)
{
    (void)workspace_gm;
    const uint32_t core = GetBlockIdx() ^ 2u;
    if (core >= rmm::BLOCK_DIM) return;

    const auto *meta = reinterpret_cast<__gm__ uint32_t *>(tiling_gm);
    const uint32_t magic = meta[0];
    const uint32_t num_slots = meta[1];
    const uint32_t num_layers = meta[2];
    const uint32_t qm = meta[3];
    const uint32_t num_data_re = meta[4];
    const uint32_t data_stride = meta[5];
    const uint32_t codeword_symbols = meta[6];
    const uint32_t codeword_stride = meta[7];
    const uint32_t num_code_blocks = meta[8];
    const uint32_t encoded_stride = meta[9];
    const uint32_t copy_elems = meta[10];
    const uint32_t max_slots = meta[11];
    if (magic != rmm::META_MAGIC || num_slots == 0 ||
        num_slots > rmm::MAX_SLOTS || max_slots != rmm::MAX_SLOTS ||
        num_layers == 0 || num_layers > rmm::MAX_LAYERS || qm != rmm::Q_M ||
        num_data_re != rmm::N_DATA_RE || data_stride != rmm::N_DATA_PAD ||
        codeword_symbols != num_layers * rmm::N_DATA_RE ||
        codeword_stride != num_layers * rmm::N_DATA_PAD ||
        num_code_blocks != rmm::C_NUM || encoded_stride != rmm::N_CB_BUF ||
        copy_elems != rmm::COPY_ELEMS) {
        return;
    }

    const uint32_t input_elems = num_code_blocks * encoded_stride;
    const uint32_t output_elems = num_slots * qm * codeword_stride;
    GlobalTensor<int8_t> codeBlocksG;
    GlobalTensor<uint32_t> descriptorsG;
    GlobalTensor<int16_t> bitsNrG;
    codeBlocksG.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(code_blocks_gm),
                                input_elems);
    descriptorsG.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t *>(descriptors_gm),
        num_code_blocks * rmm::DESC_WORDS);
    bitsNrG.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(bits_nr_gm),
                            output_elems);

    TPipe pipe;
    TBuf<TPosition::VECCALC> slot_buf, input16_buf, input8_buf;
    pipe.InitBuffer(slot_buf,
                    rmm::MAX_CODEWORD_STRIDE * sizeof(int16_t));
    pipe.InitBuffer(input16_buf, rmm::COPY_PAD * sizeof(int16_t));
    pipe.InitBuffer(input8_buf, rmm::COPY_PAD * sizeof(int8_t));
    auto slotBits = slot_buf.Get<int16_t>();
    auto inputI16 = input16_buf.Get<int16_t>();
    auto inputHalf = input16_buf.Get<half>();
    auto inputI8 = input8_buf.Get<int8_t>();

    DataCopyParams write_params;
    write_params.blockCount = 1;
    write_params.blockLen = static_cast<uint16_t>(codeword_stride / I16_ALIGN);
    write_params.srcStride = 0;
    write_params.dstStride = 0;
    const uint32_t tail = codeword_stride - codeword_symbols;

    for (uint32_t owned = 0; owned < rmm::STREAMS_PER_AIV; ++owned) {
        const uint32_t bit = core + owned * rmm::BLOCK_DIM;
        uint64_t plane_position = 0;
        for (uint32_t cb = 0; cb < num_code_blocks; ++cb) {
            const uint32_t desc_base = cb * rmm::DESC_WORDS;
            const uint32_t e = descriptorsG.GetValue(desc_base);
            const uint32_t k0 = descriptorsG.GetValue(desc_base + 1u);
            const uint32_t ncb = descriptorsG.GetValue(desc_base + 2u);
            const uint32_t cw_offset = descriptorsG.GetValue(desc_base + 3u);
            const uint32_t eq = e / qm;

            uint32_t i = 0;
            while (i < eq) {
                const uint64_t logical = plane_position + i;
                const uint32_t slot =
                    static_cast<uint32_t>(logical / codeword_symbols);
                const uint32_t re =
                    static_cast<uint32_t>(logical % codeword_symbols);
                const uint32_t source = (k0 + bit * eq + i) % ncb;
                uint32_t count = MinU32(eq - i, codeword_symbols - re);
                count = MinU32(count, ncb - source);
                count = MinU32(count, copy_elems);

                const uint32_t source_delta = source & (I8_ALIGN - 1u);
                const uint32_t source_aligned = source - source_delta;
                const uint32_t copy_count =
                    (source_delta + count + I8_ALIGN - 1u) &
                    ~(I8_ALIGN - 1u);
                DataCopy(inputI8, codeBlocksG[cw_offset + source_aligned],
                         copy_count);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(inputHalf, inputI8, RoundMode::CAST_NONE, copy_count);
                Cast(inputI16, inputHalf, RoundMode::CAST_RINT, copy_count);
                PipeBarrier<PIPE_V>();
                Adds(slotBits[re], inputI16[source_delta],
                     static_cast<int16_t>(0), static_cast<int32_t>(count));
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                i += count;
                if (re + count == codeword_symbols) {
                    Duplicate(slotBits[codeword_symbols],
                              static_cast<int16_t>(0), tail);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                    const uint64_t output_base =
                        static_cast<uint64_t>(slot) * qm * codeword_stride +
                        static_cast<uint64_t>(bit) * codeword_stride;
                    DataCopy(bitsNrG[output_base], slotBits, write_params);
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                }
            }
            plane_position += eq;
        }
    }
}
