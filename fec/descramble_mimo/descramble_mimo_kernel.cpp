










#include "kernel_operator.h"
#include "descramble_mimo.h"

using namespace AscendC;
namespace dmm = airan::descramble_mimo;

namespace {
constexpr uint32_t TILE_BYTES = dmm::TILE_ELEMS * sizeof(int16_t);
constexpr uint32_t UB_BYTES =
    dmm::META_WORDS * sizeof(uint32_t) + 2 * TILE_BYTES;
static_assert(UB_BYTES <= 192u * 1024u,
              "descramble_mimo UB working set exceeds dav_m200 capacity");

__aicore__ inline uint32_t NRBitToQamStream(uint32_t bit)
{
    return (bit >> 1) + ((bit & 1u) << 2);
}
}

extern "C" __global__ __aicore__ void descramble_mimo_kernel(
    GM_ADDR llr_qam_gm, GM_ADDR sign_gm, GM_ADDR llr_nr_gm,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm)
{
    (void)workspace_gm;
    const uint32_t core = GetBlockIdx() ^ 2;
    if (core >= dmm::BLOCK_DIM) return;

    TPipe pipe;
    TBuf<TPosition::VECCALC> meta_buf, llr_buf, sign_buf;
    pipe.InitBuffer(meta_buf, dmm::META_WORDS * sizeof(uint32_t));
    pipe.InitBuffer(llr_buf, TILE_BYTES);
    pipe.InitBuffer(sign_buf, TILE_BYTES);

    GlobalTensor<uint32_t> tiling_g;
    tiling_g.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling_gm),
                            dmm::META_WORDS);
    auto meta = meta_buf.Get<uint32_t>();
    DataCopy(meta, tiling_g, dmm::META_WORDS);
    PipeBarrier<PIPE_ALL>();

    if (meta.GetValue(0) != dmm::META_MAGIC) return;
    const uint32_t num_slots = meta.GetValue(1);
    const uint32_t num_layers = meta.GetValue(2);
    const uint32_t qm = meta.GetValue(3);
    const uint32_t num_data_re = meta.GetValue(4);
    const uint32_t data_stride = meta.GetValue(5);
    const uint32_t codeword_symbols = meta.GetValue(6);
    const uint32_t codeword_stride = meta.GetValue(7);
    const uint32_t tile_elems = meta.GetValue(8);
    const uint32_t max_slots = meta.GetValue(9);
    if (num_slots == 0 || num_slots > dmm::MAX_SLOTS ||
        max_slots != dmm::MAX_SLOTS || num_layers == 0 ||
        num_layers > dmm::MAX_LAYERS || qm != dmm::Q_M ||
        num_data_re != dmm::N_DATA_RE || data_stride != dmm::N_DATA_PAD ||
        codeword_symbols != num_layers * dmm::N_DATA_RE ||
        codeword_stride != num_layers * dmm::N_DATA_PAD ||
        tile_elems != dmm::TILE_ELEMS) return;

    const uint32_t total_elems = num_slots * qm * codeword_stride;
    GlobalTensor<int16_t> llr_qam_g, sign_g, llr_nr_g;
    llr_qam_g.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(llr_qam_gm),
                              total_elems);
    sign_g.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(sign_gm), total_elems);
    llr_nr_g.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(llr_nr_gm),
                             total_elems);
    auto llr = llr_buf.Get<int16_t>();
    auto sign = sign_buf.Get<int16_t>();


    for (uint32_t bit = core; bit < qm; bit += dmm::BLOCK_DIM) {
        const uint32_t qam_stream = NRBitToQamStream(bit);
        for (uint32_t slot = 0; slot < num_slots; ++slot) {
            const uint32_t slot_base = slot * qm * codeword_stride;
            const uint32_t input_base = slot_base + qam_stream * codeword_stride;
            const uint32_t output_base = slot_base + bit * codeword_stride;

            for (uint32_t offset = 0; offset < codeword_symbols;
                 offset += dmm::TILE_ELEMS) {
                const uint32_t count =
                    offset + dmm::TILE_ELEMS <= codeword_symbols
                        ? dmm::TILE_ELEMS : codeword_symbols - offset;
                DataCopy(llr, llr_qam_g[input_base + offset], count);
                DataCopy(sign, sign_g[output_base + offset], count);
                PipeBarrier<PIPE_ALL>();
                Mul(llr, llr, sign, count);
                PipeBarrier<PIPE_ALL>();
                DataCopy(llr_nr_g[output_base + offset], llr, count);
                PipeBarrier<PIPE_ALL>();
            }

            const uint32_t tail = codeword_stride - codeword_symbols;
            Duplicate(llr, static_cast<int16_t>(0), tail);
            PipeBarrier<PIPE_ALL>();
            DataCopy(llr_nr_g[output_base + codeword_symbols], llr, tail);
            PipeBarrier<PIPE_ALL>();
        }
    }
}
