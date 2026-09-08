






#include "kernel_operator.h"
#include "scramble_mimo.h"

using namespace AscendC;
namespace smm = airan::scramble_mimo;

namespace {
constexpr uint32_t TILE_BYTES = smm::TILE_ELEMS * sizeof(int16_t);
static_assert(smm::Q_M == 8 && smm::BLOCK_DIM == 4,
              "scramble_mimo expects two QAM streams per AIV");
static_assert(smm::N_DATA_RE % smm::TILE_ELEMS == 0,
              "native tile must divide one layer's data RE count");

__aicore__ inline uint32_t QamStreamToNrBit(uint32_t qam_stream)
{
    return qam_stream < 4u ? (qam_stream << 1u)
                           : (((qam_stream - 4u) << 1u) + 1u);
}
}

extern "C" __global__ __aicore__ void scramble_mimo_kernel(
    GM_ADDR bits_nr_gm, GM_ADDR gold_gm, GM_ADDR bits_qam_gm,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm)
{
    (void)workspace_gm;
    const uint32_t core = GetBlockIdx() ^ 2u;
    if (core >= smm::BLOCK_DIM) return;

    const auto *meta = reinterpret_cast<__gm__ uint32_t *>(tiling_gm);
    const uint32_t magic = meta[0];
    const uint32_t num_slots = meta[1];
    const uint32_t num_layers = meta[2];
    const uint32_t qm = meta[3];
    const uint32_t num_data_re = meta[4];
    const uint32_t data_stride = meta[5];
    const uint32_t codeword_symbols = meta[6];
    const uint32_t codeword_stride = meta[7];
    const uint32_t tile_elems = meta[8];
    const uint32_t max_slots = meta[9];
    if (magic != smm::META_MAGIC || num_slots == 0 ||
        num_slots > smm::MAX_SLOTS || max_slots != smm::MAX_SLOTS ||
        num_layers == 0 || num_layers > smm::MAX_LAYERS || qm != smm::Q_M ||
        num_data_re != smm::N_DATA_RE || data_stride != smm::N_DATA_PAD ||
        codeword_symbols != num_layers * smm::N_DATA_RE ||
        codeword_stride != num_layers * smm::N_DATA_PAD ||
        tile_elems != smm::TILE_ELEMS) {
        return;
    }

    const uint32_t total_elems = num_slots * qm * codeword_stride;
    GlobalTensor<int16_t> bits_nr_g, gold_g, bits_qam_g;
    bits_nr_g.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(bits_nr_gm),
                             total_elems);
    gold_g.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(gold_gm),
                          total_elems);
    bits_qam_g.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(bits_qam_gm),
                              total_elems);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bit_buf, gold_buf;
    pipe.InitBuffer(bit_buf, TILE_BYTES);
    pipe.InitBuffer(gold_buf, TILE_BYTES);
    auto bits = bit_buf.Get<int16_t>();
    auto gold = gold_buf.Get<int16_t>();

    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        const uint32_t slot_base = slot * qm * codeword_stride;
        for (uint32_t pair = 0; pair < 2; ++pair) {
            const uint32_t qam_stream = core + pair * smm::BLOCK_DIM;
            const uint32_t nr_bit = QamStreamToNrBit(qam_stream);
            const uint32_t input_base = slot_base + nr_bit * codeword_stride;
            const uint32_t output_base = slot_base + qam_stream * codeword_stride;

            for (uint32_t base = 0; base < codeword_symbols;
                 base += smm::TILE_ELEMS) {
                DataCopy(bits, bits_nr_g[input_base + base], smm::TILE_ELEMS);
                DataCopy(gold, gold_g[input_base + base], smm::TILE_ELEMS);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                Sub(bits, bits, gold, smm::TILE_ELEMS);
                Muls(gold, bits, static_cast<int16_t>(-1), smm::TILE_ELEMS);
                Max(bits, bits, gold, smm::TILE_ELEMS);

                SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                DataCopy(bits_qam_g[output_base + base], bits, smm::TILE_ELEMS);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }

            const uint32_t tail = codeword_stride - codeword_symbols;
            SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            Duplicate(bits, static_cast<int16_t>(0), tail);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            DataCopy(bits_qam_g[output_base + codeword_symbols], bits, tail);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}
