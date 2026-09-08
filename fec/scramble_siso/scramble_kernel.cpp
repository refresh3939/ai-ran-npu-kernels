











#include "kernel_operator.h"
#include "scramble.h"

using namespace AscendC;

static_assert(airan_scr::BLOCK_DIM == 4 && airan_scr::N_STREAMS == 8 &&
              airan_scr::N_DATA_SYM == 12 && airan_scr::N_SC_USED == 1596 &&
              airan_scr::QAM_SYM_STRIDE == 1600 &&
              airan_scr::N_SYM_PAD == 19200 && airan_scr::N_SLOT_MAX == 23,
              "scramble constants do not match the QAM ABI");

namespace {
constexpr uint32_t N_STREAMS = airan_scr::N_STREAMS;
constexpr uint32_t N_DATA_SYM = airan_scr::N_DATA_SYM;
constexpr uint32_t N_SC_USED = airan_scr::N_SC_USED;
constexpr uint32_t QAM_SYM_STRIDE = airan_scr::QAM_SYM_STRIDE;
constexpr uint32_t N_SYM_PAD = airan_scr::N_SYM_PAD;
constexpr uint32_t STREAM_BYTES = N_SYM_PAD * sizeof(int16_t);
constexpr uint32_t NSLOT_ELEMS = 16;
constexpr uint32_t NSLOT_BYTES = NSLOT_ELEMS * sizeof(int32_t);

__aicore__ inline uint32_t QamStreamToNRBit(uint32_t qam_stream)
{
    return (qam_stream < 4u) ? (qam_stream << 1) : (((qam_stream - 4u) << 1) + 1u);
}
}

extern "C" __global__ __aicore__ void scramble_kernel(
    GM_ADDR bits_in_gm,
    GM_ADDR gold_gm,
    GM_ADDR n_slot_gm,
    GM_ADDR bits_out_gm,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    (void)tiling;

    const uint32_t aiv_id = GetBlockIdx() ^ 2u;
    if (aiv_id >= airan_scr::BLOCK_DIM) return;

    constexpr uint32_t TOTAL_IO = airan_scr::N_SLOT_MAX * N_STREAMS * N_SYM_PAD;
    GlobalTensor<int16_t> bits_in_g, gold_g, bits_out_g;
    GlobalTensor<int32_t> n_slot_g;
    bits_in_g.SetGlobalBuffer((__gm__ int16_t*)bits_in_gm, TOTAL_IO);
    gold_g.SetGlobalBuffer((__gm__ int16_t*)gold_gm, TOTAL_IO);
    bits_out_g.SetGlobalBuffer((__gm__ int16_t*)bits_out_gm, TOTAL_IO);
    n_slot_g.SetGlobalBuffer((__gm__ int32_t*)n_slot_gm, NSLOT_ELEMS);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bit_buf, gold_buf, out_buf, nslot_buf;
    pipe.InitBuffer(bit_buf, STREAM_BYTES);
    pipe.InitBuffer(gold_buf, STREAM_BYTES);
    pipe.InitBuffer(out_buf, STREAM_BYTES);
    pipe.InitBuffer(nslot_buf, NSLOT_BYTES);

    auto bit_i16 = bit_buf.Get<int16_t>();
    auto gold_i16 = gold_buf.Get<int16_t>();
    auto out_i16 = out_buf.Get<int16_t>();
    auto nslot_i32 = nslot_buf.Get<int32_t>();

    DataCopy(nslot_i32, n_slot_g, NSLOT_ELEMS);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    uint32_t n_slots = (uint32_t)nslot_i32.GetValue(0);
    if (n_slots > airan_scr::N_SLOT_MAX) n_slots = airan_scr::N_SLOT_MAX;

    for (uint32_t slot = 0; slot < n_slots; ++slot) {
        const uint32_t slot_base = slot * N_STREAMS * N_SYM_PAD;


        for (uint32_t qam_stream = aiv_id;
             qam_stream < N_STREAMS;
             qam_stream += airan_scr::BLOCK_DIM) {
            const uint32_t nr_bit = QamStreamToNRBit(qam_stream);
            const uint32_t in_off = slot_base + nr_bit * N_SYM_PAD;
            const uint32_t out_off = slot_base + qam_stream * N_SYM_PAD;

            DataCopy(bit_i16, bits_in_g[in_off], N_SYM_PAD);
            DataCopy(gold_i16, gold_g[in_off], N_SYM_PAD);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);


            Sub(bit_i16, bit_i16, gold_i16, N_SYM_PAD);
            Muls(gold_i16, bit_i16, (int16_t)-1, N_SYM_PAD);
            Max(bit_i16, bit_i16, gold_i16, N_SYM_PAD);


            Duplicate(out_i16, (int16_t)0, N_SYM_PAD);
            for (uint32_t ds = 0; ds < N_DATA_SYM; ++ds) {
                Adds(out_i16[ds * QAM_SYM_STRIDE],
                     bit_i16[ds * N_SC_USED], (int16_t)0, N_SC_USED);
            }
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            DataCopy(bits_out_g[out_off], out_i16, N_SYM_PAD);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}
