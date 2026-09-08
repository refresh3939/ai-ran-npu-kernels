
























#include "kernel_operator.h"
#include "descramble.h"
using namespace AscendC;

static_assert(airan_descr::BLOCK_DIM == 4 && airan_descr::N_STREAMS == 8 &&
              airan_descr::N_DATA_SYM == 12 && airan_descr::N_SC_USED == 1596 &&
              airan_descr::QAM_SYM_STRIDE == 1600 &&
              airan_descr::N_SYM_PAD == 19200 && airan_descr::N_SLOT_MAX == 23,
              "descramble: airan_descr consts wrong (combined-build pollution?)");

namespace {

constexpr uint32_t N_STREAMS   = airan_descr::N_STREAMS;
constexpr uint32_t N_DATA_SYM  = airan_descr::N_DATA_SYM;
constexpr uint32_t N_SC_USED   = airan_descr::N_SC_USED;
constexpr uint32_t IN_SYM_STRIDE = airan_descr::QAM_SYM_STRIDE;
constexpr uint32_t N_SYM       = airan_descr::N_SYM;
constexpr uint32_t N_SYM_PAD   = airan_descr::N_SYM_PAD;

constexpr uint32_t STREAM_ELEMS = N_SYM_PAD;
constexpr uint32_t STREAM_BYTES = STREAM_ELEMS * sizeof(int16_t);
constexpr uint32_t INPUT_GUARD_ELEMS = 64;
constexpr uint32_t OUTPUT_GUARD_ELEMS = 80;
constexpr uint32_t GUARDED_INPUT_BYTES =
    (STREAM_ELEMS + INPUT_GUARD_ELEMS) * sizeof(int16_t);
constexpr uint32_t GUARDED_OUTPUT_BYTES =
    (STREAM_ELEMS + OUTPUT_GUARD_ELEMS) * sizeof(int16_t);
constexpr uint32_t UB_BYTES = GUARDED_INPUT_BYTES + STREAM_BYTES +
                              GUARDED_OUTPUT_BYTES + 64u;
static_assert(UB_BYTES <= 192u * 1024u, "descramble UB allocation exceeds dav_m200 budget");

constexpr uint32_t NSLOT_ELEMS = 16;
constexpr uint32_t NSLOT_BYTES = NSLOT_ELEMS * sizeof(int32_t);


__aicore__ inline uint32_t NRBitToQamStream(uint32_t b)
{
    return (b >> 1) + ((b & 1u) << 2);
}

}

extern "C" __global__ __aicore__ void descramble_kernel(
    GM_ADDR llr_in_gm,
    GM_ADDR sign_gm,
    GM_ADDR n_slot_gm,
    GM_ADDR llr_out_gm,
    GM_ADDR workspace,
    GM_ADDR tiling
)
{
    (void)workspace; (void)tiling;

    const uint32_t aiv_id = GetBlockIdx() ^ 2;
    if (aiv_id >= airan_descr::BLOCK_DIM) return;

    constexpr uint32_t TOTAL_IO = airan_descr::N_SLOT_MAX * N_STREAMS * N_SYM_PAD;

    GlobalTensor<int16_t> llr_in_g, sign_g, llr_out_g;
    GlobalTensor<int32_t> n_slot_g;
    llr_in_g .SetGlobalBuffer((__gm__ int16_t*) llr_in_gm,  TOTAL_IO);
    sign_g   .SetGlobalBuffer((__gm__ int16_t*) sign_gm,    TOTAL_IO);
    llr_out_g.SetGlobalBuffer((__gm__ int16_t*) llr_out_gm, TOTAL_IO);
    n_slot_g .SetGlobalBuffer((__gm__ int32_t*) n_slot_gm,  NSLOT_ELEMS);

    TPipe pipe;
    TBuf<TPosition::VECCALC> llr_buf, sgn_buf, out_buf, nslot_buf;
    pipe.InitBuffer(llr_buf,   GUARDED_INPUT_BYTES);
    pipe.InitBuffer(sgn_buf,   STREAM_BYTES);
    pipe.InitBuffer(out_buf,   GUARDED_OUTPUT_BYTES);
    pipe.InitBuffer(nslot_buf, NSLOT_BYTES);

    auto llr_i16 = llr_buf  .Get<int16_t>();
    auto sgn_i16 = sgn_buf  .Get<int16_t>();
    auto out_i16 = out_buf  .Get<int16_t>();
    auto nslot_u = nslot_buf.Get<int32_t>();


    DataCopy(nslot_u, n_slot_g, NSLOT_ELEMS);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    uint32_t n_slots = (uint32_t)nslot_u.GetValue(0);
    if (n_slots > airan_descr::N_SLOT_MAX) n_slots = airan_descr::N_SLOT_MAX;

    for (uint32_t slot = 0; slot < n_slots; ++slot) {
        const uint32_t slot_base = slot * N_STREAMS * N_SYM_PAD;


        for (uint32_t b = aiv_id; b < N_STREAMS; b += airan_descr::BLOCK_DIM) {
            const uint32_t qam_stream = NRBitToQamStream(b);
            const uint32_t in_off  = slot_base + qam_stream * N_SYM_PAD;
            const uint32_t out_off = slot_base + b * N_SYM_PAD;

            DataCopy(llr_i16, llr_in_g[in_off], STREAM_ELEMS);
            DataCopy(sgn_i16, sign_g[out_off], STREAM_ELEMS);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);







            const UnaryRepeatParams copy_params(1, 1, 8, 8);
            for (uint32_t ds = 0; ds < N_DATA_SYM; ++ds) {
                Adds<int16_t, false>(out_i16[ds * N_SC_USED],
                     llr_i16[ds * IN_SYM_STRIDE], (int16_t)0,
                     (uint64_t)128, (uint8_t)13, copy_params);
            }
            Duplicate<int16_t, false>(out_i16[N_SYM], (int16_t)0,
                                      (uint64_t)64, (uint8_t)1, 1, 8);
            const BinaryRepeatParams mul_params(1, 1, 1, 8, 8, 8);
            Mul<int16_t, false>(out_i16, out_i16, sgn_i16,
                                (uint64_t)128, (uint8_t)150, mul_params);

            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            DataCopy(llr_out_g[out_off], out_i16, STREAM_ELEMS);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}
