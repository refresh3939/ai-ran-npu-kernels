














#include "kernel_operator.h"
#include "llr_assemble.h"

using namespace AscendC;

namespace {
using namespace airan;
constexpr uint32_t UB_BYTES = UB_MAX_RUN * sizeof(int16_t);
}

extern "C" __global__ __aicore__ void llr_assemble_kernel(
    GM_ADDR llr_in_gm,
    GM_ADDR llr_out_gm,
    GM_ADDR ws_gm,
    GM_ADDR tiling_gm)
{
    (void)ws_gm; (void)tiling_gm;

    const uint32_t aiv_id = GetBlockIdx() ^ 2;
    if (aiv_id >= BLOCK_DIM) return;

    GlobalTensor<int16_t> srcG, dstG;
    srcG.SetGlobalBuffer((__gm__ int16_t*) llr_in_gm,
                         (uint64_t)N_SLOT_IN * N_STREAMS * SLOT_PAD);
    dstG.SetGlobalBuffer((__gm__ int16_t*) llr_out_gm,
                         (uint64_t)N_TILE * N_STREAMS * TILE_PAD);

    TPipe pipe;
    TBuf<TPosition::VECCALC> ubBuf;
    pipe.InitBuffer(ubBuf, UB_BYTES);
    auto ub = ubBuf.Get<int16_t>();


    for (uint32_t b = aiv_id; b < N_STREAMS; b += BLOCK_DIM) {
        uint32_t c = 0;
        while (c < CW_LEN) {

            const uint32_t s = c / SLOT_VALID;
            const uint32_t p = c - s * SLOT_VALID;
            const uint32_t t = c / TILE_VALID;
            const uint32_t q = c - t * TILE_VALID;


            uint32_t run = (s + 1) * SLOT_VALID - c;
            const uint32_t rdst = (t + 1) * TILE_VALID - c;
            if (rdst < run)         run = rdst;
            if (CW_LEN - c < run)   run = CW_LEN - c;

            const uint64_t src_off = (uint64_t)s * N_STREAMS * SLOT_PAD
                                   + (uint64_t)b * SLOT_PAD + p;
            const uint64_t dst_off = (uint64_t)t * N_STREAMS * TILE_PAD
                                   + (uint64_t)b * TILE_PAD + q;


            DataCopy(ub, srcG[src_off], run);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);


            Adds(ub, ub, (int16_t)0, run);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);


            DataCopy(dstG[dst_off], ub, run);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

            c += run;
        }
    }

}
