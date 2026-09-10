/** Device adapters for the natural MIMO boundary and fixed-width detector ABI. */
#include "kernel_operator.h"
#include "channel_est_lmmse.h"

using namespace AscendC;
using namespace airan::channel_est_lmmse;

namespace {
constexpr uint32_t PACK_TILE = N_PILOT_PAD * NCOL;
}

extern "C" __global__ __aicore__ void channel_est_lmmse_pack_natural_kernel(
    GM_ADDR natural_re, GM_ADDR natural_im, GM_ADDR gather_index,
    GM_ADDR private_re, GM_ADDR private_im, GM_ADDR private_neg_im,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    (void)tiling;
    const uint32_t block = GetBlockIdx();
    GlobalTensor<half> naturalReG, naturalImG, privateReG, privateImG, privateNegImG;
    GlobalTensor<uint32_t> indexG;
    naturalReG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(natural_re), NATURAL_HLS_ELEMS);
    naturalImG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(natural_im), NATURAL_HLS_ELEMS);
    privateReG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(private_re), HLS_ELEMS);
    privateImG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(private_im), HLS_ELEMS);
    privateNegImG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(private_neg_im), HLS_ELEMS);
    indexG.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather_index), PACK_INDEX_ELEMS);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bSrc, bDst, bIndex;
    pipe.InitBuffer(bSrc, PACK_TILE * sizeof(half));
    pipe.InitBuffer(bDst, PACK_TILE * sizeof(half));
    pipe.InitBuffer(bIndex, PACK_TILE * sizeof(uint32_t));
    auto src = bSrc.Get<half>();
    auto dst = bDst.Get<half>();
    auto index = bIndex.Get<uint32_t>();
    constexpr uint32_t INDEX_CHUNK = PACK_TILE / 4;
    for (uint32_t chunk = 0; chunk < 4; ++chunk) {
        DataCopy(index[chunk * INDEX_CHUNK], indexG[chunk * INDEX_CHUNK], INDEX_CHUNK);
    }
    auto indexEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(indexEvent);
    WaitFlag<HardEvent::MTE2_V>(indexEvent);

    for (uint32_t layer = block; layer < NL; layer += BLOCK_DIM) {
        for (uint32_t group = 0; group < N_RX_GROUP; ++group) {
            for (uint32_t localRx = 0; localRx < RX_GROUP; ++localRx) {
                const uint32_t rx = group * RX_GROUP + localRx;
                for (uint32_t dmrs = 0; dmrs < N_DMRS_SYMBOL; ++dmrs) {
                    const uint32_t column = N_DMRS_SYMBOL * localRx + dmrs;
                    const size_t offset = ((static_cast<size_t>(rx) * NL + layer) *
                                           N_DMRS_SYMBOL + dmrs) * N_PILOT_PAD;
                    DataCopy(src[column * N_PILOT_PAD], naturalReG[offset], N_PILOT_PAD);
                }
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Gather(dst, src, index, static_cast<uint32_t>(0), PACK_TILE);
            PipeBarrier<PIPE_V>();
            const size_t privateOffset =
                (static_cast<size_t>(layer) * N_RX_GROUP + group) * PACK_TILE;
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            DataCopy(privateReG[privateOffset], dst, PACK_TILE / 2);
            DataCopy(privateReG[privateOffset + PACK_TILE / 2], dst[PACK_TILE / 2], PACK_TILE / 2);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

            for (uint32_t localRx = 0; localRx < RX_GROUP; ++localRx) {
                const uint32_t rx = group * RX_GROUP + localRx;
                for (uint32_t dmrs = 0; dmrs < N_DMRS_SYMBOL; ++dmrs) {
                    const uint32_t column = N_DMRS_SYMBOL * localRx + dmrs;
                    const size_t offset = ((static_cast<size_t>(rx) * NL + layer) *
                                           N_DMRS_SYMBOL + dmrs) * N_PILOT_PAD;
                    DataCopy(src[column * N_PILOT_PAD], naturalImG[offset], N_PILOT_PAD);
                }
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Gather(dst, src, index, static_cast<uint32_t>(0), PACK_TILE);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            DataCopy(privateImG[privateOffset], dst, PACK_TILE / 2);
            DataCopy(privateImG[privateOffset + PACK_TILE / 2], dst[PACK_TILE / 2], PACK_TILE / 2);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            Muls(dst, dst, static_cast<half>(-1.0f), PACK_TILE);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            DataCopy(privateNegImG[privateOffset], dst, PACK_TILE / 2);
            DataCopy(privateNegImG[privateOffset + PACK_TILE / 2], dst[PACK_TILE / 2], PACK_TILE / 2);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

extern "C" __global__ __aicore__ void channel_est_lmmse_pad_layers_kernel(
    GM_ADDR active_re, GM_ADDR active_im, GM_ADDR padded_re, GM_ADDR padded_im,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    (void)tiling;
    const uint32_t block = GetBlockIdx();
    GlobalTensor<half> activeReG, activeImG, paddedReG, paddedImG;
    activeReG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(active_re), OUT_ELEMS);
    activeImG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(active_im), OUT_ELEMS);
    paddedReG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(padded_re), PADDED_OUT_ELEMS);
    paddedImG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(padded_im), PADDED_OUT_ELEMS);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bRe, bIm;
    pipe.InitBuffer(bRe, N_SC_PAD * sizeof(half));
    pipe.InitBuffer(bIm, N_SC_PAD * sizeof(half));
    auto re = bRe.Get<half>();
    auto im = bIm.Get<half>();
    for (uint32_t task = block; task < NR * NL; task += BLOCK_DIM) {
        const uint32_t rx = task / NL;
        const uint32_t layer = task % NL;
        for (uint32_t symbol = 0; symbol < N_SYMBOL; ++symbol) {
            const size_t src = ((static_cast<size_t>(rx) * NL + layer) * N_SYMBOL + symbol) * N_SC_PAD;
            const size_t dst = ((static_cast<size_t>(rx) * DETECTOR_LAYERS + layer) *
                                N_SYMBOL + symbol) * N_SC_PAD;
            DataCopy(re, activeReG[src], N_SC_PAD);
            DataCopy(im, activeImG[src], N_SC_PAD);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopy(paddedReG[dst], re, N_SC_PAD);
            DataCopy(paddedImG[dst], im, N_SC_PAD);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}
