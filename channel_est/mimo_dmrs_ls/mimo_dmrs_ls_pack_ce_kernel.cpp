/** Pack 798-point natural h_ls into channel_est_lmmse_mimo's private ABI. */
#include "kernel_operator.h"
#include "mimo_dmrs_ls.h"

using namespace AscendC;
using namespace airan::mimo_dmrs_ls;

namespace {
constexpr uint32_t PACK_TILE = N_PILOT_PAD * CE_COLUMNS;
}

extern "C" __global__ __aicore__ void mimo_dmrs_ls_pack_ce_kernel(
    GM_ADDR natural_re, GM_ADDR natural_im, GM_ADDR gather_index,
    GM_ADDR ce_re, GM_ADDR ce_im, GM_ADDR ce_neg_im,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    const uint32_t layer = GetBlockIdx();
    GlobalTensor<uint32_t> metaG;
    metaG.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling), META_WORDS);
    TPipe pipe;
    TBuf<TPosition::VECCALC> bMeta, bSrc, bDst, bIndex;
    pipe.InitBuffer(bMeta, META_WORDS * sizeof(uint32_t));
    pipe.InitBuffer(bSrc, PACK_TILE * sizeof(half));
    pipe.InitBuffer(bDst, PACK_TILE * sizeof(half));
    pipe.InitBuffer(bIndex, PACK_TILE * sizeof(uint32_t));
    auto meta = bMeta.Get<uint32_t>();
    DataCopy(meta, metaG, META_WORDS);
    auto metaEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(metaEvent); WaitFlag<HardEvent::MTE2_S>(metaEvent);
    const uint32_t nl = meta.GetValue(2);
    if (layer >= nl || layer >= MAX_LAYERS) return;

    const size_t naturalElems = static_cast<size_t>(NR_CURRENT) * nl *
                                CURRENT_DMRS_SYMBOLS * N_PILOT_PAD;
    GlobalTensor<half> naturalReG, naturalImG, ceReG, ceImG, ceNegImG;
    GlobalTensor<uint32_t> indexG;
    naturalReG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(natural_re), naturalElems);
    naturalImG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(natural_im), naturalElems);
    ceReG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ce_re), CE_PACKED_ELEMS);
    ceImG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ce_im), CE_PACKED_ELEMS);
    ceNegImG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ce_neg_im), CE_PACKED_ELEMS);
    indexG.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather_index), PACK_TILE);

    auto src = bSrc.Get<half>();
    auto dst = bDst.Get<half>();
    auto index = bIndex.Get<uint32_t>();
    constexpr uint32_t INDEX_CHUNK = PACK_TILE / 4;
    for (uint32_t chunk = 0; chunk < 4; ++chunk) {
        DataCopy(index[chunk * INDEX_CHUNK], indexG[chunk * INDEX_CHUNK], INDEX_CHUNK);
    }
    auto indexEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(indexEvent); WaitFlag<HardEvent::MTE2_V>(indexEvent);

    for (uint32_t group = 0; group < CE_RX_GROUPS; ++group) {
        for (uint32_t localRx = 0; localRx < CE_RX_GROUP; ++localRx) {
            const uint32_t rx = group * CE_RX_GROUP + localRx;
            for (uint32_t dmrs = 0; dmrs < CURRENT_DMRS_SYMBOLS; ++dmrs) {
                const uint32_t column = 2u * localRx + dmrs;
                const size_t naturalOffset = ((static_cast<size_t>(rx) * nl + layer) *
                                              CURRENT_DMRS_SYMBOLS + dmrs) * N_PILOT_PAD;
                DataCopy(src[column * N_PILOT_PAD], naturalReG[naturalOffset], N_PILOT_PAD);
            }
        }
        auto loadRe = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(loadRe); WaitFlag<HardEvent::MTE2_V>(loadRe);
        Gather(dst, src, index, static_cast<uint32_t>(0), PACK_TILE);
        PipeBarrier<PIPE_V>();
        const size_t ceOffset = (static_cast<size_t>(layer) * CE_RX_GROUPS + group) * PACK_TILE;
        auto writeRe = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(writeRe); WaitFlag<HardEvent::V_MTE3>(writeRe);
        DataCopy(ceReG[ceOffset], dst, PACK_TILE / 2);
        DataCopy(ceReG[ceOffset + PACK_TILE / 2], dst[PACK_TILE / 2], PACK_TILE / 2);
        auto reuseRe = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(reuseRe); WaitFlag<HardEvent::MTE3_MTE2>(reuseRe);

        for (uint32_t localRx = 0; localRx < CE_RX_GROUP; ++localRx) {
            const uint32_t rx = group * CE_RX_GROUP + localRx;
            for (uint32_t dmrs = 0; dmrs < CURRENT_DMRS_SYMBOLS; ++dmrs) {
                const uint32_t column = 2u * localRx + dmrs;
                const size_t naturalOffset = ((static_cast<size_t>(rx) * nl + layer) *
                                              CURRENT_DMRS_SYMBOLS + dmrs) * N_PILOT_PAD;
                DataCopy(src[column * N_PILOT_PAD], naturalImG[naturalOffset], N_PILOT_PAD);
            }
        }
        auto loadIm = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(loadIm); WaitFlag<HardEvent::MTE2_V>(loadIm);
        Gather(dst, src, index, static_cast<uint32_t>(0), PACK_TILE);
        PipeBarrier<PIPE_V>();
        auto writeIm = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(writeIm); WaitFlag<HardEvent::V_MTE3>(writeIm);
        DataCopy(ceImG[ceOffset], dst, PACK_TILE / 2);
        DataCopy(ceImG[ceOffset + PACK_TILE / 2], dst[PACK_TILE / 2], PACK_TILE / 2);
        auto imToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(imToV); WaitFlag<HardEvent::MTE3_V>(imToV);
        Muls(dst, dst, static_cast<half>(-1.0f), PACK_TILE); PipeBarrier<PIPE_V>();
        auto writeNeg = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(writeNeg); WaitFlag<HardEvent::V_MTE3>(writeNeg);
        DataCopy(ceNegImG[ceOffset], dst, PACK_TILE / 2);
        DataCopy(ceNegImG[ceOffset + PACK_TILE / 2], dst[PACK_TILE / 2], PACK_TILE / 2);
        auto reuse = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(reuse); WaitFlag<HardEvent::MTE3_MTE2>(reuse);
    }
}
