// Single-launch Rank-1..4 256-QAM batch demapper.
//
// The verified SISO batch-3 pairwise math is unchanged. Four fixed AIVs retain
// ownership of three data symbols each and traverse all layers in one launch.
#include "kernel_operator.h"
#include "qam256_demod_batch.h"
#include "../qam_demod_256_siso/qam256_demod.h"

using namespace AscendC;
namespace qdb = airan::qam256_demod_batch;

namespace {
constexpr uint32_t N = qdb::N_SC_GRID_PAD;
constexpr uint32_t N_SYM_PAD = qdb::LLR_LAYER_STRIDE;
constexpr uint32_t BLOCK_DIM = qdb::BLOCK_DIM;
constexpr uint32_t SYM_PER_AIV = qdb::N_DATA_SYMBOLS / BLOCK_DIM;
constexpr uint32_t WS_SYM_STRIDE = qdb::N_SC_LLR_PAD;
constexpr uint32_t BATCH = SYM_PER_AIV * N;
constexpr uint32_t BATCH_BYTES = BATCH * sizeof(int16_t);
constexpr uint32_t OUT_BLOCK_BYTES = 4u * BATCH_BYTES;
constexpr uint32_t MASK_BYTES = ((BATCH + 127u) / 128u) * 32u;
constexpr uint32_t LLR_LAYER_ELEMS = qdb::Q_M * qdb::LLR_LAYER_STRIDE;

static_assert(BLOCK_DIM == 4 && SYM_PER_AIV == 3,
              "expected fixed four-core batch scheduling");
static_assert(BATCH == 4992, "expected 3x1664 batch");

__aicore__ inline uint32_t DataSymToPhys(uint32_t ds)
{
    if (ds < 2) return ds;
    if (ds < 10) return ds + 1;
    return ds + 2;
}
}  // namespace

extern "C" __global__ __aicore__ void qam256_demod_batch_kernel(
    GM_ADDR x_re_gm, GM_ADDR x_im_gm, GM_ADDR no_eff_gm,
    GM_ADDR out_llr_gm, GM_ADDR ws_gm, GM_ADDR tiling_gm)
{
    (void)ws_gm;
    const uint32_t aivId = GetBlockIdx() ^ 2;

    TPipe pipe;
    TBuf<TPosition::VECCALC> bxRe, bxIm, bNe, bScale, bTf;
    TBuf<TPosition::VECCALC> byI, byQ, bT;
    TBuf<TPosition::VECCALC> bOut;
    TBuf<TPosition::VECCALC> bTmp0, bTmp1, bTmp2, bTmp3, bTmp4;
    TBuf<TPosition::VECCALC> bMask;

    pipe.InitBuffer(bxRe, BATCH_BYTES); pipe.InitBuffer(bxIm, BATCH_BYTES);
    pipe.InitBuffer(bNe, BATCH_BYTES); pipe.InitBuffer(bScale, BATCH_BYTES);
    pipe.InitBuffer(bTf, BATCH_BYTES);
    pipe.InitBuffer(byI, BATCH_BYTES); pipe.InitBuffer(byQ, BATCH_BYTES);
    pipe.InitBuffer(bT, BATCH_BYTES);
    pipe.InitBuffer(bOut, OUT_BLOCK_BYTES);
    pipe.InitBuffer(bTmp0, BATCH_BYTES); pipe.InitBuffer(bTmp1, BATCH_BYTES);
    pipe.InitBuffer(bTmp2, BATCH_BYTES); pipe.InitBuffer(bTmp3, BATCH_BYTES);
    pipe.InitBuffer(bTmp4, BATCH_BYTES);
    pipe.InitBuffer(bMask, MASK_BYTES);

    // One scalar GM read is cheaper than staging the full 128-byte descriptor.
    // Field 8 is BatchTilingData::num_layers.
    const uint32_t numLayers =
        *reinterpret_cast<__gm__ uint32_t *>(tiling_gm + 8 * sizeof(uint32_t));
    if (numLayers == 0 || numLayers > qdb::MAX_LAYERS) return;

    auto xRe = bxRe.Get<half>(); auto xIm = bxIm.Get<half>();
    auto ne = bNe.Get<half>(); auto scale = bScale.Get<half>(); auto tf = bTf.Get<half>();
    auto yI = byI.Get<int16_t>(); auto yQ = byQ.Get<int16_t>(); auto tv = bT.Get<int16_t>();
    auto out = bOut.Get<int16_t>();
    auto mask = bMask.Get<uint8_t>();

    GlobalTensor<half> xReG, xImG, neG;
    GlobalTensor<int16_t> outG;
    xReG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x_re_gm),
                         qdb::MAX_LAYERS * qdb::GRID_LAYER_STRIDE);
    xImG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x_im_gm),
                         qdb::MAX_LAYERS * qdb::GRID_LAYER_STRIDE);
    neG.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(no_eff_gm),
                        qdb::MAX_LAYERS * qdb::GRID_LAYER_STRIDE);
    outG.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(out_llr_gm),
                         qdb::MAX_LAYERS * LLR_LAYER_ELEMS);

#define PROCESS_AXIS(uHalf, streamBase)                                       \
    do {                                                                      \
        auto outH = bOut.Get<half>();                                         \
        auto m2 = outH[0u * BATCH]; auto m4 = outH[1u * BATCH];               \
        auto m6 = outH[2u * BATCH]; auto m8 = outH[3u * BATCH];               \
        auto ah=bTmp0.Get<half>(); auto tmp=bTmp1.Get<half>();                \
        auto m10=bTmp2.Get<half>(); auto m12=bTmp3.Get<half>();               \
        auto m14=bTmp4.Get<half>();                                           \
        auto r3=bScale.Get<half>(); auto r2=bTf.Get<half>();                  \
        auto r1=byI.Get<half>(); auto r0=byQ.Get<half>();                     \
        Abs(ah,(uHalf),BATCH);                                                \
        Muls(tmp,ne,(half)2.0,BATCH); Min(m2,ah,tmp,BATCH);                   \
        Muls(tmp,ne,(half)4.0,BATCH); Min(m4,ah,tmp,BATCH);                   \
        Muls(tmp,ne,(half)6.0,BATCH); Min(m6,ah,tmp,BATCH);                   \
        Muls(tmp,ne,(half)8.0,BATCH); Min(m8,ah,tmp,BATCH);                   \
        Muls(tmp,ne,(half)10.0,BATCH); Min(m10,ah,tmp,BATCH);                 \
        Muls(tmp,ne,(half)12.0,BATCH); Min(m12,ah,tmp,BATCH);                 \
        Muls(tmp,ne,(half)14.0,BATCH); Min(m14,ah,tmp,BATCH);                 \
        Muls(tmp,ne,(half)5.0,BATCH); Sub(tmp,ah,tmp,BATCH);                  \
        Muls(r2,tmp,(half)16.0,BATCH);                                        \
        Sub(tmp,m2,m10,BATCH); Axpy(r2,tmp,(half)4.0,BATCH);                  \
        Sub(tmp,m4,m12,BATCH); Axpy(r2,tmp,(half)4.0,BATCH);                  \
        Sub(tmp,m6,m14,BATCH); Axpy(r2,tmp,(half)4.0,BATCH);                  \
        Muls(tmp,ne,(half)3.0,BATCH); Add(tmp,ah,tmp,BATCH);                  \
        Muls(r3,m8,(half)2.0,BATCH); Sub(tmp,tmp,r3,BATCH);                   \
        Muls(r1,tmp,(half)8.0,BATCH);                                         \
        Sub(r3,m6,m2,BATCH); Axpy(r1,r3,(half)4.0,BATCH);                     \
        Sub(r3,m10,m14,BATCH); Axpy(r1,r3,(half)4.0,BATCH);                   \
        Add(tmp,ne,m8,BATCH); Sub(tmp,tmp,m4,BATCH); Sub(tmp,tmp,m12,BATCH);  \
        Muls(r0,tmp,(half)8.0,BATCH); Axpy(r0,ah,(half)4.0,BATCH);            \
        Sub(r3,m2,ah,BATCH);                                                  \
        Sub(tmp,m4,ah,BATCH); Add(r3,r3,tmp,BATCH);                           \
        Sub(tmp,m6,ah,BATCH); Add(r3,r3,tmp,BATCH);                           \
        Sub(tmp,m8,ah,BATCH); Add(r3,r3,tmp,BATCH);                           \
        Sub(tmp,m10,ah,BATCH); Add(r3,r3,tmp,BATCH);                          \
        Sub(tmp,m12,ah,BATCH); Add(r3,r3,tmp,BATCH);                          \
        Sub(tmp,m14,ah,BATCH); Add(r3,r3,tmp,BATCH);                          \
        Sub(r3,r3,ah,BATCH);                                                  \
        Muls(tmp,r3,(half)4.0,BATCH); Muls(m10,tmp,(half)-1.0,BATCH);         \
        CompareScalar(mask,(uHalf),(half)0.0,CMPMODE::GE,BATCH);              \
        Select(r3,mask,tmp,m10,SELMODE::VSEL_TENSOR_TENSOR_MODE,BATCH);       \
        Mins(r3,r3,(half)2560.0,BATCH); Maxs(r3,r3,(half)-2560.0,BATCH);      \
        Mins(r2,r2,(half)2560.0,BATCH); Maxs(r2,r2,(half)-2560.0,BATCH);      \
        Mins(r1,r1,(half)2560.0,BATCH); Maxs(r1,r1,(half)-2560.0,BATCH);      \
        Mins(r0,r0,(half)2560.0,BATCH); Maxs(r0,r0,(half)-2560.0,BATCH);      \
        Cast(out[0u*BATCH],r3,RoundMode::CAST_RINT,BATCH);                    \
        Cast(out[1u*BATCH],r2,RoundMode::CAST_RINT,BATCH);                    \
        Cast(out[2u*BATCH],r1,RoundMode::CAST_RINT,BATCH);                    \
        Cast(out[3u*BATCH],r0,RoundMode::CAST_RINT,BATCH);                    \
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);                                \
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);                               \
        DataCopyParams cp; cp.blockCount=SYM_PER_AIV;                         \
        cp.blockLen=WS_SYM_STRIDE/16; cp.srcStride=(N-WS_SYM_STRIDE)/16;      \
        cp.dstStride=0;                                                       \
        const uint32_t firstDs=aivId*SYM_PER_AIV;                             \
        DataCopy(outG[outLayerBase+((streamBase)+0u)*N_SYM_PAD+firstDs*WS_SYM_STRIDE],out[0u*BATCH],cp);\
        DataCopy(outG[outLayerBase+((streamBase)+1u)*N_SYM_PAD+firstDs*WS_SYM_STRIDE],out[1u*BATCH],cp);\
        DataCopy(outG[outLayerBase+((streamBase)+2u)*N_SYM_PAD+firstDs*WS_SYM_STRIDE],out[2u*BATCH],cp);\
        DataCopy(outG[outLayerBase+((streamBase)+3u)*N_SYM_PAD+firstDs*WS_SYM_STRIDE],out[3u*BATCH],cp);\
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);                                \
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);                               \
    } while (0)

    for (uint32_t layer = 0; layer < numLayers; ++layer) {
        const uint32_t gridLayerBase = layer * qdb::GRID_LAYER_STRIDE;
        const uint32_t outLayerBase = layer * LLR_LAYER_ELEMS;

        for (uint32_t ls = 0; ls < SYM_PER_AIV; ++ls) {
            const uint32_t ds = aivId * SYM_PER_AIV + ls;
            const uint32_t src = gridLayerBase + DataSymToPhys(ds) * N;
            const uint32_t dst = ls * N;
            DataCopy(xRe[dst], xReG[src], N);
            DataCopy(xIm[dst], xImG[src], N);
            DataCopy(ne[dst], neG[src], N);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        Duplicate(scale, static_cast<half>(airan::D_X_QSCALE), BATCH);
        Duplicate(tf, static_cast<half>(airan::D2_X_QSCALE), BATCH);
        Div(scale, scale, ne, BATCH); Div(tf, tf, ne, BATCH);
        Mul(xRe, xRe, scale, BATCH); Mul(xIm, xIm, scale, BATCH);
        Cast(yI, xRe, RoundMode::CAST_RINT, BATCH);
        Cast(yQ, xIm, RoundMode::CAST_RINT, BATCH);
        Cast(tv, tf, RoundMode::CAST_RINT, BATCH);
        Cast(xRe, yI, RoundMode::CAST_NONE, BATCH);
        Cast(xIm, yQ, RoundMode::CAST_NONE, BATCH);
        Cast(ne, tv, RoundMode::CAST_NONE, BATCH);
        Mins(xRe,xRe,static_cast<half>(1000.0),BATCH);
        Maxs(xRe,xRe,static_cast<half>(-1000.0),BATCH);
        Mins(xIm,xIm,static_cast<half>(1000.0),BATCH);
        Maxs(xIm,xIm,static_cast<half>(-1000.0),BATCH);

        PROCESS_AXIS(xRe, 0u);
        PROCESS_AXIS(xIm, 4u);
        PipeBarrier<PIPE_ALL>();
    }

#undef PROCESS_AXIS
}
