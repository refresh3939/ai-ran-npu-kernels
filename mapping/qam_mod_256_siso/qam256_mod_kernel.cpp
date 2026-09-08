


#include "kernel_operator.h"
#include "qam256_mod.h"

using namespace AscendC;

namespace {
constexpr uint32_t N_PAD = airan::N_SC_PAD;
constexpr uint32_t N_USED = airan::N_SC_USED;
constexpr uint32_t N_STREAM_PAD = airan::N_SYM_PAD;
constexpr uint32_t BLOCK_DIM = airan::BLOCK_DIM;
constexpr uint32_t SYM_PER_AIV = airan::N_DATA_SYM / BLOCK_DIM;
constexpr uint32_t IN_SYM_STRIDE = 1600;
constexpr uint32_t BATCH = SYM_PER_AIV * IN_SYM_STRIDE;
constexpr uint32_t BATCH_BYTES = BATCH * sizeof(int16_t);
constexpr uint32_t ROW_BYTES = N_PAD * sizeof(half);

static_assert(BLOCK_DIM == 4 && SYM_PER_AIV == 3, "fixed four-core batch");
static_assert(airan::N_DATA_SYM * IN_SYM_STRIDE == N_STREAM_PAD,
              "padded input must occupy 19200 elements per stream");
static_assert((BATCH % 16u) == 0u, "batch DataCopy must be 32-byte aligned");

__aicore__ inline uint32_t DataSymToPhys(uint32_t ds)
{
    if (ds < 2) return ds;
    if (ds < 10) return ds + 1;
    return ds + 2;
}
}

extern "C" __global__ __aicore__ void qam256_mod_kernel(
    GM_ADDR bits_gm,
    GM_ADDR x_re_gm, GM_ADDR x_im_gm,
    GM_ADDR ws_gm, GM_ADDR tiling_gm)
{
    (void)ws_gm;
    (void)tiling_gm;
    const uint32_t aiv_id = GetBlockIdx() ^ 2u;
    if (aiv_id >= BLOCK_DIM) return;

    TPipe pipe;
    TBuf<TPosition::VECCALC> bufC[8];
    TBuf<TPosition::VECCALC> bufTmp, bufXre, bufXim;
    TBuf<TPosition::VECCALC> bufRowRe, bufRowIm;
    for (int i = 0; i < 8; ++i) pipe.InitBuffer(bufC[i], BATCH_BYTES);
    pipe.InitBuffer(bufTmp, BATCH_BYTES);
    pipe.InitBuffer(bufXre, BATCH_BYTES);
    pipe.InitBuffer(bufXim, BATCH_BYTES);
    pipe.InitBuffer(bufRowRe, ROW_BYTES);
    pipe.InitBuffer(bufRowIm, ROW_BYTES);

    auto c0=bufC[0].Get<half>(); auto c1=bufC[1].Get<half>();
    auto c2=bufC[2].Get<half>(); auto c3=bufC[3].Get<half>();
    auto c4=bufC[4].Get<half>(); auto c5=bufC[5].Get<half>();
    auto c6=bufC[6].Get<half>(); auto c7=bufC[7].Get<half>();
    auto tmp=bufTmp.Get<half>(); auto xre=bufXre.Get<half>(); auto xim=bufXim.Get<half>();
    auto rowRe=bufRowRe.Get<half>(); auto rowIm=bufRowIm.Get<half>();

    const half h2=(half)2.0f, hm1=(half)-1.0f, hm2=(half)-2.0f;
    const half h3=(half)3.0f, h4=(half)4.0f, h8=(half)8.0f;
    const half hz=(half)0.0f, hD=(half)airan::D_256QAM;

    GlobalTensor<int16_t> bitsG;
    GlobalTensor<half> xReG, xImG;
    bitsG.SetGlobalBuffer((__gm__ int16_t*)bits_gm, airan::Q_M * N_STREAM_PAD);
    xReG.SetGlobalBuffer((__gm__ half*)x_re_gm, airan::N_SYMBOL_MAX * N_PAD);
    xImG.SetGlobalBuffer((__gm__ half*)x_im_gm, airan::N_SYMBOL_MAX * N_PAD);


    Duplicate(rowRe, hz, N_PAD);
    Duplicate(rowIm, hz, N_PAD);
    PipeBarrier<PIPE_V>();

    const uint32_t input_base = aiv_id * SYM_PER_AIV * IN_SYM_STRIDE;
    for (int b = 0; b < 8; ++b) {
        auto ci16 = bufC[b].Get<int16_t>();
        DataCopy(ci16, bitsG[(uint32_t)b * N_STREAM_PAD + input_base], BATCH);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    {
        auto v0=bufC[0].Get<int16_t>(); auto v1=bufC[1].Get<int16_t>();
        auto v2=bufC[2].Get<int16_t>(); auto v3=bufC[3].Get<int16_t>();
        auto v4=bufC[4].Get<int16_t>(); auto v5=bufC[5].Get<int16_t>();
        auto v6=bufC[6].Get<int16_t>(); auto v7=bufC[7].Get<int16_t>();
        Cast(c0,v0,RoundMode::CAST_NONE,BATCH); Cast(c1,v1,RoundMode::CAST_NONE,BATCH);
        Cast(c2,v2,RoundMode::CAST_NONE,BATCH); Cast(c3,v3,RoundMode::CAST_NONE,BATCH);
        Cast(c4,v4,RoundMode::CAST_NONE,BATCH); Cast(c5,v5,RoundMode::CAST_NONE,BATCH);
        Cast(c6,v6,RoundMode::CAST_NONE,BATCH); Cast(c7,v7,RoundMode::CAST_NONE,BATCH);
    }
    PipeBarrier<PIPE_V>();

#define MOD_AXIS(cc0, cc1, cc2, cc3, dst)                                    \
    do {                                                                      \
        Muls((cc0),(cc0),h2,BATCH); Adds((cc0),(cc0),hm1,BATCH);              \
        Muls((cc1),(cc1),h2,BATCH); Adds((cc1),(cc1),hm1,BATCH);              \
        Muls((cc2),(cc2),h2,BATCH); Adds((cc2),(cc2),hm1,BATCH);              \
        Muls((cc3),(cc3),hm2,BATCH); Adds((cc3),(cc3),h3,BATCH);              \
        Mul(tmp,(cc2),(cc3),BATCH);                                           \
        Muls(tmp,tmp,hm1,BATCH); Adds(tmp,tmp,h4,BATCH);                      \
        Mul(tmp,(cc1),tmp,BATCH);                                             \
        Muls(tmp,tmp,hm1,BATCH); Adds(tmp,tmp,h8,BATCH);                      \
        Mul(tmp,(cc0),tmp,BATCH);                                             \
        Muls((dst),tmp,hD,BATCH);                                             \
    } while (0)

    MOD_AXIS(c0,c1,c2,c3,xre);
    MOD_AXIS(c4,c5,c6,c7,xim);
    PipeBarrier<PIPE_V>();

    for (uint32_t local = 0; local < SYM_PER_AIV; ++local) {
        const uint32_t ds = aiv_id * SYM_PER_AIV + local;
        const uint32_t phys = DataSymToPhys(ds);
        const uint32_t src = local * IN_SYM_STRIDE;
        Adds(rowRe, xre[src], hz, N_USED);
        Adds(rowIm, xim[src], hz, N_USED);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(xReG[phys * N_PAD], rowRe, N_PAD);
        DataCopy(xImG[phys * N_PAD], rowIm, N_PAD);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    if (aiv_id == 0) {
        Duplicate(rowRe, hz, N_PAD);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        DataCopy(xReG[airan::DMRS_SYM_0*N_PAD], rowRe, N_PAD);
        DataCopy(xImG[airan::DMRS_SYM_0*N_PAD], rowRe, N_PAD);
        DataCopy(xReG[airan::DMRS_SYM_1*N_PAD], rowRe, N_PAD);
        DataCopy(xImG[airan::DMRS_SYM_1*N_PAD], rowRe, N_PAD);
    }
    PipeBarrier<PIPE_ALL>();

#undef MOD_AXIS
}
