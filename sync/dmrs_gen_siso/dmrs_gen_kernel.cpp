

















#include "kernel_operator.h"
#include "dmrs_gen.h"

using namespace AscendC;

namespace {
constexpr uint32_t N_RE    = airan::N_DMRS_RE;
constexpr uint32_t N_PAD   = airan::N_DMRS_PAD;
constexpr uint32_t PLANE   = airan::N_DMRS_PLANE;
constexpr uint32_t NBITS   = airan::GOLD_NBITS;
constexpr uint32_t OUT_GRID = airan::N_DMRS_SYM * N_PAD;
constexpr uint32_t CINIT_PAD = 16;
}


class DmrsGen {
public:
    __aicore__ inline DmrsGen() {}

    __aicore__ inline void Init(GM_ADDR cinit_gm, GM_ADDR gmat_gm, GM_ADDR g1_gm,
                                 GM_ADDR scratch_gm,
                                 GM_ADDR out_re_gm, GM_ADDR out_im_gm, GM_ADDR dbg_gm,
                                 GM_ADDR ws_gm, GM_ADDR tiling_gm, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GenSymbol(uint32_t s, int32_t c_init);

    TPipe              *pipe_;
    uint32_t            blockId_, aivId_;

    GlobalTensor<int32_t> cinitG_;
    GlobalTensor<half>    gmatG_, g1G_;
    GlobalTensor<half>    outReG_, outImG_;
    GlobalTensor<float>   dbgG_;

    TBuf<TPosition::VECCALC> bufGmat_;
    TBuf<TPosition::VECCALC> bufG1_;
    TBuf<TPosition::VECCALC> bufAcc_;
    TBuf<TPosition::VECCALC> bufT_;
    TBuf<TPosition::VECCALC> bufI16_;
    TBuf<TPosition::VECCALC> bufCinit_;
    TBuf<TPosition::VECCALC> bufDbg_;
};


__aicore__ inline void DmrsGen::Init(GM_ADDR cinit_gm, GM_ADDR gmat_gm, GM_ADDR g1_gm,
                                      GM_ADDR  ,
                                      GM_ADDR out_re_gm, GM_ADDR out_im_gm, GM_ADDR dbg_gm,
                                      GM_ADDR  , GM_ADDR  , TPipe *pipe)
{
    pipe_    = pipe;
    blockId_ = GetBlockIdx();
    aivId_   = airan::USE_XOR_AIV_MAP ? (blockId_ ^ 2u) : blockId_;

    cinitG_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cinit_gm), CINIT_PAD);
    gmatG_ .SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(gmat_gm), airan::MAT_LEN);
    g1G_   .SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(g1_gm),   PLANE);
    outReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(out_re_gm), OUT_GRID);
    outImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(out_im_gm), OUT_GRID);
    dbgG_  .SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dbg_gm), airan::OUT_DBG_LEN);

    pipe_->InitBuffer(bufGmat_,  airan::MAT_LEN * sizeof(half));
    pipe_->InitBuffer(bufG1_,    PLANE * sizeof(half));
    pipe_->InitBuffer(bufAcc_,   PLANE * sizeof(half));
    pipe_->InitBuffer(bufT_,     PLANE * sizeof(half));
    pipe_->InitBuffer(bufI16_,   PLANE * sizeof(int16_t));
    pipe_->InitBuffer(bufCinit_, CINIT_PAD * sizeof(int32_t));
    pipe_->InitBuffer(bufDbg_,   airan::OUT_DBG_LEN * sizeof(float));
}


__aicore__ inline void DmrsGen::GenSymbol(uint32_t s, int32_t c_init)
{
    auto gmat = bufGmat_.Get<half>();
    auto g1   = bufG1_  .Get<half>();
    auto acc  = bufAcc_ .Get<half>();
    auto t    = bufT_   .Get<half>();
    auto i16  = bufI16_ .Get<int16_t>();

    const half INV    = (half)0.70710678f;
    const half N2INV  = (half)(-1.41421356f);
    const half HALF   = (half)0.5f;
    const half NQUART = (half)(-0.25f);



    Adds(acc, g1, (half)0.0f, PLANE);
    for (uint32_t i = 0; i < NBITS; ++i) {
        if ((c_init >> i) & 1) {
            Add(acc, acc, gmat[i * PLANE], PLANE);
        }
    }
    PipeBarrier<PIPE_V>();


    Muls(t, acc, HALF, PLANE);
    Adds(t, t, NQUART, PLANE);
    Cast(i16, t, RoundMode::CAST_RINT, PLANE);
    Cast(t, i16, RoundMode::CAST_NONE, PLANE);
    Sub(acc, acc, t, PLANE);
    Sub(acc, acc, t, PLANE);
    PipeBarrier<PIPE_V>();


    Muls(t, acc, N2INV, PLANE);
    Adds(t, t, INV, PLANE);
    PipeBarrier<PIPE_V>();


    Duplicate(t[N_RE],         (half)0.0f, N_PAD - N_RE);
    Duplicate(t[N_PAD + N_RE], (half)0.0f, N_PAD - N_RE);
    PipeBarrier<PIPE_V>();

    auto eVM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eVM); WaitFlag<HardEvent::V_MTE3>(eVM);

    DataCopy(outReG_[s * N_PAD], t,        N_PAD);
    DataCopy(outImG_[s * N_PAD], t[N_PAD], N_PAD);


    auto eMV3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eMV3); WaitFlag<HardEvent::MTE3_V>(eMV3);
    auto eM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eM); WaitFlag<HardEvent::MTE3_MTE2>(eM);
}


__aicore__ inline void DmrsGen::Process()
{
    if (aivId_ != 0) return;

    auto gmat = bufGmat_.Get<half>();
    for (uint32_t i = 0; i < NBITS; ++i)
        DataCopy(gmat[i * PLANE], gmatG_[i * PLANE], PLANE);
    DataCopy(bufG1_.Get<half>(), g1G_, PLANE);

    auto cinit = bufCinit_.Get<int32_t>();
    DataCopy(cinit, cinitG_, CINIT_PAD);

    auto eMS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eMS); WaitFlag<HardEvent::MTE2_S>(eMS);
    auto eMV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eMV); WaitFlag<HardEvent::MTE2_V>(eMV);

    int32_t c0 = cinit.GetValue(0);
    int32_t c1 = cinit.GetValue(1);

    GenSymbol(0, c0);
    GenSymbol(1, c1);


    auto dbg = bufDbg_.Get<float>();
    dbg.SetValue(0, (float)c0);
    dbg.SetValue(1, (float)c1);
    for (uint32_t i = 2; i < 7; ++i) dbg.SetValue(i, 0.0f);
    dbg.SetValue(7, 7.0f);

    auto eSM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eSM); WaitFlag<HardEvent::S_MTE3>(eSM);
    DataCopy(dbgG_, dbg, airan::OUT_DBG_LEN);
    auto eM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eM); WaitFlag<HardEvent::MTE3_MTE2>(eM);


    auto eDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eDone); WaitFlag<HardEvent::MTE3_S>(eDone);
}


extern "C" __global__ __aicore__ void dmrs_gen_kernel(
    GM_ADDR cinit_gm, GM_ADDR gmat_gm, GM_ADDR g1_gm, GM_ADDR scratch_gm,
    GM_ADDR out_re_gm, GM_ADDR out_im_gm, GM_ADDR dbg_gm,
    GM_ADDR ws_gm, GM_ADDR tiling_gm)
{
    TPipe pipe;
    DmrsGen op;
    op.Init(cinit_gm, gmat_gm, g1_gm, scratch_gm,
            out_re_gm, out_im_gm, dbg_gm, ws_gm, tiling_gm, &pipe);
    op.Process();
}
