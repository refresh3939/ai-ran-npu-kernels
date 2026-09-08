

























#include "kernel_operator.h"
#include "mimo_dmrs_gen.h"

using namespace AscendC;

namespace {
namespace mdg = airan::mimo_dmrs_gen;
constexpr uint32_t N_RE = mdg::N_DMRS_RE;
constexpr uint32_t N_PAD = mdg::N_DMRS_PAD;
constexpr uint32_t PLANE = mdg::N_DMRS_PLANE;
constexpr uint32_t NBITS = mdg::GOLD_NBITS;
constexpr uint32_t N_SYM = mdg::CURRENT_DMRS_SYMBOLS;
constexpr uint32_t CINIT_PAD = mdg::CINIT_PAD;
constexpr uint32_t MAX_NL = mdg::MAX_LAYERS;
constexpr uint32_t OUT_GRID = MAX_NL * N_SYM * N_PAD;
}


class MimoDmrsGen {
public:
    __aicore__ inline MimoDmrsGen() {}

    __aicore__ inline void Init(GM_ADDR cinit_gm, GM_ADDR gmat_gm, GM_ADDR g1_gm,
                                 GM_ADDR scratch_gm,
                                 GM_ADDR out_re_gm, GM_ADDR out_im_gm, GM_ADDR dbg_gm,
                                 GM_ADDR ws_gm, GM_ADDR tiling_gm, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BuildOccFlip();
    __aicore__ inline void GenBase(int32_t c_init);
    __aicore__ inline void EmitSymbol(uint32_t s);
    __aicore__ inline void ClearLayer(uint32_t l);

    TPipe              *pipe_;
    uint32_t            blockId_, aivId_, nl_;
    uint32_t            wfOddNegative_[MAX_NL];

    GlobalTensor<int32_t> cinitG_;
    GlobalTensor<half>    gmatG_, g1G_;
    GlobalTensor<half>    outReG_, outImG_;
    GlobalTensor<float>   dbgG_;
    GlobalTensor<uint32_t> metadataG_;

    TBuf<TPosition::VECCALC> bufGmat_;
    TBuf<TPosition::VECCALC> bufG1_;
    TBuf<TPosition::VECCALC> bufAcc_;
    TBuf<TPosition::VECCALC> bufT_;
    TBuf<TPosition::VECCALC> bufI16_;
    TBuf<TPosition::VECCALC> bufBaseRe_;
    TBuf<TPosition::VECCALC> bufBaseIm_;
    TBuf<TPosition::VECCALC> bufOccFlip_;
    TBuf<TPosition::VECCALC> bufIdx_;
    TBuf<TPosition::VECCALC> bufOut_;
    TBuf<TPosition::VECCALC> bufCinit_;
    TBuf<TPosition::VECCALC> bufDbg_;
    TBuf<TPosition::VECCALC> bufMetadata_;
};


__aicore__ inline void MimoDmrsGen::Init(GM_ADDR cinit_gm, GM_ADDR gmat_gm, GM_ADDR g1_gm,
                                          GM_ADDR  ,
                                          GM_ADDR out_re_gm, GM_ADDR out_im_gm, GM_ADDR dbg_gm,
                                          GM_ADDR  , GM_ADDR tiling_gm, TPipe *pipe)
{
    pipe_    = pipe;
    blockId_ = GetBlockIdx();
    aivId_   = blockId_;
    nl_      = 0;

    cinitG_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cinit_gm), CINIT_PAD);
    gmatG_ .SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(gmat_gm), mdg::MAT_LEN);
    g1G_   .SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(g1_gm),   PLANE);
    outReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(out_re_gm), OUT_GRID);
    outImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(out_im_gm), OUT_GRID);
    dbgG_  .SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dbg_gm), mdg::OUT_DBG_LEN);
    metadataG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling_gm), mdg::TILING_WORDS);

    pipe_->InitBuffer(bufGmat_,    mdg::MAT_LEN * sizeof(half));
    pipe_->InitBuffer(bufG1_,      PLANE * sizeof(half));
    pipe_->InitBuffer(bufAcc_,     PLANE * sizeof(half));
    pipe_->InitBuffer(bufT_,       PLANE * sizeof(half));
    pipe_->InitBuffer(bufI16_,     PLANE * sizeof(int16_t));
    pipe_->InitBuffer(bufBaseRe_,  N_PAD * sizeof(half));
    pipe_->InitBuffer(bufBaseIm_,  N_PAD * sizeof(half));
    pipe_->InitBuffer(bufOccFlip_, N_PAD * sizeof(half));
    pipe_->InitBuffer(bufIdx_,     N_PAD * sizeof(half));
    pipe_->InitBuffer(bufOut_,     N_PAD * sizeof(half));
    pipe_->InitBuffer(bufCinit_,   CINIT_PAD * sizeof(int32_t));
    pipe_->InitBuffer(bufDbg_,     mdg::OUT_DBG_LEN * sizeof(float));
    pipe_->InitBuffer(bufMetadata_, mdg::TILING_BYTES);
}



__aicore__ inline void MimoDmrsGen::BuildOccFlip()
{
    auto idx  = bufIdx_.Get<half>();
    auto t    = bufT_.Get<half>();
    auto i16  = bufI16_.Get<int16_t>();
    auto occf = bufOccFlip_.Get<half>();

    const half HALF   = (half)0.5f;
    const half NQUART = (half)(-0.25f);
    const half N2     = (half)(-2.0f);
    const half ONE    = (half)1.0f;

    ArithProgression(idx, (half)0.0f, (half)1.0f, N_PAD);
    PipeBarrier<PIPE_V>();

    Muls(t, idx, HALF, N_PAD);
    Adds(t, t, NQUART, N_PAD);
    Cast(i16, t, RoundMode::CAST_RINT, N_PAD);
    Cast(t, i16, RoundMode::CAST_NONE, N_PAD);
    PipeBarrier<PIPE_V>();

    Muls(occf, t, N2, N_PAD);
    Add(occf, occf, idx, N_PAD);
    PipeBarrier<PIPE_V>();

    Muls(t, occf, N2, N_PAD);
    Adds(occf, t, ONE, N_PAD);
    PipeBarrier<PIPE_V>();
}


__aicore__ inline void MimoDmrsGen::GenBase(int32_t c_init)
{
    auto gmat = bufGmat_.Get<half>();
    auto g1   = bufG1_  .Get<half>();
    auto acc  = bufAcc_ .Get<half>();
    auto t    = bufT_   .Get<half>();
    auto i16  = bufI16_ .Get<int16_t>();
    auto bre  = bufBaseRe_.Get<half>();
    auto bim  = bufBaseIm_.Get<half>();

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

    Adds(bre, t,        (half)0.0f, N_PAD);
    Adds(bim, t[N_PAD], (half)0.0f, N_PAD);
    Duplicate(bre[N_RE], (half)0.0f, N_PAD - N_RE);
    Duplicate(bim[N_RE], (half)0.0f, N_PAD - N_RE);
    PipeBarrier<PIPE_V>();
}


__aicore__ inline void MimoDmrsGen::EmitSymbol(uint32_t s)
{
    auto bre  = bufBaseRe_.Get<half>();
    auto bim  = bufBaseIm_.Get<half>();
    auto occf = bufOccFlip_.Get<half>();
    auto out  = bufOut_.Get<half>();



    bool hasPlain = false;
    for (uint32_t l = 0; l < nl_; ++l) hasPlain |= wfOddNegative_[l] == 0;
    if (hasPlain) {
        auto eVM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eVM); WaitFlag<HardEvent::V_MTE3>(eVM);
        for (uint32_t l = 0; l < nl_; ++l) {
            if (wfOddNegative_[l] != 0) continue;
            const uint32_t offset = (l * N_SYM + s) * N_PAD;
            DataCopy(outReG_[offset], bre, N_PAD);
            DataCopy(outImG_[offset], bim, N_PAD);
        }
        auto eMV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eMV); WaitFlag<HardEvent::MTE3_V>(eMV);
    }

    bool hasFlip = false;
    for (uint32_t l = 0; l < nl_; ++l) hasFlip |= wfOddNegative_[l] != 0;
    if (!hasFlip) return;



    Mul(out, bre, occf, N_PAD);
    PipeBarrier<PIPE_V>();
    {
        auto eVM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eVM); WaitFlag<HardEvent::V_MTE3>(eVM);
        for (uint32_t l = 0; l < nl_; ++l) {
            if (wfOddNegative_[l] == 0) continue;
            DataCopy(outReG_[(l * N_SYM + s) * N_PAD], out, N_PAD);
        }
        auto eM1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eM1); WaitFlag<HardEvent::MTE3_V>(eM1);
    }

    Mul(out, bim, occf, N_PAD);
    PipeBarrier<PIPE_V>();
    {
        auto eVM2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eVM2); WaitFlag<HardEvent::V_MTE3>(eVM2);
        for (uint32_t l = 0; l < nl_; ++l) {
            if (wfOddNegative_[l] == 0) continue;
            DataCopy(outImG_[(l * N_SYM + s) * N_PAD], out, N_PAD);
        }
        auto eM2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eM2); WaitFlag<HardEvent::MTE3_V>(eM2);
    }
}


__aicore__ inline void MimoDmrsGen::ClearLayer(uint32_t l)
{
    auto out = bufOut_.Get<half>();
    Duplicate(out, (half)0.0f, N_PAD);
    PipeBarrier<PIPE_V>();
    auto eVM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eVM); WaitFlag<HardEvent::V_MTE3>(eVM);
    for (uint32_t s = 0; s < N_SYM; ++s) {
        const uint32_t offset = (l * N_SYM + s) * N_PAD;
        DataCopy(outReG_[offset], out, N_PAD);
        DataCopy(outImG_[offset], out, N_PAD);
    }
    auto eM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eM); WaitFlag<HardEvent::MTE3_MTE2>(eM);
}


__aicore__ inline void MimoDmrsGen::Process()
{
    if (aivId_ != 0) return;

    auto gmat = bufGmat_.Get<half>();
    DataCopy(bufG1_.Get<half>(), g1G_, PLANE);

    auto cinit = bufCinit_.Get<int32_t>();
    DataCopy(cinit, cinitG_, CINIT_PAD);
    auto metadata = bufMetadata_.Get<uint32_t>();
    DataCopy(metadata, metadataG_, mdg::TILING_WORDS);

    auto eMS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eMS); WaitFlag<HardEvent::MTE2_S>(eMS);
    auto eMV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eMV); WaitFlag<HardEvent::MTE2_V>(eMV);

    if (metadata.GetValue(0) != mdg::META_MAGIC) return;
    nl_ = metadata.GetValue(1);
    if (nl_ == 0 || nl_ > MAX_NL || metadata.GetValue(2) != N_SYM) return;

    int32_t c[N_SYM];
    c[0] = cinit.GetValue(0);
    c[1] = cinit.GetValue(1);
    const uint32_t activeBits = static_cast<uint32_t>(c[0]) | static_cast<uint32_t>(c[1]);


    uint32_t activeRowCount = 0;
    for (uint32_t i = 0; i < NBITS; ++i) {
        if (((activeBits >> i) & 1u) != 0) {
            DataCopy(gmat[i * PLANE], gmatG_[i * PLANE], PLANE);
            ++activeRowCount;
        }
    }
    if (activeBits != 0) {
        auto eBasis = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eBasis); WaitFlag<HardEvent::MTE2_V>(eBasis);
    }

    bool needsOccFlip = false;
    for (uint32_t l = 0; l < MAX_NL; ++l) {
        wfOddNegative_[l] = metadata.GetValue(mdg::META_WF_ODD_NEGATIVE_WORD + l);
        if (wfOddNegative_[l] > 1) return;
        if (l < nl_ && wfOddNegative_[l] != 0) needsOccFlip = true;
    }


    for (uint32_t l = 0; l < nl_; ++l) {
        for (uint32_t s = 0; s < N_SYM; ++s) {
            if (metadata.GetValue(mdg::META_WT_NEGATIVE_WORD + l * N_SYM + s) != 0) return;
        }
    }
    if (needsOccFlip) BuildOccFlip();

    for (uint32_t s = 0; s < N_SYM; ++s) {
        GenBase(c[s]);
        EmitSymbol(s);
    }
    for (uint32_t l = nl_; l < MAX_NL; ++l) ClearLayer(l);

    auto dbg = bufDbg_.Get<float>();
    dbg.SetValue(0, (float)c[0]);
    dbg.SetValue(1, (float)c[1]);
    dbg.SetValue(2, static_cast<float>(static_cast<int32_t>(nl_)));
    dbg.SetValue(3, static_cast<float>(static_cast<int32_t>(activeRowCount)));
    for (uint32_t i = 4; i < 7; ++i) dbg.SetValue(i, 0.0f);
    dbg.SetValue(7, 7.0f);

    auto eSM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eSM); WaitFlag<HardEvent::S_MTE3>(eSM);
    DataCopy(dbgG_, dbg, mdg::OUT_DBG_LEN);
    auto eM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eM); WaitFlag<HardEvent::MTE3_MTE2>(eM);
}


extern "C" __global__ __aicore__ void mimo_dmrs_gen_kernel(
    GM_ADDR cinit_gm, GM_ADDR gmat_gm, GM_ADDR g1_gm, GM_ADDR scratch_gm,
    GM_ADDR out_re_gm, GM_ADDR out_im_gm, GM_ADDR dbg_gm,
    GM_ADDR ws_gm, GM_ADDR tiling_gm)
{
    TPipe pipe;
    MimoDmrsGen op;
    op.Init(cinit_gm, gmat_gm, g1_gm, scratch_gm,
            out_re_gm, out_im_gm, dbg_gm, ws_gm, tiling_gm, &pipe);
    op.Process();
}
