















#include "kernel_operator.h"
#include "lib/math/atan.h"
#include "cfo_dmrs.h"

using namespace AscendC;

namespace {


constexpr uint32_t N_SC_PAD          = 1664;
constexpr uint32_t N_DMRS_PAD        = 7 * 128;
constexpr uint32_t Y_GM_HALF_LEN     = 14 * N_SC_PAD;
constexpr uint32_t X_GM_HALF_LEN     = 2 * N_DMRS_PAD;
constexpr uint32_t OUT_F_LEN         = 8;


constexpr uint32_t GM_REPEATS_EVEN   = 13;
constexpr uint32_t UB_Y_LANE_HALF    = N_SC_PAD;
constexpr uint32_t UB_DMRS_HALF      = N_DMRS_PAD;

}


class CfoDmrs {
public:
    __aicore__ inline CfoDmrs() {}

    __aicore__ inline void Init(GM_ADDR y_re_gm, GM_ADDR y_im_gm,
                                 GM_ADDR x_re_gm, GM_ADDR x_im_gm,
                                 GM_ADDR scratch_gm, GM_ADDR output_gm,
                                 GM_ADDR ws_gm, GM_ADDR tiling_gm,
                                 TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeHls(uint32_t s);
    __aicore__ inline void CrossCorrAndReduce();
    __aicore__ inline void Atan2AndOutput();

    TPipe                *pipe_;
    uint32_t              blockId_;
    uint32_t              aivId_;

    GlobalTensor<half>    yReG_, yImG_;
    GlobalTensor<half>    xReG_, xImG_;
    GlobalTensor<float>   outG_f_;

    TBuf<TPosition::VECCALC> bufYreLane_;
    TBuf<TPosition::VECCALC> bufYimLane_;
    TBuf<TPosition::VECCALC> bufYreDmrs_;
    TBuf<TPosition::VECCALC> bufYimDmrs_;
    TBuf<TPosition::VECCALC> bufXreLane_;
    TBuf<TPosition::VECCALC> bufXimLane_;
    TBuf<TPosition::VECCALC> bufH0re_;
    TBuf<TPosition::VECCALC> bufH0im_;
    TBuf<TPosition::VECCALC> bufH1re_;
    TBuf<TPosition::VECCALC> bufH1im_;
    TBuf<TPosition::VECCALC> bufProdRe_;
    TBuf<TPosition::VECCALC> bufProdIm_;
    TBuf<TPosition::VECCALC> bufTmp_;
    TBuf<TPosition::VECCALC> bufNormF0_;
    TBuf<TPosition::VECCALC> bufNormF1_;
    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufAtanTmp_;
    TBuf<TPosition::VECCALC> bufStage1_;
    TBuf<TPosition::VECCALC> bufOut_;

    float c_re_;
    float c_im_;
};


__aicore__ inline void CfoDmrs::Init(GM_ADDR y_re_gm, GM_ADDR y_im_gm,
                                      GM_ADDR x_re_gm, GM_ADDR x_im_gm,
                                      GM_ADDR  , GM_ADDR output_gm,
                                      GM_ADDR  , GM_ADDR  ,
                                      TPipe *pipe)
{
    pipe_    = pipe;
    blockId_ = GetBlockIdx();
    aivId_   = airan::USE_XOR_AIV_MAP ? (blockId_ ^ 2u) : blockId_;

    yReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(y_re_gm),  Y_GM_HALF_LEN);
    yImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(y_im_gm),  Y_GM_HALF_LEN);
    xReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(x_re_gm),  X_GM_HALF_LEN);
    xImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half  *>(x_im_gm),  X_GM_HALF_LEN);
    outG_f_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output_gm), OUT_F_LEN);

    pipe_->InitBuffer(bufYreLane_, UB_Y_LANE_HALF * sizeof(half));
    pipe_->InitBuffer(bufYimLane_, UB_Y_LANE_HALF * sizeof(half));
    pipe_->InitBuffer(bufYreDmrs_, UB_DMRS_HALF   * sizeof(half));
    pipe_->InitBuffer(bufYimDmrs_, UB_DMRS_HALF   * sizeof(half));
    pipe_->InitBuffer(bufXreLane_, UB_DMRS_HALF   * sizeof(half));
    pipe_->InitBuffer(bufXimLane_, UB_DMRS_HALF   * sizeof(half));
    pipe_->InitBuffer(bufH0re_,    UB_DMRS_HALF   * sizeof(half));
    pipe_->InitBuffer(bufH0im_,    UB_DMRS_HALF   * sizeof(half));
    pipe_->InitBuffer(bufH1re_,    UB_DMRS_HALF   * sizeof(half));
    pipe_->InitBuffer(bufH1im_,    UB_DMRS_HALF   * sizeof(half));
    pipe_->InitBuffer(bufProdRe_,  UB_DMRS_HALF   * sizeof(half));
    pipe_->InitBuffer(bufProdIm_,  UB_DMRS_HALF   * sizeof(half));
    pipe_->InitBuffer(bufTmp_,     UB_DMRS_HALF   * sizeof(half));
    pipe_->InitBuffer(bufNormF0_,  UB_DMRS_HALF   * sizeof(float));
    pipe_->InitBuffer(bufNormF1_,  UB_DMRS_HALF   * sizeof(float));
    pipe_->InitBuffer(bufScalar_,  64);
    pipe_->InitBuffer(bufAtanTmp_, 1024);
    pipe_->InitBuffer(bufStage1_,  64);
    pipe_->InitBuffer(bufOut_,     OUT_F_LEN * sizeof(float));
}


__aicore__ inline void CfoDmrs::ComputeHls(uint32_t s)
{
    auto yreLane = bufYreLane_.Get<half>();
    auto yimLane = bufYimLane_.Get<half>();
    auto yreDmrs = bufYreDmrs_.Get<half>();
    auto yimDmrs = bufYimDmrs_.Get<half>();
    auto xreLane = bufXreLane_.Get<half>();
    auto ximLane = bufXimLane_.Get<half>();
    auto tmp     = bufTmp_    .Get<half>();

    uint32_t dmrs_sym = (s == 0) ? airan::DMRS_SYM_0 : airan::DMRS_SYM_1;


    DataCopy(yreLane, yReG_[dmrs_sym * N_SC_PAD], N_SC_PAD);
    DataCopy(yimLane, yImG_[dmrs_sym * N_SC_PAD], N_SC_PAD);
    auto e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e1); WaitFlag<HardEvent::MTE2_V>(e1);


    {
        uint64_t rsvdCnt = 0;
        GatherMaskParams gmp;
        gmp.src0BlockStride  = 1;
        gmp.repeatTimes      = GM_REPEATS_EVEN;
        gmp.src0RepeatStride = 8;
        gmp.src1RepeatStride = 0;
        GatherMask(yreDmrs, yreLane, (uint8_t)1, false, 0, gmp, rsvdCnt);
        GatherMask(yimDmrs, yimLane, (uint8_t)1, false, 0, gmp, rsvdCnt);
        PipeBarrier<PIPE_V>();
    }




    {
        auto mag2f = bufNormF0_.Get<float>();
        auto invf  = bufNormF1_.Get<float>();

        Cast(mag2f, yreDmrs, RoundMode::CAST_NONE, airan::N_DMRS_RE); PipeBarrier<PIPE_V>();
        Mul(mag2f, mag2f, mag2f, airan::N_DMRS_RE);                   PipeBarrier<PIPE_V>();
        Cast(invf, yimDmrs, RoundMode::CAST_NONE, airan::N_DMRS_RE);  PipeBarrier<PIPE_V>();
        Mul(invf, invf, invf, airan::N_DMRS_RE);                      PipeBarrier<PIPE_V>();
        Add(mag2f, mag2f, invf, airan::N_DMRS_RE);                    PipeBarrier<PIPE_V>();
        Adds(mag2f, mag2f, 1e-6f, airan::N_DMRS_RE);                  PipeBarrier<PIPE_V>();
        Rsqrt(invf, mag2f, airan::N_DMRS_RE);                         PipeBarrier<PIPE_V>();

        auto invh = bufTmp_.Get<half>();
        Cast(invh, invf, RoundMode::CAST_NONE, airan::N_DMRS_RE);     PipeBarrier<PIPE_V>();
        Mul(yreDmrs, yreDmrs, invh, airan::N_DMRS_RE);               PipeBarrier<PIPE_V>();
        Mul(yimDmrs, yimDmrs, invh, airan::N_DMRS_RE);               PipeBarrier<PIPE_V>();
    }


    DataCopy(xreLane, xReG_[s * N_DMRS_PAD], N_DMRS_PAD);
    DataCopy(ximLane, xImG_[s * N_DMRS_PAD], N_DMRS_PAD);
    auto e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e2); WaitFlag<HardEvent::MTE2_V>(e2);


    auto h_re = (s == 0) ? bufH0re_.Get<half>() : bufH1re_.Get<half>();
    auto h_im = (s == 0) ? bufH0im_.Get<half>() : bufH1im_.Get<half>();

    Mul(h_re, yreDmrs, xreLane, airan::N_DMRS_RE);  PipeBarrier<PIPE_V>();
    Mul(tmp,  yimDmrs, ximLane, airan::N_DMRS_RE);  PipeBarrier<PIPE_V>();
    Add(h_re, h_re, tmp, airan::N_DMRS_RE);         PipeBarrier<PIPE_V>();
    Mul(h_im, yimDmrs, xreLane, airan::N_DMRS_RE);  PipeBarrier<PIPE_V>();
    Mul(tmp,  yreDmrs, ximLane, airan::N_DMRS_RE);  PipeBarrier<PIPE_V>();
    Sub(h_im, h_im, tmp, airan::N_DMRS_RE);         PipeBarrier<PIPE_V>();
}


__aicore__ inline void CfoDmrs::CrossCorrAndReduce()
{
    auto h0re = bufH0re_  .Get<half>();
    auto h0im = bufH0im_  .Get<half>();
    auto h1re = bufH1re_  .Get<half>();
    auto h1im = bufH1im_  .Get<half>();
    auto prd_re = bufProdRe_.Get<half>();
    auto prd_im = bufProdIm_.Get<half>();
    auto tmp    = bufTmp_  .Get<half>();
    auto stage1 = bufStage1_.Get<half>();

    Mul(prd_re, h0re, h1re, airan::N_DMRS_RE);  PipeBarrier<PIPE_V>();
    Mul(tmp,    h0im, h1im, airan::N_DMRS_RE);  PipeBarrier<PIPE_V>();
    Add(prd_re, prd_re, tmp, airan::N_DMRS_RE); PipeBarrier<PIPE_V>();

    Mul(prd_im, h0re, h1im, airan::N_DMRS_RE);  PipeBarrier<PIPE_V>();
    Mul(tmp,    h0im, h1re, airan::N_DMRS_RE);  PipeBarrier<PIPE_V>();
    Sub(prd_im, prd_im, tmp, airan::N_DMRS_RE); PipeBarrier<PIPE_V>();

    Duplicate(prd_re[airan::N_DMRS_RE], (half)0.0f, UB_DMRS_HALF - airan::N_DMRS_RE);
    Duplicate(prd_im[airan::N_DMRS_RE], (half)0.0f, UB_DMRS_HALF - airan::N_DMRS_RE);
    PipeBarrier<PIPE_V>();

    WholeReduceSum<half>(stage1,    prd_re,
                          128,  7,
                          1,  1,  8);
    WholeReduceSum<half>(stage1[8], prd_im,
                          128,  7,
                          1,  1,  8);
    PipeBarrier<PIPE_V>();

    WholeReduceSum<half>(stage1,    stage1,
                          7,   1,
                          1,  1,  0);
    WholeReduceSum<half>(stage1[8], stage1[8],
                          7,   1,
                          1,  1,  0);
    PipeBarrier<PIPE_V>();

    auto e_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e_v_s);
    WaitFlag<HardEvent::V_S>(e_v_s);

    c_re_ = (float)stage1.GetValue(0);
    c_im_ = (float)stage1.GetValue(8);
}


__aicore__ inline void CfoDmrs::Atan2AndOutput()
{
    auto scalar  = bufScalar_ .Get<float>();
    auto atanTmp = bufAtanTmp_.Get<uint8_t>();
    auto atanOut = bufStage1_ .Get<float>();


    auto absf = [](float x) -> float { return x < 0.0f ? -x : x; };
    auto fmin = [](float a, float b) -> float { return a < b ? a : b; };
    auto fmax = [](float a, float b) -> float { return a > b ? a : b; };

    float A = absf(c_re_);
    float B = absf(c_im_);
    float num = fmin(A, B);
    float den = fmax(A, B);
    den = fmax(den, 1e-30f);
    float t = num / den;

    scalar.SetValue(0, t);
    for (uint32_t i = 1; i < 8; ++i) scalar.SetValue(i, 0.0f);

    auto e_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(e_s_v);
    WaitFlag<HardEvent::S_V>(e_s_v);

    Atan(atanOut, scalar, atanTmp, 8);
    PipeBarrier<PIPE_V>();

    auto e_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e_v_s);
    WaitFlag<HardEvent::V_S>(e_v_s);

    float phi_q = atanOut.GetValue(0);

    constexpr float PI       = 3.14159265358979f;
    constexpr float HALF_PI  = 1.57079632679490f;


    float theta_q1 = (A >= B) ? phi_q : (HALF_PI - phi_q);
    float theta    = (c_re_ >= 0.0f) ? theta_q1 : (PI - theta_q1);
    if (c_im_ < 0.0f) theta = -theta;

    float delta_f_hz = theta * airan::HZ_PER_RAD;


    auto out_f = bufOut_.Get<float>();
    out_f.SetValue(0, delta_f_hz);
    out_f.SetValue(1, c_re_);
    out_f.SetValue(2, c_im_);
    for (uint32_t i = 3; i < 7; ++i) out_f.SetValue(i, 0.0f);
    out_f.SetValue(7, 7.0f);

    auto e_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e_s_mte3); WaitFlag<HardEvent::S_MTE3>(e_s_mte3);
    DataCopy(outG_f_, out_f, OUT_F_LEN);
    auto e_m = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e_m); WaitFlag<HardEvent::MTE3_MTE2>(e_m);
}


__aicore__ inline void CfoDmrs::Process()
{
    if (aivId_ != 0) return;

    ComputeHls(0);
    ComputeHls(1);
    CrossCorrAndReduce();
    Atan2AndOutput();
}


extern "C" __global__ __aicore__ void cfo_dmrs_kernel(
    GM_ADDR y_re_gm, GM_ADDR y_im_gm,
    GM_ADDR x_re_gm, GM_ADDR x_im_gm,
    GM_ADDR scratch_gm,
    GM_ADDR output_gm,
    GM_ADDR ws_gm, GM_ADDR tiling_gm)
{
    TPipe pipe;
    CfoDmrs op;
    op.Init(y_re_gm, y_im_gm, x_re_gm, x_im_gm,
            scratch_gm, output_gm, ws_gm, tiling_gm, &pipe);
    op.Process();
}
