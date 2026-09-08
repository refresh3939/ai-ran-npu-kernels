










































#include "kernel_operator.h"
#include "lib/math/atan.h"
#include "pss_cfo_estimator.h"

using namespace AscendC;

namespace {

constexpr uint32_t N_PSS              = airan::N_PSS;
constexpr uint32_t N_HALF             = airan::N_HALF;




constexpr uint32_t GM_REPEATS_Y_SPLIT = 4;
constexpr uint32_t GM_REPEATS_R_SPLIT = 4;


constexpr uint32_t UB_Y_INTERLEAVED   = 4 * 128;
constexpr uint32_t UB_R_INTERLEAVED   = 4 * 128;
constexpr uint32_t UB_Y_LANE          = 4 * 128;
constexpr uint32_t UB_Y_LANE_USED     = 256;
constexpr uint32_t UB_R_LANE          = 4 * 128;
constexpr uint32_t UB_R_LANE_USED     = 256;

}


class PssCfoEstimator {
public:
    __aicore__ inline PssCfoEstimator() {}

    __aicore__ inline void Init(GM_ADDR y_gm,
                                 GM_ADDR r_gm,
                                 GM_ADDR scratch_gm,
                                 GM_ADDR output_gm,
                                 GM_ADDR ws_gm,
                                 GM_ADDR tiling_gm,
                                 TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadAndSplitIQ();
    __aicore__ inline void ConjMulAndReduce();
    __aicore__ inline void Atan2AndWriteOut();

    TPipe                *pipe_;
    uint32_t              blockId_;
    uint32_t              aivId_;

    GlobalTensor<int16_t> yG_;
    GlobalTensor<int16_t> rG_;
    GlobalTensor<half>    twiddleG_;
    GlobalTensor<float>   outG_f_;


    TBuf<TPosition::VECCALC> bufYi16_;
    TBuf<TPosition::VECCALC> bufYhalf_;
    TBuf<TPosition::VECCALC> bufRi16_;
    TBuf<TPosition::VECCALC> bufRhalf_;


    TBuf<TPosition::VECCALC> bufYre_;
    TBuf<TPosition::VECCALC> bufYim_;

    TBuf<TPosition::VECCALC> bufTwCos_;
    TBuf<TPosition::VECCALC> bufTwSin_;
    TBuf<TPosition::VECCALC> bufRotTmp_;
    TBuf<TPosition::VECCALC> bufRotTmp2_;
    TBuf<TPosition::VECCALC> bufRre_;
    TBuf<TPosition::VECCALC> bufRim_;


    TBuf<TPosition::VECCALC> bufPrdRe0_;
    TBuf<TPosition::VECCALC> bufPrdIm0_;
    TBuf<TPosition::VECCALC> bufPrdRe1_;
    TBuf<TPosition::VECCALC> bufPrdIm1_;
    TBuf<TPosition::VECCALC> bufTmp_;



    TBuf<TPosition::VECCALC> bufStage1_;


    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufAtanOut_;
    TBuf<TPosition::VECCALC> bufAtanTmp_;


    TBuf<TPosition::VECCALC> bufOut_;


    float c0_re_;
    float c0_im_;
    float c1_re_;
    float c1_im_;
};


__aicore__ inline void PssCfoEstimator::Init(GM_ADDR y_gm,
                                              GM_ADDR r_gm,
                                              GM_ADDR scratch_gm,
                                              GM_ADDR output_gm,
                                              GM_ADDR  ,
                                              GM_ADDR  ,
                                              TPipe *pipe)
{
    pipe_    = pipe;
    blockId_ = GetBlockIdx();
    aivId_   = airan::USE_XOR_AIV_MAP ? (blockId_ ^ 2u) : blockId_;

    twiddleG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scratch_gm),
                             2u * airan::N_PSS);
    yG_   .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(y_gm),
                            airan::INPUT_GM_INT16_LEN);
    rG_   .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(r_gm),
                            airan::PILOT_GM_INT16_LEN);
    outG_f_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output_gm),
                             airan::OUT_FLOAT_PADDED);

    pipe_->InitBuffer(bufYi16_,    UB_Y_INTERLEAVED * sizeof(int16_t));
    pipe_->InitBuffer(bufYhalf_,   UB_Y_INTERLEAVED * sizeof(half));
    pipe_->InitBuffer(bufRi16_,    UB_R_INTERLEAVED * sizeof(int16_t));
    pipe_->InitBuffer(bufRhalf_,   UB_R_INTERLEAVED * sizeof(half));

    pipe_->InitBuffer(bufYre_,     UB_Y_LANE * sizeof(half));
    pipe_->InitBuffer(bufYim_,     UB_Y_LANE * sizeof(half));
    pipe_->InitBuffer(bufTwCos_,   airan::N_PSS * sizeof(half));
    pipe_->InitBuffer(bufTwSin_,   airan::N_PSS * sizeof(half));
    pipe_->InitBuffer(bufRotTmp_,  airan::N_PSS * sizeof(half));
    pipe_->InitBuffer(bufRotTmp2_, airan::N_PSS * sizeof(half));
    pipe_->InitBuffer(bufRre_,     UB_R_LANE * sizeof(half));
    pipe_->InitBuffer(bufRim_,     UB_R_LANE * sizeof(half));

    pipe_->InitBuffer(bufPrdRe0_,  N_HALF * sizeof(half));
    pipe_->InitBuffer(bufPrdIm0_,  N_HALF * sizeof(half));
    pipe_->InitBuffer(bufPrdRe1_,  N_HALF * sizeof(half));
    pipe_->InitBuffer(bufPrdIm1_,  N_HALF * sizeof(half));
    pipe_->InitBuffer(bufTmp_,     N_HALF * sizeof(half));

    pipe_->InitBuffer(bufStage1_,  32 * sizeof(half));
    pipe_->InitBuffer(bufScalar_,  16 * sizeof(float));
    pipe_->InitBuffer(bufAtanOut_, 16 * sizeof(float));
    pipe_->InitBuffer(bufAtanTmp_, 1024);

    pipe_->InitBuffer(bufOut_,     airan::OUT_GM_BYTES);
}











__aicore__ inline void PssCfoEstimator::LoadAndSplitIQ()
{
    auto yi16   = bufYi16_  .Get<int16_t>();
    auto yhalf  = bufYhalf_ .Get<half>();
    auto ri16   = bufRi16_  .Get<int16_t>();
    auto rhalf  = bufRhalf_ .Get<half>();
    auto yre    = bufYre_   .Get<half>();
    auto yim    = bufYim_   .Get<half>();
    auto rre    = bufRre_   .Get<half>();
    auto rim    = bufRim_   .Get<half>();


    DataCopy(yi16, yG_, airan::INPUT_GM_INT16_LEN);
    {
        auto e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(e);
        WaitFlag<HardEvent::MTE2_V>(e);
    }


    DataCopy(ri16, rG_, airan::PILOT_GM_INT16_LEN);
    {
        auto e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(e);
        WaitFlag<HardEvent::MTE2_V>(e);
    }


    Cast(yhalf, yi16, RoundMode::CAST_NONE, airan::INPUT_GM_INT16_LEN);
    Cast(rhalf, ri16, RoundMode::CAST_NONE, airan::PILOT_GM_INT16_LEN);
    PipeBarrier<PIPE_V>();
    Muls(yhalf, yhalf, (half)airan::Q_SCALE_INV, airan::INPUT_GM_INT16_LEN);
    Muls(rhalf, rhalf, (half)airan::Q_SCALE_INV, airan::PILOT_GM_INT16_LEN);
    PipeBarrier<PIPE_V>();




    {
        uint64_t rsvdCnt = 0;
        GatherMaskParams gmp;
        gmp.src0BlockStride  = 1;
        gmp.repeatTimes      = GM_REPEATS_Y_SPLIT;
        gmp.src0RepeatStride = 8;
        gmp.src1RepeatStride = 0;
        GatherMask(yre, yhalf, (uint8_t)1, false, 0, gmp, rsvdCnt);
        GatherMask(yim, yhalf, (uint8_t)2, false, 0, gmp, rsvdCnt);
        PipeBarrier<PIPE_V>();
    }



    {
        uint64_t rsvdCnt = 0;
        GatherMaskParams gmp;
        gmp.src0BlockStride  = 1;
        gmp.repeatTimes      = GM_REPEATS_R_SPLIT;
        gmp.src0RepeatStride = 8;
        gmp.src1RepeatStride = 0;
        GatherMask(rre, rhalf, (uint8_t)1, false, 0, gmp, rsvdCnt);
        GatherMask(rim, rhalf, (uint8_t)2, false, 0, gmp, rsvdCnt);
        PipeBarrier<PIPE_V>();
    }





    {
        auto twc    = bufTwCos_.Get<half>();
        auto tws    = bufTwSin_.Get<half>();
        auto rot_re = bufRotTmp_.Get<half>();
        auto rot_im = bufRotTmp2_.Get<half>();
        DataCopy(twc, twiddleG_[0],            airan::N_PSS);
        DataCopy(tws, twiddleG_[airan::N_PSS], airan::N_PSS);
        {
            auto e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(e);
            WaitFlag<HardEvent::MTE2_V>(e);
        }

        Mul(rot_re, yre, twc, airan::N_PSS);
        PipeBarrier<PIPE_V>();
        Mul(rot_im, yim, tws, airan::N_PSS);
        PipeBarrier<PIPE_V>();
        Add(rot_re, rot_re, rot_im, airan::N_PSS);
        PipeBarrier<PIPE_V>();

        Mul(rot_im, yim, twc, airan::N_PSS);
        PipeBarrier<PIPE_V>();
        Mul(yim,    yre, tws, airan::N_PSS);
        PipeBarrier<PIPE_V>();
        Sub(rot_im, rot_im, yim, airan::N_PSS);
        PipeBarrier<PIPE_V>();

        DataCopy(yre, rot_re, airan::N_PSS);
        DataCopy(yim, rot_im, airan::N_PSS);
        PipeBarrier<PIPE_V>();
    }
}



















__aicore__ inline void PssCfoEstimator::ConjMulAndReduce()
{
    auto yre    = bufYre_   .Get<half>();
    auto yim    = bufYim_   .Get<half>();
    auto rre    = bufRre_   .Get<half>();
    auto rim    = bufRim_   .Get<half>();
    auto prd_re0 = bufPrdRe0_.Get<half>();
    auto prd_im0 = bufPrdIm0_.Get<half>();
    auto prd_re1 = bufPrdRe1_.Get<half>();
    auto prd_im1 = bufPrdIm1_.Get<half>();
    auto tmp     = bufTmp_   .Get<half>();
    auto stage1  = bufStage1_.Get<half>();



    Mul(prd_re0, yre, rre, N_HALF);
    PipeBarrier<PIPE_V>();
    Mul(tmp,     yim, rim, N_HALF);
    PipeBarrier<PIPE_V>();
    Add(prd_re0, prd_re0, tmp, N_HALF);
    PipeBarrier<PIPE_V>();


    Mul(prd_im0, yim, rre, N_HALF);
    PipeBarrier<PIPE_V>();
    Mul(tmp,     yre, rim, N_HALF);
    PipeBarrier<PIPE_V>();
    Sub(prd_im0, prd_im0, tmp, N_HALF);
    PipeBarrier<PIPE_V>();




    Mul(prd_re1, yre[N_HALF], rre[N_HALF], N_HALF);
    PipeBarrier<PIPE_V>();
    Mul(tmp,     yim[N_HALF], rim[N_HALF], N_HALF);
    PipeBarrier<PIPE_V>();
    Add(prd_re1, prd_re1, tmp, N_HALF);
    PipeBarrier<PIPE_V>();

    Mul(prd_im1, yim[N_HALF], rre[N_HALF], N_HALF);
    PipeBarrier<PIPE_V>();
    Mul(tmp,     yre[N_HALF], rim[N_HALF], N_HALF);
    PipeBarrier<PIPE_V>();
    Sub(prd_im1, prd_im1, tmp, N_HALF);
    PipeBarrier<PIPE_V>();




    WholeReduceSum<half>(stage1,      prd_re0,
                          128,  1,
                          1,  1,  8);
    WholeReduceSum<half>(stage1[8],   prd_im0,
                          128,  1,
                          1,  1,  8);
    WholeReduceSum<half>(stage1[16],  prd_re1,
                          128,  1,
                          1,  1,  8);
    WholeReduceSum<half>(stage1[24],  prd_im1,
                          128,  1,
                          1,  1,  8);
    PipeBarrier<PIPE_V>();


    {
        auto e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(e);
        WaitFlag<HardEvent::V_S>(e);
    }

    c0_re_ = (float)stage1.GetValue(0);
    c0_im_ = (float)stage1.GetValue(8);
    c1_re_ = (float)stage1.GetValue(16);
    c1_im_ = (float)stage1.GetValue(24);
}













__aicore__ inline void PssCfoEstimator::Atan2AndWriteOut()
{
    auto scalar  = bufScalar_ .Get<float>();
    auto atanTmp = bufAtanTmp_.Get<uint8_t>();
    auto atanOut = bufAtanOut_.Get<float>();
    auto out_f   = bufOut_    .Get<float>();


    float D_re = c1_re_ * c0_re_ + c1_im_ * c0_im_;
    float D_im = c1_im_ * c0_re_ - c1_re_ * c0_im_;











    float mag2_c0 = c0_re_ * c0_re_ + c0_im_ * c0_im_;
    float mag2_c1 = c1_re_ * c1_re_ + c1_im_ * c1_im_;
    float mag2_D  = D_re * D_re + D_im * D_im;












    constexpr float MAG2_D_FLOOR = 1.0e-6f;
    float theta;
    float delta_f_frac;

    if (mag2_D < MAG2_D_FLOOR) {

        theta        = 0.0f;
        delta_f_frac = 0.0f;
    } else {

        auto absf = [](float x) -> float { return x < 0.0f ? -x : x; };
        auto fmin = [](float a, float b) -> float { return a < b ? a : b; };
        auto fmax = [](float a, float b) -> float { return a > b ? a : b; };

        float A = absf(D_re);
        float B = absf(D_im);
        float num = fmin(A, B);
        float den = fmax(A, B);
        den = fmax(den, 1e-30f);
        float t = num / den;

        scalar.SetValue(0, t);
        for (uint32_t i = 1; i < 8; ++i) scalar.SetValue(i, 0.0f);

        {
            auto e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(e);
            WaitFlag<HardEvent::S_V>(e);
        }

        Atan(atanOut, scalar, atanTmp, 8);
        PipeBarrier<PIPE_V>();

        {
            auto e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(e);
            WaitFlag<HardEvent::V_S>(e);
        }

        float phi_q = atanOut.GetValue(0);

        constexpr float PI       = 3.14159265358979f;
        constexpr float HALF_PI  = 1.57079632679490f;


        float theta_q1 = (A >= B) ? phi_q : (HALF_PI - phi_q);
        theta = (D_re >= 0.0f) ? theta_q1 : (PI - theta_q1);
        if (D_im < 0.0f) theta = -theta;


        delta_f_frac = theta * airan::HZ_PER_RAD;
    }







    out_f.SetValue(0, delta_f_frac);
    out_f.SetValue(1, theta);
    out_f.SetValue(2, mag2_c0);
    out_f.SetValue(3, mag2_c1);
    for (uint32_t i = 4; i < airan::OUT_FLOAT_PADDED; ++i) {
        out_f.SetValue(i, 0.0f);
    }

    {
        auto e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(e);
        WaitFlag<HardEvent::S_MTE3>(e);
    }
    DataCopy(outG_f_, out_f, airan::OUT_FLOAT_PADDED);
    {
        auto e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(e);
        WaitFlag<HardEvent::MTE3_MTE2>(e);
    }
}


__aicore__ inline void PssCfoEstimator::Process()
{
    if (aivId_ != 0) return;

    LoadAndSplitIQ();
    ConjMulAndReduce();
    Atan2AndWriteOut();
}


extern "C" __global__ __aicore__ void pss_cfo_estimator_kernel(
    GM_ADDR y_gm,
    GM_ADDR r_gm,
    GM_ADDR scratch_gm,
    GM_ADDR output_gm,
    GM_ADDR ws_gm,
    GM_ADDR tiling_gm)
{
    TPipe pipe;
    PssCfoEstimator op;
    op.Init(y_gm, r_gm, scratch_gm, output_gm, ws_gm, tiling_gm, &pipe);
    op.Process();
}
