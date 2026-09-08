





























#include "kernel_operator.h"
#include "lib/math/sin.h"
#include "lib/math/cos.h"
#include "pss_acquisition.h"

using namespace AscendC;

namespace {

constexpr uint32_t N_FFT             = airan::N_FFT;
constexpr uint32_t IQ_BUF            = 2 * N_FFT;
constexpr uint32_t GATHER_REPEATS    = IQ_BUF / 128;

constexpr float    TWO_PI            = 6.28318530717959f;
constexpr float    INV_FS            = 1.0f / airan::SAMPLE_RATE_HZ_F;

}


class PssAcquisition {
public:
    __aicore__ inline PssAcquisition() {}

    __aicore__ inline void Init(GM_ADDR input_gm,
                                GM_ADDR pss_tmpl_gm,
                                GM_ADDR scratch_gm,
                                GM_ADDR output_gm,
                                GM_ADDR ws_gm,
                                TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadPssTemplates();
    __aicore__ inline void ComputeCosSinBatch(uint32_t h_start, uint32_t h_cnt);
    __aicore__ inline void ProcessTimeOffset(uint32_t batch_idx,
                                              uint32_t t_idx,
                                              uint32_t h_cnt);

    TPipe                *pipe_;
    uint32_t              blockId_;
    uint32_t              aivId_;

    GlobalTensor<int16_t> rxG_;
    GlobalTensor<int16_t> pssTmplG_;
    GlobalTensor<float>   scrG_;
    GlobalTensor<float>   metricG_;



    TBuf<TPosition::VECCALC> bufPssReHalf_;
    TBuf<TPosition::VECCALC> bufPssImHalf_;


    TBuf<TPosition::VECCALC> bufCosHalf_;
    TBuf<TPosition::VECCALC> bufSinHalf_;


    TBuf<TPosition::VECCALC> bufWinI16_;
    TBuf<TPosition::VECCALC> bufWinIqHalf_;
    TBuf<TPosition::VECCALC> bufWinReHalf_;
    TBuf<TPosition::VECCALC> bufWinImHalf_;
    TBuf<TPosition::VECCALC> bufDerotReHalf_;

    TBuf<TPosition::VECCALC> bufDerotImHalf_;
    TBuf<TPosition::VECCALC> bufTmpHalf_;
    TBuf<TPosition::VECCALC> bufTmp2Half_;


    TBuf<TPosition::VECCALC> bufRedHalf_;
    TBuf<TPosition::VECCALC> bufMetricOut_;


    TBuf<TPosition::VECCALC> bufStage1Half_;
};


__aicore__ inline void PssAcquisition::Init(GM_ADDR input_gm,
                                              GM_ADDR pss_tmpl_gm,
                                              GM_ADDR scratch_gm,
                                              GM_ADDR output_gm,
                                              GM_ADDR  ,
                                              TPipe *pipe)
{
    pipe_    = pipe;
    blockId_ = GetBlockIdx();
    aivId_   = airan::USE_XOR_AIV_MAP ? (blockId_ ^ 2u) : blockId_;

    rxG_     .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(input_gm),
                              airan::INPUT_GM_INT16_LEN);
    pssTmplG_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(pss_tmpl_gm),
                              airan::PSS_TMPL_INT16_LEN);
    scrG_    .SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch_gm),
                              airan::SCR_FP32_TOTAL);
    metricG_ .SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output_gm),
                              airan::OUT_FP32_TOTAL);


    pipe_->InitBuffer(bufPssReHalf_,    airan::N_PSS * N_FFT * sizeof(half));
    pipe_->InitBuffer(bufPssImHalf_,    airan::N_PSS * N_FFT * sizeof(half));


    pipe_->InitBuffer(bufCosHalf_,      airan::HYP_BATCH * N_FFT * sizeof(half));
    pipe_->InitBuffer(bufSinHalf_,      airan::HYP_BATCH * N_FFT * sizeof(half));


    pipe_->InitBuffer(bufWinI16_,       IQ_BUF * sizeof(int16_t));
    pipe_->InitBuffer(bufWinIqHalf_,    IQ_BUF * sizeof(half));
    pipe_->InitBuffer(bufWinReHalf_,    N_FFT * sizeof(half));
    pipe_->InitBuffer(bufWinImHalf_,    N_FFT * sizeof(half));
    pipe_->InitBuffer(bufDerotReHalf_,  airan::HYP_BATCH * N_FFT * sizeof(half));
    pipe_->InitBuffer(bufDerotImHalf_,  airan::HYP_BATCH * N_FFT * sizeof(half));
    pipe_->InitBuffer(bufTmpHalf_,      N_FFT * sizeof(half));
    pipe_->InitBuffer(bufTmp2Half_,     N_FFT * sizeof(half));

    pipe_->InitBuffer(bufRedHalf_,      64);
    pipe_->InitBuffer(bufMetricOut_,    airan::OUT_FP32_PER_BLOCK * sizeof(float));


    pipe_->InitBuffer(bufStage1Half_,   64);










}








__aicore__ inline void PssAcquisition::LoadPssTemplates()
{
    auto tmplI16 = bufDerotReHalf_.Get<int16_t>();
    auto tmplH   = bufWinIqHalf_  .Get<half>();
    auto pssRe   = bufPssReHalf_  .Get<half>();
    auto pssIm   = bufPssImHalf_  .Get<half>();


    DataCopy(tmplI16, pssTmplG_, airan::PSS_TMPL_INT16_LEN);
    auto e_mte2_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e_mte2_v);
    WaitFlag<HardEvent::MTE2_V>(e_mte2_v);


    for (uint32_t p = 0; p < airan::N_PSS; ++p) {
        Cast(tmplH, tmplI16[p * IQ_BUF], RoundMode::CAST_NONE, IQ_BUF);
        PipeBarrier<PIPE_V>();
        Muls(tmplH, tmplH, (half)airan::Q_SCALE_INV, IQ_BUF);
        PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        GatherMaskParams gmp;
        gmp.src0BlockStride  = 1;
        gmp.repeatTimes      = GATHER_REPEATS;
        gmp.src0RepeatStride = 8;
        gmp.src1RepeatStride = 0;
        GatherMask(pssRe[p * N_FFT], tmplH, (uint8_t)1, false, 0, gmp, rsvdCnt);
        GatherMask(pssIm[p * N_FFT], tmplH, (uint8_t)2, false, 0, gmp, rsvdCnt);
        PipeBarrier<PIPE_V>();
    }

}














__aicore__ inline void PssAcquisition::ComputeCosSinBatch(uint32_t h_start,
                                                            uint32_t h_cnt)
{
    auto ramp   = bufWinReHalf_ .Get<half>();
    auto cosBuf = bufCosHalf_   .Get<half>();
    auto sinBuf = bufSinHalf_   .Get<half>();
    auto phase  = bufTmpHalf_   .Get<half>();


    CreateVecIndex(ramp, (half)0, N_FFT);
    PipeBarrier<PIPE_V>();

    for (uint32_t h = 0; h < h_cnt; ++h) {

        int32_t hyp_idx = (int32_t)(h_start + h);
        float freq_hz = airan::FREQ_HYP_MIN_HZ +
                        (float)hyp_idx * airan::FREQ_HYP_STEP_HZ;
        float scale = TWO_PI * freq_hz * INV_FS;

        Muls(phase, ramp, (half)scale, N_FFT);
        PipeBarrier<PIPE_V>();

        Cos(cosBuf[h * N_FFT], phase, N_FFT);
        Sin(sinBuf[h * N_FFT], phase, N_FFT);
        PipeBarrier<PIPE_V>();
    }
}















__aicore__ inline void PssAcquisition::ProcessTimeOffset(uint32_t batch_idx,
                                                          uint32_t t_idx,
                                                          uint32_t h_cnt)
{
    int32_t rx_start = (int32_t)airan::EXPECTED_SSB_START
                      + (int32_t)t_idx
                      - (int32_t)airan::TIME_SEARCH_HALF;

    auto metricOut = bufMetricOut_.Get<float>();


    Duplicate(metricOut, (float)0.0f, airan::OUT_FP32_PER_BLOCK);
    PipeBarrier<PIPE_V>();


    if (rx_start < 0 || rx_start + (int32_t)N_FFT > (int32_t)airan::N_SAMPLE_PER_SLOT) {
        auto e_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e_v_mte3);
        WaitFlag<HardEvent::V_MTE3>(e_v_mte3);
        uint32_t gm_off = batch_idx * airan::OUT_FP32_PER_BATCH
                        + t_idx * airan::OUT_FP32_PER_BLOCK;
        DataCopy(metricG_[gm_off], metricOut, airan::OUT_FP32_PER_BLOCK);
        auto e_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e_mte3_v);
        WaitFlag<HardEvent::MTE3_V>(e_mte3_v);
        return;
    }

    auto winI16  = bufWinI16_   .Get<int16_t>();
    auto winIq   = bufWinIqHalf_.Get<half>();
    auto winRe   = bufWinReHalf_.Get<half>();
    auto winIm   = bufWinImHalf_.Get<half>();
    auto cosBuf  = bufCosHalf_  .Get<half>();
    auto sinBuf  = bufSinHalf_  .Get<half>();
    auto derotRe = bufDerotReHalf_.Get<half>();
    auto derotIm = bufDerotImHalf_.Get<half>();
    auto tmp     = bufTmpHalf_  .Get<half>();
    auto tmp2    = bufTmp2Half_ .Get<half>();
    auto pssRe   = bufPssReHalf_.Get<half>();
    auto pssIm   = bufPssImHalf_.Get<half>();



    DataCopy(winI16, rxG_[(uint32_t)rx_start * 2], IQ_BUF);
    auto e_mte2_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e_mte2_v);
    WaitFlag<HardEvent::MTE2_V>(e_mte2_v);

    Cast(winIq, winI16, RoundMode::CAST_NONE, IQ_BUF);
    PipeBarrier<PIPE_V>();
    Muls(winIq, winIq, (half)airan::Q_SCALE_INV, IQ_BUF);
    PipeBarrier<PIPE_V>();

    {
        uint64_t rsvdCnt = 0;
        GatherMaskParams gmp;
        gmp.src0BlockStride  = 1;
        gmp.repeatTimes      = GATHER_REPEATS;
        gmp.src0RepeatStride = 8;
        gmp.src1RepeatStride = 0;
        GatherMask(winRe, winIq, (uint8_t)1, false, 0, gmp, rsvdCnt);
        GatherMask(winIm, winIq, (uint8_t)2, false, 0, gmp, rsvdCnt);
        PipeBarrier<PIPE_V>();
    }




    for (uint32_t h = 0; h < h_cnt; ++h) {

        Mul(derotRe[h * N_FFT], winRe, cosBuf[h * N_FFT], N_FFT);
        PipeBarrier<PIPE_V>();
        Mul(tmp, winIm, sinBuf[h * N_FFT], N_FFT);
        PipeBarrier<PIPE_V>();
        Add(derotRe[h * N_FFT], derotRe[h * N_FFT], tmp, N_FFT);
        PipeBarrier<PIPE_V>();


        Mul(derotIm[h * N_FFT], winIm, cosBuf[h * N_FFT], N_FFT);
        PipeBarrier<PIPE_V>();
        Mul(tmp, winRe, sinBuf[h * N_FFT], N_FFT);
        PipeBarrier<PIPE_V>();
        Sub(derotIm[h * N_FFT], derotIm[h * N_FFT], tmp, N_FFT);
        PipeBarrier<PIPE_V>();
    }

















    auto stage1 = bufStage1Half_.Get<half>();

    for (uint32_t h = 0; h < h_cnt; ++h) {
        for (uint32_t p = 0; p < airan::N_PSS; ++p) {

            Mul(tmp,  derotRe[h * N_FFT], pssRe[p * N_FFT], N_FFT);
            PipeBarrier<PIPE_V>();
            Mul(tmp2, derotIm[h * N_FFT], pssIm[p * N_FFT], N_FFT);
            PipeBarrier<PIPE_V>();
            Add(tmp, tmp, tmp2, N_FFT);
            PipeBarrier<PIPE_V>();

            WholeReduceSum<half>(stage1, tmp,
                 128,  16,
                 1,  1,  8);
            PipeBarrier<PIPE_V>();

            WholeReduceSum<half>(stage1, stage1,
                 16,  1,
                 1,  1,  0);
            PipeBarrier<PIPE_V>();
            auto e_v_s_re = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(e_v_s_re);
            WaitFlag<HardEvent::V_S>(e_v_s_re);
            float corr_re = (float)stage1.GetValue(0);


            Mul(tmp,  derotIm[h * N_FFT], pssRe[p * N_FFT], N_FFT);
            PipeBarrier<PIPE_V>();
            Mul(tmp2, derotRe[h * N_FFT], pssIm[p * N_FFT], N_FFT);
            PipeBarrier<PIPE_V>();
            Sub(tmp, tmp, tmp2, N_FFT);
            PipeBarrier<PIPE_V>();
            WholeReduceSum<half>(stage1, tmp,
                 128,  16,
                 1,  1,  8);
            PipeBarrier<PIPE_V>();
            WholeReduceSum<half>(stage1, stage1,
                 16,  1,
                 1,  1,  0);
            PipeBarrier<PIPE_V>();
            auto e_v_s_im = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(e_v_s_im);
            WaitFlag<HardEvent::V_S>(e_v_s_im);
            float corr_im = (float)stage1.GetValue(0);

            float metric = corr_re * corr_re + corr_im * corr_im;
            metricOut.SetValue(h * airan::OUT_PSS_PAD + p, metric);
        }
    }


    auto e_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e_s_mte3);
    WaitFlag<HardEvent::S_MTE3>(e_s_mte3);
    uint32_t gm_off = batch_idx * airan::OUT_FP32_PER_BATCH
                    + t_idx * airan::OUT_FP32_PER_BLOCK;
    DataCopy(metricG_[gm_off], metricOut, airan::OUT_FP32_PER_BLOCK);
    auto e_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e_mte3_v);
    WaitFlag<HardEvent::MTE3_V>(e_mte3_v);
}


__aicore__ inline void PssAcquisition::Process()
{

    LoadPssTemplates();


    uint32_t t_per_core = airan::TIME_PER_CORE_MAX;
    uint32_t t_start = aivId_ * t_per_core;
    uint32_t t_end   = t_start + t_per_core;
    if (t_end > airan::N_TIME_OFFSETS) t_end = airan::N_TIME_OFFSETS;
    if (t_start >= airan::N_TIME_OFFSETS) return;


    for (uint32_t b = 0; b < airan::N_HYP_BATCHES; ++b) {
        uint32_t h_start = b * airan::HYP_BATCH;
        uint32_t h_cnt   = airan::HYP_BATCH;
        if (h_start + h_cnt > airan::N_FREQ_HYP) h_cnt = airan::N_FREQ_HYP - h_start;

        ComputeCosSinBatch(h_start, h_cnt);

        for (uint32_t t = t_start; t < t_end; ++t) {
            ProcessTimeOffset(b, t, h_cnt);
        }
    }
}


extern "C" __global__ __aicore__ void pss_acquisition_kernel(
    GM_ADDR input_gm,
    GM_ADDR pss_tmpl_gm,
    GM_ADDR scratch_gm,
    GM_ADDR ws_gm,
    GM_ADDR output_gm)
{
    TPipe pipe;
    PssAcquisition op;
    op.Init(input_gm, pss_tmpl_gm, scratch_gm, output_gm, ws_gm, &pipe);
    op.Process();
}
