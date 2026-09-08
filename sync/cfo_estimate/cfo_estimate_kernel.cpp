



























#include "kernel_operator.h"
#include "lib/math/atan.h"
#include "cfo_estimate.h"

using namespace AscendC;

namespace {

constexpr uint32_t SAMPLE_MAX        = 2240;
constexpr uint32_t IQ_BUF            = 2 * SAMPLE_MAX;
constexpr uint32_t GATHER_REPEATS    = IQ_BUF / 128;

__aicore__ inline uint32_t sym_start_iq(uint32_t sym) {
    return (sym == 0) ? 0u
                      : (airan::CP_LEN_FIRST + airan::N_FFT
                         + (sym - 1u) * airan::SYM_STRIDE);
}
__aicore__ inline uint32_t cp_len_of(uint32_t sym) {
    return (sym == 0) ? airan::CP_LEN_FIRST : airan::CP_LEN_OTHER;
}

}


class CfoEstimate {
public:
    __aicore__ inline CfoEstimate() {}

    __aicore__ inline void Init(GM_ADDR input_gm,
                                GM_ADDR scratch_gm,
                                GM_ADDR output_gm,
                                GM_ADDR ws_gm,
                                TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Phase1_CpAutocorr();
    __aicore__ inline void Phase3_Atan2AndScale();
    __aicore__ inline void ProcessSymbol(uint32_t sym,
                                          float &r_re_acc,
                                          float &r_im_acc);

    TPipe                *pipe_;
    uint32_t              blockId_;
    uint32_t              aivId_;

    GlobalTensor<int16_t> inG_;
    GlobalTensor<float>   scrG_;
    GlobalTensor<float>   outG_;

    TBuf<TPosition::VECCALC> bufXi16_;
    TBuf<TPosition::VECCALC> bufXiq_;
    TBuf<TPosition::VECCALC> bufXre_;
    TBuf<TPosition::VECCALC> bufXim_;
    TBuf<TPosition::VECCALC> bufTmp1_;
    TBuf<TPosition::VECCALC> bufTmp2_;
    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufAtanTmp_;
    TBuf<TPosition::VECCALC> bufOut_;
    TBuf<TPosition::VECCALC> bufScrRead_;
};


__aicore__ inline void CfoEstimate::Init(GM_ADDR input_gm,
                                          GM_ADDR scratch_gm,
                                          GM_ADDR output_gm,
                                          GM_ADDR  ,
                                          TPipe *pipe)
{
    pipe_    = pipe;
    blockId_ = GetBlockIdx();
    aivId_   = airan::USE_XOR_AIV_MAP ? (blockId_ ^ 2u) : blockId_;

    inG_ .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(input_gm),
                          airan::INPUT_GM_INT16_LEN);
    scrG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch_gm),
                          airan::SCR_FP32_TOTAL);
    outG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output_gm),
                          airan::OUT_FP32_COUNT);

    pipe_->InitBuffer(bufXi16_,    IQ_BUF * sizeof(int16_t));
    pipe_->InitBuffer(bufXiq_,     IQ_BUF * sizeof(half));
    pipe_->InitBuffer(bufXre_,     SAMPLE_MAX * sizeof(half));
    pipe_->InitBuffer(bufXim_,     SAMPLE_MAX * sizeof(half));
    pipe_->InitBuffer(bufTmp1_,    SAMPLE_MAX * sizeof(float));
    pipe_->InitBuffer(bufTmp2_,    SAMPLE_MAX * sizeof(float));
    pipe_->InitBuffer(bufScalar_,  64);
    pipe_->InitBuffer(bufAtanTmp_, 1024);
    pipe_->InitBuffer(bufOut_,     airan::OUT_GM_BYTES);
    pipe_->InitBuffer(bufScrRead_, 128);
}


__aicore__ inline void CfoEstimate::ProcessSymbol(uint32_t sym,
                                                    float &r_re_acc,
                                                    float &r_im_acc)
{
    auto xi16   = bufXi16_  .Get<int16_t>();
    auto xhalf  = bufXiq_   .Get<half>();
    auto xre_h  = bufXre_   .Get<half>();
    auto xim_h  = bufXim_   .Get<half>();
    auto tmp1_h = bufTmp1_  .Get<half>();
    auto tmp2_h = bufTmp2_  .Get<half>();

    uint32_t gm_iq_start = sym_start_iq(sym);
    uint32_t cp_len      = cp_len_of(sym);


    DataCopy(xi16, inG_[gm_iq_start * 2], IQ_BUF);
    auto e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e1);
    WaitFlag<HardEvent::MTE2_V>(e1);


    Cast(xhalf, xi16, RoundMode::CAST_NONE, IQ_BUF);
    PipeBarrier<PIPE_V>();
    Muls(xhalf, xhalf, (half)airan::Q_SCALE_INV, IQ_BUF);
    PipeBarrier<PIPE_V>();


    uint64_t rsvdCnt = 0;
    GatherMaskParams gmp;
    gmp.src0BlockStride  = 1;
    gmp.repeatTimes      = GATHER_REPEATS;
    gmp.src0RepeatStride = 8;
    gmp.src1RepeatStride = 0;
    GatherMask(xre_h, xhalf, (uint8_t)1, false, 0, gmp, rsvdCnt);
    GatherMask(xim_h, xhalf, (uint8_t)2, false, 0, gmp, rsvdCnt);
    PipeBarrier<PIPE_V>();


    Mul(tmp1_h, xre_h[0], xre_h[airan::N_FFT], cp_len);
    PipeBarrier<PIPE_V>();
    Mul(tmp2_h, xim_h[0], xim_h[airan::N_FFT], cp_len);
    PipeBarrier<PIPE_V>();
    Add(tmp1_h, tmp1_h, tmp2_h, cp_len);
    PipeBarrier<PIPE_V>();

    auto tmp2_f = bufTmp2_.Get<float>();
    Cast(tmp2_f, tmp1_h, RoundMode::CAST_NONE, cp_len);
    PipeBarrier<PIPE_V>();

    auto e_v_s1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e_v_s1);
    WaitFlag<HardEvent::V_S>(e_v_s1);
    for (uint32_t i = 0; i < cp_len; ++i) {
        r_re_acc += tmp2_f.GetValue(i);
    }


    Mul(tmp1_h, xre_h[0], xim_h[airan::N_FFT], cp_len);
    PipeBarrier<PIPE_V>();
    Mul(tmp2_h, xim_h[0], xre_h[airan::N_FFT], cp_len);
    PipeBarrier<PIPE_V>();
    Sub(tmp1_h, tmp1_h, tmp2_h, cp_len);
    PipeBarrier<PIPE_V>();

    Cast(tmp2_f, tmp1_h, RoundMode::CAST_NONE, cp_len);
    PipeBarrier<PIPE_V>();

    auto e_v_s2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e_v_s2);
    WaitFlag<HardEvent::V_S>(e_v_s2);
    for (uint32_t i = 0; i < cp_len; ++i) {
        r_im_acc += tmp2_f.GetValue(i);
    }
}


__aicore__ inline void CfoEstimate::Phase1_CpAutocorr()
{
    float r_re_acc = 0.0f;
    float r_im_acc = 0.0f;

    uint32_t sym_start = aivId_ * airan::SYMBOLS_PER_CORE;
    for (uint32_t i = 0; i < airan::SYMBOLS_PER_CORE; ++i) {
        uint32_t sym = sym_start + i;
        if (sym >= airan::N_SYMBOL) continue;
        ProcessSymbol(sym, r_re_acc, r_im_acc);
    }


    auto scalar = bufScalar_.Get<float>();
    scalar.SetValue(0, r_re_acc);
    scalar.SetValue(1, r_im_acc);
    for (uint32_t i = 2; i < 8; ++i) scalar.SetValue(i, 0.0f);

    auto e_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e_s_mte3);
    WaitFlag<HardEvent::S_MTE3>(e_s_mte3);
    DataCopy(scrG_[aivId_ * airan::SCR_FP32_PER_CORE], scalar, 8);
    auto e_mte3_mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e_mte3_mte2);
    WaitFlag<HardEvent::MTE3_MTE2>(e_mte3_mte2);
}


__aicore__ inline void CfoEstimate::Phase3_Atan2AndScale()
{
    if (aivId_ != 0) return;

    auto scrRead = bufScrRead_.Get<float>();
    auto scalar  = bufScalar_ .Get<float>();
    auto atanTmp = bufAtanTmp_.Get<uint8_t>();
    auto out     = bufOut_    .Get<float>();

    DataCopy(scrRead, scrG_, 32);
    auto e_mte2_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e_mte2_s);
    WaitFlag<HardEvent::MTE2_S>(e_mte2_s);

    float R_re = 0.0f, R_im = 0.0f;
    for (uint32_t c = 0; c < airan::N_BLOCKS; ++c) {
        R_re += scrRead.GetValue(c * airan::SCR_FP32_PER_CORE + 0);
        R_im += scrRead.GetValue(c * airan::SCR_FP32_PER_CORE + 1);
    }

    auto absf = [](float x) -> float { return x < 0.0f ? -x : x; };
    auto fmin = [](float a, float b) -> float { return a < b ? a : b; };
    auto fmax = [](float a, float b) -> float { return a > b ? a : b; };

    float A = absf(R_re);
    float B = absf(R_im);
    float num = fmin(A, B);
    float den = fmax(A, B);
    den = fmax(den, 1e-30f);
    float t = num / den;

    scalar.SetValue(0, t);
    for (uint32_t i = 1; i < 8; ++i) scalar.SetValue(i, 0.0f);

    auto e_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(e_s_v);
    WaitFlag<HardEvent::S_V>(e_s_v);

    Atan(scrRead, scalar, atanTmp, 8);
    PipeBarrier<PIPE_V>();

    auto e_v_s3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e_v_s3);
    WaitFlag<HardEvent::V_S>(e_v_s3);

    float phi = scrRead.GetValue(0);

    constexpr float PI       = 3.14159265358979f;
    constexpr float HALF_PI  = 1.57079632679490f;
    constexpr float TWO_PI   = 6.28318530717959f;
    constexpr float SCS_HZ_F = 30000.0f;
    constexpr float SCALE    = SCS_HZ_F / TWO_PI;

    float theta_q1 = (A >= B) ? phi : (HALF_PI - phi);
    float theta    = (R_re >= 0.0f) ? theta_q1 : (PI - theta_q1);
    if (R_im < 0.0f) theta = -theta;

    float delta_f_hz = theta * SCALE;

    out.SetValue(0, R_re);
    out.SetValue(1, R_im);
    out.SetValue(2, delta_f_hz);
    for (uint32_t i = 3; i < airan::OUT_FP32_COUNT; ++i) out.SetValue(i, 0.0f);

    auto e_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e_s_mte3);
    WaitFlag<HardEvent::S_MTE3>(e_s_mte3);
    DataCopy(outG_, out, airan::OUT_FP32_COUNT);
}


__aicore__ inline void CfoEstimate::Process()
{
    Phase1_CpAutocorr();
    AscendC::SyncAll();
    Phase3_Atan2AndScale();
}


extern "C" __global__ __aicore__ void cfo_estimate_kernel(
    GM_ADDR input_gm,
    GM_ADDR scratch_gm,
    GM_ADDR ws_gm,
    GM_ADDR output_gm)
{
    TPipe pipe;
    CfoEstimate op;
    op.Init(input_gm, scratch_gm, output_gm, ws_gm, &pipe);
    op.Process();
}