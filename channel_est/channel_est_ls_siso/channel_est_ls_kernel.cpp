































#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t N_SC_USED         = 1664;
constexpr uint32_t N_SYMBOL          = 14;
constexpr uint32_t N_BLOCKS          = 4;
constexpr uint32_t DMRS_SYM_0        = 2;
constexpr uint32_t DMRS_SYM_1        = 11;
constexpr uint32_t N_DMRS_RE_PER_SYM = 832;


constexpr uint32_t SC_PER_CORE_WRITE       = 416;
constexpr uint32_t DMRS_RE_WRITE_PAIR      = 208;
constexpr uint32_t DMRS_RE_READ            = 209;

constexpr uint32_t INPUT_HALF_LEN       = N_SYMBOL * N_SC_USED;
constexpr uint32_t PILOT_HALF_LEN       = 2 * N_DMRS_RE_PER_SYM;
constexpr uint32_t OUT_HALF_LEN         = N_SYMBOL * N_SC_USED;
constexpr uint32_t WEAVE_IDX_COUNT      = 2 * DMRS_RE_WRITE_PAIR;


constexpr uint32_t Y_FULL_HALF_BUF      = 512;
constexpr uint32_t Y_DMRS_HALF_BUF      = 256;
constexpr uint32_t X_HALF_BUF           = 256;
constexpr uint32_t H_LS_HALF_BUF        = 256;
constexpr uint32_t BIG_HALF_BUF         = 832;
constexpr uint32_t NATURAL_HALF_BUF     = 416;
constexpr uint32_t WEAVE_IDX_BUF        = 416;

constexpr uint32_t Y_GATHER_REPS_DMRS  = 4;

}


class ChannelEstLs {
public:
    __aicore__ inline ChannelEstLs() {}

    __aicore__ inline void Init(GM_ADDR y_re_gm, GM_ADDR y_im_gm,
                                 GM_ADDR x_ref_re_gm, GM_ADDR x_ref_im_gm,
                                 GM_ADDR out_h_re_gm, GM_ADDR out_h_im_gm,
                                 GM_ADDR weave_idx_gm,
                                 TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Phase1_Ls();
    __aicore__ inline void Phase2_FreqInterp();
    __aicore__ inline void Phase3_TimeInterpAndNaturalWrite();

    __aicore__ inline void LoadY(uint32_t dmrs_sym,
                                  LocalTensor<half> &y_re_full,
                                  LocalTensor<half> &y_im_full);
    __aicore__ inline void LoadX(uint32_t s,
                                  LocalTensor<half> &x_re,
                                  LocalTensor<half> &x_im);
    __aicore__ inline void ExtractEvenSc(LocalTensor<half> &dst,
                                          LocalTensor<half> &src);
    __aicore__ inline void ComplexMulConj(LocalTensor<half> &hls_re,
                                           LocalTensor<half> &hls_im,
                                           LocalTensor<half> &y_re,
                                           LocalTensor<half> &y_im,
                                           LocalTensor<half> &x_re,
                                           LocalTensor<half> &x_im,
                                           uint32_t n);
    __aicore__ inline void FreqInterpOneSym(LocalTensor<half> &h_even,
                                             LocalTensor<half> &h_odd,
                                             LocalTensor<half> &h_ls);

    TPipe                *pipe_;
    uint32_t              blockId_;
    uint32_t              sc_start_;
    uint32_t              dmrs_re_start_;

    GlobalTensor<half>     yReG_;
    GlobalTensor<half>     yImG_;
    GlobalTensor<half>     xRefReG_;
    GlobalTensor<half>     xRefImG_;
    GlobalTensor<half>     outHReG_;
    GlobalTensor<half>     outHImG_;
    GlobalTensor<uint32_t> weaveIdxG_;


    TBuf<TPosition::VECCALC> bufYreFull_;
    TBuf<TPosition::VECCALC> bufYimFull_;
    TBuf<TPosition::VECCALC> bufYreDmrs_;
    TBuf<TPosition::VECCALC> bufYimDmrs_;
    TBuf<TPosition::VECCALC> bufXre_;
    TBuf<TPosition::VECCALC> bufXim_;
    TBuf<TPosition::VECCALC> bufHlsRe0_;
    TBuf<TPosition::VECCALC> bufHlsIm0_;
    TBuf<TPosition::VECCALC> bufHlsRe1_;
    TBuf<TPosition::VECCALC> bufHlsIm1_;
    TBuf<TPosition::VECCALC> bufMulTmp_;


    TBuf<TPosition::VECCALC> bufBigH0_;
    TBuf<TPosition::VECCALC> bufBigH1_;
    TBuf<TPosition::VECCALC> bufBigOut_;
    TBuf<TPosition::VECCALC> bufBigTmp_;


    TBuf<TPosition::VECCALC> bufNaturalRe_;
    TBuf<TPosition::VECCALC> bufNaturalIm_;
    TBuf<TPosition::VECCALC> bufWeaveIdx_;
};


__aicore__ inline void ChannelEstLs::Init(
    GM_ADDR y_re_gm, GM_ADDR y_im_gm,
    GM_ADDR x_ref_re_gm, GM_ADDR x_ref_im_gm,
    GM_ADDR out_h_re_gm, GM_ADDR out_h_im_gm,
    GM_ADDR weave_idx_gm,
    TPipe *pipe)
{
    pipe_    = pipe;
    blockId_ = GetBlockIdx();

    sc_start_      = blockId_ * SC_PER_CORE_WRITE;
    dmrs_re_start_ = sc_start_ / 2;


    yReG_     .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y_re_gm),
                                INPUT_HALF_LEN + 256);
    yImG_     .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y_im_gm),
                                INPUT_HALF_LEN + 256);
    xRefReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x_ref_re_gm),
                                PILOT_HALF_LEN + 256);
    xRefImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x_ref_im_gm),
                                PILOT_HALF_LEN + 256);
    outHReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(out_h_re_gm),
                                OUT_HALF_LEN);
    outHImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(out_h_im_gm),
                                OUT_HALF_LEN);
    weaveIdxG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(weave_idx_gm),
                                WEAVE_IDX_COUNT);

    pipe_->InitBuffer(bufYreFull_,   Y_FULL_HALF_BUF  * sizeof(half));
    pipe_->InitBuffer(bufYimFull_,   Y_FULL_HALF_BUF  * sizeof(half));
    pipe_->InitBuffer(bufYreDmrs_,   Y_DMRS_HALF_BUF  * sizeof(half));
    pipe_->InitBuffer(bufYimDmrs_,   Y_DMRS_HALF_BUF  * sizeof(half));
    pipe_->InitBuffer(bufXre_,       X_HALF_BUF       * sizeof(half));
    pipe_->InitBuffer(bufXim_,       X_HALF_BUF       * sizeof(half));
    pipe_->InitBuffer(bufHlsRe0_,    H_LS_HALF_BUF    * sizeof(half));
    pipe_->InitBuffer(bufHlsIm0_,    H_LS_HALF_BUF    * sizeof(half));
    pipe_->InitBuffer(bufHlsRe1_,    H_LS_HALF_BUF    * sizeof(half));
    pipe_->InitBuffer(bufHlsIm1_,    H_LS_HALF_BUF    * sizeof(half));
    pipe_->InitBuffer(bufMulTmp_,    H_LS_HALF_BUF    * sizeof(half));
    pipe_->InitBuffer(bufBigH0_,     BIG_HALF_BUF     * sizeof(half));
    pipe_->InitBuffer(bufBigH1_,     BIG_HALF_BUF     * sizeof(half));
    pipe_->InitBuffer(bufBigOut_,    BIG_HALF_BUF     * sizeof(half));
    pipe_->InitBuffer(bufBigTmp_,    BIG_HALF_BUF     * sizeof(half));
    pipe_->InitBuffer(bufNaturalRe_, NATURAL_HALF_BUF * sizeof(half));
    pipe_->InitBuffer(bufNaturalIm_, NATURAL_HALF_BUF * sizeof(half));
    pipe_->InitBuffer(bufWeaveIdx_,  WEAVE_IDX_BUF    * sizeof(uint32_t));


    auto weave_idx_ub = bufWeaveIdx_.Get<uint32_t>();
    DataCopy(weave_idx_ub, weaveIdxG_, WEAVE_IDX_COUNT);
    PipeBarrier<PIPE_ALL>();
}


__aicore__ inline void ChannelEstLs::LoadY(uint32_t dmrs_sym,
    LocalTensor<half> &y_re_full, LocalTensor<half> &y_im_full)
{
    uint32_t y_offset = dmrs_sym * N_SC_USED + sc_start_;
    DataCopy(y_re_full, yReG_[y_offset], Y_FULL_HALF_BUF);
    DataCopy(y_im_full, yImG_[y_offset], Y_FULL_HALF_BUF);
    PipeBarrier<PIPE_ALL>();
}


__aicore__ inline void ChannelEstLs::LoadX(uint32_t s,
    LocalTensor<half> &x_re, LocalTensor<half> &x_im)
{
    uint32_t x_offset = s * N_DMRS_RE_PER_SYM + dmrs_re_start_;
    DataCopy(x_re, xRefReG_[x_offset], X_HALF_BUF);
    DataCopy(x_im, xRefImG_[x_offset], X_HALF_BUF);
    PipeBarrier<PIPE_ALL>();
}


__aicore__ inline void ChannelEstLs::ExtractEvenSc(
    LocalTensor<half> &dst, LocalTensor<half> &src)
{
    uint64_t rsvdCnt = 0;
    GatherMaskParams gmp;
    gmp.src0BlockStride  = 1;
    gmp.repeatTimes      = Y_GATHER_REPS_DMRS;
    gmp.src0RepeatStride = 8;
    gmp.src1RepeatStride = 0;
    GatherMask(dst, src, (uint8_t)1, false, 0, gmp, rsvdCnt);
    PipeBarrier<PIPE_V>();
}


__aicore__ inline void ChannelEstLs::ComplexMulConj(
    LocalTensor<half> &hls_re, LocalTensor<half> &hls_im,
    LocalTensor<half> &y_re,   LocalTensor<half> &y_im,
    LocalTensor<half> &x_re,   LocalTensor<half> &x_im,
    uint32_t n)
{
    auto mul_tmp = bufMulTmp_.Get<half>();




    Mul(hls_re,  y_re, x_re, n);
    PipeBarrier<PIPE_V>();
    Mul(mul_tmp, y_im, x_im, n);
    PipeBarrier<PIPE_V>();
    Add(hls_re, hls_re, mul_tmp, n);
    PipeBarrier<PIPE_V>();

    Mul(hls_im,  y_im, x_re, n);
    PipeBarrier<PIPE_V>();
    Mul(mul_tmp, y_re, x_im, n);
    PipeBarrier<PIPE_V>();
    Sub(hls_im, hls_im, mul_tmp, n);
    PipeBarrier<PIPE_V>();
}


__aicore__ inline void ChannelEstLs::Phase1_Ls()
{
    constexpr uint32_t DMRS_SYMS[2] = { DMRS_SYM_0, DMRS_SYM_1 };

    auto y_re_full = bufYreFull_ .Get<half>();
    auto y_im_full = bufYimFull_ .Get<half>();
    auto y_re_dmrs = bufYreDmrs_ .Get<half>();
    auto y_im_dmrs = bufYimDmrs_ .Get<half>();
    auto x_re      = bufXre_     .Get<half>();
    auto x_im      = bufXim_     .Get<half>();

    for (uint32_t s = 0; s < 2; ++s) {
        uint32_t dmrs_sym = DMRS_SYMS[s];
        auto hls_re = (s == 0 ? bufHlsRe0_ : bufHlsRe1_).Get<half>();
        auto hls_im = (s == 0 ? bufHlsIm0_ : bufHlsIm1_).Get<half>();

        LoadY(dmrs_sym, y_re_full, y_im_full);
        ExtractEvenSc(y_re_dmrs, y_re_full);
        ExtractEvenSc(y_im_dmrs, y_im_full);
        LoadX(s, x_re, x_im);

        ComplexMulConj(hls_re, hls_im, y_re_dmrs, y_im_dmrs, x_re, x_im, DMRS_RE_READ);
    }



    if (blockId_ == N_BLOCKS - 1) {
        Duplicate(bufHlsRe0_.Get<half>()[DMRS_RE_WRITE_PAIR], (half)0.0f, 16);
        Duplicate(bufHlsIm0_.Get<half>()[DMRS_RE_WRITE_PAIR], (half)0.0f, 16);
        Duplicate(bufHlsRe1_.Get<half>()[DMRS_RE_WRITE_PAIR], (half)0.0f, 16);
        Duplicate(bufHlsIm1_.Get<half>()[DMRS_RE_WRITE_PAIR], (half)0.0f, 16);
        PipeBarrier<PIPE_V>();
    }

    PipeBarrier<PIPE_ALL>();
}


__aicore__ inline void ChannelEstLs::FreqInterpOneSym(
    LocalTensor<half> &h_even, LocalTensor<half> &h_odd,
    LocalTensor<half> &h_ls)
{
    Adds(h_even, h_ls, (half)0.0f, DMRS_RE_WRITE_PAIR);
    PipeBarrier<PIPE_V>();

    auto tmp = bufMulTmp_.Get<half>();
    Add(tmp,    h_ls, h_ls[1], DMRS_RE_WRITE_PAIR);
    PipeBarrier<PIPE_V>();
    Muls(h_odd, tmp, (half)0.5f, DMRS_RE_WRITE_PAIR);
    PipeBarrier<PIPE_V>();
}


__aicore__ inline void ChannelEstLs::Phase2_FreqInterp()
{


    constexpr uint32_t N = DMRS_RE_WRITE_PAIR;

    for (uint32_t s = 0; s < 2; ++s) {
        auto hls_re = (s == 0 ? bufHlsRe0_ : bufHlsRe1_).Get<half>();
        auto hls_im = (s == 0 ? bufHlsIm0_ : bufHlsIm1_).Get<half>();
        auto big_h  = (s == 0 ? bufBigH0_  : bufBigH1_ ).Get<half>();

        auto heven_re = big_h[0    ];
        auto hodd_re  = big_h[N    ];
        auto heven_im = big_h[2 * N];
        auto hodd_im  = big_h[3 * N];

        FreqInterpOneSym(heven_re, hodd_re, hls_re);
        FreqInterpOneSym(heven_im, hodd_im, hls_im);
    }

    PipeBarrier<PIPE_ALL>();
}


__aicore__ inline void ChannelEstLs::Phase3_TimeInterpAndNaturalWrite()
{
    auto big_h_0     = bufBigH0_    .Get<half>();
    auto big_h_1     = bufBigH1_    .Get<half>();
    auto big_out     = bufBigOut_   .Get<half>();
    auto big_tmp     = bufBigTmp_   .Get<half>();
    auto natural_re  = bufNaturalRe_.Get<half>();
    auto natural_im  = bufNaturalIm_.Get<half>();
    auto weave_idx   = bufWeaveIdx_ .Get<uint32_t>();

    constexpr uint32_t N         = DMRS_RE_WRITE_PAIR;
    constexpr uint32_t big_len   = 4 * N;
    constexpr uint32_t weave_cnt = 2 * N;
    constexpr uint32_t IM_BASE_BYTES = 2 * N * sizeof(half);



    constexpr event_t EVT_V_MTE3 = EVENT_ID0;
    constexpr event_t EVT_MTE3_V = EVENT_ID1;

    for (uint32_t sym = 0; sym < N_SYMBOL; ++sym) {
        half alpha_h, oma_h;
        switch (sym) {
            case  0: alpha_h = (half)(-2.0f / 9.0f); oma_h = (half)(11.0f / 9.0f); break;
            case  1: alpha_h = (half)(-1.0f / 9.0f); oma_h = (half)(10.0f / 9.0f); break;
            case  2: alpha_h = (half)( 0.0f / 9.0f); oma_h = (half)( 9.0f / 9.0f); break;
            case  3: alpha_h = (half)( 1.0f / 9.0f); oma_h = (half)( 8.0f / 9.0f); break;
            case  4: alpha_h = (half)( 2.0f / 9.0f); oma_h = (half)( 7.0f / 9.0f); break;
            case  5: alpha_h = (half)( 3.0f / 9.0f); oma_h = (half)( 6.0f / 9.0f); break;
            case  6: alpha_h = (half)( 4.0f / 9.0f); oma_h = (half)( 5.0f / 9.0f); break;
            case  7: alpha_h = (half)( 5.0f / 9.0f); oma_h = (half)( 4.0f / 9.0f); break;
            case  8: alpha_h = (half)( 6.0f / 9.0f); oma_h = (half)( 3.0f / 9.0f); break;
            case  9: alpha_h = (half)( 7.0f / 9.0f); oma_h = (half)( 2.0f / 9.0f); break;
            case 10: alpha_h = (half)( 8.0f / 9.0f); oma_h = (half)( 1.0f / 9.0f); break;
            case 11: alpha_h = (half)( 9.0f / 9.0f); oma_h = (half)( 0.0f / 9.0f); break;
            case 12: alpha_h = (half)(10.0f / 9.0f); oma_h = (half)(-1.0f / 9.0f); break;
            case 13: alpha_h = (half)(11.0f / 9.0f); oma_h = (half)(-2.0f / 9.0f); break;
            default: alpha_h = (half)0.0f; oma_h = (half)1.0f; break;
        }


        Muls(big_out, big_h_0, oma_h,   big_len);
        PipeBarrier<PIPE_V>();
        Muls(big_tmp, big_h_1, alpha_h, big_len);
        PipeBarrier<PIPE_V>();
        Add(big_out, big_out, big_tmp, big_len);
        PipeBarrier<PIPE_V>();

        Gather(natural_re, big_out, weave_idx, (uint32_t)0,             weave_cnt);
        Gather(natural_im, big_out, weave_idx, (uint32_t)IM_BASE_BYTES, weave_cnt);


        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);

        uint32_t out_offset = sym * N_SC_USED + sc_start_;
        DataCopy(outHReG_[out_offset], natural_re, weave_cnt);
        DataCopy(outHImG_[out_offset], natural_im, weave_cnt);



        if (sym < N_SYMBOL - 1) {
            SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
            WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        }
    }
}


__aicore__ inline void ChannelEstLs::Process()
{
    Phase1_Ls();
    Phase2_FreqInterp();
    Phase3_TimeInterpAndNaturalWrite();
}


extern "C" __global__ __aicore__ void channel_est_ls_kernel(
    GM_ADDR y_re_gm,
    GM_ADDR y_im_gm,
    GM_ADDR x_ref_re_gm,
    GM_ADDR x_ref_im_gm,
    GM_ADDR out_h_re_gm,
    GM_ADDR out_h_im_gm,
    GM_ADDR weave_idx_gm,
    GM_ADDR ws,
    GM_ADDR tilingGm)
{
    TPipe pipe;
    ChannelEstLs op;
    op.Init(y_re_gm, y_im_gm, x_ref_re_gm, x_ref_im_gm,
            out_h_re_gm, out_h_im_gm, weave_idx_gm, &pipe);
    op.Process();
}