










































#include "kernel_operator.h"
#include "lib/transpose/confusion_transpose.h"

using namespace AscendC;

namespace {


constexpr uint32_t N_FFT     = 4096;
constexpr uint32_t K_DMRS    = 220;
constexpr uint32_t N_SYM     = 2;


constexpr uint32_t N_SYM_TOTAL = 14;
constexpr uint32_t N_SC_PAD    = 1664;
constexpr uint32_t DMRS_SYM_0  = 2;
constexpr uint32_t DMRS_SYM_1  = 11;
constexpr uint32_t REF_OFFSET  = 0;
constexpr uint32_t DC_PER_SYM_HALF = 224;


constexpr uint32_t P         = 64;
constexpr uint32_t Q         = 64;
constexpr uint32_t TILE_PQ   = P * Q;


constexpr uint16_t M_MMAD    = 64;
constexpr uint16_t N_SUB     = 16;
constexpr uint16_t N_SPLITS  = 4;
constexpr uint16_t K_MMAD    = 64;



constexpr float    OUTPUT_SCALE = 1.0f / float(N_FFT);




constexpr uint32_t W_SEARCH = 72;
constexpr uint32_t W_SEARCH_TOTAL = 2 * W_SEARCH;

}





class TimingTracker {
public:
    __aicore__ inline TimingTracker() {}

    __aicore__ inline void Init(GM_ADDR h_re_gm, GM_ADDR h_im_gm,
                                 GM_ADDR w_re_gm, GM_ADDR w_im_gm,
                                 GM_ADDR tw_re_gm, GM_ADDR tw_im_gm,
                                 GM_ADDR dt_scale_gm,
                                 GM_ADDR cir_re_gm, GM_ADDR cir_im_gm,
                                 GM_ADDR delta_T_gm,
                                 GM_ADDR scratch_gm,
                                 TPipe *pipe);
    __aicore__ inline void Process();

private:

    __aicore__ inline void Phase0_LoadAndPad();
    __aicore__ inline void Phase1_IdftP();
    __aicore__ inline void Phase2_Twiddle();
    __aicore__ inline void Phase3_IdftQ();
    __aicore__ inline void Phase3p5_TransposeAndScale();
    __aicore__ inline void Phase5to7_ArgmaxAndDeltaT();


    __aicore__ inline void CopyNd2Nz(const LocalTensor<half> &dst,
                                      const GlobalTensor<half> &src,
                                      uint16_t height, uint16_t width);
    __aicore__ inline void LoadL1ToL0A(const LocalTensor<half> &a2,
                                        const LocalTensor<half> &a1,
                                        uint16_t mBlocks, uint16_t kBlocks);
    __aicore__ inline void NzToNdAndCastHalf(const LocalTensor<half> &dst,
                                              const LocalTensor<float> &src,
                                              const LocalTensor<float> &tmp);


    __aicore__ inline void RunCubeP1(const LocalTensor<half> &Al1,
                                      const GlobalTensor<half> &Bgm,
                                      const LocalTensor<float> &dst);
    __aicore__ inline void RunCubeP3(const GlobalTensor<half> &Agm,
                                      const LocalTensor<half> &Bl1,
                                      const LocalTensor<float> &dst);


    TPipe   *pipe_;
    uint32_t blockId_;


    GlobalTensor<half>    hReG_;
    GlobalTensor<half>    hImG_;
    GlobalTensor<float>   dtScaleG_;
    GlobalTensor<half>    wReG_, wImG_;
    GlobalTensor<half>    twReG_, twImG_;

    GlobalTensor<half>    cirReG_, cirImG_;
    GlobalTensor<float>   deltaTG_;

    GlobalTensor<half>    HpadReScr_, HpadImScr_;
    GlobalTensor<half>    X1ReScr_, X1ImScr_;
    GlobalTensor<half>    TwReScr_, TwImScr_;



    TBuf<TPosition::VECCALC> bufHpadRe_;
    TBuf<TPosition::VECCALC> bufHpadIm_;
    TBuf<TPosition::VECCALC> bufHcirRe_;
    TBuf<TPosition::VECCALC> bufHcirIm_;
    TBuf<TPosition::VECCALC> bufOutRe_;
    TBuf<TPosition::VECCALC> bufOutIm_;
    TBuf<TPosition::VECCALC> bufTmp_;
    TBuf<TPosition::VECCALC> bufCubeAcc_;
    TBuf<TPosition::VECCALC> bufCubeUB_;
    TBuf<TPosition::VECCALC> bufCubeNd_;


    TQue<TPosition::A1, 2>  qA1_;
    TQue<TPosition::A2, 2>  qA2_;
    TQue<TPosition::B1, 2>  qB1_;
    TQue<TPosition::B2, 2>  qB2_;
    TQue<TPosition::CO1, 2> qCO1_;


    TBuf<TPosition::A1> bufWreL1_;
    TBuf<TPosition::A1> bufWimL1_;
    TBuf<TPosition::B1> bufWreL1B_;
    TBuf<TPosition::B1> bufWimL1B_;
};





__aicore__ inline void TimingTracker::Init(
    GM_ADDR h_re_gm, GM_ADDR h_im_gm,
    GM_ADDR w_re_gm, GM_ADDR w_im_gm,
    GM_ADDR tw_re_gm, GM_ADDR tw_im_gm,
    GM_ADDR dt_scale_gm,
    GM_ADDR cir_re_gm, GM_ADDR cir_im_gm,
    GM_ADDR delta_T_gm,
    GM_ADDR scratch_gm,
    TPipe *pipe)
{
    pipe_      = pipe;
    blockId_   = GetBlockIdx();


    hReG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h_re_gm), N_SYM_TOTAL * N_SC_PAD);
    hImG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h_im_gm), N_SYM_TOTAL * N_SC_PAD);
    wReG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w_re_gm),  P * P);
    wImG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w_im_gm),  P * P);
    twReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(tw_re_gm), TILE_PQ);
    twImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(tw_im_gm), TILE_PQ);
    cirReG_ .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cir_re_gm), N_FFT);
    cirImG_ .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cir_im_gm), N_FFT);
    deltaTG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(delta_T_gm), 1);


    dtScaleG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dt_scale_gm), 1);


    auto sbase = reinterpret_cast<__gm__ half *>(scratch_gm);
    HpadReScr_.SetGlobalBuffer(sbase + 0 * N_FFT, N_FFT);
    HpadImScr_.SetGlobalBuffer(sbase + 1 * N_FFT, N_FFT);
    X1ReScr_  .SetGlobalBuffer(sbase + 2 * N_FFT, N_FFT);
    X1ImScr_  .SetGlobalBuffer(sbase + 3 * N_FFT, N_FFT);
    TwReScr_  .SetGlobalBuffer(sbase + 4 * N_FFT, N_FFT);
    TwImScr_  .SetGlobalBuffer(sbase + 5 * N_FFT, N_FFT);


    pipe_->InitBuffer(bufHpadRe_,  N_FFT * sizeof(half));
    pipe_->InitBuffer(bufHpadIm_,  N_FFT * sizeof(half));
    pipe_->InitBuffer(bufHcirRe_,  N_FFT * sizeof(half));
    pipe_->InitBuffer(bufHcirIm_,  N_FFT * sizeof(half));
    pipe_->InitBuffer(bufOutRe_,   N_FFT * sizeof(half));
    pipe_->InitBuffer(bufOutIm_,   N_FFT * sizeof(half));
    pipe_->InitBuffer(bufTmp_,     N_FFT * sizeof(half));
    pipe_->InitBuffer(bufCubeAcc_, TILE_PQ * sizeof(float));
    pipe_->InitBuffer(bufCubeUB_,  TILE_PQ * sizeof(float));
    pipe_->InitBuffer(bufCubeNd_,  TILE_PQ * sizeof(float));


    pipe_->InitBuffer(qA1_,  2, M_MMAD * K_MMAD * sizeof(half));
    pipe_->InitBuffer(qA2_,  2, M_MMAD * K_MMAD * sizeof(half));
    pipe_->InitBuffer(qB1_,  2, K_MMAD * Q * sizeof(half));
    pipe_->InitBuffer(qB2_,  2, K_MMAD * Q * sizeof(half));
    pipe_->InitBuffer(qCO1_, 2, M_MMAD * Q * sizeof(float));


    pipe_->InitBuffer(bufWreL1_,  P * P * sizeof(half));
    pipe_->InitBuffer(bufWimL1_,  P * P * sizeof(half));
    pipe_->InitBuffer(bufWreL1B_, P * P * sizeof(half));
    pipe_->InitBuffer(bufWimL1B_, P * P * sizeof(half));
}






__aicore__ inline void TimingTracker::CopyNd2Nz(
    const LocalTensor<half> &dst, const GlobalTensor<half> &src,
    uint16_t height, uint16_t width)
{
    for (uint16_t i = 0; i < width / 16; ++i) {
        DataCopy(dst[i * 16 * height], src[i * 16],
                 { height, 1, uint16_t(width / 16 - 1), 0 });
    }
}

__aicore__ inline void TimingTracker::LoadL1ToL0A(
    const LocalTensor<half> &a2, const LocalTensor<half> &a1,
    uint16_t mBlocks, uint16_t kBlocks)
{
    LoadData2DParams p;
    p.repeatTimes = kBlocks;
    p.srcStride   = mBlocks;
    p.ifTranspose = false;
    for (uint16_t i = 0; i < mBlocks; ++i) {
        LoadData(a2[i * kBlocks * 256], a1[i * 256], p);
    }
}

__aicore__ inline void TimingTracker::NzToNdAndCastHalf(
    const LocalTensor<half> &dst,
    const LocalTensor<float> &src,
    const LocalTensor<float> &tmp)
{

    DataCopyParams dcp;
    dcp.blockCount = M_MMAD;
    dcp.blockLen   = 2;
    dcp.srcStride  = 0;
    dcp.dstStride  = uint16_t((Q - N_SUB) * sizeof(float) / 32);

    for (uint16_t ns = 0; ns < N_SPLITS; ns++) {
        DataCopy(tmp[ns * N_SUB], src[ns * M_MMAD * N_SUB], dcp);
    }
    PipeBarrier<PIPE_V>();
    Cast(dst, tmp, RoundMode::CAST_NONE, TILE_PQ);
    PipeBarrier<PIPE_V>();
}





















__attribute__((noinline)) __aicore__ void TimingTracker::Phase0_LoadAndPad()
{
    if (blockId_ != 0) return;

    auto reSym2  = bufHcirRe_.Get<half>();
    auto reSym11 = bufHcirRe_.Get<half>()[DC_PER_SYM_HALF];
    auto imSym2  = bufHcirIm_.Get<half>();
    auto imSym11 = bufHcirIm_.Get<half>()[DC_PER_SYM_HALF];
    auto hpadRe  = bufHpadRe_.Get<half>();
    auto hpadIm  = bufHpadIm_.Get<half>();
    auto tmp     = bufTmp_.Get<half>();


    constexpr uint32_t SYM2_OFF  = DMRS_SYM_0 * N_SC_PAD + REF_OFFSET;
    constexpr uint32_t SYM11_OFF = DMRS_SYM_1 * N_SC_PAD + REF_OFFSET;
    DataCopy(reSym2,  hReG_[SYM2_OFF],  DC_PER_SYM_HALF);
    DataCopy(reSym11, hReG_[SYM11_OFF], DC_PER_SYM_HALF);
    DataCopy(imSym2,  hImG_[SYM2_OFF],  DC_PER_SYM_HALF);
    DataCopy(imSym11, hImG_[SYM11_OFF], DC_PER_SYM_HALF);
    event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e1); WaitFlag<HardEvent::MTE2_V>(e1);



    Add(tmp, reSym2, reSym11, K_DMRS);    PipeBarrier<PIPE_V>();
    Muls(tmp, tmp, (half)0.5f, K_DMRS);   PipeBarrier<PIPE_V>();
    Duplicate(hpadRe, (half)0.0f, N_FFT); PipeBarrier<PIPE_V>();

    Adds(hpadRe, tmp, (half)0.0f, K_DMRS); PipeBarrier<PIPE_V>();

    Add(tmp, imSym2, imSym11, K_DMRS);    PipeBarrier<PIPE_V>();
    Muls(tmp, tmp, (half)0.5f, K_DMRS);   PipeBarrier<PIPE_V>();
    Duplicate(hpadIm, (half)0.0f, N_FFT); PipeBarrier<PIPE_V>();
    Adds(hpadIm, tmp, (half)0.0f, K_DMRS); PipeBarrier<PIPE_V>();


    event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e2); WaitFlag<HardEvent::V_MTE3>(e2);
    DataCopy(HpadReScr_, hpadRe, N_FFT);
    DataCopy(HpadImScr_, hpadIm, N_FFT);
    event_t e3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e3); WaitFlag<HardEvent::MTE3_MTE2>(e3);
}














__aicore__ inline void TimingTracker::RunCubeP1(
    const LocalTensor<half> &Al1,
    const GlobalTensor<half> &Bgm,
    const LocalTensor<float> &dst)
{
    constexpr uint16_t mBlocks = M_MMAD / 16;
    constexpr uint16_t kBlocks = K_MMAD / 16;

    for (uint16_t ns = 0; ns < N_SPLITS; ns++) {
        uint32_t bOff   = ns * N_SUB;
        uint32_t dstOff = ns * M_MMAD * N_SUB;


        LocalTensor<half> b1 = qB1_.AllocTensor<half>();
        DataCopy(b1, Bgm[bOff], { K_MMAD, 1, uint16_t(Q / 16 - 1), 0 });
        qB1_.EnQue(b1);


        LocalTensor<half> a2 = qA2_.AllocTensor<half>();
        LoadL1ToL0A(a2, Al1, mBlocks, kBlocks);
        qA2_.EnQue(a2);


        b1 = qB1_.DeQue<half>();
        LocalTensor<half> b2 = qB2_.AllocTensor<half>();
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks;
        loadP.srcStride   = 1;
        loadP.ifTranspose = true;
        LoadData(b2, b1, loadP);
        qB2_.EnQue(b2); qB1_.FreeTensor(b1);


        a2 = qA2_.DeQue<half>(); b2 = qB2_.DeQue<half>();
        LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
        MmadParams mp;
        mp.m = M_MMAD; mp.n = N_SUB; mp.k = K_MMAD;
        Mmad(co1, a2, b2, mp);
        qCO1_.EnQue(co1); qA2_.FreeTensor(a2); qB2_.FreeTensor(b2);


        co1 = qCO1_.DeQue<float>();
        DataCopyParams dcp;
        dcp.blockCount = 1;
        dcp.blockLen   = uint16_t(M_MMAD * N_SUB / 256);
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(dst[dstOff], co1, dcp, ep);
        qCO1_.FreeTensor(co1);
    }
}

__attribute__((noinline)) __aicore__ void TimingTracker::Phase1_IdftP()
{
    if (blockId_ != 0) return;

    auto cubeAcc = bufCubeAcc_.Get<float>();
    auto cubeUB  = bufCubeUB_.Get<float>();
    auto cubeNd  = bufCubeNd_.Get<float>();


    auto wReL1 = bufWreL1_.Get<half>();
    auto wImL1 = bufWimL1_.Get<half>();
    CopyNd2Nz(wReL1, wReG_, M_MMAD, K_MMAD);
    CopyNd2Nz(wImL1, wImG_, M_MMAD, K_MMAD);
    PipeBarrier<PIPE_ALL>();


    RunCubeP1(wReL1, HpadReScr_, cubeAcc);
    RunCubeP1(wImL1, HpadImScr_, cubeUB);
    PipeBarrier<PIPE_ALL>();
    Sub(cubeAcc, cubeAcc, cubeUB, TILE_PQ);
    PipeBarrier<PIPE_V>();
    auto x1reUB = bufHpadRe_.Get<half>();
    NzToNdAndCastHalf(x1reUB, cubeAcc, cubeNd);
    event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e1); WaitFlag<HardEvent::V_MTE3>(e1);
    DataCopy(X1ReScr_, x1reUB, TILE_PQ);


    RunCubeP1(wReL1, HpadImScr_, cubeAcc);
    RunCubeP1(wImL1, HpadReScr_, cubeUB);
    PipeBarrier<PIPE_ALL>();
    Add(cubeAcc, cubeAcc, cubeUB, TILE_PQ);
    PipeBarrier<PIPE_V>();
    auto x1imUB = bufHpadIm_.Get<half>();
    NzToNdAndCastHalf(x1imUB, cubeAcc, cubeNd);
    event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e2); WaitFlag<HardEvent::V_MTE3>(e2);
    DataCopy(X1ImScr_, x1imUB, TILE_PQ);

    PipeBarrier<PIPE_ALL>();
}









__attribute__((noinline)) __aicore__ void TimingTracker::Phase2_Twiddle()
{
    if (blockId_ != 0) return;

    auto x1re = bufHpadRe_.Get<half>();
    auto x1im = bufHpadIm_.Get<half>();
    auto twre = bufHcirRe_.Get<half>();
    auto twim = bufHcirIm_.Get<half>();
    auto out  = bufTmp_.Get<half>();
    auto tmp  = bufOutRe_.Get<half>();


    DataCopy(x1re, X1ReScr_, TILE_PQ);
    DataCopy(x1im, X1ImScr_, TILE_PQ);
    DataCopy(twre, twReG_, TILE_PQ);
    DataCopy(twim, twImG_, TILE_PQ);
    event_t e0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e0); WaitFlag<HardEvent::MTE2_V>(e0);


    Mul(out, x1re, twre, TILE_PQ);  PipeBarrier<PIPE_V>();
    Mul(tmp, x1im, twim, TILE_PQ);  PipeBarrier<PIPE_V>();
    Sub(out, out, tmp, TILE_PQ);    PipeBarrier<PIPE_V>();
    event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e1); WaitFlag<HardEvent::V_MTE3>(e1);
    DataCopy(TwReScr_, out, TILE_PQ);
    event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e2); WaitFlag<HardEvent::MTE3_V>(e2);


    Mul(out, x1re, twim, TILE_PQ);  PipeBarrier<PIPE_V>();
    Mul(tmp, x1im, twre, TILE_PQ);  PipeBarrier<PIPE_V>();
    Add(out, out, tmp, TILE_PQ);    PipeBarrier<PIPE_V>();
    event_t e3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e3); WaitFlag<HardEvent::V_MTE3>(e3);
    DataCopy(TwImScr_, out, TILE_PQ);
    event_t e4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e4); WaitFlag<HardEvent::MTE3_V>(e4);

    PipeBarrier<PIPE_ALL>();
}











__aicore__ inline void TimingTracker::RunCubeP3(
    const GlobalTensor<half> &Agm,
    const LocalTensor<half> &Bl1,
    const LocalTensor<float> &dst)
{
    constexpr uint16_t mBlocks = M_MMAD / 16;
    constexpr uint16_t kBlocks = K_MMAD / 16;

    for (uint16_t ns = 0; ns < N_SPLITS; ns++) {
        uint32_t dstOff = ns * M_MMAD * N_SUB;


        LocalTensor<half> a1 = qA1_.AllocTensor<half>();
        for (uint16_t i = 0; i < kBlocks; ++i) {
            DataCopy(a1[i * 16 * M_MMAD], Agm[i * 16],
                     { M_MMAD, 1, uint16_t(K_MMAD / 16 - 1), 0 });
        }
        qA1_.EnQue(a1);

        a1 = qA1_.DeQue<half>();
        LocalTensor<half> a2 = qA2_.AllocTensor<half>();
        LoadL1ToL0A(a2, a1, mBlocks, kBlocks);
        qA2_.EnQue(a2); qA1_.FreeTensor(a1);


        LocalTensor<half> b2 = qB2_.AllocTensor<half>();
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks;
        loadP.srcStride   = 1;
        loadP.ifTranspose = true;
        LoadData(b2, Bl1[ns * kBlocks * 256], loadP);
        qB2_.EnQue(b2);


        a2 = qA2_.DeQue<half>(); b2 = qB2_.DeQue<half>();
        LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
        MmadParams mp;
        mp.m = M_MMAD; mp.n = N_SUB; mp.k = K_MMAD;
        Mmad(co1, a2, b2, mp);
        qCO1_.EnQue(co1); qA2_.FreeTensor(a2); qB2_.FreeTensor(b2);

        co1 = qCO1_.DeQue<float>();
        DataCopyParams dcp;
        dcp.blockCount = 1;
        dcp.blockLen   = uint16_t(M_MMAD * N_SUB / 256);
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(dst[dstOff], co1, dcp, ep);
        qCO1_.FreeTensor(co1);
    }
}

__attribute__((noinline)) __aicore__ void TimingTracker::Phase3_IdftQ()
{
    if (blockId_ != 0) return;

    auto cubeAcc = bufCubeAcc_.Get<float>();
    auto cubeUB  = bufCubeUB_.Get<float>();
    auto cubeNd  = bufCubeNd_.Get<float>();


    auto wReL1B = bufWreL1B_.Get<half>();
    auto wImL1B = bufWimL1B_.Get<half>();
    CopyNd2Nz(wReL1B, wReG_, K_MMAD, Q);
    CopyNd2Nz(wImL1B, wImG_, K_MMAD, Q);
    PipeBarrier<PIPE_ALL>();


    RunCubeP3(TwReScr_, wReL1B, cubeAcc);
    RunCubeP3(TwImScr_, wImL1B, cubeUB);
    PipeBarrier<PIPE_ALL>();
    Sub(cubeAcc, cubeAcc, cubeUB, TILE_PQ);
    PipeBarrier<PIPE_V>();
    auto hcirRe = bufHcirRe_.Get<half>();
    NzToNdAndCastHalf(hcirRe, cubeAcc, cubeNd);


    RunCubeP3(TwReScr_, wImL1B, cubeAcc);
    RunCubeP3(TwImScr_, wReL1B, cubeUB);
    PipeBarrier<PIPE_ALL>();
    Add(cubeAcc, cubeAcc, cubeUB, TILE_PQ);
    PipeBarrier<PIPE_V>();
    auto hcirIm = bufHcirIm_.Get<half>();
    NzToNdAndCastHalf(hcirIm, cubeAcc, cubeNd);

    PipeBarrier<PIPE_ALL>();
}








__attribute__((noinline)) __aicore__ void TimingTracker::Phase3p5_TransposeAndScale()
{
    if (blockId_ != 0) return;

    auto hcirRe = bufHcirRe_.Get<half>();
    auto hcirIm = bufHcirIm_.Get<half>();
    auto outRe  = bufOutRe_.Get<half>();
    auto outIm  = bufOutIm_.Get<half>();


    Muls(hcirRe, hcirRe, (half)OUTPUT_SCALE, TILE_PQ);  PipeBarrier<PIPE_V>();
    Muls(hcirIm, hcirIm, (half)OUTPUT_SCALE, TILE_PQ);  PipeBarrier<PIPE_V>();









    ConfusionTransposeOnlyTiling t;
    t.blockSize = 16;
    t.height    = Q;
    t.width     = P;
    t.highBlock = Q / 16;
    t.stride    = Q;
    t.repeat    = P / 16;






    ConfusionTransposeOnlyCompute<half>(outRe, hcirRe, t);
    PipeBarrier<PIPE_V>();
    ConfusionTransposeOnlyCompute<half>(outIm, hcirIm, t);
    PipeBarrier<PIPE_V>();


    event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e1); WaitFlag<HardEvent::V_MTE3>(e1);
    DataCopy(cirReG_, outRe, N_FFT);
    DataCopy(cirImG_, outIm, N_FFT);

    PipeBarrier<PIPE_ALL>();
}
























__attribute__((noinline)) __aicore__ void TimingTracker::Phase5to7_ArgmaxAndDeltaT()
{
    if (blockId_ != 0) return;

    auto outRe = bufOutRe_.Get<half>();
    auto outIm = bufOutIm_.Get<half>();
    auto winRe = bufHcirRe_.Get<half>();
    auto winIm = bufHcirIm_.Get<half>();
    auto pwr16 = bufTmp_.Get<half>();
    auto pwr32 = bufCubeAcc_.Get<float>();

    constexpr uint32_t W_PAD = 80;
    constexpr uint32_t W_TOTAL_PAD = 2 * W_PAD;
    constexpr uint32_t SEARCH_BEGIN = W_PAD - W_SEARCH;
    constexpr uint32_t SEARCH_END   = W_PAD + W_SEARCH;


    DataCopy(winRe,         outRe[N_FFT - W_PAD], W_PAD);
    DataCopy(winRe[W_PAD],  outRe,                W_PAD);
    DataCopy(winIm,         outIm[N_FFT - W_PAD], W_PAD);
    DataCopy(winIm[W_PAD],  outIm,                W_PAD);

    event_t e_mte_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e_mte_v); WaitFlag<HardEvent::MTE3_V>(e_mte_v);


    Mul(pwr16, winRe, winRe, W_TOTAL_PAD);  PipeBarrier<PIPE_V>();

    auto tmp16 = winRe;
    Mul(tmp16, winIm, winIm, W_TOTAL_PAD);  PipeBarrier<PIPE_V>();
    Add(pwr16, pwr16, tmp16, W_TOTAL_PAD);  PipeBarrier<PIPE_V>();


    Cast(pwr32, pwr16, RoundMode::CAST_NONE, W_TOTAL_PAD);
    PipeBarrier<PIPE_V>();


    Duplicate(pwr32,                (float)(-1.0f), SEARCH_BEGIN);
    Duplicate(pwr32[SEARCH_END],    (float)(-1.0f), W_TOTAL_PAD - SEARCH_END);
    PipeBarrier<PIPE_V>();




    auto reduceDst = bufCubeUB_.Get<float>();
    auto reduceWk  = bufCubeNd_.Get<float>();
    ReduceMax<float>(reduceDst, pwr32, reduceWk, (int32_t)W_TOTAL_PAD,  true);
    PipeBarrier<PIPE_V>();


    event_t e_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e_v_s); WaitFlag<HardEvent::V_S>(e_v_s);

    float   max_val       = reduceDst.GetValue(0);
    float   idx_bits      = reduceDst.GetValue(1);
    int32_t max_idx_local = reinterpret_cast<int32_t&>(idx_bits);


    float offset = 0.0f;
    if (max_idx_local > static_cast<int32_t>(SEARCH_BEGIN) &&
        max_idx_local < static_cast<int32_t>(SEARCH_END - 1)) {
        float ym = pwr32.GetValue(max_idx_local - 1);
        float yp = pwr32.GetValue(max_idx_local + 1);
        float denom = ym - 2.0f * max_val + yp;
        if (denom < -1e-12f) {
            offset = 0.5f * (ym - yp) / denom;
            if (offset < -0.5f) offset = -0.5f;
            if (offset >  0.5f) offset =  0.5f;
        }
    }





    float dt_scale = dtScaleG_.GetValue(0);
    float tau_int  = static_cast<float>(max_idx_local - static_cast<int32_t>(W_PAD));
    float tau_fine = tau_int + offset;
    float delta_T  = -tau_fine * dt_scale;

    deltaTG_.SetValue(0, delta_T);

    PipeBarrier<PIPE_ALL>();
}





__aicore__ inline void TimingTracker::Process()
{
    Phase0_LoadAndPad();                  AscendC::SyncAll();
    Phase1_IdftP();                       AscendC::SyncAll();
    Phase2_Twiddle();                     AscendC::SyncAll();
    Phase3_IdftQ();                       AscendC::SyncAll();
    Phase3p5_TransposeAndScale();         AscendC::SyncAll();
    Phase5to7_ArgmaxAndDeltaT();          AscendC::SyncAll();
}





extern "C" __global__ __aicore__ void timing_tracker_kernel(
    GM_ADDR h_re_gm,    GM_ADDR h_im_gm,
    GM_ADDR w_re_gm,    GM_ADDR w_im_gm,
    GM_ADDR tw_re_gm,   GM_ADDR tw_im_gm,
    GM_ADDR dt_scale_gm,
    GM_ADDR scratch_gm,
    GM_ADDR cir_re_gm,  GM_ADDR cir_im_gm,
    GM_ADDR delta_T_gm,
    GM_ADDR ws,
    GM_ADDR tilingGm)
{
    TPipe pipe;
    TimingTracker op;
    op.Init(h_re_gm, h_im_gm,
            w_re_gm, w_im_gm,
            tw_re_gm, tw_im_gm,
            dt_scale_gm,
            cir_re_gm, cir_im_gm,
            delta_T_gm,
            scratch_gm,
            &pipe);
    op.Process();
}