



















#include "kernel_operator.h"
#include "ssb_fft.h"

using namespace AscendC;
using namespace ssb_fft;


class SsbFft {
public:
    __aicore__ inline SsbFft() {}

    __aicore__ inline void Init(GM_ADDR input_gm,
                                 GM_ADDR w16_re_gm, GM_ADDR w16_im_gm,
                                 GM_ADDR tw_re_gm,  GM_ADDR tw_im_gm,
                                 GM_ADDR gather_idx_gm,
                                 GM_ADDR derot_re_gm, GM_ADDR derot_im_gm,
                                 GM_ADDR output_re_gm, GM_ADDR output_im_gm,
                                 GM_ADDR scratch_gm,
                                 TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void Phase0_CopyIn();
    __aicore__ inline void Phase1_Dft16();
    __aicore__ inline void Phase2_Twiddle();
    __aicore__ inline void Phase3_Dft16();
    __aicore__ inline void Phase4_GatherDerotate();

    __aicore__ inline void LoadSymbol(uint32_t s);
    __aicore__ inline void CopyNd2Nz(const LocalTensor<half> &dst,
                                      const GlobalTensor<half> &src,
                                      uint16_t height, uint16_t width);
    __aicore__ inline void LoadL1ToL0A(const LocalTensor<half> &a2,
                                        const LocalTensor<half> &a1,
                                        uint16_t mBlocks, uint16_t kBlocks);
    __aicore__ inline void NzToNdAndCast(const LocalTensor<half> &dst,
                                          const LocalTensor<float> &src,
                                          const LocalTensor<float> &tmp);
    __aicore__ inline void MakeNeg(const LocalTensor<half> &dstL1,
                                    const LocalTensor<half> &srcL1,
                                    uint32_t n);

    __aicore__ inline void RunCubeP1_Fused(
        const LocalTensor<half> &Al1_1, const GlobalTensor<half> &Bgm_1,
        const LocalTensor<half> &Al1_2, const GlobalTensor<half> &Bgm_2,
        const LocalTensor<float> &dst);

    __aicore__ inline void RunCubeP3_Fused(
        const GlobalTensor<half> &Agm_1, const LocalTensor<half> &Bl1_1,
        const GlobalTensor<half> &Agm_2, const LocalTensor<half> &Bl1_2,
        const LocalTensor<float> &dst);

    TPipe   *pipe_;

    GlobalTensor<int16_t> inG_;
    GlobalTensor<half>    w16ReG_, w16ImG_;
    GlobalTensor<half>    twReG_,  twImG_;
    GlobalTensor<int32_t> gidxG_;
    GlobalTensor<half>    derotReG_, derotImG_;
    GlobalTensor<half>    outReG_, outImG_;
    GlobalTensor<half>    XreScr_, XimScr_, TwReScr_, TwImScr_;

    TBuf<TPosition::VECCALC> bufXi16_, bufXfullHalf_;
    TBuf<TPosition::VECCALC> bufXre_,  bufXim_;
    TBuf<TPosition::VECCALC> bufTmp_,  bufTmp2_;
    TBuf<TPosition::VECCALC> bufTwRe_, bufTwIm_;
    TBuf<TPosition::VECCALC> bufX1re_, bufX1im_;
    TBuf<TPosition::VECCALC> bufCubeAcc_, bufCubeNd_;
    TBuf<TPosition::VECCALC> bufGidx_, bufDerotRe_, bufDerotIm_;
    TBuf<TPosition::VECCALC> bufRre_,  bufRim_;

    TQue<TPosition::A1, 2>  qA1_;
    TQue<TPosition::A2, 2>  qA2_;
    TQue<TPosition::B1, 2>  qB1_;
    TQue<TPosition::B2, 2>  qB2_;
    TQue<TPosition::CO1, 2> qCO1_;


    TBuf<TPosition::A1> bufWreL1_, bufWimL1_, bufWimNegL1_;
};





__aicore__ inline void SsbFft::Init(
    GM_ADDR input_gm,
    GM_ADDR w16_re_gm, GM_ADDR w16_im_gm,
    GM_ADDR tw_re_gm,  GM_ADDR tw_im_gm,
    GM_ADDR gather_idx_gm,
    GM_ADDR derot_re_gm, GM_ADDR derot_im_gm,
    GM_ADDR output_re_gm, GM_ADDR output_im_gm,
    GM_ADDR scratch_gm,
    TPipe *pipe)
{
    pipe_ = pipe;

    inG_     .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(input_gm),     INPUT_GM_INT16_LEN);
    w16ReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w16_re_gm),       P * P);
    w16ImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w16_im_gm),       P * P);
    twReG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(tw_re_gm),        TILE_PQ);
    twImG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(tw_im_gm),        TILE_PQ);
    gidxG_   .SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gather_idx_gm), N_DMRS_RE);
    derotReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(derot_re_gm),     N_DMRS_RE);
    derotImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(derot_im_gm),     N_DMRS_RE);
    outReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output_re_gm),    N_DMRS_RE);
    outImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output_im_gm),    N_DMRS_RE);


    auto sbase = reinterpret_cast<__gm__ half *>(scratch_gm);
    XreScr_ .SetGlobalBuffer(sbase + 0 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    XimScr_ .SetGlobalBuffer(sbase + 1 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    TwReScr_.SetGlobalBuffer(sbase + 2 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    TwImScr_.SetGlobalBuffer(sbase + 3 * BATCH_X_ELEMS, BATCH_X_ELEMS);


    pipe_->InitBuffer(bufXi16_,      2 * N_FFT * sizeof(int16_t));
    pipe_->InitBuffer(bufXfullHalf_, 2 * N_FFT * sizeof(half));
    pipe_->InitBuffer(bufXre_,       TILE_PQ   * sizeof(half));
    pipe_->InitBuffer(bufXim_,       TILE_PQ   * sizeof(half));
    pipe_->InitBuffer(bufTmp_,       TILE_PQ   * sizeof(half));
    pipe_->InitBuffer(bufTmp2_,      TILE_PQ   * sizeof(half));
    pipe_->InitBuffer(bufTwRe_,      TILE_PQ   * sizeof(half));
    pipe_->InitBuffer(bufTwIm_,      TILE_PQ   * sizeof(half));
    pipe_->InitBuffer(bufX1re_,      BATCH_X_ELEMS * sizeof(half));
    pipe_->InitBuffer(bufX1im_,      BATCH_X_ELEMS * sizeof(half));
    pipe_->InitBuffer(bufCubeAcc_,   TILE_PQ * sizeof(float));
    pipe_->InitBuffer(bufCubeNd_,    TILE_PQ * sizeof(float));
    pipe_->InitBuffer(bufGidx_,      N_DMRS_RE * sizeof(int32_t));
    pipe_->InitBuffer(bufDerotRe_,   N_DMRS_RE * sizeof(half));
    pipe_->InitBuffer(bufDerotIm_,   N_DMRS_RE * sizeof(half));
    pipe_->InitBuffer(bufRre_,       N_DMRS_RE * sizeof(half));
    pipe_->InitBuffer(bufRim_,       N_DMRS_RE * sizeof(half));


    pipe_->InitBuffer(qA1_,  2, M_MMAD * Q * sizeof(half));
    pipe_->InitBuffer(qA2_,  2, M_MMAD * Q * sizeof(half));
    pipe_->InitBuffer(qB1_,  2, Q * Q * sizeof(half));
    pipe_->InitBuffer(qB2_,  2, Q * Q * sizeof(half));
    pipe_->InitBuffer(qCO1_, 2, M_MMAD * Q * sizeof(float));


    pipe_->InitBuffer(bufWreL1_,    P * P * sizeof(half));
    pipe_->InitBuffer(bufWimL1_,    P * P * sizeof(half));
    pipe_->InitBuffer(bufWimNegL1_, P * P * sizeof(half));
}





__aicore__ inline void SsbFft::CopyNd2Nz(
    const LocalTensor<half> &dst, const GlobalTensor<half> &src,
    uint16_t height, uint16_t width)
{
    for (uint16_t i = 0; i < width / 16; ++i) {
        DataCopy(dst[i * 16 * height], src[i * 16],
                 { height, 1, uint16_t(width / 16 - 1), 0 });
    }
}

__aicore__ inline void SsbFft::LoadL1ToL0A(
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

__aicore__ inline void SsbFft::NzToNdAndCast(
    const LocalTensor<half> &dst, const LocalTensor<float> &src,
    const LocalTensor<float> &tmp)
{
    DataCopyParams dcp;
    dcp.blockCount = M_MMAD;
    dcp.blockLen   = uint16_t(N_SUB * sizeof(float) / 32);
    dcp.srcStride  = 0;
    dcp.dstStride  = uint16_t((Q - N_SUB) * sizeof(float) / 32);

    for (uint16_t ns = 0; ns < N_SPLITS; ns++) {
        DataCopy(tmp[ns * N_SUB], src[ns * M_MMAD * N_SUB], dcp);
    }
    PipeBarrier<PIPE_V>();
    Cast(dst, tmp, RoundMode::CAST_NONE, M_MMAD * Q);
}


__aicore__ inline void SsbFft::MakeNeg(
    const LocalTensor<half> &dstL1, const LocalTensor<half> &srcL1, uint32_t n)
{
    auto tmpUB = bufTmp_.Get<half>();
    DataCopy(tmpUB, srcL1, n);
    PipeBarrier<PIPE_ALL>();
    Muls(tmpUB, tmpUB, (half)-1.0, n);
    PipeBarrier<PIPE_V>();
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e); WaitFlag<HardEvent::V_MTE3>(e);
    DataCopy(dstL1, tmpUB, n);
    PipeBarrier<PIPE_ALL>();
}





__aicore__ inline void SsbFft::LoadSymbol(uint32_t s)
{
    auto xi16      = bufXi16_.Get<int16_t>();
    auto xfullHalf = bufXfullHalf_.Get<half>();
    auto xre       = bufXre_.Get<half>();
    auto xim       = bufXim_.Get<half>();

    uint32_t base = s * SYM_STRIDE + SSB_CP;
    DataCopy(xi16, inG_[base * 2], 2 * N_FFT);
    event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e1); WaitFlag<HardEvent::MTE2_V>(e1);

    Cast(xfullHalf, xi16, RoundMode::CAST_NONE, 2 * N_FFT);
    PipeBarrier<PIPE_V>();
    Muls(xfullHalf, xfullHalf, (half)INPUT_SCALE, 2 * N_FFT);
    PipeBarrier<PIPE_V>();

    uint64_t rsvdCnt = 0;
    GatherMaskParams gmp;
    gmp.src0BlockStride  = 1;
    gmp.repeatTimes      = uint16_t(2 * N_FFT / 128);
    gmp.src0RepeatStride = 8;
    gmp.src1RepeatStride = 0;
    GatherMask(xre, xfullHalf, (uint8_t)1, false, 0, gmp, rsvdCnt);
    GatherMask(xim, xfullHalf, (uint8_t)2, false, 0, gmp, rsvdCnt);
    PipeBarrier<PIPE_V>();

    event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e2); WaitFlag<HardEvent::V_MTE3>(e2);
    DataCopy(XreScr_[s * TILE_PQ], xre, TILE_PQ);
    DataCopy(XimScr_[s * TILE_PQ], xim, TILE_PQ);
    event_t e3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e3); WaitFlag<HardEvent::MTE3_MTE2>(e3);
}

__aicore__ inline void SsbFft::Phase0_CopyIn()
{
    for (uint32_t s = 0; s < N_SYMBOL; ++s) LoadSymbol(s);
}






__aicore__ inline void SsbFft::RunCubeP1_Fused(
    const LocalTensor<half> &Al1_1, const GlobalTensor<half> &Bgm_1,
    const LocalTensor<half> &Al1_2, const GlobalTensor<half> &Bgm_2,
    const LocalTensor<float> &dst)
{
    constexpr uint16_t mBlocks = M_MMAD / 16;
    constexpr uint16_t kBlocks = K_PHASE1 / 16;

    for (uint16_t ns = 0; ns < N_SPLITS; ns++) {
        uint32_t bOff   = ns * N_SUB;
        uint32_t dstOff = ns * M_MMAD * N_SUB;


        LocalTensor<half> b1_1 = qB1_.AllocTensor<half>();
        DataCopy(b1_1, Bgm_1[bOff], { K_PHASE1, 1, uint16_t(Q / 16 - 1), 0 });
        qB1_.EnQue(b1_1);

        LocalTensor<half> a2_1 = qA2_.AllocTensor<half>();
        LoadL1ToL0A(a2_1, Al1_1, mBlocks, kBlocks);
        qA2_.EnQue(a2_1);

        b1_1 = qB1_.DeQue<half>();
        LocalTensor<half> b2_1 = qB2_.AllocTensor<half>();
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks;
        loadP.srcStride   = 1;
        loadP.ifTranspose = true;
        LoadData(b2_1, b1_1, loadP);
        qB2_.EnQue(b2_1); qB1_.FreeTensor(b1_1);

        a2_1 = qA2_.DeQue<half>(); b2_1 = qB2_.DeQue<half>();
        LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
        MmadParams mp;
        mp.m = M_MMAD; mp.n = N_SUB; mp.k = K_PHASE1;
        mp.cmatrixInitVal = true;
        Mmad(co1, a2_1, b2_1, mp);
        qA2_.FreeTensor(a2_1); qB2_.FreeTensor(b2_1);


        LocalTensor<half> b1_2 = qB1_.AllocTensor<half>();
        DataCopy(b1_2, Bgm_2[bOff], { K_PHASE1, 1, uint16_t(Q / 16 - 1), 0 });
        qB1_.EnQue(b1_2);

        LocalTensor<half> a2_2 = qA2_.AllocTensor<half>();
        LoadL1ToL0A(a2_2, Al1_2, mBlocks, kBlocks);
        qA2_.EnQue(a2_2);

        b1_2 = qB1_.DeQue<half>();
        LocalTensor<half> b2_2 = qB2_.AllocTensor<half>();
        LoadData(b2_2, b1_2, loadP);
        qB2_.EnQue(b2_2); qB1_.FreeTensor(b1_2);

        a2_2 = qA2_.DeQue<half>(); b2_2 = qB2_.DeQue<half>();
        mp.cmatrixInitVal = false;
        Mmad(co1, a2_2, b2_2, mp);
        qCO1_.EnQue(co1);
        qA2_.FreeTensor(a2_2); qB2_.FreeTensor(b2_2);

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

__attribute__((noinline)) __aicore__ void SsbFft::Phase1_Dft16()
{
    auto x1reBatch = bufX1re_.Get<half>();
    auto x1imBatch = bufX1im_.Get<half>();
    auto cubeAcc   = bufCubeAcc_.Get<float>();
    auto cubeNd    = bufCubeNd_.Get<float>();

    auto wReL1    = bufWreL1_.Get<half>();
    auto wImL1    = bufWimL1_.Get<half>();
    auto wImNegL1 = bufWimNegL1_.Get<half>();


    CopyNd2Nz(wReL1, w16ReG_, M_MMAD, K_PHASE1);
    CopyNd2Nz(wImL1, w16ImG_, M_MMAD, K_PHASE1);
    PipeBarrier<PIPE_ALL>();
    MakeNeg(wImNegL1, wImL1, P * P);

    for (uint32_t s = 0; s < N_SYMBOL; ++s) {
        uint32_t symOff = s * TILE_PQ;
        auto xreSym = XreScr_[symOff];
        auto ximSym = XimScr_[symOff];

        RunCubeP1_Fused(wReL1, xreSym, wImNegL1, ximSym, cubeAcc);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(x1reBatch[symOff], cubeAcc, cubeNd);

        RunCubeP1_Fused(wReL1, ximSym, wImL1, xreSym, cubeAcc);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(x1imBatch[symOff], cubeAcc, cubeNd);
    }
    PipeBarrier<PIPE_ALL>();
}






__attribute__((noinline)) __aicore__ void SsbFft::Phase2_Twiddle()
{
    auto tmp       = bufTmp_.Get<half>();
    auto tmp2      = bufTmp2_.Get<half>();
    auto twre      = bufTwRe_.Get<half>();
    auto twim      = bufTwIm_.Get<half>();
    auto x1reBatch = bufX1re_.Get<half>();
    auto x1imBatch = bufX1im_.Get<half>();

    DataCopy(twre, twReG_, TILE_PQ);
    DataCopy(twim, twImG_, TILE_PQ);
    event_t e0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e0); WaitFlag<HardEvent::MTE2_V>(e0);

    for (uint32_t s = 0; s < N_SYMBOL; ++s) {
        uint32_t symOff = s * TILE_PQ;
        auto xreS = x1reBatch[symOff];
        auto ximS = x1imBatch[symOff];

        Mul(tmp,  xreS, twre, TILE_PQ); PipeBarrier<PIPE_V>();
        Mul(tmp2, ximS, twim, TILE_PQ); PipeBarrier<PIPE_V>();
        Sub(tmp, tmp, tmp2, TILE_PQ);   PipeBarrier<PIPE_V>();
        event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e1); WaitFlag<HardEvent::V_MTE3>(e1);
        DataCopy(TwReScr_[symOff], tmp, TILE_PQ);
        event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e2); WaitFlag<HardEvent::MTE3_V>(e2);

        Mul(tmp,  xreS, twim, TILE_PQ); PipeBarrier<PIPE_V>();
        Mul(tmp2, ximS, twre, TILE_PQ); PipeBarrier<PIPE_V>();
        Add(tmp, tmp, tmp2, TILE_PQ);   PipeBarrier<PIPE_V>();
        event_t e3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e3); WaitFlag<HardEvent::V_MTE3>(e3);
        DataCopy(TwImScr_[symOff], tmp, TILE_PQ);
        event_t e4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e4); WaitFlag<HardEvent::MTE3_V>(e4);
    }
    PipeBarrier<PIPE_ALL>();
}







__aicore__ inline void SsbFft::RunCubeP3_Fused(
    const GlobalTensor<half> &Agm_1, const LocalTensor<half> &Bl1_1,
    const GlobalTensor<half> &Agm_2, const LocalTensor<half> &Bl1_2,
    const LocalTensor<float> &dst)
{
    constexpr uint16_t mBlocks = M_MMAD / 16;
    constexpr uint16_t kBlocks = K_PHASE3 / 16;

    for (uint16_t ns = 0; ns < N_SPLITS; ns++) {
        uint32_t dstOff = ns * M_MMAD * N_SUB;


        LocalTensor<half> a1_1 = qA1_.AllocTensor<half>();
        for (uint16_t i = 0; i < kBlocks; ++i) {
            DataCopy(a1_1[i * 16 * M_MMAD], Agm_1[i * 16],
                     { M_MMAD, 1, uint16_t(K_PHASE3 / 16 - 1), 0 });
        }
        qA1_.EnQue(a1_1);

        a1_1 = qA1_.DeQue<half>();
        LocalTensor<half> a2_1 = qA2_.AllocTensor<half>();
        LoadL1ToL0A(a2_1, a1_1, mBlocks, kBlocks);
        qA2_.EnQue(a2_1); qA1_.FreeTensor(a1_1);

        LocalTensor<half> b2_1 = qB2_.AllocTensor<half>();
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks;
        loadP.srcStride   = 1;
        loadP.ifTranspose = true;
        LoadData(b2_1, Bl1_1[ns * kBlocks * 256], loadP);
        qB2_.EnQue(b2_1);

        a2_1 = qA2_.DeQue<half>(); b2_1 = qB2_.DeQue<half>();
        LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
        MmadParams mp;
        mp.m = M_MMAD; mp.n = N_SUB; mp.k = K_PHASE3;
        mp.cmatrixInitVal = true;
        Mmad(co1, a2_1, b2_1, mp);
        qA2_.FreeTensor(a2_1); qB2_.FreeTensor(b2_1);


        LocalTensor<half> a1_2 = qA1_.AllocTensor<half>();
        for (uint16_t i = 0; i < kBlocks; ++i) {
            DataCopy(a1_2[i * 16 * M_MMAD], Agm_2[i * 16],
                     { M_MMAD, 1, uint16_t(K_PHASE3 / 16 - 1), 0 });
        }
        qA1_.EnQue(a1_2);

        a1_2 = qA1_.DeQue<half>();
        LocalTensor<half> a2_2 = qA2_.AllocTensor<half>();
        LoadL1ToL0A(a2_2, a1_2, mBlocks, kBlocks);
        qA2_.EnQue(a2_2); qA1_.FreeTensor(a1_2);

        LocalTensor<half> b2_2 = qB2_.AllocTensor<half>();
        LoadData(b2_2, Bl1_2[ns * kBlocks * 256], loadP);
        qB2_.EnQue(b2_2);

        a2_2 = qA2_.DeQue<half>(); b2_2 = qB2_.DeQue<half>();
        mp.cmatrixInitVal = false;
        Mmad(co1, a2_2, b2_2, mp);
        qCO1_.EnQue(co1);
        qA2_.FreeTensor(a2_2); qB2_.FreeTensor(b2_2);

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

__attribute__((noinline)) __aicore__ void SsbFft::Phase3_Dft16()
{
    auto x2reBatch = bufX1re_.Get<half>();
    auto x2imBatch = bufX1im_.Get<half>();
    auto cubeAcc   = bufCubeAcc_.Get<float>();
    auto cubeNd    = bufCubeNd_.Get<float>();

    auto wReL1    = bufWreL1_.Get<half>();
    auto wImL1    = bufWimL1_.Get<half>();
    auto wImNegL1 = bufWimNegL1_.Get<half>();

    for (uint32_t s = 0; s < N_SYMBOL; ++s) {
        uint32_t symOff = s * TILE_PQ;
        auto twreSym = TwReScr_[symOff];
        auto twimSym = TwImScr_[symOff];


        RunCubeP3_Fused(twreSym, wReL1, twimSym, wImNegL1, cubeAcc);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(x2reBatch[symOff], cubeAcc, cubeNd);


        RunCubeP3_Fused(twreSym, wImL1, twimSym, wReL1, cubeAcc);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(x2imBatch[symOff], cubeAcc, cubeNd);
    }
    PipeBarrier<PIPE_ALL>();
}







__attribute__((noinline)) __aicore__ void SsbFft::Phase4_GatherDerotate()
{
    auto x2re = bufX1re_.Get<half>();
    auto x2im = bufX1im_.Get<half>();
    auto gidx = bufGidx_.Get<int32_t>();
    auto dre  = bufDerotRe_.Get<half>();
    auto dim  = bufDerotIm_.Get<half>();
    auto gre  = bufRre_.Get<half>();
    auto gim  = bufRim_.Get<half>();
    auto tmp  = bufTmp_.Get<half>();
    auto tmp2 = bufTmp2_.Get<half>();

    DataCopy(gidx, gidxG_, N_DMRS_RE);
    DataCopy(dre,  derotReG_, N_DMRS_RE);
    DataCopy(dim,  derotImG_, N_DMRS_RE);
    PipeBarrier<PIPE_ALL>();


    for (uint32_t i = 0; i < N_DMRS_RE; ++i) {
        int32_t k = gidx.GetValue(i);
        gre.SetValue(i, x2re.GetValue(static_cast<uint32_t>(k)));
        gim.SetValue(i, x2im.GetValue(static_cast<uint32_t>(k)));
    }
    PipeBarrier<PIPE_ALL>();


    Mul(tmp,  gre, dre, N_DMRS_RE); PipeBarrier<PIPE_V>();
    Mul(tmp2, gim, dim, N_DMRS_RE); PipeBarrier<PIPE_V>();
    Sub(tmp,  tmp, tmp2, N_DMRS_RE); PipeBarrier<PIPE_V>();
    event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e1); WaitFlag<HardEvent::V_MTE3>(e1);
    DataCopy(outReG_, tmp, N_DMRS_RE);

    event_t e1b = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e1b); WaitFlag<HardEvent::MTE3_V>(e1b);

    Mul(tmp,  gre, dim, N_DMRS_RE); PipeBarrier<PIPE_V>();
    Mul(tmp2, gim, dre, N_DMRS_RE); PipeBarrier<PIPE_V>();
    Add(tmp,  tmp, tmp2, N_DMRS_RE); PipeBarrier<PIPE_V>();
    event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e2); WaitFlag<HardEvent::V_MTE3>(e2);
    DataCopy(outImG_, tmp, N_DMRS_RE);
    PipeBarrier<PIPE_ALL>();
}





__aicore__ inline void SsbFft::Process()
{
    Phase0_CopyIn();
    Phase1_Dft16();
    Phase2_Twiddle();
    Phase3_Dft16();
    Phase4_GatherDerotate();
}





extern "C" __global__ __aicore__ void ssb_fft_kernel(
    GM_ADDR input_gm,
    GM_ADDR w16_re_gm,  GM_ADDR w16_im_gm,
    GM_ADDR tw_re_gm,   GM_ADDR tw_im_gm,
    GM_ADDR gather_idx_gm,
    GM_ADDR derot_re_gm, GM_ADDR derot_im_gm,
    GM_ADDR scratch_gm,
    GM_ADDR output_re_gm, GM_ADDR output_im_gm,
    GM_ADDR ws,
    GM_ADDR tilingGm)
{
    if (GetBlockIdx() != 0) return;
    TPipe pipe;
    SsbFft op;
    op.Init(input_gm,
            w16_re_gm, w16_im_gm,
            tw_re_gm,  tw_im_gm,
            gather_idx_gm,
            derot_re_gm, derot_im_gm,
            output_re_gm, output_im_gm,
            scratch_gm,
            &pipe);
    op.Process();
}