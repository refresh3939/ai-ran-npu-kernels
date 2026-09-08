


























#include "kernel_operator.h"
#include "ofdm_mod.h"

using namespace AscendC;
using namespace ofdm_mod;


class OfdmMod {
public:
    __aicore__ inline OfdmMod() {}

    __aicore__ inline void Init(GM_ADDR in_re_gm, GM_ADDR in_im_gm,
                                 GM_ADDR w32_re_gm, GM_ADDR w32_im_gm,
                                 GM_ADDR w64_re_gm, GM_ADDR w64_im_gm,
                                 GM_ADDR tw_re_gm,  GM_ADDR tw_im_gm,
                                 GM_ADDR output_re_gm, GM_ADDR output_im_gm, GM_ADDR output_iq_gm,
                                 GM_ADDR scratch_gm,
                                 TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void PhaseA_Idft64();
    __aicore__ inline void PhaseB_Twiddle();
    __aicore__ inline void PhaseC_Idft32();

    __aicore__ inline void CopyNd2Nz(const LocalTensor<half> &dst,
                                      const GlobalTensor<half> &src,
                                      uint16_t height, uint16_t width);
    __aicore__ inline void LoadL1ToL0A(const LocalTensor<half> &a2,
                                        const LocalTensor<half> &a1,
                                        uint16_t mBlocks, uint16_t kBlocks);
    __aicore__ inline void NzToNdAndCast(const LocalTensor<half> &dst,
                                          const LocalTensor<float> &src,
                                          const LocalTensor<float> &tmp,
                                          uint16_t m = M_MMAD);

    __aicore__ inline void RunCubeP1_ComplexTileReuse(
        const LocalTensor<half> &wReL0,
        const LocalTensor<half> &wImL0,
        const LocalTensor<half> &wImNegL0,
        const GlobalTensor<half> &xReGm,
        const GlobalTensor<half> &xImGm,
        const LocalTensor<float> &dstRe,
        const LocalTensor<float> &dstIm);

    __aicore__ inline void RunCubeP3_Batch(
        const GlobalTensor<half> &Agm_1, const LocalTensor<half> &Bl0_1,
        const GlobalTensor<half> &Agm_2, const LocalTensor<half> &Bl0_2,
        const LocalTensor<float> &dst, uint16_t m);

    TPipe   *pipe_;
    uint32_t blockId_;
    uint32_t sym_start_;

    GlobalTensor<half>    inReG_,   inImG_;
    GlobalTensor<half>    w32ReG_,  w32ImG_;
    GlobalTensor<half>    w64ReG_T_, w64ImG_T_;
    GlobalTensor<half>    twReG_,   twImG_;
    GlobalTensor<int16_t> outReG_,  outImG_,  outIqG_;
    GlobalTensor<half>    X1ReScr_, X1ImScr_;

    TBuf<TPosition::VECCALC> bufTmp_,   bufTmp2_;
    TBuf<TPosition::VECCALC> bufTwRe_,  bufTwIm_;
    TBuf<TPosition::VECCALC> bufX1tre_, bufX1tim_;
    TBuf<TPosition::VECCALC> bufCubeAcc_, bufCubeNd_;
    TBuf<TPosition::VECCALC> bufI16_;
    TBuf<TPosition::VECCALC> bufReI32_, bufImI32_;

    TQue<TPosition::A1, 2>  qA1_;
    TQue<TPosition::A2, 2>  qA2_;
    TQue<TPosition::B1, 2>  qB1_;
    TQue<TPosition::B2, 2>  qB2_;
    TQue<TPosition::CO1, 2> qCO1_;

    TBuf<TPosition::A1> bufW32reL1_, bufW32imL1_, bufW32imNegL1_;
    TBuf<TPosition::B1> bufW64reL1_, bufW64imL1_, bufW64imNegL1_;

    TBuf<TPosition::A2> bufW32L0A_;
    TBuf<TPosition::B2> bufW64L0B_;
};


__aicore__ inline void OfdmMod::Init(
    GM_ADDR in_re_gm, GM_ADDR in_im_gm,
    GM_ADDR w32_re_gm, GM_ADDR w32_im_gm,
    GM_ADDR w64_re_gm, GM_ADDR w64_im_gm,
    GM_ADDR tw_re_gm,  GM_ADDR tw_im_gm,
    GM_ADDR output_re_gm, GM_ADDR output_im_gm, GM_ADDR output_iq_gm,
    GM_ADDR scratch_gm,
    TPipe *pipe)
{
    pipe_      = pipe;
    blockId_   = GetBlockIdx();
    sym_start_ = blockId_ * SYMBOLS_PER_CORE;

    inReG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(in_re_gm),     INPUT_GM_HALF_LEN);
    inImG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(in_im_gm),     INPUT_GM_HALF_LEN);
    w32ReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w32_re_gm),    P * P);
    w32ImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w32_im_gm),    P * P);
    w64ReG_T_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w64_re_gm),    Q * Q);
    w64ImG_T_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w64_im_gm),    Q * Q);
    twReG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(tw_re_gm),     TILE_PQ);
    twImG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(tw_im_gm),     TILE_PQ);
    outReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(output_re_gm), N_SAMPLE_PER_SLOT);
    outImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(output_im_gm), N_SAMPLE_PER_SLOT);
    outIqG_  .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(output_iq_gm), 2 * N_SAMPLE_PER_SLOT);

    constexpr uint32_t SCR_PER_CORE = 2 * BATCH_X_ELEMS;
    auto sbase = reinterpret_cast<__gm__ half *>(scratch_gm) + blockId_ * SCR_PER_CORE;
    X1ReScr_.SetGlobalBuffer(sbase + 0 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    X1ImScr_.SetGlobalBuffer(sbase + 1 * BATCH_X_ELEMS, BATCH_X_ELEMS);

    pipe_->InitBuffer(bufTmp_,       TILE_PQ   * sizeof(half));
    pipe_->InitBuffer(bufTmp2_,      TILE_PQ   * sizeof(half));
    pipe_->InitBuffer(bufTwRe_,      TILE_PQ   * sizeof(half));
    pipe_->InitBuffer(bufTwIm_,      TILE_PQ   * sizeof(half));
    pipe_->InitBuffer(bufX1tre_,     BATCH_X_ELEMS * sizeof(half));
    pipe_->InitBuffer(bufX1tim_,     BATCH_X_ELEMS * sizeof(half));
    pipe_->InitBuffer(bufCubeAcc_,   BATCH_X_ELEMS * sizeof(float));
    pipe_->InitBuffer(bufCubeNd_,    BATCH_X_ELEMS * sizeof(float));
    pipe_->InitBuffer(bufI16_,       TILE_PQ * sizeof(int16_t));
    pipe_->InitBuffer(bufReI32_,     TILE_PQ * sizeof(int32_t));
    pipe_->InitBuffer(bufImI32_,     TILE_PQ * sizeof(int32_t));

    pipe_->InitBuffer(qA1_,  2, MAX_M_BATCH * Q * sizeof(half));
    pipe_->InitBuffer(qA2_,  2, MAX_M_BATCH * Q * sizeof(half));
    pipe_->InitBuffer(qB1_,  2, Q * Q * sizeof(half));
    pipe_->InitBuffer(qB2_,  2, Q * Q * sizeof(half));
    pipe_->InitBuffer(qCO1_, 2, M_MMAD * Q * sizeof(float));

    pipe_->InitBuffer(bufW32reL1_,    P * P * sizeof(half));
    pipe_->InitBuffer(bufW32imL1_,    P * P * sizeof(half));
    pipe_->InitBuffer(bufW32imNegL1_, P * P * sizeof(half));
    pipe_->InitBuffer(bufW64reL1_,    Q * Q * sizeof(half));
    pipe_->InitBuffer(bufW64imL1_,    Q * Q * sizeof(half));
    pipe_->InitBuffer(bufW64imNegL1_, Q * Q * sizeof(half));
    pipe_->InitBuffer(bufW32L0A_, 3 * P * P * sizeof(half));
    pipe_->InitBuffer(bufW64L0B_, 3 * Q * Q * sizeof(half));
}



__aicore__ inline void OfdmMod::CopyNd2Nz(
    const LocalTensor<half> &dst, const GlobalTensor<half> &src,
    uint16_t height, uint16_t width)
{
    for (uint16_t i = 0; i < width / 16; ++i) {
        DataCopy(dst[i * 16 * height], src[i * 16],
                 { height, 1, uint16_t(width / 16 - 1), 0 });
    }
}

__aicore__ inline void OfdmMod::LoadL1ToL0A(
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

__aicore__ inline void OfdmMod::NzToNdAndCast(
    const LocalTensor<half> &dst,
    const LocalTensor<float> &src,
    const LocalTensor<float> &tmp,
    uint16_t m)
{
    event_t eMv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_V));
    SetFlag<HardEvent::M_V>(eMv); WaitFlag<HardEvent::M_V>(eMv);
    DataCopyParams dcp;
    dcp.blockCount = m;
    dcp.blockLen   = uint16_t(N_SUB * sizeof(float) / 32);
    dcp.srcStride  = 0;
    dcp.dstStride  = uint16_t((Q - N_SUB) * sizeof(float) / 32);
    for (uint16_t ns = 0; ns < N_SPLITS; ns++) {
        DataCopy(tmp[ns * N_SUB], src[ns * m * N_SUB], dcp);
    }
    PipeBarrier<PIPE_V>();
    Cast(dst, tmp, RoundMode::CAST_NONE, m * Q);
    event_t eVm = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_M));
    SetFlag<HardEvent::V_M>(eVm); WaitFlag<HardEvent::V_M>(eVm);
}




__aicore__ inline void OfdmMod::RunCubeP3_Batch(
    const GlobalTensor<half> &Agm_1, const LocalTensor<half> &Bl0_1,
    const GlobalTensor<half> &Agm_2, const LocalTensor<half> &Bl0_2,
    const LocalTensor<float> &dst, uint16_t m)
{
    constexpr uint16_t kBlocks = K_PHASE3 / 16;
    const uint16_t mBlocks = m / 16;


    LocalTensor<half> a1_1 = qA1_.AllocTensor<half>();
    LocalTensor<half> a1_2 = qA1_.AllocTensor<half>();
    for (uint16_t i = 0; i < kBlocks; ++i) {
        DataCopy(a1_1[i * 16 * m], Agm_1[i * 16],
                 { m, 1, uint16_t(kBlocks - 1), 0 });
        DataCopy(a1_2[i * 16 * m], Agm_2[i * 16],
                 { m, 1, uint16_t(kBlocks - 1), 0 });
    }
    qA1_.EnQue(a1_1); qA1_.EnQue(a1_2);
    a1_1 = qA1_.DeQue<half>(); a1_2 = qA1_.DeQue<half>();
    LocalTensor<half> a2_1 = qA2_.AllocTensor<half>();
    LocalTensor<half> a2_2 = qA2_.AllocTensor<half>();
    LoadL1ToL0A(a2_1, a1_1, mBlocks, kBlocks);
    LoadL1ToL0A(a2_2, a1_2, mBlocks, kBlocks);
    qA2_.EnQue(a2_1); qA2_.EnQue(a2_2);
    qA1_.FreeTensor(a1_1); qA1_.FreeTensor(a1_2);
    a2_1 = qA2_.DeQue<half>(); a2_2 = qA2_.DeQue<half>();

    for (uint16_t ns = 0; ns < N_SPLITS; ns++) {
        uint32_t dstOff = ns * m * N_SUB;
        uint32_t wOff = ns * K_PHASE3 * N_SUB;
        LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
        MmadParams mp; mp.m = m; mp.n = N_SUB; mp.k = K_PHASE3; mp.cmatrixInitVal = true;
        Mmad(co1, a2_1, Bl0_1[wOff], mp);
        mp.cmatrixInitVal = false;
        Mmad(co1, a2_2, Bl0_2[wOff], mp);
        qCO1_.EnQue(co1);

        co1 = qCO1_.DeQue<float>();
        DataCopyParams dcp; dcp.blockCount = 1; dcp.blockLen = uint16_t(m * N_SUB / 256);
        DataCopyEnhancedParams ep; ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(dst[dstOff], co1, dcp, ep);
        qCO1_.FreeTensor(co1);
    }
    qA2_.FreeTensor(a2_1); qA2_.FreeTensor(a2_2);
}

__attribute__((noinline)) __aicore__ void OfdmMod::PhaseA_Idft64()
{
    auto x1tre   = bufX1tre_.Get<half>();
    auto x1tim   = bufX1tim_.Get<half>();
    auto cubeAcc = bufCubeAcc_.Get<float>();
    auto cubeNd  = bufCubeNd_.Get<float>();

    auto w64ReL1    = bufW64reL1_.Get<half>();
    auto w64ImL1    = bufW64imL1_.Get<half>();
    auto w64ImNegL1 = bufW64imNegL1_.Get<half>();

    CopyNd2Nz(w64ReL1, w64ReG_T_, K_PHASE3, Q);
    CopyNd2Nz(w64ImL1, w64ImG_T_, K_PHASE3, Q);
    PipeBarrier<PIPE_ALL>();
    {
        auto bigTmp = bufX1tre_.Get<half>();
        DataCopy(bigTmp, w64ImL1, Q * Q);
        PipeBarrier<PIPE_ALL>();
        Muls(bigTmp, bigTmp, (half)-1.0, Q * Q);
        PipeBarrier<PIPE_V>();
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e); WaitFlag<HardEvent::V_MTE3>(e);
        DataCopy(w64ImNegL1, bigTmp, Q * Q);
        PipeBarrier<PIPE_ALL>();
    }



    auto w64L0 = bufW64L0B_.Get<half>();
    auto w64ReL0 = w64L0;
    auto w64ImL0 = w64L0[Q * Q];
    auto w64ImNegL0 = w64L0[2 * Q * Q];
    LoadData2DParams loadP;
    loadP.repeatTimes = K_PHASE3 / 16; loadP.srcStride = 1; loadP.ifTranspose = true;
    for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
        uint32_t l1Off = ns * K_PHASE3 * N_SUB;
        uint32_t l0Off = ns * K_PHASE3 * N_SUB;
        LoadData(w64ReL0[l0Off],    w64ReL1[l1Off],    loadP);
        LoadData(w64ImL0[l0Off],    w64ImL1[l1Off],    loadP);
        LoadData(w64ImNegL0[l0Off], w64ImNegL1[l1Off], loadP);
    }
    PipeBarrier<PIPE_ALL>();

    uint16_t validSymbols = uint16_t(N_SYMBOL - sym_start_);
    if (validSymbols > SYMBOLS_PER_CORE) validSymbols = SYMBOLS_PER_CORE;
    uint16_t m = validSymbols * P;
    auto s4reBatch = inReG_[sym_start_ * N_FFT];
    auto s4imBatch = inImG_[sym_start_ * N_FFT];


    RunCubeP3_Batch(s4reBatch, w64ReL0, s4imBatch, w64ImNegL0, cubeAcc, m);
    PipeBarrier<PIPE_V>();
    NzToNdAndCast(x1tre, cubeAcc, cubeNd, m);

    RunCubeP3_Batch(s4reBatch, w64ImL0, s4imBatch, w64ReL0, cubeAcc, m);
    PipeBarrier<PIPE_V>();
    NzToNdAndCast(x1tim, cubeAcc, cubeNd, m);

    PipeBarrier<PIPE_ALL>();
}



__attribute__((noinline)) __aicore__ void OfdmMod::PhaseB_Twiddle()
{
    auto tmp     = bufTmp_.Get<half>();
    auto tmp2    = bufTmp2_.Get<half>();
    auto twre    = bufTwRe_.Get<half>();
    auto twim    = bufTwIm_.Get<half>();
    auto x1tre   = bufX1tre_.Get<half>();
    auto x1tim   = bufX1tim_.Get<half>();

    DataCopy(twre, twReG_, TILE_PQ);
    DataCopy(twim, twImG_, TILE_PQ);
    event_t e0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e0); WaitFlag<HardEvent::MTE2_V>(e0);

    for (uint32_t s = 0; s < SYMBOLS_PER_CORE; ++s) {
        uint32_t sym = sym_start_ + s;
        if (sym >= N_SYMBOL) continue;
        uint32_t symOff = s * TILE_PQ;
        auto xreS = x1tre[symOff];
        auto ximS = x1tim[symOff];


        Mul(tmp,  xreS, twre, TILE_PQ); PipeBarrier<PIPE_V>();
        Mul(tmp2, ximS, twim, TILE_PQ); PipeBarrier<PIPE_V>();
        Sub(tmp, tmp, tmp2, TILE_PQ);   PipeBarrier<PIPE_V>();
        event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e1); WaitFlag<HardEvent::V_MTE3>(e1);
        DataCopy(X1ReScr_[symOff], tmp, TILE_PQ);
        event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e2); WaitFlag<HardEvent::MTE3_V>(e2);


        Mul(tmp,  xreS, twim, TILE_PQ); PipeBarrier<PIPE_V>();
        Mul(tmp2, ximS, twre, TILE_PQ); PipeBarrier<PIPE_V>();
        Add(tmp, tmp, tmp2, TILE_PQ);   PipeBarrier<PIPE_V>();
        event_t e3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e3); WaitFlag<HardEvent::V_MTE3>(e3);
        DataCopy(X1ImScr_[symOff], tmp, TILE_PQ);
        event_t e4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e4); WaitFlag<HardEvent::MTE3_V>(e4);
    }


    event_t eScratch = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eScratch); WaitFlag<HardEvent::MTE3_MTE2>(eScratch);
}





__aicore__ inline void OfdmMod::RunCubeP1_ComplexTileReuse(
    const LocalTensor<half> &wReL0,
    const LocalTensor<half> &wImL0,
    const LocalTensor<half> &wImNegL0,
    const GlobalTensor<half> &xReGm,
    const GlobalTensor<half> &xImGm,
    const LocalTensor<float> &dstRe,
    const LocalTensor<float> &dstIm)
{
    constexpr uint16_t kBlocks = K_PHASE1 / 16;
    LoadData2DParams loadP;
    loadP.repeatTimes = kBlocks;
    loadP.srcStride = 1;
    loadP.ifTranspose = true;
    DataCopyParams dcp;
    dcp.blockCount = 1;
    dcp.blockLen = uint16_t(M_MMAD * N_SUB / 256);
    DataCopyEnhancedParams ep;
    ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;

    for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
        const uint32_t bOff = ns * N_SUB;
        const uint32_t dstOff = ns * M_MMAD * N_SUB;
        MmadParams mp;
        mp.m = M_MMAD; mp.n = N_SUB; mp.k = K_PHASE1;


        LocalTensor<float> co = qCO1_.AllocTensor<float>();
        LocalTensor<half> b1 = qB1_.AllocTensor<half>();
        DataCopy(b1, xReGm[bOff], {K_PHASE1, 1, uint16_t(Q / 16 - 1), 0});
        qB1_.EnQue(b1); b1 = qB1_.DeQue<half>();
        LocalTensor<half> b2 = qB2_.AllocTensor<half>();
        LoadData(b2, b1, loadP);
        qB2_.EnQue(b2); qB1_.FreeTensor(b1); b2 = qB2_.DeQue<half>();
        mp.cmatrixInitVal = true;
        Mmad(co, wReL0, b2, mp);

        SetFlag<HardEvent::M_S>(EVENT_ID1);
        WaitFlag<HardEvent::M_S>(EVENT_ID1);
        qB2_.FreeTensor(b2);

        b1 = qB1_.AllocTensor<half>();
        DataCopy(b1, xImGm[bOff], {K_PHASE1, 1, uint16_t(Q / 16 - 1), 0});
        qB1_.EnQue(b1); b1 = qB1_.DeQue<half>();
        b2 = qB2_.AllocTensor<half>();
        LoadData(b2, b1, loadP);
        qB2_.EnQue(b2); qB1_.FreeTensor(b1); b2 = qB2_.DeQue<half>();
        mp.cmatrixInitVal = false;
        Mmad(co, wImNegL0, b2, mp);
        SetFlag<HardEvent::M_S>(EVENT_ID1);
        WaitFlag<HardEvent::M_S>(EVENT_ID1);
        qCO1_.EnQue(co); qB2_.FreeTensor(b2); co = qCO1_.DeQue<float>();
        DataCopy(dstRe[dstOff], co, dcp, ep);

        SetFlag<HardEvent::V_M>(EVENT_ID2);
        WaitFlag<HardEvent::V_M>(EVENT_ID2);
        qCO1_.FreeTensor(co);


        co = qCO1_.AllocTensor<float>();
        b1 = qB1_.AllocTensor<half>();
        DataCopy(b1, xImGm[bOff], {K_PHASE1, 1, uint16_t(Q / 16 - 1), 0});
        qB1_.EnQue(b1); b1 = qB1_.DeQue<half>();
        b2 = qB2_.AllocTensor<half>();
        LoadData(b2, b1, loadP);
        qB2_.EnQue(b2); qB1_.FreeTensor(b1); b2 = qB2_.DeQue<half>();
        mp.cmatrixInitVal = true;
        Mmad(co, wReL0, b2, mp);
        SetFlag<HardEvent::M_S>(EVENT_ID1);
        WaitFlag<HardEvent::M_S>(EVENT_ID1);
        qB2_.FreeTensor(b2);

        b1 = qB1_.AllocTensor<half>();
        DataCopy(b1, xReGm[bOff], {K_PHASE1, 1, uint16_t(Q / 16 - 1), 0});
        qB1_.EnQue(b1); b1 = qB1_.DeQue<half>();
        b2 = qB2_.AllocTensor<half>();
        LoadData(b2, b1, loadP);
        qB2_.EnQue(b2); qB1_.FreeTensor(b1); b2 = qB2_.DeQue<half>();
        mp.cmatrixInitVal = false;
        Mmad(co, wImL0, b2, mp);
        SetFlag<HardEvent::M_S>(EVENT_ID1);
        WaitFlag<HardEvent::M_S>(EVENT_ID1);
        qCO1_.EnQue(co); qB2_.FreeTensor(b2); co = qCO1_.DeQue<float>();
        DataCopy(dstIm[dstOff], co, dcp, ep);
        SetFlag<HardEvent::V_M>(EVENT_ID2);
        WaitFlag<HardEvent::V_M>(EVENT_ID2);
        qCO1_.FreeTensor(co);
    }
}

__attribute__((noinline)) __aicore__ void OfdmMod::PhaseC_Idft32()
{
    auto cubeAccRe = bufCubeAcc_.Get<float>();
    auto cubeAccIm = cubeAccRe[TILE_PQ];
    auto cubeNd  = bufCubeNd_.Get<float>();
    auto halfBuf = bufTmp_.Get<half>();
    auto i16Buf  = bufI16_.Get<int16_t>();
    auto reI32   = bufReI32_.Get<int32_t>();
    auto imI32   = bufImI32_.Get<int32_t>();

    auto w32ReL1    = bufW32reL1_.Get<half>();
    auto w32ImL1    = bufW32imL1_.Get<half>();
    auto w32ImNegL1 = bufW32imNegL1_.Get<half>();

    CopyNd2Nz(w32ReL1, w32ReG_, M_MMAD, K_PHASE1);
    CopyNd2Nz(w32ImL1, w32ImG_, M_MMAD, K_PHASE1);
    PipeBarrier<PIPE_ALL>();
    {
        auto tmpUB = bufTmp2_.Get<half>();
        DataCopy(tmpUB, w32ImL1, P * P);
        PipeBarrier<PIPE_ALL>();
        Muls(tmpUB, tmpUB, (half)-1.0, P * P);
        PipeBarrier<PIPE_V>();
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e); WaitFlag<HardEvent::V_MTE3>(e);
        DataCopy(w32ImNegL1, tmpUB, P * P);
        PipeBarrier<PIPE_ALL>();
    }


    auto w32L0 = bufW32L0A_.Get<half>();
    auto w32ReL0 = w32L0;
    auto w32ImL0 = w32L0[P * P];
    auto w32ImNegL0 = w32L0[2 * P * P];
    LoadL1ToL0A(w32ReL0,    w32ReL1,    M_MMAD / 16, K_PHASE1 / 16);
    LoadL1ToL0A(w32ImL0,    w32ImL1,    M_MMAD / 16, K_PHASE1 / 16);
    LoadL1ToL0A(w32ImNegL0, w32ImNegL1, M_MMAD / 16, K_PHASE1 / 16);
    PipeBarrier<PIPE_ALL>();

    for (uint32_t s = 0; s < SYMBOLS_PER_CORE; ++s) {
        uint32_t sym = sym_start_ + s;
        if (sym >= N_SYMBOL) continue;
        uint32_t symOff = s * TILE_PQ;
        auto x1reSym = X1ReScr_[symOff];
        auto x1imSym = X1ImScr_[symOff];

        uint32_t cp    = (sym == 0) ? CP_LEN_FIRST : CP_LEN_OTHER;
        uint32_t start = SYM0_CP_OFFSET + sym * SYM_STRIDE;


        RunCubeP1_ComplexTileReuse(w32ReL0, w32ImL0, w32ImNegL0,
                                   x1reSym, x1imSym, cubeAccRe, cubeAccIm);


        PipeBarrier<PIPE_V>();
        NzToNdAndCast(halfBuf, cubeAccRe, cubeNd);
        PipeBarrier<PIPE_V>();
        Muls(halfBuf, halfBuf, (half)OUT_SCALE, TILE_PQ);
        PipeBarrier<PIPE_V>();
        Cast(i16Buf, halfBuf, RoundMode::CAST_RINT, TILE_PQ);
        PipeBarrier<PIPE_V>();
        Cast(reI32, halfBuf, RoundMode::CAST_RINT, TILE_PQ);
        PipeBarrier<PIPE_V>();
        event_t er = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(er); WaitFlag<HardEvent::V_MTE3>(er);
        DataCopy(outReG_[start],      i16Buf,                TILE_PQ);
        DataCopy(outReG_[start - cp], i16Buf[TILE_PQ - cp],  cp);
        event_t er2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(er2); WaitFlag<HardEvent::MTE3_V>(er2);


        PipeBarrier<PIPE_V>();
        NzToNdAndCast(halfBuf, cubeAccIm, cubeNd);
        PipeBarrier<PIPE_V>();
        Muls(halfBuf, halfBuf, (half)OUT_SCALE, TILE_PQ);
        PipeBarrier<PIPE_V>();
        Cast(i16Buf, halfBuf, RoundMode::CAST_RINT, TILE_PQ);
        PipeBarrier<PIPE_V>();
        Cast(imI32, halfBuf, RoundMode::CAST_RINT, TILE_PQ);
        PipeBarrier<PIPE_V>();
        event_t ei = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ei); WaitFlag<HardEvent::V_MTE3>(ei);
        DataCopy(outImG_[start],      i16Buf,                TILE_PQ);
        DataCopy(outImG_[start - cp], i16Buf[TILE_PQ - cp],  cp);
        event_t ei2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(ei2); WaitFlag<HardEvent::MTE3_V>(ei2);


        Cast(cubeNd, reI32, RoundMode::CAST_NONE, TILE_PQ);
        PipeBarrier<PIPE_V>();
        Muls(cubeAccRe, cubeNd, (float)(1.0 / 65536.0), TILE_PQ);
        PipeBarrier<PIPE_V>();
        auto qI32 = cubeNd.ReinterpretCast<int32_t>();
        Cast(qI32, cubeAccRe, RoundMode::CAST_FLOOR, TILE_PQ);
        PipeBarrier<PIPE_V>();
        Muls(qI32, qI32, (int32_t)65536, TILE_PQ);
        PipeBarrier<PIPE_V>();
        Sub(reI32, reI32, qI32, TILE_PQ);
        PipeBarrier<PIPE_V>();
        Muls(imI32, imI32, (int32_t)65536, TILE_PQ);
        PipeBarrier<PIPE_V>();
        Add(reI32, reI32, imI32, TILE_PQ);
        PipeBarrier<PIPE_V>();
        auto iqI16 = reI32.ReinterpretCast<int16_t>();
        event_t eq = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eq); WaitFlag<HardEvent::V_MTE3>(eq);
        DataCopy(outIqG_[2 * start],        iqI16,                     2 * TILE_PQ);
        DataCopy(outIqG_[2 * (start - cp)], iqI16[2 * (TILE_PQ - cp)], 2 * cp);
        event_t eq2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eq2); WaitFlag<HardEvent::MTE3_V>(eq2);



        PipeBarrier<PIPE_ALL>();
    }

    PipeBarrier<PIPE_ALL>();
}


__aicore__ inline void OfdmMod::Process()
{
    PhaseA_Idft64();
    PhaseB_Twiddle();
    PhaseC_Idft32();
}



extern "C" __global__ __aicore__ void ofdm_mod_kernel(
    GM_ADDR in_re_gm,     GM_ADDR in_im_gm,
    GM_ADDR w32_re_gm,    GM_ADDR w32_im_gm,
    GM_ADDR w64_re_gm,    GM_ADDR w64_im_gm,
    GM_ADDR tw_re_gm,     GM_ADDR tw_im_gm,
    GM_ADDR scratch_gm,
    GM_ADDR output_re_gm, GM_ADDR output_im_gm, GM_ADDR output_iq_gm,
    GM_ADDR ws,
    GM_ADDR tilingGm)
{
    TPipe pipe;
    OfdmMod op;
    op.Init(in_re_gm, in_im_gm,
            w32_re_gm, w32_im_gm,
            w64_re_gm, w64_im_gm,
            tw_re_gm,  tw_im_gm,
            output_re_gm, output_im_gm, output_iq_gm,
            scratch_gm,
            &pipe);
    op.Process();
}
