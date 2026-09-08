



























#include "kernel_operator.h"
#include "ofdm_demod.h"

using namespace AscendC;
using namespace ofdm_demod;





class OfdmDemod {
public:
    __aicore__ inline OfdmDemod() {}

    __aicore__ inline void Init(GM_ADDR input_gm,
                                 GM_ADDR w32_re_gm, GM_ADDR w32_im_gm,
                                 GM_ADDR w64_re_gm, GM_ADDR w64_im_gm,
                                 GM_ADDR tw_re_gm,  GM_ADDR tw_im_gm,
                                 GM_ADDR output_re_gm, GM_ADDR output_im_gm,
                                 GM_ADDR scratch_gm,
#ifdef OFDM_DEMOD_BATCH
                                 uint32_t batch_size,
#endif
                                 TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void Phase0_CopyIn();
    __aicore__ inline void Phase1_Dft32();
    __aicore__ inline void Phase2_Twiddle();
    __aicore__ inline void Phase3_Dft64();

    __aicore__ inline void LoadSymbol(uint32_t s, uint32_t sym);
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

#ifdef OFDM_DEMOD_BATCH
    __aicore__ inline void RunCubeP1_Stable(
        const LocalTensor<half> &w1L0, const GlobalTensor<half> &x1Gm,
        const LocalTensor<half> &w2L0, const GlobalTensor<half> &x2Gm,
        const LocalTensor<float> &dst);

    __aicore__ inline void RunCubeP3_Stable(
        const GlobalTensor<half> &a1Gm, const LocalTensor<half> &b1L1,
        const GlobalTensor<half> &a2Gm, const LocalTensor<half> &b2L1,
        const LocalTensor<float> &dst);
#endif

    __aicore__ inline void RunCubeP3_Batch(
        const GlobalTensor<half> &Agm_1, const LocalTensor<half> &Bl0_1,
        const GlobalTensor<half> &Agm_2, const LocalTensor<half> &Bl0_2,
        const LocalTensor<float> &dst, uint16_t m);

    TPipe   *pipe_;
    uint32_t blockId_;
#ifdef OFDM_DEMOD_BATCH
    uint32_t batchSize_;
    uint32_t batchIdx_;
#endif
    uint32_t sym_start_;

    GlobalTensor<int16_t> inG_;
    GlobalTensor<half>    w32ReG_,  w32ImG_;
    GlobalTensor<half>    w64ReG_T_, w64ImG_T_;
    GlobalTensor<half>    twReG_,   twImG_;
    GlobalTensor<half>    outReG_,  outImG_;
    GlobalTensor<half>    XreScr_,  XimScr_;
    GlobalTensor<half>    TwReScr_, TwImScr_;

    TBuf<TPosition::VECCALC> bufXi16_,  bufXfullHalf_;
    TBuf<TPosition::VECCALC> bufXre_,   bufXim_;
    TBuf<TPosition::VECCALC> bufTmp_,   bufTmp2_;
    TBuf<TPosition::VECCALC> bufTwRe_,  bufTwIm_;
    TBuf<TPosition::VECCALC> bufX1re_,  bufX1im_;
    TBuf<TPosition::VECCALC> bufCubeAcc_, bufCubeNd_;

    TQue<TPosition::A1, 2>  qA1_;
    TQue<TPosition::A2, 2>  qA2_;
    TQue<TPosition::B1, 2>  qB1_;
    TQue<TPosition::B2, 2>  qB2_;
#ifdef OFDM_DEMOD_BATCH


    TQue<TPosition::CO1, 8> qCO1_;
#else
    TQue<TPosition::CO1, 2> qCO1_;
#endif


    TBuf<TPosition::A1> bufW32reL1_, bufW32imL1_, bufW32imNegL1_;
    TBuf<TPosition::B1> bufW64reL1_, bufW64imL1_, bufW64imNegL1_;
    TBuf<TPosition::A2> bufW32L0A_;
    TBuf<TPosition::B2> bufW64L0B_;
};





__aicore__ inline void OfdmDemod::Init(
    GM_ADDR input_gm,
    GM_ADDR w32_re_gm, GM_ADDR w32_im_gm,
    GM_ADDR w64_re_gm, GM_ADDR w64_im_gm,
    GM_ADDR tw_re_gm,  GM_ADDR tw_im_gm,
    GM_ADDR output_re_gm, GM_ADDR output_im_gm,
    GM_ADDR scratch_gm,
#ifdef OFDM_DEMOD_BATCH
    uint32_t batch_size,
#endif
    TPipe *pipe)
{
    pipe_      = pipe;
    blockId_   = GetBlockIdx();
#ifdef OFDM_DEMOD_BATCH
    batchSize_ = batch_size;
    batchIdx_  = 0;
    sym_start_ = 0;
#else
    sym_start_ = blockId_ * SYMBOLS_PER_CORE;
#endif

#ifdef OFDM_DEMOD_BATCH
    inG_     .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(input_gm), batchSize_ * INPUT_INT16_PER_BATCH);
#else
    inG_     .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(input_gm),     INPUT_GM_INT16_LEN);
#endif
    w32ReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w32_re_gm),       P * P);
    w32ImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w32_im_gm),       P * P);
    w64ReG_T_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w64_re_gm),       Q * Q);
    w64ImG_T_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w64_im_gm),       Q * Q);
    twReG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(tw_re_gm),        TILE_PQ);
    twImG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(tw_im_gm),        TILE_PQ);
#ifdef OFDM_DEMOD_BATCH
    outReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output_re_gm), batchSize_ * OUTPUT_ELEMS_PER_BATCH);
    outImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output_im_gm), batchSize_ * OUTPUT_ELEMS_PER_BATCH);
#else
    outReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output_re_gm), N_SYMBOL * N_FFT);
    outImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output_im_gm), N_SYMBOL * N_FFT);
#endif

#ifdef OFDM_DEMOD_BATCH


    constexpr uint32_t SCR_PER_CORE = 4 * BATCH_X_ELEMS;
    auto sbase = reinterpret_cast<__gm__ half *>(scratch_gm) + blockId_ * SCR_PER_CORE;
    XreScr_ .SetGlobalBuffer(sbase + 0 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    XimScr_ .SetGlobalBuffer(sbase + 1 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    TwReScr_.SetGlobalBuffer(sbase + 2 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    TwImScr_.SetGlobalBuffer(sbase + 3 * BATCH_X_ELEMS, BATCH_X_ELEMS);
#else

    constexpr uint32_t SCR_PER_CORE = 2 * BATCH_X_ELEMS;
    auto sbase = reinterpret_cast<__gm__ half *>(scratch_gm) + blockId_ * SCR_PER_CORE;
    XreScr_ .SetGlobalBuffer(sbase + 0 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    XimScr_ .SetGlobalBuffer(sbase + 1 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    TwReScr_.SetGlobalBuffer(sbase + 0 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    TwImScr_.SetGlobalBuffer(sbase + 1 * BATCH_X_ELEMS, BATCH_X_ELEMS);
#endif


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
    pipe_->InitBuffer(bufCubeAcc_,   BATCH_X_ELEMS * sizeof(float));
    pipe_->InitBuffer(bufCubeNd_,    BATCH_X_ELEMS * sizeof(float));


    pipe_->InitBuffer(qA1_,  2, MAX_M_BATCH * Q * sizeof(half));
    pipe_->InitBuffer(qA2_,  2, MAX_M_BATCH * Q * sizeof(half));
    pipe_->InitBuffer(qB1_,  2, Q * Q * sizeof(half));
    pipe_->InitBuffer(qB2_,  2, Q * Q * sizeof(half));
#ifdef OFDM_DEMOD_BATCH
    pipe_->InitBuffer(qCO1_, 8, M_MMAD * Q * sizeof(float));
#else
    pipe_->InitBuffer(qCO1_, 2, M_MMAD * Q * sizeof(float));
#endif


    pipe_->InitBuffer(bufW32reL1_,    P * P * sizeof(half));
    pipe_->InitBuffer(bufW32imL1_,    P * P * sizeof(half));
    pipe_->InitBuffer(bufW32imNegL1_, P * P * sizeof(half));
    pipe_->InitBuffer(bufW64reL1_,    Q * Q * sizeof(half));
    pipe_->InitBuffer(bufW64imL1_,    Q * Q * sizeof(half));
    pipe_->InitBuffer(bufW64imNegL1_, Q * Q * sizeof(half));
    pipe_->InitBuffer(bufW32L0A_, 3 * P * P * sizeof(half));
    pipe_->InitBuffer(bufW64L0B_, 3 * Q * Q * sizeof(half));
}






__aicore__ inline void OfdmDemod::CopyNd2Nz(
    const LocalTensor<half> &dst, const GlobalTensor<half> &src,
    uint16_t height, uint16_t width)
{
    for (uint16_t i = 0; i < width / 16; ++i) {
        DataCopy(dst[i * 16 * height], src[i * 16],
                 { height, 1, uint16_t(width / 16 - 1), 0 });
    }
}

__aicore__ inline void OfdmDemod::LoadL1ToL0A(
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

__aicore__ inline void OfdmDemod::NzToNdAndCast(
    const LocalTensor<half> &dst,
    const LocalTensor<float> &src,
    const LocalTensor<float> &tmp,
    uint16_t m)
{
    event_t eMv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_V));
    SetFlag<HardEvent::M_V>(eMv); WaitFlag<HardEvent::M_V>(eMv);
    DataCopyParams dcp;
    dcp.blockCount = m;
    dcp.blockLen   = 2;
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





__attribute__((noinline)) __aicore__ void OfdmDemod::LoadSymbol(uint32_t s, uint32_t sym)
{
    auto xi16      = bufXi16_.Get<int16_t>();
    auto xfullHalf = bufXfullHalf_.Get<half>();
    auto xre       = bufXre_.Get<half>();
    auto xim       = bufXim_.Get<half>();

    uint32_t base = SYM0_CP_OFFSET + sym * SYM_STRIDE;
#ifdef OFDM_DEMOD_BATCH
    base += batchIdx_ * N_SAMPLE_PER_SLOT;
#endif
    DataCopy(xi16, inG_[base * 2], 2 * N_FFT);
    event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e1); WaitFlag<HardEvent::MTE2_V>(e1);

    Cast(xfullHalf, xi16, RoundMode::CAST_NONE, 2 * N_FFT);
    PipeBarrier<PIPE_V>();
    Muls(xfullHalf, xfullHalf, (half)INPUT_SCALE, 2 * N_FFT);
    PipeBarrier<PIPE_V>();

    uint64_t rsvdCnt = 0;
    GatherMaskParams gmp;
    gmp.src0BlockStride = 1;
    gmp.repeatTimes     = 32;
    gmp.src0RepeatStride = 8;
    gmp.src1RepeatStride = 0;
    GatherMask(xre, xfullHalf, (uint8_t)1, false, 0, gmp, rsvdCnt);
    GatherMask(xim, xfullHalf, (uint8_t)2, false, 0, gmp, rsvdCnt);
    PipeBarrier<PIPE_V>();

    event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e2); WaitFlag<HardEvent::V_MTE3>(e2);
    DataCopy(XreScr_[s * TILE_PQ], xre, TILE_PQ);
    DataCopy(XimScr_[s * TILE_PQ], xim, TILE_PQ);
    event_t e3v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e3v); WaitFlag<HardEvent::MTE3_V>(e3v);
    event_t e3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e3); WaitFlag<HardEvent::MTE3_MTE2>(e3);
}

__aicore__ inline void OfdmDemod::Phase0_CopyIn()
{
    for (uint32_t s = 0; s < SYMBOLS_PER_CORE; ++s) {
        uint32_t sym = sym_start_ + s;
        if (sym >= N_SYMBOL) continue;
        LoadSymbol(s, sym);
    }
}








__aicore__ inline void OfdmDemod::RunCubeP1_ComplexTileReuse(
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


        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        qB2_.FreeTensor(b2);

        b1 = qB1_.AllocTensor<half>();
        DataCopy(b1, xImGm[bOff], {K_PHASE1, 1, uint16_t(Q / 16 - 1), 0});
        qB1_.EnQue(b1); b1 = qB1_.DeQue<half>();
        b2 = qB2_.AllocTensor<half>();
        LoadData(b2, b1, loadP);
        qB2_.EnQue(b2); qB1_.FreeTensor(b1); b2 = qB2_.DeQue<half>();
        mp.cmatrixInitVal = false;
        Mmad(co, wImNegL0, b2, mp);
        qCO1_.EnQue(co); qB2_.FreeTensor(b2); co = qCO1_.DeQue<float>();
        DataCopy(dstRe[dstOff], co, dcp, ep);
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
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        qB2_.FreeTensor(b2);

        b1 = qB1_.AllocTensor<half>();
        DataCopy(b1, xReGm[bOff], {K_PHASE1, 1, uint16_t(Q / 16 - 1), 0});
        qB1_.EnQue(b1); b1 = qB1_.DeQue<half>();
        b2 = qB2_.AllocTensor<half>();
        LoadData(b2, b1, loadP);
        qB2_.EnQue(b2); qB1_.FreeTensor(b1); b2 = qB2_.DeQue<half>();
        mp.cmatrixInitVal = false;
        Mmad(co, wImL0, b2, mp);
        qCO1_.EnQue(co); qB2_.FreeTensor(b2); co = qCO1_.DeQue<float>();
        DataCopy(dstIm[dstOff], co, dcp, ep);
        qCO1_.FreeTensor(co);
    }
}

#ifdef OFDM_DEMOD_BATCH
__aicore__ inline void OfdmDemod::RunCubeP1_Stable(
    const LocalTensor<half> &w1L0, const GlobalTensor<half> &x1Gm,
    const LocalTensor<half> &w2L0, const GlobalTensor<half> &x2Gm,
    const LocalTensor<float> &dst)
{
    constexpr uint16_t kBlocks = K_PHASE1 / 16;
    for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
        const uint32_t bOff = ns * N_SUB;
        const uint32_t dstOff = ns * M_MMAD * N_SUB;
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks;
        loadP.srcStride = 1;
        loadP.ifTranspose = true;

        LocalTensor<half> b1 = qB1_.AllocTensor<half>();
        DataCopy(b1, x1Gm[bOff], {K_PHASE1, 1, uint16_t(Q / 16 - 1), 0});
        qB1_.EnQue(b1);
        b1 = qB1_.DeQue<half>();
        LocalTensor<half> b2 = qB2_.AllocTensor<half>();
        LoadData(b2, b1, loadP);
        qB2_.EnQue(b2); qB1_.FreeTensor(b1);
        b2 = qB2_.DeQue<half>();
        LocalTensor<float> co = qCO1_.AllocTensor<float>();
        MmadParams mp;
        mp.m = M_MMAD; mp.n = N_SUB; mp.k = K_PHASE1; mp.cmatrixInitVal = true;
        Mmad(co, w1L0, b2, mp);
        qB2_.FreeTensor(b2);

        b1 = qB1_.AllocTensor<half>();
        DataCopy(b1, x2Gm[bOff], {K_PHASE1, 1, uint16_t(Q / 16 - 1), 0});
        qB1_.EnQue(b1);
        b1 = qB1_.DeQue<half>();
        b2 = qB2_.AllocTensor<half>();
        LoadData(b2, b1, loadP);
        qB2_.EnQue(b2); qB1_.FreeTensor(b1);
        b2 = qB2_.DeQue<half>();
        mp.cmatrixInitVal = false;
        Mmad(co, w2L0, b2, mp);
        qCO1_.EnQue(co); qB2_.FreeTensor(b2);

        co = qCO1_.DeQue<float>();
        DataCopyParams dcp;
        dcp.blockCount = 1;
        dcp.blockLen = uint16_t(M_MMAD * N_SUB / 256);
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(dst[dstOff], co, dcp, ep);
        PipeBarrier<PIPE_ALL>();
        qCO1_.FreeTensor(co);
    }
}
#endif

__attribute__((noinline)) __aicore__ void OfdmDemod::Phase1_Dft32()
{
    auto x1reBatch = bufX1re_.Get<half>();
    auto x1imBatch = bufX1im_.Get<half>();
    auto cubeAccRe = bufCubeAcc_.Get<float>();
    auto cubeAccIm = cubeAccRe[TILE_PQ];
    auto cubeNd    = bufCubeNd_.Get<float>();

    auto w32ReL1    = bufW32reL1_.Get<half>();
    auto w32ImL1    = bufW32imL1_.Get<half>();
    auto w32ImNegL1 = bufW32imNegL1_.Get<half>();
    auto w32L0 = bufW32L0A_.Get<half>();
    auto w32ReL0 = w32L0;
    auto w32ImL0 = w32L0[P * P];
    auto w32ImNegL0 = w32L0[2 * P * P];



    CopyNd2Nz(w32ReL1, w32ReG_, M_MMAD, K_PHASE1);
    CopyNd2Nz(w32ImL1, w32ImG_, M_MMAD, K_PHASE1);
    PipeBarrier<PIPE_ALL>();




















    {
        auto tmpUB = bufTmp_.Get<half>();

        DataCopy(tmpUB, w32ImL1, P * P);
        PipeBarrier<PIPE_ALL>();

        Muls(tmpUB, tmpUB, (half)-1.0, P * P);
        PipeBarrier<PIPE_V>();

        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e); WaitFlag<HardEvent::V_MTE3>(e);
        DataCopy(w32ImNegL1, tmpUB, P * P);
        PipeBarrier<PIPE_ALL>();
    }


    LoadL1ToL0A(w32ReL0,    w32ReL1,    M_MMAD / 16, K_PHASE1 / 16);
    LoadL1ToL0A(w32ImL0,    w32ImL1,    M_MMAD / 16, K_PHASE1 / 16);
    LoadL1ToL0A(w32ImNegL0, w32ImNegL1, M_MMAD / 16, K_PHASE1 / 16);
    PipeBarrier<PIPE_ALL>();

    for (uint32_t s = 0; s < SYMBOLS_PER_CORE; ++s) {
        uint32_t sym = sym_start_ + s;
        if (sym >= N_SYMBOL) continue;
        uint32_t symOff = s * TILE_PQ;
        auto xreSym = XreScr_[symOff];
        auto ximSym = XimScr_[symOff];

#ifdef OFDM_DEMOD_BATCH
        RunCubeP1_Stable(w32ReL0, xreSym, w32ImNegL0, ximSym, cubeAccRe);
        PipeBarrier<PIPE_ALL>();
        NzToNdAndCast(x1reBatch[symOff], cubeAccRe, cubeNd);
        PipeBarrier<PIPE_ALL>();
        RunCubeP1_Stable(w32ReL0, ximSym, w32ImL0, xreSym, cubeAccIm);
        PipeBarrier<PIPE_ALL>();
        NzToNdAndCast(x1imBatch[symOff], cubeAccIm, cubeNd);
#else
        RunCubeP1_ComplexTileReuse(w32ReL0, w32ImL0, w32ImNegL0,
                                   xreSym, ximSym, cubeAccRe, cubeAccIm);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(x1reBatch[symOff], cubeAccRe, cubeNd);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(x1imBatch[symOff], cubeAccIm, cubeNd);
#endif
        PipeBarrier<PIPE_ALL>();
    }

    PipeBarrier<PIPE_ALL>();
}





__attribute__((noinline)) __aicore__ void OfdmDemod::Phase2_Twiddle()
{
    auto tmp       = bufTmp_.Get<half>();
    auto tmp2      = bufTmp2_.Get<half>();
    auto twre      = bufTwRe_.Get<half>();
    auto twim      = bufTwIm_.Get<half>();
    auto x1reBatch = bufX1re_.Get<half>();
    auto x1imBatch = bufX1im_.Get<half>();


#ifdef OFDM_DEMOD_BATCH
    if (batchIdx_ == 0) {
#endif
        DataCopy(twre, twReG_, TILE_PQ);
        DataCopy(twim, twImG_, TILE_PQ);
        event_t e0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(e0); WaitFlag<HardEvent::MTE2_V>(e0);
#ifdef OFDM_DEMOD_BATCH
    }
#endif

    for (uint32_t s = 0; s < SYMBOLS_PER_CORE; ++s) {
        uint32_t sym = sym_start_ + s;
        if (sym >= N_SYMBOL) continue;
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

    event_t eScratch = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eScratch); WaitFlag<HardEvent::MTE3_MTE2>(eScratch);
}








#ifdef OFDM_DEMOD_BATCH
__aicore__ inline void OfdmDemod::RunCubeP3_Stable(
    const GlobalTensor<half> &a1Gm, const LocalTensor<half> &b1L1,
    const GlobalTensor<half> &a2Gm, const LocalTensor<half> &b2L1,
    const LocalTensor<float> &dst)
{
    constexpr uint16_t mBlocks = M_MMAD / 16;
    constexpr uint16_t kBlocks = K_PHASE3 / 16;
    for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
        const uint32_t dstOff = ns * M_MMAD * N_SUB;
        LocalTensor<half> a1 = qA1_.AllocTensor<half>();
        for (uint16_t i = 0; i < kBlocks; ++i) {
            DataCopy(a1[i * 16 * M_MMAD], a1Gm[i * 16],
                     {M_MMAD, 1, uint16_t(kBlocks - 1), 0});
        }
        qA1_.EnQue(a1);
        a1 = qA1_.DeQue<half>();
        LocalTensor<half> a2 = qA2_.AllocTensor<half>();
        LoadL1ToL0A(a2, a1, mBlocks, kBlocks);
        qA2_.EnQue(a2); qA1_.FreeTensor(a1);

        LocalTensor<half> b2 = qB2_.AllocTensor<half>();
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks;
        loadP.srcStride = 1;
        loadP.ifTranspose = true;
        LoadData(b2, b1L1[ns * kBlocks * 256], loadP);
        qB2_.EnQue(b2);
        a2 = qA2_.DeQue<half>(); b2 = qB2_.DeQue<half>();
        LocalTensor<float> co = qCO1_.AllocTensor<float>();
        MmadParams mp;
        mp.m = M_MMAD; mp.n = N_SUB; mp.k = K_PHASE3; mp.cmatrixInitVal = true;
        Mmad(co, a2, b2, mp);
        qA2_.FreeTensor(a2); qB2_.FreeTensor(b2);

        a1 = qA1_.AllocTensor<half>();
        for (uint16_t i = 0; i < kBlocks; ++i) {
            DataCopy(a1[i * 16 * M_MMAD], a2Gm[i * 16],
                     {M_MMAD, 1, uint16_t(kBlocks - 1), 0});
        }
        qA1_.EnQue(a1);
        a1 = qA1_.DeQue<half>();
        a2 = qA2_.AllocTensor<half>();
        LoadL1ToL0A(a2, a1, mBlocks, kBlocks);
        qA2_.EnQue(a2); qA1_.FreeTensor(a1);
        b2 = qB2_.AllocTensor<half>();
        LoadData(b2, b2L1[ns * kBlocks * 256], loadP);
        qB2_.EnQue(b2);
        a2 = qA2_.DeQue<half>(); b2 = qB2_.DeQue<half>();
        mp.cmatrixInitVal = false;
        Mmad(co, a2, b2, mp);
        qCO1_.EnQue(co);
        qA2_.FreeTensor(a2); qB2_.FreeTensor(b2);

        co = qCO1_.DeQue<float>();
        DataCopyParams dcp;
        dcp.blockCount = 1;
        dcp.blockLen = uint16_t(M_MMAD * N_SUB / 256);
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(dst[dstOff], co, dcp, ep);
        PipeBarrier<PIPE_ALL>();
        qCO1_.FreeTensor(co);
    }
}
#endif

__aicore__ inline void OfdmDemod::RunCubeP3_Batch(
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

    for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
        uint32_t dstOff = ns * m * N_SUB;
        uint32_t wOff = ns * K_PHASE3 * N_SUB;
        LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
        MmadParams mp;
        mp.m = m; mp.n = N_SUB; mp.k = K_PHASE3; mp.cmatrixInitVal = true;
        Mmad(co1, a2_1, Bl0_1[wOff], mp);
        mp.cmatrixInitVal = false;
        Mmad(co1, a2_2, Bl0_2[wOff], mp);
        qCO1_.EnQue(co1);

        co1 = qCO1_.DeQue<float>();
        DataCopyParams dcp; dcp.blockCount = 1;
        dcp.blockLen = uint16_t(m * N_SUB / 256);
        DataCopyEnhancedParams ep; ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(dst[dstOff], co1, dcp, ep);
        qCO1_.FreeTensor(co1);
    }
    qA2_.FreeTensor(a2_1); qA2_.FreeTensor(a2_2);
}

__attribute__((noinline)) __aicore__ void OfdmDemod::Phase3_Dft64()
{
    auto cubeAcc = bufCubeAcc_.Get<float>();
    auto cubeNd  = bufCubeNd_.Get<float>();

    auto w64ReL1    = bufW64reL1_.Get<half>();
    auto w64ImL1    = bufW64imL1_.Get<half>();
    auto w64ImNegL1 = bufW64imNegL1_.Get<half>();
    auto w64L0 = bufW64L0B_.Get<half>();
    auto w64ReL0 = w64L0;
    auto w64ImL0 = w64L0[Q * Q];
    auto w64ImNegL0 = w64L0[2 * Q * Q];


        CopyNd2Nz(w64ReL1, w64ReG_T_, K_PHASE3, Q);
        CopyNd2Nz(w64ImL1, w64ImG_T_, K_PHASE3, Q);
        PipeBarrier<PIPE_ALL>();


    {

        auto bigTmp = bufX1re_.Get<half>();
        DataCopy(bigTmp, w64ImL1, Q * Q);
        PipeBarrier<PIPE_ALL>();
        Muls(bigTmp, bigTmp, (half)-1.0, Q * Q);
        PipeBarrier<PIPE_V>();
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e); WaitFlag<HardEvent::V_MTE3>(e);
        DataCopy(w64ImNegL1, bigTmp, Q * Q);
        PipeBarrier<PIPE_ALL>();
    }


    LoadData2DParams loadP;
    loadP.repeatTimes = K_PHASE3 / 16;
    loadP.srcStride = 1;
    loadP.ifTranspose = true;
    for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
        uint32_t off = ns * K_PHASE3 * N_SUB;
        LoadData(w64ReL0[off],    w64ReL1[off],    loadP);
        LoadData(w64ImL0[off],    w64ImL1[off],    loadP);
        LoadData(w64ImNegL0[off], w64ImNegL1[off], loadP);
    }
    PipeBarrier<PIPE_ALL>();

    uint16_t validSymbols = uint16_t(N_SYMBOL - sym_start_);
    if (validSymbols > SYMBOLS_PER_CORE) validSymbols = SYMBOLS_PER_CORE;
#ifdef OFDM_DEMOD_BATCH



    auto yreBatch = bufX1re_.Get<half>();
    auto yimBatch = bufX1im_.Get<half>();
    for (uint32_t s = 0; s < validSymbols; ++s) {
        const uint32_t sym = sym_start_ + s;
        const uint32_t symOff = s * TILE_PQ;
        const uint32_t outOff = batchIdx_ * OUTPUT_ELEMS_PER_BATCH + sym * N_FFT;

        RunCubeP3_Stable(TwReScr_[symOff], w64ReL1,
                         TwImScr_[symOff], w64ImNegL1, cubeAcc);
        PipeBarrier<PIPE_ALL>();
        NzToNdAndCast(yreBatch, cubeAcc, cubeNd, M_MMAD);
        PipeBarrier<PIPE_ALL>();
        event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e1); WaitFlag<HardEvent::V_MTE3>(e1);
        DataCopy(outReG_[outOff], yreBatch, TILE_PQ);
        event_t e1Done = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e1Done); WaitFlag<HardEvent::MTE3_V>(e1Done);

        RunCubeP3_Stable(TwReScr_[symOff], w64ImL1,
                         TwImScr_[symOff], w64ReL1, cubeAcc);
        PipeBarrier<PIPE_ALL>();
        NzToNdAndCast(yimBatch, cubeAcc, cubeNd, M_MMAD);
        PipeBarrier<PIPE_ALL>();
        event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e2); WaitFlag<HardEvent::V_MTE3>(e2);
        DataCopy(outImG_[outOff], yimBatch, TILE_PQ);
        event_t e2Done = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e2Done); WaitFlag<HardEvent::MTE3_V>(e2Done);
    }
#else
    uint16_t m = validSymbols * P;
    uint32_t outOff = sym_start_ * N_FFT;


    auto yreBatch = bufX1re_.Get<half>();
    RunCubeP3_Batch(TwReScr_, w64ReL0, TwImScr_, w64ImNegL0, cubeAcc, m);
    PipeBarrier<PIPE_V>();
    NzToNdAndCast(yreBatch, cubeAcc, cubeNd, m);
    event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e1); WaitFlag<HardEvent::V_MTE3>(e1);
    DataCopy(outReG_[outOff], yreBatch, m * Q);

    auto yimBatch = bufX1im_.Get<half>();
    RunCubeP3_Batch(TwReScr_, w64ImL0, TwImScr_, w64ReL0, cubeAcc, m);
    PipeBarrier<PIPE_V>();
    NzToNdAndCast(yimBatch, cubeAcc, cubeNd, m);
    event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e2); WaitFlag<HardEvent::V_MTE3>(e2);
    DataCopy(outImG_[outOff], yimBatch, m * Q);
#endif

    PipeBarrier<PIPE_ALL>();
}





__aicore__ inline void OfdmDemod::Process()
{
#ifdef OFDM_DEMOD_BATCH
    if (batchSize_ == 0) return;
    const uint32_t totalTiles = batchSize_ * TILES_PER_BATCH;
    for (uint32_t tile = blockId_; tile < totalTiles; tile += BLOCK_DIM) {
        batchIdx_  = tile / TILES_PER_BATCH;
        sym_start_ = (tile % TILES_PER_BATCH) * SYMBOLS_PER_CORE;
        Phase0_CopyIn();  AscendC::SyncAll();
        Phase1_Dft32();   AscendC::SyncAll();
        Phase2_Twiddle(); AscendC::SyncAll();
        Phase3_Dft64();   AscendC::SyncAll();
    }
#else
    Phase0_CopyIn();
    Phase1_Dft32();
    Phase2_Twiddle();
    Phase3_Dft64();
#endif
}





extern "C" __global__ __aicore__ void ofdm_demod_kernel(
    GM_ADDR input_gm,
    GM_ADDR w32_re_gm,    GM_ADDR w32_im_gm,
    GM_ADDR w64_re_gm,    GM_ADDR w64_im_gm,
    GM_ADDR tw_re_gm,     GM_ADDR tw_im_gm,
    GM_ADDR scratch_gm,
    GM_ADDR output_re_gm, GM_ADDR output_im_gm,
#ifdef OFDM_DEMOD_BATCH
    uint32_t batch_size,
#endif
    GM_ADDR ws,
    GM_ADDR tilingGm)
{
    TPipe pipe;
    OfdmDemod op;
    op.Init(input_gm,
            w32_re_gm, w32_im_gm,
            w64_re_gm, w64_im_gm,
            tw_re_gm,  tw_im_gm,
            output_re_gm, output_im_gm,
            scratch_gm,
#ifdef OFDM_DEMOD_BATCH
            batch_size,
#endif
            &pipe);
    op.Process();
}
