





























#include "kernel_operator.h"
#include "ofdm_mod.h"

using namespace AscendC;
using namespace ofdm_mod_batch;


class OfdmModBatch {
public:
    __aicore__ inline OfdmModBatch() {}

    __aicore__ inline void Init(GM_ADDR in_re_gm, GM_ADDR in_im_gm,
                                 GM_ADDR w32_re_gm, GM_ADDR w32_im_gm,
                                 GM_ADDR w64_re_gm, GM_ADDR w64_im_gm,
                                 GM_ADDR tw_re_gm,  GM_ADDR tw_im_gm,
                                 GM_ADDR output_re_gm, GM_ADDR output_im_gm, GM_ADDR output_iq_gm,
                                 uint32_t batch_size,
                                 TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void PrepareConstants();
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

    __aicore__ inline void RunCubeP1_Fused(
        const LocalTensor<half> &Al0_1, const LocalTensor<half> &Bub_1,
        const LocalTensor<half> &Al0_2, const LocalTensor<half> &Bub_2,
        const LocalTensor<float> &dst);

    __aicore__ inline void RunCubeP3_Batch(
        const GlobalTensor<half> &Agm_1, const LocalTensor<half> &Bl0_1,
        const GlobalTensor<half> &Agm_2, const LocalTensor<half> &Bl0_2,
        const LocalTensor<float> &dst, uint16_t m);

    TPipe   *pipe_;
    uint32_t blockId_;
    uint32_t batchSize_;
    uint32_t batchIdx_;
    uint32_t sym_start_;

    GlobalTensor<half>    inReG_,   inImG_;
    GlobalTensor<half>    w32ReG_,  w32ImG_;
    GlobalTensor<half>    w64ReG_T_, w64ImG_T_;
    GlobalTensor<half>    twReG_,   twImG_;
    GlobalTensor<int16_t> outReG_,  outImG_,  outIqG_;

    TBuf<TPosition::VECCALC> bufTmp_,   bufTmp2_;
    TBuf<TPosition::VECCALC> bufTwRe_,  bufTwIm_;
    TBuf<TPosition::VECCALC> bufX1tre_, bufX1tim_;
    TBuf<TPosition::VECCALC> bufCubeAcc_, bufCubeNd_;
    TBuf<TPosition::VECCALC> bufI16_;
    TBuf<TPosition::VECCALC> bufReI32_, bufImI32_;

    TQue<TPosition::A1, 2>  qA1_;
    TQue<TPosition::A2, 2>  qA2_;
    TQue<TPosition::B2, 2>  qB2_;
    TQue<TPosition::CO1, 4> qCO1_;

    TBuf<TPosition::A1> bufW32reL1_, bufW32imL1_, bufW32imNegL1_;
    TBuf<TPosition::B1> bufW64reL1_, bufW64imL1_, bufW64imNegL1_;
    TBuf<TPosition::B1> bufX1ReL1_, bufX1ImL1_;

    TBuf<TPosition::A2> bufW32L0A_;
    TBuf<TPosition::B2> bufW64L0B_;
};


__aicore__ inline void OfdmModBatch::Init(
    GM_ADDR in_re_gm, GM_ADDR in_im_gm,
    GM_ADDR w32_re_gm, GM_ADDR w32_im_gm,
    GM_ADDR w64_re_gm, GM_ADDR w64_im_gm,
    GM_ADDR tw_re_gm,  GM_ADDR tw_im_gm,
    GM_ADDR output_re_gm, GM_ADDR output_im_gm, GM_ADDR output_iq_gm,
    uint32_t batch_size,
    TPipe *pipe)
{
    pipe_      = pipe;
    blockId_   = GetBlockIdx();
    batchSize_ = batch_size;
    batchIdx_  = 0;
    sym_start_ = 0;

    inReG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(in_re_gm), batchSize_ * INPUT_ELEMS_PER_BATCH);
    inImG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(in_im_gm), batchSize_ * INPUT_ELEMS_PER_BATCH);
    w32ReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w32_re_gm),    P * P);
    w32ImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w32_im_gm),    P * P);
    w64ReG_T_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w64_re_gm),    Q * Q);
    w64ImG_T_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w64_im_gm),    Q * Q);
    twReG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(tw_re_gm),     TILE_PQ);
    twImG_   .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(tw_im_gm),     TILE_PQ);
    outReG_  .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(output_re_gm), batchSize_ * OUTPUT_ELEMS_PER_BATCH);
    outImG_  .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(output_im_gm), batchSize_ * OUTPUT_ELEMS_PER_BATCH);
    outIqG_  .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(output_iq_gm), 2 * batchSize_ * OUTPUT_ELEMS_PER_BATCH);

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
    pipe_->InitBuffer(qB2_,  2, Q * Q * sizeof(half));
    pipe_->InitBuffer(qCO1_, 4, M_MMAD * Q * sizeof(float));

    pipe_->InitBuffer(bufW32reL1_,    P * P * sizeof(half));
    pipe_->InitBuffer(bufW32imL1_,    P * P * sizeof(half));
    pipe_->InitBuffer(bufW32imNegL1_, P * P * sizeof(half));
    pipe_->InitBuffer(bufW64reL1_,    Q * Q * sizeof(half));
    pipe_->InitBuffer(bufW64imL1_,    Q * Q * sizeof(half));
    pipe_->InitBuffer(bufW64imNegL1_, Q * Q * sizeof(half));
    pipe_->InitBuffer(bufX1ReL1_, BATCH_X_ELEMS * sizeof(half));
    pipe_->InitBuffer(bufX1ImL1_, BATCH_X_ELEMS * sizeof(half));
    pipe_->InitBuffer(bufW32L0A_, 3 * P * P * sizeof(half));
    pipe_->InitBuffer(bufW64L0B_, 3 * Q * Q * sizeof(half));

}



__aicore__ inline void OfdmModBatch::CopyNd2Nz(
    const LocalTensor<half> &dst, const GlobalTensor<half> &src,
    uint16_t height, uint16_t width)
{
    for (uint16_t i = 0; i < width / 16; ++i) {
        DataCopy(dst[i * 16 * height], src[i * 16],
                 { height, 1, uint16_t(width / 16 - 1), 0 });
    }
}

__aicore__ inline void OfdmModBatch::LoadL1ToL0A(
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

__aicore__ inline void OfdmModBatch::NzToNdAndCast(
    const LocalTensor<half> &dst,
    const LocalTensor<float> &src,
    const LocalTensor<float> &tmp,
    uint16_t m)
{
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
}




__aicore__ inline void OfdmModBatch::RunCubeP3_Batch(
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

__aicore__ inline void OfdmModBatch::PrepareConstants()
{
    auto twre = bufTwRe_.Get<half>();
    auto twim = bufTwIm_.Get<half>();
    DataCopy(twre, twReG_, TILE_PQ);
    DataCopy(twim, twImG_, TILE_PQ);
    PipeBarrier<PIPE_ALL>();
}

__attribute__((noinline)) __aicore__ void OfdmModBatch::PhaseA_Idft64()
{
    auto x1tre   = bufX1tre_.Get<half>();
    auto x1tim   = bufX1tim_.Get<half>();
    auto cubeAcc = bufCubeAcc_.Get<float>();
    auto cubeNd  = bufCubeNd_.Get<float>();

    auto w64ReL1    = bufW64reL1_.Get<half>();
    auto w64ImL1    = bufW64imL1_.Get<half>();
    auto w64ImNegL1 = bufW64imNegL1_.Get<half>();
    auto w64L0 = bufW64L0B_.Get<half>();
    auto w64ReL0 = w64L0;
    auto w64ImL0 = w64L0[Q * Q];
    auto w64ImNegL0 = w64L0[2 * Q * Q];
    if (batchIdx_ == 0) {
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

        LoadData2DParams loadP;
        loadP.repeatTimes = K_PHASE3 / 16; loadP.srcStride = 1; loadP.ifTranspose = true;
        for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
            uint32_t off = ns * K_PHASE3 * N_SUB;
            LoadData(w64ReL0[off],    w64ReL1[off],    loadP);
            LoadData(w64ImL0[off],    w64ImL1[off],    loadP);
            LoadData(w64ImNegL0[off], w64ImNegL1[off], loadP);
        }
        PipeBarrier<PIPE_ALL>();
    }

    uint16_t validSymbols = uint16_t(N_SYMBOL - sym_start_);
    if (validSymbols > SYMBOLS_PER_CORE) validSymbols = SYMBOLS_PER_CORE;
    uint16_t m = validSymbols * P;
    const uint32_t inBase = batchIdx_ * INPUT_ELEMS_PER_BATCH;
    auto s4reBatch = inReG_[inBase + sym_start_ * N_FFT];
    auto s4imBatch = inImG_[inBase + sym_start_ * N_FFT];


    RunCubeP3_Batch(s4reBatch, w64ReL0, s4imBatch, w64ImNegL0, cubeAcc, m);
    PipeBarrier<PIPE_V>();
    NzToNdAndCast(x1tre, cubeAcc, cubeNd, m);

    RunCubeP3_Batch(s4reBatch, w64ImL0, s4imBatch, w64ReL0, cubeAcc, m);
    PipeBarrier<PIPE_V>();
    NzToNdAndCast(x1tim, cubeAcc, cubeNd, m);

    PipeBarrier<PIPE_ALL>();
}



__attribute__((noinline)) __aicore__ void OfdmModBatch::PhaseB_Twiddle()
{
    auto outRe   = bufTmp_.Get<half>();
    auto outIm   = bufTmp2_.Get<half>();
    auto work    = bufCubeAcc_.Get<half>();
    auto twre    = bufTwRe_.Get<half>();
    auto twim    = bufTwIm_.Get<half>();
    auto x1tre   = bufX1tre_.Get<half>();
    auto x1tim   = bufX1tim_.Get<half>();
    auto x1reL1  = bufX1ReL1_.Get<half>();
    auto x1imL1  = bufX1ImL1_.Get<half>();

    for (uint32_t s = 0; s < SYMBOLS_PER_CORE; ++s) {
        uint32_t sym = sym_start_ + s;
        if (sym >= N_SYMBOL) continue;
        uint32_t symOff = s * TILE_PQ;
        auto xreS = x1tre[symOff];
        auto ximS = x1tim[symOff];


        Mul(outRe, xreS, twre, TILE_PQ); PipeBarrier<PIPE_V>();
        Mul(work,  ximS, twim, TILE_PQ); PipeBarrier<PIPE_V>();
        Sub(outRe, outRe, work, TILE_PQ); PipeBarrier<PIPE_V>();
        Mul(outIm, xreS, twim, TILE_PQ); PipeBarrier<PIPE_V>();
        Mul(work,  ximS, twre, TILE_PQ); PipeBarrier<PIPE_V>();
        Add(outIm, outIm, work, TILE_PQ); PipeBarrier<PIPE_V>();

        event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e1); WaitFlag<HardEvent::V_MTE3>(e1);


        for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
            uint32_t panelOff = symOff + ns * K_PHASE1 * N_SUB;
            DataCopy(x1reL1[panelOff], outRe[ns * N_SUB],
                     { K_PHASE1, 1, uint16_t(Q / 16 - 1), 0 });
            DataCopy(x1imL1[panelOff], outIm[ns * N_SUB],
                     { K_PHASE1, 1, uint16_t(Q / 16 - 1), 0 });
        }
        event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e2); WaitFlag<HardEvent::MTE3_V>(e2);
    }
}





__aicore__ inline void OfdmModBatch::RunCubeP1_Fused(
    const LocalTensor<half> &Al0_1, const LocalTensor<half> &B1_1,
    const LocalTensor<half> &Al0_2, const LocalTensor<half> &B1_2,
    const LocalTensor<float> &dst)
{
    constexpr uint16_t kBlocks = K_PHASE1 / 16;
    for (uint16_t ns = 0; ns < N_SPLITS; ns++) {
        uint32_t bOff = ns * K_PHASE1 * N_SUB;
        uint32_t dstOff = ns * M_MMAD * N_SUB;
        LocalTensor<half> b2_1 = qB2_.AllocTensor<half>();
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks; loadP.srcStride = 1; loadP.ifTranspose = true;
        LoadData(b2_1, B1_1[bOff], loadP);
        qB2_.EnQue(b2_1);

        b2_1 = qB2_.DeQue<half>();
        LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
        MmadParams mp; mp.m = M_MMAD; mp.n = N_SUB; mp.k = K_PHASE1; mp.cmatrixInitVal = true;
        Mmad(co1, Al0_1, b2_1, mp);
        qB2_.FreeTensor(b2_1);

        LocalTensor<half> b2_2 = qB2_.AllocTensor<half>();
        LoadData(b2_2, B1_2[bOff], loadP);
        qB2_.EnQue(b2_2);

        b2_2 = qB2_.DeQue<half>();
        mp.cmatrixInitVal = false;
        Mmad(co1, Al0_2, b2_2, mp);
        qCO1_.EnQue(co1);
        qB2_.FreeTensor(b2_2);

        co1 = qCO1_.DeQue<float>();
        DataCopyParams dcp; dcp.blockCount = 1;
        dcp.blockLen = uint16_t(M_MMAD * N_SUB / 256);
        DataCopyEnhancedParams ep; ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(dst[dstOff], co1, dcp, ep);
        qCO1_.FreeTensor(co1);
    }
}

__attribute__((noinline)) __aicore__ void OfdmModBatch::PhaseC_Idft32()
{
    auto cubeAcc = bufCubeAcc_.Get<float>();
    auto iqTmp   = cubeAcc[TILE_PQ];
    auto cubeNd  = bufCubeNd_.Get<float>();
    auto halfBuf = bufTmp_.Get<half>();
    auto i16Buf  = bufI16_.Get<int16_t>();
    auto reI32   = bufReI32_.Get<int32_t>();
    auto imI32   = bufImI32_.Get<int32_t>();
    auto x1reL1  = bufX1ReL1_.Get<half>();
    auto x1imL1  = bufX1ImL1_.Get<half>();

    auto w32ReL1    = bufW32reL1_.Get<half>();
    auto w32ImL1    = bufW32imL1_.Get<half>();
    auto w32ImNegL1 = bufW32imNegL1_.Get<half>();
    auto w32L0 = bufW32L0A_.Get<half>();
    auto w32ReL0 = w32L0;
    auto w32ImL0 = w32L0[P * P];
    auto w32ImNegL0 = w32L0[2 * P * P];
    if (batchIdx_ == 0) {
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

        LoadL1ToL0A(w32ReL0,    w32ReL1,    M_MMAD / 16, K_PHASE1 / 16);
        LoadL1ToL0A(w32ImL0,    w32ImL1,    M_MMAD / 16, K_PHASE1 / 16);
        LoadL1ToL0A(w32ImNegL0, w32ImNegL1, M_MMAD / 16, K_PHASE1 / 16);
        PipeBarrier<PIPE_ALL>();
    }

    for (uint32_t s = 0; s < SYMBOLS_PER_CORE; ++s) {
        uint32_t sym = sym_start_ + s;
        if (sym >= N_SYMBOL) continue;
        uint32_t symOff = s * TILE_PQ;
        auto x1reSym = x1reL1[symOff];
        auto x1imSym = x1imL1[symOff];

        uint32_t cp    = (sym == 0) ? CP_LEN_FIRST : CP_LEN_OTHER;
        uint32_t start = batchIdx_ * OUTPUT_ELEMS_PER_BATCH
                       + SYM0_CP_OFFSET + sym * SYM_STRIDE;


        RunCubeP1_Fused(w32ReL0, x1reSym, w32ImNegL0, x1imSym, cubeAcc);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(halfBuf, cubeAcc, cubeNd);
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


        RunCubeP1_Fused(w32ReL0, x1imSym, w32ImL0, x1reSym, cubeAcc);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(halfBuf, cubeAcc, cubeNd);
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
        Muls(iqTmp, cubeNd, (float)(1.0 / 65536.0), TILE_PQ);
        PipeBarrier<PIPE_V>();
        auto qI32 = cubeNd.ReinterpretCast<int32_t>();
        Cast(qI32, iqTmp, RoundMode::CAST_FLOOR, TILE_PQ);
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
        const uint32_t iqBatchBase = 2 * batchIdx_ * OUTPUT_ELEMS_PER_BATCH;
        const uint32_t iqStart = 2 * (start - batchIdx_ * OUTPUT_ELEMS_PER_BATCH);
        DataCopy(outIqG_[iqBatchBase + iqStart],        iqI16,                     2 * TILE_PQ);
        DataCopy(outIqG_[iqBatchBase + iqStart - 2*cp], iqI16[2 * (TILE_PQ - cp)], 2 * cp);
        event_t eq2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eq2); WaitFlag<HardEvent::MTE3_V>(eq2);
    }

    PipeBarrier<PIPE_ALL>();
}


__aicore__ inline void OfdmModBatch::Process()
{
    if (batchSize_ == 0) return;
    PrepareConstants();
    const uint32_t totalTiles = batchSize_ * TILES_PER_BATCH;
    for (uint32_t tile = blockId_; tile < totalTiles; tile += BLOCK_DIM) {
        batchIdx_  = tile / TILES_PER_BATCH;
        sym_start_ = (tile % TILES_PER_BATCH) * SYMBOLS_PER_CORE;
        PhaseA_Idft64();
        PhaseB_Twiddle();
        PhaseC_Idft32();
    }
}



extern "C" __global__ __aicore__ void ofdm_mod_batch_kernel(
    GM_ADDR in_re_gm,     GM_ADDR in_im_gm,
    GM_ADDR w32_re_gm,    GM_ADDR w32_im_gm,
    GM_ADDR w64_re_gm,    GM_ADDR w64_im_gm,
    GM_ADDR tw_re_gm,     GM_ADDR tw_im_gm,
    GM_ADDR output_re_gm, GM_ADDR output_im_gm, GM_ADDR output_iq_gm,
    uint32_t batch_size,
    GM_ADDR ws,
    GM_ADDR tilingGm)
{
    TPipe pipe;
    OfdmModBatch op;
    op.Init(in_re_gm, in_im_gm,
            w32_re_gm, w32_im_gm,
            w64_re_gm, w64_im_gm,
            tw_re_gm,  tw_im_gm,
            output_re_gm, output_im_gm, output_iq_gm,
            batch_size,
            &pipe);
    op.Process();
}
