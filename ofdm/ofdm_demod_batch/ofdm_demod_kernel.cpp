/**
 * @file ofdm_demod_kernel.cpp — 5G NR OFDM Demodulator (Ascend 310P3 / dav_m200)
 *
 * 算法: 2048-FFT 分解为 32×64 mixed-radix Cooley-Tukey
 *   Phase 0: CP removal + IQ de-interleave   (vector, GatherMask)
 *   Phase 1: DFT-32  X1 = (W32/sqrt(32)) @ X (cube, K=32)
 *   Phase 2: twiddle Tw = X1 ⊙ twiddle       (vector)
 *   Phase 3: DFT-64  Y  = Tw @ (W64/sqrt(64))^T (cube, K=64)
 *
 * 两级矩阵预折叠 1/sqrt(32) 与 1/sqrt(64)，输出直接满足 unitary FFT；
 * kernel 不增加额外归一化 Vector 指令。
 *
 * V2 优化: cmatrixInitVal=false fused Mmad
 *   - 复数实部 X1_re = W_re·X_re + (-W_im)·X_im  (W_im 预先取负)
 *   - 复数虚部 X1_im = W_re·X_im + W_im·X_re     (本来就是 Add)
 *   - 两次 Mmad 累加到同一 L0C, 省外面的 Sub/Add + PIPE_ALL barrier
 *
 * ofdm_mod 对称优化:
 *   - Phase 1 的 W32 常驻 L0A，并让同一 RHS tile 同时服务实/虚两个输出
 *   - Phase 3 将同核 symbol 沿 M 合并为 M<=128，W64 常驻 L0B
 *   - 每核数据完全独立，用局部 pipe event 取代跨核 SyncAll
 *   - batch 路径将两个 phase hand-off 留在 L1，删除 GM scratch；常量驻留
 *     在 L1，并在每个 tile 重新装入 L0，避免相邻 Cube 阶段覆盖 L0 内容
 *
 * dav_m200 限制:
 *   - Matmul 高层 API 不能在同一 kernel 内多 mm 不同 tiling → 用基础 Mmad ISA
 *   - 实测 M=32 N=64 单 Mmad 输出后半零 → N=64 切 4 个 N=16 sub-cube
 *
 * 共享常量见 ofdm_demod.h
 */
#define OFDM_DEMOD_BATCH

#include "kernel_operator.h"
#include "ofdm_demod.h"

using namespace AscendC;
using namespace ofdm_demod_batch;


// ═══════════════════════════════════════════════════════════════
//  OfdmDemod class
// ═══════════════════════════════════════════════════════════════
class OfdmDemod {
public:
    __aicore__ inline OfdmDemod() {}

    __aicore__ inline void Init(GM_ADDR input_gm,
                                 GM_ADDR w32_re_gm, GM_ADDR w32_im_gm,
                                 GM_ADDR w64_re_gm, GM_ADDR w64_im_gm,
                                 GM_ADDR tw_re_gm,  GM_ADDR tw_im_gm,
                                 GM_ADDR output_re_gm, GM_ADDR output_im_gm,
#ifndef OFDM_DEMOD_BATCH
                                 GM_ADDR scratch_gm,
#endif
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

    // ★ 优化: fused 2-Mmad 累加, 输出 dst = A1·B1 + A2·B2
    __aicore__ inline void RunCubeP1_ComplexTileReuse(
        const LocalTensor<half> &wReL0,
        const LocalTensor<half> &wImL0,
        const LocalTensor<half> &wImNegL0,
        const GlobalTensor<half> &xReGm,
        const GlobalTensor<half> &xImGm,
        const LocalTensor<float> &dstRe,
        const LocalTensor<float> &dstIm);

#ifdef OFDM_DEMOD_BATCH
    __aicore__ inline void RunCubeP1_Fused(
        const LocalTensor<half> &w1L0, const LocalTensor<half> &x1L1,
        const LocalTensor<half> &w2L0, const LocalTensor<half> &x2L1,
        const LocalTensor<float> &dst);

    __aicore__ inline void RunCubeP3_L1Batch(
        const LocalTensor<half> &a1L1, const LocalTensor<half> &b1L0,
        const LocalTensor<half> &a2L1, const LocalTensor<half> &b2L0,
        const LocalTensor<float> &dst, uint16_t m);
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
    TQue<TPosition::CO1, 4> qCO1_;
#else
    TQue<TPosition::CO1, 2> qCO1_;
#endif

    // 持久 L1: W matrix + W_im 取负副本
    TBuf<TPosition::A1> bufW32reL1_, bufW32imL1_, bufW32imNegL1_;
    TBuf<TPosition::B1> bufW64reL1_, bufW64imL1_, bufW64imNegL1_;
#ifdef OFDM_DEMOD_BATCH
    // mod_batch-style on-chip hand-off: Phase 0→1 uses B1, Phase 2→3 uses A1.
    TBuf<TPosition::B1> bufXReL1_, bufXImL1_;
    TBuf<TPosition::A1> bufTwReL1_, bufTwImL1_;
#endif
    TBuf<TPosition::A2> bufW32L0A_;  // 3 × 32 × 32 half = 6 KiB
    TBuf<TPosition::B2> bufW64L0B_;  // 3 × 64 × 64 half = 24 KiB
};


// ═══════════════════════════════════════════════════════════════
//  Init
// ═══════════════════════════════════════════════════════════════
__aicore__ inline void OfdmDemod::Init(
    GM_ADDR input_gm,
    GM_ADDR w32_re_gm, GM_ADDR w32_im_gm,
    GM_ADDR w64_re_gm, GM_ADDR w64_im_gm,
    GM_ADDR tw_re_gm,  GM_ADDR tw_im_gm,
    GM_ADDR output_re_gm, GM_ADDR output_im_gm,
#ifndef OFDM_DEMOD_BATCH
    GM_ADDR scratch_gm,
#endif
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

#ifndef OFDM_DEMOD_BATCH
    // In the one-shot kernel Phase 0/1 X scratch can be reused by Phase 2 Tw.
    constexpr uint32_t SCR_PER_CORE = 2 * BATCH_X_ELEMS;
    auto sbase = reinterpret_cast<__gm__ half *>(scratch_gm) + blockId_ * SCR_PER_CORE;
    XreScr_ .SetGlobalBuffer(sbase + 0 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    XimScr_ .SetGlobalBuffer(sbase + 1 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    TwReScr_.SetGlobalBuffer(sbase + 0 * BATCH_X_ELEMS, BATCH_X_ELEMS);
    TwImScr_.SetGlobalBuffer(sbase + 1 * BATCH_X_ELEMS, BATCH_X_ELEMS);
#endif

    // UB buffers
    pipe_->InitBuffer(bufXi16_,      2 * N_FFT * sizeof(int16_t));   // 8 KB
    pipe_->InitBuffer(bufXfullHalf_, 2 * N_FFT * sizeof(half));      // 8 KB
    pipe_->InitBuffer(bufXre_,       TILE_PQ   * sizeof(half));      // 4 KB
    pipe_->InitBuffer(bufXim_,       TILE_PQ   * sizeof(half));      // 4 KB
    pipe_->InitBuffer(bufTmp_,       TILE_PQ   * sizeof(half));      // 4 KB
    pipe_->InitBuffer(bufTmp2_,      TILE_PQ   * sizeof(half));      // 4 KB
    pipe_->InitBuffer(bufTwRe_,      TILE_PQ   * sizeof(half));      // 4 KB
    pipe_->InitBuffer(bufTwIm_,      TILE_PQ   * sizeof(half));      // 4 KB
    pipe_->InitBuffer(bufX1re_,      BATCH_X_ELEMS * sizeof(half));  // 16 KB
    pipe_->InitBuffer(bufX1im_,      BATCH_X_ELEMS * sizeof(half));  // 16 KB
    pipe_->InitBuffer(bufCubeAcc_,   BATCH_X_ELEMS * sizeof(float)); // 32 KB
    pipe_->InitBuffer(bufCubeNd_,    BATCH_X_ELEMS * sizeof(float)); // 32 KB

    // Mmad queues
    pipe_->InitBuffer(qA1_,  2, MAX_M_BATCH * Q * sizeof(half));     // 2×16 KB
    pipe_->InitBuffer(qA2_,  2, MAX_M_BATCH * Q * sizeof(half));     // 2×16 KB
    pipe_->InitBuffer(qB1_,  2, Q * Q * sizeof(half));               // 2×8 KB
    pipe_->InitBuffer(qB2_,  2, Q * Q * sizeof(half));               // 2×8 KB
#ifdef OFDM_DEMOD_BATCH
    pipe_->InitBuffer(qCO1_, 4, M_MMAD * Q * sizeof(float));         // 4×8 KB
#else
    pipe_->InitBuffer(qCO1_, 2, M_MMAD * Q * sizeof(float));         // 2×8 KB
#endif

    // 持久 L1 + W_im 取负副本
    pipe_->InitBuffer(bufW32reL1_,    P * P * sizeof(half));         // 2 KB
    pipe_->InitBuffer(bufW32imL1_,    P * P * sizeof(half));         // 2 KB
    pipe_->InitBuffer(bufW32imNegL1_, P * P * sizeof(half));         // 2 KB (新增)
    pipe_->InitBuffer(bufW64reL1_,    Q * Q * sizeof(half));         // 8 KB
    pipe_->InitBuffer(bufW64imL1_,    Q * Q * sizeof(half));         // 8 KB
    pipe_->InitBuffer(bufW64imNegL1_, Q * Q * sizeof(half));         // 8 KB (新增)
#ifdef OFDM_DEMOD_BATCH
    pipe_->InitBuffer(bufXReL1_, BATCH_X_ELEMS * sizeof(half));      // 16 KB
    pipe_->InitBuffer(bufXImL1_, BATCH_X_ELEMS * sizeof(half));      // 16 KB
    pipe_->InitBuffer(bufTwReL1_, BATCH_X_ELEMS * sizeof(half));     // 16 KB
    pipe_->InitBuffer(bufTwImL1_, BATCH_X_ELEMS * sizeof(half));     // 16 KB
#endif
    pipe_->InitBuffer(bufW32L0A_, 3 * P * P * sizeof(half));         // 6 KB
    pipe_->InitBuffer(bufW64L0B_, 3 * Q * Q * sizeof(half));         // 24 KB
}


// ═══════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════

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
    // ★ X1 优化: 删原来 Cast 后的 PipeBarrier<PIPE_V>,
    //    调用者 (Phase 1/3 sym 循环) 在下一步 RunCubeXX_Fused 前/末尾
    //    已有兜底 PIPE_V/PIPE_ALL barrier, 此处重复.
}


// ═══════════════════════════════════════════════════════════════
//  Phase 0: CP removal + de-interleave
// ═══════════════════════════════════════════════════════════════
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
#ifdef OFDM_DEMOD_BATCH
    auto xReL1 = bufXReL1_.Get<half>();
    auto xImL1 = bufXImL1_.Get<half>();
    // Pack four [32,16] panels directly into the B1 layout consumed by
    // Phase 1, eliminating both GM scratch planes and the GM→B1 copy.
    for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
        const uint32_t panelOff = s * TILE_PQ + ns * K_PHASE1 * N_SUB;
        DataCopy(xReL1[panelOff], xre[ns * N_SUB],
                 {K_PHASE1, 1, uint16_t(Q / 16 - 1), 0});
        DataCopy(xImL1[panelOff], xim[ns * N_SUB],
                 {K_PHASE1, 1, uint16_t(Q / 16 - 1), 0});
    }
    event_t e3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e3); WaitFlag<HardEvent::MTE3_MTE2>(e3);
#else
    DataCopy(XreScr_[s * TILE_PQ], xre, TILE_PQ);
    DataCopy(XimScr_[s * TILE_PQ], xim, TILE_PQ);
    event_t e3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e3); WaitFlag<HardEvent::MTE3_MTE2>(e3);
#endif
}

__aicore__ inline void OfdmDemod::Phase0_CopyIn()
{
    for (uint32_t s = 0; s < SYMBOLS_PER_CORE; ++s) {
        uint32_t sym = sym_start_ + s;
        if (sym >= N_SYMBOL) continue;
        LoadSymbol(s, sym);
    }
}


// ═══════════════════════════════════════════════════════════════
//  Phase 1: DFT-32 — X1 = W32 @ X (fused Mmad)
//  X1_re = W_re·X_re + (-W_im)·X_im
//  X1_im = W_re·X_im + W_im·X_re
// ═══════════════════════════════════════════════════════════════

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
    for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
        uint32_t bOff = ns * N_SUB;
        uint32_t dstOff = ns * M_MMAD * N_SUB;

        LocalTensor<float> coRe = qCO1_.AllocTensor<float>();
        LocalTensor<float> coIm = qCO1_.AllocTensor<float>();
        MmadParams mp;
        mp.m = M_MMAD; mp.n = N_SUB; mp.k = K_PHASE1;

        // 同一 Xre tile 同时用于 Yre=Wre*Xre 与 Yim=Wim*Xre。
        LocalTensor<half> b1Re = qB1_.AllocTensor<half>();
        DataCopy(b1Re, xReGm[bOff], { K_PHASE1, 1, uint16_t(Q / 16 - 1), 0 });
        qB1_.EnQue(b1Re);
        b1Re = qB1_.DeQue<half>();
        LocalTensor<half> b2Re = qB2_.AllocTensor<half>();
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks; loadP.srcStride = 1; loadP.ifTranspose = true;
        LoadData(b2Re, b1Re, loadP);
        qB2_.EnQue(b2Re); qB1_.FreeTensor(b1Re);
        b2Re = qB2_.DeQue<half>();
        mp.cmatrixInitVal = true;
        Mmad(coRe, wReL0, b2Re, mp);
        Mmad(coIm, wImL0, b2Re, mp);
        qB2_.FreeTensor(b2Re);

        // 同一 Xim tile 同时用于 Yre=(-Wim)*Xim 与 Yim=Wre*Xim。
        LocalTensor<half> b1Im = qB1_.AllocTensor<half>();
        DataCopy(b1Im, xImGm[bOff], { K_PHASE1, 1, uint16_t(Q / 16 - 1), 0 });
        qB1_.EnQue(b1Im);
        b1Im = qB1_.DeQue<half>();
        LocalTensor<half> b2Im = qB2_.AllocTensor<half>();
        LoadData(b2Im, b1Im, loadP);
        qB2_.EnQue(b2Im); qB1_.FreeTensor(b1Im);
        b2Im = qB2_.DeQue<half>();
        mp.cmatrixInitVal = false;
        Mmad(coRe, wImNegL0, b2Im, mp);
        Mmad(coIm, wReL0, b2Im, mp);
        qCO1_.EnQue(coRe); qCO1_.EnQue(coIm);
        qB2_.FreeTensor(b2Im);
        coRe = qCO1_.DeQue<float>(); coIm = qCO1_.DeQue<float>();

        DataCopyParams dcp; dcp.blockCount = 1;
        dcp.blockLen = uint16_t(M_MMAD * N_SUB / 256);
        DataCopyEnhancedParams ep; ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(dstRe[dstOff], coRe, dcp, ep);
        DataCopy(dstIm[dstOff], coIm, dcp, ep);
        qCO1_.FreeTensor(coRe); qCO1_.FreeTensor(coIm);
    }
}

#ifdef OFDM_DEMOD_BATCH
__aicore__ inline void OfdmDemod::RunCubeP1_Fused(
    const LocalTensor<half> &w1L0, const LocalTensor<half> &x1L1,
    const LocalTensor<half> &w2L0, const LocalTensor<half> &x2L1,
    const LocalTensor<float> &dst)
{
    constexpr uint16_t kBlocks = K_PHASE1 / 16;
    for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
        const uint32_t bOff = ns * K_PHASE1 * N_SUB;
        const uint32_t dstOff = ns * M_MMAD * N_SUB;
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks;
        loadP.srcStride = 1;
        loadP.ifTranspose = true;

        LocalTensor<half> b2 = qB2_.AllocTensor<half>();
        LoadData(b2, x1L1[bOff], loadP);
        qB2_.EnQue(b2);
        b2 = qB2_.DeQue<half>();
        LocalTensor<float> co = qCO1_.AllocTensor<float>();
        MmadParams mp;
        mp.m = M_MMAD; mp.n = N_SUB; mp.k = K_PHASE1; mp.cmatrixInitVal = true;
        Mmad(co, w1L0, b2, mp);
        qB2_.FreeTensor(b2);

        b2 = qB2_.AllocTensor<half>();
        LoadData(b2, x2L1[bOff], loadP);
        qB2_.EnQue(b2);
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

#ifdef OFDM_DEMOD_BATCH
    if (batchIdx_ == 0) {
#endif
    CopyNd2Nz(w32ReL1, w32ReG_, M_MMAD, K_PHASE1);
    CopyNd2Nz(w32ImL1, w32ImG_, M_MMAD, K_PHASE1);
    PipeBarrier<PIPE_ALL>();

    // ⚠️ Muls 是 Vector 操作, 不能直接对 L1 操作!
    // L1 不是 Vector 工作区, Muls 输入要求 UB.
    // 解决方案: 先 L1 → UB → Muls 取负 → UB → L1
    // 但这样代价高. 改成在 GM 端预生成? 太麻烦.
    //
    // 正确方法: 直接对 W_im 在 L1 中做取负? 不行, Muls 操作不了 L1.
    //
    // 折中: 把 W_im 取负后存到 GM scratch, 然后 CopyNd2Nz 加载?
    // 太复杂. 改用第二个方案:
    //
    // 方案 B: W_im 不取负, 第二个 Mmad 用 -1 的 Muls 后再 Mmad? 也不对.
    //
    // 方案 C: 让 host 端生成 W_im_neg.bin, kernel 直接加载.
    // 这个最干净, 但要改 ref.py 和 main.cpp.
    //
    // 当前实现: 我们在 UB 里准备 W_im_neg, 然后 CopyNd2Nz 到 L1.
    // 但这需要 UB 中转 + 额外 P*P + Q*Q half 空间.
    //
    // 最简洁 — 直接拿 w32ImL1, 走 UB 中转一次:
    {
        auto tmpUB = bufTmp_.Get<half>();   // 复用 tmp buffer (4 KB > P*P*2B=2KB)
        // 拷 L1 → UB
        DataCopy(tmpUB, w32ImL1, P * P);
        PipeBarrier<PIPE_ALL>();
        // UB 内取负
        Muls(tmpUB, tmpUB, (half)-1.0, P * P);
        PipeBarrier<PIPE_V>();
        // UB → L1 (NZ 格式不变, 直接 DataCopy)
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e); WaitFlag<HardEvent::V_MTE3>(e);
        DataCopy(w32ImNegL1, tmpUB, P * P);
        PipeBarrier<PIPE_ALL>();
    }
#ifdef OFDM_DEMOD_BATCH
    }
#endif

    // L1 constants persist across batch items. Reload L0 per tile because the
    // later qA2 activity may invalidate L0A residency.
    LoadL1ToL0A(w32ReL0,    w32ReL1,    M_MMAD / 16, K_PHASE1 / 16);
    LoadL1ToL0A(w32ImL0,    w32ImL1,    M_MMAD / 16, K_PHASE1 / 16);
    LoadL1ToL0A(w32ImNegL0, w32ImNegL1, M_MMAD / 16, K_PHASE1 / 16);
    PipeBarrier<PIPE_ALL>();

    for (uint32_t s = 0; s < SYMBOLS_PER_CORE; ++s) {
        uint32_t sym = sym_start_ + s;
        if (sym >= N_SYMBOL) continue;
        uint32_t symOff = s * TILE_PQ;
#ifdef OFDM_DEMOD_BATCH
        auto xReL1 = bufXReL1_.Get<half>();
        auto xImL1 = bufXImL1_.Get<half>();
        auto xreSym = xReL1[symOff];
        auto ximSym = xImL1[symOff];
        RunCubeP1_Fused(w32ReL0, xreSym, w32ImNegL0, ximSym, cubeAccRe);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(x1reBatch[symOff], cubeAccRe, cubeNd);
        PipeBarrier<PIPE_V>();
        RunCubeP1_Fused(w32ReL0, ximSym, w32ImL0, xreSym, cubeAccIm);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(x1imBatch[symOff], cubeAccIm, cubeNd);
#else
        auto xreSym = XreScr_[symOff];
        auto ximSym = XimScr_[symOff];
        RunCubeP1_ComplexTileReuse(w32ReL0, w32ImL0, w32ImNegL0,
                                   xreSym, ximSym, cubeAccRe, cubeAccIm);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(x1reBatch[symOff], cubeAccRe, cubeNd);
        PipeBarrier<PIPE_V>();
        NzToNdAndCast(x1imBatch[symOff], cubeAccIm, cubeNd);
#endif
        PipeBarrier<PIPE_V>();
    }

    PipeBarrier<PIPE_ALL>();
}


// ═══════════════════════════════════════════════════════════════
//  Phase 2: Twiddle — Tw = X1 ⊙ twiddle
// ═══════════════════════════════════════════════════════════════
__attribute__((noinline)) __aicore__ void OfdmDemod::Phase2_Twiddle()
{
    auto tmp       = bufTmp_.Get<half>();
    auto tmp2      = bufTmp2_.Get<half>();
    auto twre      = bufTwRe_.Get<half>();
    auto twim      = bufTwIm_.Get<half>();
    auto x1reBatch = bufX1re_.Get<half>();
    auto x1imBatch = bufX1im_.Get<half>();
#ifdef OFDM_DEMOD_BATCH
    auto twReL1 = bufTwReL1_.Get<half>();
    auto twImL1 = bufTwImL1_.Get<half>();
    uint16_t validSymbols = uint16_t(N_SYMBOL - sym_start_);
    if (validSymbols > SYMBOLS_PER_CORE) validSymbols = SYMBOLS_PER_CORE;
    const uint16_t m = validSymbols * P;
#endif

    // Twiddle 常驻 UB；batch 内所有 tile 共享同一份系数。
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
#ifdef OFDM_DEMOD_BATCH
        // A1 NZ layout for one fused [m,64] matrix: K panels outermost,
        // followed by the 32 rows belonging to this symbol.
        for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
            const uint32_t panelOff = ns * N_SUB * m + s * P * N_SUB;
            DataCopy(twReL1[panelOff], tmp[ns * N_SUB],
                     {P, 1, uint16_t(Q / 16 - 1), 0});
        }
#else
        DataCopy(TwReScr_[symOff], tmp, TILE_PQ);
#endif
        event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e2); WaitFlag<HardEvent::MTE3_V>(e2);

        Mul(tmp,  xreS, twim, TILE_PQ); PipeBarrier<PIPE_V>();
        Mul(tmp2, ximS, twre, TILE_PQ); PipeBarrier<PIPE_V>();
        Add(tmp, tmp, tmp2, TILE_PQ);   PipeBarrier<PIPE_V>();
        event_t e3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e3); WaitFlag<HardEvent::V_MTE3>(e3);
#ifdef OFDM_DEMOD_BATCH
        for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
            const uint32_t panelOff = ns * N_SUB * m + s * P * N_SUB;
            DataCopy(twImL1[panelOff], tmp[ns * N_SUB],
                     {P, 1, uint16_t(Q / 16 - 1), 0});
        }
#else
        DataCopy(TwImScr_[symOff], tmp, TILE_PQ);
#endif
        event_t e4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e4); WaitFlag<HardEvent::MTE3_V>(e4);
    }
    // Phase 3 读取前只需本地 MTE3→MTE2 依赖。
    event_t eScratch = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eScratch); WaitFlag<HardEvent::MTE3_MTE2>(eScratch);
}


// ═══════════════════════════════════════════════════════════════
//  Phase 3: DFT-64 — Y = Tw @ W64^T (fused Mmad)
//  Y_re = Tw_re·W64re + Tw_im·(-W64im)
//  Y_im = Tw_re·W64im + Tw_im·W64re
// ═══════════════════════════════════════════════════════════════

#ifdef OFDM_DEMOD_BATCH
__aicore__ inline void OfdmDemod::RunCubeP3_L1Batch(
    const LocalTensor<half> &a1L1, const LocalTensor<half> &b1L0,
    const LocalTensor<half> &a2L1, const LocalTensor<half> &b2L0,
    const LocalTensor<float> &dst, uint16_t m)
{
    constexpr uint16_t kBlocks = K_PHASE3 / 16;
    const uint16_t mBlocks = m / 16;

    // The twiddle outputs already reside in packed A1 layout. Load each
    // complex operand once for all four N=16 sub-cubes.
    LocalTensor<half> a2_1 = qA2_.AllocTensor<half>();
    LocalTensor<half> a2_2 = qA2_.AllocTensor<half>();
    LoadL1ToL0A(a2_1, a1L1, mBlocks, kBlocks);
    LoadL1ToL0A(a2_2, a2L1, mBlocks, kBlocks);
    qA2_.EnQue(a2_1); qA2_.EnQue(a2_2);
    a2_1 = qA2_.DeQue<half>();
    a2_2 = qA2_.DeQue<half>();

    for (uint16_t ns = 0; ns < N_SPLITS; ++ns) {
        const uint32_t dstOff = ns * m * N_SUB;
        const uint32_t wOff = ns * K_PHASE3 * N_SUB;
        LocalTensor<float> co = qCO1_.AllocTensor<float>();
        MmadParams mp;
        mp.m = m; mp.n = N_SUB; mp.k = K_PHASE3; mp.cmatrixInitVal = true;
        Mmad(co, a2_1, b1L0[wOff], mp);
        mp.cmatrixInitVal = false;
        Mmad(co, a2_2, b2L0[wOff], mp);
        qCO1_.EnQue(co);

        co = qCO1_.DeQue<float>();
        DataCopyParams dcp;
        dcp.blockCount = 1;
        dcp.blockLen = uint16_t(m * N_SUB / 256);
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(dst[dstOff], co, dcp, ep);
        qCO1_.FreeTensor(co);
    }
    qA2_.FreeTensor(a2_1);
    qA2_.FreeTensor(a2_2);
}
#endif

__aicore__ inline void OfdmDemod::RunCubeP3_Batch(
    const GlobalTensor<half> &Agm_1, const LocalTensor<half> &Bl0_1,
    const GlobalTensor<half> &Agm_2, const LocalTensor<half> &Bl0_2,
    const LocalTensor<float> &dst, uint16_t m)
{
    constexpr uint16_t kBlocks = K_PHASE3 / 16;
    const uint16_t mBlocks = m / 16;

    // 把同核最多 4 个 [32,64] symbol 沿 M 合并；每个操作数只搬一次。
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

#ifdef OFDM_DEMOD_BATCH
    if (batchIdx_ == 0) {
#endif
        CopyNd2Nz(w64ReL1, w64ReG_T_, K_PHASE3, Q);
        CopyNd2Nz(w64ImL1, w64ImG_T_, K_PHASE3, Q);
        PipeBarrier<PIPE_ALL>();

    // W64_im 取负副本 (UB 中转)
    {
        // bufTmp_ 只有 4 KB；借用 Phase 3 输出前尚空闲的 16 KB buffer。
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
#ifdef OFDM_DEMOD_BATCH
    }
#endif

    // Keep GM→L1 and negation persistent, but reload L0B after Phase 1 qB2 use.
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
    const uint16_t m = validSymbols * P;
    const uint32_t outOff = batchIdx_ * OUTPUT_ELEMS_PER_BATCH + sym_start_ * N_FFT;
    auto twReL1 = bufTwReL1_.Get<half>();
    auto twImL1 = bufTwImL1_.Get<half>();
    auto yreBatch = bufX1re_.Get<half>();
    auto yimBatch = bufX1im_.Get<half>();

    // Fuse all symbols assigned to this core along M, exactly as mod_batch's
    // inverse Phase A. L1 inputs eliminate the GM→L1 leg entirely.
    RunCubeP3_L1Batch(twReL1, w64ReL0, twImL1, w64ImNegL0, cubeAcc, m);
    PipeBarrier<PIPE_V>();
    NzToNdAndCast(yreBatch, cubeAcc, cubeNd, m);

    RunCubeP3_L1Batch(twReL1, w64ImL0, twImL1, w64ReL0, cubeAcc, m);
    PipeBarrier<PIPE_V>();
    NzToNdAndCast(yimBatch, cubeAcc, cubeNd, m);
    PipeBarrier<PIPE_V>();

    event_t eOut = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eOut); WaitFlag<HardEvent::V_MTE3>(eOut);
    DataCopy(outReG_[outOff], yreBatch, m * Q);
    DataCopy(outImG_[outOff], yimBatch, m * Q);
#else
    uint16_t m = validSymbols * P;
    uint32_t outOff = sym_start_ * N_FFT;

    // [validSymbols,32,64] 合并为一次 [M,64]×[64,64]，M=128/64。
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


// ═══════════════════════════════════════════════════════════════
//  Process
// ═══════════════════════════════════════════════════════════════
__aicore__ inline void OfdmDemod::Process()
{
#ifdef OFDM_DEMOD_BATCH
    if (batchSize_ == 0) return;
    const uint32_t totalTiles = batchSize_ * TILES_PER_BATCH;
    for (uint32_t tile = blockId_; tile < totalTiles; tile += BLOCK_DIM) {
        batchIdx_  = tile / TILES_PER_BATCH;
        sym_start_ = (tile % TILES_PER_BATCH) * SYMBOLS_PER_CORE;
        // Each core owns its complete tile; phase hand-offs stay in local L1,
        // so no cross-core barrier is required.
        Phase0_CopyIn();
        Phase1_Dft32();
        Phase2_Twiddle();
        Phase3_Dft64();
    }
#else
    Phase0_CopyIn();
    Phase1_Dft32();
    Phase2_Twiddle();
    Phase3_Dft64();
#endif
}


// ═══════════════════════════════════════════════════════════════
//  Kernel entry
// ═══════════════════════════════════════════════════════════════
extern "C" __global__ __aicore__ void ofdm_demod_batch_kernel(
    GM_ADDR input_gm,
    GM_ADDR w32_re_gm,    GM_ADDR w32_im_gm,
    GM_ADDR w64_re_gm,    GM_ADDR w64_im_gm,
    GM_ADDR tw_re_gm,     GM_ADDR tw_im_gm,
#ifndef OFDM_DEMOD_BATCH
    GM_ADDR scratch_gm,
#endif
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
#ifndef OFDM_DEMOD_BATCH
            scratch_gm,
#endif
#ifdef OFDM_DEMOD_BATCH
            batch_size,
#endif
            &pipe);
    op.Process();
}
