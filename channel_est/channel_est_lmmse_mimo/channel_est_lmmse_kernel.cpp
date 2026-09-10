/**
 * Compile-time configurable low-rank LMMSE channel estimator for Ascend 310P.
 *
 * Optimizations borrowed from the 64x16 detector/precoder kernels:
 *   - eight-Rx grouping: one 16-column Cube tile is 8 Rx x 2 DMRS;
 *   - per-layer, pre-folded B=D*U^H and A=diag(sf)*Rhp*U factors;
 *   - Phase-A B factors remain in L0 across eight independent Rx groups;
 *   - raw L0 TBufs and explicit MTE1<->M events;
 *   - FixPipe converts FP32 accumulators directly to FP16 in UB;
 *   - Wt is loaded once per layer/SC tile and reused by all eight Rx groups.
 */
#include "kernel_operator.h"
#include "channel_est_lmmse.h"

using namespace AscendC;
using namespace airan::channel_est_lmmse;

#ifndef CE_PHASE2_SKINNY
#define CE_PHASE2_SKINNY 1
#endif
#ifndef CE_LAYER_BATCH
#define CE_LAYER_BATCH 1
#endif
#ifndef CE_CUBE_TIME_FUSED
#define CE_CUBE_TIME_FUSED 0
#endif
#ifndef CE_CUBE_TIME_POST_GEMM
#define CE_CUBE_TIME_POST_GEMM 0
#endif
namespace {
constexpr uint32_t TILE_ELEMS = 16 * 16;
constexpr uint32_t FACTOR_TILE_ELEMS = 16 * N_PILOT_PAD;
constexpr uint32_t T_LAYER_ELEMS = N_RX_GROUP * RANK * NCOL;
constexpr uint32_t A_TILE_ELEMS = RANK * SC_TILE;
constexpr uint32_t WT_TILE_ELEMS = N_SYMBOL * N_DMRS_SYMBOL * SC_TILE;
constexpr uint32_t ALL_RX_COLS = N_RX_GROUP * NCOL;
constexpr uint32_t ALL_RX_TILE_ELEMS = N_RX_GROUP * TILE_ELEMS;
constexpr uint32_t OUT_GROUP_ELEMS = RX_GROUP * N_SYMBOL * SC_TILE;
constexpr uint32_t CE_GROUP_BATCH = N_RX_GROUP < 4 ? N_RX_GROUP : 4;
#if CE_CUBE_TIME_FUSED
constexpr uint32_t CUBE_TIME_LAYER_BATCH = NL < 4 ? NL : 4;
constexpr uint32_t CUBE_TIME_RX_ELEMS = NR * SC_TILE;
#endif
#if CE_CUBE_TIME_POST_GEMM
constexpr uint32_t TIME_POST_LAYER_BATCH = 2;
constexpr uint32_t TIME_POST_K = N_DMRS_SYMBOL * SC_TILE;
constexpr uint32_t TIME_POST_N = N_SYMBOL * SC_TILE;
constexpr uint32_t TIME_POST_M = 2 * NR;
constexpr uint32_t TIME_POST_A_ELEMS = TIME_POST_M * TIME_POST_K;
constexpr uint32_t TIME_POST_P_ELEMS = TIME_POST_K * TIME_POST_N;
constexpr uint32_t TIME_POST_C_ELEMS = TIME_POST_M * TIME_POST_N;
constexpr uint32_t TIME_POST_P_L0_OFFSET = 3 * A_TILE_ELEMS;
static_assert(TIME_POST_P_L0_OFFSET + TIME_POST_P_ELEMS <= N_PILOT_PAD * NCOL,
              "post-frequency time matrix must fit existing L0B allocation");
static_assert(NL % TIME_POST_LAYER_BATCH == 0, "post-frequency layer batch must divide NL");
static_assert(NR % 16 == 0, "post-frequency Cube path requires NR=16,32,64");
#endif
static_assert(N_RX_GROUP % CE_GROUP_BATCH == 0, "group batch must divide Rx groups");
static_assert(ALL_RX_COLS % 16 == 0, "skinny Phase B rows must be Cube aligned");
static_assert(ALL_RX_COLS * RANK <= 2 * FACTOR_TILE_ELEMS,
              "all-Rx skinny left operand exceeds the inherited L0A allocation");
static_assert(3 * A_TILE_ELEMS <= N_PILOT_PAD * NCOL,
              "three frequency factors exceed the inherited L0B allocation");
static_assert(NL % CE_LAYER_BATCH == 0, "layer batch must divide NL");
static_assert(CE_LAYER_BATCH == 1, "current output path requires one layer at a time");
}

class ChannelEstLmmse {
public:
    __aicore__ inline ChannelEstLmmse() = default;

    __aicore__ inline void Init(
        GM_ADDR b_re, GM_ADDR b_im,
        GM_ADDR hls_re, GM_ADDR hls_im, GM_ADDR hls_neg_im,
        GM_ADDR a_re, GM_ADDR a_im, GM_ADDR a_neg_im, GM_ADDR wt_re, GM_ADDR wt_im,
        GM_ADDR t_re, GM_ADDR t_im,
        GM_ADDR out_re, GM_ADDR out_im,
        GM_ADDR workspace, GM_ADDR, TPipe *pipe)
    {
        pipe_ = pipe;
        blockId_ = GetBlockIdx();
        bReG_.SetGlobalBuffer((__gm__ half *)b_re, B_ELEMS);
        bImG_.SetGlobalBuffer((__gm__ half *)b_im, B_ELEMS);
        hlsReG_.SetGlobalBuffer((__gm__ half *)hls_re, HLS_ELEMS);
        hlsImG_.SetGlobalBuffer((__gm__ half *)hls_im, HLS_ELEMS);
        hlsNegImG_.SetGlobalBuffer((__gm__ half *)hls_neg_im, HLS_ELEMS);
        aReG_.SetGlobalBuffer((__gm__ half *)a_re, A_ELEMS);
        aImG_.SetGlobalBuffer((__gm__ half *)a_im, A_ELEMS);
        aNegImG_.SetGlobalBuffer((__gm__ half *)a_neg_im, A_ELEMS);
        wtReG_.SetGlobalBuffer((__gm__ half *)wt_re, WT_ELEMS);
        wtImG_.SetGlobalBuffer((__gm__ half *)wt_im, WT_ELEMS);
        tReG_.SetGlobalBuffer((__gm__ half *)t_re, T_ELEMS);
        tImG_.SetGlobalBuffer((__gm__ half *)t_im, T_ELEMS);
        outReG_.SetGlobalBuffer((__gm__ half *)out_re, OUT_ELEMS);
        outImG_.SetGlobalBuffer((__gm__ half *)out_im, OUT_ELEMS);
        syncG_.SetGlobalBuffer((__gm__ int32_t *)workspace);
#if CE_CUBE_TIME_FUSED
        tNegImG_.SetGlobalBuffer(
            (__gm__ half *)(workspace + MIN_SYNC_WORKSPACE_BYTES), T_ELEMS);
#endif

        // L1.  Phase-A and Phase-B buffers coexist but remain far below 1 MiB.
        pipe_->InitBuffer(factorReL1_, FACTOR_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(factorImL1_, FACTOR_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(hlsReL1_, N_PILOT_PAD * NCOL * sizeof(half));
        pipe_->InitBuffer(hlsImL1_, N_PILOT_PAD * NCOL * sizeof(half));
        pipe_->InitBuffer(hlsNegImL1_, N_PILOT_PAD * NCOL * sizeof(half));
#if CE_CUBE_TIME_FUSED
        pipe_->InitBuffer(tReL1_, CUBE_TIME_LAYER_BATCH * T_LAYER_ELEMS * sizeof(half));
        pipe_->InitBuffer(tImL1_, CUBE_TIME_LAYER_BATCH * T_LAYER_ELEMS * sizeof(half));
        pipe_->InitBuffer(tNegImL1_, CUBE_TIME_LAYER_BATCH * T_LAYER_ELEMS * sizeof(half));
#elif CE_CUBE_TIME_POST_GEMM
        pipe_->InitBuffer(tReL1_, TIME_POST_LAYER_BATCH * T_LAYER_ELEMS * sizeof(half));
        pipe_->InitBuffer(tImL1_, TIME_POST_LAYER_BATCH * T_LAYER_ELEMS * sizeof(half));
#else
        pipe_->InitBuffer(tReL1_, CE_LAYER_BATCH * T_LAYER_ELEMS * sizeof(half));
        pipe_->InitBuffer(tImL1_, CE_LAYER_BATCH * T_LAYER_ELEMS * sizeof(half));
#endif
        pipe_->InitBuffer(aReL1_, A_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(aImL1_, A_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(aNegImL1_, A_TILE_ELEMS * sizeof(half));
#if CE_CUBE_TIME_FUSED
        pipe_->InitBuffer(fused1ImL1_, A_TILE_ELEMS * sizeof(half));
#elif CE_CUBE_TIME_POST_GEMM
        pipe_->InitBuffer(timePostWeightL1_, TIME_POST_P_ELEMS * sizeof(half));
        pipe_->InitBuffer(timePostInputL1_, TIME_POST_A_ELEMS * sizeof(half));
#endif

        // Raw L0 buffers.  A2 holds two full 16x832 Phase-A factors.
        pipe_->InitBuffer(a2Buf_, 2 * FACTOR_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(b2Buf_, N_PILOT_PAD * NCOL * sizeof(half));
#if CE_CUBE_TIME_POST_GEMM
        pipe_->InitBuffer(coBuf_, TIME_POST_C_ELEMS * sizeof(float));
#else
        pipe_->InitBuffer(coBuf_, 2 * ALL_RX_TILE_ELEMS * sizeof(float));
#endif

        // UB. FixPipe writes half directly into hfRe/hfIm.
        pipe_->InitBuffer(hfReBuf_, ALL_RX_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(hfImBuf_, ALL_RX_TILE_ELEMS * sizeof(half));
#if CE_CUBE_TIME_POST_GEMM
        pipe_->InitBuffer(timePostOutBuf_,
                          TIME_POST_LAYER_BATCH * TIME_POST_C_ELEMS * sizeof(half));
#elif CE_CUBE_TIME_FUSED
        pipe_->InitBuffer(fused1ReBuf_, ALL_RX_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(fused1ImBuf_, ALL_RX_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(cubeOutReBuf_, CUBE_TIME_RX_ELEMS * sizeof(half));
        pipe_->InitBuffer(cubeOutImBuf_, CUBE_TIME_RX_ELEMS * sizeof(half));
        pipe_->InitBuffer(cubeTmpReBuf_, CUBE_TIME_RX_ELEMS * sizeof(half));
        pipe_->InitBuffer(cubeTmpImBuf_, CUBE_TIME_RX_ELEMS * sizeof(half));
        pipe_->InitBuffer(gatherIdxBuf_, CUBE_TIME_RX_ELEMS * sizeof(uint32_t));
#else
        pipe_->InitBuffer(wtReBuf_, WT_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(outGroupRe0Buf_, OUT_GROUP_ELEMS * sizeof(half));
        pipe_->InitBuffer(outGroupRe1Buf_, OUT_GROUP_ELEMS * sizeof(half));
        pipe_->InitBuffer(outGroupRe2Buf_, OUT_GROUP_ELEMS * sizeof(half));
        pipe_->InitBuffer(outGroupRe3Buf_, OUT_GROUP_ELEMS * sizeof(half));
        pipe_->InitBuffer(outGroupIm0Buf_, OUT_GROUP_ELEMS * sizeof(half));
        pipe_->InitBuffer(outGroupIm1Buf_, OUT_GROUP_ELEMS * sizeof(half));
        pipe_->InitBuffer(outGroupIm2Buf_, OUT_GROUP_ELEMS * sizeof(half));
        pipe_->InitBuffer(outGroupIm3Buf_, OUT_GROUP_ELEMS * sizeof(half));
#endif
        pipe_->InitBuffer(syncBuf_, BLOCK_DIM * 32 * sizeof(int32_t));

        evMte2Mte1_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
        evMte1M_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
        evMMte1_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
        evVMte3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
#if CE_CUBE_TIME_POST_GEMM
        evMte3Mte1_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE1));
#endif
    }

    __aicore__ inline void Process()
    {
        PhaseA();
        PipeBarrier<PIPE_ALL>();
        SyncAll(syncG_, syncBuf_.Get<int32_t>(), static_cast<int32_t>(BLOCK_DIM));
#if CE_CUBE_TIME_FUSED
        InitGatherIndex();
        PhaseBCubeTimeFused();
#elif CE_CUBE_TIME_POST_GEMM
        PhaseBPostFrequencyCubeTime();
#else
        PhaseBAndTime();
#endif
    }

private:
    __aicore__ inline void Mte2ToMte1()
    {
        SetFlag<HardEvent::MTE2_MTE1>(evMte2Mte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(evMte2Mte1_);
    }

    __aicore__ inline void Mte1ToM()
    {
        SetFlag<HardEvent::MTE1_M>(evMte1M_);
        WaitFlag<HardEvent::MTE1_M>(evMte1M_);
    }

    __aicore__ inline void MToMte1()
    {
        SetFlag<HardEvent::M_MTE1>(evMMte1_);
        WaitFlag<HardEvent::M_MTE1>(evMMte1_);
    }

    __aicore__ inline void VToMte3()
    {
        SetFlag<HardEvent::V_MTE3>(evVMte3_);
        WaitFlag<HardEvent::V_MTE3>(evVMte3_);
    }

#if CE_CUBE_TIME_POST_GEMM
    __aicore__ inline void Mte3ToMte1()
    {
        SetFlag<HardEvent::MTE3_MTE1>(evMte3Mte1_);
        WaitFlag<HardEvent::MTE3_MTE1>(evMte3Mte1_);
    }
#endif

    __aicore__ inline void LoadFactor(uint32_t layer, uint32_t rankTile)
    {
        const uint32_t base = layer * B_LAYER_ELEMS + rankTile * FACTOR_TILE_ELEMS;
        DataCopy(factorReL1_.Get<half>(), bReG_[base], FACTOR_TILE_ELEMS);
        DataCopy(factorImL1_.Get<half>(), bImG_[base], FACTOR_TILE_ELEMS);
        Mte2ToMte1();

        auto a2 = a2Buf_.Get<half>();
        LoadData2DParams p;
        p.repeatTimes = PILOT_K_BLOCK;
        p.srcStride = 1;
        p.ifTranspose = false;
        LoadData(a2, factorReL1_.Get<half>(), p);
        LoadData(a2[FACTOR_TILE_ELEMS], factorImL1_.Get<half>(), p);
        Mte1ToM();
    }

    __aicore__ inline void LoadHls(uint32_t layer, uint32_t group)
    {
        const uint32_t base = (layer * N_RX_GROUP + group) * N_PILOT_PAD * NCOL;
        const uint32_t count = N_PILOT_PAD * NCOL;
        DataCopy(hlsReL1_.Get<half>(), hlsReG_[base], count);
        DataCopy(hlsImL1_.Get<half>(), hlsImG_[base], count);
        DataCopy(hlsNegImL1_.Get<half>(), hlsNegImG_[base], count);
        Mte2ToMte1();
    }

    __aicore__ inline void LoadHlsOperand(const LocalTensor<half> &src)
    {
        LoadData2DParams p;
        p.repeatTimes = PILOT_K_BLOCK;
        p.srcStride = 1;
        p.ifTranspose = true;
        LoadData(b2Buf_.Get<half>(), src, p);
        Mte1ToM();
    }

    __aicore__ inline void GemmT(uint32_t layer, uint32_t group, uint32_t rankTile)
    {
        auto a2 = a2Buf_.Get<half>();
        auto b2 = b2Buf_.Get<half>();
        auto co = coBuf_.Get<float>();
        constexpr uint32_t coImOffset = TILE_ELEMS;
        MmadParams mp;
        mp.m = 16;
        mp.n = 16;
        mp.k = N_PILOT_PAD;

        // B = Br + jBi, h = hr + jhi.
        LoadHlsOperand(hlsReL1_.Get<half>());
        mp.cmatrixInitVal = true;
        Mmad(co, a2, b2, mp);                                  // tr = Br*hr
        Mmad(co[coImOffset], a2[FACTOR_TILE_ELEMS], b2, mp);   // ti = Bi*hr
        MToMte1();

        LoadHlsOperand(hlsImL1_.Get<half>());
        mp.cmatrixInitVal = false;
        Mmad(co[coImOffset], a2, b2, mp);                       // ti += Br*hi
        MToMte1();

        LoadHlsOperand(hlsNegImL1_.Get<half>());
        Mmad(co, a2[FACTOR_TILE_ELEMS], b2, mp);                // tr += Bi*(-hi)
        MToMte1();
        PipeBarrier<PIPE_ALL>();

        auto tr = hfReBuf_.Get<half>();
        auto ti = hfImBuf_.Get<half>();
        DataCopyParams dcp;
        dcp.blockCount = 1;
        dcp.blockLen = 1;
        dcp.srcStride = 0;
        dcp.dstStride = 0;
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(tr, co, dcp, ep);
        DataCopy(ti, co[coImOffset], dcp, ep);
        PipeBarrier<PIPE_ALL>();

        VToMte3();
        const uint32_t dst = ((layer * N_RX_GROUP + group) * RANK + rankTile * 16) * NCOL;
        SetAtomicAdd<half>();
        DataCopy(tReG_[dst], tr, TILE_ELEMS);
        DataCopy(tImG_[dst], ti, TILE_ELEMS);
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
#if CE_CUBE_TIME_FUSED
        Muls(ti, ti, static_cast<half>(-1.0f), TILE_ELEMS);
        VToMte3();
        DataCopy(tNegImG_[dst], ti, TILE_ELEMS);
        PipeBarrier<PIPE_ALL>();
#endif
    }

    __aicore__ inline void PhaseA()
    {
        // NL x rank-tile independent tasks; each task consumes every Rx group.
        for (uint32_t task = blockId_; task < NL * N_RANK_TILE; task += BLOCK_DIM) {
            const uint32_t layer = task / N_RANK_TILE;
            const uint32_t rankTile = task % N_RANK_TILE;
            LoadFactor(layer, rankTile);
            for (uint32_t group = 0; group < N_RX_GROUP; ++group) {
                LoadHls(layer, group);
                GemmT(layer, group, rankTile);
            }
        }
    }

    __aicore__ inline void LoadTBatch(uint32_t firstLayer)
    {
        const uint32_t base = firstLayer * T_LAYER_ELEMS;
        constexpr uint32_t count = CE_LAYER_BATCH * T_LAYER_ELEMS;
        DataCopy(tReL1_.Get<half>(), tReG_[base], count);
        DataCopy(tImL1_.Get<half>(), tImG_[base], count);
        Mte2ToMte1();
    }

    __aicore__ inline void LoadAAndWt(uint32_t layer, uint32_t scTile)
    {
        const uint32_t aBase = layer * A_LAYER_ELEMS + scTile * A_TILE_ELEMS;
        DataCopy(aReL1_.Get<half>(), aReG_[aBase], A_TILE_ELEMS);
        DataCopy(aImL1_.Get<half>(), aImG_[aBase], A_TILE_ELEMS);
        DataCopy(aNegImL1_.Get<half>(), aNegImG_[aBase], A_TILE_ELEMS);
        Mte2ToMte1();

        LoadAWeightsToL0();

        const uint32_t wBase = layer * WT_LAYER_ELEMS + scTile * WT_TILE_ELEMS;
        DataCopy(wtReBuf_.Get<half>(), wtReG_[wBase], WT_TILE_ELEMS);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadAWeightsToL0()
    {
        // Three immutable right operands share one raw L0B allocation.
        auto b2 = b2Buf_.Get<half>();
        LoadData2DParams p;
        p.repeatTimes = N_RANK_TILE;
        p.srcStride = 1;
        p.ifTranspose = true;
        LoadData(b2, aReL1_.Get<half>(), p);
        LoadData(b2[A_TILE_ELEMS], aImL1_.Get<half>(), p);
        LoadData(b2[2 * A_TILE_ELEMS], aNegImL1_.Get<half>(), p);
        Mte1ToM();
    }

    __aicore__ inline void LoadTOperand(const LocalTensor<half> &dst, const LocalTensor<half> &src)
    {
        LoadData2DParams p;
        p.repeatTimes = N_RANK_TILE;
        p.srcStride = 1;
        p.ifTranspose = true;
        LoadData(dst, src, p);
    }

    __aicore__ inline void GemmHfGroup(uint32_t layerInBatch, uint32_t group)
    {
        const uint32_t tBase = layerInBatch * T_LAYER_ELEMS + group * RANK * NCOL;
        auto a2 = a2Buf_.Get<half>();
        auto b2 = b2Buf_.Get<half>();
        auto ar = b2;
        auto ai = b2[A_TILE_ELEMS];
        auto ani = b2[2 * A_TILE_ELEMS];
        auto co = coBuf_.Get<float>();
        constexpr uint32_t coImOffset = TILE_ELEMS;
        MmadParams mp;
        mp.m = NCOL;
        mp.n = SC_TILE;
        mp.k = RANK;

        LoadTOperand(a2, tReL1_.Get<half>()[tBase]);
        Mte1ToM();
        mp.cmatrixInitVal = true;
        Mmad(co, a2, ar, mp);                       // hr = tr*Ar
        Mmad(co[coImOffset], a2, ai, mp);           // hi = tr*Ai
        MToMte1();

        LoadTOperand(a2, tImL1_.Get<half>()[tBase]);
        Mte1ToM();
        mp.cmatrixInitVal = false;
        Mmad(co, a2, ani, mp);                      // hr += ti*(-Ai)
        Mmad(co[coImOffset], a2, ar, mp);           // hi += ti*Ar
        MToMte1();
        PipeBarrier<PIPE_ALL>();

        auto hr = hfReBuf_.Get<half>();
        auto hi = hfImBuf_.Get<half>();
        DataCopyParams dcp;
        dcp.blockCount = 1;
        dcp.blockLen = 1;
        dcp.srcStride = 0;
        dcp.dstStride = 0;
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(hr, co, dcp, ep);
        DataCopy(hi, co[coImOffset], dcp, ep);
        PipeBarrier<PIPE_ALL>();
    }

    // True cross-group skinny GEMM:
    //   t^T[128,96] * A[96,16] -> hf[128,16].
    __aicore__ inline void LoadTAllToL0A(const LocalTensor<half> &src)
    {
        auto a2 = a2Buf_.Get<half>();
        LoadData2DParams p;
        p.repeatTimes = N_RANK_TILE;
        p.srcStride = 1;
        p.ifTranspose = true;
        for (uint32_t group = 0; group < N_RX_GROUP; ++group) {
            const uint32_t offset = group * A_TILE_ELEMS;
            LoadData(a2[offset], src[offset], p);
        }
        Mte1ToM();
    }

    __aicore__ inline void GemmHfAllGroups(uint32_t layerInBatch)
    {
        auto a2 = a2Buf_.Get<half>();
        auto b2 = b2Buf_.Get<half>();
        auto ar = b2;
        auto ai = b2[A_TILE_ELEMS];
        auto ani = b2[2 * A_TILE_ELEMS];
        auto co = coBuf_.Get<float>();
        constexpr uint32_t coImOffset = ALL_RX_TILE_ELEMS;
        MmadParams mp;
        mp.m = ALL_RX_COLS;
        mp.n = SC_TILE;
        mp.k = RANK;

        const uint32_t tBase = layerInBatch * T_LAYER_ELEMS;
        LoadTAllToL0A(tReL1_.Get<half>()[tBase]);
        mp.cmatrixInitVal = true;
        Mmad(co, a2, ar, mp);
        Mmad(co[coImOffset], a2, ai, mp);
        MToMte1();

        LoadTAllToL0A(tImL1_.Get<half>()[tBase]);
        mp.cmatrixInitVal = false;
        Mmad(co, a2, ani, mp);
        Mmad(co[coImOffset], a2, ar, mp);
        MToMte1();
        PipeBarrier<PIPE_ALL>();

        auto hr = hfReBuf_.Get<half>();
        auto hi = hfImBuf_.Get<half>();
        DataCopyParams dcp;
        dcp.blockCount = N_RX_GROUP;
        dcp.blockLen = 1;
        dcp.srcStride = 0;
        dcp.dstStride = 0;
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(hr, co, dcp, ep);
        DataCopy(hi, co[coImOffset], dcp, ep);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void TimeToGroup(uint32_t hfGroupBase, uint32_t batchSlot)
    {
        auto hfR = hfReBuf_.Get<half>();
        auto hfI = hfImBuf_.Get<half>();
        auto wtR = wtReBuf_.Get<half>();
        uint64_t mask[2] = {0x000000000000FFFFULL, 0};
        LocalTensor<half> outAllR;
        LocalTensor<half> outAllI;
        if (batchSlot == 0) {
            outAllR = outGroupRe0Buf_.Get<half>();
            outAllI = outGroupIm0Buf_.Get<half>();
        } else if (batchSlot == 1) {
            outAllR = outGroupRe1Buf_.Get<half>();
            outAllI = outGroupIm1Buf_.Get<half>();
        } else if (batchSlot == 2) {
            outAllR = outGroupRe2Buf_.Get<half>();
            outAllI = outGroupIm2Buf_.Get<half>();
        } else {
            outAllR = outGroupRe3Buf_.Get<half>();
            outAllI = outGroupIm3Buf_.Get<half>();
        }
        const BinaryRepeatParams rp(1, 1, 1, 1, 2, 0);

        for (uint32_t rx = 0; rx < RX_GROUP; ++rx) {
            const uint32_t ubBase = rx * N_SYMBOL * SC_TILE;
            auto outR = outAllR[ubBase];
            auto outI = outAllI[ubBase];
            const uint32_t f0 = hfGroupBase + (2 * rx) * SC_TILE;
            const uint32_t f1 = f0 + SC_TILE;
            Mul(outR, wtR, hfR[f0], mask, N_SYMBOL, rp);
            Mul(outI, wtR, hfI[f0], mask, N_SYMBOL, rp);
            MulAddDst(outR, wtR[SC_TILE], hfR[f1], mask, N_SYMBOL, rp);
            MulAddDst(outI, wtR[SC_TILE], hfI[f1], mask, N_SYMBOL, rp);
        }
    }

    __aicore__ inline void StoreGroupBatch(uint32_t layer, uint32_t scTile,
                                           uint32_t firstGroup)
    {
        VToMte3();
        DataCopyParams dp;
        dp.blockCount = N_SYMBOL;
        dp.blockLen = 1;
        dp.srcStride = 0;
        dp.dstStride = (N_SC_PAD - SC_TILE) / 16;
        const uint32_t sc0 = scTile * SC_TILE;
        for (uint32_t batchSlot = 0; batchSlot < CE_GROUP_BATCH; ++batchSlot) {
            LocalTensor<half> outAllR;
            LocalTensor<half> outAllI;
            if (batchSlot == 0) {
                outAllR = outGroupRe0Buf_.Get<half>();
                outAllI = outGroupIm0Buf_.Get<half>();
            } else if (batchSlot == 1) {
                outAllR = outGroupRe1Buf_.Get<half>();
                outAllI = outGroupIm1Buf_.Get<half>();
            } else if (batchSlot == 2) {
                outAllR = outGroupRe2Buf_.Get<half>();
                outAllI = outGroupIm2Buf_.Get<half>();
            } else {
                outAllR = outGroupRe3Buf_.Get<half>();
                outAllI = outGroupIm3Buf_.Get<half>();
            }
            const uint32_t group = firstGroup + batchSlot;
            for (uint32_t rx = 0; rx < RX_GROUP; ++rx) {
                const uint32_t globalRx = group * RX_GROUP + rx;
                const uint32_t outBase = ((globalRx * NL + layer) * N_SYMBOL) * N_SC_PAD + sc0;
                const uint32_t ubBase = rx * N_SYMBOL * SC_TILE;
                DataCopy(outReG_[outBase], outAllR[ubBase], dp);
                DataCopy(outImG_[outBase], outAllI[ubBase], dp);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

#if CE_CUBE_TIME_POST_GEMM
    __aicore__ inline void LoadAAndPostTimeMatrix(uint32_t layer, uint32_t scTile)
    {
        const uint32_t aBase = layer * A_LAYER_ELEMS + scTile * A_TILE_ELEMS;
        DataCopy(aReL1_.Get<half>(), aReG_[aBase], A_TILE_ELEMS);
        DataCopy(aImL1_.Get<half>(), aImG_[aBase], A_TILE_ELEMS);
        DataCopy(aNegImL1_.Get<half>(), aNegImG_[aBase], A_TILE_ELEMS);
        DataCopy(timePostWeightL1_.Get<half>(),
                 wtReG_[layer * WT_LAYER_ELEMS + scTile * TIME_POST_P_ELEMS],
                 TIME_POST_P_ELEMS);
        Mte2ToMte1();

        auto b2 = b2Buf_.Get<half>();
        LoadData2DParams pA;
        pA.repeatTimes = N_RANK_TILE;
        pA.srcStride = 1;
        pA.ifTranspose = true;
        LoadData(b2, aReL1_.Get<half>(), pA);
        LoadData(b2[A_TILE_ELEMS], aImL1_.Get<half>(), pA);
        LoadData(b2[2 * A_TILE_ELEMS], aNegImL1_.Get<half>(), pA);

        LoadData2DParams pTime;
        pTime.repeatTimes = (TIME_POST_K / 16) * N_SYMBOL;
        pTime.srcStride = 1;
        pTime.ifTranspose = true;
        LoadData(b2[TIME_POST_P_L0_OFFSET], timePostWeightL1_.Get<half>(), pTime);
        Mte1ToM();
    }

    __aicore__ inline void LoadTimePostLeftOperand(const LocalTensor<half> &src)
    {
        auto a2 = a2Buf_.Get<half>();
        LoadData2DParams p;
        p.repeatTimes = TIME_POST_K / 16;
        p.srcStride = 1;
        p.ifTranspose = false;
        constexpr uint32_t mBlockElems = 16 * TIME_POST_K;
        for (uint32_t mBlock = 0; mBlock < TIME_POST_M / 16; ++mBlock) {
            const uint32_t offset = mBlock * mBlockElems;
            LoadData(a2[offset], src[offset], p);
        }
        Mte1ToM();
    }

    __aicore__ inline void StoreTimePostComponent(const LocalTensor<half> &src,
                                                  GlobalTensor<half> &dst,
                                                  uint32_t mBlockBase,
                                                  uint32_t firstLayer, uint32_t scTile)
    {
        DataCopyParams dp;
        dp.blockCount = TIME_POST_LAYER_BATCH * N_SYMBOL;
        dp.blockLen = 1;
        dp.srcStride = (TIME_POST_M / 16) * 16 - 1;
        dp.dstStride = (N_SC_PAD - SC_TILE) / 16;
        const uint32_t sc0 = scTile * SC_TILE;
        for (uint32_t rx = 0; rx < NR; ++rx) {
            const uint32_t outBase =
                ((rx * NL + firstLayer) * N_SYMBOL) * N_SC_PAD + sc0;
            const uint32_t srcBase = (mBlockBase + rx / 16) * 256 + (rx % 16) * 16;
            DataCopy(dst[outBase], src[srcBase], dp);
        }
    }

    __aicore__ inline void PackTimePostComponent(const LocalTensor<half> &srcUb,
                                                  uint32_t dstMBlockBase)
    {
        // Pack the [NR,32] row-major frequency result directly into 16x16
        // A-fractal source tiles in L1.  This removes a full UB Gather;
        // each MTE3 command copies one strided 16x16 tile.
        DataCopyParams pack;
        pack.blockCount = 16;
        pack.blockLen = 1;
        pack.srcStride = 1;
        pack.dstStride = 0;
        constexpr uint32_t kBlocks = TIME_POST_K / 16;
        for (uint32_t mBlock = 0; mBlock < NR / 16; ++mBlock) {
            for (uint32_t kBlock = 0; kBlock < kBlocks; ++kBlock) {
                const uint32_t srcOffset = mBlock * 16 * TIME_POST_K + kBlock * 16;
                const uint32_t dstOffset =
                    ((dstMBlockBase + mBlock) * kBlocks + kBlock) * TILE_ELEMS;
                DataCopy(timePostInputL1_.Get<half>()[dstOffset], srcUb[srcOffset], pack);
            }
        }
    }

    __aicore__ inline void RunTimePostComplex(uint32_t layerInBatch)
    {
        auto staging = timePostOutBuf_.Get<half>()[layerInBatch * TIME_POST_C_ELEMS];
        PackTimePostComponent(hfReBuf_.Get<half>(), 0);
        PackTimePostComponent(hfImBuf_.Get<half>(), NR / 16);
        Mte3ToMte1();
        LoadTimePostLeftOperand(timePostInputL1_.Get<half>());
        auto co = coBuf_.Get<float>();
        MmadParams mp;
        mp.m = TIME_POST_M;
        mp.n = TIME_POST_N;
        mp.k = TIME_POST_K;
        mp.cmatrixInitVal = true;
        Mmad(co, a2Buf_.Get<half>(),
             b2Buf_.Get<half>()[TIME_POST_P_L0_OFFSET], mp);
        MToMte1();
        PipeBarrier<PIPE_ALL>();

        DataCopyParams dcp;
        dcp.blockCount = (TIME_POST_M / 16) * (TIME_POST_N / 16);
        dcp.blockLen = 1;
        dcp.srcStride = 0;
        dcp.dstStride = 0;
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(staging, co, dcp, ep);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreTimePostBatch(uint32_t firstLayer, uint32_t scTile)
    {
        auto staging = timePostOutBuf_.Get<half>();
        StoreTimePostComponent(staging, outReG_, 0, firstLayer, scTile);
        StoreTimePostComponent(staging, outImG_, NR / 16, firstLayer, scTile);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PhaseBPostFrequencyCubeTime()
    {
        for (uint32_t firstLayer = 0; firstLayer < NL;
             firstLayer += TIME_POST_LAYER_BATCH) {
            const uint32_t tBase = firstLayer * T_LAYER_ELEMS;
            constexpr uint32_t tCount = TIME_POST_LAYER_BATCH * T_LAYER_ELEMS;
            DataCopy(tReL1_.Get<half>(), tReG_[tBase], tCount);
            DataCopy(tImL1_.Get<half>(), tImG_[tBase], tCount);
            Mte2ToMte1();
            for (uint32_t localTile = 0; localTile < SC_TILE_PER_CORE; ++localTile) {
                const uint32_t scTile = blockId_ * SC_TILE_PER_CORE + localTile;
                for (uint32_t layerInBatch = 0;
                     layerInBatch < TIME_POST_LAYER_BATCH; ++layerInBatch) {
                    LoadAAndPostTimeMatrix(firstLayer + layerInBatch, scTile);
                    GemmHfAllGroups(layerInBatch);
                    RunTimePostComplex(layerInBatch);
                }
                StoreTimePostBatch(firstLayer, scTile);
            }
        }
    }
#endif

#if CE_CUBE_TIME_FUSED
    __aicore__ inline void InitGatherIndex()
    {
        // Gather the DMRS0 row of every Rx from the interleaved
        // [group,rx,dmrs,sc] Cube result.  Passing source[SC_TILE] with the
        // same offsets selects the adjacent DMRS1 rows.
        auto index = gatherIdxBuf_.Get<uint32_t>();
        for (uint32_t rx = 0; rx < NR; ++rx) {
            const uint32_t group = rx / RX_GROUP;
            const uint32_t localRx = rx % RX_GROUP;
            const uint32_t cubeRow = group * NCOL + 2 * localRx;
            for (uint32_t sc = 0; sc < SC_TILE; ++sc) {
                index.SetValue(rx * SC_TILE + sc,
                    (cubeRow * SC_TILE + sc) * sizeof(half));
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadTCubeBatch(uint32_t firstLayer)
    {
        const uint32_t base = firstLayer * T_LAYER_ELEMS;
        constexpr uint32_t count = CUBE_TIME_LAYER_BATCH * T_LAYER_ELEMS;
        DataCopy(tReL1_.Get<half>(), tReG_[base], count);
        DataCopy(tImL1_.Get<half>(), tImG_[base], count);
        DataCopy(tNegImL1_.Get<half>(), tNegImG_[base], count);
        Mte2ToMte1();
    }

    __aicore__ inline void LoadCubeTimeFactors(uint32_t layer, uint32_t scTile,
                                               uint32_t symbol)
    {
        const uint32_t base = layer * A_LAYER_ELEMS +
                              (scTile * N_SYMBOL + symbol) * A_TILE_ELEMS;
        DataCopy(aReL1_.Get<half>(), aReG_[base], A_TILE_ELEMS);
        DataCopy(aImL1_.Get<half>(), aImG_[base], A_TILE_ELEMS);
        DataCopy(aNegImL1_.Get<half>(), aNegImG_[base], A_TILE_ELEMS);
        DataCopy(fused1ImL1_.Get<half>(), wtReG_[base], A_TILE_ELEMS);
        Mte2ToMte1();

        auto b2 = b2Buf_.Get<half>();
        LoadData2DParams p;
        p.repeatTimes = N_RANK_TILE;
        p.srcStride = 1;
        p.ifTranspose = true;
        LoadData(b2, aReL1_.Get<half>(), p);
        LoadData(b2[A_TILE_ELEMS], aImL1_.Get<half>(), p);
        LoadData(b2[2 * A_TILE_ELEMS], aNegImL1_.Get<half>(), p);
        LoadData(b2[3 * A_TILE_ELEMS], fused1ImL1_.Get<half>(), p);
        Mte1ToM();
    }

    __aicore__ inline void GemmCubeTime(uint32_t layerInBatch, uint32_t dmrs,
                                        const LocalTensor<half> &dstRe,
                                        const LocalTensor<half> &dstIm)
    {
        auto a2 = a2Buf_.Get<half>();
        auto b2 = b2Buf_.Get<half>();
        auto factorRe = b2[dmrs * 2 * A_TILE_ELEMS];
        auto factorIm = b2[(dmrs * 2 + 1) * A_TILE_ELEMS];
        auto co = coBuf_.Get<float>();
        constexpr uint32_t coImOffset = ALL_RX_TILE_ELEMS;
        MmadParams mp;
        mp.m = ALL_RX_COLS;
        mp.n = SC_TILE;
        mp.k = RANK;
        const uint32_t tBase = layerInBatch * T_LAYER_ELEMS;

        LoadTAllToL0A(tReL1_.Get<half>()[tBase]);
        mp.cmatrixInitVal = true;
        Mmad(co, a2, factorRe, mp);
        Mmad(co[coImOffset], a2, factorIm, mp);
        MToMte1();

        LoadTAllToL0A(tNegImL1_.Get<half>()[tBase]);
        mp.cmatrixInitVal = false;
        Mmad(co, a2, factorIm, mp);
        MToMte1();

        LoadTAllToL0A(tImL1_.Get<half>()[tBase]);
        Mmad(co[coImOffset], a2, factorRe, mp);
        MToMte1();
        PipeBarrier<PIPE_ALL>();

        DataCopyParams dcp;
        dcp.blockCount = N_RX_GROUP;
        dcp.blockLen = 1;
        dcp.srcStride = 0;
        dcp.dstStride = 0;
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(dstRe, co, dcp, ep);
        DataCopy(dstIm, co[coImOffset], dcp, ep);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CombineAndStoreCubeTime(uint32_t layer, uint32_t symbol,
                                                   uint32_t scTile)
    {
        auto d0Re = hfReBuf_.Get<half>();
        auto d0Im = hfImBuf_.Get<half>();
        auto d1Re = fused1ReBuf_.Get<half>();
        auto d1Im = fused1ImBuf_.Get<half>();
        auto outRe = cubeOutReBuf_.Get<half>();
        auto outIm = cubeOutImBuf_.Get<half>();
        auto tmpRe = cubeTmpReBuf_.Get<half>();
        auto tmpIm = cubeTmpImBuf_.Get<half>();
        auto index = gatherIdxBuf_.Get<uint32_t>();

        Gather(outRe, d0Re, index, static_cast<uint32_t>(0), CUBE_TIME_RX_ELEMS);
        Gather(outIm, d0Im, index, static_cast<uint32_t>(0), CUBE_TIME_RX_ELEMS);
        Gather(tmpRe, d1Re[SC_TILE], index, static_cast<uint32_t>(0), CUBE_TIME_RX_ELEMS);
        Gather(tmpIm, d1Im[SC_TILE], index, static_cast<uint32_t>(0), CUBE_TIME_RX_ELEMS);
        PipeBarrier<PIPE_V>();
        Add(outRe, outRe, tmpRe, CUBE_TIME_RX_ELEMS);
        Add(outIm, outIm, tmpIm, CUBE_TIME_RX_ELEMS);

        VToMte3();
        DataCopyParams dp;
        dp.blockCount = NR;
        dp.blockLen = 1;
        dp.srcStride = 0;
        dp.dstStride = (NL * N_SYMBOL * N_SC_PAD - SC_TILE) / 16;
        const uint32_t sc0 = scTile * SC_TILE;
        const uint32_t outBase = (layer * N_SYMBOL + symbol) * N_SC_PAD + sc0;
        DataCopy(outReG_[outBase], outRe, dp);
        DataCopy(outImG_[outBase], outIm, dp);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PhaseBCubeTimeFused()
    {
        for (uint32_t firstLayer = 0; firstLayer < NL; firstLayer += CUBE_TIME_LAYER_BATCH) {
            LoadTCubeBatch(firstLayer);
            for (uint32_t localTile = 0; localTile < SC_TILE_PER_CORE; ++localTile) {
                const uint32_t scTile = blockId_ * SC_TILE_PER_CORE + localTile;
                for (uint32_t symbol = 0; symbol < N_SYMBOL; ++symbol) {
                    for (uint32_t layerInBatch = 0;
                         layerInBatch < CUBE_TIME_LAYER_BATCH; ++layerInBatch) {
                        LoadCubeTimeFactors(firstLayer + layerInBatch, scTile, symbol);
                        GemmCubeTime(layerInBatch, 0,
                            hfReBuf_.Get<half>(), hfImBuf_.Get<half>());
                        GemmCubeTime(layerInBatch, 1,
                            fused1ReBuf_.Get<half>(), fused1ImBuf_.Get<half>());
                        CombineAndStoreCubeTime(firstLayer + layerInBatch,
                                                symbol, scTile);
                    }
                }
            }
        }
    }
#endif

    __aicore__ inline void PhaseBAndTime()
    {
        for (uint32_t firstLayer = 0; firstLayer < NL; firstLayer += CE_LAYER_BATCH) {
            LoadTBatch(firstLayer);
            for (uint32_t localTile = 0; localTile < SC_TILE_PER_CORE; ++localTile) {
                const uint32_t scTile = blockId_ * SC_TILE_PER_CORE + localTile;
                for (uint32_t layerInBatch = 0; layerInBatch < CE_LAYER_BATCH; ++layerInBatch) {
                    LoadAAndWt(firstLayer + layerInBatch, scTile);
#if CE_PHASE2_SKINNY
                    GemmHfAllGroups(layerInBatch);
                    for (uint32_t firstGroup = 0; firstGroup < N_RX_GROUP; firstGroup += CE_GROUP_BATCH) {
                        for (uint32_t slot = 0; slot < CE_GROUP_BATCH; ++slot) {
                            TimeToGroup((firstGroup + slot) * TILE_ELEMS, slot);
                        }
                        StoreGroupBatch(firstLayer + layerInBatch, scTile, firstGroup);
                    }
#else
                    for (uint32_t firstGroup = 0; firstGroup < N_RX_GROUP; firstGroup += CE_GROUP_BATCH) {
                        for (uint32_t slot = 0; slot < CE_GROUP_BATCH; ++slot) {
                            GemmHfGroup(layerInBatch, firstGroup + slot);
                            TimeToGroup(0, slot);
                        }
                        StoreGroupBatch(firstLayer + layerInBatch, scTile, firstGroup);
                    }
#endif
                }
            }
        }
    }

    TPipe *pipe_;
    uint32_t blockId_;
    event_t evMte2Mte1_, evMte1M_, evMMte1_, evVMte3_;
#if CE_CUBE_TIME_POST_GEMM
    event_t evMte3Mte1_;
#endif

    GlobalTensor<half> bReG_, bImG_, hlsReG_, hlsImG_, hlsNegImG_;
    GlobalTensor<half> aReG_, aImG_, aNegImG_, wtReG_, wtImG_;
    GlobalTensor<half> tReG_, tImG_, outReG_, outImG_;
#if CE_CUBE_TIME_FUSED
    GlobalTensor<half> tNegImG_;
#endif
    GlobalTensor<int32_t> syncG_;

    TBuf<TPosition::A1> factorReL1_, factorImL1_, tReL1_, tImL1_;
#if CE_CUBE_TIME_FUSED
    TBuf<TPosition::A1> tNegImL1_;
#elif CE_CUBE_TIME_POST_GEMM
    TBuf<TPosition::A1> timePostInputL1_;
#endif
    TBuf<TPosition::B1> hlsReL1_, hlsImL1_, hlsNegImL1_, aReL1_, aImL1_, aNegImL1_;
#if CE_CUBE_TIME_FUSED
    TBuf<TPosition::B1> fused1ImL1_;
#elif CE_CUBE_TIME_POST_GEMM
    TBuf<TPosition::B1> timePostWeightL1_;
#endif
    TBuf<TPosition::A2> a2Buf_;
    TBuf<TPosition::B2> b2Buf_;
    TBuf<TPosition::CO1> coBuf_;
    TBuf<TPosition::VECCALC> hfReBuf_, hfImBuf_, wtReBuf_;
#if CE_CUBE_TIME_FUSED
    TBuf<TPosition::VECCALC> fused1ReBuf_, fused1ImBuf_;
    TBuf<TPosition::VECCALC> cubeOutReBuf_, cubeOutImBuf_, cubeTmpReBuf_, cubeTmpImBuf_;
    TBuf<TPosition::VECCALC> gatherIdxBuf_;
#elif CE_CUBE_TIME_POST_GEMM
    TBuf<TPosition::VECCALC> timePostOutBuf_;
#endif
    TBuf<TPosition::VECCALC> outGroupRe0Buf_, outGroupRe1Buf_;
    TBuf<TPosition::VECCALC> outGroupRe2Buf_, outGroupRe3Buf_;
    TBuf<TPosition::VECCALC> outGroupIm0Buf_, outGroupIm1Buf_;
    TBuf<TPosition::VECCALC> outGroupIm2Buf_, outGroupIm3Buf_, syncBuf_;
};

extern "C" __global__ __aicore__ void channel_est_lmmse_kernel(
    GM_ADDR b_re, GM_ADDR b_im,
    GM_ADDR hls_re, GM_ADDR hls_im, GM_ADDR hls_neg_im,
    GM_ADDR a_re, GM_ADDR a_im, GM_ADDR a_neg_im, GM_ADDR wt_re, GM_ADDR wt_im,
    GM_ADDR t_re, GM_ADDR t_im,
    GM_ADDR out_re, GM_ADDR out_im,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    ChannelEstLmmse op;
    op.Init(b_re, b_im, hls_re, hls_im, hls_neg_im,
            a_re, a_im, a_neg_im, wt_re, wt_im, t_re, t_im,
            out_re, out_im, workspace, tiling, &pipe);
    op.Process();
}
