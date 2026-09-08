
























#include "kernel_operator.h"
#include "sss_correlator.h"
#include "sss_tiling.h"

using namespace AscendC;

namespace {






constexpr uint32_t GM_REPEATS         = 5u;
constexpr uint32_t SEG_HALF_ALIGNED   = GM_REPEATS * 128u;
constexpr uint32_t IQ_SPLIT_HALF      = GM_REPEATS * 64u;


constexpr uint32_t RBATCH_HALF_PER_PLANE = N_TAU_PADDED * N_FFT;
constexpr uint32_t RBATCH_PLANE_BYTES    = RBATCH_HALF_PER_PLANE * 2u;


constexpr uint32_t CORR_FP32_PER_PLANE   = N_TAU_PADDED * N_ID_1_PADDED;
constexpr uint32_t CORR_PLANE_BYTES      = CORR_FP32_PER_PLANE * 4u;

}





class SssCorrelator {
public:
    __aicore__ inline SssCorrelator() {}

    __aicore__ inline void Init(GM_ADDR rx_gm,
                                 GM_ADDR sss_re_gm, GM_ADDR sss_im_gm,
                                 GM_ADDR twid_re_gm, GM_ADDR twid_im_gm,
                                 GM_ADDR scratch_gm,
                                 GM_ADDR output_gm,
                                 const SssTilingV1 &tiling,
                                 TPipe *pipe);
    __aicore__ inline void Process();

private:

    __aicore__ inline void Stage01_LoadAndDerot();
    __aicore__ inline void Stage2_BuildRBatch();
    __aicore__ inline void Stage3_CubeGemm(uint16_t ns_start, uint16_t ns_end);
    __aicore__ inline void Stage4_MetricSquare();
    __aicore__ inline void Stage5_LocalArgmax(uint16_t ns_start, uint16_t ns_end);
    __aicore__ inline void Stage6_GlobalReduceAndOutput();


    __aicore__ inline void RunCubeMnAcc(
        const LocalTensor<half> &A_L1,
        const GlobalTensor<half> &Bgm,
        const LocalTensor<float> &acc,
        bool initAcc);

    __aicore__ inline void NzToNd_FromCo1(
        const LocalTensor<float> &dst_nd,
        const LocalTensor<float> &src_co1,
        uint16_t n_offset);


    TPipe   *pipe_;
    uint32_t blockId_;
    SssTilingV1 tiling_;


    GlobalTensor<int16_t> rxG_;
    GlobalTensor<half>    sssReG_;
    GlobalTensor<half>    sssImG_;
    GlobalTensor<half>    twidReG_;
    GlobalTensor<half>    twidImG_;
    GlobalTensor<half>    rbatReScr_;
    GlobalTensor<half>    rbatImScr_;
    GlobalTensor<float>   corrReScr_;
    GlobalTensor<float>   corrImScr_;

    GlobalTensor<float>   localBestG_;
    GlobalTensor<float>   perJMaxG_;
    GlobalTensor<uint16_t> outG_;


    TBuf<TPosition::VECCALC> bufSegI16_, bufSegHalf_;
    TBuf<TPosition::VECCALC> bufRxRe_, bufRxIm_;
    TBuf<TPosition::VECCALC> bufTwRe_, bufTwIm_;
    TBuf<TPosition::VECCALC> bufDerotRe_, bufDerotIm_;
    TBuf<TPosition::VECCALC> bufRbatRe_, bufRbatIm_;
    TBuf<TPosition::VECCALC> bufCorrReFp32_, bufCorrImFp32_;
    TBuf<TPosition::VECCALC> bufMetric_;
    TBuf<TPosition::VECCALC> bufNdScratch_;


    TQue<TPosition::A1, 2>  qA1_;
    TQue<TPosition::A2, 2>  qA2_;
    TQue<TPosition::B1, 2>  qB1_;
    TQue<TPosition::B2, 2>  qB2_;
    TQue<TPosition::CO1, 2> qCO1_;


    TBuf<TPosition::A1> bufAReL1_;
    TBuf<TPosition::A1> bufAImL1_;


    float dbg_s0_;
    float dbg_s1_;
    float dbg_s2_;
    float dbg_s3_;
};





__aicore__ inline void SssCorrelator::Init(
    GM_ADDR rx_gm,
    GM_ADDR sss_re_gm, GM_ADDR sss_im_gm,
    GM_ADDR twid_re_gm, GM_ADDR twid_im_gm,
    GM_ADDR scratch_gm,
    GM_ADDR output_gm,
    const SssTilingV1 &tiling,
    TPipe *pipe)
{
    pipe_    = pipe;
    blockId_ = GetBlockIdx();
    tiling_  = tiling;


    rxG_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(rx_gm),
                          static_cast<uint64_t>(N_SEARCH) * 2ull);
    sssReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(sss_re_gm),
                             static_cast<uint64_t>(N_ID_2_COUNT) * N_ID_1_COUNT * N_FFT);
    sssImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(sss_im_gm),
                             static_cast<uint64_t>(N_ID_2_COUNT) * N_ID_1_COUNT * N_FFT);
    twidReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(twid_re_gm),
                              static_cast<uint64_t>(N_ID_2_COUNT) * SEG_LEN);
    twidImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(twid_im_gm),
                              static_cast<uint64_t>(N_ID_2_COUNT) * SEG_LEN);








    auto sbase = reinterpret_cast<__gm__ half *>(scratch_gm);
    rbatReScr_.SetGlobalBuffer(sbase + blockId_ * RBATCH_HALF_PER_PLANE,
                                RBATCH_HALF_PER_PLANE);
    rbatImScr_.SetGlobalBuffer(sbase + 4u * RBATCH_HALF_PER_PLANE
                                     + blockId_ * RBATCH_HALF_PER_PLANE,
                                RBATCH_HALF_PER_PLANE);
    auto sbase_f32 = reinterpret_cast<__gm__ float *>(
        scratch_gm + 4u * 2u * RBATCH_PLANE_BYTES);
    corrReScr_.SetGlobalBuffer(sbase_f32 + 0u, CORR_FP32_PER_PLANE);
    corrImScr_.SetGlobalBuffer(sbase_f32 + CORR_FP32_PER_PLANE, CORR_FP32_PER_PLANE);

    auto sbase_mc = reinterpret_cast<__gm__ float *>(
        scratch_gm + 4u * 2u * RBATCH_PLANE_BYTES + 2u * CORR_PLANE_BYTES);
    localBestG_.SetGlobalBuffer(sbase_mc, 32u);
    perJMaxG_.SetGlobalBuffer(sbase_mc + 32u, N_ID_1_COUNT);

    outG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(output_gm), 16u);


    pipe_->InitBuffer(bufSegI16_,    SEG_HALF_ALIGNED * sizeof(int16_t));
    pipe_->InitBuffer(bufSegHalf_,   SEG_HALF_ALIGNED * sizeof(half));

    pipe_->InitBuffer(bufRxRe_,      IQ_SPLIT_HALF * sizeof(half));
    pipe_->InitBuffer(bufRxIm_,      IQ_SPLIT_HALF * sizeof(half));
    pipe_->InitBuffer(bufTwRe_,      IQ_SPLIT_HALF * sizeof(half));
    pipe_->InitBuffer(bufTwIm_,      IQ_SPLIT_HALF * sizeof(half));
    pipe_->InitBuffer(bufDerotRe_,   IQ_SPLIT_HALF * sizeof(half));
    pipe_->InitBuffer(bufDerotIm_,   IQ_SPLIT_HALF * sizeof(half));
    pipe_->InitBuffer(bufRbatRe_,    RBATCH_HALF_PER_PLANE * sizeof(half));
    pipe_->InitBuffer(bufRbatIm_,    RBATCH_HALF_PER_PLANE * sizeof(half));
    pipe_->InitBuffer(bufCorrReFp32_, CORR_FP32_PER_PLANE * sizeof(float));
    pipe_->InitBuffer(bufCorrImFp32_, CORR_FP32_PER_PLANE * sizeof(float));
    pipe_->InitBuffer(bufMetric_,    CORR_FP32_PER_PLANE * sizeof(float));
    pipe_->InitBuffer(bufNdScratch_, M_MMAD * N_SUB * sizeof(float));


    pipe_->InitBuffer(qA1_,  2, M_MMAD * K_SUB * sizeof(half));
    pipe_->InitBuffer(qA2_,  2, M_MMAD * K_SUB * sizeof(half));
    pipe_->InitBuffer(qB1_,  2, N_SUB  * K_SUB * sizeof(half));
    pipe_->InitBuffer(qB2_,  2, N_SUB  * K_SUB * sizeof(half));
    pipe_->InitBuffer(qCO1_, 2, M_MMAD * N_SUB * sizeof(float));


    pipe_->InitBuffer(bufAReL1_, M_MMAD * N_FFT * sizeof(half));
    pipe_->InitBuffer(bufAImL1_, M_MMAD * N_FFT * sizeof(half));
}





__attribute__((noinline)) __aicore__ void SssCorrelator::Stage01_LoadAndDerot()
{
    auto segI16   = bufSegI16_.Get<int16_t>();
    auto segHalf  = bufSegHalf_.Get<half>();
    auto rxRe     = bufRxRe_.Get<half>();
    auto rxIm     = bufRxIm_.Get<half>();
    auto twRe     = bufTwRe_.Get<half>();
    auto twIm     = bufTwIm_.Get<half>();
    auto derotRe  = bufDerotRe_.Get<half>();
    auto derotIm  = bufDerotIm_.Get<half>();


    int32_t seg_start_i16 = tiling_.seg_start * 2;
    DataCopy(segI16, rxG_[seg_start_i16], SEG_HALF_ALIGNED);

    event_t e_mte2v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e_mte2v); WaitFlag<HardEvent::MTE2_V>(e_mte2v);




    Cast(segHalf, segI16, RoundMode::CAST_NONE, SEG_HALF_ALIGNED);
    PipeBarrier<PIPE_V>();


    {
        event_t e_v_s_a = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(e_v_s_a); WaitFlag<HardEvent::V_S>(e_v_s_a);
        dbg_s0_ = static_cast<float>(segHalf.GetValue(0u));
    }

    constexpr half QINV = static_cast<half>(1.0f / static_cast<float>(Q_SCALE));
    Muls(segHalf, segHalf, QINV, SEG_HALF_ALIGNED);
    PipeBarrier<PIPE_V>();


    {
        event_t e_v_s_b = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(e_v_s_b); WaitFlag<HardEvent::V_S>(e_v_s_b);
        dbg_s1_ = static_cast<float>(segHalf.GetValue(0u));
    }







    {
        uint64_t rsvdCnt = 0;
        GatherMaskParams gmp;
        gmp.src0BlockStride = 1u;
        gmp.repeatTimes     = static_cast<uint16_t>((SEG_HALF_ALIGNED + 127u) / 128u);
        gmp.src0RepeatStride = 8u;
        gmp.src1RepeatStride = 0u;
        GatherMask(rxRe, segHalf, (uint8_t)1, false, 0, gmp, rsvdCnt);
        GatherMask(rxIm, segHalf, (uint8_t)2, false, 0, gmp, rsvdCnt);
    }
    PipeBarrier<PIPE_V>();


    int32_t tw_start = tiling_.g_idx * static_cast<int32_t>(SEG_LEN);




    constexpr uint32_t SEG_HALF_LOAD = ((SEG_LEN + 15u) / 16u) * 16u;
    DataCopy(twRe, twidReG_[tw_start], SEG_HALF_LOAD);
    DataCopy(twIm, twidImG_[tw_start], SEG_HALF_LOAD);

    event_t e_mte2v2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e_mte2v2); WaitFlag<HardEvent::MTE2_V>(e_mte2v2);









    Mul(derotRe, rxRe, twRe, SEG_LEN);
    PipeBarrier<PIPE_V>();
    Mul(derotIm, rxIm, twIm, SEG_LEN);
    PipeBarrier<PIPE_V>();
    Sub(derotRe, derotRe, derotIm, SEG_LEN);
    PipeBarrier<PIPE_V>();

    Mul(derotIm, rxRe, twIm, SEG_LEN);
    PipeBarrier<PIPE_V>();

    Mul(rxRe, rxIm, twRe, SEG_LEN);
    PipeBarrier<PIPE_V>();
    Add(derotIm, derotIm, rxRe, SEG_LEN);
    PipeBarrier<PIPE_V>();
}






__attribute__((noinline)) __aicore__ void SssCorrelator::Stage2_BuildRBatch()
{
    auto derotRe = bufDerotRe_.Get<half>();
    auto derotIm = bufDerotIm_.Get<half>();
    auto rbatRe  = bufRbatRe_.Get<half>();
    auto rbatIm  = bufRbatIm_.Get<half>();


    Duplicate(rbatRe, static_cast<half>(0.0f), RBATCH_HALF_PER_PLANE);
    Duplicate(rbatIm, static_cast<half>(0.0f), RBATCH_HALF_PER_PLANE);
    PipeBarrier<PIPE_V>();


    Duplicate(rbatRe, static_cast<half>(0.0f), RBATCH_HALF_PER_PLANE);
    Duplicate(rbatIm, static_cast<half>(0.0f), RBATCH_HALF_PER_PLANE);
    PipeBarrier<PIPE_V>();


    for (uint32_t tau = 0u; tau < N_TAU; ++tau) {
        auto src_re = derotRe[tau];
        auto src_im = derotIm[tau];
        auto dst_re = rbatRe[tau * N_FFT];
        auto dst_im = rbatIm[tau * N_FFT];
        DataCopy(dst_re, src_re, N_FFT);
        DataCopy(dst_im, src_im, N_FFT);
    }
    event_t e_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e_v_mte3); WaitFlag<HardEvent::V_MTE3>(e_v_mte3);


    DataCopy(rbatReScr_, rbatRe, RBATCH_HALF_PER_PLANE);
    DataCopy(rbatImScr_, rbatIm, RBATCH_HALF_PER_PLANE);
    event_t e_mte3_mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e_mte3_mte2); WaitFlag<HardEvent::MTE3_MTE2>(e_mte3_mte2);


    event_t e_v_s2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e_v_s2); WaitFlag<HardEvent::V_S>(e_v_s2);
    dbg_s2_ = static_cast<float>(rbatRe.GetValue(4u * N_FFT));
}











__aicore__ inline void SssCorrelator::RunCubeMnAcc(
    const LocalTensor<half> &A_L1,
    const GlobalTensor<half> &Bgm,
    const LocalTensor<float> &acc,
    bool initAcc)
{
    constexpr uint16_t mBlocks = M_MMAD / 16;
    constexpr uint16_t kBlocks = K_SUB  / 16;


    constexpr uint16_t SRC_STRIDE_GM_B = uint16_t(N_ID_1_COUNT / 16 - 1);
    LocalTensor<half> b1 = qB1_.AllocTensor<half>();
    for (uint16_t i = 0; i < kBlocks; ++i) {
        DataCopy(b1[i * 16 * N_SUB], Bgm[i * 16 * N_ID_1_COUNT],
                 { 16, 1, SRC_STRIDE_GM_B, 0 });
    }
    qB1_.EnQue(b1);


    LocalTensor<half> a2 = qA2_.AllocTensor<half>();
    {
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks;
        loadP.srcStride   = mBlocks;
        loadP.ifTranspose = false;
        for (uint16_t i = 0; i < mBlocks; ++i) {
            LoadData(a2[i * kBlocks * 256], A_L1[i * 256], loadP);
        }
    }
    qA2_.EnQue(a2);


    b1 = qB1_.DeQue<half>();
    LocalTensor<half> b2 = qB2_.AllocTensor<half>();
    {
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks;
        loadP.srcStride   = 1;
        loadP.ifTranspose = true;
        LoadData(b2, b1, loadP);
    }
    qB2_.EnQue(b2); qB1_.FreeTensor(b1);


    a2 = qA2_.DeQue<half>();
    b2 = qB2_.DeQue<half>();

    MmadParams mp;
    mp.m = M_MMAD; mp.n = N_SUB; mp.k = K_SUB;
    mp.cmatrixInitVal = initAcc;

    Mmad(acc, a2, b2, mp);

    qA2_.FreeTensor(a2);
    qB2_.FreeTensor(b2);
}


__aicore__ inline void SssCorrelator::NzToNd_FromCo1(
    const LocalTensor<float> &dst_nd,
    const LocalTensor<float> &src_co1,
    uint16_t n_offset)
{


    DataCopyParams dcp;
    dcp.blockCount = 1;
    dcp.blockLen   = uint16_t(M_MMAD * N_SUB / 256);


    dcp.blockLen   = uint16_t(M_MMAD * N_SUB / 256);
    dcp.srcStride  = 0;
    dcp.dstStride  = 0;
    DataCopyEnhancedParams ep;
    ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;



    DataCopy(dst_nd, src_co1, dcp, ep);
}





























__attribute__((noinline)) __aicore__ void SssCorrelator::Stage3_CubeGemm(
    uint16_t ns_start, uint16_t ns_end)
{
    auto corrReFp32_ub = bufCorrReFp32_.Get<float>();
    auto corrImFp32_ub = bufCorrImFp32_.Get<float>();


    Duplicate(corrReFp32_ub, 0.0f, CORR_FP32_PER_PLANE);
    Duplicate(corrImFp32_ub, 0.0f, CORR_FP32_PER_PLANE);
    PipeBarrier<PIPE_V>();

    const int64_t n2_offset = static_cast<int64_t>(tiling_.n_id_2)
                            * static_cast<int64_t>(N_ID_1_COUNT)
                            * static_cast<int64_t>(N_FFT);


    auto ndScratch = bufNdScratch_.Get<float>();



    auto aReL1 = bufAReL1_.Get<half>();
    auto aImL1 = bufAImL1_.Get<half>();
    {
        constexpr uint16_t kBlocks_full = N_FFT / 16;
        constexpr uint16_t SRC_STRIDE_GM_A = uint16_t(N_FFT / 16 - 1);
        for (uint16_t i = 0; i < kBlocks_full; ++i) {
            DataCopy(aReL1[i * 16 * M_MMAD], rbatReScr_[i * 16],
                     { M_MMAD, 1, SRC_STRIDE_GM_A, 0 });
            DataCopy(aImL1[i * 16 * M_MMAD], rbatImScr_[i * 16],
                     { M_MMAD, 1, SRC_STRIDE_GM_A, 0 });
        }
        event_t e_mte2_mte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
        SetFlag<HardEvent::MTE2_MTE1>(e_mte2_mte1);
        WaitFlag<HardEvent::MTE2_MTE1>(e_mte2_mte1);
    }

    for (uint16_t ns = ns_start; ns < ns_end; ++ns) {
        uint32_t n_start = ns * N_SUB;
        if (n_start >= N_ID_1_COUNT) continue;


        {
            LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
            auto Bgm = sssReG_[n2_offset + n_start];
            RunCubeMnAcc(aReL1, Bgm, co1,  true);
            qCO1_.EnQue(co1);
            co1 = qCO1_.DeQue<float>();


            DataCopyParams dcp;
            dcp.blockCount = 1u;
            dcp.blockLen   = uint16_t(M_MMAD * N_SUB / 256);
            dcp.srcStride  = 0u;
            dcp.dstStride  = 0u;
            DataCopyEnhancedParams ep;
            ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
            DataCopy(ndScratch, co1, dcp, ep);
            qCO1_.FreeTensor(co1);

            {
                event_t e_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(e_mte3_v);
                WaitFlag<HardEvent::MTE3_V>(e_mte3_v);
            }


            {
                BinaryRepeatParams bp;
                bp.dstBlkStride = 1; bp.src0BlkStride = 1; bp.src1BlkStride = 1;
                bp.dstRepStride  = N_ID_1_PADDED / 8;
                bp.src0RepStride = N_ID_1_PADDED / 8;
                bp.src1RepStride = N_SUB / 8;
                Add(corrReFp32_ub[n_start], corrReFp32_ub[n_start], ndScratch,
                    N_SUB, M_MMAD, bp);
                PipeBarrier<PIPE_V>();
            }
        }


        {
            LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
            auto Bgm = sssImG_[n2_offset + n_start];
            RunCubeMnAcc(aImL1, Bgm, co1, true);
            qCO1_.EnQue(co1);
            co1 = qCO1_.DeQue<float>();

            DataCopyParams dcp;
            dcp.blockCount = 1u;
            dcp.blockLen   = uint16_t(M_MMAD * N_SUB / 256);
            dcp.srcStride  = 0u;
            dcp.dstStride  = 0u;
            DataCopyEnhancedParams ep;
            ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
            DataCopy(ndScratch, co1, dcp, ep);
            qCO1_.FreeTensor(co1);

            {
                event_t e_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(e_mte3_v);
                WaitFlag<HardEvent::MTE3_V>(e_mte3_v);
            }


            {
                BinaryRepeatParams bp;
                bp.dstBlkStride = 1; bp.src0BlkStride = 1; bp.src1BlkStride = 1;
                bp.dstRepStride  = N_ID_1_PADDED / 8;
                bp.src0RepStride = N_ID_1_PADDED / 8;
                bp.src1RepStride = N_SUB / 8;
                Add(corrReFp32_ub[n_start], corrReFp32_ub[n_start], ndScratch,
                    N_SUB, M_MMAD, bp);
                PipeBarrier<PIPE_V>();
            }
        }


        {
            LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
            auto Bgm = sssReG_[n2_offset + n_start];
            RunCubeMnAcc(aImL1, Bgm, co1, true);
            qCO1_.EnQue(co1);
            co1 = qCO1_.DeQue<float>();

            DataCopyParams dcp;
            dcp.blockCount = 1u;
            dcp.blockLen   = uint16_t(M_MMAD * N_SUB / 256);
            dcp.srcStride  = 0u;
            dcp.dstStride  = 0u;
            DataCopyEnhancedParams ep;
            ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
            DataCopy(ndScratch, co1, dcp, ep);
            qCO1_.FreeTensor(co1);

            {
                event_t e_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(e_mte3_v);
                WaitFlag<HardEvent::MTE3_V>(e_mte3_v);
            }


            {
                BinaryRepeatParams bp;
                bp.dstBlkStride = 1; bp.src0BlkStride = 1; bp.src1BlkStride = 1;
                bp.dstRepStride  = N_ID_1_PADDED / 8;
                bp.src0RepStride = N_ID_1_PADDED / 8;
                bp.src1RepStride = N_SUB / 8;
                Add(corrImFp32_ub[n_start], corrImFp32_ub[n_start], ndScratch,
                    N_SUB, M_MMAD, bp);
                PipeBarrier<PIPE_V>();
            }
        }


        {
            LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
            auto Bgm = sssImG_[n2_offset + n_start];
            RunCubeMnAcc(aReL1, Bgm, co1, true);
            qCO1_.EnQue(co1);
            co1 = qCO1_.DeQue<float>();

            DataCopyParams dcp;
            dcp.blockCount = 1u;
            dcp.blockLen   = uint16_t(M_MMAD * N_SUB / 256);
            dcp.srcStride  = 0u;
            dcp.dstStride  = 0u;
            DataCopyEnhancedParams ep;
            ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
            DataCopy(ndScratch, co1, dcp, ep);
            qCO1_.FreeTensor(co1);

            {
                event_t e_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(e_mte3_v);
                WaitFlag<HardEvent::MTE3_V>(e_mte3_v);
            }


            {
                BinaryRepeatParams bp;
                bp.dstBlkStride = 1; bp.src0BlkStride = 1; bp.src1BlkStride = 1;
                bp.dstRepStride  = N_ID_1_PADDED / 8;
                bp.src0RepStride = N_ID_1_PADDED / 8;
                bp.src1RepStride = N_SUB / 8;
                Sub(corrImFp32_ub[n_start], corrImFp32_ub[n_start], ndScratch,
                    N_SUB, M_MMAD, bp);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    PipeBarrier<PIPE_ALL>();


    dbg_s3_ = corrReFp32_ub.GetValue(4u * N_ID_1_PADDED + 0u);
}
__attribute__((noinline)) __aicore__ void SssCorrelator::Stage4_MetricSquare()
{
    auto corrRe = bufCorrReFp32_.Get<float>();
    auto corrIm = bufCorrImFp32_.Get<float>();
    auto metric = bufMetric_.Get<float>();



    Mul(metric, corrRe, corrRe, CORR_FP32_PER_PLANE);
    PipeBarrier<PIPE_V>();

    MulAddDst(metric, corrIm, corrIm, CORR_FP32_PER_PLANE);
    PipeBarrier<PIPE_V>();
}





__attribute__((noinline)) __aicore__ void SssCorrelator::Stage5_LocalArgmax(
    uint16_t ns_start, uint16_t ns_end)
{
    auto metric = bufMetric_.Get<float>();


    uint32_t j_lo = ns_start * N_SUB;
    uint32_t j_hi = ns_end   * N_SUB;
    if (j_hi > N_ID_1_COUNT) j_hi = N_ID_1_COUNT;
    if (j_lo > N_ID_1_COUNT) j_lo = N_ID_1_COUNT;


    auto per_j_max = bufRbatRe_.Get<float>();
    if (j_hi > j_lo) {
        Duplicate(per_j_max[j_lo], -1.0f, j_hi - j_lo);
        PipeBarrier<PIPE_V>();
    }
    event_t e_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e_v_s); WaitFlag<HardEvent::V_S>(e_v_s);


    float best_v = -1.0f;
    uint32_t best_idx = 0u;
    for (uint32_t tau = 0u; tau < N_TAU; ++tau) {
        for (uint32_t j = j_lo; j < j_hi; ++j) {
            uint32_t idx = tau * N_ID_1_PADDED + j;
            float v = metric.GetValue(idx);
            if (v > best_v) {
                best_v = v;
                best_idx = idx;
            }
            float pjm = per_j_max.GetValue(j);
            if (v > pjm) per_j_max.SetValue(j, v);
        }
    }
    int32_t best_tau = static_cast<int32_t>(best_idx / N_ID_1_PADDED);
    int32_t best_j   = static_cast<int32_t>(best_idx % N_ID_1_PADDED);


    LocalTensor<float> outBuf = bufNdScratch_.Get<float>();
    outBuf.SetValue(0u, best_v);
    outBuf.SetValue(1u, static_cast<float>(best_tau));
    outBuf.SetValue(2u, static_cast<float>(best_j));
    outBuf.SetValue(3u, (j_hi > j_lo) ? 1.0f : 0.0f);
    outBuf.SetValue(4u, 0.0f);
    outBuf.SetValue(5u, 0.0f);
    outBuf.SetValue(6u, 0.0f);
    outBuf.SetValue(7u, 0.0f);
    event_t e_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e_s_mte3); WaitFlag<HardEvent::S_MTE3>(e_s_mte3);
    DataCopy(localBestG_[blockId_ * 8u], outBuf, 8u);


    if (j_hi > j_lo) {
        event_t e_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e_v_mte3); WaitFlag<HardEvent::V_MTE3>(e_v_mte3);
        DataCopy(perJMaxG_[j_lo], per_j_max[j_lo], j_hi - j_lo);
    }
    event_t e_mte3_mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e_mte3_mte2); WaitFlag<HardEvent::MTE3_MTE2>(e_mte3_mte2);
}





__attribute__((noinline)) __aicore__ void SssCorrelator::Stage6_GlobalReduceAndOutput()
{

    auto buf = bufNdScratch_.Get<float>();
    DataCopy(buf, localBestG_, 32u);
    event_t e_mte2_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e_mte2_s); WaitFlag<HardEvent::MTE2_S>(e_mte2_s);


    float global_best_v = -1.0f;
    int32_t global_best_tau = 0, global_best_j = 0;
    for (uint32_t b = 0u; b < 4u; ++b) {
        float valid = buf.GetValue(b * 8u + 3u);
        if (valid < 0.5f) continue;
        float v = buf.GetValue(b * 8u + 0u);
        if (v > global_best_v) {
            global_best_v = v;
            global_best_tau = static_cast<int32_t>(buf.GetValue(b * 8u + 1u));
            global_best_j   = static_cast<int32_t>(buf.GetValue(b * 8u + 2u));
        }
    }
    int32_t tau_star = global_best_tau - static_cast<int32_t>(T_HALF);
    int32_t n_id_1   = global_best_j;


    auto pjmBuf = bufCorrReFp32_.Get<float>();
    DataCopy(pjmBuf, perJMaxG_, N_ID_1_COUNT);
    event_t e_mte2_s2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e_mte2_s2); WaitFlag<HardEvent::MTE2_S>(e_mte2_s2);

    float second = -1.0f;
    for (int32_t j = 0; j < static_cast<int32_t>(N_ID_1_COUNT); ++j) {
        if (j == global_best_j) continue;
        float pjm = pjmBuf.GetValue(static_cast<uint32_t>(j));
        if (pjm > second) second = pjm;
    }


    int32_t pcid = 3 * n_id_1 + static_cast<int32_t>(tiling_.n_id_2);
    buf.SetValue(0u, static_cast<float>(n_id_1));
    buf.SetValue(1u, static_cast<float>(pcid));
    buf.SetValue(2u, global_best_v);
    buf.SetValue(3u, second);
    buf.SetValue(4u, static_cast<float>(tau_star));
    buf.SetValue(5u, 0.0f);
    buf.SetValue(6u, 0.0f);
    buf.SetValue(7u, 7.0f);

    GlobalTensor<float> outG_f32;
    outG_f32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        const_cast<__gm__ uint16_t *>(outG_.GetPhyAddr())), 8u);
    event_t e_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e_s_mte3); WaitFlag<HardEvent::S_MTE3>(e_s_mte3);
    DataCopy(outG_f32, buf, 8u);
    event_t e_mte3_mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e_mte3_mte2); WaitFlag<HardEvent::MTE3_MTE2>(e_mte3_mte2);
}









__aicore__ inline void SssCorrelator::Process()
{



    uint16_t ns_start, ns_end;
    if (blockId_ == 0u) { ns_start = 0u;  ns_end = 6u;  }
    else if (blockId_ == 1u) { ns_start = 6u;  ns_end = 11u; }
    else if (blockId_ == 2u) { ns_start = 11u; ns_end = 16u; }
    else { ns_start = 16u; ns_end = 22u; }


    Stage01_LoadAndDerot();
    PipeBarrier<PIPE_ALL>();
    Stage2_BuildRBatch();
    PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll();


    Stage3_CubeGemm(ns_start, ns_end);
    PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll();

    Stage4_MetricSquare();
    PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll();

    Stage5_LocalArgmax(ns_start, ns_end);
    PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll();


    if (blockId_ == 0u) {
        Stage6_GlobalReduceAndOutput();
        PipeBarrier<PIPE_ALL>();
    }
}






extern "C" __global__ __aicore__ void sss_correlator_kernel(
    GM_ADDR rx_gm,
    GM_ADDR sss_re_gm,
    GM_ADDR sss_im_gm,
    GM_ADDR twid_re_gm,
    GM_ADDR twid_im_gm,
    GM_ADDR scratch_gm,
    GM_ADDR output_gm,
    GM_ADDR ws,
    GM_ADDR tiling_gm)
{

    (void)ws;


    SssTilingV1 tiling;
    {
        const __gm__ uint32_t *src = reinterpret_cast<const __gm__ uint32_t*>(tiling_gm);
        uint32_t *dst = reinterpret_cast<uint32_t*>(&tiling);
        for (uint32_t i = 0; i < sizeof(SssTilingV1) / sizeof(uint32_t); ++i) {
            dst[i] = src[i];
        }
    }

    TPipe pipe;
    SssCorrelator op;
    op.Init(rx_gm, sss_re_gm, sss_im_gm,
            twid_re_gm, twid_im_gm,
            scratch_gm, output_gm,
            tiling, &pipe);
    op.Process();
}