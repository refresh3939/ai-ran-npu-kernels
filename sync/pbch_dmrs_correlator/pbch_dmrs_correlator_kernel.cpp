





























#include "kernel_operator.h"
#include "pbch_dmrs_correlator.h"
#include "pbch_dmrs_tiling.h"

using namespace AscendC;





class PbchDmrsCorrelator {
public:
    __aicore__ inline PbchDmrsCorrelator() {}

    __aicore__ inline void Init(GM_ADDR r_re_gm,
                                 GM_ADDR r_im_gm,
                                 GM_ADDR d_re_gm,
                                 GM_ADDR d_im_gm,
                                 GM_ADDR d_im_neg_gm,
                                 GM_ADDR output_gm,
                                 const PbchDmrsTilingV1 &tiling,
                                 TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Stage1_CubeGemm();
    __aicore__ inline void Stage2_Metric();
    __aicore__ inline void Stage3_Argmax();
    __aicore__ inline void Stage4_Output();


    __aicore__ inline void RunCube(
        const GlobalTensor<half> &Agm,
        const GlobalTensor<half> &Bgm,
        const LocalTensor<float> &acc,
        bool initAcc);


    __aicore__ inline void CopyCo1ToUb(
        const LocalTensor<float> &dst_nd,
        const LocalTensor<float> &src_co1);


    TPipe   *pipe_;
    uint32_t blockId_;
    PbchDmrsTilingV1 tiling_;


    GlobalTensor<half>     rReG_;
    GlobalTensor<half>     rImG_;
    GlobalTensor<half>     dReG_;
    GlobalTensor<half>     dImG_;
    GlobalTensor<half>     dImNegG_;
    GlobalTensor<uint16_t> outG_;


    TBuf<TPosition::VECCALC> bufCorrReNd_;
    TBuf<TPosition::VECCALC> bufCorrImNd_;
    TBuf<TPosition::VECCALC> bufOut_;


    TQue<TPosition::A1, 2>  qA1_;
    TQue<TPosition::A2, 2>  qA2_;
    TQue<TPosition::B1, 2>  qB1_;
    TQue<TPosition::B2, 2>  qB2_;
    TQue<TPosition::CO1, 2> qCO1_;


    float dbg_r_re_0_;
    float dbg_d_re_00_;
    float best_v_;
    int32_t i_ssb_;
    float second_v_;
};





__aicore__ inline void PbchDmrsCorrelator::Init(
    GM_ADDR r_re_gm,
    GM_ADDR r_im_gm,
    GM_ADDR d_re_gm,
    GM_ADDR d_im_gm,
    GM_ADDR d_im_neg_gm,
    GM_ADDR output_gm,
    const PbchDmrsTilingV1 &tiling,
    TPipe *pipe)
{
    pipe_    = pipe;
    blockId_ = GetBlockIdx();
    tiling_  = tiling;

    rReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(r_re_gm),
                           static_cast<uint64_t>(M_MMAD) * K_SUB);
    rImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(r_im_gm),
                           static_cast<uint64_t>(M_MMAD) * K_SUB);
    dReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(d_re_gm),
                           static_cast<uint64_t>(K_SUB) * N_SUB);
    dImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(d_im_gm),
                           static_cast<uint64_t>(K_SUB) * N_SUB);
    dImNegG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(d_im_neg_gm),
                              static_cast<uint64_t>(K_SUB) * N_SUB);
    outG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(output_gm),
                           OUT_FP32_COUNT * 2u);


    pipe_->InitBuffer(bufCorrReNd_, M_MMAD * N_SUB * sizeof(float));
    pipe_->InitBuffer(bufCorrImNd_, M_MMAD * N_SUB * sizeof(float));
    pipe_->InitBuffer(bufOut_,      OUT_FP32_COUNT * sizeof(float));





    pipe_->InitBuffer(qA1_,  2, M_MMAD * K_SUB * sizeof(half));
    pipe_->InitBuffer(qA2_,  2, M_MMAD * K_SUB * sizeof(half));
    pipe_->InitBuffer(qB1_,  2, N_SUB  * K_SUB * sizeof(half));
    pipe_->InitBuffer(qB2_,  2, N_SUB  * K_SUB * sizeof(half));
    pipe_->InitBuffer(qCO1_, 2, M_MMAD * N_SUB * sizeof(float));

    dbg_r_re_0_  = 0.0f;
    dbg_d_re_00_ = 0.0f;
    best_v_   = 0.0f;
    i_ssb_    = 0;
    second_v_ = 0.0f;
}














__aicore__ inline void PbchDmrsCorrelator::RunCube(
    const GlobalTensor<half> &Agm,
    const GlobalTensor<half> &Bgm,
    const LocalTensor<float> &acc,
    bool initAcc)
{
    constexpr uint16_t kBlocks = K_SUB / 16;
    constexpr uint16_t mBlocks = M_MMAD / 16;
    constexpr uint16_t SRC_STRIDE_A = K_SUB / 16 - 1;
    constexpr uint16_t SRC_STRIDE_B = 0;


    LocalTensor<half> a1 = qA1_.AllocTensor<half>();
    for (uint16_t i = 0; i < kBlocks; ++i) {
        DataCopy(a1[i * 16 * M_MMAD], Agm[i * 16],
                 { M_MMAD, 1, SRC_STRIDE_A, 0 });
    }
    qA1_.EnQue(a1);


    LocalTensor<half> b1 = qB1_.AllocTensor<half>();
    for (uint16_t i = 0; i < kBlocks; ++i) {
        DataCopy(b1[i * 16 * N_SUB], Bgm[i * 16 * N_SUB],
                 { 16, 1, SRC_STRIDE_B, 0 });
    }
    qB1_.EnQue(b1);


    a1 = qA1_.DeQue<half>();
    LocalTensor<half> a2 = qA2_.AllocTensor<half>();
    {
        LoadData2DParams loadP;
        loadP.repeatTimes = kBlocks;
        loadP.srcStride   = mBlocks;
        loadP.ifTranspose = false;
        for (uint16_t i = 0; i < mBlocks; ++i) {
            LoadData(a2[i * kBlocks * 256], a1[i * 256], loadP);
        }
    }
    qA2_.EnQue(a2); qA1_.FreeTensor(a1);


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
    mp.m = M_MMAD;
    mp.n = N_SUB;
    mp.k = K_SUB;
    mp.cmatrixInitVal = initAcc;

    Mmad(acc, a2, b2, mp);

    qA2_.FreeTensor(a2);
    qB2_.FreeTensor(b2);
}






__aicore__ inline void PbchDmrsCorrelator::CopyCo1ToUb(
    const LocalTensor<float> &dst_nd,
    const LocalTensor<float> &src_co1)
{
    DataCopyParams dcp;
    dcp.blockCount = 1u;
    dcp.blockLen   = uint16_t(M_MMAD * N_SUB / 256);
    dcp.srcStride  = 0u;
    dcp.dstStride  = 0u;
    DataCopyEnhancedParams ep;
    ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(dst_nd, src_co1, dcp, ep);
}















__attribute__((noinline)) __aicore__ void PbchDmrsCorrelator::Stage1_CubeGemm()
{
    auto corrReNd = bufCorrReNd_.Get<float>();
    auto corrImNd = bufCorrImNd_.Get<float>();


    {
        LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
        RunCube(rReG_, dReG_, co1,  true);
        RunCube(rImG_, dImG_, co1,  false);
        qCO1_.EnQue(co1);
        co1 = qCO1_.DeQue<float>();
        CopyCo1ToUb(corrReNd, co1);
        qCO1_.FreeTensor(co1);
    }
    PipeBarrier<PIPE_ALL>();


    {
        LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
        RunCube(rImG_, dReG_,    co1,  true);
        RunCube(rReG_, dImNegG_, co1,  false);
        qCO1_.EnQue(co1);
        co1 = qCO1_.DeQue<float>();
        CopyCo1ToUb(corrImNd, co1);
        qCO1_.FreeTensor(co1);
    }
    PipeBarrier<PIPE_ALL>();



    dbg_r_re_0_  = static_cast<float>(rReG_.GetValue(0));
    dbg_d_re_00_ = static_cast<float>(dReG_.GetValue(0));
}













__attribute__((noinline)) __aicore__ void PbchDmrsCorrelator::Stage2_Metric()
{
    auto corrReNd = bufCorrReNd_.Get<float>();
    auto corrImNd = bufCorrImNd_.Get<float>();




    Mul(corrReNd, corrReNd, corrReNd, M_MMAD * N_SUB);
    PipeBarrier<PIPE_V>();
    Mul(corrImNd, corrImNd, corrImNd, M_MMAD * N_SUB);
    PipeBarrier<PIPE_V>();
    Add(corrReNd, corrReNd, corrImNd, M_MMAD * N_SUB);
    PipeBarrier<PIPE_V>();
}








__attribute__((noinline)) __aicore__ void PbchDmrsCorrelator::Stage3_Argmax()
{
    auto metricBuf = bufCorrReNd_.Get<float>();

    float best   = -1.0f;
    int32_t best_i = 0;
    const int32_t l_max = tiling_.l_max;

    for (int32_t i = 0; i < l_max; ++i) {
        float v = metricBuf.GetValue(static_cast<uint32_t>(i));
        if (v > best) { best = v; best_i = i; }
    }


    float second = -1.0f;
    for (int32_t i = 0; i < l_max; ++i) {
        if (i == best_i) continue;
        float v = metricBuf.GetValue(static_cast<uint32_t>(i));
        if (v > second) second = v;
    }
    if (l_max <= 1) second = 0.0f;

    best_v_   = best;
    i_ssb_    = best_i;
    second_v_ = (second < 0.0f) ? 0.0f : second;
}

















__attribute__((noinline)) __aicore__ void PbchDmrsCorrelator::Stage4_Output()
{
    auto outBuf = bufOut_.Get<float>();


    outBuf.SetValue(0u,  static_cast<float>(i_ssb_));
    outBuf.SetValue(1u,  best_v_);
    outBuf.SetValue(2u,  second_v_);
    outBuf.SetValue(3u,  static_cast<float>(tiling_.l_max));
    outBuf.SetValue(20u, dbg_r_re_0_);
    outBuf.SetValue(21u, dbg_d_re_00_);
    outBuf.SetValue(22u, best_v_);
    outBuf.SetValue(23u, SENTINEL_F32);
    PipeBarrier<PIPE_ALL>();


    GlobalTensor<float> outG_f32;
    outG_f32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        const_cast<__gm__ uint16_t *>(outG_.GetPhyAddr())), OUT_FP32_COUNT);

    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e); WaitFlag<HardEvent::S_MTE3>(e);

    DataCopy(outG_f32, outBuf, OUT_FP32_COUNT);

    event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e2); WaitFlag<HardEvent::MTE3_MTE2>(e2);
}








__aicore__ inline void PbchDmrsCorrelator::Process()
{
    Stage1_CubeGemm();
    PipeBarrier<PIPE_ALL>();



    {
        auto corrReNd = bufCorrReNd_.Get<float>();
        auto corrImNd = bufCorrImNd_.Get<float>();
        auto outBuf   = bufOut_.Get<float>();
        for (uint32_t i = 0u; i < 8u; ++i) {
            outBuf.SetValue(4u + i,  corrReNd.GetValue(i));
            outBuf.SetValue(12u + i, corrImNd.GetValue(i));
        }
        PipeBarrier<PIPE_ALL>();
    }

    Stage2_Metric();
    PipeBarrier<PIPE_ALL>();

    Stage3_Argmax();
    PipeBarrier<PIPE_ALL>();

    Stage4_Output();
    PipeBarrier<PIPE_ALL>();
}





extern "C" __global__ __aicore__ void pbch_dmrs_correlator_kernel(
    GM_ADDR r_re_gm,
    GM_ADDR r_im_gm,
    GM_ADDR d_re_gm,
    GM_ADDR d_im_gm,
    GM_ADDR d_im_neg_gm,
    GM_ADDR output_gm,
    GM_ADDR ws,
    GM_ADDR tiling_gm)
{
    if (GetBlockIdx() != 0u) return;
    (void)ws;


    PbchDmrsTilingV1 tiling;
    {
        const __gm__ uint32_t *src = reinterpret_cast<const __gm__ uint32_t *>(tiling_gm);
        uint32_t *dst = reinterpret_cast<uint32_t *>(&tiling);
        for (uint32_t i = 0; i < sizeof(PbchDmrsTilingV1) / sizeof(uint32_t); ++i) {
            dst[i] = src[i];
        }
    }

    TPipe pipe;
    PbchDmrsCorrelator op;
    op.Init(r_re_gm, r_im_gm, d_re_gm, d_im_gm, d_im_neg_gm, output_gm, tiling, &pipe);
    op.Process();
}
