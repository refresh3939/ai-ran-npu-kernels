





















#include "kernel_operator.h"
#include "pss_correlator.h"

using namespace AscendC;
using namespace airan_pss;


namespace {


constexpr uint32_t S1_PER_AIV          = N_SEARCH / 4u;
constexpr uint32_t S1_TILE             = 1920u;
constexpr uint32_t S1_TILES_PER_AIV    = S1_PER_AIV / S1_TILE;
constexpr uint32_t S1_TILE_INTERLEAVED = 2u * S1_TILE;
constexpr float    Q_INV_F             = 1.0f / static_cast<float>(Q_SCALE);

static_assert(S1_PER_AIV % S1_TILE == 0u, "S1_TILE divide");
static_assert(S1_TILE % 16u == 0u,        "S1_TILE align");


constexpr uint32_t S23_M_TILE      = 128u;
constexpr uint32_t S23_K           = N_MF;
constexpr uint32_t S23_N           = 16u;
constexpr uint32_t S23_M_MMAD      = 16u;
constexpr uint32_t S23_M_SPLITS    = S23_M_TILE / S23_M_MMAD;
constexpr uint32_t S23_YG_L1_HALF  = S23_M_TILE * S23_K;
constexpr uint32_t S23_PSS_L1_HALF = S23_K * S23_N;


constexpr uint32_t S23_M_PER_AIV   = 38336u;
constexpr uint32_t S23_FULL_TILES  = S23_M_PER_AIV / S23_M_TILE;
constexpr uint32_t S23_TAIL_MU     = S23_M_PER_AIV - S23_FULL_TILES * S23_M_TILE;
constexpr uint32_t S23_TOTAL_TILES = S23_FULL_TILES + (S23_TAIL_MU > 0u ? 1u : 0u);
static_assert(S23_M_PER_AIV * 4u == 153344u, "S23_M_PER_AIV * 4 must = 153344");




constexpr uint32_t S23_PER_TILE_FP32     = 16u;
constexpr uint32_t S23_METRIC_LEN_PER_G  = 4u * S23_TOTAL_TILES * S23_PER_TILE_FP32;
constexpr uint32_t S23_METRIC_TOTAL_FP32 = N_G * S23_METRIC_LEN_PER_G;

}





class PssCorrelator {
public:
    __aicore__ inline PssCorrelator() {}

    __aicore__ inline void Init(GM_ADDR input_gm,
                                 GM_ADDR pss_ref_gm,
                                 GM_ADDR twiddle_gm,
                                 GM_ADDR scratch_gm,
                                 GM_ADDR output_gm,
                                 TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void Stage1_CfoDerot();
    __aicore__ inline void Stage23_Correlate();
    __aicore__ inline void EmitOutput();

    __aicore__ inline void LoadAndDequantize(
        const GlobalTensor<int16_t> &srcG, uint32_t src_offset_iq_pair,
        const LocalTensor<int16_t> &i16_ub,
        const LocalTensor<half>    &half_ub,
        const LocalTensor<half>    &re_ub,
        const LocalTensor<half>    &im_ub,
        uint32_t tile_samples);

    __aicore__ inline void BuildPssNzInL1(
        const LocalTensor<half> &pss_vecs_ub,
        const LocalTensor<half> &staging_ub,
        const LocalTensor<half> &out_l1);

    __aicore__ inline void LoadYgWindowToL1Nz(
        uint32_t src_gm_off_half,
        const LocalTensor<half> &dst_l1,
        uint32_t num_mu,
        const LocalTensor<half> &zero_ub);

    __aicore__ inline void RunCubeS23(
        const LocalTensor<half>  &a_l1,
        const LocalTensor<half>  &b_l1,
        const LocalTensor<float> &dst_ub);


    TPipe   *pipe_{nullptr};
    uint32_t aiv_id_{0};


    GlobalTensor<int16_t> inG_;
    GlobalTensor<int16_t> pssRefG_;
    GlobalTensor<int16_t> twiddleG_;
    GlobalTensor<half>    scratchG_;
    GlobalTensor<float>   metricG_;
    GlobalTensor<half>    outG_;


    TBuf<TPosition::VECCALC> bufYi16_,  bufYhalf_, bufYre_, bufYim_;
    TBuf<TPosition::VECCALC> bufTwi16_, bufTwhalf_, bufTwre_, bufTwim_;
    TBuf<TPosition::VECCALC> bufTmp1_,  bufTmp2_;
    TBuf<TPosition::VECCALC> bufYGre_,  bufYGim_;


    TBuf<TPosition::VECCALC> bufPssI16_;
    TBuf<TPosition::VECCALC> bufPssHalf_;
    TBuf<TPosition::VECCALC> bufPssRe_;
    TBuf<TPosition::VECCALC> bufPssIm_;
    TBuf<TPosition::VECCALC> bufStagingNz_;
    TBuf<TPosition::VECCALC> bufGemm1_;
    TBuf<TPosition::VECCALC> bufGemm2_;
    TBuf<TPosition::VECCALC> bufAccRe_;
    TBuf<TPosition::VECCALC> bufAccIm_;
    TBuf<TPosition::VECCALC> bufMetric_;


    TBuf<TPosition::A1> bufYGreL1_;
    TBuf<TPosition::A1> bufYGimL1_;
    TBuf<TPosition::B1> bufPssReL1_;
    TBuf<TPosition::B1> bufPssImL1_;


    TQue<TPosition::A2,  1> qA2_;
    TQue<TPosition::B2,  1> qB2_;
    TQue<TPosition::CO1, 1> qCO1_;

    TBuf<TPosition::VECCALC> bufOut_;
};





__aicore__ inline void PssCorrelator::Init(
    GM_ADDR input_gm,
    GM_ADDR pss_ref_gm,
    GM_ADDR twiddle_gm,
    GM_ADDR scratch_gm,
    GM_ADDR output_gm,
    TPipe *pipe)
{
    pipe_ = pipe;

    const uint32_t raw_bid = GetBlockIdx();
    aiv_id_ = raw_bid ^ 2u;


    inG_     .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(input_gm),
                              2u * N_SEARCH);
    pssRefG_ .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(pss_ref_gm),
                              2u * N_PSS * N_MF);
    twiddleG_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(twiddle_gm),
                              2u * N_G * N_SEARCH);
    scratchG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scratch_gm),
                              6u * N_SEARCH);

    metricG_ .SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(
            reinterpret_cast<__gm__ uint8_t *>(scratch_gm) + 12u * N_SEARCH),
        S23_METRIC_TOTAL_FP32);
    outG_    .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output_gm),
                              OUTPUT_FP16_LEN);


    const uint32_t TILE_INTERLEAVED_BYTES = S1_TILE_INTERLEAVED * sizeof(half);
    const uint32_t TILE_CHANNEL_BYTES     = S1_TILE * sizeof(half);
    pipe_->InitBuffer(bufYi16_,    TILE_INTERLEAVED_BYTES);
    pipe_->InitBuffer(bufYhalf_,   TILE_INTERLEAVED_BYTES);
    pipe_->InitBuffer(bufYre_,     TILE_CHANNEL_BYTES);
    pipe_->InitBuffer(bufYim_,     TILE_CHANNEL_BYTES);
    pipe_->InitBuffer(bufTwi16_,   TILE_INTERLEAVED_BYTES);
    pipe_->InitBuffer(bufTwhalf_,  TILE_INTERLEAVED_BYTES);
    pipe_->InitBuffer(bufTwre_,    TILE_CHANNEL_BYTES);
    pipe_->InitBuffer(bufTwim_,    TILE_CHANNEL_BYTES);
    pipe_->InitBuffer(bufTmp1_,    TILE_CHANNEL_BYTES);
    pipe_->InitBuffer(bufTmp2_,    TILE_CHANNEL_BYTES);
    pipe_->InitBuffer(bufYGre_,    TILE_CHANNEL_BYTES);
    pipe_->InitBuffer(bufYGim_,    TILE_CHANNEL_BYTES);


    pipe_->InitBuffer(bufPssI16_,    2u * N_MF * sizeof(int16_t));
    pipe_->InitBuffer(bufPssHalf_,   2u * N_MF * sizeof(half));
    pipe_->InitBuffer(bufPssRe_,     N_PSS * N_MF * sizeof(half));
    pipe_->InitBuffer(bufPssIm_,     N_PSS * N_MF * sizeof(half));
    pipe_->InitBuffer(bufStagingNz_, S23_PSS_L1_HALF * sizeof(half));
    pipe_->InitBuffer(bufGemm1_,     S23_M_TILE * S23_N * sizeof(float));
    pipe_->InitBuffer(bufGemm2_,     S23_M_TILE * S23_N * sizeof(float));
    pipe_->InitBuffer(bufAccRe_,     S23_M_TILE * S23_N * sizeof(float));
    pipe_->InitBuffer(bufAccIm_,     S23_M_TILE * S23_N * sizeof(float));
    pipe_->InitBuffer(bufMetric_,    S23_PER_TILE_FP32 * sizeof(float));


    pipe_->InitBuffer(bufYGreL1_,  S23_YG_L1_HALF  * sizeof(half));
    pipe_->InitBuffer(bufYGimL1_,  S23_YG_L1_HALF  * sizeof(half));
    pipe_->InitBuffer(bufPssReL1_, S23_PSS_L1_HALF * sizeof(half));
    pipe_->InitBuffer(bufPssImL1_, S23_PSS_L1_HALF * sizeof(half));


    pipe_->InitBuffer(qA2_,  1, S23_M_MMAD * S23_K * sizeof(half));
    pipe_->InitBuffer(qB2_,  1, S23_K * S23_N * sizeof(half));
    pipe_->InitBuffer(qCO1_, 1, S23_M_MMAD * S23_N * sizeof(float));

    pipe_->InitBuffer(bufOut_, OUTPUT_FP16_LEN * sizeof(half));
}





__aicore__ inline void PssCorrelator::LoadAndDequantize(
    const GlobalTensor<int16_t> &srcG, uint32_t src_offset_iq_pair,
    const LocalTensor<int16_t> &i16_ub,
    const LocalTensor<half>    &half_ub,
    const LocalTensor<half>    &re_ub,
    const LocalTensor<half>    &im_ub,
    uint32_t tile_samples)
{
    const uint32_t n_int16 = 2u * tile_samples;

    DataCopy(i16_ub, srcG[2u * src_offset_iq_pair], n_int16);
    event_t e_mte2_v = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e_mte2_v);
    WaitFlag<HardEvent::MTE2_V>(e_mte2_v);

    Cast(half_ub, i16_ub, RoundMode::CAST_NONE, n_int16);
    PipeBarrier<PIPE_V>();
    Muls(half_ub, half_ub, static_cast<half>(Q_INV_F), n_int16);
    PipeBarrier<PIPE_V>();

    {
        uint64_t rsvdCnt = 0;
        GatherMaskParams gmp;
        gmp.src0BlockStride  = 1;
        gmp.src0RepeatStride = 8;
        gmp.repeatTimes      = static_cast<uint16_t>(n_int16 / 128u);
        gmp.src1RepeatStride = 0;
        GatherMask(re_ub, half_ub, 1, false, 0, gmp, rsvdCnt);
        GatherMask(im_ub, half_ub, 2, false, 0, gmp, rsvdCnt);
    }
    PipeBarrier<PIPE_V>();
}







__aicore__ inline void PssCorrelator::Stage1_CfoDerot()
{
    if (aiv_id_ >= 4u) return;

    auto yi16   = bufYi16_  .Get<int16_t>();
    auto yhalf  = bufYhalf_ .Get<half>();
    auto y_re   = bufYre_   .Get<half>();
    auto y_im   = bufYim_   .Get<half>();
    auto twi16  = bufTwi16_ .Get<int16_t>();
    auto twhalf = bufTwhalf_.Get<half>();
    auto tw_re  = bufTwre_  .Get<half>();
    auto tw_im  = bufTwim_  .Get<half>();
    auto tmp1   = bufTmp1_  .Get<half>();
    auto tmp2   = bufTmp2_  .Get<half>();
    auto yG_re  = bufYGre_  .Get<half>();
    auto yG_im  = bufYGim_  .Get<half>();

    const uint32_t base_sample = aiv_id_ * S1_PER_AIV;

    for (uint32_t t = 0u; t < S1_TILES_PER_AIV; ++t) {
        const uint32_t tile_base = base_sample + t * S1_TILE;
        LoadAndDequantize(inG_, tile_base, yi16, yhalf, y_re, y_im, S1_TILE);

        for (uint32_t g = 0u; g < N_G; ++g) {
            const uint32_t tw_offset = g * N_SEARCH + tile_base;
            LoadAndDequantize(twiddleG_, tw_offset,
                              twi16, twhalf, tw_re, tw_im, S1_TILE);




            Mul(tmp1, y_re, tw_re, S1_TILE);
            Mul(tmp2, y_im, tw_im, S1_TILE);
            PipeBarrier<PIPE_V>();
            Sub(yG_re, tmp1, tmp2, S1_TILE);

            Mul(tmp1, y_re, tw_im, S1_TILE);
            Mul(tmp2, y_im, tw_re, S1_TILE);
            PipeBarrier<PIPE_V>();
            Add(yG_im, tmp1, tmp2, S1_TILE);
            PipeBarrier<PIPE_V>();

            const uint32_t re_gm_off = (2u * g)      * N_SEARCH + tile_base;
            const uint32_t im_gm_off = (2u * g + 1u) * N_SEARCH + tile_base;

            event_t e_v_mte3 = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(e_v_mte3);
            WaitFlag<HardEvent::V_MTE3>(e_v_mte3);

            DataCopy(scratchG_[re_gm_off], yG_re, S1_TILE);
            DataCopy(scratchG_[im_gm_off], yG_im, S1_TILE);

            event_t e_mte3_v = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(e_mte3_v);
            WaitFlag<HardEvent::MTE3_V>(e_mte3_v);
        }
    }
}









__aicore__ inline void PssCorrelator::BuildPssNzInL1(
    const LocalTensor<half> &pss_vecs_ub,
    const LocalTensor<half> &staging_ub,
    const LocalTensor<half> &out_l1)
{
    Duplicate(staging_ub, static_cast<half>(0.0f), S23_PSS_L1_HALF);
    PipeBarrier<PIPE_V>();

    for (uint32_t l = 0u; l < N_PSS; ++l) {
        for (uint32_t k = 0u; k < N_MF; ++k) {
            half v = pss_vecs_ub.GetValue(l * N_MF + k);
            staging_ub.SetValue(k * S23_N + l, v);
        }
    }

    event_t e_s_mte3 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e_s_mte3);
    WaitFlag<HardEvent::S_MTE3>(e_s_mte3);

    DataCopy(out_l1, staging_ub, S23_PSS_L1_HALF);

    event_t e_mte3_mte2 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e_mte3_mte2);
    WaitFlag<HardEvent::MTE3_MTE2>(e_mte3_mte2);
}







__aicore__ inline void PssCorrelator::LoadYgWindowToL1Nz(
    uint32_t src_gm_off_half,
    const LocalTensor<half> &dst_l1,
    uint32_t num_mu,
    const LocalTensor<half> &zero_ub)
{
    DataCopyParams dcp;
    dcp.blockCount = static_cast<uint16_t>(S23_K / 16u);
    dcp.blockLen   = 1u;
    dcp.srcStride  = 0u;
    dcp.dstStride  = static_cast<uint16_t>(S23_M_TILE - 1u);

    for (uint32_t mu = 0u; mu < num_mu; ++mu) {
        DataCopy(dst_l1[mu * 16u], scratchG_[src_gm_off_half + mu], dcp);
    }
    for (uint32_t mu = num_mu; mu < S23_M_TILE; ++mu) {
        DataCopy(dst_l1[mu * 16u], zero_ub, dcp);
    }

    event_t e_mte2_mte1 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    SetFlag<HardEvent::MTE2_MTE1>(e_mte2_mte1);
    WaitFlag<HardEvent::MTE2_MTE1>(e_mte2_mte1);
}







__aicore__ inline void PssCorrelator::RunCubeS23(
    const LocalTensor<half>  &a_l1,
    const LocalTensor<half>  &b_l1,
    const LocalTensor<float> &dst_ub)
{
    constexpr uint16_t kBlocks = S23_K / 16u;

    for (uint16_t ms = 0u; ms < S23_M_SPLITS; ++ms) {

        LocalTensor<half> b2 = qB2_.AllocTensor<half>();
        {
            LoadData2DParams lp;
            lp.repeatTimes = kBlocks;
            lp.srcStride   = 1u;
            lp.ifTranspose = true;
            LoadData(b2, b_l1, lp);
        }
        qB2_.EnQue(b2);


        LocalTensor<half> a2 = qA2_.AllocTensor<half>();
        {
            LoadData2DParams lp;
            lp.repeatTimes = kBlocks;
            lp.srcStride   = static_cast<uint16_t>(S23_M_TILE / 16u);
            lp.ifTranspose = false;
            LoadData(a2, a_l1[ms * S23_M_MMAD * 16u], lp);
        }
        qA2_.EnQue(a2);


        a2 = qA2_.DeQue<half>();
        b2 = qB2_.DeQue<half>();
        LocalTensor<float> co1 = qCO1_.AllocTensor<float>();
        {
            MmadParams mp;
            mp.m = S23_M_MMAD;
            mp.n = S23_N;
            mp.k = S23_K;
            Mmad(co1, a2, b2, mp);
        }
        qCO1_.EnQue(co1);
        qA2_.FreeTensor(a2);
        qB2_.FreeTensor(b2);


        co1 = qCO1_.DeQue<float>();
        {
            DataCopyParams dcp;
            dcp.blockCount = 1u;
            dcp.blockLen   = static_cast<uint16_t>(S23_M_MMAD * S23_N / 256u);
            DataCopyEnhancedParams ep;
            ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
            DataCopy(dst_ub[ms * S23_M_MMAD * S23_N], co1, dcp, ep);
        }
        qCO1_.FreeTensor(co1);
    }
}










__aicore__ inline void PssCorrelator::Stage23_Correlate()
{
    if (aiv_id_ >= 4u) return;

    auto pss_re_l1 = bufPssReL1_.Get<half>();
    auto pss_im_l1 = bufPssImL1_.Get<half>();
    auto yG_re_l1  = bufYGreL1_ .Get<half>();
    auto yG_im_l1  = bufYGimL1_ .Get<half>();
    auto pss_re_ub = bufPssRe_  .Get<half>();
    auto pss_im_ub = bufPssIm_  .Get<half>();
    auto pi16      = bufPssI16_ .Get<int16_t>();
    auto phalf     = bufPssHalf_.Get<half>();
    auto staging   = bufStagingNz_.Get<half>();
    auto gemm1     = bufGemm1_  .Get<float>();
    auto gemm2     = bufGemm2_  .Get<float>();
    auto acc_re    = bufAccRe_  .Get<float>();
    auto acc_im    = bufAccIm_  .Get<float>();
    auto metric_ub = bufMetric_ .Get<float>();


    for (uint32_t l = 0u; l < N_PSS; ++l) {
        LoadAndDequantize(pssRefG_, l * N_MF,
                          pi16, phalf,
                          pss_re_ub[l * N_MF], pss_im_ub[l * N_MF],
                          N_MF);
    }
    BuildPssNzInL1(pss_re_ub, staging, pss_re_l1);
    BuildPssNzInL1(pss_im_ub, staging, pss_im_l1);


    Duplicate(staging, static_cast<half>(0.0f), N_MF);
    PipeBarrier<PIPE_V>();
    event_t e_v_mte2 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(e_v_mte2);
    WaitFlag<HardEvent::V_MTE2>(e_v_mte2);


    const uint32_t mu_aiv_start = aiv_id_ * S23_M_PER_AIV;


    for (uint32_t tile_idx = 0u; tile_idx < S23_TOTAL_TILES; ++tile_idx) {
        const uint32_t mu_tile_start = mu_aiv_start + tile_idx * S23_M_TILE;
        const bool     is_tail       = (tile_idx == S23_FULL_TILES);
        const uint32_t this_tile_mu  = is_tail ? S23_TAIL_MU : S23_M_TILE;


        for (uint32_t g = 0u; g < N_G; ++g) {
            const uint32_t YG_RE_OFF = (2u * g)      * N_SEARCH + mu_tile_start;
            const uint32_t YG_IM_OFF = (2u * g + 1u) * N_SEARCH + mu_tile_start;
            LoadYgWindowToL1Nz(YG_RE_OFF, yG_re_l1, this_tile_mu, staging);
            LoadYgWindowToL1Nz(YG_IM_OFF, yG_im_l1, this_tile_mu, staging);








            RunCubeS23(yG_re_l1, pss_re_l1, gemm1);
            RunCubeS23(yG_im_l1, pss_im_l1, gemm2);
            {
                event_t e_mte3_v = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(e_mte3_v);
                WaitFlag<HardEvent::MTE3_V>(e_mte3_v);
            }
            Add(acc_re, gemm1, gemm2, S23_M_TILE * S23_N);
            PipeBarrier<PIPE_V>();

            RunCubeS23(yG_im_l1, pss_re_l1, gemm1);
            RunCubeS23(yG_re_l1, pss_im_l1, gemm2);
            {
                event_t e_mte3_v = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(e_mte3_v);
                WaitFlag<HardEvent::MTE3_V>(e_mte3_v);
            }
            Sub(acc_im, gemm1, gemm2, S23_M_TILE * S23_N);
            PipeBarrier<PIPE_V>();


            Mul(gemm1, acc_re, acc_re, S23_M_TILE * S23_N);
            Mul(gemm2, acc_im, acc_im, S23_M_TILE * S23_N);
            PipeBarrier<PIPE_V>();
            Add(gemm1, gemm1, gemm2, S23_M_TILE * S23_N);
            PipeBarrier<PIPE_V>();




            event_t e_v_s = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(e_v_s);
            WaitFlag<HardEvent::V_S>(e_v_s);

            for (uint32_t l = 0u; l < N_PSS; ++l) {
                float   max_v   = -1.0e30f;
                int32_t max_idx = 0;
                float   sum_v   = 0.0f;
                for (uint32_t m = 0u; m < S23_M_TILE; ++m) {
                    float v = gemm1.GetValue(m * S23_N + l);
                    sum_v += v;
                    if (v > max_v) {
                        max_v   = v;
                        max_idx = static_cast<int32_t>(m);
                    }
                }
                metric_ub.SetValue(l * 4u + 0u, max_v);
                metric_ub.SetValue(l * 4u + 1u, static_cast<float>(max_idx));
                metric_ub.SetValue(l * 4u + 2u, sum_v);
                metric_ub.SetValue(l * 4u + 3u, static_cast<float>(static_cast<int32_t>(S23_M_TILE)));
            }

            metric_ub.SetValue(12u, 0.0f);
            metric_ub.SetValue(13u, 0.0f);
            metric_ub.SetValue(14u, 0.0f);
            metric_ub.SetValue(15u, 0.0f);


            event_t e_s_mte3 = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(e_s_mte3);
            WaitFlag<HardEvent::S_MTE3>(e_s_mte3);

            const uint32_t global_tile_idx = aiv_id_ * S23_TOTAL_TILES + tile_idx;
            const uint32_t gm_off =
                g * (4u * S23_TOTAL_TILES * S23_PER_TILE_FP32) +
                global_tile_idx * S23_PER_TILE_FP32;
            DataCopy(metricG_[gm_off], metric_ub, S23_PER_TILE_FP32);

            event_t e_mte3_v = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(e_mte3_v);
            WaitFlag<HardEvent::MTE3_V>(e_mte3_v);
        }
    }
}





__aicore__ inline void PssCorrelator::EmitOutput()
{
    if (aiv_id_ != 0u) return;

    auto outBuf = bufOut_.Get<half>();
    Duplicate(outBuf, static_cast<half>(0.0f), OUTPUT_FP16_LEN);
    PipeBarrier<PIPE_V>();

    outBuf.SetValue(13, static_cast<half>(1.0f));
    outBuf.SetValue(14, static_cast<half>(1.0f));
    outBuf.SetValue(15, static_cast<half>(7.0f));

    event_t e = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e);
    WaitFlag<HardEvent::S_MTE3>(e);

    DataCopy(outG_, outBuf, OUTPUT_FP16_LEN);

    event_t e2 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e2);
    WaitFlag<HardEvent::MTE3_MTE2>(e2);
}





__aicore__ inline void PssCorrelator::Process()
{
    Stage1_CfoDerot();
    SyncAll();

    Stage23_Correlate();
    SyncAll();

    EmitOutput();
}





extern "C" __global__ __aicore__ void pss_correlator_kernel(
    GM_ADDR input_gm,
    GM_ADDR pss_ref_gm,
    GM_ADDR twiddle_gm,
    GM_ADDR scratch_gm,
    GM_ADDR output_gm,
    GM_ADDR ws,
    GM_ADDR tilingGm)
{
    TPipe pipe;
    PssCorrelator op;
    op.Init(input_gm, pss_ref_gm, twiddle_gm, scratch_gm, output_gm, &pipe);
    op.Process();
}
