






#include "kernel_operator.h"
#include "mimo_dmrs_ls.h"

using namespace AscendC;
using namespace airan::mimo_dmrs_ls;

namespace {
constexpr uint32_t RX_PER_CORE = NR_CURRENT / BLOCK_DIM;
constexpr uint32_t PIPELINE_DEPTH = 2;
constexpr uint32_t FULL_BUF = 1696;
constexpr uint32_t PILOT_WORK = N_DMRS_REF_PAD;
constexpr uint32_t GATHER_FULL_REPEATS = 13;
constexpr uint32_t GATHER_PAIR_REPEATS = 7;
constexpr uint32_t REDUCE_REPEATS = 7;
constexpr uint32_t REF_CACHE_ELEMS = MAX_LAYERS * CURRENT_DMRS_SYMBOLS * N_DMRS_REF_PAD;
}

class MimoDmrsLs {
public:
    __aicore__ inline MimoDmrsLs() {}
    __aicore__ inline void Init(GM_ADDR rx_re, GM_ADDR rx_im,
                                GM_ADDR ref_re, GM_ADDR ref_im,
                                GM_ADDR out_re, GM_ADDR out_im,
                                GM_ADDR pilot_sc, GM_ADDR pilot_count,
                                GM_ADDR noise_var, GM_ADDR tiling,
                                TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ExtractComb(LocalTensor<half> dst, LocalTensor<half> src,
                                        uint32_t delta);
    __aicore__ inline void ComplexMulConj(LocalTensor<half> hre, LocalTensor<half> him,
                                           LocalTensor<half> yre, LocalTensor<half> yim,
                                           LocalTensor<half> xre, LocalTensor<half> xim);
    __aicore__ inline void DespreadPairs(LocalTensor<half> dst,
                                          LocalTensor<half> src,
                                          LocalTensor<half> even,
                                          LocalTensor<half> odd);
    __aicore__ inline void AccumulateNoise(LocalTensor<half> hre,
                                            LocalTensor<half> him,
                                            uint32_t count,
                                            bool first);
    __aicore__ inline float ReduceNoise();
    __aicore__ inline void WritePilotDescription();

    TPipe *pipe_;
    uint32_t block_, nl_, ndmrs_, dmrs_symbol_[CURRENT_DMRS_SYMBOLS];
    uint32_t comb_[MAX_LAYERS], model_[MAX_LAYERS];
    bool useComb0_, useComb1_;

    GlobalTensor<half> rxReG_, rxImG_, refReG_, refImG_, outReG_, outImG_, noiseG_;
    GlobalTensor<int16_t> pilotScG_, pilotCountG_;
    GlobalTensor<uint32_t> metaG_;

    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> qOut_;
    TBuf<TPosition::VECCALC> bMeta_, bYreFull_, bYimFull_;
    TBuf<TPosition::VECCALC> bComb0Re_, bComb0Im_, bComb1Re_, bComb1Im_;
    TBuf<TPosition::VECCALC> bRefReCache_, bRefImCache_, bPairEven_, bPairOdd_;
    TBuf<TPosition::VECCALC> bRawRe_, bRawIm_;
    TBuf<TPosition::VECCALC> bTmp_, bStage_, bNoise_, bNoisePower_;
};

__aicore__ inline void MimoDmrsLs::Init(
    GM_ADDR rx_re, GM_ADDR rx_im, GM_ADDR ref_re, GM_ADDR ref_im,
    GM_ADDR out_re, GM_ADDR out_im, GM_ADDR pilot_sc, GM_ADDR pilot_count,
    GM_ADDR noise_var, GM_ADDR tiling, TPipe *pipe)
{
    pipe_ = pipe;
    block_ = GetBlockIdx();
    metaG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling), META_WORDS);

    pipe_->InitBuffer(bMeta_, META_WORDS * sizeof(uint32_t));


    pipe_->InitBuffer(bYreFull_, FULL_BUF * sizeof(half));
    pipe_->InitBuffer(bYimFull_, FULL_BUF * sizeof(half));
    pipe_->InitBuffer(bComb0Re_, PILOT_WORK * sizeof(half));
    pipe_->InitBuffer(bComb0Im_, PILOT_WORK * sizeof(half));
    pipe_->InitBuffer(bComb1Re_, PILOT_WORK * sizeof(half));
    pipe_->InitBuffer(bComb1Im_, PILOT_WORK * sizeof(half));
    pipe_->InitBuffer(bRefReCache_, REF_CACHE_ELEMS * sizeof(half));
    pipe_->InitBuffer(bRefImCache_, REF_CACHE_ELEMS * sizeof(half));
    pipe_->InitBuffer(bPairEven_, PILOT_WORK * sizeof(half));
    pipe_->InitBuffer(bPairOdd_, PILOT_WORK * sizeof(half));
    pipe_->InitBuffer(bRawRe_, PILOT_WORK * sizeof(half));
    pipe_->InitBuffer(bRawIm_, PILOT_WORK * sizeof(half));
    pipe_->InitBuffer(bTmp_, PILOT_WORK * sizeof(half));
    pipe_->InitBuffer(bStage_, 16 * sizeof(half));
    pipe_->InitBuffer(bNoise_, RX_PER_CORE * sizeof(half));
    pipe_->InitBuffer(bNoisePower_, PILOT_WORK * sizeof(half));

    auto meta = bMeta_.Get<uint32_t>();
    DataCopy(meta, metaG_, META_WORDS);
    auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(event); WaitFlag<HardEvent::MTE2_S>(event);
    nl_ = meta.GetValue(2);
    ndmrs_ = meta.GetValue(3);
    dmrs_symbol_[0] = meta.GetValue(10);
    dmrs_symbol_[1] = meta.GetValue(11);
    useComb0_ = false;
    useComb1_ = false;
    for (uint32_t l = 0; l < MAX_LAYERS; ++l) {
        comb_[l] = meta.GetValue(18 + l);
        model_[l] = meta.GetValue(22 + l);
        if (l < nl_ && comb_[l] == 0u) useComb0_ = true;
        if (l < nl_ && comb_[l] == 1u) useComb1_ = true;
    }

    const size_t rxElems = static_cast<size_t>(NR_CURRENT) * N_SYMBOLS * N_SC_PAD;
    const size_t refElems = static_cast<size_t>(nl_) * ndmrs_ * N_DMRS_REF_PAD;
    const size_t outElems = static_cast<size_t>(NR_CURRENT) * nl_ * ndmrs_ * N_PILOT_PAD;
    rxReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(rx_re), rxElems);
    rxImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(rx_im), rxElems);
    refReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ref_re), refElems);
    refImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ref_im), refElems);
    outReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(out_re), outElems);
    outImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(out_im), outElems);
    pilotScG_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(pilot_sc),
                              MAX_LAYERS * CURRENT_DMRS_SYMBOLS * N_PILOT_PAD);
    pilotCountG_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(pilot_count), COUNT_PAD);
    noiseG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(noise_var), NR_CURRENT);

}

__aicore__ inline void MimoDmrsLs::ExtractComb(LocalTensor<half> dst,
                                                LocalTensor<half> src,
                                                uint32_t delta)
{
    uint64_t ignored = 0;
    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = GATHER_FULL_REPEATS;
    params.src0RepeatStride = 8;
    params.src1RepeatStride = 0;
    const uint8_t laneMask = delta == 0u ? static_cast<uint8_t>(1) : static_cast<uint8_t>(2);
    GatherMask(dst, src, laneMask, false, 0, params, ignored);
}

__aicore__ inline void MimoDmrsLs::ComplexMulConj(
    LocalTensor<half> hre, LocalTensor<half> him,
    LocalTensor<half> yre, LocalTensor<half> yim,
    LocalTensor<half> xre, LocalTensor<half> xim)
{
    auto tmp = bTmp_.Get<half>();


    Mul(hre, yre, xre, N_PILOT_PAD);
    Mul(tmp, yim, xim, N_PILOT_PAD);
    Add(hre, hre, tmp, N_PILOT_PAD);
    Mul(him, yim, xre, N_PILOT_PAD);
    Mul(tmp, yre, xim, N_PILOT_PAD);
    Sub(him, him, tmp, N_PILOT_PAD);
}

__aicore__ inline void MimoDmrsLs::DespreadPairs(
    LocalTensor<half> dst, LocalTensor<half> src,
    LocalTensor<half> even, LocalTensor<half> odd)
{
    uint64_t ignored = 0;
    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = GATHER_PAIR_REPEATS;
    params.src0RepeatStride = 8;
    params.src1RepeatStride = 0;
    GatherMask(even, src, static_cast<uint8_t>(1), false, 0, params, ignored);
    GatherMask(odd, src, static_cast<uint8_t>(2), false, 0, params, ignored);
    Duplicate(dst, static_cast<half>(0.0f), PILOT_WORK);
    Add(dst, even, odd, N_OCC_PILOT);
    Muls(dst, dst, static_cast<half>(0.5f), N_OCC_PILOT);
}

__aicore__ inline void MimoDmrsLs::AccumulateNoise(LocalTensor<half> hre,
                                                    LocalTensor<half> him,
                                                    uint32_t count,
                                                    bool first)
{
    auto dre = bRawRe_.Get<half>();
    auto dim = bRawIm_.Get<half>();
    auto power = bTmp_.Get<half>();
    auto accumulated = bNoisePower_.Get<half>();
    const uint32_t n = count - 1;
    Sub(dre, hre[1], hre, n);
    Sub(dim, him[1], him, n);
    Mul(power, dre, dre, n);
    Mul(dre, dim, dim, n);
    Add(power, power, dre, n);
    if (first) Adds(accumulated, power, static_cast<half>(0.0f), n);
    else Add(accumulated, accumulated, power, n);
}

__aicore__ inline float MimoDmrsLs::ReduceNoise()
{
    auto accumulated = bNoisePower_.Get<half>();
    auto stage = bStage_.Get<half>();
    WholeReduceSum<half>(stage, accumulated, 128, REDUCE_REPEATS, 1, 1, 8);
    WholeReduceSum<half>(stage, stage, REDUCE_REPEATS, 1, 1, 1, 0);
    auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event); WaitFlag<HardEvent::V_S>(event);
    return static_cast<float>(stage.GetValue(0));
}

__aicore__ inline void MimoDmrsLs::WritePilotDescription()
{
    if (block_ != 0) return;
    auto seqHalf = bYreFull_.Get<half>();
    auto seqI16 = bYreFull_.Get<int16_t>();
    auto counts = bYimFull_.Get<int16_t>();
    Duplicate(counts, static_cast<int16_t>(0), COUNT_PAD);
    for (uint32_t layer = 0; layer < nl_; ++layer) {
        const uint32_t count = model_[layer] == FD_OCC2_399 ? N_OCC_PILOT : N_DMRS_RE;
        half start = static_cast<half>(0.0f);
        if (comb_[layer] == 1u) start = static_cast<half>(1.0f);
        if (model_[layer] == FD_OCC2_399) start = comb_[layer] == 1u ?
            static_cast<half>(2.0f) : static_cast<half>(1.0f);
        const half step = static_cast<half>(model_[layer] == FD_OCC2_399 ? 4.0f : 2.0f);
        for (uint32_t dmrs = 0; dmrs < ndmrs_; ++dmrs) {
            ArithProgression(seqHalf, start, step, N_PILOT_PAD);
            Cast(seqI16, seqHalf, RoundMode::CAST_RINT, N_PILOT_PAD);
            Duplicate(seqI16[count], static_cast<int16_t>(0), N_PILOT_PAD - count);
            auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(event); WaitFlag<HardEvent::V_MTE3>(event);
            DataCopy(pilotScG_[(layer * ndmrs_ + dmrs) * N_PILOT_PAD], seqI16, N_PILOT_PAD);
            auto back = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(back); WaitFlag<HardEvent::MTE3_V>(back);
            counts.SetValue(layer * ndmrs_ + dmrs, static_cast<int16_t>(count));
        }
    }
    auto scalarToMte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(scalarToMte); WaitFlag<HardEvent::S_MTE3>(scalarToMte);
    DataCopy(pilotCountG_, counts, COUNT_PAD);
    auto countDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(countDone); WaitFlag<HardEvent::MTE3_MTE2>(countDone);
}

__aicore__ inline void MimoDmrsLs::Process()
{
    if (block_ >= BLOCK_DIM || nl_ == 0 || nl_ > MAX_LAYERS ||
        ndmrs_ != CURRENT_DMRS_SYMBOLS) return;
    WritePilotDescription();

    auto comb0Re = bComb0Re_.Get<half>();
    auto comb0Im = bComb0Im_.Get<half>();
    auto comb1Re = bComb1Re_.Get<half>();
    auto comb1Im = bComb1Im_.Get<half>();
    auto refReCache = bRefReCache_.Get<half>();
    auto refImCache = bRefImCache_.Get<half>();
    auto pairEven = bPairEven_.Get<half>();
    auto pairOdd = bPairOdd_.Get<half>();
    auto rawRe = bRawRe_.Get<half>();
    auto rawIm = bRawIm_.Get<half>();
    auto noise = bNoise_.Get<half>();
    Duplicate(noise, static_cast<half>(0.0f), RX_PER_CORE);



    for (uint32_t layer = 0; layer < nl_; ++layer) {
        for (uint32_t dmrs = 0; dmrs < ndmrs_; ++dmrs) {
            const size_t refOffset = (static_cast<size_t>(layer) * ndmrs_ + dmrs) * N_DMRS_REF_PAD;
            DataCopy(refReCache[refOffset], refReG_[refOffset], N_DMRS_REF_PAD);
            DataCopy(refImCache[refOffset], refImG_[refOffset], N_DMRS_REF_PAD);
        }
    }
    auto refsReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(refsReady); WaitFlag<HardEvent::MTE2_V>(refsReady);



    pipe_->InitBuffer(qOut_, PIPELINE_DEPTH, 2 * PILOT_WORK * sizeof(half));
    const uint32_t rxStart = block_ * RX_PER_CORE;
    for (uint32_t localRx = 0; localRx < RX_PER_CORE; ++localRx) {
        const uint32_t rx = rxStart + localRx;
        Duplicate(bNoisePower_.Get<half>(), static_cast<half>(0.0f), PILOT_WORK);
        for (uint32_t dmrs = 0; dmrs < ndmrs_; ++dmrs) {
            auto yreFull = bYreFull_.Get<half>();
            auto yimFull = bYimFull_.Get<half>();
            const size_t gridOffset = (static_cast<size_t>(rx) * N_SYMBOLS +
                                       dmrs_symbol_[dmrs]) * N_SC_PAD;
            DataCopy(yreFull, rxReG_[gridOffset], N_SC_PAD);
            DataCopy(yimFull, rxImG_[gridOffset], N_SC_PAD);
            auto loadEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(loadEvent); WaitFlag<HardEvent::MTE2_V>(loadEvent);
            Duplicate(yreFull[N_SC_PAD], static_cast<half>(0.0f), FULL_BUF - N_SC_PAD);
            Duplicate(yimFull[N_SC_PAD], static_cast<half>(0.0f), FULL_BUF - N_SC_PAD);

            if (useComb0_) {
                ExtractComb(comb0Re, yreFull, 0);
                ExtractComb(comb0Im, yimFull, 0);
            }
            if (useComb1_) {
                ExtractComb(comb1Re, yreFull, 1);
                ExtractComb(comb1Im, yimFull, 1);
            }

            for (uint32_t layer = 0; layer < nl_; ++layer) {
                auto packedOut = qOut_.AllocTensor<half>();
                auto outRe = packedOut;
                auto outIm = packedOut[PILOT_WORK];
                auto yre = comb_[layer] == 0u ? comb0Re : comb1Re;
                auto yim = comb_[layer] == 0u ? comb0Im : comb1Im;
                const size_t refOffset = (static_cast<size_t>(layer) * ndmrs_ + dmrs) * N_DMRS_REF_PAD;
                auto xre = refReCache[refOffset];
                auto xim = refImCache[refOffset];
                ComplexMulConj(rawRe, rawIm, yre, yim, xre, xim);

                uint32_t count = N_DMRS_RE;
                if (model_[layer] == FD_OCC2_399) {
                    DespreadPairs(outRe, rawRe, pairEven, pairOdd);
                    DespreadPairs(outIm, rawIm, pairEven, pairOdd);
                    count = N_OCC_PILOT;
                } else {
                    Adds(outRe, rawRe, static_cast<half>(0.0f), N_PILOT_PAD);
                    Adds(outIm, rawIm, static_cast<half>(0.0f), N_PILOT_PAD);
                }

                if (layer == 0) AccumulateNoise(outRe, outIm, count, dmrs == 0u);


                const size_t outOffset = ((static_cast<size_t>(rx) * nl_ + layer) *
                                          ndmrs_ + dmrs) * N_PILOT_PAD;
                qOut_.EnQue(packedOut);
                auto readyOut = qOut_.DeQue<half>();
                DataCopy(outReG_[outOffset], readyOut, N_PILOT_PAD);
                DataCopy(outImG_[outOffset], readyOut[PILOT_WORK], N_PILOT_PAD);
                qOut_.FreeTensor(readyOut);
            }
        }
        const float noiseSum = ReduceNoise();
        const float noiseValue = model_[0] == FD_OCC2_399 ?
            (noiseSum / 796.0f) : (noiseSum / 3188.0f);
        noise.SetValue(localRx, static_cast<half>(noiseValue));
    }
    auto noiseEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(noiseEvent); WaitFlag<HardEvent::S_MTE3>(noiseEvent);
    DataCopy(noiseG_[rxStart], noise, RX_PER_CORE);
}

extern "C" __global__ __aicore__ void mimo_dmrs_ls_kernel(
    GM_ADDR rx_re, GM_ADDR rx_im, GM_ADDR ref_re, GM_ADDR ref_im,
    GM_ADDR out_re, GM_ADDR out_im, GM_ADDR pilot_sc, GM_ADDR pilot_count,
    GM_ADDR noise_var, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    TPipe pipe;
    MimoDmrsLs op;
    op.Init(rx_re, rx_im, ref_re, ref_im, out_re, out_im,
            pilot_sc, pilot_count, noise_var, tiling, &pipe);
    op.Process();
}
