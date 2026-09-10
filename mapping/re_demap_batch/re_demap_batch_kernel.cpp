/**
 * @file re_demap_batch_kernel.cpp
 * Runtime-batch RE demapper for Ascend 310P (dav_m200).
 *
 * Input planes use the exact [NR,14,32,64] stage-4 layout emitted by
 * ofdm_demod_batch. Output planes are [NR,14,1664] in natural used-SC order.
 * Every [1596,1664) tail is explicitly zeroed before it reaches channel
 * estimation or detector packing.
 */
#include "kernel_operator.h"
#include "re_demap_batch_kernel.h"

using namespace AscendC;
using namespace airan::re_demap_batch;

class ReDemapBatch {
public:
    __aicore__ inline ReDemapBatch() = default;

    __aicore__ inline void Init(GM_ADDR input_re, GM_ADDR input_im,
                                GM_ADDR gather_index, GM_ADDR output_re,
                                GM_ADDR output_im, uint32_t batch_size,
                                TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GatherOne(const GlobalTensor<half> &input,
                                     const GlobalTensor<half> &output,
                                     uint32_t batch, uint32_t symbol);

    TPipe *pipe_;
    uint32_t blockId_;
    uint32_t batchSize_;

    GlobalTensor<half> inputRe_;
    GlobalTensor<half> inputIm_;
    GlobalTensor<half> outputRe_;
    GlobalTensor<half> outputIm_;
    GlobalTensor<uint32_t> index_;

    TBuf<TPosition::VECCALC> sourceBuffer_;
    TBuf<TPosition::VECCALC> destinationBuffer_;
    TBuf<TPosition::VECCALC> indexBuffer_;
};

__aicore__ inline void ReDemapBatch::Init(
    GM_ADDR input_re, GM_ADDR input_im, GM_ADDR gather_index,
    GM_ADDR output_re, GM_ADDR output_im, uint32_t batch_size, TPipe *pipe)
{
    pipe_ = pipe;
    blockId_ = GetBlockIdx();
    batchSize_ = batch_size;

    inputRe_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input_re),
                             batchSize_ * GRID_IN_ELEMS);
    inputIm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input_im),
                             batchSize_ * GRID_IN_ELEMS);
    outputRe_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output_re),
                              batchSize_ * GRID_OUT_ELEMS);
    outputIm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output_im),
                              batchSize_ * GRID_OUT_ELEMS);
    index_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather_index),
                           N_SC_PAD);

    pipe_->InitBuffer(sourceBuffer_, N_FFT * sizeof(half));
    pipe_->InitBuffer(destinationBuffer_, N_SC_PAD * sizeof(half));
    pipe_->InitBuffer(indexBuffer_, N_SC_PAD * sizeof(uint32_t));
}

__aicore__ inline void ReDemapBatch::GatherOne(
    const GlobalTensor<half> &input, const GlobalTensor<half> &output,
    uint32_t batch, uint32_t symbol)
{
    auto source = sourceBuffer_.Get<half>();
    auto destination = destinationBuffer_.Get<half>();
    auto index = indexBuffer_.Get<uint32_t>();
    const uint32_t inputOffset = batch * GRID_IN_ELEMS + symbol * N_FFT;
    const uint32_t outputOffset = batch * GRID_OUT_ELEMS + symbol * N_SC_PAD;

    DataCopy(source, input[inputOffset], N_FFT);
    event_t copyIn = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(copyIn);
    WaitFlag<HardEvent::MTE2_V>(copyIn);

    // Keep the already validated 1664-element Gather path, then overwrite the
    // storage tail. This does not rely on the input element selected by the
    // padding entries in the shared offset table.
    Gather(destination, source, index, static_cast<uint32_t>(0), N_SC_PAD);
    Duplicate(destination[N_SC_USED], static_cast<half>(0), N_SC_PAD - N_SC_USED);

    event_t copyOut = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(copyOut);
    WaitFlag<HardEvent::V_MTE3>(copyOut);
    DataCopy(output[outputOffset], destination, N_SC_PAD);
    event_t reuse = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(reuse);
    WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}

__aicore__ inline void ReDemapBatch::Process()
{
    if (batchSize_ == 0) return;

    auto index = indexBuffer_.Get<uint32_t>();
    DataCopy(index, index_, N_SC_PAD);
    event_t indexReady = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(indexReady);
    WaitFlag<HardEvent::MTE2_V>(indexReady);

    const uint32_t totalTiles = batchSize_ * TILES_PER_BATCH;
    for (uint32_t tile = blockId_; tile < totalTiles; tile += BLOCK_DIM) {
        const uint32_t batch = tile / TILES_PER_BATCH;
        const uint32_t symbolStart =
            (tile % TILES_PER_BATCH) * SYMBOLS_PER_TILE;
        for (uint32_t localSymbol = 0; localSymbol < SYMBOLS_PER_TILE;
             ++localSymbol) {
            const uint32_t symbol = symbolStart + localSymbol;
            if (symbol >= N_SYMBOLS) break;
            GatherOne(inputRe_, outputRe_, batch, symbol);
            GatherOne(inputIm_, outputIm_, batch, symbol);
        }
    }
}

extern "C" __global__ __aicore__ void re_demap_batch_kernel(
    GM_ADDR input_re, GM_ADDR input_im, GM_ADDR gather_index,
    GM_ADDR output_re, GM_ADDR output_im, uint32_t batch_size,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    (void)tiling;
    TPipe pipe;
    ReDemapBatch operation;
    operation.Init(input_re, input_im, gather_index, output_re, output_im,
                   batch_size, &pipe);
    operation.Process();
}
