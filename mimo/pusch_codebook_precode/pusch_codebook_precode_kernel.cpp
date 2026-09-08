













#include "kernel_operator.h"
#include "pusch_codebook_precode_kernel.h"

using namespace AscendC;
using namespace airan::pusch_precode_kernel;

class PuschCodebookPrecode {
public:
    __aicore__ inline PuschCodebookPrecode() = default;

    __aicore__ inline void Init(GM_ADDR layer_re, GM_ADDR layer_im,
                                GM_ADDR weight_re, GM_ADDR weight_im,
                                GM_ADDR prg_of_rb, GM_ADDR port_re, GM_ADDR port_im,
                                GM_ADDR tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void PrecodeSymbol(uint32_t symbol);

    TPipe* pipe_ = nullptr;
    uint32_t core_id_ = 0;
    uint32_t num_layers_ = 0;
    uint32_t num_ports_ = 0;
    uint32_t num_prgs_ = 0;
    uint32_t weight_count_padded_ = 0;

    GlobalTensor<half> layerReG_, layerImG_;
    GlobalTensor<half> weightReG_, weightImG_;
    GlobalTensor<uint16_t> prgG_;
    GlobalTensor<half> portReG_, portImG_;
    GlobalTensor<uint32_t> tilingG_;

    TBuf<TPosition::VECCALC> layerReBuf_, layerImBuf_;
    TBuf<TPosition::VECCALC> outReBuf_, outImBuf_;
    TBuf<TPosition::VECCALC> negLayerImBuf_;
    TBuf<TPosition::VECCALC> weightReBuf_, weightImBuf_;
    TBuf<TPosition::VECCALC> prgBuf_, tilingBuf_;
};

__aicore__ inline void PuschCodebookPrecode::Init(
    GM_ADDR layer_re, GM_ADDR layer_im, GM_ADDR weight_re, GM_ADDR weight_im,
    GM_ADDR prg_of_rb, GM_ADDR port_re, GM_ADDR port_im, GM_ADDR tiling, TPipe* pipe) {
    pipe_ = pipe;
    core_id_ = GetBlockIdx();
    layerReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(layer_re), MAX_LAYERS * N_RE_GRID);
    layerImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(layer_im), MAX_LAYERS * N_RE_GRID);
    weightReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight_re), MAX_WEIGHT_ELEMS);
    weightImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight_im), MAX_WEIGHT_ELEMS);
    prgG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(prg_of_rb), PRG_MAP_PAD);
    portReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(port_re), MAX_PORTS * N_RE_GRID);
    portImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(port_im), MAX_PORTS * N_RE_GRID);
    tilingG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tiling), TILING_WORDS);

    pipe_->InitBuffer(layerReBuf_, N_SC_PAD * sizeof(half));
    pipe_->InitBuffer(layerImBuf_, N_SC_PAD * sizeof(half));
    pipe_->InitBuffer(outReBuf_, MAX_PORTS * N_SC_PAD * sizeof(half));
    pipe_->InitBuffer(outImBuf_, MAX_PORTS * N_SC_PAD * sizeof(half));
    pipe_->InitBuffer(negLayerImBuf_, N_SC_PAD * sizeof(half));
    pipe_->InitBuffer(weightReBuf_, MAX_WEIGHT_ELEMS * sizeof(half));
    pipe_->InitBuffer(weightImBuf_, MAX_WEIGHT_ELEMS * sizeof(half));
    pipe_->InitBuffer(prgBuf_, PRG_MAP_PAD * sizeof(uint16_t));
    pipe_->InitBuffer(tilingBuf_, TILING_WORDS * sizeof(uint32_t));
}

__aicore__ inline void PuschCodebookPrecode::PrecodeSymbol(uint32_t symbol) {
    auto layerRe = layerReBuf_.Get<half>();
    auto layerIm = layerImBuf_.Get<half>();
    auto outRe = outReBuf_.Get<half>();
    auto outIm = outImBuf_.Get<half>();
    auto negLayerIm = negLayerImBuf_.Get<half>();
    auto weightRe = weightReBuf_.Get<half>();
    auto weightIm = weightImBuf_.Get<half>();
    auto prg = prgBuf_.Get<uint16_t>();
    auto tiling = tilingBuf_.Get<uint32_t>();

    for (uint32_t port = 0; port < num_ports_; ++port) {
        Duplicate(outRe[port * N_SC_PAD], static_cast<half>(0.0f), N_SC_PAD);
        Duplicate(outIm[port * N_SC_PAD], static_cast<half>(0.0f), N_SC_PAD);
    }
    PipeBarrier<PIPE_V>();

    const uint32_t matrix_elems = num_ports_ * num_layers_;
    for (uint32_t layer = 0; layer < num_layers_; ++layer) {
        const uint32_t input_offset = (layer * N_SYMBOLS + symbol) * N_SC_PAD;
        DataCopy(layerRe, layerReG_[input_offset], N_SC_PAD);
        DataCopy(layerIm, layerImG_[input_offset], N_SC_PAD);
        auto load_event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(load_event);
        WaitFlag<HardEvent::MTE2_V>(load_event);
        Muls(negLayerIm, layerIm, static_cast<half>(-1.0f), N_SC_USED);
        PipeBarrier<PIPE_V>();

        uint32_t rb = 0;
        while (rb < N_RB) {
            const uint32_t group = static_cast<uint32_t>(prg.GetValue(rb));
            if (group >= num_prgs_) {
                ++rb;
                continue;
            }
            uint32_t rb_end = rb + 1;
            while (rb_end < N_RB && static_cast<uint32_t>(prg.GetValue(rb_end)) == group) {
                ++rb_end;
            }
            const uint32_t offset = rb * N_SC_PER_RB;
            const uint32_t count = (rb_end - rb) * N_SC_PER_RB;
            for (uint32_t port = 0; port < num_ports_; ++port) {
                const uint32_t weight_index =
                    group * matrix_elems + port * num_layers_ + layer;
                const half wr = weightRe.GetValue(weight_index);
                const half wi = weightIm.GetValue(weight_index);
                const uint32_t output_offset = port * N_SC_PAD + offset;
                const uint32_t matrix_index = port * num_layers_ + layer;
                const uint32_t weight_kind =
                    tiling.GetValue(TILING_WEIGHT_KIND_OFFSET + matrix_index);
                if (weight_kind == WEIGHT_KIND_REAL) {
                    Axpy(outRe[output_offset], layerRe[offset], wr, count);
                    Axpy(outIm[output_offset], layerIm[offset], wr, count);
                    PipeBarrier<PIPE_V>();
                } else if (weight_kind == WEIGHT_KIND_IMAG) {
                    Axpy(outRe[output_offset], negLayerIm[offset], wi, count);
                    Axpy(outIm[output_offset], layerRe[offset], wi, count);
                    PipeBarrier<PIPE_V>();
                }
            }
            rb = rb_end;
        }

        auto reuse_event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(reuse_event);
        WaitFlag<HardEvent::V_MTE2>(reuse_event);
    }


    auto store_event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(store_event);
    WaitFlag<HardEvent::V_MTE3>(store_event);
    for (uint32_t port = 0; port < num_ports_; ++port) {
        const uint32_t output_offset = (port * N_SYMBOLS + symbol) * N_SC_PAD;
        DataCopy(portReG_[output_offset], outRe[port * N_SC_PAD], N_SC_PAD);
        DataCopy(portImG_[output_offset], outIm[port * N_SC_PAD], N_SC_PAD);
    }

    auto next_event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(next_event);
    WaitFlag<HardEvent::MTE3_V>(next_event);
}

__aicore__ inline void PuschCodebookPrecode::Process() {
    auto tiling = tilingBuf_.Get<uint32_t>();
    DataCopy(tiling, tilingG_, TILING_WORDS);
    auto tiling_event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(tiling_event);
    WaitFlag<HardEvent::MTE2_S>(tiling_event);

    if (tiling.GetValue(0) != TILING_MAGIC || tiling.GetValue(1) != 1u) return;
    num_layers_ = tiling.GetValue(2);
    num_ports_ = tiling.GetValue(3);
    num_prgs_ = tiling.GetValue(4);
    weight_count_padded_ = tiling.GetValue(5);
    if (num_layers_ < 1 || num_layers_ > MAX_LAYERS ||
        num_ports_ < 1 || num_ports_ > MAX_PORTS || num_layers_ > num_ports_ ||
        num_prgs_ < 1 || num_prgs_ > MAX_PRGS ||
        weight_count_padded_ < num_prgs_ * num_ports_ * num_layers_ ||
        weight_count_padded_ > MAX_WEIGHT_ELEMS || (weight_count_padded_ & 15u) != 0u) return;

    auto weightRe = weightReBuf_.Get<half>();
    auto weightIm = weightImBuf_.Get<half>();
    auto prg = prgBuf_.Get<uint16_t>();
    DataCopy(weightRe, weightReG_, weight_count_padded_);
    DataCopy(weightIm, weightImG_, weight_count_padded_);
    DataCopy(prg, prgG_, PRG_MAP_PAD);
    auto resource_event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(resource_event);
    WaitFlag<HardEvent::MTE2_S>(resource_event);

    for (uint32_t symbol = core_id_; symbol < N_SYMBOLS; symbol += GetBlockNum()) {
        PrecodeSymbol(symbol);
    }
}

extern "C" __global__ __aicore__ void pusch_codebook_precode_kernel(
    GM_ADDR layer_re, GM_ADDR layer_im, GM_ADDR weight_re, GM_ADDR weight_im,
    GM_ADDR prg_of_rb, GM_ADDR port_re, GM_ADDR port_im,
    GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    TPipe pipe;
    PuschCodebookPrecode op;
    op.Init(layer_re, layer_im, weight_re, weight_im, prg_of_rb,
            port_re, port_im, tiling, &pipe);
    op.Process();
}
