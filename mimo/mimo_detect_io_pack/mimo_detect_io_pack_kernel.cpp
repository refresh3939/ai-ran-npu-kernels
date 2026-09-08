








#include "kernel_operator.h"
#include "mimo_detect_io_pack.h"

using namespace AscendC;
namespace pack = airan::mimo_detect_io_pack;

namespace {

constexpr uint32_t HALF_PER_BLOCK = 16;
constexpr uint32_t GRID_ROW_BLOCKS = pack::N_RE / HALF_PER_BLOCK;
constexpr uint32_t PACKED_ROW_BLOCKS = pack::NR * pack::NL / HALF_PER_BLOCK;
constexpr uint32_t H_TILE_ELEMS = pack::RX_TILE * pack::NL * pack::RE_TILE;
constexpr uint32_t RX_TILE_ELEMS = pack::RX_GROUP * pack::RE_TILE;
constexpr uint32_t Y_TILE_ELEMS = pack::RE_TILE * pack::RX_GROUP * pack::NL;
constexpr uint32_t META_WORDS = pack::TILING_BYTES / sizeof(uint32_t);
constexpr size_t UB_WORKING_BYTES =
    2 * H_TILE_ELEMS * sizeof(half) +
    2 * RX_TILE_ELEMS * sizeof(half) +
    Y_TILE_ELEMS * sizeof(half) +
    pack::NOISE_ELEMS * sizeof(half) +
    pack::NOISE_ELEMS * sizeof(float) +
    pack::RE_PER_CORE * sizeof(half) +
    pack::TILING_BYTES;
static_assert(UB_WORKING_BYTES <= 128 * 1024,
              "mimo_detect_io_pack UB working set exceeds dav_m200 capacity");

class MimoDetectIoPack {
public:
    __aicore__ inline MimoDetectIoPack() {}

    __aicore__ inline void Init(
        GM_ADDR rx_re_gm, GM_ADDR rx_im_gm,
        GM_ADDR h_re_gm, GM_ADDR h_im_gm, GM_ADDR noise_gm,
        GM_ADDR hrm_re_gm, GM_ADDR hrm_im_gm,
        GM_ADDR yvpad_re_gm, GM_ADDR yvpad_im_gm, GM_ADDR no_gm,
        GM_ADDR tiling_gm,
        TPipe *pipe)
    {
        pipe_ = pipe;
        core_ = GetBlockIdx();
        re_begin_ = core_ * pack::RE_PER_CORE;

        rx_re_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(rx_re_gm), pack::RX_ELEMS);
        rx_im_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(rx_im_gm), pack::RX_ELEMS);
        h_re_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h_re_gm), pack::H_ELEMS);
        h_im_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h_im_gm), pack::H_ELEMS);
        noise_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(noise_gm), pack::NOISE_ELEMS);
        hrm_re_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(hrm_re_gm), pack::PACKED_ELEMS);
        hrm_im_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(hrm_im_gm), pack::PACKED_ELEMS);
        yvpad_re_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(yvpad_re_gm), pack::PACKED_ELEMS);
        yvpad_im_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(yvpad_im_gm), pack::PACKED_ELEMS);
        no_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(no_gm), pack::NO_ELEMS);
        metadata_g_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling_gm), META_WORDS);

        pipe_->InitBuffer(h_in_buf_, H_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(h_out_buf_, H_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(rx_in_buf_, RX_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(rx_trans_buf_, RX_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(y_out_buf_, Y_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(noise_half_buf_, pack::NOISE_ELEMS * sizeof(half));
        pipe_->InitBuffer(noise_float_buf_, pack::NOISE_ELEMS * sizeof(float));
        pipe_->InitBuffer(no_out_buf_, pack::RE_PER_CORE * sizeof(half));
        pipe_->InitBuffer(metadata_buf_, pack::TILING_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (core_ >= pack::BLOCK_DIM) return;
        if (!LoadAndValidateMetadata()) return;
        PackNoise();
        for (uint32_t local_re = 0; local_re < pack::RE_PER_CORE;
             local_re += pack::RE_TILE) {
            const uint32_t re = re_begin_ + local_re;
            PackHPlane(h_re_g_, hrm_re_g_, re);
            PackHPlane(h_im_g_, hrm_im_g_, re);
            PackYPlane(rx_re_g_, yvpad_re_g_, re);
            PackYPlane(rx_im_g_, yvpad_im_g_, re);
        }
    }

private:
    __aicore__ inline bool LoadAndValidateMetadata()
    {
        auto metadata = metadata_buf_.Get<uint32_t>();
        DataCopy(metadata, metadata_g_, META_WORDS);
        auto metadata_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(metadata_event);
        WaitFlag<HardEvent::MTE2_S>(metadata_event);

        active_layers_ = metadata.GetValue(1);
        return metadata.GetValue(0) == pack::META_MAGIC &&
               active_layers_ >= 1 && active_layers_ <= pack::NL &&
               metadata.GetValue(2) == pack::NR &&
               metadata.GetValue(3) == pack::NL &&
               metadata.GetValue(4) == pack::N_RE &&
               metadata.GetValue(5) == pack::ABI_VERSION &&
               metadata.GetValue(6) == 0 && metadata.GetValue(7) == 0;
    }

    __aicore__ inline void PackNoise()
    {
        auto noise_h = noise_half_buf_.Get<half>();
        auto noise_f = noise_float_buf_.Get<float>();
        auto no_out = no_out_buf_.Get<half>();
        DataCopy(noise_h, noise_g_, pack::NOISE_ELEMS);
        auto load_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(load_event);
        WaitFlag<HardEvent::MTE2_V>(load_event);
        Cast(noise_f, noise_h, RoundMode::CAST_NONE, pack::NOISE_ELEMS);
        auto scalar_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(scalar_event);
        WaitFlag<HardEvent::V_S>(scalar_event);
        float sum = 0.0f;
        for (uint32_t rx = 0; rx < pack::NR; ++rx) sum += noise_f.GetValue(rx);
        const half mean = static_cast<half>(sum * pack::NOISE_MEAN_SCALE);
        Duplicate(no_out, mean, pack::RE_PER_CORE);
        auto store_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(store_event);
        WaitFlag<HardEvent::V_MTE3>(store_event);
        DataCopy(no_g_[re_begin_], no_out, pack::RE_PER_CORE);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PackHPlane(
        const GlobalTensor<half> &input, GlobalTensor<half> &output, uint32_t re)
    {
        auto h_in = h_in_buf_.Get<half>();
        auto h_out = h_out_buf_.Get<half>();

        DataCopyParams load;
        load.blockCount = pack::RX_TILE * pack::NL;
        load.blockLen = 1;
        load.srcStride = GRID_ROW_BLOCKS - 1;
        load.dstStride = 0;
        DataCopyParams store;
        store.blockCount = pack::RE_TILE;
        store.blockLen = 1;
        store.srcStride = 0;
        store.dstStride = PACKED_ROW_BLOCKS - 1;

        for (uint32_t rx_base = 0; rx_base < pack::NR; rx_base += pack::RX_TILE) {
            const size_t input_offset =
                (static_cast<size_t>(rx_base) * pack::NL) * pack::N_RE + re;
            DataCopy(h_in, input[input_offset], load);
            auto load_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(load_event);
            WaitFlag<HardEvent::MTE2_V>(load_event);



            if (active_layers_ < pack::NL) {
                const uint32_t inactive_elems =
                    (pack::NL - active_layers_) * pack::RE_TILE;
                for (uint32_t rx = 0; rx < pack::RX_TILE; ++rx) {
                    const uint32_t offset =
                        (rx * pack::NL + active_layers_) * pack::RE_TILE;
                    Duplicate(h_in[offset], static_cast<half>(0.0f), inactive_elems);
                }
            }
            for (uint32_t rx = 0; rx < pack::RX_TILE; ++rx) {
                const uint32_t tile = rx * pack::NL * pack::RE_TILE;
                Transpose(h_out[tile], h_in[tile]);
            }
            auto store_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(store_event);
            WaitFlag<HardEvent::V_MTE3>(store_event);
            for (uint32_t rx = 0; rx < pack::RX_TILE; ++rx) {
                const size_t output_offset =
                    static_cast<size_t>(re) * pack::NR * pack::NL +
                    (rx_base + rx) * pack::NL;
                DataCopy(output[output_offset],
                         h_out[rx * pack::NL * pack::RE_TILE], store);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void PackYPlane(
        const GlobalTensor<half> &input, GlobalTensor<half> &output, uint32_t re)
    {
        auto rx_in = rx_in_buf_.Get<half>();
        auto rx_trans = rx_trans_buf_.Get<half>();
        auto y_out = y_out_buf_.Get<half>();

        DataCopyParams load;
        load.blockCount = pack::RX_GROUP;
        load.blockLen = 1;
        load.srcStride = GRID_ROW_BLOCKS - 1;
        load.dstStride = 0;
        DataCopyParams store;
        store.blockCount = pack::RE_TILE;
        store.blockLen = pack::RX_GROUP;
        store.srcStride = 0;
        store.dstStride = (pack::NR - pack::RX_GROUP) * pack::NL / HALF_PER_BLOCK;

        for (uint32_t rx_base = 0; rx_base < pack::NR; rx_base += pack::RX_GROUP) {
            const size_t input_offset = static_cast<size_t>(rx_base) * pack::N_RE + re;
            DataCopy(rx_in, input[input_offset], load);
            auto load_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(load_event);
            WaitFlag<HardEvent::MTE2_V>(load_event);


            Transpose(rx_trans, rx_in);
            Brcb(y_out, rx_trans, (pack::RE_TILE * pack::RX_GROUP) / 8, {1, 8});
            auto store_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(store_event);
            WaitFlag<HardEvent::V_MTE3>(store_event);
            const size_t output_offset =
                static_cast<size_t>(re) * pack::NR * pack::NL + rx_base * pack::NL;
            DataCopy(output[output_offset], y_out, store);
            PipeBarrier<PIPE_ALL>();
        }
    }

    TPipe *pipe_ = nullptr;
    uint32_t core_ = 0;
    uint32_t re_begin_ = 0;
    uint32_t active_layers_ = 0;

    GlobalTensor<half> rx_re_g_, rx_im_g_, h_re_g_, h_im_g_, noise_g_;
    GlobalTensor<half> hrm_re_g_, hrm_im_g_, yvpad_re_g_, yvpad_im_g_, no_g_;
    GlobalTensor<uint32_t> metadata_g_;
    TBuf<TPosition::VECCALC> h_in_buf_, h_out_buf_;
    TBuf<TPosition::VECCALC> rx_in_buf_, rx_trans_buf_, y_out_buf_;
    TBuf<TPosition::VECCALC> noise_half_buf_, noise_float_buf_, no_out_buf_;
    TBuf<TPosition::VECCALC> metadata_buf_;
};

}

extern "C" __global__ __aicore__ void mimo_detect_io_pack_kernel(
    GM_ADDR rx_grid_re, GM_ADDR rx_grid_im,
    GM_ADDR h_grid_re, GM_ADDR h_grid_im, GM_ADDR noise_var_rx,
    GM_ADDR hrm_re, GM_ADDR hrm_im,
    GM_ADDR yvpad_re, GM_ADDR yvpad_im, GM_ADDR no,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    TPipe pipe;
    MimoDetectIoPack op;
    op.Init(rx_grid_re, rx_grid_im, h_grid_re, h_grid_im, noise_var_rx,
            hrm_re, hrm_im, yvpad_re, yvpad_im, no, tiling, &pipe);
    op.Process();
}
