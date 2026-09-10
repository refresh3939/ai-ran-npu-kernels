/**
 * @file mimo_detect_io_pack_kernel.cpp
 * Four-core, bit-preserving mixed-radix layout adapter for a selected
 * NR x 16 detector.
 *
 * Natural input:
 *   H [rx,layer,re], y [rx,re]
 * Physical BRI input:
 *   hrm [re,rx,layer], yvpad [re,rx,layer]
 */
#include "kernel_operator.h"
#include "mimo_detect_io_pack.h"

using namespace AscendC;
namespace pack = airan::mimo_detect_io_pack;

namespace {

constexpr uint32_t HALF_PER_BLOCK = 16;
constexpr uint32_t GRID_ROW_BLOCKS = pack::N_RE / HALF_PER_BLOCK;
constexpr uint32_t H_RX_TILE = pack::NR;
constexpr uint32_t H_TILE_ELEMS = H_RX_TILE * pack::NL * pack::RE_TILE;
constexpr uint32_t H_TRANSPOSE_BATCH = H_RX_TILE;
constexpr uint32_t H_TRANSPOSE_GROUPS = H_RX_TILE / H_TRANSPOSE_BATCH;
constexpr uint32_t H_VA_WORDS = 4;
constexpr uint32_t H_ADDRS_PER_GROUP = 2 * H_VA_WORDS;
constexpr uint32_t H_ADDR_ELEMS =
    H_TRANSPOSE_GROUPS * H_ADDRS_PER_GROUP;
constexpr uint32_t RX_TILE_ELEMS = pack::RX_GROUP * pack::RE_TILE;
constexpr uint32_t Y_TILE_ELEMS = pack::RE_TILE * pack::RX_GROUP * pack::NL;
constexpr uint32_t Y_BUFFER_SLOTS = 3;
constexpr uint32_t META_WORDS = pack::TILING_BYTES / sizeof(uint32_t);

static_assert(pack::NL == 16 && pack::RE_TILE == 16,
              "mixed-radix base tile must be 16x16");
static_assert(pack::NR == 16 || pack::NR == 32 || pack::NR == 64,
              "mixed-radix path supports 16, 32, or 64 receivers");
static_assert(H_RX_TILE % H_TRANSPOSE_BATCH == 0,
              "batched H transpose must cover every receiver");

constexpr size_t UB_WORKING_BYTES =
    3 * H_TILE_ELEMS * sizeof(half) +
    2 * RX_TILE_ELEMS * sizeof(half) +
    Y_BUFFER_SLOTS * Y_TILE_ELEMS * sizeof(half) +
    H_ADDR_ELEMS * sizeof(uint64_t) +
    pack::NOISE_ELEMS * sizeof(half) +
    pack::NOISE_ELEMS * sizeof(float) +
    pack::TILING_BYTES;
static_assert(UB_WORKING_BYTES <= 128 * 1024,
              "mixed-radix io-pack exceeds dav_m200 UB capacity");

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
#if defined(AIRAN_FUSED_RX_GROUPED_Y) && AIRAN_FUSED_RX_GROUPED_Y
        constexpr size_t grouped_elems =
            static_cast<size_t>(pack::N_RE / 8u) * pack::NR * pack::NL;
        yvpad_re_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(yvpad_re_gm), grouped_elems);
        yvpad_im_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(yvpad_im_gm), grouped_elems);
#else
        yvpad_re_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(yvpad_re_gm), pack::PACKED_ELEMS);
        yvpad_im_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(yvpad_im_gm), pack::PACKED_ELEMS);
#endif
        no_g_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(no_gm), pack::NO_ELEMS);
        metadata_g_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling_gm), META_WORDS);

        metadata_event_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        load_event_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        scalar_event_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        store_event_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        store_reuse_event_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        address_event_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));

        h_load_.blockCount = H_RX_TILE * pack::NL;
        h_load_.blockLen = 1;
        h_load_.srcStride = GRID_ROW_BLOCKS - 1;
        h_load_.dstStride = 0;
        h_radix2_.blockCount = H_RX_TILE;
        h_radix2_.blockLen = 1;
        h_radix2_.srcStride = pack::RE_TILE - 1;
        h_radix2_.dstStride = 0;
        h_store_.blockCount = pack::RE_TILE;
        h_store_.blockLen = H_RX_TILE * pack::NL / HALF_PER_BLOCK;
        h_store_.srcStride = 0;
        h_store_.dstStride = 0;
        h_transpose_.repeatTimes = H_TRANSPOSE_BATCH;
        h_transpose_.dstRepStride = pack::NL;
        h_transpose_.srcRepStride = pack::NL;
        y_load_.blockCount = pack::RX_GROUP;
        y_load_.blockLen = 1;
        y_load_.srcStride = GRID_ROW_BLOCKS - 1;
        y_load_.dstStride = 0;
        y_store_.blockCount = pack::RE_TILE;
        y_store_.blockLen = pack::RX_GROUP;
        y_store_.srcStride = 0;
        y_store_.dstStride =
            (pack::NR - pack::RX_GROUP) * pack::NL / HALF_PER_BLOCK;

        pipe_->InitBuffer(h_in_buf_, H_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(h_stage1_buf_, H_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(h_stage2_buf_, H_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(rx_in_buf_, RX_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(rx_trans_buf_, RX_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(y_out_buf_,
                          Y_BUFFER_SLOTS * Y_TILE_ELEMS * sizeof(half));
        pipe_->InitBuffer(noise_half_buf_, pack::NOISE_ELEMS * sizeof(half));
        pipe_->InitBuffer(noise_float_buf_, pack::NOISE_ELEMS * sizeof(float));
        pipe_->InitBuffer(metadata_buf_, pack::TILING_BYTES);
        pipe_->InitBuffer(h_addr_buf_, H_ADDR_ELEMS * sizeof(uint64_t));
        InitHTransposeAddresses();
    }

    __aicore__ inline void Process()
    {
        if (core_ >= pack::BLOCK_DIM || !LoadAndValidateMetadata()) return;
        PackNoise();
#pragma unroll 2
        for (uint32_t local_re = 0; local_re < pack::RE_PER_CORE;
             local_re += pack::RE_TILE) {
            const uint32_t re = re_begin_ + local_re;
            const size_t packed_base =
                static_cast<size_t>(re) * pack::NR * pack::NL;
            PackHPlane(h_re_g_, hrm_re_g_, re, packed_base);
            PackHPlane(h_im_g_, hrm_im_g_, re, packed_base);
            PackYPlane(rx_re_g_, yvpad_re_g_, re, packed_base);
            PackYPlane(rx_im_g_, yvpad_im_g_, re, packed_base);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline bool LoadAndValidateMetadata()
    {
        auto metadata = metadata_buf_.Get<uint32_t>();
        DataCopy(metadata, metadata_g_, META_WORDS);
        SetFlag<HardEvent::MTE2_S>(metadata_event_);
        WaitFlag<HardEvent::MTE2_S>(metadata_event_);

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
        auto no_out = h_stage2_buf_.Get<half>();
        DataCopy(noise_h, noise_g_, pack::NOISE_ELEMS);
        SetFlag<HardEvent::MTE2_V>(load_event_);
        WaitFlag<HardEvent::MTE2_V>(load_event_);
        Cast(noise_f, noise_h, RoundMode::CAST_NONE, pack::NOISE_ELEMS);
        SetFlag<HardEvent::V_S>(scalar_event_);
        WaitFlag<HardEvent::V_S>(scalar_event_);
        float sum = 0.0f;
        for (uint32_t rx = 0; rx < pack::NR; ++rx) sum += noise_f.GetValue(rx);
        const half mean = static_cast<half>(sum * pack::NOISE_MEAN_SCALE);
        Duplicate(no_out, mean, pack::RE_PER_CORE);
        SetFlag<HardEvent::V_MTE3>(store_event_);
        WaitFlag<HardEvent::V_MTE3>(store_event_);
        DataCopy(no_g_[re_begin_], no_out, pack::RE_PER_CORE);
    }

    __aicore__ inline void ZeroInactiveLayers(
        const LocalTensor<half> &h_in)
    {
        if (active_layers_ >= pack::NL) return;

        constexpr uint32_t LAYERS_PER_REPEAT = 8;
        constexpr uint8_t RX_REPEAT_STRIDE =
            pack::NL * pack::RE_TILE / HALF_PER_BLOCK;
        if (active_layers_ < LAYERS_PER_REPEAT) {
            const uint32_t count =
                (LAYERS_PER_REPEAT - active_layers_) * pack::RE_TILE;
            Duplicate(h_in[active_layers_ * pack::RE_TILE],
                      static_cast<half>(0.0f),
                      static_cast<uint64_t>(count),
                      static_cast<uint8_t>(H_RX_TILE), 1,
                      RX_REPEAT_STRIDE);
        }

        const uint32_t upper_begin =
            active_layers_ > LAYERS_PER_REPEAT
                ? active_layers_ : LAYERS_PER_REPEAT;
        if (upper_begin < pack::NL) {
            const uint32_t count = (pack::NL - upper_begin) * pack::RE_TILE;
            Duplicate(h_in[upper_begin * pack::RE_TILE],
                      static_cast<half>(0.0f),
                      static_cast<uint64_t>(count),
                      static_cast<uint8_t>(H_RX_TILE), 1,
                      RX_REPEAT_STRIDE);
        }
    }

    __aicore__ inline void InitHTransposeAddresses()
    {
        auto addresses = h_addr_buf_.Get<uint64_t>();
        auto dst = h_stage1_buf_.Get<half>();
        auto src = h_in_buf_.Get<half>();
#pragma unroll
        for (uint32_t group = 0; group < H_TRANSPOSE_GROUPS; ++group) {
            const uint32_t tile =
                group * H_TRANSPOSE_BATCH * pack::NL * pack::RE_TILE;
#pragma unroll
            for (uint32_t word = 0; word < H_VA_WORDS; ++word) {
                const uint32_t table = group * H_ADDRS_PER_GROUP;
                uint64_t dst_config = 0;
                uint64_t src_config = 0;
#pragma unroll
                for (uint32_t lane = 0; lane < 4; ++lane) {
                    const uint32_t row = word * 4 + lane;
                    const uint64_t dst_addr = reinterpret_cast<uint64_t>(
                        dst[tile + row * pack::RE_TILE].GetPhyAddr());
                    const uint64_t src_addr = reinterpret_cast<uint64_t>(
                        src[tile + row * pack::RE_TILE].GetPhyAddr());
                    dst_config |=
                        ((dst_addr >> 5) & 0x1fffULL) << (lane * 16);
                    src_config |=
                        ((src_addr >> 5) & 0x1fffULL) << (lane * 16);
                }
                addresses.SetValue(table + word, dst_config);
                addresses.SetValue(
                    table + H_VA_WORDS + word, src_config);
            }
        }
        SetFlag<HardEvent::S_V>(address_event_);
        WaitFlag<HardEvent::S_V>(address_event_);
    }

    __aicore__ inline void TransposeHReceivers()
    {
        auto addresses = h_addr_buf_.Get<uint64_t>();
#pragma unroll
        for (uint32_t group = 0; group < H_TRANSPOSE_GROUPS; ++group) {
            const uint32_t table = group * H_ADDRS_PER_GROUP;
            VldVaReg(
                reinterpret_cast<__ubuf__ uint64_t *>(
                    addresses[table].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint64_t *>(
                    addresses[table + H_VA_WORDS].GetPhyAddr()));
            SetFlag<HardEvent::S_V>(address_event_);
            WaitFlag<HardEvent::S_V>(address_event_);
            uint64_t unused_dst[16] = {0};
            uint64_t unused_src[16] = {0};
            TransDataTo5HDIntrinsicsImpl<half>(
                unused_dst, unused_src, h_transpose_);
            SetFlag<HardEvent::V_S>(scalar_event_);
            WaitFlag<HardEvent::V_S>(scalar_event_);
        }
    }

    __aicore__ inline void PackHPlane(
        const GlobalTensor<half> &input, GlobalTensor<half> &output,
        uint32_t re, size_t packed_base)
    {
        auto h_in = h_in_buf_.Get<half>();
        auto h_stage1 = h_stage1_buf_.Get<half>();
        auto h_stage2 = h_stage2_buf_.Get<half>();

        DataCopy(h_in, input[re], h_load_);
        SetFlag<HardEvent::MTE2_V>(load_event_);
        WaitFlag<HardEvent::MTE2_V>(load_event_);
        ZeroInactiveLayers(h_in);

        // [rx,layer,re] -> [rx,re,layer], one NR-repeat vnchwconv.
        TransposeHReceivers();

        // Delay the reuse wait until load/radix-1 have overlapped the
        // preceding GM store.
        SetFlag<HardEvent::MTE3_V>(store_reuse_event_);
        WaitFlag<HardEvent::MTE3_V>(store_reuse_event_);
        y_stores_since_wait_ = 0;
        y_slot_cursor_ = 0;

        // [rx,re,layer] -> [re,rx,layer], with one 32-byte layer vector
        // as the radix-2 copy atom.
#pragma unroll
        for (uint32_t sre = 0; sre < pack::RE_TILE; ++sre) {
            DataCopy(h_stage2[sre * H_RX_TILE * pack::NL],
                     h_stage1[sre * pack::NL], h_radix2_);
        }

        SetFlag<HardEvent::V_MTE3>(store_event_);
        WaitFlag<HardEvent::V_MTE3>(store_event_);
        DataCopy(output[packed_base], h_stage2, h_store_);
    }

    __aicore__ inline void PackYPlane(
        const GlobalTensor<half> &input, GlobalTensor<half> &output,
        uint32_t re, size_t packed_base)
    {
        auto rx_in = rx_in_buf_.Get<half>();
        auto rx_trans = rx_trans_buf_.Get<half>();
        auto y_out_base = y_out_buf_.Get<half>();
#if defined(AIRAN_FUSED_RX_GROUPED_Y) && AIRAN_FUSED_RX_GROUPED_Y
        (void)packed_base;
        auto y_out = y_out_base;
        for (uint32_t rx_base = 0; rx_base < pack::NR; rx_base += pack::RX_GROUP) {
            const size_t input_offset = static_cast<size_t>(rx_base) * pack::N_RE + re;
            DataCopy(rx_in, input[input_offset], y_load_);
            SetFlag<HardEvent::MTE2_V>(load_event_);
            WaitFlag<HardEvent::MTE2_V>(load_event_);

            // BRI consumes one [NR,16] RHS for every eight REs.  Columns
            // [0,8) carry the eight independent y samples; [8,16) are zero.
            // Keeping this conversion on device removes the staged host
            // canonical-y validation/copy and avoids materializing 16 copies.
            for (uint32_t group = 0; group < 2; ++group) {
                Duplicate(y_out, static_cast<half>(0.0f), pack::RX_GROUP * pack::NL);
                PipeBarrier<PIPE_V>();
                for (uint32_t rx = 0; rx < pack::RX_GROUP; ++rx) {
                    DataCopy(y_out[rx * pack::NL],
                             rx_in[rx * pack::RE_TILE + group * 8], pack::NL);
                    Duplicate(y_out[rx * pack::NL + 8],
                              static_cast<half>(0.0f), 8);
                }
                PipeBarrier<PIPE_ALL>();
                const size_t output_offset =
                    static_cast<size_t>(re / 8 + group) * pack::NR * pack::NL +
                    rx_base * pack::NL;
                DataCopy(output[output_offset], y_out, pack::RX_GROUP * pack::NL);
                PipeBarrier<PIPE_ALL>();
            }
        }
#else
#pragma unroll
        for (uint32_t rx_base = 0; rx_base < pack::NR; rx_base += pack::RX_GROUP) {
            auto y_out = y_out_base[y_slot_cursor_ * Y_TILE_ELEMS];
            const size_t input_offset = static_cast<size_t>(rx_base) * pack::N_RE + re;
            DataCopy(rx_in, input[input_offset], y_load_);
            SetFlag<HardEvent::MTE2_V>(load_event_);
            WaitFlag<HardEvent::MTE2_V>(load_event_);

            Transpose(rx_trans, rx_in);
            if (y_stores_since_wait_ == Y_BUFFER_SLOTS) {
                SetFlag<HardEvent::MTE3_V>(store_reuse_event_);
                WaitFlag<HardEvent::MTE3_V>(store_reuse_event_);
                y_stores_since_wait_ = 0;
            }
            Brcb(y_out, rx_trans, (pack::RE_TILE * pack::RX_GROUP) / 8, {1, 8});
            SetFlag<HardEvent::V_MTE3>(store_event_);
            WaitFlag<HardEvent::V_MTE3>(store_event_);
            DataCopy(output[packed_base + rx_base * pack::NL],
                     y_out, y_store_);
            ++y_stores_since_wait_;
            if (++y_slot_cursor_ == Y_BUFFER_SLOTS) y_slot_cursor_ = 0;
        }
#endif
    }

    TPipe *pipe_ = nullptr;
    uint32_t core_ = 0;
    uint32_t re_begin_ = 0;
    uint32_t active_layers_ = 0;
    uint32_t y_stores_since_wait_ = 0;
    uint32_t y_slot_cursor_ = 0;
    event_t metadata_event_, load_event_, scalar_event_, store_event_;
    event_t store_reuse_event_, address_event_;
    DataCopyParams h_load_, h_radix2_, h_store_, y_load_, y_store_;
    TransDataTo5HDParams h_transpose_;

    GlobalTensor<half> rx_re_g_, rx_im_g_, h_re_g_, h_im_g_, noise_g_;
    GlobalTensor<half> hrm_re_g_, hrm_im_g_, yvpad_re_g_, yvpad_im_g_, no_g_;
    GlobalTensor<uint32_t> metadata_g_;
    TBuf<TPosition::VECCALC> h_in_buf_, h_stage1_buf_, h_stage2_buf_;
    TBuf<TPosition::VECCALC> rx_in_buf_, rx_trans_buf_, y_out_buf_;
    TBuf<TPosition::VECCALC> noise_half_buf_, noise_float_buf_;
    TBuf<TPosition::VECCALC> metadata_buf_, h_addr_buf_;
};

}  // namespace

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
