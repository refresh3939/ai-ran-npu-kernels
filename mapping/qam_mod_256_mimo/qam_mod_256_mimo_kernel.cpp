







#include "kernel_operator.h"
#include "qam_mod_256_mimo.h"

using namespace AscendC;
namespace qmm = airan::qam_mod_256_mimo;

namespace {
constexpr uint32_t TILE_BYTES = qmm::TILE_ELEMS * sizeof(int16_t);
static_assert(qmm::N_DATA_RE % qmm::COMPUTE_CORES == 0,
              "valid codeword must split exactly across compute cores");
static_assert(qmm::CORE_RE % 16 == 0,
              "each core segment must preserve 32-byte DMA alignment");
static_assert((11 * TILE_BYTES + qmm::TILING_BYTES) <= 128 * 1024,
              "qam_mod_256_mimo UB working set exceeds dav_m200 capacity");
}

extern "C" __global__ __aicore__ void qam_mod_256_mimo_kernel(
    GM_ADDR bits_qam_gm, GM_ADDR d_re_gm, GM_ADDR d_im_gm,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm)
{
    (void)workspace_gm;
    const uint32_t core = GetBlockIdx() ^ 2u;
    if (core >= qmm::BLOCK_DIM) return;

    TPipe pipe;
    TBuf<TPosition::VECCALC> meta_buf;
    TBuf<TPosition::VECCALC> bit_buf[8];
    TBuf<TPosition::VECCALC> tmp_buf, re_buf, im_buf;
    pipe.InitBuffer(meta_buf, qmm::TILING_BYTES);
    for (uint32_t stream = 0; stream < qmm::Q_M; ++stream) {
        pipe.InitBuffer(bit_buf[stream], TILE_BYTES);
    }
    pipe.InitBuffer(tmp_buf, TILE_BYTES);
    pipe.InitBuffer(re_buf, TILE_BYTES);
    pipe.InitBuffer(im_buf, TILE_BYTES);

    GlobalTensor<uint32_t> tiling_g;
    tiling_g.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling_gm),
                             qmm::META_WORDS);
    auto meta = meta_buf.Get<uint32_t>();
    DataCopy(meta, tiling_g, qmm::META_WORDS);
    auto metadata_event = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(metadata_event);
    WaitFlag<HardEvent::MTE2_S>(metadata_event);

    const uint32_t magic = meta.GetValue(0);
    const uint32_t num_layers = meta.GetValue(1);
    const uint32_t qm = meta.GetValue(2);
    const uint32_t num_data_re = meta.GetValue(3);
    const uint32_t data_stride = meta.GetValue(4);
    const uint32_t codeword_symbols = meta.GetValue(5);
    const uint32_t codeword_stride = meta.GetValue(6);
    const uint32_t tile_elems = meta.GetValue(7);
    const uint32_t compute_cores = meta.GetValue(8);
    if (magic != qmm::META_MAGIC || num_layers == 0 ||
        num_layers > qmm::MAX_LAYERS || qm != qmm::Q_M ||
        num_data_re != qmm::N_DATA_RE || data_stride != qmm::N_DATA_PAD ||
        codeword_symbols != num_layers * qmm::N_DATA_RE ||
        codeword_stride != num_layers * qmm::N_DATA_PAD ||
        tile_elems != qmm::TILE_ELEMS || compute_cores != qmm::COMPUTE_CORES) {
        return;
    }

    GlobalTensor<int16_t> bits_g;
    GlobalTensor<half> d_re_g, d_im_g;
    bits_g.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(bits_qam_gm),
                           qm * codeword_stride);
    d_re_g.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(d_re_gm),
                           codeword_stride);
    d_im_g.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(d_im_gm),
                           codeword_stride);

    auto c0 = bit_buf[0].Get<half>(); auto c1 = bit_buf[1].Get<half>();
    auto c2 = bit_buf[2].Get<half>(); auto c3 = bit_buf[3].Get<half>();
    auto c4 = bit_buf[4].Get<half>(); auto c5 = bit_buf[5].Get<half>();
    auto c6 = bit_buf[6].Get<half>(); auto c7 = bit_buf[7].Get<half>();
    auto tmp = tmp_buf.Get<half>();
    auto d_re = re_buf.Get<half>();
    auto d_im = im_buf.Get<half>();

    const half h2 = static_cast<half>(2.0f);
    const half hm1 = static_cast<half>(-1.0f);
    const half hm2 = static_cast<half>(-2.0f);
    const half h3 = static_cast<half>(3.0f);
    const half h4 = static_cast<half>(4.0f);
    const half h8 = static_cast<half>(8.0f);
    const half hz = static_cast<half>(0.0f);
    const half hD = static_cast<half>(qmm::D_256QAM);

#define MOD_AXIS(cc0, cc1, cc2, cc3, dst, count)                              \
    do {                                                                       \
        Muls((cc0), (cc0), h2, (count)); Adds((cc0), (cc0), hm1, (count));    \
        Muls((cc1), (cc1), h2, (count)); Adds((cc1), (cc1), hm1, (count));    \
        Muls((cc2), (cc2), h2, (count)); Adds((cc2), (cc2), hm1, (count));    \
        Muls((cc3), (cc3), hm2, (count)); Adds((cc3), (cc3), h3, (count));    \
        Mul(tmp, (cc2), (cc3), (count));                                       \
        Muls(tmp, tmp, hm1, (count)); Adds(tmp, tmp, h4, (count));             \
        Mul(tmp, (cc1), tmp, (count));                                         \
        Muls(tmp, tmp, hm1, (count)); Adds(tmp, tmp, h8, (count));             \
        Mul(tmp, (cc0), tmp, (count));                                         \
        Muls((dst), tmp, hD, (count));                                         \
    } while (0)

    if (core < compute_cores) {
        const uint32_t core_count = num_layers * qmm::CORE_RE;
        const uint32_t core_base = core * core_count;
        for (uint32_t offset = 0; offset < core_count; offset += tile_elems) {
            uint32_t count = core_count - offset;
            if (count > tile_elems) count = tile_elems;
            for (uint32_t stream = 0; stream < qmm::Q_M; ++stream) {
                auto input = bit_buf[stream].Get<int16_t>();
                DataCopy(input,
                         bits_g[stream * codeword_stride + core_base + offset],
                         count);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            Cast(c0, bit_buf[0].Get<int16_t>(), RoundMode::CAST_NONE, count);
            Cast(c1, bit_buf[1].Get<int16_t>(), RoundMode::CAST_NONE, count);
            Cast(c2, bit_buf[2].Get<int16_t>(), RoundMode::CAST_NONE, count);
            Cast(c3, bit_buf[3].Get<int16_t>(), RoundMode::CAST_NONE, count);
            Cast(c4, bit_buf[4].Get<int16_t>(), RoundMode::CAST_NONE, count);
            Cast(c5, bit_buf[5].Get<int16_t>(), RoundMode::CAST_NONE, count);
            Cast(c6, bit_buf[6].Get<int16_t>(), RoundMode::CAST_NONE, count);
            Cast(c7, bit_buf[7].Get<int16_t>(), RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();

            MOD_AXIS(c0, c1, c2, c3, d_re, count);
            MOD_AXIS(c4, c5, c6, c7, d_im, count);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            DataCopy(d_re_g[core_base + offset], d_re, count);
            DataCopy(d_im_g[core_base + offset], d_im, count);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    } else {
        const uint32_t tail = codeword_stride - codeword_symbols;
        Duplicate(d_re, hz, tail);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        DataCopy(d_re_g[codeword_symbols], d_re, tail);
        DataCopy(d_im_g[codeword_symbols], d_re, tail);
    }
    PipeBarrier<PIPE_ALL>();

#undef MOD_AXIS
}
