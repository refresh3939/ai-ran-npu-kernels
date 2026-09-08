






#include "kernel_operator.h"
#include "layer_map.h"

using namespace AscendC;

namespace lm = airan::layer_map;

extern "C" __global__ __aicore__ void layer_map_kernel(
    GM_ADDR d_re_gm, GM_ADDR d_im_gm,
    GM_ADDR layer_re_gm, GM_ADDR layer_im_gm,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm)
{
    (void)workspace_gm;
    const uint32_t layer = GetBlockIdx();
    if (layer >= lm::BLOCK_DIM) return;

    TPipe pipe;
    TBuf<TPosition::VECCALC> meta_buf, source_re_buf, source_im_buf;
    TBuf<TPosition::VECCALC> output_re_buf, output_im_buf, index_buf;
    pipe.InitBuffer(meta_buf, lm::META_WORDS * sizeof(uint32_t));
    pipe.InitBuffer(source_re_buf,
                    lm::MAX_LAYERS * lm::TILE_SYMBOLS * sizeof(half));
    pipe.InitBuffer(source_im_buf,
                    lm::MAX_LAYERS * lm::TILE_SYMBOLS * sizeof(half));
    pipe.InitBuffer(output_re_buf, lm::TILE_SYMBOLS * sizeof(half));
    pipe.InitBuffer(output_im_buf, lm::TILE_SYMBOLS * sizeof(half));
    pipe.InitBuffer(index_buf,
                    lm::INDEX_ELEMS_PER_LAYER * sizeof(uint32_t));

    GlobalTensor<uint32_t> tiling_g;
    tiling_g.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling_gm),
                            lm::META_WORDS + lm::MAX_INDEX_ELEMS);
    auto metadata = meta_buf.Get<uint32_t>();
    DataCopy(metadata, tiling_g, lm::META_WORDS);
    auto metadata_event =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(metadata_event);
    WaitFlag<HardEvent::MTE2_S>(metadata_event);

    if (metadata.GetValue(0) != lm::META_MAGIC) return;
    const uint32_t num_layers = metadata.GetValue(1);
    const uint32_t num_data_re = metadata.GetValue(2);
    const uint32_t data_stride = metadata.GetValue(3);
    const uint32_t codeword_symbols = metadata.GetValue(4);
    const uint32_t codeword_stride = metadata.GetValue(5);
    const uint32_t tile_symbols = metadata.GetValue(6);
    const uint32_t index_elems = metadata.GetValue(7);
    if (layer >= num_layers || num_layers == 0 || num_layers > lm::MAX_LAYERS ||
        num_data_re != lm::N_DATA_RE || data_stride != lm::N_DATA_PAD ||
        codeword_symbols != num_layers * num_data_re ||
        codeword_stride != num_layers * data_stride ||
        tile_symbols != lm::TILE_SYMBOLS ||
        index_elems != lm::INDEX_ELEMS_PER_LAYER) {
        return;
    }

    GlobalTensor<half> d_re_g, d_im_g, layer_re_g, layer_im_g;
    d_re_g.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(d_re_gm),
                           lm::MAX_LAYERS * lm::N_DATA_PAD);
    d_im_g.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(d_im_gm),
                           lm::MAX_LAYERS * lm::N_DATA_PAD);
    layer_re_g.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(layer_re_gm),
                               lm::MAX_LAYERS * lm::N_DATA_PAD);
    layer_im_g.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(layer_im_gm),
                               lm::MAX_LAYERS * lm::N_DATA_PAD);

    auto source_re = source_re_buf.Get<half>();
    auto source_im = source_im_buf.Get<half>();
    auto output_re = output_re_buf.Get<half>();
    auto output_im = output_im_buf.Get<half>();



    if (num_layers == 1) {
        constexpr uint32_t direct_tile = lm::MAX_LAYERS * lm::TILE_SYMBOLS;
        for (uint32_t base = 0; base < num_data_re; base += direct_tile) {
            uint32_t valid = num_data_re - base;
            if (valid > direct_tile) valid = direct_tile;
            DataCopy(source_re, d_re_g[base], valid);
            DataCopy(source_im, d_im_g[base], valid);
            auto load_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(load_event);
            WaitFlag<HardEvent::MTE2_MTE3>(load_event);
            DataCopy(layer_re_g[base], source_re, valid);
            DataCopy(layer_im_g[base], source_im, valid);
            auto store_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(store_event);
            WaitFlag<HardEvent::MTE3_MTE2>(store_event);
        }
    } else {
        auto index = index_buf.Get<uint32_t>();
        const uint32_t index_base =
            lm::META_WORDS + layer * lm::INDEX_ELEMS_PER_LAYER;
        DataCopy(index, tiling_g[index_base], lm::INDEX_ELEMS_PER_LAYER);
        auto index_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(index_event);
        WaitFlag<HardEvent::MTE2_V>(index_event);

        const uint32_t num_tiles =
            (num_data_re + tile_symbols - 1) / tile_symbols;
        for (uint32_t tile = 0; tile < num_tiles; ++tile) {
            const uint32_t symbol_base = tile * tile_symbols;
            uint32_t valid = num_data_re - symbol_base;
            if (valid > tile_symbols) valid = tile_symbols;
            const uint32_t source_base = num_layers * symbol_base;
            const uint32_t source_count = num_layers * valid;
            DataCopy(source_re, d_re_g[source_base], source_count);
            DataCopy(source_im, d_im_g[source_base], source_count);
            auto load_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(load_event);
            WaitFlag<HardEvent::MTE2_V>(load_event);

            if (valid == tile_symbols) {
                Gather(output_re, source_re, index, static_cast<uint32_t>(0),
                       lm::TILE_SYMBOLS);
                Gather(output_im, source_im, index, static_cast<uint32_t>(0),
                       lm::TILE_SYMBOLS);
            } else {
                Gather(output_re, source_re, index, static_cast<uint32_t>(0),
                       lm::TAIL_SYMBOLS);
                Gather(output_im, source_im, index, static_cast<uint32_t>(0),
                       lm::TAIL_SYMBOLS);
            }
            auto gather_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(gather_event);
            WaitFlag<HardEvent::V_MTE3>(gather_event);
            const uint32_t destination = layer * data_stride + symbol_base;
            DataCopy(layer_re_g[destination], output_re, valid);
            DataCopy(layer_im_g[destination], output_im, valid);
            auto store_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(store_event);
            WaitFlag<HardEvent::MTE3_MTE2>(store_event);
        }
    }




    auto reuse_event = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(reuse_event);
    WaitFlag<HardEvent::MTE3_V>(reuse_event);
    const uint32_t tail = data_stride - num_data_re;
    Duplicate(output_re, static_cast<half>(0.0f), tail);
    auto zero_event = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(zero_event);
    WaitFlag<HardEvent::V_MTE3>(zero_event);
    const uint32_t tail_base = layer * data_stride + num_data_re;
    DataCopy(layer_re_g[tail_base], output_re, tail);
    DataCopy(layer_im_g[tail_base], output_re, tail);
}
