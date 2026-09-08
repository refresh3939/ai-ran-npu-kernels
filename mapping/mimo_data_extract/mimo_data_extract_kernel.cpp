



#include "kernel_operator.h"
#include "mimo_data_extract.h"

using namespace AscendC;
namespace mde = airan::mimo_data_extract;

namespace {





__aicore__ inline void ExtractPlane(const GlobalTensor<half> &source,
                                    const GlobalTensor<half> &destination,
                                    LocalTensor<half> row,
                                    LocalTensor<half> compact,
                                    LocalTensor<half> zero_tail,
                                    LocalTensor<uint32_t> metadata,
                                    uint32_t layer)
{
    const half zero = static_cast<half>(0.0f);
    const uint32_t grid_base = layer * mde::N_GRID;
    const uint32_t data_base = layer * mde::N_DATA_PAD;

    for (uint32_t group = 0; group < mde::NUM_GROUPS; ++group) {
        for (uint32_t local = 0; local < mde::GROUP_SYMBOLS; ++local) {
            const uint32_t ordinal = group * mde::GROUP_SYMBOLS + local;
            const uint32_t physical_symbol = metadata.GetValue(10 + ordinal);
            DataCopy(row, source[grid_base + physical_symbol * mde::N_SC_PAD],
                     mde::N_SC_PAD);
            auto load_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(load_event);
            WaitFlag<HardEvent::MTE2_V>(load_event);
            Adds(compact[local * mde::N_SC_USED], row, zero, mde::N_SC_USED);
            auto reuse_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(reuse_event);
            WaitFlag<HardEvent::V_MTE2>(reuse_event);
        }
        auto store_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(store_event);
        WaitFlag<HardEvent::V_MTE3>(store_event);
        DataCopy(destination[data_base + group * mde::GROUP_RE], compact,
                 mde::GROUP_RE);
        auto output_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(output_event);
        WaitFlag<HardEvent::MTE3_V>(output_event);
    }

    Duplicate(zero_tail, zero, mde::OUTPUT_TAIL);
    auto tail_event = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(tail_event);
    WaitFlag<HardEvent::V_MTE3>(tail_event);
    DataCopy(destination[data_base + mde::N_DATA_RE], zero_tail, mde::OUTPUT_TAIL);
}

}

extern "C" __global__ __aicore__ void mimo_data_extract_kernel(
    GM_ADDR xhat_re_gm, GM_ADDR xhat_im_gm, GM_ADDR no_eff_gm,
    GM_ADDR data_re_gm, GM_ADDR data_im_gm, GM_ADDR data_no_eff_gm,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm)
{
    (void)workspace_gm;
    const uint32_t layer = GetBlockIdx();
    if (layer >= mde::BLOCK_DIM) return;

    TPipe pipe;
    TBuf<TPosition::VECCALC> metadata_buf, row_buf, compact_buf, zero_buf;
    pipe.InitBuffer(metadata_buf, mde::TILING_BYTES);
    pipe.InitBuffer(row_buf, mde::N_SC_PAD * sizeof(half));
    pipe.InitBuffer(compact_buf, mde::GROUP_RE * sizeof(half));
    pipe.InitBuffer(zero_buf, 64 * sizeof(half));

    GlobalTensor<uint32_t> tiling;
    tiling.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling_gm),
                           mde::META_WORDS);
    auto metadata = metadata_buf.Get<uint32_t>();
    DataCopy(metadata, tiling, mde::META_WORDS);
    auto metadata_event = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(metadata_event);
    WaitFlag<HardEvent::MTE2_S>(metadata_event);

    const uint32_t num_layers = metadata.GetValue(1);
    if (metadata.GetValue(0) != mde::META_MAGIC || layer >= num_layers ||
        num_layers == 0 || num_layers > mde::MAX_LAYERS ||
        metadata.GetValue(2) != mde::N_SYMBOLS ||
        metadata.GetValue(3) != mde::N_SC_USED ||
        metadata.GetValue(4) != mde::N_SC_PAD ||
        metadata.GetValue(5) != mde::N_DATA_SYMBOLS ||
        metadata.GetValue(6) != mde::N_DATA_RE ||
        metadata.GetValue(7) != mde::N_DATA_PAD ||
        metadata.GetValue(8) != mde::N_GRID) {
        return;
    }

    GlobalTensor<half> xhat_re, xhat_im, no_eff;
    GlobalTensor<half> data_re, data_im, data_no_eff;
    xhat_re.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(xhat_re_gm),
                            mde::MAX_LAYERS * mde::N_GRID);
    xhat_im.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(xhat_im_gm),
                            mde::MAX_LAYERS * mde::N_GRID);
    no_eff.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(no_eff_gm),
                           mde::MAX_LAYERS * mde::N_GRID);
    data_re.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(data_re_gm),
                            mde::MAX_LAYERS * mde::N_DATA_PAD);
    data_im.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(data_im_gm),
                            mde::MAX_LAYERS * mde::N_DATA_PAD);
    data_no_eff.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(data_no_eff_gm),
                                mde::MAX_LAYERS * mde::N_DATA_PAD);

    auto row = row_buf.Get<half>();
    auto compact = compact_buf.Get<half>();
    auto zero_tail = zero_buf.Get<half>();
    ExtractPlane(xhat_re, data_re, row, compact, zero_tail, metadata, layer);
    ExtractPlane(xhat_im, data_im, row, compact, zero_tail, metadata, layer);
    ExtractPlane(no_eff, data_no_eff, row, compact, zero_tail, metadata, layer);
}
