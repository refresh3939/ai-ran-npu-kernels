

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t N_SC_PAD = 1664;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_DMRS_PAD = 896;
constexpr uint32_t ZERO_SLOT = N_DMRS_PAD;
constexpr uint32_t SRC_LEN = N_DMRS_PAD + 16;
constexpr uint32_t N_SYMBOL = 14;
constexpr uint32_t DMRS_SYMBOL_0 = 2;
constexpr uint32_t DMRS_SYMBOL_1 = 11;
}

extern "C" __global__ __aicore__ void merge_dmrs_device_kernel(
    GM_ADDR grid_re_gm, GM_ADDR grid_im_gm,
    GM_ADDR dmrs_re_gm, GM_ADDR dmrs_im_gm,
    GM_ADDR gather_idx_gm, GM_ADDR workspace_gm, GM_ADDR tiling_gm)
{
    (void)workspace_gm;
    (void)tiling_gm;
    const uint32_t block = GetBlockIdx();
    if (block >= 4) return;
    const uint32_t dmrs_row = block >> 1;
    const bool imag = (block & 1u) != 0;
    const uint32_t symbol = dmrs_row == 0 ? DMRS_SYMBOL_0 : DMRS_SYMBOL_1;

    TPipe pipe;
    TBuf<TPosition::VECCALC> src_buf;
    TBuf<TPosition::VECCALC> dst_buf;
    TBuf<TPosition::VECCALC> idx_buf;
    pipe.InitBuffer(src_buf, SRC_LEN * sizeof(half));
    pipe.InitBuffer(dst_buf, N_SC_PAD * sizeof(half));
    pipe.InitBuffer(idx_buf, N_SC_PAD * sizeof(uint32_t));

    auto src = src_buf.Get<half>();
    auto dst = dst_buf.Get<half>();
    auto idx = idx_buf.Get<uint32_t>();
    GlobalTensor<half> grid_re, grid_im, dmrs_re, dmrs_im;
    GlobalTensor<uint32_t> gather_idx;
    grid_re.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(grid_re_gm), N_SYMBOL * N_SC_PAD);
    grid_im.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(grid_im_gm), N_SYMBOL * N_SC_PAD);
    dmrs_re.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dmrs_re_gm), 2 * N_DMRS_PAD);
    dmrs_im.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dmrs_im_gm), 2 * N_DMRS_PAD);
    gather_idx.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather_idx_gm), N_SC_PAD);

    DataCopy(idx, gather_idx, N_SC_PAD);
    if (imag) DataCopy(src, dmrs_im[dmrs_row * N_DMRS_PAD], N_DMRS_PAD);
    else DataCopy(src, dmrs_re[dmrs_row * N_DMRS_PAD], N_DMRS_PAD);
    event_t load_done = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(load_done);
    WaitFlag<HardEvent::MTE2_V>(load_done);



    Duplicate(src[ZERO_SLOT], static_cast<half>(0), 16);
    Gather(dst, src, idx, static_cast<uint32_t>(0), N_SC_PAD);
    PipeBarrier<PIPE_V>();
    event_t store_ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(store_ready);
    WaitFlag<HardEvent::V_MTE3>(store_ready);
    if (imag) DataCopy(grid_im[symbol * N_SC_PAD], dst, N_SC_PAD);
    else DataCopy(grid_re[symbol * N_SC_PAD], dst, N_SC_PAD);
    event_t store_done = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(store_done);
    WaitFlag<HardEvent::MTE3_S>(store_done);
}
