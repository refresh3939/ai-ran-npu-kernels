#pragma once
#include <cstddef>
#include <cstdint>
























namespace airan_scr {

constexpr uint32_t N_STREAMS    = 8;
constexpr uint32_t N_DATA_SYM   = 12;
constexpr uint32_t N_SC_USED    = 1596;
constexpr uint32_t QAM_SYM_STRIDE = 1600;
constexpr uint32_t N_SYM        = N_DATA_SYM * N_SC_USED;
constexpr uint32_t N_SYM_PAD    = 19200;
constexpr uint32_t N_SLOT_MAX   = 23;
constexpr uint32_t BLOCK_DIM    = 4;

static_assert(N_DATA_SYM * QAM_SYM_STRIDE == N_SYM_PAD, "padded QAM geometry mismatch");

constexpr size_t TILING_TOTAL_SIZE = 128;
constexpr size_t WS_TOTAL          = 4 * 1024 * 1024;

}
