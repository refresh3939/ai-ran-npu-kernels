




































#pragma once
#include <cstddef>
#include <cstdint>

namespace airan {


constexpr uint32_t N_FFT       = 2048;
constexpr uint32_t N_SC_USED   = 1596;
constexpr uint32_t N_SYMBOL    = 14;
constexpr uint32_t BLOCK_DIM   = 4;



constexpr uint32_t SC_PER_CORE = 416;
constexpr uint32_t N_SC_PAD    = BLOCK_DIM * SC_PER_CORE;

static_assert(N_SC_PAD >= N_SC_USED,  "pad must cover used SC");
static_assert(N_SC_PAD % 16 == 0,     "padded SC stride must be 16-aligned");
static_assert(SC_PER_CORE % 16 == 0,  "per-core SC must be 16-aligned");


constexpr uint32_t STREAM_HALF_LEN     = N_SYMBOL * N_SC_PAD;
constexpr uint32_t STREAM_HALF_LOGICAL = N_SYMBOL * N_SC_USED;

constexpr size_t STREAM_BYTES = STREAM_HALF_LEN * sizeof(uint16_t);




constexpr float G_FLOOR = 1.0e-4f;


constexpr size_t TILING_TOTAL_SIZE = 128;
constexpr size_t WS_TOTAL          = 1 * 1024 * 1024;

}