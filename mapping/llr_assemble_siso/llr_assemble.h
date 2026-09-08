






















#pragma once
#include <cstddef>
#include <cstdint>

namespace airan {

constexpr uint32_t N_STREAMS  = 8;
constexpr uint32_t BLOCK_DIM  = 4;


constexpr uint32_t SLOT_VALID = 19152;
constexpr uint32_t SLOT_PAD   = 19200;
constexpr uint32_t N_SLOT_IN  = 24;


constexpr uint32_t TILE_VALID = 10960;
constexpr uint32_t TILE_PAD   = 11264;
constexpr uint32_t N_TILE     = 41;


constexpr uint32_t CW_LEN     = N_TILE * TILE_VALID;

static_assert(N_SLOT_IN * SLOT_VALID >= CW_LEN, "input slots must cover codeword");
static_assert(CW_LEN     % 16 == 0, "codeword len must be 16-aligned");
static_assert(SLOT_VALID % 16 == 0, "source chunk must be 16-aligned");
static_assert(TILE_VALID % 16 == 0, "dest tile must be 16-aligned");
static_assert(SLOT_PAD   % 16 == 0 && TILE_PAD % 16 == 0, "strides must be 16-aligned");



constexpr uint32_t UB_MAX_RUN = TILE_VALID;


constexpr size_t IN_BYTES  = (size_t)N_SLOT_IN * N_STREAMS * SLOT_PAD * sizeof(int16_t);
constexpr size_t OUT_BYTES = (size_t)N_TILE    * N_STREAMS * TILE_PAD * sizeof(int16_t);

constexpr size_t TILING_TOTAL_SIZE = 128;
constexpr size_t WS_TOTAL          = 1 * 1024 * 1024;

}
