














#pragma once
#include <cstddef>
#include <cstdint>

namespace cfo_compensate {

constexpr uint32_t N_SAMPLE_PER_SLOT = 30720;
constexpr uint32_t SUB_TILE          = 2048;
constexpr uint32_t N_SUBTILE         = 15;
constexpr uint32_t BLOCK_DIM         = 4;

constexpr uint32_t IN_INT16_LEN      = 2 * N_SAMPLE_PER_SLOT;
constexpr uint32_t OUT_INT16_LEN     = 2 * N_SAMPLE_PER_SLOT;

static_assert(SUB_TILE * N_SUBTILE == N_SAMPLE_PER_SLOT, "subtile 切分不整除");

constexpr size_t TILING_TOTAL_SIZE = 128;
constexpr size_t WS_TOTAL          = 1 * 1024 * 1024;

}