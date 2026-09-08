
#pragma once

#include <cstdint>

namespace airan::pusch_precode_kernel {

constexpr uint32_t N_SYMBOLS = 14;
constexpr uint32_t N_RB = 133;
constexpr uint32_t N_SC_PER_RB = 12;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_SC_PAD = 1664;
constexpr uint32_t N_RE_GRID = N_SYMBOLS * N_SC_PAD;
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t MAX_PORTS = 4;
constexpr uint32_t MAX_PRGS = N_RB;
constexpr uint32_t MAX_WEIGHT_ELEMS = MAX_PRGS * MAX_PORTS * MAX_LAYERS;
constexpr uint32_t PRG_MAP_PAD = 144;
constexpr uint32_t TILING_MAGIC = 0x50435031u;
constexpr uint32_t TILING_WORDS = 32;
constexpr uint32_t TILING_WEIGHT_KIND_OFFSET = 9;
constexpr uint32_t WEIGHT_KIND_REAL = 1;
constexpr uint32_t WEIGHT_KIND_IMAG = 2;

}
