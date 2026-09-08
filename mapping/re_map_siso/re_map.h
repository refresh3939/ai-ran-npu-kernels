






#ifndef RE_MAP_H
#define RE_MAP_H

#include <cstdint>

namespace re_map {

constexpr uint32_t N_FFT            = 2048;
constexpr uint32_t N_SYMBOL         = 14;
constexpr uint32_t N_SC_USED        = 1596;
constexpr uint32_t N_SC_PAD         = 1664;
constexpr uint32_t SYMBOLS_PER_CORE = 4;
constexpr uint32_t BLOCK_DIM        = 4;
constexpr uint32_t ZERO_SLOT        = N_SC_PAD;
constexpr uint32_t SRC_UB_LEN       = N_SC_PAD + 16;

}

#endif
