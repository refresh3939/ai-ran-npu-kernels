


















#pragma once
#include <stddef.h>
#include <stdint.h>

namespace airan {


constexpr uint32_t N_DMRS_RE    = 798;
constexpr uint32_t N_DMRS_PAD   = 896;
constexpr uint32_t N_DMRS_SYM   = 2;
constexpr uint32_t N_DMRS_PLANE = 1792;
constexpr uint32_t DMRS_SYM_0   = 2;
constexpr uint32_t DMRS_SYM_1   = 11;


constexpr uint32_t GOLD_NC      = 1600;
constexpr uint32_t GOLD_NBITS   = 31;
constexpr uint32_t MAT_LEN      = GOLD_NBITS * N_DMRS_PLANE;


constexpr uint32_t USE_XOR_AIV_MAP = 0;
constexpr uint32_t BLOCK_DIM       = 1;


constexpr uint32_t OUT_DBG_LEN  = 8;

}