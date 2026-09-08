









#pragma once
#include <cstddef>
#include <cstdint>

namespace ssb_fft {


constexpr uint32_t N_FFT       = 256;
constexpr uint32_t N_SYMBOL    = 3;
constexpr uint32_t SSB_CP      = 36;
constexpr uint32_t SYM_STRIDE  = 292;
constexpr uint32_t N_DMRS_RE   = 144;


constexpr uint32_t P           = 16;
constexpr uint32_t Q           = 16;
constexpr uint32_t TILE_PQ     = P * Q;


constexpr uint16_t M_MMAD      = 16;
constexpr uint16_t N_SUB       = 16;
constexpr uint16_t N_SPLITS    = 1;
constexpr uint16_t K_PHASE1    = 16;
constexpr uint16_t K_PHASE3    = 16;


constexpr uint32_t BLOCK_DIM        = 4;
constexpr uint32_t SYMBOLS_PER_CORE = N_SYMBOL;
constexpr uint32_t BATCH_X_ELEMS    = SYMBOLS_PER_CORE * TILE_PQ;


constexpr float    INPUT_SCALE        = 1.0f / 256.0f;
constexpr uint32_t INPUT_GM_INT16_LEN = 1760;

}
