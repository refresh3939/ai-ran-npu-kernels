















#pragma once
#include <cstddef>
#include <cstdint>


namespace ofdm_mod {


constexpr uint32_t N_FFT             = 2048;
constexpr uint32_t N_SYMBOL          = 14;
constexpr uint32_t SYM_STRIDE        = 2192;
constexpr uint32_t SYM0_CP_OFFSET    = 176;
constexpr uint32_t N_SAMPLE_PER_SLOT = 30720;

constexpr uint32_t CP_LEN_FIRST      = 176;
constexpr uint32_t CP_LEN_OTHER      = 144;


constexpr uint32_t P                 = 32;
constexpr uint32_t Q                 = 64;
constexpr uint32_t TILE_PQ           = P * Q;


constexpr uint16_t M_MMAD            = 32;
constexpr uint16_t N_SUB             = 16;
constexpr uint16_t N_SPLITS          = 4;
constexpr uint16_t K_PHASE1          = 32;
constexpr uint16_t K_PHASE3          = 64;


constexpr uint32_t BLOCK_DIM         = 4;
constexpr uint32_t SYMBOLS_PER_CORE  = 4;
constexpr uint32_t BATCH_X_ELEMS     = SYMBOLS_PER_CORE * TILE_PQ;
constexpr uint32_t BATCH_TILE        = SYMBOLS_PER_CORE * P * Q;
constexpr uint32_t BATCH_HALF_TILE   = SYMBOLS_PER_CORE * P * P;
constexpr uint16_t MAX_M_BATCH       = SYMBOLS_PER_CORE * P;








constexpr float    OUT_SCALE         = 1600.0f;
constexpr uint32_t INPUT_GM_HALF_LEN = N_SYMBOL * N_FFT;
constexpr uint32_t OUTPUT_GM_LEN     = N_SAMPLE_PER_SLOT;

}
