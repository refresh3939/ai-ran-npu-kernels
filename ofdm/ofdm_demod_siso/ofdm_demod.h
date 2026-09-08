








#pragma once
#include <cstddef>
#include <cstdint>


namespace ofdm_demod {


constexpr uint32_t N_FFT             = 2048;
constexpr uint32_t N_SYMBOL          = 14;
constexpr uint32_t SYM_STRIDE        = 2192;
constexpr uint32_t SYM0_CP_OFFSET    = 176;
constexpr uint32_t N_SAMPLE_PER_SLOT = 30720;


constexpr uint32_t P                 = 32;
constexpr uint32_t Q                 = 64;
constexpr uint32_t TILE_PQ           = P * Q;


constexpr uint16_t M_MMAD            = 32;
constexpr uint16_t N_SUB             = 16;
constexpr uint16_t N_SPLITS          = 4;
constexpr uint16_t K_PHASE1          = 32;
constexpr uint16_t K_PHASE3          = 64;
constexpr uint16_t MAX_M_BATCH       = 128;


constexpr uint32_t BLOCK_DIM         = 4;
constexpr uint32_t SYMBOLS_PER_CORE  = 4;
constexpr uint32_t TILES_PER_BATCH   = (N_SYMBOL + SYMBOLS_PER_CORE - 1) / SYMBOLS_PER_CORE;
constexpr uint32_t BATCH_X_ELEMS     = SYMBOLS_PER_CORE * TILE_PQ;
constexpr uint32_t BATCH_TILE        = SYMBOLS_PER_CORE * P * Q;
constexpr uint32_t BATCH_HALF_TILE   = SYMBOLS_PER_CORE * P * P;


constexpr float    INPUT_SCALE       = 1.0f / 256.0f;
constexpr double   SLOT_DURATION_US  = 500.0;
constexpr uint32_t INPUT_GM_INT16_LEN = 65536;
constexpr uint32_t INPUT_INT16_PER_BATCH = 2 * N_SAMPLE_PER_SLOT;
constexpr uint32_t OUTPUT_ELEMS_PER_BATCH = N_SYMBOL * N_FFT;
constexpr uint32_t DEFAULT_BATCH_SIZE = 8;

}
