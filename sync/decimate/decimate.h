













#pragma once
#include <cstddef>
#include <cstdint>

namespace decimate {


constexpr uint32_t DECIM   = 8;
constexpr uint32_t N_IN    = 1228800;
constexpr uint32_t N_OUT   = 153600;


constexpr uint32_t L_TAP       = 96;
constexpr uint32_t HALO        = L_TAP;
constexpr uint32_t N_ACC       = 4;


constexpr uint32_t BLOCK_DIM    = 4;
constexpr uint32_t OUT_PER_CORE = N_OUT / BLOCK_DIM;


constexpr uint32_t OUT_TILE     = 1280;
constexpr uint32_t IN_PER_TILE  = OUT_TILE * DECIM;
constexpr uint32_t N_TILES      = OUT_PER_CORE / OUT_TILE;
constexpr uint32_t WINDOW_CPX   = IN_PER_TILE + HALO;


constexpr uint32_t LOAD_CPX     = 11264;
constexpr uint32_t LOAD_I16     = 2 * LOAD_CPX;
constexpr uint32_t STREAM_LEN   = LOAD_CPX / DECIM;


constexpr uint32_t LOAD_CHUNK_I16 = 12288;


constexpr uint32_t IN_GM_I16_LEN  = 2 * N_IN;
constexpr uint32_t OUT_GM_I16_LEN = 2 * N_OUT;




constexpr uint32_t INPUT_Q = 512;

}