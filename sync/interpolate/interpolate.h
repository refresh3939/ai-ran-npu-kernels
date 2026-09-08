




















#pragma once
#include <cstddef>
#include <cstdint>

namespace interpolate {


constexpr uint32_t INTERP  = 8;
constexpr uint32_t N_IN    = 153600;
constexpr uint32_t N_OUT   = 1228800;


constexpr uint32_t L_TAP   = 96;
constexpr uint32_t N_PHASE = INTERP;
constexpr uint32_t NPT     = 12;
constexpr uint32_t HALO_IN = 16;


constexpr uint32_t BLOCK_DIM    = 4;
constexpr uint32_t IN_PER_CORE  = N_IN / BLOCK_DIM;
constexpr uint32_t OUT_PER_CORE = N_OUT / BLOCK_DIM;


constexpr uint32_t IN_TILE   = 1280;
constexpr uint32_t N_TILES   = IN_PER_CORE / IN_TILE;
constexpr uint32_t OUT_TILE  = IN_TILE * INTERP;


constexpr uint32_t WINDOW_CPX = IN_TILE + HALO_IN;
constexpr uint32_t LOAD_CPX   = 1344;
constexpr uint32_t LOAD_I16   = 2 * LOAD_CPX;


constexpr uint32_t CMAT_I32   = N_PHASE * IN_TILE;
constexpr uint32_t IDX_LEN    = OUT_TILE;
constexpr uint32_t OUT_I16    = OUT_TILE * 2;


constexpr uint32_t IDX_CHUNK_I32 = 6144;
constexpr uint32_t OUT_CHUNK_I16 = 12288;


constexpr uint32_t IN_GM_I16_LEN  = 2 * N_IN;
constexpr uint32_t IDX_GM_LEN     = IDX_LEN;
constexpr uint32_t OUT_GM_I16_LEN = 2 * N_OUT;




constexpr uint32_t INPUT_Q = 512;

}