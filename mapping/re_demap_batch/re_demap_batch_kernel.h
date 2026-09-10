/**
 * @file re_demap_batch_kernel.h
 * Constants shared by the batched RE demapper kernel and its host adapter.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace airan::re_demap_batch {

constexpr uint32_t N_FFT = 2048;
constexpr uint32_t N_SYMBOLS = 14;
constexpr uint32_t N_RB = 133;
constexpr uint32_t N_SC_USED = N_RB * 12;       // 1596
constexpr uint32_t N_SC_PAD = 1664;
constexpr uint32_t GRID_IN_ELEMS = N_SYMBOLS * N_FFT;
constexpr uint32_t GRID_OUT_ELEMS = N_SYMBOLS * N_SC_PAD;

constexpr uint32_t BLOCK_DIM = 4;
constexpr uint32_t SYMBOLS_PER_TILE = 4;
constexpr uint32_t TILES_PER_BATCH =
    (N_SYMBOLS + SYMBOLS_PER_TILE - 1) / SYMBOLS_PER_TILE;
constexpr uint32_t MAX_RX_ANTENNAS = 64;
constexpr uint32_t DEFAULT_BATCH_SIZE = 8;

constexpr uint32_t META_WORDS = 32;
constexpr size_t TILING_BYTES = META_WORDS * sizeof(uint32_t);
// Kept as valid tail storage for combined-build HAVE_WORKSPACE compatibility.
constexpr size_t WORKSPACE_BYTES = 128;

}  // namespace airan::re_demap_batch
