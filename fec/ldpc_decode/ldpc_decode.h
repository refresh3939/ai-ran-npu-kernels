/**
 * @file ldpc_decode.h
 * @brief Constants for the BG1 Z=384 normalized min-sum decoder.
 *
 * Pipeline:
 *   1. Run LDPC_MAX_ITER layered NMS iterations.
 *   2. Hard-decision the first K posterior LLRs.
 *
 * Inputs:
 *   lam_in           int16[N_CB, NFULL*Z]
 *   packed_bc/s      int16[MB, MAX_DEG]
 *   edge_offsets     int32[MB+1]
 *   degrees          int16[MB]
 * Internal (iter buffer, host inits to 0):
 *   prev_msg         int8 [N_CB, TOTAL_EDGES*Z] in-place message buffer
 * Outputs:
 *   lam_out          int16[N_CB, NFULL*Z]       final lam
 *   decoded_bits     int8 [N_CB, K]             0/1 hard-decided bits
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace airan {

constexpr uint32_t LDPC_C_NUM     = 143;

constexpr uint32_t LDPC_Z         = 384;
constexpr uint32_t LDPC_KB        = 22;
constexpr uint32_t LDPC_MB        = 46;
constexpr uint32_t LDPC_NFULL     = LDPC_KB + LDPC_MB;
constexpr uint32_t LDPC_K         = LDPC_KB * LDPC_Z;
constexpr uint32_t LDPC_N_RAW     = 25344;

constexpr uint32_t BATCH_ROWS     = 128;
constexpr uint32_t PAD            = 128;
constexpr uint32_t Z_PAD          = LDPC_Z + PAD;

constexpr uint32_t SHIFT_ELEMS    = LDPC_MB * LDPC_NFULL;

constexpr uint32_t LDPC_MAX_DEG     = 19;
constexpr uint32_t LDPC_TOTAL_EDGES = 316;

constexpr uint32_t PACKED_ELEMS_TOTAL = LDPC_MB * LDPC_MAX_DEG;
constexpr size_t   PACKED_BYTES       = PACKED_ELEMS_TOTAL * sizeof(int16_t);

constexpr int16_t  Q_SCALE        = 256;
constexpr int16_t  LLR_CLIP_FX    = 20 * Q_SCALE;

constexpr float    NMS_ALPHA_F    = 0.75f;

constexpr uint32_t LDPC_MAX_ITER  = 3;

constexpr size_t LAM_ELEMS_PER_CB  = (size_t)LDPC_NFULL * LDPC_Z;
constexpr size_t LAM_BYTES         = (size_t)LDPC_C_NUM * LAM_ELEMS_PER_CB * sizeof(int16_t);

constexpr size_t PREV_ELEMS_PER_CB = (size_t)LDPC_TOTAL_EDGES * LDPC_Z;
constexpr size_t PREV_BYTES        = (size_t)LDPC_C_NUM * PREV_ELEMS_PER_CB * sizeof(int8_t);
constexpr int16_t PREV_CLIP_FX     = 127;

// Kept in the public ABI because rx_chain already passes this workspace.
constexpr size_t LAM_SCRATCH_PER_AIV = LAM_ELEMS_PER_CB * sizeof(int16_t);
constexpr size_t LAM_SCRATCH_BYTES   = 4 * LAM_SCRATCH_PER_AIV;

constexpr size_t BITS_ELEMS_PER_CB = (size_t)LDPC_K;
constexpr size_t BITS_BYTES        = (size_t)LDPC_C_NUM * BITS_ELEMS_PER_CB * sizeof(int8_t);

constexpr size_t TILING_TOTAL_SIZE = 128;
constexpr size_t WS_TOTAL          = 1 * 1024 * 1024;
constexpr size_t DBG_WORDS_PER_CB  = 64;

}  // namespace airan
