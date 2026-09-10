/**
 * @file mimo_dmrs_ls.h
 * Natural-layout PUSCH MIMO DMRS LS estimator and LMMSE adapter contract.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include "../../common/pusch_mimo_types.h"

namespace airan::mimo_dmrs_ls {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;
#ifndef MIMO_DMRS_LS_NR
#define MIMO_DMRS_LS_NR 64
#endif
#ifndef MIMO_DMRS_LS_BLOCK_DIM
#define MIMO_DMRS_LS_BLOCK_DIM 4
#endif
constexpr uint32_t NR_CURRENT = MIMO_DMRS_LS_NR;
// PuschMimoConfig carries four DMRS ports.  The detector's shared ABI can have
// a wider layer axis, but this physical estimator supports Rank-1..4 only.
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t N_SYMBOLS = 14;
constexpr uint32_t MAX_DMRS_SYMBOLS = 4;
constexpr uint32_t CURRENT_DMRS_SYMBOLS = 2;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_SC_PAD = 1664;
constexpr uint32_t N_DMRS_RE = 798;
constexpr uint32_t N_DMRS_REF_PAD = 896;
constexpr uint32_t N_PILOT_PAD = 832;
constexpr uint32_t N_OCC_PILOT = 399;
constexpr uint32_t BLOCK_DIM = MIMO_DMRS_LS_BLOCK_DIM;
constexpr uint32_t CE_LAYER_CAPACITY = 16;
constexpr uint32_t CE_RX_GROUP = 8;
constexpr uint32_t CE_RX_GROUPS = NR_CURRENT / CE_RX_GROUP;
constexpr uint32_t CE_COLUMNS = CE_RX_GROUP * CURRENT_DMRS_SYMBOLS;
constexpr uint32_t COUNT_PAD = 16;
constexpr uint32_t META_WORDS = 32;
constexpr size_t TILING_BYTES = META_WORDS * sizeof(uint32_t);
constexpr size_t CE_PACKED_ELEMS = static_cast<size_t>(CE_LAYER_CAPACITY) *
                                   CE_RX_GROUPS * N_PILOT_PAD * CE_COLUMNS;
static_assert(NR_CURRENT == 16 || NR_CURRENT == 32 || NR_CURRENT == 64,
              "DMRS-LS execution RX capacity must be 16, 32, or 64");
static_assert(NR_CURRENT % CE_RX_GROUP == 0,
              "DMRS-LS RX grouping must be exact");
static_assert(BLOCK_DIM >= 1 && BLOCK_DIM <= 4 &&
              NR_CURRENT / BLOCK_DIM == 16,
              "each DMRS-LS core must own exactly 16 RX rows");

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    LMMSE_OBSERVATION_MODEL_MISMATCH = -3,
    CE_OBSERVATION_MODEL_MISMATCH = LMMSE_OBSERVATION_MODEL_MISMATCH,
};

// Canonical per-layer observation model consumed together with pilot_count and
// pilot_sc.  Values intentionally match the device metadata words.
enum ObservationModel : uint32_t {
    COMB2_798 = 0,
    FD_OCC2_399 = 1,
};

using PuschMimoConfig = ::airan::PuschMimoConfig;
using PuschMimoLayout = ::airan::PuschMimoLayout;

// Public buffers are logical shapes; callers may allocate current-profile maxima.
struct MimoDmrsLsOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *rx_grid_re;       // fp16 [NR,NSYM,NSC_PAD]
    const void *rx_grid_im;
    const void *dmrs_ref_re;      // fp16 [L,D,NDMRS_REF_PAD]
    const void *dmrs_ref_im;
    void *h_ls_re;                // fp16 [NR,L,D,NPILOT_PAD]
    void *h_ls_im;
    void *pilot_sc;               // uint16 [L,D,NPILOT_PAD]
    void *pilot_count;            // uint16 [L,D], device backing is COUNT_PAD
    void *noise_var_rx;           // fp16 [NR]
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    void *stream;
};

// Internal 128-byte descriptor consumed as the low-level kernel tiling tail.
// The public chain does not expose this derived resource.
struct KernelMetadata {
    uint32_t magic;
    uint32_t num_rx;
    uint32_t num_layers;
    uint32_t num_dmrs_symbols;
    uint32_t num_symbols;
    uint32_t used_subcarriers;
    uint32_t padded_subcarriers;
    uint32_t dmrs_ref_stride;
    uint32_t pilot_stride;
    uint32_t block_dim;
    uint32_t dmrs_symbols[MAX_DMRS_SYMBOLS];
    uint32_t ports[MAX_LAYERS];
    uint32_t comb_delta[MAX_LAYERS];
    uint32_t observation_model[MAX_LAYERS];
    uint32_t reserved[6];
};
static_assert(sizeof(KernelMetadata) == TILING_BYTES, "metadata ABI must be 128 bytes");

constexpr uint32_t META_MAGIC = 0x4d4c5331u; // "MLS1"

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           uint16_t *pilot_count,
                           uint16_t *pilot_sc);

Status ValidateOpArgs(const MimoDmrsLsOpArgsV1 &args);

// Classifies every layer independently.  Both 798 and 399 are canonical CE
// inputs, including a mixed model such as [399,399,798].
Status DescribeObservationModels(const uint16_t *pilot_count,
                                 uint32_t num_layers,
                                 uint32_t num_dmrs_symbols,
                                 ObservationModel *models);

Status ValidateNaturalLmmseContract(const uint16_t *pilot_count,
                                    const uint16_t *pilot_sc,
                                    uint32_t num_layers,
                                    uint32_t num_dmrs_symbols);

// Legacy private pack only: old channel_est_lmmse_mimo weights have 798 valid
// observations per (layer,DMRS).  Canonical CE callers must use the natural
// contract above and must not use this function as a global compatibility gate.
Status ValidateLmmse798Compatibility(const uint16_t *pilot_count,
                                     uint32_t num_layers,
                                     uint32_t num_dmrs_symbols);

// Retained as a source-compatible spelling for the existing pack adapter.
inline Status ValidateCe64x16Packing(const uint16_t *pilot_count,
                                     uint32_t num_layers,
                                     uint32_t num_dmrs_symbols)
{
    return ValidateLmmse798Compatibility(pilot_count, num_layers,
                                         num_dmrs_symbols);
}

constexpr size_t NaturalElems(uint32_t num_layers) {
    return static_cast<size_t>(NR_CURRENT) * num_layers * CURRENT_DMRS_SYMBOLS * N_PILOT_PAD;
}
constexpr size_t NaturalOffset(uint32_t rx, uint32_t layer, uint32_t dmrs,
                               uint32_t pilot, uint32_t num_layers) {
    return (((static_cast<size_t>(rx) * num_layers + layer) *
             CURRENT_DMRS_SYMBOLS + dmrs) * N_PILOT_PAD + pilot);
}
constexpr size_t PilotOffset(uint32_t layer, uint32_t dmrs, uint32_t pilot) {
    return ((static_cast<size_t>(layer) * CURRENT_DMRS_SYMBOLS + dmrs) *
            N_PILOT_PAD + pilot);
}
constexpr size_t CePackedElems() {
    return CE_PACKED_ELEMS;
}

}  // namespace airan::mimo_dmrs_ls
