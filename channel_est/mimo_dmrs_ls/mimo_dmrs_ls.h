



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



enum ObservationModel : uint32_t {
    COMB2_798 = 0,
    FD_OCC2_399 = 1,
};

using PuschMimoConfig = ::airan::PuschMimoConfig;
using PuschMimoLayout = ::airan::PuschMimoLayout;


struct MimoDmrsLsOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *rx_grid_re;
    const void *rx_grid_im;
    const void *dmrs_ref_re;
    const void *dmrs_ref_im;
    void *h_ls_re;
    void *h_ls_im;
    void *pilot_sc;
    void *pilot_count;
    void *noise_var_rx;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    void *stream;
};



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

constexpr uint32_t META_MAGIC = 0x4d4c5331u;

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           uint16_t *pilot_count,
                           uint16_t *pilot_sc);

Status ValidateOpArgs(const MimoDmrsLsOpArgsV1 &args);



Status DescribeObservationModels(const uint16_t *pilot_count,
                                 uint32_t num_layers,
                                 uint32_t num_dmrs_symbols,
                                 ObservationModel *models);

Status ValidateNaturalLmmseContract(const uint16_t *pilot_count,
                                    const uint16_t *pilot_sc,
                                    uint32_t num_layers,
                                    uint32_t num_dmrs_symbols);




Status ValidateLmmse798Compatibility(const uint16_t *pilot_count,
                                     uint32_t num_layers,
                                     uint32_t num_dmrs_symbols);


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

}
