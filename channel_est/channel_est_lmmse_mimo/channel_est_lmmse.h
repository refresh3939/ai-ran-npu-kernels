






#pragma once

#include <cstddef>
#include <cstdint>
#include "../../common/pusch_mimo_types.h"

namespace airan::channel_est_lmmse {

#ifndef CE_NR
#define CE_NR 64
#endif
#ifndef CE_NL
#define CE_NL 2
#endif
#ifndef CE_RANK
#define CE_RANK 96
#endif
#ifndef CE_BLOCK_DIM
#define CE_BLOCK_DIM 4
#endif

constexpr uint32_t NR = CE_NR;
constexpr uint32_t NL = CE_NL;
constexpr uint32_t MAX_ACTIVE_LAYERS = 4;
constexpr uint32_t DETECTOR_LAYERS = ::airan::PUSCH_MIMO_MAX_DETECT_LAYERS;
constexpr uint32_t N_SYMBOL = 14;
constexpr uint32_t N_DMRS_SYMBOL = 2;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_SC_PAD = 1664;
constexpr uint32_t N_PILOT = N_SC_USED / 2;
constexpr uint32_t N_PILOT_PAD = 832;
constexpr uint32_t N_OCC_PILOT = N_PILOT / 2;
constexpr uint32_t PILOT_COUNT_PAD = 16;
constexpr uint32_t RANK = CE_RANK;

constexpr uint32_t RX_GROUP = 8;
constexpr uint32_t N_RX_GROUP = NR / RX_GROUP;
constexpr uint32_t NCOL = RX_GROUP * N_DMRS_SYMBOL;
constexpr uint32_t BLOCK_DIM = CE_BLOCK_DIM;
constexpr uint32_t SC_TILE = 16;
constexpr uint32_t N_SC_TILE = (N_SC_USED + SC_TILE - 1) / SC_TILE;
constexpr uint32_t SC_TILE_PER_CORE = N_SC_TILE / BLOCK_DIM;
constexpr uint32_t RANK_TILE = 16;
constexpr uint32_t N_RANK_TILE = RANK / RANK_TILE;
constexpr uint32_t PILOT_K_BLOCK = N_PILOT_PAD / 16;















constexpr size_t B_LAYER_ELEMS = static_cast<size_t>(RANK) * N_PILOT_PAD;
constexpr size_t B_ELEMS = static_cast<size_t>(NL) * B_LAYER_ELEMS;
constexpr size_t HLS_ELEMS = static_cast<size_t>(NL) * N_RX_GROUP * N_PILOT_PAD * NCOL;
#if defined(CE_CUBE_TIME_FUSED) && CE_CUBE_TIME_FUSED
constexpr size_t A_LAYER_ELEMS = static_cast<size_t>(N_SC_TILE) * N_SYMBOL * RANK * SC_TILE;
constexpr size_t WT_LAYER_ELEMS = A_LAYER_ELEMS;
constexpr size_t A_ELEMS = static_cast<size_t>(NL) * A_LAYER_ELEMS;
constexpr size_t WT_ELEMS = A_ELEMS;
#elif defined(CE_CUBE_TIME_POST_GEMM) && CE_CUBE_TIME_POST_GEMM
constexpr size_t A_LAYER_ELEMS = static_cast<size_t>(N_SC_TILE) * RANK * SC_TILE;
constexpr size_t WT_LAYER_ELEMS = static_cast<size_t>(N_SC_TILE) * 14 * 32 * 16;
constexpr size_t A_ELEMS = static_cast<size_t>(NL) * A_LAYER_ELEMS;
constexpr size_t WT_ELEMS = static_cast<size_t>(NL) * WT_LAYER_ELEMS;
#else
constexpr size_t A_LAYER_ELEMS = static_cast<size_t>(N_SC_TILE) * RANK * SC_TILE;
constexpr size_t WT_LAYER_ELEMS = static_cast<size_t>(N_SC_TILE) * N_SYMBOL * N_DMRS_SYMBOL * SC_TILE;
constexpr size_t A_ELEMS = static_cast<size_t>(NL) * A_LAYER_ELEMS;
constexpr size_t WT_ELEMS = static_cast<size_t>(NL) * WT_LAYER_ELEMS;
#endif
constexpr size_t T_ELEMS = static_cast<size_t>(NL) * N_RX_GROUP * RANK * NCOL;
constexpr size_t OUT_ELEMS = static_cast<size_t>(NR) * NL * N_SYMBOL * N_SC_PAD;
constexpr size_t NATURAL_HLS_ELEMS = static_cast<size_t>(NR) * NL * N_DMRS_SYMBOL * N_PILOT_PAD;
constexpr size_t PADDED_OUT_ELEMS = static_cast<size_t>(NR) * DETECTOR_LAYERS * N_SYMBOL * N_SC_PAD;
constexpr size_t PACK_INDEX_ELEMS = static_cast<size_t>(N_PILOT_PAD) * NCOL;

constexpr size_t TILING_BYTES = 256;
constexpr size_t MIN_SYNC_WORKSPACE_BYTES = 8192;
#define CE_STRINGIFY_IMPL(x) #x
#define CE_STRINGIFY(x) CE_STRINGIFY_IMPL(x)
constexpr const char *CASE_NAME =
    "case_0_m" CE_STRINGIFY(CE_NR) "_k" CE_STRINGIFY(CE_NL) "_r" CE_STRINGIFY(CE_RANK);

static_assert(NR == 8 || NR == 16 || NR == 32 || NR == 64,
              "native receive-antenna profiles are NR=8,16,32,64");
static_assert(NL >= 1 && NL <= 4,
              "public MIMO profiles are active L=1,2,3,4");
static_assert(NR % RX_GROUP == 0, "RX grouping must be exact");
static_assert(NCOL == 16, "one Cube tile holds 8 Rx x 2 DMRS symbols");
static_assert(RANK % 16 == 0 && N_PILOT_PAD % 16 == 0, "Cube K dimensions must align");
static_assert(RANK >= 16 && RANK <= 128, "RANK must be a 16-aligned value in [16,128]");
static_assert(BLOCK_DIM == 1 || BLOCK_DIM == 2 || BLOCK_DIM == 4,
              "fixed 100-SC-tile grid supports blockDim 1, 2, or 4");
static_assert(N_SC_TILE % BLOCK_DIM == 0, "computed SC tiles must split evenly across cores");
static_assert(N_SC_TILE * SC_TILE <= N_SC_PAD, "computed SC tiles must fit padded output");

constexpr uint32_t WEIGHT_MODEL_MAGIC = 0x43455731u;

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    CE_OBSERVATION_MODEL_MISMATCH = -3,
    CE_WEIGHT_MODEL_MISMATCH = -4,
    CE_BUILD_PROFILE_MISMATCH = -5,
};

using PuschMimoConfig = ::airan::PuschMimoConfig;
using PuschMimoLayout = ::airan::PuschMimoLayout;

struct LmmseWeightModelV1 {
    uint32_t magic;
    uint16_t abi_version;
    uint16_t struct_size;
    uint16_t num_layers;
    uint16_t num_dmrs_symbols;
    uint16_t rank;
    uint16_t pilot_stride;
    uint16_t pilot_count[MAX_ACTIVE_LAYERS][N_DMRS_SYMBOL];
    uint64_t pilot_sc_hash[MAX_ACTIVE_LAYERS][N_DMRS_SYMBOL];
    uint32_t reserved[8];
};




struct ChannelEstLmmseOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *h_ls_re;
    const void *h_ls_im;
    const void *pilot_count;
    const void *pilot_sc;
    const uint16_t *pilot_count_host;
    const uint16_t *pilot_sc_host;
    const LmmseWeightModelV1 *weight_model;
    void *h_grid_re;
    void *h_grid_im;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    void *stream;
};

uint64_t PilotScHash(const uint16_t *pilot_sc, uint32_t count);
Status BuildWeightModel(const PuschMimoConfig &config,
                        const PuschMimoLayout &layout,
                        const uint16_t *pilot_count,
                        const uint16_t *pilot_sc,
                        LmmseWeightModelV1 *model);
Status ValidateObservationModel(const PuschMimoConfig &config,
                                const PuschMimoLayout &layout,
                                const uint16_t *pilot_count,
                                const uint16_t *pilot_sc,
                                const LmmseWeightModelV1 &model);
Status ValidateOpArgs(const ChannelEstLmmseOpArgsV1 &args);

}
