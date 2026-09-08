



#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"

namespace airan::qam256_demod_batch {

constexpr uint32_t ABI_VERSION = ::airan::PUSCH_MIMO_ABI_VERSION;
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t Q_M = 8;
constexpr uint32_t N_SYMBOLS = 14;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_SC_GRID_PAD = 1664;
constexpr uint32_t N_DATA_SYMBOLS = 12;
constexpr uint32_t N_SC_LLR_PAD = 1600;
constexpr uint32_t N_DATA_RE = N_DATA_SYMBOLS * N_SC_USED;
constexpr uint32_t LLR_LAYER_STRIDE = N_DATA_SYMBOLS * N_SC_LLR_PAD;
constexpr uint32_t GRID_LAYER_STRIDE = N_SYMBOLS * N_SC_GRID_PAD;

constexpr uint32_t BLOCK_DIM = 4;
constexpr uint32_t DMRS_SYMBOL_MASK = (1u << 2) | (1u << 11);
constexpr size_t TILING_BYTES = 128;


constexpr size_t WORKSPACE_BYTES = 3 * 1024 * 1024;

using PuschMimoConfig = ::airan::PuschMimoConfig;
using PuschMimoLayout = ::airan::PuschMimoLayout;

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    LAYOUT_MISMATCH = -3,
    RESOURCE_TOO_SMALL = -4,
    LAUNCH_FAILED = -5,
};



struct QamDemod256BatchOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *x_re;
    const void *x_im;
    const void *no_eff;
    void *layer_llr;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    void *stream;
};



struct BatchTilingData {
    int32_t n_sc_used;
    int32_t n_sc_pad;
    int32_t n_data_symbols;
    int32_t n_data_re;
    int32_t llr_layer_stride;
    int32_t block_dim;
    int32_t llr_clip;
    int32_t y_scaled_clip;
    int32_t num_layers;
    int32_t grid_layer_stride;
    int32_t physical_data_symbols[N_SYMBOLS];
    int32_t group_size;
    int32_t reserved[7];
};
static_assert(sizeof(BatchTilingData) == TILING_BYTES,
              "QAM batch tiling ABI must remain 128 bytes");

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           BatchTilingData *tiling);

Status ValidateOpArgs(const QamDemod256BatchOpArgsV1 &args);




Status Enqueue(const QamDemod256BatchOpArgsV1 &args,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes);

constexpr size_t GridElems(uint32_t layers) {
    return static_cast<size_t>(layers) * GRID_LAYER_STRIDE;
}

constexpr size_t LlrElems(uint32_t layers) {
    return static_cast<size_t>(layers) * Q_M * LLR_LAYER_STRIDE;
}

}
