



#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"

namespace airan::mimo_data_extract {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t N_SYMBOLS = 14;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_SC_PAD = 1664;
constexpr uint32_t N_GRID = N_SYMBOLS * N_SC_PAD;
constexpr uint32_t CURRENT_DMRS_SYMBOLS = 2;
constexpr uint32_t N_DATA_SYMBOLS = N_SYMBOLS - CURRENT_DMRS_SYMBOLS;
constexpr uint32_t N_DATA_RE = N_DATA_SYMBOLS * N_SC_USED;
constexpr uint32_t N_DATA_PAD = 19200;
constexpr uint32_t GROUP_SYMBOLS = 4;
constexpr uint32_t GROUP_RE = GROUP_SYMBOLS * N_SC_USED;
constexpr uint32_t NUM_GROUPS = N_DATA_SYMBOLS / GROUP_SYMBOLS;
constexpr uint32_t OUTPUT_TAIL = N_DATA_PAD - N_DATA_RE;
constexpr uint32_t BLOCK_DIM = MAX_LAYERS;
constexpr uint32_t META_WORDS = 32;
constexpr size_t TILING_BYTES = META_WORDS * sizeof(uint32_t);

using PuschMimoConfig = ::airan::PuschMimoConfig;
using PuschMimoLayout = ::airan::PuschMimoLayout;



enum TensorLayout : uint32_t {
    TENSOR_LAYOUT_UNSPECIFIED = 0,
    TENSOR_LAYOUT_FULL_GRID_FP16 = 1,
    TENSOR_LAYOUT_COMPACT_DATA_FP16 = 2,
};

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    LAYOUT_MISMATCH = -3,
    TENSOR_LAYOUT_MISMATCH = -4,
};




struct MimoDataExtractOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *xhat_re;
    const void *xhat_im;
    const void *no_eff;
    void *data_re;
    void *data_im;
    void *data_no_eff;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    void *stream;
    TensorLayout input_layout;
    TensorLayout output_layout;
};




struct KernelMetadata {
    uint32_t magic;
    uint32_t num_layers;
    uint32_t num_symbols;
    uint32_t used_subcarriers;
    uint32_t padded_subcarriers;
    uint32_t num_data_symbols;
    uint32_t num_data_re;
    uint32_t data_stride;
    uint32_t grid_stride;
    uint32_t dmrs_symbol_mask;
    uint32_t data_symbol_to_grid[N_DATA_SYMBOLS];
    uint32_t reserved[META_WORDS - 10 - N_DATA_SYMBOLS];
};
static_assert(sizeof(KernelMetadata) == TILING_BYTES,
              "mimo_data_extract metadata ABI must be 128 bytes");

constexpr uint32_t META_MAGIC = 0x4d444531u;

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata);

Status ValidateOpArgs(const MimoDataExtractOpArgsV1 &args);




Status ValidateConsumerLayout(TensorLayout consumer_input_layout);



Status ReferenceExtract(const uint16_t *xhat_re,
                        const uint16_t *xhat_im,
                        const uint16_t *no_eff,
                        const PuschMimoConfig &config,
                        const PuschMimoLayout &layout,
                        uint16_t *data_re,
                        uint16_t *data_im,
                        uint16_t *data_no_eff);

constexpr size_t GridElems(uint32_t layers) {
    return static_cast<size_t>(layers) * N_GRID;
}

constexpr size_t DataElems(uint32_t layers) {
    return static_cast<size_t>(layers) * N_DATA_PAD;
}

}
