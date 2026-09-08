



#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"

namespace airan::mimo_resource_grid_map {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;


constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t N_SYMBOLS = 14;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_SC_PAD = 1664;
constexpr uint32_t N_GRID = N_SYMBOLS * N_SC_PAD;
constexpr uint32_t CURRENT_DMRS_SYMBOLS = 2;
constexpr uint32_t MAX_DMRS_SYMBOLS = 4;
constexpr uint32_t N_DMRS_RE = 798;
constexpr uint32_t N_DMRS_PAD = 896;
constexpr uint32_t N_DATA_RE = 12 * N_SC_USED;
constexpr uint32_t N_DATA_PAD = 19200;
constexpr uint32_t DATA_GROUP_SYMBOLS = 4;
constexpr uint32_t DATA_GROUP_RE = DATA_GROUP_SYMBOLS * N_SC_USED;
constexpr uint32_t DATA_GROUP_PAD = 6400;
constexpr uint32_t BLOCK_DIM = MAX_LAYERS;
constexpr uint32_t INVALID_OFFSET = 0xffffffffu;
constexpr uint32_t META_WORDS = 32;
constexpr size_t TILING_BYTES = META_WORDS * sizeof(uint32_t);

using ::airan::PuschMimoConfig;
using ::airan::PuschMimoLayout;

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    RESOURCE_MISMATCH = -3,
};



struct MimoResourceGridMapOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *layer_re;
    const void *layer_im;
    const void *dmrs_re;
    const void *dmrs_im;
    void *layer_grid_re;
    void *layer_grid_im;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    void *stream;
};


struct KernelMetadata {
    uint32_t magic;
    uint32_t num_layers;
    uint32_t num_dmrs_symbols;
    uint32_t num_symbols;
    uint32_t used_subcarriers;
    uint32_t padded_subcarriers;
    uint32_t num_data_re;
    uint32_t data_stride;
    uint32_t dmrs_stride;
    uint32_t grid_stride;
    uint32_t invalid_offset;
    uint32_t dmrs_symbol_mask;
    uint32_t reserved[20];
};
static_assert(sizeof(KernelMetadata) == TILING_BYTES, "metadata ABI must be 128 bytes");

constexpr uint32_t META_MAGIC = 0x52474d31u;





Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           uint32_t *data_dst_offset,
                           uint32_t *dmrs_dst_offset);

Status ValidateOpArgs(const MimoResourceGridMapOpArgsV1 &args);




Status ReferenceMap(const uint16_t *layer_re,
                    const uint16_t *layer_im,
                    const uint16_t *dmrs_re,
                    const uint16_t *dmrs_im,
                    const PuschMimoConfig &config,
                    const PuschMimoLayout &layout,
                    const uint32_t *data_dst_offset,
                    const uint32_t *dmrs_dst_offset,
                    uint16_t *layer_grid_re,
                    uint16_t *layer_grid_im);

constexpr size_t DataElems(uint32_t layers) {
    return static_cast<size_t>(layers) * N_DATA_PAD;
}
constexpr size_t DmrsElems(uint32_t layers) {
    return static_cast<size_t>(layers) * CURRENT_DMRS_SYMBOLS * N_DMRS_PAD;
}
constexpr size_t GridElems(uint32_t layers) {
    return static_cast<size_t>(layers) * N_GRID;
}
constexpr size_t DmrsOffsetElems() {
    return static_cast<size_t>(MAX_LAYERS) * CURRENT_DMRS_SYMBOLS * N_DMRS_PAD;
}

}
