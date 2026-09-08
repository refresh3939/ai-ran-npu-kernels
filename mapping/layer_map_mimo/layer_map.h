



#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"

namespace airan::layer_map {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t Q_M = 8;
constexpr uint32_t N_DATA_RE = 19152;
constexpr uint32_t N_DATA_PAD = 19200;
constexpr uint32_t TILE_SYMBOLS = 4096;
constexpr uint32_t TAIL_SYMBOLS = N_DATA_RE % TILE_SYMBOLS;
constexpr uint32_t BLOCK_DIM = MAX_LAYERS;
constexpr uint32_t META_WORDS = 32;
constexpr uint32_t INDEX_ELEMS_PER_LAYER = TILE_SYMBOLS;
constexpr uint32_t MAX_INDEX_ELEMS = MAX_LAYERS * INDEX_ELEMS_PER_LAYER;
constexpr size_t UB_WORKING_BYTES =
    META_WORDS * sizeof(uint32_t) +
    2 * MAX_LAYERS * TILE_SYMBOLS * sizeof(uint16_t) +
    2 * TILE_SYMBOLS * sizeof(uint16_t) +
    INDEX_ELEMS_PER_LAYER * sizeof(uint32_t);
static_assert(UB_WORKING_BYTES <= 128 * 1024,
              "layer_map UB working set exceeds dav_m200 capacity");
constexpr size_t TILING_BYTES =
    (META_WORDS + MAX_INDEX_ELEMS) * sizeof(uint32_t);

using PuschMimoConfig = ::airan::PuschMimoConfig;
using PuschMimoLayout = ::airan::PuschMimoLayout;

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    LAYOUT_MISMATCH = -3,
};



struct LayerMapOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *d_re;
    const void *d_im;
    void *layer_re;
    void *layer_im;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    void *stream;
};



struct KernelMetadata {
    uint32_t magic;
    uint32_t num_layers;
    uint32_t num_data_re;
    uint32_t data_stride;
    uint32_t codeword_symbols;
    uint32_t codeword_stride;
    uint32_t tile_symbols;
    uint32_t index_elems_per_layer;
    uint32_t reserved[24];
};
static_assert(sizeof(KernelMetadata) == META_WORDS * sizeof(uint32_t),
              "metadata ABI must be 128 bytes");

constexpr uint32_t META_MAGIC = 0x4c4d5031u;



Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           uint32_t *gather_index);

Status ValidateOpArgs(const LayerMapOpArgsV1 &args);



Status ReferenceMap(const uint16_t *d_re,
                    const uint16_t *d_im,
                    const PuschMimoConfig &config,
                    const PuschMimoLayout &layout,
                    uint16_t *layer_re,
                    uint16_t *layer_im);

constexpr size_t CodewordElems(uint32_t layers) {
    return static_cast<size_t>(layers) * N_DATA_PAD;
}
constexpr size_t LayerElems(uint32_t layers) {
    return static_cast<size_t>(layers) * N_DATA_PAD;
}

}
