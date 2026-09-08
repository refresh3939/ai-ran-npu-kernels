



#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"

namespace airan::layer_demap {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t Q_M = 8;
constexpr uint32_t N_DATA_SYMBOLS = 12;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_SC_LLR_PAD = 1600;
constexpr uint32_t N_DATA_RE = 19152;
constexpr uint32_t N_DATA_PAD = 19200;
constexpr uint32_t INPUT_LAYER_Q_STRIDE = N_DATA_SYMBOLS * N_SC_LLR_PAD;
constexpr uint32_t SYMBOLS_PER_GROUP = 4;
constexpr uint32_t NUM_GROUPS = N_DATA_SYMBOLS / SYMBOLS_PER_GROUP;
constexpr uint32_t GROUP_DATA_RE = SYMBOLS_PER_GROUP * N_SC_USED;
constexpr uint32_t GATHER_CHUNK = 128;
constexpr uint32_t GATHER_TAIL = GROUP_DATA_RE % GATHER_CHUNK;
constexpr uint32_t BLOCK_DIM = 4;
constexpr uint32_t META_WORDS = 32;
constexpr uint32_t MAX_INDEX_ELEMS = MAX_LAYERS * GATHER_CHUNK;
constexpr uint32_t SOURCE_ELEMS = MAX_LAYERS * GROUP_DATA_RE;
constexpr uint32_t OUTPUT_ELEMS = MAX_LAYERS * GROUP_DATA_RE;
constexpr size_t UB_WORKING_BYTES =
    META_WORDS * sizeof(uint32_t) +
    N_SC_LLR_PAD * sizeof(int16_t) +
    SOURCE_ELEMS * sizeof(int16_t) +
    OUTPUT_ELEMS * sizeof(int16_t) +
    MAX_INDEX_ELEMS * sizeof(uint32_t);
static_assert(UB_WORKING_BYTES <= 128 * 1024,
              "layer_demap UB working set exceeds dav_m200 capacity");
constexpr size_t TILING_BYTES =
    (META_WORDS + MAX_INDEX_ELEMS) * sizeof(uint32_t);

static_assert(INPUT_LAYER_Q_STRIDE == N_DATA_PAD,
              "current physical QAM and compact storage strides must match");
static_assert(N_DATA_RE == N_DATA_SYMBOLS * N_SC_USED,
              "data RE geometry must be self-consistent");
static_assert(BLOCK_DIM == 4, "layer_demap must launch exactly four AI Cores");

using PuschMimoConfig = ::airan::PuschMimoConfig;
using PuschMimoLayout = ::airan::PuschMimoLayout;

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    LAYOUT_MISMATCH = -3,
};




struct LayerDemapOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *layer_llr;
    void *cw_llr;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    void *stream;
};



struct KernelMetadata {
    uint32_t magic;
    uint32_t num_layers;
    uint32_t qm;
    uint32_t num_data_re;
    uint32_t data_stride;
    uint32_t codeword_symbols;
    uint32_t codeword_stride;
    uint32_t input_symbol_stride;
    uint32_t symbols_per_group;
    uint32_t group_data_re;
    uint32_t gather_chunk;
    uint32_t gather_index_elems;
    uint32_t reserved[20];
};
static_assert(sizeof(KernelMetadata) == META_WORDS * sizeof(uint32_t),
              "metadata ABI must be 128 bytes");

constexpr uint32_t META_MAGIC = 0x4c444d31u;



Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           uint32_t *gather_index);

Status ValidateOpArgs(const LayerDemapOpArgsV1 &args);



Status ReferenceDemap(const int16_t *layer_llr,
                      const PuschMimoConfig &config,
                      const PuschMimoLayout &layout,
                      int16_t *cw_llr);

constexpr size_t LayerElems(uint32_t layers) {
    return static_cast<size_t>(layers) * Q_M * INPUT_LAYER_Q_STRIDE;
}
constexpr size_t CodewordElems(uint32_t layers) {
    return static_cast<size_t>(Q_M) * layers * N_DATA_PAD;
}

}
