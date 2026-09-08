



#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"

namespace airan::rate_dematch_mimo {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t MAX_SLOTS = 23;
constexpr uint32_t Q_M = 8;
constexpr uint32_t N_DATA_SYMBOLS = 12;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_DATA_RE = N_DATA_SYMBOLS * N_SC_USED;
constexpr uint32_t N_DATA_PAD = 19200;


constexpr uint32_t C_NUM = 143;
constexpr uint32_t LDPC_Z = 384;
constexpr uint32_t LDPC_N = 26112;
constexpr uint32_t N_2Z = 2 * LDPC_Z;
constexpr uint32_t N_CB_BUF = LDPC_N - N_2Z;
constexpr uint32_t BLOCK_DIM = 4;
constexpr int16_t LLR_SCALE = 8;
constexpr int16_t LLR_CLIP = 5120;

constexpr uint32_t DESC_WORDS = 4;
constexpr uint32_t DESC_TABLE_WORDS = C_NUM * DESC_WORDS;
constexpr uint32_t DESC_PAD_WORDS = 576;
constexpr uint32_t META_WORDS = 32;
constexpr uint32_t TILE_ELEMS = 8192;
constexpr uint32_t TILE_PAD_ELEMS = TILE_ELEMS + 16;
constexpr size_t TILING_BYTES = META_WORDS * sizeof(uint32_t);
constexpr size_t DESCRIPTOR_BYTES = DESC_PAD_WORDS * sizeof(uint32_t);
constexpr size_t WORKSPACE_BYTES = 128;
constexpr uint32_t META_MAGIC = 0x52444d31u;

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



struct RateMatchDescriptor {
    uint32_t e;
    uint32_t k0;
    uint32_t ncb;
    uint32_t cw_symbol_offset;
};
static_assert(sizeof(RateMatchDescriptor) == DESC_WORDS * sizeof(uint32_t),
              "rate-match descriptor ABI must remain four uint32 words");

struct KernelMetadata {
    uint32_t magic;
    uint32_t num_slots;
    uint32_t num_layers;
    uint32_t qm;
    uint32_t num_data_re;
    uint32_t codeword_symbols;
    uint32_t codeword_stride;
    uint32_t total_coded_bits;
    uint32_t code_blocks;
    uint32_t ldpc_n;
    uint32_t n_2z;
    uint32_t ncb;
    uint32_t llr_scale;
    uint32_t llr_clip;
    uint32_t tile_elems;
    uint32_t max_slots;
    uint32_t descriptor_words;
    uint32_t reserved[15];
};
static_assert(sizeof(KernelMetadata) == TILING_BYTES,
              "rate_dematch_mimo metadata ABI must remain 128 bytes");



struct RateDematchMimoOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *cw_llr;
    void *ldpc_llr;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    uint32_t num_slots;
    void *stream;
};

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           uint32_t num_slots,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           RateMatchDescriptor *descriptors,
                           size_t descriptor_count);

Status ValidateOpArgs(const RateDematchMimoOpArgsV1 &args);



Status ReferenceRateDematch(const int16_t *cw_llr,
                            const PuschMimoConfig &config,
                            const PuschMimoLayout &layout,
                            uint32_t num_slots,
                            const RateMatchDescriptor *descriptors,
                            size_t descriptor_count,
                            int16_t *ldpc_llr);



Status Enqueue(const RateDematchMimoOpArgsV1 &args,
               const void *descriptors,
               size_t descriptor_bytes,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes);

constexpr size_t InputElems(uint32_t num_slots, uint32_t layers) {
    return static_cast<size_t>(num_slots) * Q_M * layers * N_DATA_PAD;
}
constexpr size_t OutputElems() {
    return static_cast<size_t>(C_NUM) * LDPC_N;
}

}
