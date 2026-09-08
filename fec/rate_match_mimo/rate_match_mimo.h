



#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../../common/pusch_mimo_types.h"

namespace airan::rate_match_mimo {

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
constexpr uint32_t N_CB_BUF = LDPC_N - 2 * LDPC_Z;
constexpr uint32_t BLOCK_DIM = 4;
constexpr uint32_t STREAMS_PER_AIV = Q_M / BLOCK_DIM;

constexpr uint32_t DESC_WORDS = 4;
constexpr uint32_t META_WORDS = 32;
constexpr uint32_t COPY_ELEMS = 4096;
constexpr uint32_t COPY_PAD = COPY_ELEMS + 32;
constexpr uint32_t MAX_CODEWORD_STRIDE = MAX_LAYERS * N_DATA_PAD;
constexpr size_t TILING_BYTES = META_WORDS * sizeof(uint32_t);
constexpr size_t WORKSPACE_BYTES = 128;

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

struct RateMatchFecConfigV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    uint32_t flags;
    uint32_t num_code_blocks;
    uint32_t encoded_stride;
    uint16_t rv_index;
    uint16_t base_graph;
    uint16_t lifting_size;
    uint16_t filler_bits;
    uint32_t reserved[8];
};



struct RateMatchDescriptor {
    uint32_t e;
    uint32_t k0;
    uint32_t ncb;
    uint32_t cw_bit_offset;
};
static_assert(sizeof(RateMatchDescriptor) == DESC_WORDS * sizeof(uint32_t),
              "rate-match descriptor ABI must remain four uint32 words");

struct RateMatchMimoOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *code_blocks;
    void *bits_nr;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    const RateMatchFecConfigV1 *fec;
    uint32_t num_slots;
    void *stream;
};

struct KernelMetadata {
    uint32_t magic;
    uint32_t num_slots;
    uint32_t num_layers;
    uint32_t qm;
    uint32_t num_data_re;
    uint32_t data_stride;
    uint32_t codeword_symbols;
    uint32_t codeword_stride;
    uint32_t num_code_blocks;
    uint32_t encoded_stride;
    uint32_t copy_elems;
    uint32_t max_slots;
    uint32_t reserved[20];
};
static_assert(sizeof(KernelMetadata) == TILING_BYTES,
              "rate_match_mimo metadata ABI must remain 128 bytes");

constexpr uint32_t META_MAGIC = 0x524d4d31u;

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           const RateMatchFecConfigV1 &fec,
                           uint32_t num_slots,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata);

Status BuildRateMatchDescriptors(const PuschMimoConfig &config,
                                 const RateMatchFecConfigV1 &fec,
                                 uint32_t num_slots,
                                 RateMatchDescriptor *descriptors,
                                 size_t descriptor_count);

Status ValidateOpArgs(const RateMatchMimoOpArgsV1 &args);

Status ReferenceRateMatch(const int8_t *code_blocks,
                          const PuschMimoConfig &config,
                          const PuschMimoLayout &layout,
                          const RateMatchFecConfigV1 &fec,
                          uint32_t num_slots,
                          const RateMatchDescriptor *descriptors,
                          int16_t *bits_nr);

Status Enqueue(const RateMatchMimoOpArgsV1 &args,
               const void *descriptors,
               size_t descriptor_bytes,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes);

constexpr size_t InputElems(const RateMatchFecConfigV1 &fec) {
    return static_cast<size_t>(fec.num_code_blocks) * fec.encoded_stride;
}

constexpr size_t OutputElems(uint32_t num_slots, uint32_t layers) {
    return static_cast<size_t>(num_slots) * Q_M * layers * N_DATA_PAD;
}

}
