



#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"

namespace airan::descramble_mimo {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t Q_M = 8;
constexpr uint32_t N_DATA_SYMBOLS = 12;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_DATA_RE = N_DATA_SYMBOLS * N_SC_USED;
constexpr uint32_t N_DATA_PAD = 19200;
constexpr uint32_t MAX_SLOTS = 23;
constexpr uint32_t BLOCK_DIM = 4;
constexpr uint32_t TILE_ELEMS = 32768;
constexpr uint32_t META_WORDS = 32;
constexpr size_t TILING_BYTES = META_WORDS * sizeof(uint32_t);

constexpr size_t WORKSPACE_BYTES = 128;

constexpr uint32_t QAM_STREAM_TO_NR_BIT[Q_M] = {0, 2, 4, 6, 1, 3, 5, 7};
constexpr uint32_t NR_BIT_TO_QAM_STREAM[Q_M] = {0, 4, 1, 5, 2, 6, 3, 7};

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





struct DescrambleMimoOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *llr_qam;
    void *llr_nr;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
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
    uint32_t tile_elems;
    uint32_t max_slots;
    uint32_t reserved[22];
};
static_assert(sizeof(KernelMetadata) == TILING_BYTES,
              "descramble_mimo metadata ABI must remain 128 bytes");

constexpr uint32_t META_MAGIC = 0x444d4d31u;

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           uint32_t num_slots,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata);

Status ValidateOpArgs(const DescrambleMimoOpArgsV1 &args);



Status BuildGoldSign(const PuschMimoConfig &config,
                     const PuschMimoLayout &layout,
                     uint32_t num_slots,
                     int16_t *sign,
                     size_t sign_elems);


Status ReferenceDescramble(const int16_t *llr_qam,
                           const int16_t *sign,
                           const PuschMimoConfig &config,
                           const PuschMimoLayout &layout,
                           uint32_t num_slots,
                           int16_t *llr_nr);



Status Enqueue(const DescrambleMimoOpArgsV1 &args,
               const void *sign,
               size_t sign_bytes,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes);

constexpr size_t BufferElems(uint32_t num_slots, uint32_t layers) {
    return static_cast<size_t>(num_slots) * Q_M * layers * N_DATA_PAD;
}

}
