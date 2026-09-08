



#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"

namespace airan::qam_mod_256_mimo {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t Q_M = 8;
constexpr uint32_t N_DATA_SYMBOLS = 12;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_DATA_RE = N_DATA_SYMBOLS * N_SC_USED;
constexpr uint32_t N_DATA_PAD = 19200;
constexpr uint32_t BLOCK_DIM = 4;
constexpr uint32_t COMPUTE_CORES = 3;
constexpr uint32_t CORE_RE = N_DATA_RE / COMPUTE_CORES;
constexpr uint32_t TILE_ELEMS = 4096;
constexpr uint32_t META_WORDS = 32;
constexpr size_t TILING_BYTES = META_WORDS * sizeof(uint32_t);
constexpr size_t WORKSPACE_BYTES = 128;
constexpr float D_256QAM = 0.07669649888473704f;



constexpr uint32_t QAM_STREAM_TO_NR_BIT[Q_M] = {0, 2, 4, 6, 1, 3, 5, 7};

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




struct QamMod256MimoOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *bits_qam;
    void *d_re;
    void *d_im;
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
    uint32_t tile_elems;
    uint32_t compute_cores;
    uint32_t qam_stream_to_nr_bit[Q_M];
    uint32_t reserved[15];
};
static_assert(sizeof(KernelMetadata) == TILING_BYTES,
              "qam_mod_256_mimo metadata ABI must remain 128 bytes");

constexpr uint32_t META_MAGIC = 0x514d4d31u;

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata);

Status ValidateOpArgs(const QamMod256MimoOpArgsV1 &args);



Status ReferenceModulate(const int16_t *bits_qam,
                         const PuschMimoConfig &config,
                         const PuschMimoLayout &layout,
                         uint16_t *d_re,
                         uint16_t *d_im);

Status Enqueue(const QamMod256MimoOpArgsV1 &args,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes);

constexpr size_t InputElems(uint32_t layers) {
    return static_cast<size_t>(Q_M) * layers * N_DATA_PAD;
}

constexpr size_t OutputElems(uint32_t layers) {
    return static_cast<size_t>(layers) * N_DATA_PAD;
}

}
