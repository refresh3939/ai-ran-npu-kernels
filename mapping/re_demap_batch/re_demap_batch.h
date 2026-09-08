



#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"
#include "re_demap_batch_kernel.h"

namespace airan::re_demap_batch {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;
constexpr uint32_t META_MAGIC = 0x52444231u;

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    LAYOUT_MISMATCH = -3,
    RESOURCE_TOO_SMALL = -4,
    LAUNCH_FAILED = -5,
};




struct ReDemapBatchOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *fft_grid_re;
    const void *fft_grid_im;
    void *rx_grid_re;
    void *rx_grid_im;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    void *stream;
};




struct KernelMetadata {
    uint32_t magic;
    uint32_t num_rx_antennas;
    uint32_t num_symbols;
    uint32_t fft_size;
    uint32_t used_subcarriers;
    uint32_t padded_subcarriers;
    uint32_t block_dim;
    uint32_t symbols_per_tile;
    uint32_t reserved[24];
};
static_assert(sizeof(KernelMetadata) == TILING_BYTES,
              "re_demap_batch metadata ABI must remain 128 bytes");

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata);

Status ValidateOpArgs(const ReDemapBatchOpArgsV1 &args);



Status BuildGatherIndex(uint32_t *index, size_t index_elems);



Status Reference(const uint16_t *fft_grid_re,
                 const uint16_t *fft_grid_im,
                 uint32_t num_rx_antennas,
                 const uint32_t *gather_index,
                 uint16_t *rx_grid_re,
                 uint16_t *rx_grid_im);



Status Enqueue(const ReDemapBatchOpArgsV1 &args,
               const void *gather_index,
               size_t gather_index_bytes,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes);

constexpr size_t InputElems(uint32_t num_rx_antennas) {
    return static_cast<size_t>(num_rx_antennas) * GRID_IN_ELEMS;
}

constexpr size_t OutputElems(uint32_t num_rx_antennas) {
    return static_cast<size_t>(num_rx_antennas) * GRID_OUT_ELEMS;
}

}
