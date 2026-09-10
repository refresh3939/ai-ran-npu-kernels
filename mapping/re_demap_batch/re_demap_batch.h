/**
 * @file re_demap_batch.h
 * Stable chain-facing contract for the batched PUSCH RE demapper.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"
#include "re_demap_batch_kernel.h"

namespace airan::re_demap_batch {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;
constexpr uint32_t META_MAGIC = 0x52444231u;  // "RDB1"

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    LAYOUT_MISMATCH = -3,
    RESOURCE_TOO_SMALL = -4,
    LAUNCH_FAILED = -5,
};

// Standard chain ABI:
//   fft_grid_* [NR,14,2048] in upstream OFDM stage-4 [32,64] memory order
//   rx_grid_*  [NR,14,1664] in natural used-subcarrier order
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

// Adapter-owned descriptor copied to the low-level tiling tail. The device
// kernel takes NR explicitly today; retaining this descriptor freezes a stable
// resource ABI for later profile extensions.
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

// Build the shared fp16 byte-offset table. Only [0,1596) is consumed as
// gather data; the output padding is produced as explicit zeros by the kernel.
Status BuildGatherIndex(uint32_t *index, size_t index_elems);

// Bit-exact host reference with zero-filled [1596,1664) for every antenna and
// OFDM symbol. Input and output must not alias.
Status Reference(const uint16_t *fft_grid_re,
                 const uint16_t *fft_grid_im,
                 uint32_t num_rx_antennas,
                 const uint32_t *gather_index,
                 uint16_t *rx_grid_re,
                 uint16_t *rx_grid_im);

// Enqueue one native runtime-batch kernel. gather_index, workspace and tiling
// are adapter-owned cached device resources. This function does not synchronize.
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

}  // namespace airan::re_demap_batch
