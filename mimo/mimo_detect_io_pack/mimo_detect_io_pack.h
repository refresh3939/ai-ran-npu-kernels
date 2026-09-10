/**
 * @file mimo_detect_io_pack.h
 * Fixed-profile adapter between the natural PUSCH MIMO grid layout and the
 * physical ABI consumed by mimo_detect_bri_batch.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"

namespace airan::mimo_detect_io_pack {

constexpr uint16_t ABI_VERSION = 1;
#ifndef MIMO_DETECT_IO_NR
#define MIMO_DETECT_IO_NR 64
#endif
#ifndef MIMO_DETECT_IO_NL
#define MIMO_DETECT_IO_NL 16
#endif
constexpr uint32_t NR = MIMO_DETECT_IO_NR;
constexpr uint32_t NL = MIMO_DETECT_IO_NL;
constexpr uint32_t MAX_PUSCH_LAYERS = 4;
constexpr uint32_t MAX_ALLOCATIONS = ::airan::PUSCH_MIMO_MAX_ALLOCATIONS;
constexpr uint32_t STANDARD_CHAIN_ALLOCATIONS = 1;
constexpr uint32_t N_SYMBOL = 14;
constexpr uint32_t N_SC_PAD = 1664;
constexpr uint32_t N_RE = N_SYMBOL * N_SC_PAD;
constexpr uint32_t BLOCK_DIM = 4;
constexpr uint32_t RE_PER_CORE = N_RE / BLOCK_DIM;
constexpr uint32_t RE_TILE = 16;
constexpr uint32_t RX_TILE = NR >= 32 ? 32 : 16;
constexpr uint32_t RX_GROUP = 16;
constexpr uint32_t TILING_BYTES = 32;
constexpr uint32_t META_MAGIC = 0x4d495031u;  // "MIP1"
constexpr float NOISE_MEAN_SCALE =
    NR == 16 ? 0.0625f : (NR == 32 ? 0.03125f : 0.015625f);

constexpr size_t RX_ELEMS = static_cast<size_t>(NR) * N_RE;
constexpr size_t H_ELEMS = static_cast<size_t>(NR) * NL * N_RE;
constexpr size_t PACKED_ELEMS = static_cast<size_t>(N_RE) * NR * NL;
constexpr size_t NOISE_ELEMS = NR;
constexpr size_t NO_ELEMS = N_RE;

static_assert(N_RE % BLOCK_DIM == 0, "REs must split exactly over four AI Cores");
static_assert(RE_PER_CORE % RE_TILE == 0, "each core range must use full 16-RE tiles");
static_assert(NR % RX_TILE == 0 && RX_TILE % RX_GROUP == 0,
              "receiver tiling must be exact");
static_assert(NR == 16 || NR == 32 || NR == 64,
              "io-pack execution RX capacity must be 16, 32, or 64");
static_assert(NL == 16, "BRI physical layer width must match one fp16 block");
static_assert(NL == ::airan::PUSCH_MIMO_MAX_DETECT_LAYERS,
              "BRI width must match the common detector layer capacity");

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    PLAN_MISMATCH = -3,
    RESOURCE_TOO_SMALL = -4,
    LAUNCH_FAILED = -5,
};

using PuschMimoConfig = ::airan::PuschMimoConfig;
using MimoDetectAllocation = ::airan::MimoDetectAllocation;
using MimoDetectLayerPlan = ::airan::MimoDetectLayerPlan;

struct KernelMetadata {
    uint32_t magic;
    uint32_t active_layers;
    uint32_t num_rx_antennas;
    uint32_t layer_capacity;
    uint32_t n_re;
    uint32_t abi_version;
    uint32_t reserved[2];
};
static_assert(sizeof(KernelMetadata) == TILING_BYTES,
              "kernel metadata must remain one 32-byte DMA block");

// Public wrapper arguments. Complex tensors use separate fp16 planes:
//   rx_grid_* [NR,N_SYMBOL,N_SC_PAD]
//   h_grid_*  [NR,NL,N_SYMBOL,N_SC_PAD], where NL=16 is storage alignment;
//             only [0,num_layers) with num_layers<=4 is active
//   noise_var_rx [NR]
//   hrm_*     [N_RE,NR,NL]
//   yvpad_*   [N_RE,NR,NL]
//   no        [N_RE]
struct MimoDetectIoPackOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *rx_grid_re;
    const void *rx_grid_im;
    const void *h_grid_re;
    const void *h_grid_im;
    const void *noise_var_rx;
    void *hrm_re;
    void *hrm_im;
    void *yvpad_re;
    void *yvpad_im;
    void *no;
    const PuschMimoConfig *configs;
    uint16_t num_configs;
    uint16_t reserved;
    const MimoDetectLayerPlan *layer_plan;
    void *stream;
};

Status BuildCurrentProfile(const PuschMimoConfig *configs,
                           size_t num_configs,
                           const MimoDetectLayerPlan &layer_plan,
                           KernelMetadata *metadata);

Status ValidateOpArgs(const MimoDetectIoPackOpArgsV1 &args);

// Internal runtime-adapter entry. The caller owns a device-side metadata
// buffer populated by BuildCurrentProfile. The function enqueues exactly one
// four-core kernel and never synchronizes the caller's stream.
Status Enqueue(const MimoDetectIoPackOpArgsV1 &args,
               const void *metadata,
               size_t metadata_bytes);

}  // namespace airan::mimo_detect_io_pack
