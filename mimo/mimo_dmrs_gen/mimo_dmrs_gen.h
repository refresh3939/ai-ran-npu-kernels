/**
 * @file mimo_dmrs_gen.h
 * Standard NR PUSCH Type-1 MIMO DMRS contract and current kernel ABI metadata.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"

namespace airan::mimo_dmrs_gen {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;
// The canonical detector ABI allows larger tensors; this TX profile and the
// PuschMimoConfig DMRS-port array intentionally support Rank 1..4.
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t N_SYMBOLS = 14;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_SC_PAD = 1664;
constexpr uint32_t CURRENT_DMRS_SYMBOLS = 2;
constexpr uint32_t MAX_DMRS_SYMBOLS = 4;
constexpr uint32_t N_DMRS_RE = 798;
constexpr uint32_t N_DMRS_PAD = 896;
constexpr uint32_t N_DMRS_PLANE = 2 * N_DMRS_PAD;

constexpr uint32_t GOLD_NC = 1600;
constexpr uint32_t GOLD_NBITS = 31;
constexpr uint32_t MAT_LEN = GOLD_NBITS * N_DMRS_PLANE;
constexpr uint32_t CINIT_PAD = 16;
constexpr uint32_t OUT_DBG_LEN = 8;
constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t TILING_WORDS = 32;
constexpr size_t TILING_BYTES = TILING_WORDS * sizeof(uint32_t);

using ::airan::PuschMimoConfig;
using ::airan::PuschMimoLayout;

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    RESOURCE_ERROR = -3,
    LAUNCH_FAILED = -4,
};

// Public chain arguments. The logical outputs are [L,D,896]. The current
// low-level kernel writes a [4,2,896] physical buffer; the runtime adapter
// owns that compatibility detail and exposes only the active logical prefix.
struct MimoDmrsGenOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    void *dmrs_re;
    void *dmrs_im;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    void *stream;
};

// Opaque adapter state. Create/Destroy must run after ACL initialization and
// with the intended device context current. A handle has one private staging
// slot: synchronize its stream before another Enqueue or DestroyRuntime.
struct MimoDmrsGenRuntimeV1;

// Private 128-byte descriptor carried in the existing trailing tiling slot.
// This keeps the low-level parameter order unchanged.
struct KernelMetadata {
    uint32_t magic;
    uint32_t num_layers;
    uint32_t num_dmrs_symbols;
    uint32_t dmrs_symbols[MAX_DMRS_SYMBOLS];
    uint32_t port_indices[MAX_LAYERS];
    uint32_t comb_delta[MAX_LAYERS];
    uint32_t wf_odd_negative[MAX_LAYERS];
    uint32_t wt_negative[MAX_LAYERS * CURRENT_DMRS_SYMBOLS];
    uint32_t output_layer_stride;
    uint32_t output_dmrs_stride;
    uint32_t reserved[3];
};
static_assert(sizeof(KernelMetadata) == TILING_BYTES, "metadata ABI must be 128 bytes");

constexpr uint32_t META_MAGIC = 0x4d444731u;  // "MDG1"
constexpr uint32_t META_WF_ODD_NEGATIVE_WORD =
    offsetof(KernelMetadata, wf_odd_negative) / sizeof(uint32_t);
constexpr uint32_t META_WT_NEGATIVE_WORD =
    offsetof(KernelMetadata, wt_negative) / sizeof(uint32_t);

// TS 38.211 Type-1, single-symbol DMRS port semantics supported by the
// current TX/RX chain. Port is the configured 1000-based antenna port.
struct PortOccSemantics {
    uint32_t port_index;
    uint32_t comb_delta;
    uint32_t wf_odd_negative;
    uint32_t wt_negative[CURRENT_DMRS_SYMBOLS];
};

Status GetCurrentPortOccSemantics(uint16_t port, PortOccSemantics *semantics);

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           int32_t *cinit);

Status ValidateOpArgs(const MimoDmrsGenOpArgsV1 &args);

// Public runtime adapter. It owns c_init, Gold basis, tiling metadata and the
// legacy [4,2,896] output. Enqueue copies only the contiguous active logical
// [L,D,896] prefix to args.dmrs_re/im and does not synchronize the stream.
Status CreateRuntime(MimoDmrsGenRuntimeV1 **runtime);
Status Enqueue(MimoDmrsGenRuntimeV1 *runtime, const MimoDmrsGenOpArgsV1 &args);
void DestroyRuntime(MimoDmrsGenRuntimeV1 *runtime);

// Host-generated, c_init-independent Gold basis consumed by the legacy ABI.
void BuildGoldBasis(uint16_t *gmat, uint16_t *g1);

constexpr size_t LogicalOutputElems(uint32_t num_layers, uint32_t num_dmrs_symbols) {
    return static_cast<size_t>(num_layers) * num_dmrs_symbols * N_DMRS_PAD;
}
constexpr size_t PhysicalOutputElems() {
    return static_cast<size_t>(MAX_LAYERS) * CURRENT_DMRS_SYMBOLS * N_DMRS_PAD;
}

}  // namespace airan::mimo_dmrs_gen
