#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace airan {

// Physical ABI shared with rx/mimo_detect_io_pack.  These are storage
// dimensions, not the number of active spatial layers.
constexpr uint32_t MIMO_IO_ABI_VERSION = 1;
#ifndef MIMO_IO_NR_VALUE
#define MIMO_IO_NR_VALUE 64
#endif
#ifndef MIMO_IO_LAYER_VALUE
#define MIMO_IO_LAYER_VALUE 16
#endif
constexpr uint32_t MIMO_IO_NR = MIMO_IO_NR_VALUE;
constexpr uint32_t MIMO_IO_PHYSICAL_LAYERS = MIMO_IO_LAYER_VALUE;
constexpr uint32_t MIMO_IO_SYMBOLS = 14;
constexpr uint32_t MIMO_IO_SUBCARRIERS = 1664;
constexpr uint32_t MIMO_IO_RE = MIMO_IO_SYMBOLS * MIMO_IO_SUBCARRIERS;

struct MimoDetectBriHostContract {
    uint32_t abi_version;
    uint32_t nr;
    uint32_t physical_layers;
    uint32_t active_layers;
    uint32_t symbols;
    uint32_t subcarriers;
    uint32_t bri_block_size;
    uint32_t pack_mode;
};

constexpr size_t MimoIoMatrixElements()
{
    return static_cast<size_t>(MIMO_IO_RE) * MIMO_IO_NR * MIMO_IO_PHYSICAL_LAYERS;
}

constexpr size_t MimoIoNoiseElements()
{
    return MIMO_IO_RE;
}

constexpr size_t MimoIoOutputElements()
{
    return static_cast<size_t>(MIMO_IO_PHYSICAL_LAYERS) * MIMO_IO_RE;
}

// Validates the selected io_pack -> detector profile and the BRI convergence
// constraints.  On failure, reason contains a host-facing diagnostic.
bool ValidateMimoDetectBriHostContract(
    const MimoDetectBriHostContract &contract, std::string *reason);

// Validates the actual fp16 buffers before launching the kernel:
//   hrm_*   [23296,NR,16], inactive columns [active_layers,16) == 0
//   yvpad_* [23296,NR,16], all 16 columns repeat the same y sample
//   no      [23296], finite and non-negative
// The length arguments are element counts, not byte counts.
bool ValidateMimoDetectBriPhysicalInputs(
    const MimoDetectBriHostContract &contract,
    const uint16_t *hrm_re, size_t hrm_re_elements,
    const uint16_t *hrm_im, size_t hrm_im_elements,
    const uint16_t *yvpad_re, size_t yvpad_re_elements,
    const uint16_t *yvpad_im, size_t yvpad_im_elements,
    const uint16_t *noise, size_t noise_elements,
    std::string *reason);

}  // namespace airan
