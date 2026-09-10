/** @file pusch_mimo_runtime_config.h Host validation/loading for profile ABI. */
#pragma once

#include "pusch_mimo_types.h"

#include <string>

namespace airan {

enum class MimoRuntimeConfigStatus : int32_t {
    kSuccess = 0,
    kNullArgument = 1,
    kAbiMismatch = 2,
    kUnsupportedValue = 3,
    kInconsistentShape = 4,
    kIoError = 5,
};

MimoRuntimeConfigStatus ValidateMimoRuntimeConfig(
    const PuschMimoRuntimeConfig &config, std::string *why = nullptr);

MimoRuntimeConfigStatus LoadMimoRuntimeConfig(
    const std::string &path, PuschMimoRuntimeConfig *config,
    std::string *why = nullptr);

/** Load the active profile from PUSCH_MIMO_RUNTIME_CONFIG when present.
 * expected_rank=0 accepts the Rank carried by the runtime profile.
 */
MimoRuntimeConfigStatus LoadMimoRuntimeConfigFromEnv(
    uint16_t expected_rank, PuschMimoRuntimeConfig *config, bool *enabled,
    std::string *why = nullptr);

/** Derive the unchanged operator ABI v1 from the common runtime ABI. */
MimoRuntimeConfigStatus DerivePuschMimoConfig(
    const PuschMimoRuntimeConfig &runtime, uint16_t operator_rx_capacity,
    PuschMimoConfig *config, std::string *why = nullptr,
    uint16_t slot_number = 0);

}  // namespace airan
