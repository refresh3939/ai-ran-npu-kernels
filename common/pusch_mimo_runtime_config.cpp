#include "pusch_mimo_runtime_config.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>

namespace airan {
namespace {

MimoRuntimeConfigStatus Fail(MimoRuntimeConfigStatus status,
                             const char *message, std::string *why) {
    if (why != nullptr) *why = message;
    return status;
}

bool IsZero(const uint8_t *data, size_t count) {
    return std::all_of(data, data + count, [](uint8_t value) { return value == 0; });
}

}  // namespace

MimoRuntimeConfigStatus ValidateMimoRuntimeConfig(
    const PuschMimoRuntimeConfig &config, std::string *why) {
    if (config.abi_version != PUSCH_MIMO_RUNTIME_ABI_VERSION ||
        config.struct_size != sizeof(PuschMimoRuntimeConfig)) {
        return Fail(MimoRuntimeConfigStatus::kAbiMismatch,
                    "runtime config ABI version/size mismatch", why);
    }
    if ((config.flags & ~PUSCH_MIMO_RUNTIME_KNOWN_FLAGS) != 0) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "runtime config contains unknown flags", why);
    }
    if ((config.flags & PUSCH_MIMO_RUNTIME_FLAG_PROFILE_HASH) != 0 &&
        IsZero(config.profile_sha256, sizeof(config.profile_sha256))) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "profile hash flag set with an all-zero digest", why);
    }
    if (config.architecture < static_cast<uint16_t>(MimoArchitecture::kFullDigital) ||
        config.architecture > static_cast<uint16_t>(MimoArchitecture::kAnalog)) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "unsupported MIMO architecture", why);
    }
    const auto valid_tx = [](uint16_t value) {
        return value >= 1 && value <= PUSCH_MIMO_MAX_TX_ANTENNAS;
    };
    const auto valid_rx = [](uint16_t value) {
        return value >= 1 && value <= PUSCH_MIMO_MAX_RX_ANTENNAS;
    };
    if (!valid_tx(config.num_tx_antennas) ||
        !valid_tx(config.num_tx_rf_chains) ||
        !valid_rx(config.num_rx_antennas) ||
        !valid_rx(config.num_rx_rf_chains) ||
        config.num_tx_rf_chains > config.num_tx_antennas ||
        config.num_rx_rf_chains > config.num_rx_antennas) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "invalid antenna/RF-chain topology", why);
    }
    if (config.architecture == static_cast<uint16_t>(MimoArchitecture::kFullDigital) &&
        (config.num_tx_rf_chains != config.num_tx_antennas ||
         config.num_rx_rf_chains != config.num_rx_antennas)) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "full-digital topology requires one RF chain per antenna", why);
    }
    if (config.num_layers < 1 || config.num_layers > PUSCH_MIMO_MAX_PUSCH_LAYERS ||
        config.num_layers > config.num_tx_rf_chains ||
        config.num_layers > config.num_rx_rf_chains ||
        config.num_tx_ports < config.num_layers ||
        config.num_tx_ports > PUSCH_MIMO_MAX_PUSCH_LAYERS ||
        config.num_tx_ports > config.num_tx_rf_chains) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "invalid layer/port dimensions", why);
    }
    if (config.max_rx_antennas < config.num_rx_antennas ||
        config.max_rx_antennas > PUSCH_MIMO_MAX_RX_ANTENNAS ||
        config.max_layers < config.num_layers ||
        config.max_layers > PUSCH_MIMO_MAX_PUSCH_LAYERS ||
        config.rx_bucket < config.num_rx_antennas ||
        config.rx_bucket > config.max_rx_antennas ||
        config.layer_bucket < config.num_layers ||
        config.layer_bucket > PUSCH_MIMO_MAX_DETECT_LAYERS) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "active dimensions exceed declared bucket/capacity", why);
    }
    if ((config.num_rx_antennas != config.rx_bucket ||
         config.num_layers != config.layer_bucket) &&
        (config.flags & PUSCH_MIMO_RUNTIME_FLAG_ADAPTER_REQUIRED) == 0) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "capacity padding requires the adapter flag", why);
    }
    if (config.qm != 2 && config.qm != 4 && config.qm != 6 && config.qm != 8) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "unsupported modulation order", why);
    }
    if (config.num_symbols < 1 || config.num_symbols > 14 ||
        config.fft_size < 128 || config.fft_size > 4096 ||
        config.num_rb < 1 || config.num_rb > 275 ||
        config.rb_start + config.num_rb > 275 ||
        config.used_subcarriers != config.num_rb * 12 ||
        config.used_subcarriers > config.padded_subcarriers ||
        config.padded_subcarriers > config.fft_size) {
        return Fail(MimoRuntimeConfigStatus::kInconsistentShape,
                    "invalid waveform geometry", why);
    }
    if (config.num_dmrs_symbols < 1 ||
        config.num_dmrs_symbols >= config.num_symbols ||
        config.dmrs_port_count != config.num_layers ||
        config.dmrs_port_count > PUSCH_MIMO_MAX_DMRS_PORTS ||
        (config.dmrs_type != 1 && config.dmrs_type != 2) ||
        (config.dmrs_length != 1 && config.dmrs_length != 2) ||
        config.num_cdm_groups_without_data < 1 ||
        config.num_cdm_groups_without_data > 3 || config.n_scid > 1 ||
        __builtin_popcount(static_cast<unsigned>(config.dmrs_symbol_mask)) !=
            config.num_dmrs_symbols) {
        return Fail(MimoRuntimeConfigStatus::kInconsistentShape,
                    "invalid DMRS dimensions", why);
    }
    for (uint16_t i = 0; i < config.dmrs_port_count; ++i) {
        for (uint16_t j = i + 1; j < config.dmrs_port_count; ++j) {
            if (config.dmrs_ports[i] == config.dmrs_ports[j]) {
                return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                            "duplicate active DMRS port", why);
            }
        }
    }
    for (uint16_t i = config.dmrs_port_count; i < PUSCH_MIMO_MAX_DMRS_PORTS; ++i) {
        if (config.dmrs_ports[i] != 0) {
            return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                        "inactive DMRS ports must be zero", why);
        }
    }
    if (config.precoding_mode > static_cast<uint16_t>(MimoPrecodingMode::kNonCodebook) ||
        (config.precoding_mode == static_cast<uint16_t>(MimoPrecodingMode::kBypass) &&
         config.num_tx_ports != config.num_layers)) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "invalid precoding configuration", why);
    }
    if (config.slot_number > 1023 || config.data_scrambling_id > 1023 ||
        config.rnti == 0 ||
        config.start_symbol > 13 || config.num_allocated_symbols == 0 ||
        config.start_symbol + config.num_allocated_symbols > 14 ||
        config.dmrs_additional_position > 3 || config.mapping_type > 1 ||
        config.codeword_index > 1 || config.reserved16 != 0) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "invalid dynamic PUSCH scheduler grant", why);
    }
    if (!std::isfinite(config.channel_gain) || config.channel_gain <= 0.0f ||
        config.channel_gain > 1.0f || !std::isfinite(config.awgn_std_int16) ||
        config.awgn_std_int16 < 0.0f || config.awgn_std_int16 > 200.0f) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "invalid channel gain/noise", why);
    }
    const uint32_t data_symbols = config.num_symbols - config.num_dmrs_symbols;
    const uint32_t data_re = data_symbols * config.used_subcarriers;
    if (config.data_re_per_layer != data_re ||
        config.data_stride_per_layer < data_re ||
        config.qam_row_stride < config.used_subcarriers ||
        config.grid_re_per_port !=
            static_cast<uint32_t>(config.num_symbols) * config.padded_subcarriers) {
        return Fail(MimoRuntimeConfigStatus::kInconsistentShape,
                    "derived tensor geometry mismatch", why);
    }
    if (std::any_of(std::begin(config.reserved), std::end(config.reserved),
                    [](uint32_t value) { return value != 0; })) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "reserved runtime config fields must be zero", why);
    }
    if (why != nullptr) why->clear();
    return MimoRuntimeConfigStatus::kSuccess;
}

MimoRuntimeConfigStatus LoadMimoRuntimeConfig(
    const std::string &path, PuschMimoRuntimeConfig *config, std::string *why) {
    if (config == nullptr) {
        return Fail(MimoRuntimeConfigStatus::kNullArgument,
                    "runtime config output is null", why);
    }
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || static_cast<size_t>(stream.tellg()) != sizeof(*config)) {
        return Fail(MimoRuntimeConfigStatus::kIoError,
                    "runtime config file missing or wrong size", why);
    }
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(config), sizeof(*config));
    if (!stream) {
        return Fail(MimoRuntimeConfigStatus::kIoError,
                    "runtime config read failed", why);
    }
    return ValidateMimoRuntimeConfig(*config, why);
}

MimoRuntimeConfigStatus LoadMimoRuntimeConfigFromEnv(
    uint16_t expected_rank, PuschMimoRuntimeConfig *config, bool *enabled,
    std::string *why) {
    if (config == nullptr || enabled == nullptr) {
        return Fail(MimoRuntimeConfigStatus::kNullArgument,
                    "runtime config/environment output is null", why);
    }
    const char *path = std::getenv("PUSCH_MIMO_RUNTIME_CONFIG");
    if (path == nullptr || path[0] == '\0') {
        *enabled = false;
        *config = {};
        if (why != nullptr) why->clear();
        return MimoRuntimeConfigStatus::kSuccess;
    }
    const auto status = LoadMimoRuntimeConfig(path, config, why);
    if (status != MimoRuntimeConfigStatus::kSuccess) return status;
    if (expected_rank != 0 && config->num_layers != expected_rank) {
        return Fail(MimoRuntimeConfigStatus::kInconsistentShape,
                    "runtime config Rank does not match stage Rank", why);
    }
    *enabled = true;
    return MimoRuntimeConfigStatus::kSuccess;
}

MimoRuntimeConfigStatus DerivePuschMimoConfig(
    const PuschMimoRuntimeConfig &runtime, uint16_t operator_rx_capacity,
    PuschMimoConfig *config, std::string *why, uint16_t slot_number) {
    if (config == nullptr) {
        return Fail(MimoRuntimeConfigStatus::kNullArgument,
                    "derived PuschMimoConfig output is null", why);
    }
    const auto status = ValidateMimoRuntimeConfig(runtime, why);
    if (status != MimoRuntimeConfigStatus::kSuccess) return status;
    if (runtime.dmrs_port_count > 4 ||
        operator_rx_capacity < runtime.num_rx_antennas ||
        operator_rx_capacity > PUSCH_MIMO_MAX_RX_ANTENNAS ||
        runtime.num_symbols > 255) {
        return Fail(MimoRuntimeConfigStatus::kUnsupportedValue,
                    "runtime config cannot be represented by operator ABI v1", why);
    }
    PuschMimoConfig result{};
    result.abi_version = PUSCH_MIMO_ABI_VERSION;
    result.struct_size = sizeof(result);
    result.num_layers = runtime.num_layers;
    result.num_tx_ports = runtime.num_tx_ports;
    result.num_rx_antennas = operator_rx_capacity;
    result.qm = runtime.qm;
    result.num_symbols = runtime.num_symbols;
    result.fft_size = runtime.fft_size;
    result.num_rb = runtime.num_rb;
    result.rb_start = runtime.rb_start;
    result.slot_number = slot_number == 0 ? runtime.slot_number : slot_number;
    result.start_symbol = static_cast<uint8_t>(runtime.start_symbol);
    result.num_allocated_symbols =
        static_cast<uint8_t>(runtime.num_allocated_symbols);
    result.used_subcarriers = runtime.used_subcarriers;
    result.padded_subcarriers = runtime.padded_subcarriers;
    result.dmrs_symbol_mask = runtime.dmrs_symbol_mask;
    for (uint16_t index = 0; index < runtime.dmrs_port_count; ++index) {
        result.dmrs_ports[index] = runtime.dmrs_ports[index];
    }
    result.dmrs_type = static_cast<uint8_t>(runtime.dmrs_type);
    result.dmrs_length = static_cast<uint8_t>(runtime.dmrs_length);
    result.dmrs_scrambling_id = runtime.dmrs_scrambling_id;
    result.data_scrambling_id = runtime.data_scrambling_id;
    result.rnti = runtime.rnti;
    result.dmrs_additional_position =
        static_cast<uint8_t>(runtime.dmrs_additional_position);
    result.mapping_type = static_cast<uint8_t>(runtime.mapping_type);
    result.num_cdm_groups_without_data =
        static_cast<uint8_t>(runtime.num_cdm_groups_without_data);
    result.n_scid = static_cast<uint8_t>(runtime.n_scid);
    result.codeword_index = static_cast<uint8_t>(runtime.codeword_index);
    result.codebook_enabled = runtime.precoding_mode ==
        static_cast<uint16_t>(MimoPrecodingMode::kCodebook);
    result.tpmi = runtime.tpmi;
    result.prg_size_rb = runtime.prg_size_rb;
    *config = result;
    if (why != nullptr) why->clear();
    return MimoRuntimeConfigStatus::kSuccess;
}

}  // namespace airan
