#include "pusch_mimo_runtime_config.h"

#include <cstdlib>
#include <cstdio>
#include <string>

namespace {

bool ExpectRejected(const airan::PuschMimoRuntimeConfig &config,
                    const char *case_name) {
    std::string why;
    if (airan::ValidateMimoRuntimeConfig(config, &why) ==
        airan::MimoRuntimeConfigStatus::kSuccess) {
        std::fprintf(stderr, "[FAIL] invalid %s accepted\n", case_name);
        return false;
    }
    if (why.empty()) {
        std::fprintf(stderr, "[FAIL] invalid %s had no diagnostic\n", case_name);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s RUNTIME_CONFIG.bin\n", argv[0]);
        return 2;
    }
    airan::PuschMimoRuntimeConfig config{};
    std::string why;
    if (airan::LoadMimoRuntimeConfig(argv[1], &config, &why) !=
        airan::MimoRuntimeConfigStatus::kSuccess) {
        std::fprintf(stderr, "[FAIL] %s\n", why.c_str());
        return 1;
    }
    auto invalid_rank = config;
    invalid_rank.num_layers = 5;
    auto invalid_rx = config;
    invalid_rx.num_rx_antennas = 65;
    auto invalid_tx = config;
    invalid_tx.num_tx_antennas = 17;
    auto invalid_semantic_capacity = config;
    invalid_semantic_capacity.max_layers = 16;
    auto invalid_hash = config;
    for (auto &byte : invalid_hash.profile_sha256) byte = 0;
    auto invalid_shape = config;
    ++invalid_shape.data_re_per_layer;
    auto invalid_reserved = config;
    invalid_reserved.reserved[0] = 1;
    auto invalid_adapter_flag = config;
    invalid_adapter_flag.flags &= ~airan::PUSCH_MIMO_RUNTIME_FLAG_ADAPTER_REQUIRED;
    auto invalid_rnti = config;
    invalid_rnti.rnti = 0;
    auto invalid_allocation = config;
    invalid_allocation.start_symbol = 13;
    invalid_allocation.num_allocated_symbols = 2;
    if (!ExpectRejected(invalid_rank, "single-PUSCH Rank5") ||
        !ExpectRejected(invalid_rx, "65Rx") ||
        !ExpectRejected(invalid_tx, "17Tx") ||
        !ExpectRejected(invalid_semantic_capacity, "single-PUSCH max_layers=16") ||
        !ExpectRejected(invalid_hash, "zero profile hash") ||
        !ExpectRejected(invalid_shape, "derived tensor shape") ||
        !ExpectRejected(invalid_reserved, "reserved field") ||
        !ExpectRejected(invalid_adapter_flag, "missing capacity-adapter flag") ||
        !ExpectRejected(invalid_rnti, "zero RNTI") ||
        !ExpectRejected(invalid_allocation, "out-of-slot symbol allocation")) {
        return 1;
    }
    airan::PuschMimoConfig operator_config{};
    if (airan::DerivePuschMimoConfig(config, config.max_rx_antennas,
                                     &operator_config, &why) !=
            airan::MimoRuntimeConfigStatus::kSuccess ||
        operator_config.num_layers != config.num_layers ||
        operator_config.num_tx_ports != config.num_tx_ports ||
        operator_config.num_rx_antennas != config.max_rx_antennas ||
        operator_config.dmrs_ports[3] != config.dmrs_ports[3] ||
        operator_config.slot_number != config.slot_number ||
        operator_config.dmrs_scrambling_id != config.dmrs_scrambling_id ||
        operator_config.data_scrambling_id != config.data_scrambling_id ||
        operator_config.rnti != config.rnti ||
        operator_config.start_symbol != config.start_symbol ||
        operator_config.num_allocated_symbols != config.num_allocated_symbols ||
        operator_config.dmrs_additional_position != config.dmrs_additional_position ||
        operator_config.mapping_type != config.mapping_type) {
        std::fprintf(stderr, "[FAIL] runtime-to-operator ABI derivation: %s\n",
                     why.c_str());
        return 1;
    }
    if (setenv("PUSCH_MIMO_RUNTIME_CONFIG", argv[1], 1) != 0) {
        std::fprintf(stderr, "[FAIL] cannot set runtime-config test environment\n");
        return 1;
    }
    airan::PuschMimoRuntimeConfig from_env{};
    bool enabled = false;
    if (airan::LoadMimoRuntimeConfigFromEnv(config.num_layers, &from_env,
                                            &enabled, &why) !=
            airan::MimoRuntimeConfigStatus::kSuccess ||
        !enabled || from_env.num_layers != config.num_layers) {
        std::fprintf(stderr, "[FAIL] environment runtime load: %s\n", why.c_str());
        return 1;
    }
    if (airan::LoadMimoRuntimeConfigFromEnv(config.num_layers + 1, &from_env,
                                            &enabled, &why) ==
        airan::MimoRuntimeConfigStatus::kSuccess) {
        std::fprintf(stderr, "[FAIL] environment Rank mismatch accepted\n");
        return 1;
    }
    unsetenv("PUSCH_MIMO_RUNTIME_CONFIG");
    std::printf("[PASS] runtime ABI bytes=%zu tx=%u rx=%u rank=%u ports=%u bucket=%ux%u\n",
                sizeof(config), config.num_tx_antennas, config.num_rx_antennas,
                config.num_layers, config.num_tx_ports, config.rx_bucket,
                config.layer_bucket);
    return 0;
}
