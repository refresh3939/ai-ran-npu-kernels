#include "mimo_detect_io_pack.h"

#include <cstring>

namespace airan::mimo_detect_io_pack {

namespace {

bool IsZero(const uint32_t *values, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (values[i] != 0) return false;
    }
    return true;
}

Status ValidateConfig(const PuschMimoConfig &config)
{
    if (config.abi_version != ABI_VERSION ||
        config.struct_size < sizeof(PuschMimoConfig)) {
        return UNSUPPORTED_PROFILE;
    }
    if (config.flags != 0 || config.num_layers < 1 ||
        config.num_layers > MAX_PUSCH_LAYERS ||
        (config.num_tx_ports != 1 && config.num_tx_ports != 2 &&
         config.num_tx_ports != 4) ||
        config.num_tx_ports < config.num_layers || config.num_rx_antennas != NR ||
        config.qm != 8 || config.num_symbols != N_SYMBOL ||
        config.fft_size != 2048 || config.num_rb != 133 || config.rb_start != 0 ||
        config.start_symbol != 0 || config.num_allocated_symbols != N_SYMBOL ||
        config.used_subcarriers != 1596 ||
        config.padded_subcarriers != N_SC_PAD || config.transform_precoding != 0 ||
        !IsZero(config.reserved, sizeof(config.reserved) / sizeof(config.reserved[0]))) {
        return UNSUPPORTED_PROFILE;
    }
    return OK;
}

}  // namespace

Status BuildCurrentProfile(const PuschMimoConfig *configs,
                           size_t num_configs,
                           const MimoDetectLayerPlan &layer_plan,
                           KernelMetadata *metadata)
{
    if (configs == nullptr || metadata == nullptr ||
        num_configs != STANDARD_CHAIN_ALLOCATIONS) {
        return INVALID_ARGUMENT;
    }
    if (layer_plan.abi_version != ABI_VERSION ||
        layer_plan.struct_size < sizeof(MimoDetectLayerPlan)) {
        return UNSUPPORTED_PROFILE;
    }
    if (layer_plan.num_rx_antennas != NR || layer_plan.layer_capacity != NL ||
        layer_plan.total_layers < 1 ||
        layer_plan.total_layers > MAX_PUSCH_LAYERS ||
        layer_plan.num_allocations != num_configs ||
        !IsZero(layer_plan.reserved,
                sizeof(layer_plan.reserved) / sizeof(layer_plan.reserved[0]))) {
        return PLAN_MISMATCH;
    }

    uint16_t expected_offset = 0;
    for (size_t i = 0; i < num_configs; ++i) {
        const Status config_status = ValidateConfig(configs[i]);
        if (config_status != OK) return config_status;
        const MimoDetectAllocation &allocation = layer_plan.allocations[i];
        if (allocation.pusch_index != i || allocation.layer_offset != expected_offset ||
            allocation.num_layers != configs[i].num_layers ||
            allocation.num_tx_ports != configs[i].num_tx_ports) {
            return PLAN_MISMATCH;
        }
        expected_offset = static_cast<uint16_t>(expected_offset + configs[i].num_layers);
        if (expected_offset > MAX_PUSCH_LAYERS) return PLAN_MISMATCH;
    }
    if (expected_offset != layer_plan.total_layers) return PLAN_MISMATCH;

    for (size_t i = num_configs; i < MAX_ALLOCATIONS; ++i) {
        const MimoDetectAllocation &allocation = layer_plan.allocations[i];
        if (allocation.pusch_index != 0 || allocation.layer_offset != 0 ||
            allocation.num_layers != 0 || allocation.num_tx_ports != 0) {
            return PLAN_MISMATCH;
        }
    }

    std::memset(metadata, 0, sizeof(*metadata));
    metadata->magic = META_MAGIC;
    metadata->active_layers = layer_plan.total_layers;
    metadata->num_rx_antennas = NR;
    metadata->layer_capacity = NL;
    metadata->n_re = N_RE;
    metadata->abi_version = ABI_VERSION;
    return OK;
}

Status ValidateOpArgs(const MimoDetectIoPackOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(MimoDetectIoPackOpArgsV1)) {
        return UNSUPPORTED_PROFILE;
    }
    if (args.rx_grid_re == nullptr || args.rx_grid_im == nullptr ||
        args.h_grid_re == nullptr || args.h_grid_im == nullptr ||
        args.noise_var_rx == nullptr || args.hrm_re == nullptr ||
        args.hrm_im == nullptr || args.yvpad_re == nullptr ||
        args.yvpad_im == nullptr || args.no == nullptr || args.configs == nullptr ||
        args.layer_plan == nullptr || args.num_configs == 0 || args.reserved != 0 ||
        args.stream == nullptr) {
        return INVALID_ARGUMENT;
    }
    KernelMetadata metadata {};
    return BuildCurrentProfile(args.configs, args.num_configs, *args.layer_plan,
                               &metadata);
}

}  // namespace airan::mimo_detect_io_pack
