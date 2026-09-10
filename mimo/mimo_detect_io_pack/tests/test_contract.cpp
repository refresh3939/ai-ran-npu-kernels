#include "mimo_detect_io_pack.h"

#include <array>
#include <cstdio>

namespace pack = airan::mimo_detect_io_pack;

namespace {

bool Check(bool condition, const char *message)
{
    if (!condition) std::fprintf(stderr, "[contract] FAIL: %s\n", message);
    return condition;
}

pack::PuschMimoConfig MakeConfig(uint16_t layers, uint16_t ports)
{
    pack::PuschMimoConfig config {};
    config.abi_version = pack::ABI_VERSION;
    config.struct_size = static_cast<uint16_t>(sizeof(config));
    config.num_layers = layers;
    config.num_tx_ports = ports;
    config.num_rx_antennas = pack::NR;
    config.qm = 8;
    config.num_symbols = pack::N_SYMBOL;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.slot_number = 7;
    config.num_allocated_symbols = pack::N_SYMBOL;
    config.used_subcarriers = 1596;
    config.padded_subcarriers = pack::N_SC_PAD;
    return config;
}

template <size_t N>
pack::MimoDetectLayerPlan MakePlan(
    const std::array<pack::PuschMimoConfig, N> &configs)
{
    pack::MimoDetectLayerPlan plan {};
    plan.abi_version = pack::ABI_VERSION;
    plan.struct_size = static_cast<uint16_t>(sizeof(plan));
    plan.num_rx_antennas = pack::NR;
    plan.layer_capacity = pack::NL;
    plan.num_allocations = static_cast<uint16_t>(configs.size());
    uint16_t offset = 0;
    for (size_t i = 0; i < configs.size(); ++i) {
        plan.allocations[i].pusch_index = static_cast<uint16_t>(i);
        plan.allocations[i].layer_offset = offset;
        plan.allocations[i].num_layers = configs[i].num_layers;
        plan.allocations[i].num_tx_ports = configs[i].num_tx_ports;
        offset = static_cast<uint16_t>(offset + configs[i].num_layers);
    }
    plan.total_layers = offset;
    return plan;
}

}  // namespace

int main()
{
    const std::array<pack::PuschMimoConfig, 1> configs = {MakeConfig(4, 4)};
    const pack::MimoDetectLayerPlan plan = MakePlan(configs);
    pack::KernelMetadata metadata {};
    bool ok = true;
    ok &= Check(pack::BuildCurrentProfile(configs.data(), configs.size(), plan,
                                          &metadata) == pack::OK,
                "valid standard Rank4 PUSCH plan rejected");
    ok &= Check(metadata.magic == pack::META_MAGIC && metadata.active_layers == 4 &&
                    metadata.num_rx_antennas == pack::NR &&
                    metadata.layer_capacity == pack::NL &&
                    metadata.n_re == 23296,
                "kernel metadata mismatch");

    pack::MimoDetectLayerPlan bad = plan;
    bad.allocations[0].layer_offset = 1;
    ok &= Check(pack::BuildCurrentProfile(configs.data(), configs.size(), bad,
                                          &metadata) == pack::PLAN_MISMATCH,
                "nonzero single-PUSCH layer offset was accepted");
    bad = plan;
    bad.total_layers = 3;
    ok &= Check(pack::BuildCurrentProfile(configs.data(), configs.size(), bad,
                                          &metadata) == pack::PLAN_MISMATCH,
                "incorrect total layer count was accepted");
    bad = plan;
    bad.allocations[1].num_layers = 1;
    ok &= Check(pack::BuildCurrentProfile(configs.data(), configs.size(), bad,
                                          &metadata) == pack::PLAN_MISMATCH,
                "non-zero unused allocation was accepted");

    auto bad_configs = configs;
    bad_configs[0].num_rx_antennas = pack::NR == 64 ? 32 : 64;
    ok &= Check(pack::BuildCurrentProfile(bad_configs.data(), bad_configs.size(), plan,
                                          &metadata) == pack::UNSUPPORTED_PROFILE,
                "unsupported receiver count was accepted");

    const std::array<pack::PuschMimoConfig, 2> multiple_configs = {
        MakeConfig(2, 2), MakeConfig(2, 2)};
    const pack::MimoDetectLayerPlan multiple_plan = MakePlan(multiple_configs);
    ok &= Check(pack::BuildCurrentProfile(multiple_configs.data(),
                                          multiple_configs.size(), multiple_plan,
                                          &metadata) == pack::INVALID_ARGUMENT,
                "multi-PUSCH aggregate plan was accepted by standard chain");

    pack::MimoDetectIoPackOpArgsV1 args {};
    args.abi_version = pack::ABI_VERSION;
    args.struct_size = static_cast<uint16_t>(sizeof(args));
    args.rx_grid_re = args.rx_grid_im = args.h_grid_re = args.h_grid_im =
        args.noise_var_rx = reinterpret_cast<const void *>(1);
    args.hrm_re = args.hrm_im = args.yvpad_re = args.yvpad_im = args.no =
        args.stream = reinterpret_cast<void *>(1);
    args.configs = configs.data();
    args.num_configs = static_cast<uint16_t>(configs.size());
    args.layer_plan = &plan;
    ok &= Check(pack::ValidateOpArgs(args) == pack::OK,
                "valid public OpArgs rejected");
    args.num_configs = 0;
    ok &= Check(pack::ValidateOpArgs(args) == pack::INVALID_ARGUMENT,
                "zero config count was accepted");

    std::printf("[contract] %s: one standard PUSCH Rank4, activeL=4, "
                "physical storage width=16\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
