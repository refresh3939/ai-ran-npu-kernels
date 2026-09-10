#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#include "mimo_dmrs_gen.h"

namespace mdg = airan::mimo_dmrs_gen;

namespace {

static_assert(std::is_same<mdg::PuschMimoConfig, ::airan::PuschMimoConfig>::value,
              "generator must use canonical PUSCH MIMO config");
static_assert(std::is_same<mdg::PuschMimoLayout, ::airan::PuschMimoLayout>::value,
              "generator must use canonical PUSCH MIMO layout");
static_assert(mdg::ABI_VERSION == ::airan::PUSCH_MIMO_ABI_VERSION,
              "generator ABI version must match the canonical chain ABI");

struct Case {
    uint16_t layers;
    uint16_t slot;
    uint16_t n_id;
    uint8_t n_scid;
    uint16_t ports[mdg::MAX_LAYERS];
};

constexpr Case CASES[] = {
    {1, 0, 1, 0, {1000, 0, 0, 0}},
    {2, 0, 0, 0, {1000, 1002, 0, 0}},
    {2, 7, 1, 0, {1001, 1000, 0, 0}},
    {3, 0, 1, 1, {1002, 1000, 1003, 0}},
    {4, 0, 300, 0, {1003, 1002, 1001, 1000}},
};

mdg::PuschMimoConfig MakeConfig(const Case &test)
{
    mdg::PuschMimoConfig config {};
    config.abi_version = mdg::ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = test.layers;
    config.num_tx_ports = test.layers == 1 ? 1 : (test.layers == 2 ? 2 : 4);
    config.num_rx_antennas = 64;
    config.qm = 8;
    config.num_symbols = mdg::N_SYMBOLS;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.slot_number = test.slot;
    config.num_allocated_symbols = mdg::N_SYMBOLS;
    config.used_subcarriers = mdg::N_SC_USED;
    config.padded_subcarriers = mdg::N_SC_PAD;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
    config.dmrs_scrambling_id = test.n_id;
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    config.n_scid = test.n_scid;
    for (uint32_t layer = 0; layer < test.layers; ++layer) {
        config.dmrs_ports[layer] = test.ports[layer];
    }
    return config;
}

int32_t ExpectedCinit(const mdg::PuschMimoConfig &config, uint32_t symbol)
{
    const uint64_t value =
        (uint64_t{1} << 17) *
            (static_cast<uint64_t>(mdg::N_SYMBOLS) * config.slot_number + symbol + 1u) *
            (2u * static_cast<uint64_t>(config.dmrs_scrambling_id) + 1u) +
        2u * static_cast<uint64_t>(config.dmrs_scrambling_id) + config.n_scid;
    return static_cast<int32_t>(value & 0x7fffffffu);
}

bool CheckCase(const Case &test)
{
    const mdg::PuschMimoConfig config = MakeConfig(test);
    mdg::PuschMimoLayout layout {};
    mdg::KernelMetadata metadata {};
    int32_t cinit[mdg::CINIT_PAD] = {};
    if (mdg::BuildCurrentProfile(config, &layout, &metadata, cinit) != mdg::OK) return false;

    bool ok = metadata.magic == mdg::META_MAGIC && metadata.num_layers == test.layers &&
              metadata.num_dmrs_symbols == 2 && metadata.dmrs_symbols[0] == 2 &&
              metadata.dmrs_symbols[1] == 11 && metadata.output_layer_stride == 1792 &&
              metadata.output_dmrs_stride == 896 && cinit[0] == ExpectedCinit(config, 2) &&
              cinit[1] == ExpectedCinit(config, 11) && layout.num_dmrs_symbols == 2 &&
              layout.num_data_symbols == 12 && layout.num_data_re == 19152 &&
              layout.data_stride == 19200 &&
              layout.codeword_symbols == test.layers * 19152u &&
              layout.codeword_stride == test.layers * 19200u;
    for (uint32_t index = 2; index < mdg::CINIT_PAD; ++index) ok &= cinit[index] == 0;
    for (uint32_t layer = 0; layer < test.layers; ++layer) {
        const uint32_t port_index = test.ports[layer] - 1000u;
        ok &= metadata.port_indices[layer] == port_index;
        ok &= metadata.comb_delta[layer] == port_index / 2u;
        ok &= metadata.wf_odd_negative[layer] == (port_index & 1u);
        ok &= metadata.wt_negative[layer * 2] == 0;
        ok &= metadata.wt_negative[layer * 2 + 1] == 0;
    }
    for (uint32_t layer = test.layers; layer < mdg::MAX_LAYERS; ++layer) {
        ok &= metadata.port_indices[layer] == 0 && metadata.comb_delta[layer] == 0 &&
              metadata.wf_odd_negative[layer] == 0;
    }
    for (uint32_t word : metadata.reserved) ok &= word == 0;

    mdg::MimoDmrsGenOpArgsV1 args {};
    args.abi_version = mdg::ABI_VERSION;
    args.struct_size = sizeof(args);
    args.dmrs_re = reinterpret_cast<void *>(uintptr_t{1});
    args.dmrs_im = reinterpret_cast<void *>(uintptr_t{2});
    args.config = &config;
    args.layout = &layout;
    args.stream = reinterpret_cast<void *>(uintptr_t{3});
    ok &= mdg::ValidateOpArgs(args) == mdg::OK;
    mdg::PuschMimoLayout wrong = layout;
    --wrong.data_stride;
    args.layout = &wrong;
    ok &= mdg::ValidateOpArgs(args) == mdg::INVALID_ARGUMENT;
    return ok;
}

}  // namespace

int main()
{
    static_assert(sizeof(mdg::KernelMetadata) == 128, "metadata size");
    bool ok = true;
    for (const Case &test : CASES) ok &= CheckCase(test);

    for (uint16_t port = 1000; port <= 1003; ++port) {
        mdg::PortOccSemantics semantics {};
        const uint32_t index = port - 1000u;
        ok &= mdg::GetCurrentPortOccSemantics(port, &semantics) == mdg::OK;
        ok &= semantics.port_index == index && semantics.comb_delta == index / 2u &&
              semantics.wf_odd_negative == (index & 1u) &&
              semantics.wt_negative[0] == 0 && semantics.wt_negative[1] == 0;
    }
    mdg::PortOccSemantics unused {};
    ok &= mdg::GetCurrentPortOccSemantics(999, &unused) == mdg::UNSUPPORTED_PROFILE;
    ok &= mdg::GetCurrentPortOccSemantics(1004, &unused) == mdg::UNSUPPORTED_PROFILE;

    mdg::PuschMimoConfig invalid = MakeConfig(CASES[4]);
    mdg::PuschMimoLayout layout {};
    mdg::KernelMetadata metadata {};
    int32_t cinit[mdg::CINIT_PAD] = {};
    invalid.dmrs_ports[1] = invalid.dmrs_ports[0];
    ok &= mdg::BuildCurrentProfile(invalid, &layout, &metadata, cinit) ==
          mdg::UNSUPPORTED_PROFILE;
    invalid = MakeConfig(CASES[0]);
    invalid.num_layers = 0;
    ok &= mdg::BuildCurrentProfile(invalid, &layout, &metadata, cinit) ==
          mdg::UNSUPPORTED_PROFILE;
    invalid = MakeConfig(CASES[0]);
    invalid.dmrs_length = 2;
    ok &= mdg::BuildCurrentProfile(invalid, &layout, &metadata, cinit) ==
          mdg::UNSUPPORTED_PROFILE;

    std::vector<uint16_t> gmat(mdg::MAT_LEN);
    std::vector<uint16_t> g1(mdg::N_DMRS_PLANE);
    mdg::BuildGoldBasis(gmat.data(), g1.data());
    for (uint32_t index = mdg::N_DMRS_RE; index < mdg::N_DMRS_PAD; ++index) {
        ok &= g1[index] == 0 && g1[mdg::N_DMRS_PAD + index] == 0;
        for (uint32_t bit = 0; bit < mdg::GOLD_NBITS; ++bit) {
            const size_t base = static_cast<size_t>(bit) * mdg::N_DMRS_PLANE;
            ok &= gmat[base + index] == 0 &&
                  gmat[base + mdg::N_DMRS_PAD + index] == 0;
        }
    }

    std::printf("mimo_dmrs_gen host contract: %s (Rank1-4, comb/Wf/Wt, metadata, padding, negatives)\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
