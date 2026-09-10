#include "../mimo_dmrs_ls.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace airan::mimo_dmrs_ls;

namespace {

[[noreturn]] void Fail(const char *message)
{
    std::fprintf(stderr, "[contract] FAIL: %s\n", message);
    std::exit(EXIT_FAILURE);
}

void Require(bool condition, const char *message)
{
    if (!condition) Fail(message);
}

PuschMimoConfig MakeConfig(const uint16_t *ports, uint16_t numLayers)
{
    PuschMimoConfig config {};
    config.abi_version = ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = numLayers;
    config.num_tx_ports = numLayers;
    config.num_rx_antennas = NR_CURRENT;
    config.qm = 8;
    config.num_symbols = N_SYMBOLS;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = N_SYMBOLS;
    config.used_subcarriers = N_SC_USED;
    config.padded_subcarriers = N_SC_PAD;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2u) | (1u << 11u));
    config.dmrs_type = 1;
    config.num_cdm_groups_without_data = 2;
    for (uint16_t layer = 0; layer < numLayers; ++layer) {
        config.dmrs_ports[layer] = ports[layer];
    }
    return config;
}

void CheckCase(const uint16_t *ports, uint16_t numLayers,
               const uint16_t *expectedCounts, Status expectedLmmse)
{
    const PuschMimoConfig config = MakeConfig(ports, numLayers);
    PuschMimoLayout layout {};
    KernelMetadata metadata {};
    std::array<uint16_t, COUNT_PAD> counts {};
    std::array<uint16_t,
               MAX_LAYERS * CURRENT_DMRS_SYMBOLS * N_PILOT_PAD> subcarriers {};
    Require(BuildCurrentProfile(config, &layout, &metadata, counts.data(),
                                subcarriers.data()) == OK,
            "BuildCurrentProfile rejected a supported profile");
    Require(layout.num_dmrs_symbols == CURRENT_DMRS_SYMBOLS &&
            layout.num_data_symbols == 12 && layout.num_data_re == 19152 &&
            layout.data_stride == 19200,
            "shared PuschMimoLayout does not match the natural chain layout");

    for (uint32_t layer = 0; layer < numLayers; ++layer) {
        const uint16_t expectedCount = expectedCounts[layer];
        const ObservationModel expectedModel = expectedCount == N_OCC_PILOT ?
            FD_OCC2_399 : COMB2_798;
        Require(metadata.observation_model[layer] == expectedModel,
                "kernel metadata observation model mismatch");
        const uint16_t delta = static_cast<uint16_t>((ports[layer] - 1000u) / 2u);
        const uint16_t first = expectedCount == N_OCC_PILOT ?
            static_cast<uint16_t>(delta + 1u) : delta;
        const uint16_t step = expectedCount == N_OCC_PILOT ? 4u : 2u;
        for (uint32_t dmrs = 0; dmrs < CURRENT_DMRS_SYMBOLS; ++dmrs) {
            Require(counts[layer * CURRENT_DMRS_SYMBOLS + dmrs] == expectedCount,
                    "pilot_count mismatch");
            const size_t base = PilotOffset(layer, dmrs, 0);
            Require(subcarriers[base] == first, "pilot_sc first position mismatch");
            Require(subcarriers[base + expectedCount - 1u] ==
                    static_cast<uint16_t>(first + step * (expectedCount - 1u)),
                    "pilot_sc last position mismatch");
            Require(subcarriers[base + expectedCount] == 0,
                    "pilot_sc padding must be zero");
        }
    }
    ObservationModel models[MAX_LAYERS] = {};
    Require(DescribeObservationModels(counts.data(), numLayers,
                                      CURRENT_DMRS_SYMBOLS, models) == OK,
            "canonical observation-model description rejected a valid layer");
    Require(ValidateNaturalLmmseContract(counts.data(), subcarriers.data(), numLayers,
                                         CURRENT_DMRS_SYMBOLS) == OK,
            "canonical natural CE contract rejected 399/798 observations");
    for (uint32_t layer = 0; layer < numLayers; ++layer) {
        const ObservationModel expectedModel = expectedCounts[layer] == N_OCC_PILOT ?
            FD_OCC2_399 : COMB2_798;
        Require(models[layer] == expectedModel,
                "per-layer observation-model classification mismatch");
    }
    Require(ValidateLmmse798Compatibility(counts.data(), numLayers,
                                           CURRENT_DMRS_SYMBOLS) == expectedLmmse,
            "LMMSE observation-model gate mismatch");
    Require(NaturalOffset(1, 0, 0, 0, numLayers) ==
            static_cast<size_t>(numLayers) * CURRENT_DMRS_SYMBOLS * N_PILOT_PAD,
            "natural [rx,layer,dmrs,pilot] stride mismatch");
}

}  // namespace

int main()
{
    constexpr uint16_t rank1[] = {1000};
    constexpr uint16_t rank2Disjoint[] = {1000, 1002};
    constexpr uint16_t rank2Occ[] = {1000, 1001};
    constexpr uint16_t rank3Mixed[] = {1000, 1001, 1002};
    constexpr uint16_t rank4Occ[] = {1000, 1001, 1002, 1003};
    constexpr uint16_t rank1Counts[] = {N_DMRS_RE};
    constexpr uint16_t rank2DisjointCounts[] = {N_DMRS_RE, N_DMRS_RE};
    constexpr uint16_t rank2OccCounts[] = {N_OCC_PILOT, N_OCC_PILOT};
    constexpr uint16_t rank3MixedCounts[] = {N_OCC_PILOT, N_OCC_PILOT, N_DMRS_RE};
    constexpr uint16_t rank4OccCounts[] = {
        N_OCC_PILOT, N_OCC_PILOT, N_OCC_PILOT, N_OCC_PILOT};
    CheckCase(rank1, 1, rank1Counts, OK);
    CheckCase(rank2Disjoint, 2, rank2DisjointCounts, OK);
    CheckCase(rank2Occ, 2, rank2OccCounts, LMMSE_OBSERVATION_MODEL_MISMATCH);
    CheckCase(rank3Mixed, 3, rank3MixedCounts, LMMSE_OBSERVATION_MODEL_MISMATCH);
    CheckCase(rank4Occ, 4, rank4OccCounts, LMMSE_OBSERVATION_MODEL_MISMATCH);
    std::puts("[contract] PASS: Rank1/2/3/4, including mixed [399,399,798]");
    return EXIT_SUCCESS;
}
