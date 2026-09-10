#include "channel_est_lmmse.h"

#include <array>
#include <cstdio>

using namespace airan::channel_est_lmmse;

namespace {

PuschMimoConfig Config(std::initializer_list<uint16_t> ports)
{
    PuschMimoConfig config {};
    config.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = static_cast<uint16_t>(ports.size());
    config.num_tx_ports = config.num_layers;
    config.num_rx_antennas = NR;
    config.qm = 8;
    config.num_symbols = N_SYMBOL;
    config.used_subcarriers = N_SC_USED;
    config.padded_subcarriers = N_SC_PAD;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2u) | (1u << 11u));
    config.dmrs_type = 1;
    uint32_t layer = 0;
    for (uint16_t port : ports) config.dmrs_ports[layer++] = port;
    return config;
}

PuschMimoLayout Layout(uint32_t layers)
{
    PuschMimoLayout layout {};
    layout.num_dmrs_symbols = N_DMRS_SYMBOL;
    layout.num_data_symbols = N_SYMBOL - N_DMRS_SYMBOL;
    layout.num_data_re = layout.num_data_symbols * N_SC_USED;
    layout.data_stride = (layout.num_data_re + 127u) / 128u * 128u;
    layout.codeword_symbols = layers * layout.num_data_re;
    layout.codeword_stride = layers * layout.data_stride;
    return layout;
}

void FillPilots(const PuschMimoConfig &config, uint16_t *count, uint16_t *sc)
{
    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        const uint16_t comb = static_cast<uint16_t>((config.dmrs_ports[layer] - 1000u) / 2u);
        uint32_t sharing = 0;
        for (uint32_t peer = 0; peer < config.num_layers; ++peer) {
            sharing += ((config.dmrs_ports[peer] - 1000u) / 2u) == comb ? 1u : 0u;
        }
        const bool occ = sharing == 2;
        const uint16_t n = occ ? N_OCC_PILOT : N_PILOT;
        for (uint32_t dmrs = 0; dmrs < N_DMRS_SYMBOL; ++dmrs) {
            count[layer * N_DMRS_SYMBOL + dmrs] = n;
            uint16_t *dst = sc + (layer * N_DMRS_SYMBOL + dmrs) * N_PILOT_PAD;
            for (uint32_t p = 0; p < n; ++p) {
                dst[p] = comb + (occ ? 1u + 4u * p : 2u * p);
            }
        }
    }
}

bool Expect(Status actual, Status expected, const char *name)
{
    const bool pass = actual == expected;
    std::printf("[%s] %s actual=%d expected=%d\n", pass ? "PASS" : "FAIL", name,
                static_cast<int>(actual), static_cast<int>(expected));
    return pass;
}

}  // namespace

int main()
{
    std::array<uint16_t, MAX_ACTIVE_LAYERS * N_DMRS_SYMBOL> count {};
    std::array<uint16_t, MAX_ACTIVE_LAYERS * N_DMRS_SYMBOL * N_PILOT_PAD> sc {};
    LmmseWeightModelV1 model {};
    bool pass = true;

    auto rank1 = Config({1000});
    auto rank1Layout = Layout(1);
    FillPilots(rank1, count.data(), sc.data());
    pass &= Expect(BuildWeightModel(rank1, rank1Layout, count.data(), sc.data(), &model), OK,
                   "Rank1 build weight model");
    pass &= Expect(ValidateObservationModel(rank1, rank1Layout, count.data(), sc.data(), model), OK,
                   "Rank1 798-point direct contract");

    count.fill(0); sc.fill(0);
    auto rank2 = Config({1000, 1002});
    auto rank2Layout = Layout(2);
    FillPilots(rank2, count.data(), sc.data());
    pass &= Expect(BuildWeightModel(rank2, rank2Layout, count.data(), sc.data(), &model), OK,
                   "Rank2 disjoint-comb build weight model");
    pass &= Expect(ValidateObservationModel(rank2, rank2Layout, count.data(), sc.data(), model), OK,
                   "Rank2 disjoint-comb direct contract");
    ++model.pilot_sc_hash[1][0];
    pass &= Expect(ValidateObservationModel(rank2, rank2Layout, count.data(), sc.data(), model),
                   CE_WEIGHT_MODEL_MISMATCH, "wrong per-layer weights rejected");

    count.fill(0); sc.fill(0);
    auto rank3 = Config({1000, 1001, 1002});
    auto rank3Layout = Layout(3);
    FillPilots(rank3, count.data(), sc.data());
    pass &= Expect(BuildWeightModel(rank3, rank3Layout, count.data(), sc.data(), &model), OK,
                   "Rank3 mixed 399/399/798 build weight model");
    pass &= Expect(ValidateObservationModel(rank3, rank3Layout, count.data(), sc.data(), model), OK,
                   "Rank3 mixed observation contract");

    count.fill(0); sc.fill(0);
    auto rank4 = Config({1000, 1001, 1002, 1003});
    auto rank4Layout = Layout(4);
    FillPilots(rank4, count.data(), sc.data());
    pass &= Expect(BuildWeightModel(rank4, rank4Layout, count.data(), sc.data(), &model), OK,
                   "Rank4 399-point build weight model");
    pass &= Expect(ValidateObservationModel(rank4, rank4Layout, count.data(), sc.data(), model),
                   OK, "Rank4 true 399-point observation contract");
    count[0] = N_PILOT;
    pass &= Expect(ValidateObservationModel(rank4, rank4Layout, count.data(), sc.data(), model),
                   CE_OBSERVATION_MODEL_MISMATCH, "399 cannot masquerade as 798");

    std::printf("=== %s ===\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
