#include "qam256_demod_batch.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

using namespace airan::qam256_demod_batch;

namespace {

void Require(bool value, const char *message)
{
    if (!value) {
        std::fprintf(stderr, "[FAIL] %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

PuschMimoConfig MakeConfig(uint16_t layers)
{
    PuschMimoConfig config {};
    config.abi_version = ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = layers;
    config.num_tx_ports = layers == 3 ? 4 : layers;
    config.num_rx_antennas = 64;
    config.qm = Q_M;
    config.num_symbols = N_SYMBOLS;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = N_SYMBOLS;
    config.used_subcarriers = N_SC_USED;
    config.padded_subcarriers = N_SC_GRID_PAD;
    config.dmrs_symbol_mask = DMRS_SYMBOL_MASK;
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    return config;
}

}

int main()
{
    static_assert(std::is_same<PuschMimoConfig, ::airan::PuschMimoConfig>::value,
                  "operator must use common PUSCH MIMO config");
    static_assert(std::is_same<PuschMimoLayout, ::airan::PuschMimoLayout>::value,
                  "operator must use common PUSCH MIMO layout");
    static_assert(BLOCK_DIM == 4 && MAX_LAYERS == 4,
                  "production batch profile is fixed to four cores/rank <= 4");

    constexpr std::array<int32_t, N_DATA_SYMBOLS> expected_symbols =
        {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13};
    for (uint16_t layers = 1; layers <= MAX_LAYERS; ++layers) {
        const PuschMimoConfig config = MakeConfig(layers);
        PuschMimoLayout layout {};
        BatchTilingData tiling {};
        Require(BuildCurrentProfile(config, &layout, &tiling) == OK,
                "valid Rank 1-4 profile rejected");
        Require(layout.num_dmrs_symbols == 2 &&
                    layout.num_data_symbols == N_DATA_SYMBOLS &&
                    layout.num_data_re == N_DATA_RE &&
                    layout.data_stride == LLR_LAYER_STRIDE &&
                    layout.codeword_symbols == layers * N_DATA_RE &&
                    layout.codeword_stride == layers * LLR_LAYER_STRIDE,
                "derived full-grid to symbol-padded layout mismatch");
        Require(tiling.num_layers == layers && tiling.block_dim == BLOCK_DIM &&
                    tiling.grid_layer_stride == GRID_LAYER_STRIDE,
                "rank-dependent native tiling mismatch");
        for (uint32_t ds = 0; ds < N_DATA_SYMBOLS; ++ds) {
            Require(tiling.physical_data_symbols[ds] == expected_symbols[ds],
                    "data-symbol to physical-symbol map mismatch");
        }

        int input = 0;
        int output = 0;
        QamDemod256BatchOpArgsV1 args {
            static_cast<uint16_t>(ABI_VERSION),
            static_cast<uint16_t>(sizeof(QamDemod256BatchOpArgsV1)),
            &input, &input, &input, &output, &config, &layout,
            reinterpret_cast<void *>(1)};
        Require(ValidateOpArgs(args) == OK, "valid public OpArgs rejected");
        ++layout.data_stride;
        Require(ValidateOpArgs(args) == LAYOUT_MISMATCH,
                "mismatched output layout accepted");
    }

    PuschMimoLayout layout {};
    BatchTilingData tiling {};
    PuschMimoConfig bad = MakeConfig(0);
    Require(BuildCurrentProfile(bad, &layout, &tiling) == UNSUPPORTED_PROFILE,
            "Rank 0 accepted");
    bad = MakeConfig(5);
    Require(BuildCurrentProfile(bad, &layout, &tiling) == UNSUPPORTED_PROFILE,
            "Rank 5 accepted");
    bad = MakeConfig(1);
    bad.dmrs_symbol_mask = static_cast<uint16_t>(1u << 2);
    Require(BuildCurrentProfile(bad, &layout, &tiling) == UNSUPPORTED_PROFILE,
            "noncanonical DMRS mask accepted");
    bad = MakeConfig(1);
    bad.num_symbols = 1;
    bad.used_subcarriers = N_DATA_RE;
    bad.padded_subcarriers = 19200;
    Require(BuildCurrentProfile(bad, &layout, &tiling) == UNSUPPORTED_PROFILE,
            "compact QAM input geometry accepted as full-grid");
    bad = MakeConfig(1);
    bad.reserved[0] = 1;
    Require(BuildCurrentProfile(bad, &layout, &tiling) == UNSUPPORTED_PROFILE,
            "nonzero reserved field accepted");
    Require(BuildCurrentProfile(MakeConfig(1), nullptr, &tiling) == INVALID_ARGUMENT,
            "null layout accepted");

    std::puts("HOST_CONTRACT PASS: common ABI, Rank1-4, full-grid layout, negatives");
    return EXIT_SUCCESS;
}
