#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "mimo_resource_grid_map.h"

#if __has_include("../layer_map/layer_map.h")
#include "../layer_map/layer_map.h"
#define RGM_HAS_LAYER_MAP_CONTRACT 1
#elif __has_include("../layer_map_mimo/layer_map.h")
#include "../layer_map_mimo/layer_map.h"
#define RGM_HAS_LAYER_MAP_CONTRACT 1
#endif

#if __has_include("../mimo_dmrs_gen/mimo_dmrs_gen.h")
#include "../mimo_dmrs_gen/mimo_dmrs_gen.h"
#define RGM_HAS_DMRS_CONTRACT 1
#elif __has_include("../../mimo/mimo_dmrs_gen/mimo_dmrs_gen.h")
#include "../../mimo/mimo_dmrs_gen/mimo_dmrs_gen.h"
#define RGM_HAS_DMRS_CONTRACT 1
#endif

#if __has_include("../pusch_codebook_precode/pusch_codebook_precode.h")
#include "../pusch_codebook_precode/pusch_codebook_precode.h"
#define RGM_HAS_PRECODE_CONTRACT 1
#elif __has_include("../../mimo/pusch_codebook_precode/pusch_codebook_precode.h")
#include "../../mimo/pusch_codebook_precode/pusch_codebook_precode.h"
#define RGM_HAS_PRECODE_CONTRACT 1
#endif

namespace {
namespace rgm = airan::mimo_resource_grid_map;

static_assert(std::is_same<rgm::PuschMimoConfig, airan::PuschMimoConfig>::value,
              "mapper must use the canonical PUSCH MIMO config");
static_assert(std::is_same<rgm::PuschMimoLayout, airan::PuschMimoLayout>::value,
              "mapper must use the canonical PUSCH MIMO layout");
#if RGM_HAS_LAYER_MAP_CONTRACT
static_assert(rgm::N_DATA_RE == airan::layer_map::N_DATA_RE,
              "layer_map valid length must feed the mapper directly");
static_assert(rgm::N_DATA_PAD == airan::layer_map::N_DATA_PAD,
              "layer_map stride must feed the mapper directly");
#endif
#if RGM_HAS_DMRS_CONTRACT
static_assert(rgm::N_DMRS_RE == airan::mimo_dmrs_gen::N_DMRS_RE,
              "DMRS valid length must feed the mapper directly");
static_assert(rgm::N_DMRS_PAD == airan::mimo_dmrs_gen::N_DMRS_PAD,
              "DMRS stride must feed the mapper directly");
static_assert(rgm::CURRENT_DMRS_SYMBOLS == airan::mimo_dmrs_gen::CURRENT_DMRS_SYMBOLS,
              "DMRS symbol count must agree across adjacent operators");
#endif
#if RGM_HAS_PRECODE_CONTRACT
static_assert(rgm::N_SYMBOLS == airan::pusch_precode::N_SYMBOLS,
              "precoder symbol axis must consume mapper output directly");
static_assert(rgm::N_SC_USED == airan::pusch_precode::N_SC_USED,
              "precoder valid subcarrier extent must match mapper output");
static_assert(rgm::N_SC_PAD == airan::pusch_precode::N_SC_PAD,
              "precoder subcarrier stride must match mapper output");
static_assert(rgm::N_GRID == airan::pusch_precode::N_RE_GRID,
              "precoder layer stride must match mapper output");
#endif

struct Case {
    const char *name;
    uint16_t layers;
    uint16_t tx_ports;
    uint16_t ports[4];
    uint16_t dmrs_mask;
};

constexpr Case CASES[] = {
    {"rank1", 1, 1, {1000, 0, 0, 0}, static_cast<uint16_t>((1u << 2) | (1u << 11))},
    {"rank2_shared_comb", 2, 2, {1000, 1001, 0, 0},
     static_cast<uint16_t>((1u << 2) | (1u << 11))},
    {"rank3_reordered", 3, 4, {1003, 1000, 1002, 0},
     static_cast<uint16_t>((1u << 2) | (1u << 11))},
    {"rank4_alt_mask", 4, 4, {1003, 1002, 1001, 1000},
     static_cast<uint16_t>((1u << 1) | (1u << 12))},
};

airan::PuschMimoConfig MakeConfig(const Case &test)
{
    airan::PuschMimoConfig config {};
    config.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = test.layers;
    config.num_tx_ports = test.tx_ports;
    config.num_rx_antennas = 64;
    config.qm = 8;
    config.num_symbols = rgm::N_SYMBOLS;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = rgm::N_SYMBOLS;
    config.used_subcarriers = rgm::N_SC_USED;
    config.padded_subcarriers = rgm::N_SC_PAD;
    config.dmrs_symbol_mask = test.dmrs_mask;
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    std::copy(test.ports, test.ports + test.layers, config.dmrs_ports);
    return config;
}

bool Check(bool condition, const Case &test, const char *detail)
{
    if (condition) return true;
    std::fprintf(stderr, "[FAIL] %s: %s\n", test.name, detail);
    return false;
}

bool RunCase(const Case &test)
{
    const airan::PuschMimoConfig config = MakeConfig(test);
    airan::PuschMimoLayout layout {};
    rgm::KernelMetadata metadata {};
    std::vector<uint32_t> data_offsets(rgm::N_DATA_PAD);
    std::vector<uint32_t> dmrs_offsets(rgm::DmrsOffsetElems());
    if (!Check(rgm::BuildCurrentProfile(config, &layout, &metadata, data_offsets.data(),
                                        dmrs_offsets.data()) == rgm::OK,
               test, "profile rejected")) {
        return false;
    }

    bool ok = true;
    ok &= Check(layout.num_dmrs_symbols == 2 && layout.num_data_symbols == 12 &&
                    layout.num_data_re == rgm::N_DATA_RE &&
                    layout.data_stride == rgm::N_DATA_PAD &&
                    layout.codeword_symbols == test.layers * rgm::N_DATA_RE &&
                    layout.codeword_stride == test.layers * rgm::N_DATA_PAD,
                test, "canonical layout mismatch");
    ok &= Check(metadata.magic == rgm::META_MAGIC && metadata.num_layers == test.layers &&
                    metadata.dmrs_symbol_mask == test.dmrs_mask,
                test, "kernel metadata mismatch");

    uint32_t data_source = 0;
    uint32_t dmrs_symbol = 0;
    for (uint32_t symbol = 0; symbol < rgm::N_SYMBOLS; ++symbol) {
        if (((test.dmrs_mask >> symbol) & 1u) == 0) {
            for (uint32_t sc = 0; sc < rgm::N_SC_USED; ++sc) {
                const uint32_t expected = 2u * (symbol * rgm::N_SC_PAD + sc);
                ok &= Check(data_offsets[data_source++] == expected, test,
                            "data byte-offset order mismatch");
            }
            continue;
        }
        for (uint32_t layer = 0; layer < test.layers; ++layer) {
            const uint32_t delta = (test.ports[layer] - 1000u) / 2u;
            const size_t base =
                (static_cast<size_t>(layer) * rgm::CURRENT_DMRS_SYMBOLS + dmrs_symbol) *
                rgm::N_DMRS_PAD;
            for (uint32_t pilot = 0; pilot < rgm::N_DMRS_RE; ++pilot) {
                const uint32_t expected =
                    2u * (symbol * rgm::N_SC_PAD + 2u * pilot + delta);
                ok &= Check(dmrs_offsets[base + pilot] == expected, test,
                            "DMRS port-comb byte offset mismatch");
            }
        }
        ++dmrs_symbol;
    }
    ok &= Check(data_source == rgm::N_DATA_RE && dmrs_symbol == 2, test,
                "profile element count mismatch");
    ok &= Check(std::all_of(data_offsets.begin() + rgm::N_DATA_RE, data_offsets.end(),
                           [](uint32_t value) { return value == rgm::INVALID_OFFSET; }),
                test, "data padding offset is active");
    for (uint32_t layer = 0; layer < rgm::MAX_LAYERS; ++layer) {
        for (uint32_t d = 0; d < rgm::CURRENT_DMRS_SYMBOLS; ++d) {
            const size_t base =
                (static_cast<size_t>(layer) * rgm::CURRENT_DMRS_SYMBOLS + d) *
                rgm::N_DMRS_PAD;
            const size_t first_invalid = layer < test.layers ? base + rgm::N_DMRS_RE : base;
            ok &= Check(std::all_of(dmrs_offsets.begin() + first_invalid,
                                   dmrs_offsets.begin() + base + rgm::N_DMRS_PAD,
                                   [](uint32_t value) {
                                       return value == rgm::INVALID_OFFSET;
                                   }),
                        test, "DMRS padding or inactive-layer offset is active");
        }
    }

    std::vector<uint16_t> layer_re(rgm::DataElems(test.layers), 0x7e01u);
    std::vector<uint16_t> layer_im(rgm::DataElems(test.layers), 0xfe01u);
    std::vector<uint16_t> dmrs_re(rgm::DmrsElems(test.layers), 0x7d01u);
    std::vector<uint16_t> dmrs_im(rgm::DmrsElems(test.layers), 0xfd01u);
    for (uint32_t layer = 0; layer < test.layers; ++layer) {
        for (uint32_t i = 0; i < rgm::N_DATA_RE; ++i) {
            layer_re[static_cast<size_t>(layer) * rgm::N_DATA_PAD + i] =
                static_cast<uint16_t>(1u + ((layer * 97u + i) % 0x6fffu));
            layer_im[static_cast<size_t>(layer) * rgm::N_DATA_PAD + i] =
                static_cast<uint16_t>(0x8001u + ((layer * 89u + i) % 0x6fffu));
        }
        for (uint32_t d = 0; d < rgm::CURRENT_DMRS_SYMBOLS; ++d) {
            const size_t base =
                (static_cast<size_t>(layer) * rgm::CURRENT_DMRS_SYMBOLS + d) *
                rgm::N_DMRS_PAD;
            for (uint32_t i = 0; i < rgm::N_DMRS_RE; ++i) {
                dmrs_re[base + i] =
                    static_cast<uint16_t>(0x1001u + ((layer * 71u + d * 31u + i) % 0x5fffu));
                dmrs_im[base + i] =
                    static_cast<uint16_t>(0x9001u + ((layer * 67u + d * 29u + i) % 0x5fffu));
            }
        }
    }

    std::vector<uint16_t> grid_re(rgm::GridElems(test.layers), 0x3555u);
    std::vector<uint16_t> grid_im(rgm::GridElems(test.layers), 0xb555u);
    ok &= Check(rgm::ReferenceMap(layer_re.data(), layer_im.data(), dmrs_re.data(),
                                  dmrs_im.data(), config, layout, data_offsets.data(),
                                  dmrs_offsets.data(), grid_re.data(), grid_im.data()) == rgm::OK,
                test, "host reference rejected valid resources");

    std::vector<uint16_t> expected_re(rgm::GridElems(test.layers), 0);
    std::vector<uint16_t> expected_im(rgm::GridElems(test.layers), 0);
    for (uint32_t layer = 0; layer < test.layers; ++layer) {
        const size_t grid_base = static_cast<size_t>(layer) * rgm::N_GRID;
        const size_t data_base = static_cast<size_t>(layer) * rgm::N_DATA_PAD;
        for (uint32_t i = 0; i < rgm::N_DATA_RE; ++i) {
            const size_t dst = grid_base + data_offsets[i] / sizeof(uint16_t);
            expected_re[dst] = layer_re[data_base + i];
            expected_im[dst] = layer_im[data_base + i];
        }
        for (uint32_t d = 0; d < rgm::CURRENT_DMRS_SYMBOLS; ++d) {
            const size_t dmrs_base =
                (static_cast<size_t>(layer) * rgm::CURRENT_DMRS_SYMBOLS + d) *
                rgm::N_DMRS_PAD;
            for (uint32_t i = 0; i < rgm::N_DMRS_RE; ++i) {
                const size_t dst = grid_base + dmrs_offsets[dmrs_base + i] / sizeof(uint16_t);
                expected_re[dst] = dmrs_re[dmrs_base + i];
                expected_im[dst] = dmrs_im[dmrs_base + i];
            }
        }
    }
    ok &= Check(grid_re == expected_re && grid_im == expected_im, test,
                "reference output/layout/padding mismatch");

    rgm::MimoResourceGridMapOpArgsV1 args {
        rgm::ABI_VERSION,
        static_cast<uint16_t>(sizeof(rgm::MimoResourceGridMapOpArgsV1)),
        layer_re.data(), layer_im.data(), dmrs_re.data(), dmrs_im.data(),
        grid_re.data(), grid_im.data(), &config, &layout,
        reinterpret_cast<void *>(static_cast<uintptr_t>(1)),
    };
    ok &= Check(rgm::ValidateOpArgs(args) == rgm::OK, test,
                "public arguments rejected");

    std::printf("[PASS] %s\n", test.name);
    return ok;
}

}

int main()
{
    bool ok = true;
    for (const Case &test : CASES) ok &= RunCase(test);

    airan::PuschMimoConfig invalid = MakeConfig(CASES[1]);
    airan::PuschMimoLayout layout {};
    rgm::KernelMetadata metadata {};
    std::vector<uint32_t> data_offsets(rgm::N_DATA_PAD);
    std::vector<uint32_t> dmrs_offsets(rgm::DmrsOffsetElems());
    invalid.dmrs_ports[1] = invalid.dmrs_ports[0];
    ok &= rgm::BuildCurrentProfile(invalid, &layout, &metadata, data_offsets.data(),
                                   dmrs_offsets.data()) == rgm::UNSUPPORTED_PROFILE;
    invalid = MakeConfig(CASES[0]);
    invalid.dmrs_symbol_mask |= static_cast<uint16_t>(1u << 5);
    ok &= rgm::BuildCurrentProfile(invalid, &layout, &metadata, data_offsets.data(),
                                   dmrs_offsets.data()) == rgm::UNSUPPORTED_PROFILE;
    invalid = MakeConfig(CASES[0]);
    invalid.num_tx_ports = 2;
    invalid.dmrs_ports[0] = 1004;
    ok &= rgm::BuildCurrentProfile(invalid, &layout, &metadata, data_offsets.data(),
                                   dmrs_offsets.data()) == rgm::UNSUPPORTED_PROFILE;
    if (!ok) {
        std::fprintf(stderr, "mimo_resource_grid_map contract test FAILED\n");
        return 1;
    }
    std::printf("mimo_resource_grid_map contract test PASSED\n");
    return 0;
}
