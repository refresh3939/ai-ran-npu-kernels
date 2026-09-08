#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "../pusch_codebook_precode.h"
#include "../../../mapping/mimo_resource_grid_map/mimo_resource_grid_map.h"
#include "../../../mapping/re_map_batch/re_map_batch.h"

namespace pc = airan::pusch_precode;
namespace rgm = airan::mimo_resource_grid_map;
namespace remap = airan::re_map_batch;

namespace {

struct Case {
    uint16_t ports;
    uint16_t layers;
    uint16_t tpmi;
    bool codebook;
};

float HalfToFloat(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t fraction = bits & 0x03ffu;
    uint32_t value = 0;
    if (exponent == 0) {
        if (fraction == 0) {
            value = sign;
        } else {
            exponent = 1;
            while ((fraction & 0x0400u) == 0) {
                fraction <<= 1;
                --exponent;
            }
            fraction &= 0x03ffu;
            value = sign | ((exponent + 112u) << 23) | (fraction << 13);
        }
    } else if (exponent == 31) {
        value = sign | 0x7f800000u | (fraction << 13);
    } else {
        value = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

uint16_t FloatToHalf(float input) {
    uint32_t value = 0;
    std::memcpy(&value, &input, sizeof(value));
    const uint16_t sign = static_cast<uint16_t>((value >> 16) & 0x8000u);
    const uint32_t abs = value & 0x7fffffffu;
    if (abs >= 0x7f800000u) return static_cast<uint16_t>(sign | 0x7c00u);
    if (abs < 0x33000000u) return sign;
    if (abs < 0x38800000u) {
        const uint32_t shift = 113u - (abs >> 23);
        const uint32_t mantissa = (abs & 0x7fffffu) | 0x800000u;
        return static_cast<uint16_t>(sign | ((mantissa + (1u << (shift + 12u))) >> (shift + 13u)));
    }
    const uint32_t rounded = abs + 0x1000u;
    if (rounded >= 0x47800000u) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | ((rounded - 0x38000000u) >> 13));
}

bool SameLayout(const airan::PuschMimoLayout& a, const airan::PuschMimoLayout& b) {
    return a.num_dmrs_symbols == b.num_dmrs_symbols &&
           a.num_data_symbols == b.num_data_symbols &&
           a.num_data_re == b.num_data_re && a.data_stride == b.data_stride &&
           a.codeword_symbols == b.codeword_symbols &&
           a.codeword_stride == b.codeword_stride;
}

void RunCase(const Case& test) {
    airan::PuschMimoConfig config = pc::MakeDefaultConfig(test.layers, test.ports);
    config.tpmi = test.tpmi;
    config.codebook_enabled = test.codebook ? 1 : 0;

    airan::PuschMimoLayout upstream_layout{};
    rgm::KernelMetadata metadata{};
    std::vector<uint32_t> data_offsets(rgm::N_DATA_PAD);
    std::vector<uint32_t> dmrs_offsets(rgm::DmrsOffsetElems());
    assert(rgm::BuildCurrentProfile(config, &upstream_layout, &metadata,
                                    data_offsets.data(), dmrs_offsets.data()) == rgm::OK);

    std::vector<uint16_t> layer_data_re(rgm::DataElems(test.layers), 0x3c00u);
    std::vector<uint16_t> layer_data_im(rgm::DataElems(test.layers), 0xb800u);
    std::vector<uint16_t> dmrs_re(rgm::DmrsElems(test.layers), 0x3400u);
    std::vector<uint16_t> dmrs_im(rgm::DmrsElems(test.layers), 0xbc00u);
    std::vector<uint16_t> layer_grid_re(rgm::GridElems(test.layers), 0x7e00u);
    std::vector<uint16_t> layer_grid_im(rgm::GridElems(test.layers), 0x7e00u);
    assert(rgm::ReferenceMap(layer_data_re.data(), layer_data_im.data(),
                             dmrs_re.data(), dmrs_im.data(), config, upstream_layout,
                             data_offsets.data(), dmrs_offsets.data(),
                             layer_grid_re.data(), layer_grid_im.data()) == rgm::OK);

    pc::CodebookPlan plan{};
    std::string why;
    assert(pc::BuildCodebookPlan(config, &plan, &why) == pc::Status::kSuccess);
    assert(plan.num_layers == test.layers && plan.num_ports == test.ports);
    assert(layer_grid_re.size() == static_cast<size_t>(test.layers) * pc::N_RE_GRID);

    std::vector<float> layer_re(layer_grid_re.size());
    std::vector<float> layer_im(layer_grid_im.size());
    for (size_t i = 0; i < layer_re.size(); ++i) {
        layer_re[i] = HalfToFloat(layer_grid_re[i]);
        layer_im[i] = HalfToFloat(layer_grid_im[i]);
    }
    std::vector<float> port_re(static_cast<size_t>(test.ports) * pc::N_RE_GRID);
    std::vector<float> port_im(static_cast<size_t>(test.ports) * pc::N_RE_GRID);
    assert(pc::Reference(layer_re.data(), layer_im.data(), config,
                         port_re.data(), port_im.data(), &why) == pc::Status::kSuccess);

    if (!test.codebook) {
        assert(test.layers == test.ports);
        assert(port_re == layer_re);
        assert(port_im == layer_im);
    }

    airan::PuschMimoLayout downstream_layout{};
    std::array<uint32_t, remap::SCATTER_INDEX_ELEMS> scatter{};
    assert(remap::BuildCurrentProfile(config, &downstream_layout, scatter.data(),
                                      scatter.size()) == remap::OK);
    assert(SameLayout(upstream_layout, downstream_layout));

    std::vector<uint16_t> port_half_re(port_re.size());
    std::vector<uint16_t> port_half_im(port_im.size());
    for (size_t i = 0; i < port_re.size(); ++i) {
        port_half_re[i] = FloatToHalf(port_re[i]);
        port_half_im[i] = FloatToHalf(port_im[i]);
    }
    std::vector<uint16_t> fft_re(remap::FftGridElems(test.ports), 0xffffu);
    std::vector<uint16_t> fft_im(remap::FftGridElems(test.ports), 0xffffu);
    assert(remap::ReferenceMap(port_half_re.data(), port_half_im.data(), config,
                               downstream_layout, scatter.data(), scatter.size(),
                               fft_re.data(), fft_im.data()) == remap::OK);
    assert(fft_re.size() == static_cast<size_t>(test.ports) * remap::FFT_PORT_STRIDE);
}

}

int main() {
    static_assert(std::is_same<rgm::PuschMimoConfig, airan::PuschMimoConfig>::value,
                  "resource-grid mapper must consume the shared config type");
    static_assert(std::is_same<remap::PuschMimoConfig, airan::PuschMimoConfig>::value,
                  "RE mapper must consume the shared config type");
    static_assert(rgm::N_GRID == pc::N_RE_GRID, "upstream layer-grid stride mismatch");
    static_assert(remap::GRID_PORT_STRIDE == pc::N_RE_GRID,
                  "downstream port-grid stride mismatch");

    constexpr std::array<Case, 8> cases{{
        {1, 1, 0, false}, {1, 1, 0, true}, {2, 1, 4, true},
        {2, 2, 2, true}, {2, 2, 0, false}, {4, 2, 21, true},
        {4, 3, 6, true}, {4, 4, 4, true},
    }};
    for (const Case& test : cases) RunCase(test);

    std::string why;
    auto invalid_bypass = pc::MakeDefaultConfig(3, 4);
    invalid_bypass.codebook_enabled = 0;
    assert(pc::ValidateConfig(invalid_bypass, &why) == pc::Status::kInvalidConfig);
    airan::PuschMimoLayout layout{};
    std::array<uint32_t, remap::SCATTER_INDEX_ELEMS> scatter{};
    assert(remap::BuildCurrentProfile(invalid_bypass, &layout, scatter.data(),
                                      scatter.size()) == remap::UNSUPPORTED_PROFILE);

    auto invalid_tpmi = pc::MakeDefaultConfig(3, 4);
    invalid_tpmi.tpmi = pc::TpmiCount(4, 3);
    assert(pc::ValidateConfig(invalid_tpmi, &why) == pc::Status::kUnsupportedTpmi);

    auto invalid_flags = pc::MakeDefaultConfig(4, 4);
    invalid_flags.flags = 1;
    assert(pc::ValidateConfig(invalid_flags, &why) == pc::Status::kInvalidConfig);

    std::puts("mimo_resource_grid_map -> pusch_codebook_precode -> re_map_batch contract: PASS");
    return 0;
}
