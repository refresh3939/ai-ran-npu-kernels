#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

#include "../pusch_codebook_precode.h"

namespace pc = airan::pusch_precode;

namespace {

bool Near(float a, float b) { return std::fabs(a - b) < 1.0e-6f; }

float Re(const pc::CodebookPlan& p, uint32_t port, uint32_t layer) {
    return p.weight_re[port * p.num_layers + layer];
}
float Im(const pc::CodebookPlan& p, uint32_t port, uint32_t layer) {
    return p.weight_im[port * p.num_layers + layer];
}

}

int main() {
    std::string why;
    pc::CodebookPlan plan;


    constexpr std::array<uint16_t, 6> ranks{{4, 4, 3, 2, 2, 1}};
    constexpr std::array<uint16_t, 6> ports{{4, 4, 4, 2, 2, 1}};
    std::array<airan::PuschMimoConfig, ranks.size()> scheduled{};
    for (size_t i = 0; i < scheduled.size(); ++i) {
        scheduled[i] = pc::MakeDefaultConfig(ranks[i], ports[i]);
    }
    airan::MimoDetectLayerPlan detect_plan{};
    assert(pc::BuildMimoDetectLayerPlan(scheduled.data(), scheduled.size(), 16,
                                        &detect_plan, &why) == pc::Status::kSuccess);
    assert(detect_plan.num_rx_antennas == 64);
    assert(detect_plan.total_layers == 16);
    assert(detect_plan.num_allocations == scheduled.size());
    constexpr std::array<uint16_t, 6> offsets{{0, 4, 8, 11, 13, 15}};
    for (size_t i = 0; i < scheduled.size(); ++i) {
        assert(detect_plan.allocations[i].pusch_index == i);
        assert(detect_plan.allocations[i].layer_offset == offsets[i]);
        assert(detect_plan.allocations[i].num_layers == ranks[i]);
        assert(detect_plan.allocations[i].num_tx_ports == ports[i]);
    }
    scheduled[5] = pc::MakeDefaultConfig(2, 2);
    assert(pc::BuildMimoDetectLayerPlan(scheduled.data(), scheduled.size(), 16,
                                        &detect_plan, &why) ==
           pc::Status::kDetectorCapacityExceeded);
    scheduled[5] = pc::MakeDefaultConfig(1, 1);
    scheduled[5].num_rx_antennas = 32;
    assert(pc::BuildMimoDetectLayerPlan(scheduled.data(), scheduled.size(), 16,
                                        &detect_plan, &why) == pc::Status::kInvalidConfig);
    scheduled[5] = pc::MakeDefaultConfig(1, 1);
    scheduled[5].slot_number = 1;
    assert(pc::BuildMimoDetectLayerPlan(scheduled.data(), scheduled.size(), 16,
                                        &detect_plan, &why) == pc::Status::kInvalidConfig);


    for (uint16_t ports : {uint16_t{1}, uint16_t{2}, uint16_t{4}}) {
        for (uint16_t layers = 1; layers <= ports; ++layers) {
            const uint16_t count = pc::TpmiCount(ports, layers);
            assert(count > 0);
            for (uint16_t tpmi = 0; tpmi < count; ++tpmi) {
                auto c = pc::MakeDefaultConfig(layers, ports);
                c.tpmi = tpmi;
                assert(pc::BuildCodebookPlan(c, &plan, &why) == pc::Status::kSuccess);
            }
        }
    }


    auto c = pc::MakeDefaultConfig(2, 2);
    c.tpmi = 2;
    assert(pc::BuildCodebookPlan(c, &plan, &why) == pc::Status::kSuccess);
    assert(Near(Re(plan, 0, 0), 0.5f) && Near(Re(plan, 0, 1), 0.5f));
    assert(Near(Im(plan, 1, 0), 0.5f) && Near(Im(plan, 1, 1), -0.5f));


    c = pc::MakeDefaultConfig(3, 4);
    c.tpmi = 6;
    assert(pc::BuildCodebookPlan(c, &plan, &why) == pc::Status::kSuccess);
    const float s3 = 1.0f / (2.0f * std::sqrt(3.0f));
    assert(Near(Im(plan, 3, 0), -s3));
    assert(Near(Im(plan, 3, 1), s3));
    assert(Near(Im(plan, 3, 2), s3));


    c = pc::MakeDefaultConfig(1, 4);
    c.tpmi = 25;
    assert(pc::BuildCodebookPlan(c, &plan, &why) == pc::Status::kSuccess);
    assert(Near(Re(plan, 0, 0), 0.5f));
    assert(Near(Im(plan, 1, 0), -0.5f));
    assert(Near(Im(plan, 2, 0), 0.5f));
    assert(Near(Re(plan, 3, 0), -0.5f));


    c = pc::MakeDefaultConfig(4, 4);
    c.tpmi = 4;
    assert(pc::BuildCodebookPlan(c, &plan, &why) == pc::Status::kSuccess);
    assert(Near(Im(plan, 3, 0), 0.25f));
    assert(Near(Im(plan, 3, 1), -0.25f));
    assert(Near(Im(plan, 3, 2), -0.25f));
    assert(Near(Im(plan, 3, 3), 0.25f));
    assert(plan.tiling[pc::TILING_WEIGHT_KIND_OFFSET + 0] == pc::WEIGHT_KIND_REAL);
    assert(plan.tiling[pc::TILING_WEIGHT_KIND_OFFSET + 8] == pc::WEIGHT_KIND_IMAG);


    c = pc::MakeDefaultConfig(2, 4);
    c.prg_size_rb = 5;
    assert(pc::BuildCodebookPlan(c, &plan, &why) == pc::Status::kSuccess);
    assert(plan.num_prgs == 27);
    assert(plan.prg_of_rb[0] == 0 && plan.prg_of_rb[4] == 0);
    assert(plan.prg_of_rb[5] == 1 && plan.prg_of_rb[132] == 26);
    assert((plan.weight_count_padded % 16) == 0);


    c = pc::MakeDefaultConfig(4, 4);
    c.codebook_enabled = 0;
    assert(pc::BuildCodebookPlan(c, &plan, &why) == pc::Status::kSuccess);
    for (uint32_t p = 0; p < 4; ++p) {
        for (uint32_t l = 0; l < 4; ++l) {
            assert(Near(Re(plan, p, l), p == l ? 1.0f : 0.0f));
            assert(Near(Im(plan, p, l), 0.0f));
        }
    }
    c = pc::MakeDefaultConfig(2, 4);
    c.codebook_enabled = 0;
    assert(pc::ValidateConfig(c, &why) == pc::Status::kInvalidConfig);

    c = pc::MakeDefaultConfig(2, 2);
    c.tpmi = pc::TpmiCount(2, 2);
    assert(pc::ValidateConfig(c, &why) == pc::Status::kUnsupportedTpmi);

    c = pc::MakeDefaultConfig(2, 2);
    c.flags = 1;
    assert(pc::ValidateConfig(c, &why) == pc::Status::kInvalidConfig);
    c = pc::MakeDefaultConfig(2, 2);
    c.rb_start = 1;
    assert(pc::ValidateConfig(c, &why) == pc::Status::kInvalidConfig);

    std::puts("codebook host tests: PASS");
    return 0;
}
