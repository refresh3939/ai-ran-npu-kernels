#include "pusch_codebook_precode.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>

namespace airan::pusch_precode {
namespace {

struct Coeff {
    int8_t re;
    int8_t im;
};

constexpr Coeff Z{0, 0};
constexpr Coeff O{1, 0};
constexpr Coeff N{-1, 0};
constexpr Coeff J{0, 1};
constexpr Coeff K{0, -1};

constexpr Coeff P2L1[6][2] = {
    {O, Z}, {Z, O}, {O, O}, {O, N}, {O, J}, {O, K},
};

constexpr Coeff P2L2[3][4] = {
    {O, Z, Z, O},
    {O, O, O, N},
    {O, O, J, K},
};


constexpr Coeff P4L1[28][4] = {
    {O,Z,Z,Z}, {Z,O,Z,Z}, {Z,Z,O,Z}, {Z,Z,Z,O},
    {O,Z,O,Z}, {O,Z,N,Z}, {O,Z,J,Z}, {O,Z,K,Z},
    {Z,O,Z,O}, {Z,O,Z,N}, {Z,O,Z,J}, {Z,O,Z,K},
    {O,O,O,O}, {O,O,J,J}, {O,O,N,N}, {O,O,K,K},
    {O,J,O,J}, {O,J,J,N}, {O,J,N,K}, {O,J,K,O},
    {O,N,O,N}, {O,N,J,K}, {O,N,N,O}, {O,N,K,J},
    {O,K,O,K}, {O,K,J,N}, {O,K,N,J}, {O,K,K,N},
};


constexpr Coeff P4L2[22][8] = {
    {O,Z, Z,O, Z,Z, Z,Z},
    {O,Z, Z,Z, Z,O, Z,Z},
    {O,Z, Z,Z, Z,Z, Z,O},
    {Z,Z, O,Z, Z,O, Z,Z},
    {Z,Z, O,Z, Z,Z, Z,O},
    {Z,Z, Z,Z, O,Z, Z,O},
    {O,Z, Z,O, O,Z, Z,K},
    {O,Z, Z,O, O,Z, Z,J},
    {O,Z, Z,O, K,Z, Z,O},
    {O,Z, Z,O, K,Z, Z,N},
    {O,Z, Z,O, N,Z, Z,K},
    {O,Z, Z,O, N,Z, Z,J},
    {O,Z, Z,O, J,Z, Z,O},
    {O,Z, Z,O, J,Z, Z,N},
    {O,O, O,O, O,N, O,N},
    {O,O, O,O, J,K, J,K},
    {O,O, J,J, O,N, J,K},
    {O,O, J,J, J,K, N,O},
    {O,O, N,N, O,N, N,O},
    {O,O, N,N, J,K, K,J},
    {O,O, K,K, O,N, K,J},
    {O,O, K,K, J,K, O,N},
};


constexpr Coeff P4L3[7][12] = {
    {O,Z,Z, Z,O,Z, Z,Z,O, Z,Z,Z},
    {O,Z,Z, Z,O,Z, O,Z,Z, Z,Z,O},
    {O,Z,Z, Z,O,Z, N,Z,Z, Z,Z,O},
    {O,O,O, O,N,O, O,O,N, O,N,N},
    {O,O,O, O,N,O, J,J,K, J,K,K},
    {O,O,O, N,O,N, O,O,N, N,O,O},
    {O,O,O, N,O,N, J,J,K, K,J,J},
};


constexpr Coeff P4L4[5][16] = {
    {O,Z,Z,Z, Z,O,Z,Z, Z,Z,O,Z, Z,Z,Z,O},
    {O,O,Z,Z, Z,Z,O,O, O,N,Z,Z, Z,Z,O,N},
    {O,O,Z,Z, Z,Z,O,O, J,K,Z,Z, Z,Z,J,K},
    {O,O,O,O, O,N,O,N, O,O,N,N, O,N,N,O},
    {O,O,O,O, O,N,O,N, J,J,K,K, J,K,K,J},
};

void Fail(std::string* why, const char* message) {
    if (why != nullptr) *why = message;
}

uint32_t Align16(uint32_t value) {
    return (value + 15u) & ~15u;
}

Status SelectMatrix(uint16_t ports, uint16_t layers, uint16_t tpmi,
                    const Coeff** matrix, float* scale) {
    if (ports == 1 && layers == 1 && tpmi == 0) {
        static constexpr Coeff identity[1] = {O};
        *matrix = identity;
        *scale = 1.0f;
        return Status::kSuccess;
    }
    if (ports == 2 && layers == 1 && tpmi < 6) {
        *matrix = P2L1[tpmi];
        *scale = 1.0f / std::sqrt(2.0f);
        return Status::kSuccess;
    }
    if (ports == 2 && layers == 2 && tpmi < 3) {
        *matrix = P2L2[tpmi];
        *scale = tpmi == 0 ? 1.0f / std::sqrt(2.0f) : 0.5f;
        return Status::kSuccess;
    }
    if (ports == 4 && layers == 1 && tpmi < 28) {
        *matrix = P4L1[tpmi];
        *scale = 0.5f;
        return Status::kSuccess;
    }
    if (ports == 4 && layers == 2 && tpmi < 22) {
        *matrix = P4L2[tpmi];
        *scale = tpmi < 14 ? 0.5f : 1.0f / (2.0f * std::sqrt(2.0f));
        return Status::kSuccess;
    }
    if (ports == 4 && layers == 3 && tpmi < 7) {
        *matrix = P4L3[tpmi];
        *scale = tpmi < 3 ? 0.5f : 1.0f / (2.0f * std::sqrt(3.0f));
        return Status::kSuccess;
    }
    if (ports == 4 && layers == 4 && tpmi < 5) {
        *matrix = P4L4[tpmi];
        if (tpmi == 0) *scale = 0.5f;
        else if (tpmi < 3) *scale = 1.0f / (2.0f * std::sqrt(2.0f));
        else *scale = 0.25f;
        return Status::kSuccess;
    }
    return Status::kUnsupportedTpmi;
}

}

uint16_t TpmiCount(uint16_t ports, uint16_t layers) {
    if (ports == 1 && layers == 1) return 1;
    if (ports == 2 && layers == 1) return 6;
    if (ports == 2 && layers == 2) return 3;
    if (ports == 4 && layers == 1) return 28;
    if (ports == 4 && layers == 2) return 22;
    if (ports == 4 && layers == 3) return 7;
    if (ports == 4 && layers == 4) return 5;
    return 0;
}

PuschMimoConfig MakeDefaultConfig(uint16_t num_layers, uint16_t num_ports) {
    PuschMimoConfig c{};
    c.abi_version = PUSCH_MIMO_ABI_VERSION;
    c.struct_size = sizeof(PuschMimoConfig);
    c.num_layers = num_layers;
    c.num_tx_ports = num_ports;
    c.num_rx_antennas = 64;
    c.qm = 8;
    c.num_symbols = N_SYMBOLS;
    c.fft_size = 2048;
    c.num_rb = N_RB;
    c.start_symbol = 0;
    c.num_allocated_symbols = N_SYMBOLS;
    c.used_subcarriers = N_SC_USED;
    c.padded_subcarriers = N_SC_PAD;
    c.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
    for (uint16_t i = 0; i < MAX_LAYERS; ++i) c.dmrs_ports[i] = 1000 + i;
    c.dmrs_type = 1;
    c.dmrs_length = 1;
    c.num_cdm_groups_without_data = 2;
    c.codebook_enabled = 1;
    c.prg_size_rb = N_RB;
    return c;
}

Status ValidateConfig(const PuschMimoConfig& c, std::string* why) {
    if (c.abi_version != PUSCH_MIMO_ABI_VERSION ||
        c.struct_size < sizeof(PuschMimoConfig)) {
        Fail(why, "PuschMimoConfig ABI version/size mismatch");
        return Status::kAbiMismatch;
    }
    if (c.flags != 0) {
        Fail(why, "config flags must be zero for ABI v1");
        return Status::kInvalidConfig;
    }
    if (c.num_layers < 1 || c.num_layers > MAX_LAYERS ||
        (c.num_tx_ports != 1 && c.num_tx_ports != 2 && c.num_tx_ports != 4) ||
        c.num_layers > c.num_tx_ports) {
        Fail(why, "supported geometry is L=1..4, P in {1,2,4}, L<=P");
        return Status::kInvalidConfig;
    }
    if (c.num_rx_antennas == 0) {
        Fail(why, "num_rx_antennas must be non-zero");
        return Status::kInvalidConfig;
    }
    if (c.num_symbols != N_SYMBOLS || c.num_rb != N_RB || c.rb_start != 0 ||
        c.used_subcarriers != N_SC_USED || c.padded_subcarriers != N_SC_PAD) {
        Fail(why, "this kernel profile requires grid [*,14,1664] with 133 RB/1596 used SC");
        return Status::kInvalidConfig;
    }
    if (c.start_symbol != 0 || c.num_allocated_symbols != N_SYMBOLS) {
        Fail(why, "v1 precoder requires the full 14-symbol slot grid");
        return Status::kInvalidConfig;
    }
    if (c.transform_precoding != 0) {
        Fail(why, "transform precoding is unsupported in v1");
        return Status::kInvalidConfig;
    }
    if (c.prg_size_rb < 1 || c.prg_size_rb > N_RB) {
        Fail(why, "prg_size_rb must be in [1,133]");
        return Status::kInvalidConfig;
    }
    for (uint32_t v : c.reserved) {
        if (v != 0) {
            Fail(why, "reserved config fields must be zero");
            return Status::kInvalidConfig;
        }
    }
    if (c.codebook_enabled == 0) {
        if (c.num_tx_ports != c.num_layers) {
            Fail(why, "non-codebook bypass requires P=L");
            return Status::kInvalidConfig;
        }
        return Status::kSuccess;
    }
    if (c.codebook_enabled != 1 || c.tpmi >= TpmiCount(c.num_tx_ports, c.num_layers)) {
        Fail(why, "TPMI is outside the 38.211 table for this (P,L)");
        return Status::kUnsupportedTpmi;
    }
    return Status::kSuccess;
}

Status BuildMimoDetectLayerPlan(const PuschMimoConfig* configs,
                                size_t num_configs,
                                uint16_t layer_capacity,
                                MimoDetectLayerPlan* plan,
                                std::string* why) {
    if (configs == nullptr || plan == nullptr) {
        Fail(why, "configs/plan is null");
        return Status::kNullArgument;
    }
    if (num_configs == 0 || num_configs > PUSCH_MIMO_MAX_ALLOCATIONS ||
        layer_capacity == 0 || layer_capacity > PUSCH_MIMO_MAX_DETECT_LAYERS) {
        Fail(why, "num_configs and detector layer_capacity must be in [1,16]");
        return Status::kInvalidConfig;
    }

    *plan = MimoDetectLayerPlan{};
    plan->abi_version = PUSCH_MIMO_ABI_VERSION;
    plan->struct_size = sizeof(MimoDetectLayerPlan);
    plan->num_rx_antennas = configs[0].num_rx_antennas;
    plan->layer_capacity = layer_capacity;
    plan->num_allocations = static_cast<uint16_t>(num_configs);

    uint16_t layer_offset = 0;
    for (size_t i = 0; i < num_configs; ++i) {
        Status st = ValidateConfig(configs[i], why);
        if (st != Status::kSuccess) return st;
        if (configs[i].num_rx_antennas != plan->num_rx_antennas) {
            Fail(why, "all PUSCH allocations in one detector plan must use the same NR");
            return Status::kInvalidConfig;
        }
        if (configs[i].slot_number != configs[0].slot_number ||
            configs[i].rb_start != configs[0].rb_start ||
            configs[i].num_rb != configs[0].num_rb ||
            configs[i].start_symbol != configs[0].start_symbol ||
            configs[i].num_allocated_symbols != configs[0].num_allocated_symbols) {
            Fail(why, "all PUSCH allocations in one detector plan must share the same "
                      "time-frequency detection region");
            return Status::kInvalidConfig;
        }
        if (static_cast<uint32_t>(layer_offset) + configs[i].num_layers > layer_capacity) {
            Fail(why, "sum of scheduled PUSCH layers exceeds detector layer capacity");
            return Status::kDetectorCapacityExceeded;
        }

        MimoDetectAllocation& allocation = plan->allocations[i];
        allocation.pusch_index = static_cast<uint16_t>(i);
        allocation.layer_offset = layer_offset;
        allocation.num_layers = configs[i].num_layers;
        allocation.num_tx_ports = configs[i].num_tx_ports;
        layer_offset = static_cast<uint16_t>(layer_offset + configs[i].num_layers);
    }
    plan->total_layers = layer_offset;
    return Status::kSuccess;
}

Status BuildCodebookPlan(const PuschMimoConfig& c, CodebookPlan* plan,
                         std::string* why) {
    if (plan == nullptr) {
        Fail(why, "plan is null");
        return Status::kNullArgument;
    }
    Status st = ValidateConfig(c, why);
    if (st != Status::kSuccess) return st;

    *plan = CodebookPlan{};
    plan->num_layers = c.num_layers;
    plan->num_ports = c.num_tx_ports;
    plan->num_prgs = static_cast<uint16_t>((N_RB + c.prg_size_rb - 1) / c.prg_size_rb);
    for (uint32_t rb = 0; rb < N_RB; ++rb) {
        plan->prg_of_rb[rb] = static_cast<uint16_t>(rb / c.prg_size_rb);
    }

    Coeff identity[MAX_PORTS * MAX_LAYERS]{};
    const Coeff* matrix = nullptr;
    float scale = 1.0f;
    if (c.codebook_enabled == 0) {
        for (uint32_t i = 0; i < c.num_layers; ++i) identity[i * c.num_layers + i] = O;
        matrix = identity;
    } else {
        st = SelectMatrix(c.num_tx_ports, c.num_layers, c.tpmi, &matrix, &scale);
        if (st != Status::kSuccess) {
            Fail(why, "TPMI lookup failed");
            return st;
        }
    }

    const uint32_t matrix_elems = c.num_tx_ports * c.num_layers;
    const uint32_t weight_count = plan->num_prgs * matrix_elems;
    plan->weight_count_padded = static_cast<uint16_t>(Align16(weight_count));
    for (uint32_t g = 0; g < plan->num_prgs; ++g) {
        for (uint32_t i = 0; i < matrix_elems; ++i) {
            const uint32_t dst = g * matrix_elems + i;
            plan->weight_re[dst] = scale * static_cast<float>(matrix[i].re);
            plan->weight_im[dst] = scale * static_cast<float>(matrix[i].im);
        }
    }

    plan->tiling[0] = TILING_MAGIC;
    plan->tiling[1] = PUSCH_MIMO_ABI_VERSION;
    plan->tiling[2] = c.num_layers;
    plan->tiling[3] = c.num_tx_ports;
    plan->tiling[4] = plan->num_prgs;
    plan->tiling[5] = plan->weight_count_padded;
    plan->tiling[6] = N_SYMBOLS;
    plan->tiling[7] = N_SC_PAD;
    plan->tiling[8] = N_SC_USED;
    for (uint32_t i = 0; i < matrix_elems; ++i) {
        plan->tiling[TILING_WEIGHT_KIND_OFFSET + i] = matrix[i].re != 0
            ? WEIGHT_KIND_REAL
            : (matrix[i].im != 0 ? WEIGHT_KIND_IMAG : WEIGHT_KIND_ZERO);
    }
    return Status::kSuccess;
}

Status Reference(const float* layer_re, const float* layer_im,
                 const PuschMimoConfig& c, float* port_re, float* port_im,
                 std::string* why) {
    if (layer_re == nullptr || layer_im == nullptr || port_re == nullptr || port_im == nullptr) {
        Fail(why, "reference tensor pointer is null");
        return Status::kNullArgument;
    }
    CodebookPlan plan;
    Status st = BuildCodebookPlan(c, &plan, why);
    if (st != Status::kSuccess) return st;

    std::fill(port_re, port_re + static_cast<size_t>(c.num_tx_ports) * N_RE_GRID, 0.0f);
    std::fill(port_im, port_im + static_cast<size_t>(c.num_tx_ports) * N_RE_GRID, 0.0f);
    const uint32_t matrix_elems = c.num_tx_ports * c.num_layers;
    for (uint32_t p = 0; p < c.num_tx_ports; ++p) {
        for (uint32_t s = 0; s < N_SYMBOLS; ++s) {
            for (uint32_t k = 0; k < N_SC_USED; ++k) {
                const uint32_t rb = k / N_SC_PER_RB;
                const uint32_t g = plan.prg_of_rb[rb];
                const uint32_t out = (p * N_SYMBOLS + s) * N_SC_PAD + k;
                for (uint32_t l = 0; l < c.num_layers; ++l) {
                    const uint32_t in = (l * N_SYMBOLS + s) * N_SC_PAD + k;
                    const uint32_t wi = g * matrix_elems + p * c.num_layers + l;
                    const float wr = plan.weight_re[wi];
                    const float wj = plan.weight_im[wi];
                    port_re[out] += wr * layer_re[in] - wj * layer_im[in];
                    port_im[out] += wr * layer_im[in] + wj * layer_re[in];
                }
            }
        }
    }
    return Status::kSuccess;
}

}
