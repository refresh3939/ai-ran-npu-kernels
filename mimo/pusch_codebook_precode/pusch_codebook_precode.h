










#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "../../common/pusch_mimo_types.h"

namespace airan::pusch_precode {

constexpr uint32_t N_SYMBOLS = 14;
constexpr uint32_t N_RB = 133;
constexpr uint32_t N_SC_PER_RB = 12;
constexpr uint32_t N_SC_USED = N_RB * N_SC_PER_RB;
constexpr uint32_t N_SC_PAD = 1664;
constexpr uint32_t N_RE_GRID = N_SYMBOLS * N_SC_PAD;
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t MAX_PORTS = 4;
constexpr uint32_t MAX_PRGS = N_RB;
constexpr uint32_t PRG_MAP_PAD = 144;
constexpr uint32_t MAX_WEIGHT_ELEMS = MAX_PRGS * MAX_PORTS * MAX_LAYERS;

constexpr uint32_t BLOCK_DIM = 8;

constexpr uint32_t TILING_MAGIC = 0x50435031u;
constexpr uint32_t TILING_WORDS = 32;
constexpr uint32_t TILING_BYTES = TILING_WORDS * sizeof(uint32_t);
constexpr uint32_t TILING_WEIGHT_KIND_OFFSET = 9;
constexpr uint32_t WEIGHT_KIND_ZERO = 0;
constexpr uint32_t WEIGHT_KIND_REAL = 1;
constexpr uint32_t WEIGHT_KIND_IMAG = 2;

enum class Status : int32_t {
    kSuccess = 0,
    kNullArgument = -1,
    kAbiMismatch = -2,
    kInvalidConfig = -3,
    kUnsupportedTpmi = -4,
    kRuntimeError = -5,
    kNotInitialized = -6,
    kDetectorCapacityExceeded = -7,
};


struct CodebookPlan {
    uint16_t num_prgs = 0;
    uint16_t num_layers = 0;
    uint16_t num_ports = 0;
    uint16_t weight_count_padded = 0;
    std::array<uint16_t, PRG_MAP_PAD> prg_of_rb{};
    std::array<float, MAX_WEIGHT_ELEMS> weight_re{};
    std::array<float, MAX_WEIGHT_ELEMS> weight_im{};
    std::array<uint32_t, TILING_WORDS> tiling{};
};

PuschMimoConfig MakeDefaultConfig(uint16_t num_layers, uint16_t num_ports);

Status ValidateConfig(const PuschMimoConfig& config, std::string* why = nullptr);






Status BuildMimoDetectLayerPlan(const PuschMimoConfig* configs,
                                size_t num_configs,
                                uint16_t layer_capacity,
                                MimoDetectLayerPlan* plan,
                                std::string* why = nullptr);





Status BuildCodebookPlan(const PuschMimoConfig& config, CodebookPlan* plan,
                         std::string* why = nullptr);


Status Reference(const float* layer_re, const float* layer_im,
                 const PuschMimoConfig& config, float* port_re, float* port_im,
                 std::string* why = nullptr);


uint16_t TpmiCount(uint16_t num_ports, uint16_t num_layers);

}
