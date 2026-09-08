






#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace airan {

constexpr uint16_t PUSCH_MIMO_ABI_VERSION = 1;
constexpr uint16_t PUSCH_MIMO_RUNTIME_ABI_VERSION = 1;
constexpr uint16_t PUSCH_MIMO_MAX_PHYSICAL_ANTENNAS = 64;
constexpr uint16_t PUSCH_MIMO_MAX_DETECT_LAYERS = 16;
constexpr uint16_t PUSCH_MIMO_MAX_DMRS_PORTS = 16;
constexpr uint16_t PUSCH_MIMO_MAX_ALLOCATIONS = 16;

enum class MimoArchitecture : uint16_t {
    kFullDigital = 1,
    kHybrid = 2,
    kAnalog = 3,
};

enum class MimoPrecodingMode : uint16_t {
    kBypass = 0,
    kCodebook = 1,
    kNonCodebook = 2,
};

constexpr uint32_t PUSCH_MIMO_RUNTIME_FLAG_PROFILE_HASH = 1u << 0;
constexpr uint32_t PUSCH_MIMO_RUNTIME_FLAG_ADAPTER_REQUIRED = 1u << 1;
constexpr uint32_t PUSCH_MIMO_RUNTIME_KNOWN_FLAGS =
    PUSCH_MIMO_RUNTIME_FLAG_PROFILE_HASH |
    PUSCH_MIMO_RUNTIME_FLAG_ADAPTER_REQUIRED;

struct PuschMimoConfig {
    uint16_t abi_version;
    uint16_t struct_size;
    uint32_t flags;

    uint16_t num_layers;
    uint16_t num_tx_ports;
    uint16_t num_rx_antennas;
    uint16_t qm;

    uint16_t num_symbols;
    uint16_t fft_size;
    uint16_t num_rb;
    uint16_t rb_start;
    uint16_t slot_number;
    uint8_t start_symbol;
    uint8_t num_allocated_symbols;
    uint16_t used_subcarriers;
    uint16_t padded_subcarriers;

    uint16_t dmrs_symbol_mask;
    uint16_t dmrs_ports[4];
    uint16_t dmrs_scrambling_id;
    uint16_t data_scrambling_id;
    uint16_t rnti;
    uint8_t dmrs_type;
    uint8_t dmrs_length;
    uint8_t dmrs_additional_position;
    uint8_t mapping_type;
    uint8_t num_cdm_groups_without_data;
    uint8_t n_scid;
    uint8_t codeword_index;

    uint8_t transform_precoding;
    uint8_t codebook_enabled;
    uint16_t tpmi;
    uint16_t prg_size_rb;
    uint32_t reserved[8];
};


struct MimoDetectAllocation {
    uint16_t pusch_index;
    uint16_t layer_offset;
    uint16_t num_layers;
    uint16_t num_tx_ports;
};





struct MimoDetectLayerPlan {
    uint16_t abi_version;
    uint16_t struct_size;
    uint16_t num_rx_antennas;
    uint16_t layer_capacity;
    uint16_t total_layers;
    uint16_t num_allocations;
    MimoDetectAllocation allocations[PUSCH_MIMO_MAX_ALLOCATIONS];
    uint32_t reserved[8];
};

struct PuschMimoLayout {
    uint16_t num_dmrs_symbols;
    uint16_t num_data_symbols;
    uint32_t num_data_re;
    uint32_t data_stride;
    uint32_t codeword_symbols;
    uint32_t codeword_stride;
};









struct PuschMimoRuntimeConfig {
    uint16_t abi_version;
    uint16_t struct_size;
    uint32_t flags;
    uint8_t profile_sha256[32];

    uint16_t architecture;
    uint16_t num_tx_antennas;
    uint16_t num_tx_rf_chains;
    uint16_t num_rx_antennas;
    uint16_t num_rx_rf_chains;
    uint16_t num_layers;
    uint16_t num_tx_ports;
    uint16_t rx_bucket;
    uint16_t layer_bucket;
    uint16_t max_rx_antennas;
    uint16_t max_layers;

    uint16_t qm;
    uint16_t num_symbols;
    uint16_t fft_size;
    uint16_t num_rb;
    uint16_t rb_start;
    uint16_t used_subcarriers;
    uint16_t padded_subcarriers;

    uint16_t num_dmrs_symbols;
    uint16_t dmrs_symbol_mask;
    uint16_t dmrs_port_count;
    uint16_t dmrs_ports[PUSCH_MIMO_MAX_DMRS_PORTS];

    uint16_t dmrs_type;
    uint16_t dmrs_length;
    uint16_t num_cdm_groups_without_data;
    uint16_t n_scid;
    uint16_t precoding_mode;
    uint16_t tpmi;
    uint16_t prg_size_rb;
    float channel_gain;
    float awgn_std_int16;

    uint32_t data_re_per_layer;
    uint32_t data_stride_per_layer;
    uint32_t qam_row_stride;
    uint32_t grid_re_per_port;
    uint32_t reserved[10];
};

static_assert(std::is_standard_layout<PuschMimoConfig>::value,
              "PuschMimoConfig must remain a C-compatible standard-layout type");
static_assert(std::is_trivially_copyable<PuschMimoConfig>::value,
              "PuschMimoConfig must remain trivially copyable");
static_assert(std::is_standard_layout<MimoDetectAllocation>::value,
              "MimoDetectAllocation must remain a standard-layout type");
static_assert(std::is_standard_layout<MimoDetectLayerPlan>::value,
              "MimoDetectLayerPlan must remain a standard-layout type");
static_assert(std::is_trivially_copyable<MimoDetectLayerPlan>::value,
              "MimoDetectLayerPlan must remain trivially copyable");
static_assert(std::is_standard_layout<PuschMimoRuntimeConfig>::value,
              "PuschMimoRuntimeConfig must remain standard-layout");
static_assert(std::is_trivially_copyable<PuschMimoRuntimeConfig>::value,
              "PuschMimoRuntimeConfig must remain trivially copyable");
static_assert(sizeof(PuschMimoRuntimeConfig) == 192,
              "PuschMimoRuntimeConfig binary ABI size changed");
static_assert(offsetof(PuschMimoRuntimeConfig, profile_sha256) == 8,
              "PuschMimoRuntimeConfig hash offset changed");
static_assert(offsetof(PuschMimoRuntimeConfig, architecture) == 40,
              "PuschMimoRuntimeConfig topology offset changed");
static_assert(offsetof(PuschMimoRuntimeConfig, dmrs_ports) == 82,
              "PuschMimoRuntimeConfig DMRS offset changed");
static_assert(offsetof(PuschMimoRuntimeConfig, channel_gain) == 128,
              "PuschMimoRuntimeConfig channel offset changed");
static_assert(offsetof(PuschMimoRuntimeConfig, data_re_per_layer) == 136,
              "PuschMimoRuntimeConfig geometry offset changed");

}
