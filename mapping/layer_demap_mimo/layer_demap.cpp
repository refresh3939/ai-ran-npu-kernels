#include "layer_demap.h"

#include <algorithm>
#include <cstring>

namespace airan::layer_demap {
namespace {

bool ReservedIsZero(const PuschMimoConfig &config)
{
    for (uint32_t value : config.reserved) {
        if (value != 0) return false;
    }
    return true;
}

uint32_t Popcount16(uint16_t value)
{
    uint32_t count = 0;
    while (value != 0) {
        count += value & 1u;
        value >>= 1u;
    }
    return count;
}

bool SameLayout(const PuschMimoLayout &left, const PuschMimoLayout &right)
{
    return left.num_dmrs_symbols == right.num_dmrs_symbols &&
           left.num_data_symbols == right.num_data_symbols &&
           left.num_data_re == right.num_data_re &&
           left.data_stride == right.data_stride &&
           left.codeword_symbols == right.codeword_symbols &&
           left.codeword_stride == right.codeword_stride;
}

Status ValidateConfig(const PuschMimoConfig &config)
{
    if (config.abi_version != ABI_VERSION ||
        config.struct_size < sizeof(PuschMimoConfig) || config.flags != 0 ||
        config.num_layers == 0 || config.num_layers > MAX_LAYERS ||
        (config.num_tx_ports != 1 && config.num_tx_ports != 2 &&
         config.num_tx_ports != 4) ||
        config.num_tx_ports < config.num_layers || config.num_rx_antennas == 0 ||
        config.num_rx_antennas > ::airan::PUSCH_MIMO_MAX_PHYSICAL_ANTENNAS ||
        config.qm != Q_M || config.num_symbols != 14 || config.fft_size != 2048 ||
        config.num_rb != 133 || config.rb_start != 0 || config.start_symbol != 0 ||
        config.num_allocated_symbols != 14 || config.used_subcarriers != 1596 ||
        config.padded_subcarriers != 1664 || config.dmrs_type != 1 ||
        config.dmrs_length != 1 || config.num_cdm_groups_without_data != 2 ||
        config.n_scid > 1 || config.codeword_index != 0 ||
        config.transform_precoding != 0 || !ReservedIsZero(config)) {
        return UNSUPPORTED_PROFILE;
    }
    const uint32_t dmrs = Popcount16(config.dmrs_symbol_mask);
    if (dmrs != 2 ||
        (config.dmrs_symbol_mask & static_cast<uint16_t>(~((1u << 14) - 1u))) != 0) {
        return UNSUPPORTED_PROFILE;
    }
    return OK;
}

}  // namespace

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           uint32_t *gather_index)
{
    if (layout == nullptr || metadata == nullptr || gather_index == nullptr) {
        return INVALID_ARGUMENT;
    }
    const Status status = ValidateConfig(config);
    if (status != OK) return status;

    layout->num_dmrs_symbols = 2;
    layout->num_data_symbols = 12;
    layout->num_data_re = N_DATA_RE;
    layout->data_stride = N_DATA_PAD;
    layout->codeword_symbols = config.num_layers * N_DATA_RE;
    layout->codeword_stride = config.num_layers * N_DATA_PAD;

    std::memset(metadata, 0, sizeof(*metadata));
    metadata->magic = META_MAGIC;
    metadata->num_layers = config.num_layers;
    metadata->qm = config.qm;
    metadata->num_data_re = layout->num_data_re;
    metadata->data_stride = layout->data_stride;
    metadata->codeword_symbols = layout->codeword_symbols;
    metadata->codeword_stride = layout->codeword_stride;
    metadata->input_symbol_stride = N_SC_LLR_PAD;
    metadata->symbols_per_group = SYMBOLS_PER_GROUP;
    metadata->group_data_re = GROUP_DATA_RE;
    metadata->gather_chunk = GATHER_CHUNK;
    metadata->gather_index_elems = config.num_layers * GATHER_CHUNK;

    std::fill(gather_index, gather_index + MAX_INDEX_ELEMS, uint32_t{0});
    // Each UB source group is [L,GROUP_DATA_RE]. Gather a 128-RE window to
    // symbol-major [re,L]; passing source[chunk_base] selects each window.
    for (uint32_t symbol = 0; symbol < GATHER_CHUNK; ++symbol) {
        for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
            gather_index[symbol * config.num_layers + layer] =
                sizeof(int16_t) * (layer * GROUP_DATA_RE + symbol);
        }
    }
    return OK;
}

Status ValidateOpArgs(const LayerDemapOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(LayerDemapOpArgsV1) ||
        args.layer_llr == nullptr || args.cw_llr == nullptr ||
        args.config == nullptr || args.layout == nullptr || args.stream == nullptr) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    uint32_t gather_index[MAX_INDEX_ELEMS] = {};
    const Status status = BuildCurrentProfile(*args.config, &expected, &metadata,
                                               gather_index);
    if (status != OK) return status;
    return SameLayout(expected, *args.layout) ? OK : LAYOUT_MISMATCH;
}

Status ReferenceDemap(const int16_t *layer_llr,
                      const PuschMimoConfig &config,
                      const PuschMimoLayout &layout,
                      int16_t *cw_llr)
{
    if (layer_llr == nullptr || cw_llr == nullptr) return INVALID_ARGUMENT;

    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    uint32_t gather_index[MAX_INDEX_ELEMS] = {};
    const Status status = BuildCurrentProfile(config, &expected, &metadata, gather_index);
    if (status != OK) return status;
    if (!SameLayout(expected, layout)) return LAYOUT_MISMATCH;

    std::fill(cw_llr, cw_llr + CodewordElems(config.num_layers), int16_t{0});
    for (uint32_t q = 0; q < config.qm; ++q) {
        const size_t output_base = static_cast<size_t>(q) * layout.codeword_stride;
        for (uint32_t data_symbol = 0; data_symbol < N_DATA_SYMBOLS; ++data_symbol) {
            for (uint32_t sc = 0; sc < N_SC_USED; ++sc) {
                const uint32_t data_re = data_symbol * N_SC_USED + sc;
                for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
                    const size_t input =
                        (static_cast<size_t>(layer) * config.qm + q) * layout.data_stride +
                        static_cast<size_t>(data_symbol) * N_SC_LLR_PAD + sc;
                    cw_llr[output_base + static_cast<size_t>(data_re) * config.num_layers + layer] =
                        layer_llr[input];
                }
            }
        }
    }
    return OK;
}

}  // namespace airan::layer_demap
