#include "layer_map.h"

#include <algorithm>
#include <cstring>

namespace airan::layer_map {
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
        config.num_tx_ports < config.num_layers || config.num_rx_antennas != 64 ||
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
    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        const uint16_t port = config.dmrs_ports[layer];
        if (port < 1000 || port > 1003) return UNSUPPORTED_PROFILE;
        for (uint32_t previous = 0; previous < layer; ++previous) {
            if (config.dmrs_ports[previous] == port) return UNSUPPORTED_PROFILE;
        }
    }
    return OK;
}

}

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
    metadata->num_data_re = layout->num_data_re;
    metadata->data_stride = layout->data_stride;
    metadata->codeword_symbols = layout->codeword_symbols;
    metadata->codeword_stride = layout->codeword_stride;
    metadata->tile_symbols = TILE_SYMBOLS;
    metadata->index_elems_per_layer = INDEX_ELEMS_PER_LAYER;

    std::fill(gather_index, gather_index + MAX_INDEX_ELEMS, uint32_t{0});
    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        uint32_t *index = gather_index + layer * INDEX_ELEMS_PER_LAYER;
        for (uint32_t symbol = 0; symbol < TILE_SYMBOLS; ++symbol) {
            index[symbol] = sizeof(uint16_t) *
                            (symbol * config.num_layers + layer);
        }
    }
    return OK;
}

Status ValidateOpArgs(const LayerMapOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(LayerMapOpArgsV1) ||
        args.d_re == nullptr || args.d_im == nullptr ||
        args.layer_re == nullptr || args.layer_im == nullptr ||
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

Status ReferenceMap(const uint16_t *d_re,
                    const uint16_t *d_im,
                    const PuschMimoConfig &config,
                    const PuschMimoLayout &layout,
                    uint16_t *layer_re,
                    uint16_t *layer_im)
{
    if (d_re == nullptr || d_im == nullptr || layer_re == nullptr ||
        layer_im == nullptr) {
        return INVALID_ARGUMENT;
    }

    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    uint32_t gather_index[MAX_INDEX_ELEMS] = {};
    const Status status = BuildCurrentProfile(config, &expected, &metadata,
                                               gather_index);
    if (status != OK) return status;
    if (!SameLayout(expected, layout)) return LAYOUT_MISMATCH;

    const size_t output_elems = LayerElems(config.num_layers);
    std::fill(layer_re, layer_re + output_elems, uint16_t{0});
    std::fill(layer_im, layer_im + output_elems, uint16_t{0});
    for (uint32_t symbol = 0; symbol < layout.num_data_re; ++symbol) {
        for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
            const size_t source = static_cast<size_t>(symbol) * config.num_layers + layer;
            const size_t destination =
                static_cast<size_t>(layer) * layout.data_stride + symbol;
            layer_re[destination] = d_re[source];
            layer_im[destination] = d_im[source];
        }
    }
    return OK;
}

}
