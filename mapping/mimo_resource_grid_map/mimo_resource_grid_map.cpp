#include "mimo_resource_grid_map.h"

#include <algorithm>
#include <cstring>

namespace airan::mimo_resource_grid_map {
namespace {

uint32_t Popcount16(uint16_t value)
{
    uint32_t count = 0;
    while (value != 0) {
        count += value & 1u;
        value >>= 1u;
    }
    return count;
}

bool ReservedIsZero(const PuschMimoConfig &config)
{
    for (uint32_t value : config.reserved) {
        if (value != 0) return false;
    }
    return true;
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

bool OffsetIsValid(uint32_t byte_offset)
{
    return byte_offset != INVALID_OFFSET && (byte_offset & 1u) == 0 &&
           byte_offset / sizeof(uint16_t) < N_GRID;
}

}

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           uint32_t *data_dst_offset,
                           uint32_t *dmrs_dst_offset)
{
    if (layout == nullptr || metadata == nullptr || data_dst_offset == nullptr ||
        dmrs_dst_offset == nullptr) {
        return INVALID_ARGUMENT;
    }
    if (config.abi_version != ABI_VERSION || config.struct_size < sizeof(PuschMimoConfig) ||
        config.flags != 0 || config.num_layers == 0 || config.num_layers > MAX_LAYERS ||
        (config.num_tx_ports != 1 && config.num_tx_ports != 2 && config.num_tx_ports != 4) ||
        config.num_tx_ports < config.num_layers || config.num_rx_antennas != 64 ||
        config.qm != 8 || config.num_symbols != N_SYMBOLS || config.fft_size != 2048 ||
        config.num_rb != 133 || config.rb_start != 0 || config.start_symbol != 0 ||
        config.num_allocated_symbols != N_SYMBOLS || config.used_subcarriers != N_SC_USED ||
        config.padded_subcarriers != N_SC_PAD || config.dmrs_type != 1 ||
        config.dmrs_length != 1 || config.num_cdm_groups_without_data != 2 ||
        config.n_scid > 1 || config.codeword_index != 0 || config.transform_precoding != 0 ||
        !ReservedIsZero(config)) {
        return UNSUPPORTED_PROFILE;
    }

    const uint32_t d = Popcount16(config.dmrs_symbol_mask);
    if (d != CURRENT_DMRS_SYMBOLS ||
        (config.dmrs_symbol_mask & static_cast<uint16_t>(~((1u << N_SYMBOLS) - 1u))) != 0) {
        return UNSUPPORTED_PROFILE;
    }

    uint32_t comb_delta[MAX_LAYERS] = {};
    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        const uint16_t port = config.dmrs_ports[layer];
        if (port < 1000 || port > 1003) return UNSUPPORTED_PROFILE;
        for (uint32_t previous = 0; previous < layer; ++previous) {
            if (config.dmrs_ports[previous] == port) return UNSUPPORTED_PROFILE;
        }
        comb_delta[layer] = (port - 1000u) / 2u;
    }

    layout->num_dmrs_symbols = static_cast<uint16_t>(d);
    layout->num_data_symbols = static_cast<uint16_t>(N_SYMBOLS - d);
    layout->num_data_re = (N_SYMBOLS - d) * N_SC_USED;
    layout->data_stride = (layout->num_data_re + 127u) / 128u * 128u;
    layout->codeword_symbols = config.num_layers * layout->num_data_re;
    layout->codeword_stride = config.num_layers * layout->data_stride;
    if (layout->num_data_re != N_DATA_RE || layout->data_stride != N_DATA_PAD) {
        return UNSUPPORTED_PROFILE;
    }

    std::fill(data_dst_offset, data_dst_offset + N_DATA_PAD, INVALID_OFFSET);
    std::fill(dmrs_dst_offset, dmrs_dst_offset + DmrsOffsetElems(), INVALID_OFFSET);

    uint32_t data = 0;
    uint32_t dmrs = 0;
    for (uint32_t symbol = 0; symbol < N_SYMBOLS; ++symbol) {
        if (((config.dmrs_symbol_mask >> symbol) & 1u) == 0) {
            for (uint32_t subcarrier = 0; subcarrier < N_SC_USED; ++subcarrier) {
                data_dst_offset[data++] = sizeof(uint16_t) * (symbol * N_SC_PAD + subcarrier);
            }
            continue;
        }
        for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
            uint32_t *dst = dmrs_dst_offset +
                            (layer * CURRENT_DMRS_SYMBOLS + dmrs) * N_DMRS_PAD;
            for (uint32_t pilot = 0; pilot < N_DMRS_RE; ++pilot) {
                const uint32_t subcarrier = 2u * pilot + comb_delta[layer];
                dst[pilot] = sizeof(uint16_t) * (symbol * N_SC_PAD + subcarrier);
            }
        }
        ++dmrs;
    }

    std::memset(metadata, 0, sizeof(*metadata));
    metadata->magic = META_MAGIC;
    metadata->num_layers = config.num_layers;
    metadata->num_dmrs_symbols = d;
    metadata->num_symbols = N_SYMBOLS;
    metadata->used_subcarriers = N_SC_USED;
    metadata->padded_subcarriers = N_SC_PAD;
    metadata->num_data_re = layout->num_data_re;
    metadata->data_stride = layout->data_stride;
    metadata->dmrs_stride = N_DMRS_PAD;
    metadata->grid_stride = N_GRID;
    metadata->invalid_offset = INVALID_OFFSET;
    metadata->dmrs_symbol_mask = config.dmrs_symbol_mask;
    return OK;
}

Status ValidateOpArgs(const MimoResourceGridMapOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(MimoResourceGridMapOpArgsV1) ||
        args.layer_re == nullptr || args.layer_im == nullptr ||
        args.dmrs_re == nullptr || args.dmrs_im == nullptr ||
        args.layer_grid_re == nullptr || args.layer_grid_im == nullptr ||
        args.config == nullptr || args.layout == nullptr || args.stream == nullptr) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    uint32_t data_offset[N_DATA_PAD] = {};
    uint32_t dmrs_offset[MAX_LAYERS * CURRENT_DMRS_SYMBOLS * N_DMRS_PAD] = {};
    const Status status = BuildCurrentProfile(*args.config, &expected, &metadata,
                                               data_offset, dmrs_offset);
    if (status != OK) return status;
    return SameLayout(expected, *args.layout) ? OK : INVALID_ARGUMENT;
}

Status ReferenceMap(const uint16_t *layer_re,
                    const uint16_t *layer_im,
                    const uint16_t *dmrs_re,
                    const uint16_t *dmrs_im,
                    const PuschMimoConfig &config,
                    const PuschMimoLayout &layout,
                    const uint32_t *data_dst_offset,
                    const uint32_t *dmrs_dst_offset,
                    uint16_t *layer_grid_re,
                    uint16_t *layer_grid_im)
{
    if (layer_re == nullptr || layer_im == nullptr || dmrs_re == nullptr || dmrs_im == nullptr ||
        data_dst_offset == nullptr || dmrs_dst_offset == nullptr ||
        layer_grid_re == nullptr || layer_grid_im == nullptr) {
        return INVALID_ARGUMENT;
    }

    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    uint32_t expected_data[N_DATA_PAD] = {};
    uint32_t expected_dmrs[MAX_LAYERS * CURRENT_DMRS_SYMBOLS * N_DMRS_PAD] = {};
    Status status = BuildCurrentProfile(config, &expected, &metadata, expected_data, expected_dmrs);
    if (status != OK) return status;
    if (!SameLayout(expected, layout) ||
        std::memcmp(expected_data, data_dst_offset, sizeof(expected_data)) != 0 ||
        std::memcmp(expected_dmrs, dmrs_dst_offset, sizeof(expected_dmrs)) != 0) {
        return RESOURCE_MISMATCH;
    }

    const size_t grid_elems = GridElems(config.num_layers);
    std::fill(layer_grid_re, layer_grid_re + grid_elems, uint16_t{0});
    std::fill(layer_grid_im, layer_grid_im + grid_elems, uint16_t{0});

    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        const size_t data_base = static_cast<size_t>(layer) * layout.data_stride;
        const size_t grid_base = static_cast<size_t>(layer) * N_GRID;
        for (uint32_t source = 0; source < layout.num_data_re; ++source) {
            if (!OffsetIsValid(data_dst_offset[source])) return RESOURCE_MISMATCH;
            const size_t destination = grid_base + data_dst_offset[source] / sizeof(uint16_t);
            layer_grid_re[destination] = layer_re[data_base + source];
            layer_grid_im[destination] = layer_im[data_base + source];
        }
        for (uint32_t dmrs = 0; dmrs < layout.num_dmrs_symbols; ++dmrs) {
            const size_t source_base =
                (static_cast<size_t>(layer) * layout.num_dmrs_symbols + dmrs) * N_DMRS_PAD;
            const uint32_t *offset = dmrs_dst_offset + source_base;
            for (uint32_t source = 0; source < N_DMRS_RE; ++source) {
                if (!OffsetIsValid(offset[source])) return RESOURCE_MISMATCH;
                const size_t destination = grid_base + offset[source] / sizeof(uint16_t);
                layer_grid_re[destination] = dmrs_re[source_base + source];
                layer_grid_im[destination] = dmrs_im[source_base + source];
            }
        }
    }
    return OK;
}

}
