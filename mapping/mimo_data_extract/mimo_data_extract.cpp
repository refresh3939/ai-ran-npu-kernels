#include "mimo_data_extract.h"

#include <algorithm>
#include <cstring>

namespace airan::mimo_data_extract {
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

Status ValidateConfig(const PuschMimoConfig &config)
{
    if (config.abi_version != ABI_VERSION ||
        config.struct_size < sizeof(PuschMimoConfig) || config.flags != 0 ||
        config.num_layers == 0 || config.num_layers > MAX_LAYERS ||
        (config.num_tx_ports != 1 && config.num_tx_ports != 2 &&
         config.num_tx_ports != 4) ||
        config.num_tx_ports < config.num_layers || config.num_rx_antennas != 64 ||
        config.qm != 8 || config.num_symbols != N_SYMBOLS || config.fft_size != 2048 ||
        config.num_rb != 133 || config.rb_start != 0 || config.start_symbol != 0 ||
        config.num_allocated_symbols != N_SYMBOLS ||
        config.used_subcarriers != N_SC_USED ||
        config.padded_subcarriers != N_SC_PAD || config.dmrs_type != 1 ||
        config.dmrs_length != 1 || config.num_cdm_groups_without_data != 2 ||
        config.n_scid > 1 || config.codeword_index != 0 ||
        config.transform_precoding != 0 || !ReservedIsZero(config)) {
        return UNSUPPORTED_PROFILE;
    }
    const uint16_t valid_symbol_mask = static_cast<uint16_t>((1u << N_SYMBOLS) - 1u);
    if ((config.dmrs_symbol_mask & static_cast<uint16_t>(~valid_symbol_mask)) != 0 ||
        Popcount16(config.dmrs_symbol_mask) != CURRENT_DMRS_SYMBOLS) {
        return UNSUPPORTED_PROFILE;
    }
    return OK;
}

}

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata)
{
    if (layout == nullptr || metadata == nullptr) return INVALID_ARGUMENT;
    const Status status = ValidateConfig(config);
    if (status != OK) return status;

    layout->num_dmrs_symbols = CURRENT_DMRS_SYMBOLS;
    layout->num_data_symbols = N_DATA_SYMBOLS;
    layout->num_data_re = N_DATA_RE;
    layout->data_stride = N_DATA_PAD;
    layout->codeword_symbols = config.num_layers * N_DATA_RE;
    layout->codeword_stride = config.num_layers * N_DATA_PAD;

    std::memset(metadata, 0, sizeof(*metadata));
    metadata->magic = META_MAGIC;
    metadata->num_layers = config.num_layers;
    metadata->num_symbols = N_SYMBOLS;
    metadata->used_subcarriers = N_SC_USED;
    metadata->padded_subcarriers = N_SC_PAD;
    metadata->num_data_symbols = N_DATA_SYMBOLS;
    metadata->num_data_re = N_DATA_RE;
    metadata->data_stride = N_DATA_PAD;
    metadata->grid_stride = N_GRID;
    metadata->dmrs_symbol_mask = config.dmrs_symbol_mask;

    uint32_t data_symbol = 0;
    for (uint32_t symbol = 0; symbol < N_SYMBOLS; ++symbol) {
        if (((config.dmrs_symbol_mask >> symbol) & 1u) == 0) {
            metadata->data_symbol_to_grid[data_symbol++] = symbol;
        }
    }
    return data_symbol == N_DATA_SYMBOLS ? OK : UNSUPPORTED_PROFILE;
}

Status ValidateOpArgs(const MimoDataExtractOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(MimoDataExtractOpArgsV1) ||
        args.xhat_re == nullptr || args.xhat_im == nullptr || args.no_eff == nullptr ||
        args.data_re == nullptr || args.data_im == nullptr ||
        args.data_no_eff == nullptr || args.config == nullptr ||
        args.layout == nullptr || args.stream == nullptr) {
        return INVALID_ARGUMENT;
    }
    if (args.input_layout != TENSOR_LAYOUT_FULL_GRID_FP16 ||
        args.output_layout != TENSOR_LAYOUT_COMPACT_DATA_FP16) {
        return TENSOR_LAYOUT_MISMATCH;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    const Status status = BuildCurrentProfile(*args.config, &expected, &metadata);
    if (status != OK) return status;
    return SameLayout(expected, *args.layout) ? OK : LAYOUT_MISMATCH;
}

Status ValidateConsumerLayout(TensorLayout consumer_input_layout)
{
    return consumer_input_layout == TENSOR_LAYOUT_COMPACT_DATA_FP16
        ? OK : TENSOR_LAYOUT_MISMATCH;
}

Status ReferenceExtract(const uint16_t *xhat_re,
                        const uint16_t *xhat_im,
                        const uint16_t *no_eff,
                        const PuschMimoConfig &config,
                        const PuschMimoLayout &layout,
                        uint16_t *data_re,
                        uint16_t *data_im,
                        uint16_t *data_no_eff)
{
    if (xhat_re == nullptr || xhat_im == nullptr || no_eff == nullptr ||
        data_re == nullptr || data_im == nullptr || data_no_eff == nullptr) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    const Status status = BuildCurrentProfile(config, &expected, &metadata);
    if (status != OK) return status;
    if (!SameLayout(expected, layout)) return LAYOUT_MISMATCH;

    const size_t output_elems = DataElems(config.num_layers);
    std::fill(data_re, data_re + output_elems, uint16_t{0});
    std::fill(data_im, data_im + output_elems, uint16_t{0});
    std::fill(data_no_eff, data_no_eff + output_elems, uint16_t{0});
    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        const size_t grid_base = static_cast<size_t>(layer) * N_GRID;
        const size_t data_base = static_cast<size_t>(layer) * layout.data_stride;
        for (uint32_t ordinal = 0; ordinal < layout.num_data_symbols; ++ordinal) {
            const size_t source = grid_base +
                static_cast<size_t>(metadata.data_symbol_to_grid[ordinal]) * N_SC_PAD;
            const size_t destination = data_base + static_cast<size_t>(ordinal) * N_SC_USED;
            std::copy_n(xhat_re + source, N_SC_USED, data_re + destination);
            std::copy_n(xhat_im + source, N_SC_USED, data_im + destination);
            std::copy_n(no_eff + source, N_SC_USED, data_no_eff + destination);
        }
    }
    return OK;
}

}
