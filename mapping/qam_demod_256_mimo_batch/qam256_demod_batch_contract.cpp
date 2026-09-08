#include "qam256_demod_batch.h"

#include <cstring>

namespace airan::qam256_demod_batch {
namespace {

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
        config.num_tx_ports < config.num_layers || config.num_rx_antennas == 0 ||
        config.num_rx_antennas > ::airan::PUSCH_MIMO_MAX_PHYSICAL_ANTENNAS ||
        config.qm != Q_M || config.num_symbols != N_SYMBOLS ||
        config.fft_size != 2048 || config.num_rb != 133 || config.rb_start != 0 ||
        config.start_symbol != 0 || config.num_allocated_symbols != N_SYMBOLS ||
        config.used_subcarriers != N_SC_USED ||
        config.padded_subcarriers != N_SC_GRID_PAD ||
        config.dmrs_symbol_mask != DMRS_SYMBOL_MASK || config.dmrs_type != 1 ||
        config.dmrs_length != 1 || config.num_cdm_groups_without_data != 2 ||
        config.n_scid > 1 || config.codeword_index != 0 ||
        config.transform_precoding != 0 || !ReservedIsZero(config)) {
        return UNSUPPORTED_PROFILE;
    }
    return OK;
}

}

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           BatchTilingData *tiling)
{
    if (layout == nullptr || tiling == nullptr) return INVALID_ARGUMENT;
    const Status status = ValidateConfig(config);
    if (status != OK) return status;

    layout->num_dmrs_symbols = 2;
    layout->num_data_symbols = N_DATA_SYMBOLS;
    layout->num_data_re = N_DATA_RE;
    layout->data_stride = LLR_LAYER_STRIDE;
    layout->codeword_symbols = config.num_layers * N_DATA_RE;
    layout->codeword_stride = config.num_layers * LLR_LAYER_STRIDE;

    std::memset(tiling, 0, sizeof(*tiling));
    tiling->n_sc_used = N_SC_USED;
    tiling->n_sc_pad = N_SC_GRID_PAD;
    tiling->n_data_symbols = N_DATA_SYMBOLS;
    tiling->n_data_re = N_DATA_RE;
    tiling->llr_layer_stride = LLR_LAYER_STRIDE;
    tiling->block_dim = BLOCK_DIM;
    tiling->llr_clip = 2560;
    tiling->y_scaled_clip = 1000;
    tiling->num_layers = config.num_layers;
    tiling->grid_layer_stride = GRID_LAYER_STRIDE;
    uint32_t ordinal = 0;
    for (uint32_t symbol = 0; symbol < N_SYMBOLS; ++symbol) {
        if (((DMRS_SYMBOL_MASK >> symbol) & 1u) == 0) {
            tiling->physical_data_symbols[ordinal++] = static_cast<int32_t>(symbol);
        }
    }
    for (; ordinal < N_SYMBOLS; ++ordinal) {
        tiling->physical_data_symbols[ordinal] = -1;
    }
    tiling->group_size = 4;
    return OK;
}

Status ValidateOpArgs(const QamDemod256BatchOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(QamDemod256BatchOpArgsV1) ||
        args.x_re == nullptr || args.x_im == nullptr || args.no_eff == nullptr ||
        args.layer_llr == nullptr || args.config == nullptr ||
        args.layout == nullptr || args.stream == nullptr) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    BatchTilingData tiling {};
    const Status status = BuildCurrentProfile(*args.config, &expected, &tiling);
    if (status != OK) return status;
    return SameLayout(expected, *args.layout) ? OK : LAYOUT_MISMATCH;
}

}
