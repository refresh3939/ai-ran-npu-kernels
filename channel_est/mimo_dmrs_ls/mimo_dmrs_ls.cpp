#include "mimo_dmrs_ls.h"

#include <algorithm>
#include <cstring>

namespace airan::mimo_dmrs_ls {
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

}

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           uint16_t *pilot_count,
                           uint16_t *pilot_sc)
{
    if (layout == nullptr || metadata == nullptr || pilot_count == nullptr || pilot_sc == nullptr) {
        return INVALID_ARGUMENT;
    }
    if (config.abi_version != ABI_VERSION || config.struct_size < sizeof(PuschMimoConfig) ||
        config.num_layers == 0 || config.num_layers > MAX_LAYERS ||
        config.num_tx_ports < config.num_layers || config.num_rx_antennas != NR_CURRENT ||
        config.num_symbols != N_SYMBOLS || config.used_subcarriers != N_SC_USED ||
        config.padded_subcarriers != N_SC_PAD || config.dmrs_type != 1 ||
        config.transform_precoding != 0 || !ReservedIsZero(config)) {
        return UNSUPPORTED_PROFILE;
    }
    const uint32_t d = Popcount16(config.dmrs_symbol_mask);
    if (d != CURRENT_DMRS_SYMBOLS ||
        (config.dmrs_symbol_mask & static_cast<uint16_t>(~((1u << N_SYMBOLS) - 1u))) != 0) {
        return UNSUPPORTED_PROFILE;
    }

    std::memset(metadata, 0, sizeof(*metadata));
    metadata->magic = META_MAGIC;
    metadata->num_rx = NR_CURRENT;
    metadata->num_layers = config.num_layers;
    metadata->num_dmrs_symbols = d;
    metadata->num_symbols = N_SYMBOLS;
    metadata->used_subcarriers = N_SC_USED;
    metadata->padded_subcarriers = N_SC_PAD;
    metadata->dmrs_ref_stride = N_DMRS_REF_PAD;
    metadata->pilot_stride = N_PILOT_PAD;
    metadata->block_dim = BLOCK_DIM;

    uint32_t di = 0;
    for (uint32_t symbol = 0; symbol < N_SYMBOLS; ++symbol) {
        if ((config.dmrs_symbol_mask >> symbol) & 1u) metadata->dmrs_symbols[di++] = symbol;
    }
    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        const uint16_t port = config.dmrs_ports[layer];
        if (port < 1000 || port > 1003) return UNSUPPORTED_PROFILE;
        for (uint32_t previous = 0; previous < layer; ++previous) {
            if (config.dmrs_ports[previous] == port) return UNSUPPORTED_PROFILE;
        }
        metadata->ports[layer] = port;
        metadata->comb_delta[layer] = (port - 1000u) / 2u;
    }
    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        uint32_t sharing = 0;
        for (uint32_t peer = 0; peer < config.num_layers; ++peer) {
            sharing += metadata->comb_delta[peer] == metadata->comb_delta[layer] ? 1u : 0u;
        }
        if (sharing > 2) return UNSUPPORTED_PROFILE;
        metadata->observation_model[layer] = sharing == 2 ? FD_OCC2_399 : COMB2_798;
    }

    std::memset(pilot_count, 0, COUNT_PAD * sizeof(uint16_t));
    std::fill(pilot_sc, pilot_sc + MAX_LAYERS * CURRENT_DMRS_SYMBOLS * N_PILOT_PAD, uint16_t{0});
    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        const bool despread = metadata->observation_model[layer] == FD_OCC2_399;
        const uint16_t count = despread ? N_OCC_PILOT : N_DMRS_RE;
        for (uint32_t dmrs = 0; dmrs < d; ++dmrs) {
            pilot_count[layer * d + dmrs] = count;
            uint16_t *dst = pilot_sc + PilotOffset(layer, dmrs, 0);
            for (uint32_t p = 0; p < count; ++p) {
                dst[p] = static_cast<uint16_t>(metadata->comb_delta[layer] +
                         (despread ? (1u + 4u * p) : (2u * p)));
            }
        }
    }

    layout->num_dmrs_symbols = static_cast<uint16_t>(d);
    layout->num_data_symbols = static_cast<uint16_t>(N_SYMBOLS - d);
    layout->num_data_re = (N_SYMBOLS - d) * N_SC_USED;
    layout->data_stride = (layout->num_data_re + 127u) / 128u * 128u;
    layout->codeword_symbols = config.num_layers * layout->num_data_re;
    layout->codeword_stride = config.num_layers * layout->data_stride;
    return OK;
}

Status ValidateOpArgs(const MimoDmrsLsOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION || args.struct_size < sizeof(MimoDmrsLsOpArgsV1) ||
        args.rx_grid_re == nullptr || args.rx_grid_im == nullptr ||
        args.dmrs_ref_re == nullptr || args.dmrs_ref_im == nullptr ||
        args.h_ls_re == nullptr || args.h_ls_im == nullptr || args.pilot_sc == nullptr ||
        args.pilot_count == nullptr || args.noise_var_rx == nullptr ||
        args.config == nullptr || args.layout == nullptr) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    uint16_t count[COUNT_PAD] = {};
    uint16_t sc[MAX_LAYERS * CURRENT_DMRS_SYMBOLS * N_PILOT_PAD] = {};
    const Status status = BuildCurrentProfile(*args.config, &expected, &metadata, count, sc);
    if (status != OK) return status;
    return std::memcmp(&expected, args.layout, sizeof(expected)) == 0 ? OK : INVALID_ARGUMENT;
}

Status DescribeObservationModels(const uint16_t *pilot_count,
                                 uint32_t num_layers,
                                 uint32_t num_dmrs_symbols,
                                 ObservationModel *models)
{
    if (pilot_count == nullptr || models == nullptr || num_layers == 0 ||
        num_layers > MAX_LAYERS || num_dmrs_symbols != CURRENT_DMRS_SYMBOLS) {
        return INVALID_ARGUMENT;
    }
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        const uint16_t count = pilot_count[layer * num_dmrs_symbols];
        if (count == N_DMRS_RE) models[layer] = COMB2_798;
        else if (count == N_OCC_PILOT) models[layer] = FD_OCC2_399;
        else return UNSUPPORTED_PROFILE;
        for (uint32_t dmrs = 1; dmrs < num_dmrs_symbols; ++dmrs) {
            if (pilot_count[layer * num_dmrs_symbols + dmrs] != count) {
                return UNSUPPORTED_PROFILE;
            }
        }
    }
    return OK;
}

Status ValidateNaturalLmmseContract(const uint16_t *pilot_count,
                                    const uint16_t *pilot_sc,
                                    uint32_t num_layers,
                                    uint32_t num_dmrs_symbols)
{
    if (pilot_sc == nullptr) return INVALID_ARGUMENT;
    ObservationModel models[MAX_LAYERS] = {};
    const Status status = DescribeObservationModels(pilot_count, num_layers,
                                                    num_dmrs_symbols, models);
    if (status != OK) return status;
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        const uint16_t count = pilot_count[layer * num_dmrs_symbols];
        const uint16_t step = models[layer] == FD_OCC2_399 ? 4u : 2u;
        for (uint32_t dmrs = 0; dmrs < num_dmrs_symbols; ++dmrs) {
            const uint16_t *sc = pilot_sc + PilotOffset(layer, dmrs, 0);
            const uint16_t first = sc[0];
            const bool validFirst = models[layer] == FD_OCC2_399 ?
                (first == 1u || first == 2u) : (first == 0u || first == 1u);
            if (!validFirst) return UNSUPPORTED_PROFILE;
            for (uint32_t p = 0; p < count; ++p) {
                if (sc[p] != static_cast<uint16_t>(first + step * p)) {
                    return UNSUPPORTED_PROFILE;
                }
            }
            for (uint32_t p = count; p < N_PILOT_PAD; ++p) {
                if (sc[p] != 0u) return UNSUPPORTED_PROFILE;
            }
        }
    }
    return OK;
}

Status ValidateLmmse798Compatibility(const uint16_t *pilot_count,
                                     uint32_t num_layers,
                                     uint32_t num_dmrs_symbols)
{
    ObservationModel models[MAX_LAYERS] = {};
    const Status status = DescribeObservationModels(pilot_count, num_layers,
                                                    num_dmrs_symbols, models);
    if (status != OK) return status;
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        if (models[layer] != COMB2_798) return LMMSE_OBSERVATION_MODEL_MISMATCH;
    }
    return OK;
}

}
