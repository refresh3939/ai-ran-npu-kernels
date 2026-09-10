#include "channel_est_lmmse.h"

#include <cstring>

namespace airan::channel_est_lmmse {
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

bool ReservedIsZero(const uint32_t *values, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (values[i] != 0) return false;
    }
    return true;
}

Status ValidateProfile(const PuschMimoConfig &config, const PuschMimoLayout &layout)
{
    if (config.abi_version != ::airan::PUSCH_MIMO_ABI_VERSION ||
        config.struct_size < sizeof(PuschMimoConfig) ||
        config.num_layers == 0 || config.num_layers > MAX_ACTIVE_LAYERS ||
        config.num_tx_ports < config.num_layers || config.num_rx_antennas != NR ||
        config.num_symbols != N_SYMBOL || config.used_subcarriers != N_SC_USED ||
        config.padded_subcarriers != N_SC_PAD || config.dmrs_type != 1 ||
        config.transform_precoding != 0 || !ReservedIsZero(config.reserved, 8)) {
        return UNSUPPORTED_PROFILE;
    }
    if (Popcount16(config.dmrs_symbol_mask) != N_DMRS_SYMBOL) {
        return UNSUPPORTED_PROFILE;
    }
    if ((config.dmrs_symbol_mask & static_cast<uint16_t>(~((1u << N_SYMBOL) - 1u))) != 0) {
        return UNSUPPORTED_PROFILE;
    }
    const uint32_t dataSymbols = N_SYMBOL - N_DMRS_SYMBOL;
    const uint32_t dataRe = dataSymbols * N_SC_USED;
    const uint32_t dataStride = (dataRe + 127u) / 128u * 128u;
    if (layout.num_dmrs_symbols != N_DMRS_SYMBOL ||
        layout.num_data_symbols != dataSymbols || layout.num_data_re != dataRe ||
        layout.data_stride != dataStride ||
        layout.codeword_symbols != config.num_layers * dataRe ||
        layout.codeword_stride != config.num_layers * dataStride) {
        return INVALID_ARGUMENT;
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

uint32_t SharingCount(const PuschMimoConfig &config, uint32_t layer)
{
    const uint32_t comb = (config.dmrs_ports[layer] - 1000u) / 2u;
    uint32_t sharing = 0;
    for (uint32_t peer = 0; peer < config.num_layers; ++peer) {
        sharing += ((config.dmrs_ports[peer] - 1000u) / 2u) == comb ? 1u : 0u;
    }
    return sharing;
}

}  // namespace

uint64_t PilotScHash(const uint16_t *pilot_sc, uint32_t count)
{
    if (pilot_sc == nullptr) return 0;
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t i = 0; i < count; ++i) {
        const uint16_t value = pilot_sc[i];
        hash ^= static_cast<uint8_t>(value & 0xffu);
        hash *= 1099511628211ull;
        hash ^= static_cast<uint8_t>(value >> 8u);
        hash *= 1099511628211ull;
    }
    return hash;
}

Status BuildWeightModel(const PuschMimoConfig &config,
                        const PuschMimoLayout &layout,
                        const uint16_t *pilot_count,
                        const uint16_t *pilot_sc,
                        LmmseWeightModelV1 *model)
{
    if (pilot_count == nullptr || pilot_sc == nullptr || model == nullptr) {
        return INVALID_ARGUMENT;
    }
    const Status profile = ValidateProfile(config, layout);
    if (profile != OK) return profile;
    std::memset(model, 0, sizeof(*model));
    model->magic = WEIGHT_MODEL_MAGIC;
    model->abi_version = ::airan::PUSCH_MIMO_ABI_VERSION;
    model->struct_size = sizeof(*model);
    model->num_layers = config.num_layers;
    model->num_dmrs_symbols = N_DMRS_SYMBOL;
    model->rank = RANK;
    model->pilot_stride = N_PILOT_PAD;
    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        for (uint32_t dmrs = 0; dmrs < N_DMRS_SYMBOL; ++dmrs) {
            const uint32_t slot = layer * N_DMRS_SYMBOL + dmrs;
            const uint16_t count = pilot_count[slot];
            model->pilot_count[layer][dmrs] = count;
            model->pilot_sc_hash[layer][dmrs] =
                PilotScHash(pilot_sc + slot * N_PILOT_PAD, count);
        }
    }
    return OK;
}

Status ValidateObservationModel(const PuschMimoConfig &config,
                                const PuschMimoLayout &layout,
                                const uint16_t *pilot_count,
                                const uint16_t *pilot_sc,
                                const LmmseWeightModelV1 &model)
{
    if (pilot_count == nullptr || pilot_sc == nullptr) return INVALID_ARGUMENT;
    const Status profile = ValidateProfile(config, layout);
    if (profile != OK) return profile;
    if (model.magic != WEIGHT_MODEL_MAGIC ||
        model.abi_version != ::airan::PUSCH_MIMO_ABI_VERSION ||
        model.struct_size < sizeof(LmmseWeightModelV1) ||
        model.num_layers != config.num_layers ||
        model.num_dmrs_symbols != N_DMRS_SYMBOL || model.rank != RANK ||
        model.pilot_stride != N_PILOT_PAD || !ReservedIsZero(model.reserved, 8)) {
        return CE_WEIGHT_MODEL_MISMATCH;
    }

    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        const uint32_t sharing = SharingCount(config, layer);
        if (sharing == 0 || sharing > 2) return UNSUPPORTED_PROFILE;
        const uint16_t expectedCount = sharing == 2 ? N_OCC_PILOT : N_PILOT;
        const uint16_t comb = static_cast<uint16_t>((config.dmrs_ports[layer] - 1000u) / 2u);
        for (uint32_t dmrs = 0; dmrs < N_DMRS_SYMBOL; ++dmrs) {
            const uint32_t slot = layer * N_DMRS_SYMBOL + dmrs;
            if (pilot_count[slot] != expectedCount) return CE_OBSERVATION_MODEL_MISMATCH;
            const uint16_t *locations = pilot_sc + slot * N_PILOT_PAD;
            for (uint32_t p = 0; p < expectedCount; ++p) {
                const uint16_t expectedSc = sharing == 2 ?
                    static_cast<uint16_t>(comb + 1u + 4u * p) :
                    static_cast<uint16_t>(comb + 2u * p);
                if (locations[p] != expectedSc) return CE_OBSERVATION_MODEL_MISMATCH;
            }
            if (model.pilot_count[layer][dmrs] != expectedCount ||
                model.pilot_sc_hash[layer][dmrs] != PilotScHash(locations, expectedCount)) {
                return CE_WEIGHT_MODEL_MISMATCH;
            }
        }
    }
    return OK;
}

Status ValidateOpArgs(const ChannelEstLmmseOpArgsV1 &args)
{
    if (args.abi_version != ::airan::PUSCH_MIMO_ABI_VERSION ||
        args.struct_size < sizeof(ChannelEstLmmseOpArgsV1) ||
        args.h_ls_re == nullptr || args.h_ls_im == nullptr ||
        args.pilot_count == nullptr || args.pilot_sc == nullptr ||
        args.pilot_count_host == nullptr || args.pilot_sc_host == nullptr ||
        args.weight_model == nullptr || args.h_grid_re == nullptr ||
        args.h_grid_im == nullptr || args.config == nullptr || args.layout == nullptr ||
        args.stream == nullptr) {
        return INVALID_ARGUMENT;
    }
    if (args.config->num_layers != NL || args.config->num_rx_antennas != NR) {
        return CE_BUILD_PROFILE_MISMATCH;
    }
    return ValidateObservationModel(*args.config, *args.layout, args.pilot_count_host,
                                    args.pilot_sc_host, *args.weight_model);
}

}  // namespace airan::channel_est_lmmse
