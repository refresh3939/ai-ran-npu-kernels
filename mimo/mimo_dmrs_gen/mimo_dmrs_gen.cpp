#include "mimo_dmrs_gen.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace airan::mimo_dmrs_gen {
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

int32_t DmrsCinit(const PuschMimoConfig &config, uint32_t symbol)
{
    const uint64_t term = (uint64_t{1} << 17) *
                          (static_cast<uint64_t>(N_SYMBOLS) * config.slot_number + symbol + 1u) *
                          (2u * static_cast<uint64_t>(config.dmrs_scrambling_id) + 1u);
    const uint64_t tail = 2u * static_cast<uint64_t>(config.dmrs_scrambling_id) + config.n_scid;
    return static_cast<int32_t>((term + tail) & 0x7fffffffu);
}

}

Status GetCurrentPortOccSemantics(uint16_t port, PortOccSemantics *semantics)
{
    if (semantics == nullptr) return INVALID_ARGUMENT;
    if (port < 1000 || port > 1003) return UNSUPPORTED_PROFILE;

    const uint32_t port_index = static_cast<uint32_t>(port - 1000u);
    semantics->port_index = port_index;


    semantics->comb_delta = port_index / 2u;
    semantics->wf_odd_negative = port_index & 1u;
    for (uint32_t dmrs = 0; dmrs < CURRENT_DMRS_SYMBOLS; ++dmrs) {
        semantics->wt_negative[dmrs] = 0;
    }
    return OK;
}

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           int32_t *cinit)
{
    if (layout == nullptr || metadata == nullptr || cinit == nullptr) return INVALID_ARGUMENT;
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

    std::memset(metadata, 0, sizeof(*metadata));
    std::fill(cinit, cinit + CINIT_PAD, int32_t{0});
    metadata->magic = META_MAGIC;
    metadata->num_layers = config.num_layers;
    metadata->num_dmrs_symbols = d;
    metadata->output_layer_stride = d * N_DMRS_PAD;
    metadata->output_dmrs_stride = N_DMRS_PAD;

    uint32_t dmrs = 0;
    for (uint32_t symbol = 0; symbol < N_SYMBOLS; ++symbol) {
        if (((config.dmrs_symbol_mask >> symbol) & 1u) == 0) continue;
        metadata->dmrs_symbols[dmrs] = symbol;
        cinit[dmrs] = DmrsCinit(config, symbol);
        ++dmrs;
    }

    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        const uint16_t port = config.dmrs_ports[layer];
        for (uint32_t previous = 0; previous < layer; ++previous) {
            if (config.dmrs_ports[previous] == port) return UNSUPPORTED_PROFILE;
        }
        PortOccSemantics semantics {};
        const Status port_status = GetCurrentPortOccSemantics(port, &semantics);
        if (port_status != OK) return port_status;
        metadata->port_indices[layer] = semantics.port_index;
        metadata->comb_delta[layer] = semantics.comb_delta;
        metadata->wf_odd_negative[layer] = semantics.wf_odd_negative;
        for (uint32_t dmrs_index = 0; dmrs_index < d; ++dmrs_index) {
            metadata->wt_negative[layer * d + dmrs_index] =
                semantics.wt_negative[dmrs_index];
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

Status ValidateOpArgs(const MimoDmrsGenOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION || args.struct_size < sizeof(MimoDmrsGenOpArgsV1) ||
        args.dmrs_re == nullptr || args.dmrs_im == nullptr || args.config == nullptr ||
        args.layout == nullptr || args.stream == nullptr) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    int32_t cinit[CINIT_PAD] = {};
    const Status status = BuildCurrentProfile(*args.config, &expected, &metadata, cinit);
    if (status != OK) return status;
    return SameLayout(expected, *args.layout) ? OK : INVALID_ARGUMENT;
}

void BuildGoldBasis(uint16_t *gmat, uint16_t *g1)
{
    if (gmat == nullptr || g1 == nullptr) return;
    constexpr size_t m = 2 * N_DMRS_RE;
    std::vector<uint8_t> x1(m + GOLD_NC + GOLD_NBITS, 0);
    x1[0] = 1;
    for (size_t n = 0; n < m + GOLD_NC; ++n) x1[n + 31] = x1[n + 3] ^ x1[n];

    const auto pack = [](uint16_t *dst, const uint8_t *sequence) {
        constexpr uint16_t fp16_one = 0x3c00;
        std::fill(dst, dst + N_DMRS_PLANE, uint16_t{0});
        for (size_t k = 0; k < N_DMRS_RE; ++k) {
            dst[k] = sequence[2 * k] ? fp16_one : 0;
            dst[N_DMRS_PAD + k] = sequence[2 * k + 1] ? fp16_one : 0;
        }
    };
    pack(g1, x1.data() + GOLD_NC);

    std::vector<uint8_t> x2(m + GOLD_NC + GOLD_NBITS, 0);
    for (size_t bit = 0; bit < GOLD_NBITS; ++bit) {
        std::fill(x2.begin(), x2.end(), uint8_t{0});
        x2[bit] = 1;
        for (size_t n = 0; n < m + GOLD_NC; ++n) {
            x2[n + 31] = x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n];
        }
        pack(gmat + bit * N_DMRS_PLANE, x2.data() + GOLD_NC);
    }
}

}
