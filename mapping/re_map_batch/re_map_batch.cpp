#include "re_map_batch.h"

#include <algorithm>
#include <cstring>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_re_map_kernel.h"
#endif

namespace airan::re_map_batch {
namespace {

constexpr uint32_t RADIX_P = 32;
constexpr uint32_t RADIX_Q = 64;

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
        config.num_layers == 0 || config.num_layers > MAX_PORTS ||
        (config.num_tx_ports != 1 && config.num_tx_ports != 2 &&
         config.num_tx_ports != 4) ||
        config.num_tx_ports < config.num_layers || config.num_rx_antennas != 64 ||
        config.qm != 8 || config.num_symbols != N_SYMBOLS ||
        config.fft_size != N_FFT || config.num_rb != 133 || config.rb_start != 0 ||
        config.start_symbol != 0 || config.num_allocated_symbols != N_SYMBOLS ||
        config.used_subcarriers != N_SC_USED ||
        config.padded_subcarriers != N_SC_PAD || config.dmrs_type != 1 ||
        config.dmrs_length != 1 || config.num_cdm_groups_without_data != 2 ||
        config.n_scid > 1 || config.codeword_index != 0 ||
        config.transform_precoding != 0 || config.codebook_enabled > 1 ||
        !ReservedIsZero(config)) {
        return UNSUPPORTED_PROFILE;
    }



    if (config.codebook_enabled == 0 && config.num_tx_ports != config.num_layers) {
        return UNSUPPORTED_PROFILE;
    }

    const uint16_t valid_symbol_mask = static_cast<uint16_t>((1u << N_SYMBOLS) - 1u);
    const uint32_t dmrs_symbols = Popcount16(config.dmrs_symbol_mask);
    if ((config.dmrs_symbol_mask & static_cast<uint16_t>(~valid_symbol_mask)) != 0 ||
        dmrs_symbols != 2) {
        return UNSUPPORTED_PROFILE;
    }
    return OK;
}

uint32_t UsedNaturalBin(uint32_t used_subcarrier)
{
    constexpr uint32_t negative_subcarriers = N_SC_USED / 2;
    if (used_subcarrier < negative_subcarriers) {
        return N_FFT - negative_subcarriers + used_subcarrier;
    }
    return used_subcarrier - negative_subcarriers;
}

uint32_t NaturalBinToS4(uint32_t natural_bin)
{
    return (natural_bin % RADIX_P) * RADIX_Q + natural_bin / RADIX_P;
}

}

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           uint32_t *scatter_index,
                           size_t scatter_index_elems)
{
    if (layout == nullptr || scatter_index == nullptr ||
        scatter_index_elems < SCATTER_INDEX_ELEMS) {
        return INVALID_ARGUMENT;
    }
    const Status status = ValidateConfig(config);
    if (status != OK) return status;

    const uint32_t dmrs_symbols = Popcount16(config.dmrs_symbol_mask);
    layout->num_dmrs_symbols = static_cast<uint16_t>(dmrs_symbols);
    layout->num_data_symbols = static_cast<uint16_t>(N_SYMBOLS - dmrs_symbols);
    layout->num_data_re = layout->num_data_symbols * N_SC_USED;
    layout->data_stride = (layout->num_data_re + 127u) / 128u * 128u;
    layout->codeword_symbols = config.num_layers * layout->num_data_re;
    layout->codeword_stride = config.num_layers * layout->data_stride;

    std::fill(scatter_index, scatter_index + SCATTER_INDEX_ELEMS,
              ZERO_SLOT * sizeof(uint16_t));
    for (uint32_t source = 0; source < N_SC_USED; ++source) {
        const uint32_t destination = NaturalBinToS4(UsedNaturalBin(source));
        scatter_index[destination] = source * sizeof(uint16_t);
    }
    return OK;
}

Status ValidateOpArgs(const ReMapBatchOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(ReMapBatchOpArgsV1) ||
        args.port_grid_re == nullptr || args.port_grid_im == nullptr ||
        args.fft_grid_re == nullptr || args.fft_grid_im == nullptr ||
        args.config == nullptr || args.layout == nullptr || args.stream == nullptr) {
        return INVALID_ARGUMENT;
    }

    PuschMimoLayout expected {};
    uint32_t scatter_index[SCATTER_INDEX_ELEMS] = {};
    const Status status = BuildCurrentProfile(*args.config, &expected, scatter_index,
                                               SCATTER_INDEX_ELEMS);
    if (status != OK) return status;
    return SameLayout(expected, *args.layout) ? OK : LAYOUT_MISMATCH;
}

Status ReferenceMap(const uint16_t *port_grid_re,
                    const uint16_t *port_grid_im,
                    const PuschMimoConfig &config,
                    const PuschMimoLayout &layout,
                    const uint32_t *scatter_index,
                    size_t scatter_index_elems,
                    uint16_t *fft_grid_re,
                    uint16_t *fft_grid_im)
{
    if (port_grid_re == nullptr || port_grid_im == nullptr ||
        scatter_index == nullptr || scatter_index_elems < SCATTER_INDEX_ELEMS ||
        fft_grid_re == nullptr || fft_grid_im == nullptr) {
        return INVALID_ARGUMENT;
    }

    PuschMimoLayout expected_layout {};
    uint32_t expected_index[SCATTER_INDEX_ELEMS] = {};
    Status status = BuildCurrentProfile(config, &expected_layout, expected_index,
                                        SCATTER_INDEX_ELEMS);
    if (status != OK) return status;
    if (!SameLayout(expected_layout, layout) ||
        std::memcmp(expected_index, scatter_index, sizeof(expected_index)) != 0) {
        return LAYOUT_MISMATCH;
    }

    const size_t output_elems = FftGridElems(config.num_tx_ports);
    std::fill(fft_grid_re, fft_grid_re + output_elems, uint16_t{0});
    std::fill(fft_grid_im, fft_grid_im + output_elems, uint16_t{0});
    for (uint32_t port = 0; port < config.num_tx_ports; ++port) {
        const size_t input_port = static_cast<size_t>(port) * GRID_PORT_STRIDE;
        const size_t output_port = static_cast<size_t>(port) * FFT_PORT_STRIDE;
        for (uint32_t symbol = 0; symbol < N_SYMBOLS; ++symbol) {
            const size_t input_symbol = input_port + static_cast<size_t>(symbol) * N_SC_PAD;
            const size_t output_symbol = output_port + static_cast<size_t>(symbol) * N_FFT;
            for (uint32_t destination = 0; destination < N_FFT; ++destination) {
                const uint32_t byte_offset = scatter_index[destination];
                if (byte_offset == ZERO_SLOT * sizeof(uint16_t)) continue;
                if ((byte_offset & 1u) != 0 || byte_offset / sizeof(uint16_t) >= N_SC_USED) {
                    return LAYOUT_MISMATCH;
                }
                const size_t source = input_symbol + byte_offset / sizeof(uint16_t);
                fft_grid_re[output_symbol + destination] = port_grid_re[source];
                fft_grid_im[output_symbol + destination] = port_grid_im[source];
            }
        }
    }
    return OK;
}

Status Enqueue(const ReMapBatchOpArgsV1 &args,
               const void *scatter_index,
               size_t scatter_index_bytes,
               void *workspace,
               size_t workspace_bytes,
               void *tiling,
               size_t tiling_bytes)
{
    const Status status = ValidateOpArgs(args);
    if (status != OK) return status;
    if (scatter_index == nullptr || workspace == nullptr || tiling == nullptr ||
        scatter_index_bytes < SCATTER_INDEX_ELEMS * sizeof(uint32_t) ||
        workspace_bytes < DUMMY_WORKSPACE_BYTES || tiling_bytes < DUMMY_TILING_BYTES) {
        return RESOURCE_TOO_SMALL;
    }

#ifdef ASCENDC_CPU_DEBUG
    (void)scatter_index;
    (void)workspace;
    (void)tiling;
    return LAUNCH_FAILED;
#else
    const auto *input_re = static_cast<const uint8_t *>(args.port_grid_re);
    const auto *input_im = static_cast<const uint8_t *>(args.port_grid_im);
    auto *output_re = static_cast<uint8_t *>(args.fft_grid_re);
    auto *output_im = static_cast<uint8_t *>(args.fft_grid_im);
    const size_t input_port_bytes = GRID_PORT_STRIDE * sizeof(uint16_t);
    const size_t output_port_bytes = FFT_PORT_STRIDE * sizeof(uint16_t);
    const auto stream = static_cast<aclrtStream>(args.stream);

    for (uint32_t port = 0; port < args.config->num_tx_ports; ++port) {
        const uint32_t launch_status = ACLRT_LAUNCH_KERNEL(re_map_kernel)(
            SISO_BLOCK_DIM, stream,
            const_cast<uint8_t *>(input_re + port * input_port_bytes),
            const_cast<uint8_t *>(input_im + port * input_port_bytes),
            const_cast<void *>(scatter_index),
            output_re + port * output_port_bytes,
            output_im + port * output_port_bytes,
            workspace, tiling);
        if (launch_status != ACL_ERROR_NONE) return LAUNCH_FAILED;
    }
    return OK;
#endif
}

}
