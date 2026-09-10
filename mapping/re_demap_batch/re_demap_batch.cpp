#include "re_demap_batch.h"

#include <algorithm>
#include <cstring>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_re_demap_batch_kernel.h"
#endif

namespace airan::re_demap_batch {
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
        config.num_layers == 0 || config.num_layers > 4 ||
        config.num_rx_antennas == 0 ||
        config.num_rx_antennas > MAX_RX_ANTENNAS ||
        config.qm != 8 || config.num_symbols != N_SYMBOLS ||
        config.fft_size != N_FFT || config.num_rb != N_RB ||
        config.rb_start != 0 || config.start_symbol != 0 ||
        config.num_allocated_symbols != N_SYMBOLS ||
        config.used_subcarriers != N_SC_USED ||
        config.padded_subcarriers != N_SC_PAD ||
        config.dmrs_symbol_mask != ((1u << 2) | (1u << 11)) ||
        config.transform_precoding != 0 || !ReservedIsZero(config)) {
        return UNSUPPORTED_PROFILE;
    }
    return OK;
}

}  // namespace

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata)
{
    if (layout == nullptr || metadata == nullptr) return INVALID_ARGUMENT;
    const Status status = ValidateConfig(config);
    if (status != OK) return status;

    layout->num_dmrs_symbols = 2;
    layout->num_data_symbols = 12;
    layout->num_data_re = 12 * N_SC_USED;
    layout->data_stride = 19200;
    layout->codeword_symbols = config.num_layers * layout->num_data_re;
    layout->codeword_stride = config.num_layers * layout->data_stride;

    std::memset(metadata, 0, sizeof(*metadata));
    metadata->magic = META_MAGIC;
    metadata->num_rx_antennas = config.num_rx_antennas;
    metadata->num_symbols = N_SYMBOLS;
    metadata->fft_size = N_FFT;
    metadata->used_subcarriers = N_SC_USED;
    metadata->padded_subcarriers = N_SC_PAD;
    metadata->block_dim = BLOCK_DIM;
    metadata->symbols_per_tile = SYMBOLS_PER_TILE;
    return OK;
}

Status ValidateOpArgs(const ReDemapBatchOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(ReDemapBatchOpArgsV1) ||
        args.fft_grid_re == nullptr || args.fft_grid_im == nullptr ||
        args.rx_grid_re == nullptr || args.rx_grid_im == nullptr ||
        args.config == nullptr || args.layout == nullptr || args.stream == nullptr) {
        return INVALID_ARGUMENT;
    }
    if (args.fft_grid_re == args.rx_grid_re ||
        args.fft_grid_re == args.rx_grid_im ||
        args.fft_grid_im == args.rx_grid_re ||
        args.fft_grid_im == args.rx_grid_im ||
        args.rx_grid_re == args.rx_grid_im) {
        return INVALID_ARGUMENT;
    }

    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    const Status status = BuildCurrentProfile(*args.config, &expected, &metadata);
    if (status != OK) return status;
    return SameLayout(expected, *args.layout) ? OK : LAYOUT_MISMATCH;
}

Status BuildGatherIndex(uint32_t *index, size_t index_elems)
{
    if (index == nullptr || index_elems < N_SC_PAD) return RESOURCE_TOO_SMALL;

    // Used subcarriers in fftshift order: [-798..-1, DC, +1..+797].
    // Upstream OFDM stage-4 stores natural bin n at [n%32,n/32].
    for (uint32_t used = 0; used < N_SC_USED; ++used) {
        uint32_t bin;
        if (used < N_SC_USED / 2) {
            bin = N_FFT - N_SC_USED / 2 + used;
        } else {
            bin = used - N_SC_USED / 2;
        }
        const uint32_t element = (bin % 32) * 64 + bin / 32;
        index[used] = element * sizeof(uint16_t);
    }
    std::fill(index + N_SC_USED, index + N_SC_PAD, uint32_t{0});
    return OK;
}

Status Reference(const uint16_t *fft_grid_re,
                 const uint16_t *fft_grid_im,
                 uint32_t num_rx_antennas,
                 const uint32_t *gather_index,
                 uint16_t *rx_grid_re,
                 uint16_t *rx_grid_im)
{
    if (fft_grid_re == nullptr || fft_grid_im == nullptr ||
        gather_index == nullptr || rx_grid_re == nullptr || rx_grid_im == nullptr ||
        num_rx_antennas == 0 || num_rx_antennas > MAX_RX_ANTENNAS ||
        fft_grid_re == rx_grid_re || fft_grid_re == rx_grid_im ||
        fft_grid_im == rx_grid_re || fft_grid_im == rx_grid_im ||
        rx_grid_re == rx_grid_im) {
        return INVALID_ARGUMENT;
    }

    for (uint32_t rx = 0; rx < num_rx_antennas; ++rx) {
        for (uint32_t symbol = 0; symbol < N_SYMBOLS; ++symbol) {
            const size_t input_base = static_cast<size_t>(rx) * GRID_IN_ELEMS +
                                      static_cast<size_t>(symbol) * N_FFT;
            const size_t output_base = static_cast<size_t>(rx) * GRID_OUT_ELEMS +
                                       static_cast<size_t>(symbol) * N_SC_PAD;
            for (uint32_t sc = 0; sc < N_SC_USED; ++sc) {
                if ((gather_index[sc] & 1u) != 0 ||
                    gather_index[sc] >= N_FFT * sizeof(uint16_t)) {
                    return INVALID_ARGUMENT;
                }
                const uint32_t input_index = gather_index[sc] / sizeof(uint16_t);
                rx_grid_re[output_base + sc] = fft_grid_re[input_base + input_index];
                rx_grid_im[output_base + sc] = fft_grid_im[input_base + input_index];
            }
            std::fill(rx_grid_re + output_base + N_SC_USED,
                      rx_grid_re + output_base + N_SC_PAD, uint16_t{0});
            std::fill(rx_grid_im + output_base + N_SC_USED,
                      rx_grid_im + output_base + N_SC_PAD, uint16_t{0});
        }
    }
    return OK;
}

Status Enqueue(const ReDemapBatchOpArgsV1 &args,
               const void *gather_index,
               size_t gather_index_bytes,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes)
{
    const Status status = ValidateOpArgs(args);
    if (status != OK) return status;
    if (gather_index == nullptr || workspace == nullptr || tiling == nullptr ||
        gather_index_bytes < N_SC_PAD * sizeof(uint32_t) ||
        workspace_bytes < WORKSPACE_BYTES || tiling_bytes < TILING_BYTES) {
        return RESOURCE_TOO_SMALL;
    }

#ifdef ASCENDC_CPU_DEBUG
    (void)gather_index;
    (void)workspace;
    (void)tiling;
    return LAUNCH_FAILED;
#else
    const uint32_t launch_status = ACLRT_LAUNCH_KERNEL(re_demap_batch_kernel)(
        BLOCK_DIM, static_cast<aclrtStream>(args.stream),
        const_cast<void *>(args.fft_grid_re),
        const_cast<void *>(args.fft_grid_im),
        const_cast<void *>(gather_index), args.rx_grid_re, args.rx_grid_im,
        args.config->num_rx_antennas, workspace, const_cast<void *>(tiling));
    return launch_status == ACL_ERROR_NONE ? OK : LAUNCH_FAILED;
#endif
}

}  // namespace airan::re_demap_batch
