#include "qam_mod_256_mimo.h"

#include <algorithm>
#include <cstring>

#ifndef QAM_MOD_256_MIMO_NO_ACL
#include "acl/acl.h"
#include "aclrtlaunch_qam_mod_256_mimo_kernel.h"
#endif

namespace airan::qam_mod_256_mimo {
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
        config.num_allocated_symbols != 14 || config.used_subcarriers != N_SC_USED ||
        config.padded_subcarriers != 1664 || config.dmrs_type != 1 ||
        config.dmrs_length != 1 || config.num_cdm_groups_without_data != 2 ||
        config.n_scid > 1 || config.codeword_index != 0 ||
        config.transform_precoding != 0 || !ReservedIsZero(config)) {
        return UNSUPPORTED_PROFILE;
    }
    if (Popcount16(config.dmrs_symbol_mask) != 2 ||
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



constexpr uint16_t PAM_FP16_BY_NIBBLE[16] = {
    0xbc9a, 0x3c9a, 0xace9, 0x2ce9,
    0xb986, 0x3986, 0xb84c, 0x384c,
    0xbbfb, 0x3bfb, 0xb35e, 0x335e,
    0xbac0, 0x3ac0, 0xb623, 0x3623,
};

}

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata)
{
    if (layout == nullptr || metadata == nullptr) return INVALID_ARGUMENT;
    const Status status = ValidateConfig(config);
    if (status != OK) return status;

    layout->num_dmrs_symbols = 2;
    layout->num_data_symbols = N_DATA_SYMBOLS;
    layout->num_data_re = N_DATA_RE;
    layout->data_stride = N_DATA_PAD;
    layout->codeword_symbols = config.num_layers * N_DATA_RE;
    layout->codeword_stride = config.num_layers * N_DATA_PAD;

    std::memset(metadata, 0, sizeof(*metadata));
    metadata->magic = META_MAGIC;
    metadata->num_layers = config.num_layers;
    metadata->qm = Q_M;
    metadata->num_data_re = N_DATA_RE;
    metadata->data_stride = N_DATA_PAD;
    metadata->codeword_symbols = layout->codeword_symbols;
    metadata->codeword_stride = layout->codeword_stride;
    metadata->tile_elems = TILE_ELEMS;
    metadata->compute_cores = COMPUTE_CORES;
    for (uint32_t stream = 0; stream < Q_M; ++stream) {
        metadata->qam_stream_to_nr_bit[stream] = QAM_STREAM_TO_NR_BIT[stream];
    }
    return OK;
}

Status ValidateOpArgs(const QamMod256MimoOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(QamMod256MimoOpArgsV1) ||
        args.bits_qam == nullptr || args.d_re == nullptr || args.d_im == nullptr ||
        args.config == nullptr || args.layout == nullptr || args.stream == nullptr ||
        args.bits_qam == args.d_re || args.bits_qam == args.d_im ||
        args.d_re == args.d_im) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    const Status status = BuildCurrentProfile(*args.config, &expected, &metadata);
    if (status != OK) return status;
    return SameLayout(expected, *args.layout) ? OK : LAYOUT_MISMATCH;
}

Status ReferenceModulate(const int16_t *bits_qam,
                         const PuschMimoConfig &config,
                         const PuschMimoLayout &layout,
                         uint16_t *d_re,
                         uint16_t *d_im)
{
    if (bits_qam == nullptr || d_re == nullptr || d_im == nullptr ||
        reinterpret_cast<const void *>(bits_qam) == d_re ||
        reinterpret_cast<const void *>(bits_qam) == d_im || d_re == d_im) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    const Status status = BuildCurrentProfile(config, &expected, &metadata);
    if (status != OK) return status;
    if (!SameLayout(expected, layout)) return LAYOUT_MISMATCH;

    std::fill(d_re, d_re + layout.codeword_stride, uint16_t{0});
    std::fill(d_im, d_im + layout.codeword_stride, uint16_t{0});
    for (uint32_t symbol = 0; symbol < layout.codeword_symbols; ++symbol) {
        uint32_t i_nibble = 0;
        uint32_t q_nibble = 0;
        for (uint32_t bit = 0; bit < 4; ++bit) {
            const int16_t i_value =
                bits_qam[static_cast<size_t>(bit) * layout.codeword_stride + symbol];
            const int16_t q_value = bits_qam[
                static_cast<size_t>(bit + 4) * layout.codeword_stride + symbol];
            if ((i_value != 0 && i_value != 1) ||
                (q_value != 0 && q_value != 1)) {
                return INVALID_ARGUMENT;
            }
            i_nibble |= static_cast<uint32_t>(i_value) << bit;
            q_nibble |= static_cast<uint32_t>(q_value) << bit;
        }
        d_re[symbol] = PAM_FP16_BY_NIBBLE[i_nibble];
        d_im[symbol] = PAM_FP16_BY_NIBBLE[q_nibble];
    }
    return OK;
}

#ifndef QAM_MOD_256_MIMO_NO_ACL
Status Enqueue(const QamMod256MimoOpArgsV1 &args,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes)
{
    const Status status = ValidateOpArgs(args);
    if (status != OK) return status;
    if (workspace == nullptr || tiling == nullptr ||
        workspace_bytes < WORKSPACE_BYTES || tiling_bytes < TILING_BYTES) {
        return RESOURCE_TOO_SMALL;
    }
    const uint32_t launch_status = ACLRT_LAUNCH_KERNEL(qam_mod_256_mimo_kernel)(
        BLOCK_DIM, static_cast<aclrtStream>(args.stream),
        const_cast<void *>(args.bits_qam), args.d_re, args.d_im,
        workspace, const_cast<void *>(tiling));
    return launch_status == ACL_ERROR_NONE ? OK : LAUNCH_FAILED;
}
#endif

}
