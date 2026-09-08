#include "rate_match_mimo.h"

#include <algorithm>
#include <cstring>

#ifndef RATE_MATCH_MIMO_HOST_ONLY
#include "acl/acl.h"
#include "aclrtlaunch_rate_match_mimo_kernel.h"
#endif

namespace airan::rate_match_mimo {
namespace {

bool ReservedIsZero(const PuschMimoConfig &config)
{
    for (uint32_t value : config.reserved) {
        if (value != 0) return false;
    }
    return true;
}

bool ReservedIsZero(const RateMatchFecConfigV1 &fec)
{
    for (uint32_t value : fec.reserved) {
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

Status ValidateConfig(const PuschMimoConfig &config,
                      const RateMatchFecConfigV1 &fec)
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
        config.padded_subcarriers != 1664 ||
        config.dmrs_symbol_mask != ((1u << 2) | (1u << 11)) ||
        config.dmrs_type != 1 || config.dmrs_length != 1 ||
        config.num_cdm_groups_without_data != 2 || config.n_scid > 1 ||
        config.codeword_index != 0 || config.transform_precoding != 0 ||
        !ReservedIsZero(config)) {
        return UNSUPPORTED_PROFILE;
    }
    if (fec.abi_version != ABI_VERSION ||
        fec.struct_size < sizeof(RateMatchFecConfigV1) || fec.flags != 0 ||
        fec.num_code_blocks != C_NUM || fec.encoded_stride != N_CB_BUF ||
        fec.rv_index != 0 || fec.base_graph != 1 || fec.lifting_size != LDPC_Z ||
        fec.filler_bits != 0 || !ReservedIsZero(fec)) {
        return UNSUPPORTED_PROFILE;
    }
    return OK;
}

}

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           const RateMatchFecConfigV1 &fec,
                           uint32_t num_slots,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata)
{
    if (layout == nullptr || metadata == nullptr || num_slots == 0 ||
        num_slots > MAX_SLOTS) {
        return INVALID_ARGUMENT;
    }
    const Status status = ValidateConfig(config, fec);
    if (status != OK) return status;

    layout->num_dmrs_symbols = 2;
    layout->num_data_symbols = N_DATA_SYMBOLS;
    layout->num_data_re = N_DATA_RE;
    layout->data_stride = N_DATA_PAD;
    layout->codeword_symbols = config.num_layers * N_DATA_RE;
    layout->codeword_stride = config.num_layers * N_DATA_PAD;

    std::memset(metadata, 0, sizeof(*metadata));
    metadata->magic = META_MAGIC;
    metadata->num_slots = num_slots;
    metadata->num_layers = config.num_layers;
    metadata->qm = config.qm;
    metadata->num_data_re = layout->num_data_re;
    metadata->data_stride = layout->data_stride;
    metadata->codeword_symbols = layout->codeword_symbols;
    metadata->codeword_stride = layout->codeword_stride;
    metadata->num_code_blocks = fec.num_code_blocks;
    metadata->encoded_stride = fec.encoded_stride;
    metadata->copy_elems = COPY_ELEMS;
    metadata->max_slots = MAX_SLOTS;
    return OK;
}

Status BuildRateMatchDescriptors(const PuschMimoConfig &config,
                                 const RateMatchFecConfigV1 &fec,
                                 uint32_t num_slots,
                                 RateMatchDescriptor *descriptors,
                                 size_t descriptor_count)
{
    if (descriptors == nullptr || descriptor_count < fec.num_code_blocks) {
        return RESOURCE_TOO_SMALL;
    }
    PuschMimoLayout layout {};
    KernelMetadata metadata {};
    const Status status = BuildCurrentProfile(config, fec, num_slots,
                                               &layout, &metadata);
    if (status != OK) return status;



    const uint64_t g_prime = static_cast<uint64_t>(num_slots) * N_DATA_RE;
    const uint32_t floor_symbols = static_cast<uint32_t>(g_prime / C_NUM);
    const uint32_t gamma = static_cast<uint32_t>(g_prime % C_NUM);
    const uint32_t first_high = C_NUM - gamma;
    for (uint32_t cb = 0; cb < C_NUM; ++cb) {
        const uint32_t symbols = floor_symbols + (cb >= first_high ? 1u : 0u);
        descriptors[cb].e = config.num_layers * Q_M * symbols;
        descriptors[cb].k0 = 0;
        descriptors[cb].ncb = N_CB_BUF;
        descriptors[cb].cw_bit_offset = cb * N_CB_BUF;
    }
    return OK;
}

Status ValidateOpArgs(const RateMatchMimoOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(RateMatchMimoOpArgsV1) ||
        args.code_blocks == nullptr || args.bits_nr == nullptr ||
        args.config == nullptr || args.layout == nullptr || args.fec == nullptr ||
        args.stream == nullptr || args.code_blocks == args.bits_nr) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    const Status status = BuildCurrentProfile(*args.config, *args.fec,
                                               args.num_slots, &expected,
                                               &metadata);
    if (status != OK) return status;
    return SameLayout(expected, *args.layout) ? OK : LAYOUT_MISMATCH;
}

Status ReferenceRateMatch(const int8_t *code_blocks,
                          const PuschMimoConfig &config,
                          const PuschMimoLayout &layout,
                          const RateMatchFecConfigV1 &fec,
                          uint32_t num_slots,
                          const RateMatchDescriptor *descriptors,
                          int16_t *bits_nr)
{
    if (code_blocks == nullptr || descriptors == nullptr || bits_nr == nullptr) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    const Status status = BuildCurrentProfile(config, fec, num_slots,
                                               &expected, &metadata);
    if (status != OK) return status;
    if (!SameLayout(expected, layout)) return LAYOUT_MISMATCH;

    const size_t total = OutputElems(num_slots, config.num_layers);
    std::fill(bits_nr, bits_nr + total, int16_t{0});
    uint64_t plane_position = 0;
    uint64_t sum_e = 0;
    for (uint32_t cb = 0; cb < fec.num_code_blocks; ++cb) {
        const RateMatchDescriptor &desc = descriptors[cb];
        if (desc.e == 0 || desc.e % Q_M != 0 || desc.ncb == 0 ||
            desc.ncb > fec.encoded_stride ||
            desc.cw_bit_offset != cb * fec.encoded_stride) {
            return INVALID_ARGUMENT;
        }
        const uint32_t eq = desc.e / Q_M;
        if (plane_position + eq >
            static_cast<uint64_t>(num_slots) * layout.codeword_symbols) {
            return INVALID_ARGUMENT;
        }
        for (uint32_t bit = 0; bit < Q_M; ++bit) {
            for (uint32_t i = 0; i < eq; ++i) {
                const uint64_t logical = plane_position + i;
                const uint32_t slot = static_cast<uint32_t>(
                    logical / layout.codeword_symbols);
                const uint32_t symbol = static_cast<uint32_t>(
                    logical % layout.codeword_symbols);
                const uint32_t selected =
                    (desc.k0 + bit * eq + i) % desc.ncb;
                const int8_t value = code_blocks[desc.cw_bit_offset + selected];
                if (value != 0 && value != 1) return INVALID_ARGUMENT;
                const size_t output = static_cast<size_t>(slot) * Q_M *
                                          layout.codeword_stride +
                                      static_cast<size_t>(bit) *
                                          layout.codeword_stride + symbol;
                bits_nr[output] = static_cast<int16_t>(value);
            }
        }
        plane_position += eq;
        sum_e += desc.e;
    }
    const uint64_t expected_g = static_cast<uint64_t>(num_slots) * Q_M *
                                layout.codeword_symbols;
    if (sum_e != expected_g ||
        plane_position != static_cast<uint64_t>(num_slots) *
                              layout.codeword_symbols) {
        return INVALID_ARGUMENT;
    }
    return OK;
}

Status Enqueue(const RateMatchMimoOpArgsV1 &args,
               const void *descriptors,
               size_t descriptor_bytes,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes)
{
    const Status status = ValidateOpArgs(args);
    if (status != OK) return status;
    const size_t required_desc = static_cast<size_t>(args.fec->num_code_blocks) *
                                 sizeof(RateMatchDescriptor);
    if (descriptors == nullptr || workspace == nullptr || tiling == nullptr ||
        descriptors == args.code_blocks || descriptors == args.bits_nr ||
        descriptor_bytes < required_desc || workspace_bytes < WORKSPACE_BYTES ||
        tiling_bytes < TILING_BYTES) {
        return RESOURCE_TOO_SMALL;
    }
#ifdef RATE_MATCH_MIMO_HOST_ONLY
    (void)args;
    return LAUNCH_FAILED;
#else
    const uint32_t launch_status = ACLRT_LAUNCH_KERNEL(rate_match_mimo_kernel)(
        BLOCK_DIM, static_cast<aclrtStream>(args.stream),
        const_cast<void *>(args.code_blocks), const_cast<void *>(descriptors),
        args.bits_nr, workspace, const_cast<void *>(tiling));
    return launch_status == ACL_ERROR_NONE ? OK : LAUNCH_FAILED;
#endif
}

}
