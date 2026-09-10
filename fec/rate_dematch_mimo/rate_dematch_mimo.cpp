#include "rate_dematch_mimo.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "acl/acl.h"
#include "aclrtlaunch_rate_dematch_mimo_kernel.h"
#include "../ldpc_decode/ldpc_decode.h"

namespace airan::rate_dematch_mimo {
static_assert(C_NUM == ::airan::LDPC_C_NUM,
              "rate_dematch_mimo C must match ldpc_decode batch");
static_assert(LDPC_N == ::airan::LAM_ELEMS_PER_CB,
              "rate_dematch_mimo row width must match ldpc_decode lam_in");
static_assert(LLR_CLIP == ::airan::LLR_CLIP_FX,
              "rate_dematch_mimo Q8.8 clip must match ldpc_decode");
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
    return OK;
}

int16_t ScaleClip(int16_t value)
{
    const int32_t scaled = static_cast<int32_t>(value) * LLR_SCALE;
    return static_cast<int16_t>(std::max<int32_t>(
        -LLR_CLIP, std::min<int32_t>(LLR_CLIP, scaled)));
}

}  // namespace

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           uint32_t num_slots,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata,
                           RateMatchDescriptor *descriptors,
                           size_t descriptor_count)
{
    if (layout == nullptr || metadata == nullptr || descriptors == nullptr ||
        descriptor_count < C_NUM || num_slots == 0 || num_slots > MAX_SLOTS) {
        return INVALID_ARGUMENT;
    }
    const Status config_status = ValidateConfig(config);
    if (config_status != OK) return config_status;

    layout->num_dmrs_symbols = 2;
    layout->num_data_symbols = N_DATA_SYMBOLS;
    layout->num_data_re = N_DATA_RE;
    layout->data_stride = N_DATA_PAD;
    layout->codeword_symbols = config.num_layers * N_DATA_RE;
    layout->codeword_stride = config.num_layers * N_DATA_PAD;

    // TS 38.212 5.4.2.1: E_r is quantized by N_L*Q_m. Since this profile's
    // G is exactly num_slots*N_L*Q_m*N_DATA_RE, use symbol units to avoid
    // division rounding ambiguity.
    const uint32_t quantum = config.num_layers * Q_M;
    const uint32_t total_units = num_slots * N_DATA_RE;
    const uint32_t low_units = total_units / C_NUM;
    const uint32_t num_high = total_units % C_NUM;
    const uint32_t first_high = C_NUM - num_high;
    uint32_t cw_symbol_offset = 0;
    uint64_t e_sum = 0;
    for (uint32_t cb = 0; cb < C_NUM; ++cb) {
        const uint32_t units = low_units + (cb >= first_high ? 1u : 0u);
        const uint32_t e = quantum * units;
        descriptors[cb] = {e, 0u, N_CB_BUF, cw_symbol_offset};
        cw_symbol_offset += e / Q_M;
        e_sum += e;
    }
    const uint32_t total_coded_bits =
        num_slots * Q_M * layout->codeword_symbols;
    if (e_sum != total_coded_bits ||
        cw_symbol_offset != num_slots * layout->codeword_symbols) {
        return UNSUPPORTED_PROFILE;
    }

    std::memset(metadata, 0, sizeof(*metadata));
    metadata->magic = META_MAGIC;
    metadata->num_slots = num_slots;
    metadata->num_layers = config.num_layers;
    metadata->qm = Q_M;
    metadata->num_data_re = N_DATA_RE;
    metadata->codeword_symbols = layout->codeword_symbols;
    metadata->codeword_stride = layout->codeword_stride;
    metadata->total_coded_bits = total_coded_bits;
    metadata->code_blocks = C_NUM;
    metadata->ldpc_n = LDPC_N;
    metadata->n_2z = N_2Z;
    metadata->ncb = N_CB_BUF;
    metadata->llr_scale = static_cast<uint32_t>(LLR_SCALE);
    metadata->llr_clip = static_cast<uint32_t>(LLR_CLIP);
    metadata->tile_elems = TILE_ELEMS;
    metadata->max_slots = MAX_SLOTS;
    metadata->descriptor_words = DESC_TABLE_WORDS;
    return OK;
}

Status ValidateOpArgs(const RateDematchMimoOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(RateDematchMimoOpArgsV1) ||
        args.cw_llr == nullptr || args.ldpc_llr == nullptr ||
        args.config == nullptr || args.layout == nullptr || args.stream == nullptr ||
        args.cw_llr == args.ldpc_llr) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    RateMatchDescriptor descriptors[C_NUM] {};
    const Status status = BuildCurrentProfile(
        *args.config, args.num_slots, &expected, &metadata, descriptors, C_NUM);
    if (status != OK) return status;
    return SameLayout(expected, *args.layout) ? OK : LAYOUT_MISMATCH;
}

Status ReferenceRateDematch(const int16_t *cw_llr,
                            const PuschMimoConfig &config,
                            const PuschMimoLayout &layout,
                            uint32_t num_slots,
                            const RateMatchDescriptor *descriptors,
                            size_t descriptor_count,
                            int16_t *ldpc_llr)
{
    if (cw_llr == nullptr || descriptors == nullptr || ldpc_llr == nullptr ||
        cw_llr == ldpc_llr || descriptor_count < C_NUM) {
        return INVALID_ARGUMENT;
    }
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    RateMatchDescriptor expected_desc[C_NUM] {};
    const Status status = BuildCurrentProfile(
        config, num_slots, &expected, &metadata, expected_desc, C_NUM);
    if (status != OK) return status;
    if (!SameLayout(expected, layout) ||
        std::memcmp(expected_desc, descriptors, sizeof(expected_desc)) != 0) {
        return LAYOUT_MISMATCH;
    }

    std::fill(ldpc_llr, ldpc_llr + OutputElems(), int16_t{0});
    for (uint32_t cb = 0; cb < C_NUM; ++cb) {
        const RateMatchDescriptor &desc = descriptors[cb];
        const uint32_t eq = desc.e / Q_M;
        int32_t accumulator[N_CB_BUF] {};
        for (uint32_t bit = 0; bit < Q_M; ++bit) {
            for (uint32_t j = 0; j < eq; ++j) {
                const uint32_t logical = desc.cw_symbol_offset + j;
                const uint32_t slot = logical / layout.codeword_symbols;
                const uint32_t symbol = logical % layout.codeword_symbols;
                const size_t input_index =
                    static_cast<size_t>(slot) * Q_M * layout.codeword_stride +
                    static_cast<size_t>(bit) * layout.codeword_stride + symbol;
                const uint32_t circular = (desc.k0 + bit * eq + j) % desc.ncb;
                accumulator[circular] += ScaleClip(cw_llr[input_index]);
            }
        }
        const size_t row = static_cast<size_t>(cb) * LDPC_N;
        for (uint32_t k = 0; k < N_CB_BUF; ++k) {
            ldpc_llr[row + N_2Z + k] = static_cast<int16_t>(
                std::max<int32_t>(-LLR_CLIP,
                                  std::min<int32_t>(LLR_CLIP, accumulator[k])));
        }
    }
    return OK;
}

Status Enqueue(const RateDematchMimoOpArgsV1 &args,
               const void *descriptors,
               size_t descriptor_bytes,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes)
{
    const Status status = ValidateOpArgs(args);
    if (status != OK) return status;
    if (descriptors == nullptr || workspace == nullptr || tiling == nullptr ||
        descriptors == args.cw_llr || descriptors == args.ldpc_llr ||
        descriptor_bytes < DESCRIPTOR_BYTES ||
        workspace_bytes < WORKSPACE_BYTES || tiling_bytes < TILING_BYTES) {
        return RESOURCE_TOO_SMALL;
    }
    const uint32_t launch_status = ACLRT_LAUNCH_KERNEL(rate_dematch_mimo_kernel)(
        BLOCK_DIM, static_cast<aclrtStream>(args.stream),
        const_cast<void *>(args.cw_llr), const_cast<void *>(descriptors),
        args.ldpc_llr, workspace, const_cast<void *>(tiling));
    return launch_status == ACL_ERROR_NONE ? OK : LAUNCH_FAILED;
}

}  // namespace airan::rate_dematch_mimo
