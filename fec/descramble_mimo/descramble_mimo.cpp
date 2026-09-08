#include "descramble_mimo.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "acl/acl.h"
#include "aclrtlaunch_descramble_mimo_kernel.h"

namespace airan::descramble_mimo {
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
        config.data_scrambling_id > 1023 || !ReservedIsZero(config)) {
        return UNSUPPORTED_PROFILE;
    }
    return OK;
}

class GoldGenerator {
public:
    explicit GoldGenerator(uint32_t c_init)
    {
        std::memset(x1_, 0, sizeof(x1_));
        std::memset(x2_, 0, sizeof(x2_));
        x1_[0] = 1;
        for (uint32_t bit = 0; bit < 31; ++bit) {
            x2_[bit] = static_cast<uint8_t>((c_init >> bit) & 1u);
        }
        for (uint32_t n = 0; n < 1600; ++n) Step();
    }

    uint8_t Next()
    {
        const uint8_t value = static_cast<uint8_t>(x1_[position_] ^ x2_[position_]);
        Step();
        return value;
    }

private:
    void Step()
    {
        const uint32_t p0 = position_;
        const uint32_t p1 = (position_ + 1) % 31;
        const uint32_t p2 = (position_ + 2) % 31;
        const uint32_t p3 = (position_ + 3) % 31;
        const uint8_t next_x1 = static_cast<uint8_t>(x1_[p3] ^ x1_[p0]);
        const uint8_t next_x2 = static_cast<uint8_t>(
            x2_[p3] ^ x2_[p2] ^ x2_[p1] ^ x2_[p0]);
        x1_[p0] = next_x1;
        x2_[p0] = next_x2;
        position_ = (position_ + 1) % 31;
    }

    uint8_t x1_[31] {};
    uint8_t x2_[31] {};
    uint32_t position_ = 0;
};

int16_t SaturatingProduct(int16_t value, int16_t sign)
{
    const int32_t product = static_cast<int32_t>(value) * static_cast<int32_t>(sign);
    return static_cast<int16_t>(std::max<int32_t>(
        std::numeric_limits<int16_t>::min(),
        std::min<int32_t>(std::numeric_limits<int16_t>::max(), product)));
}

}

Status BuildCurrentProfile(const PuschMimoConfig &config,
                           uint32_t num_slots,
                           PuschMimoLayout *layout,
                           KernelMetadata *metadata)
{
    if (layout == nullptr || metadata == nullptr || num_slots == 0 ||
        num_slots > MAX_SLOTS) {
        return INVALID_ARGUMENT;
    }
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
    metadata->num_slots = num_slots;
    metadata->num_layers = config.num_layers;
    metadata->qm = config.qm;
    metadata->num_data_re = layout->num_data_re;
    metadata->data_stride = layout->data_stride;
    metadata->codeword_symbols = layout->codeword_symbols;
    metadata->codeword_stride = layout->codeword_stride;
    metadata->tile_elems = TILE_ELEMS;
    metadata->max_slots = MAX_SLOTS;
    return OK;
}

Status ValidateOpArgs(const DescrambleMimoOpArgsV1 &args)
{
    if (args.abi_version != ABI_VERSION ||
        args.struct_size < sizeof(DescrambleMimoOpArgsV1) ||
        args.llr_qam == nullptr || args.llr_nr == nullptr ||
        args.config == nullptr || args.layout == nullptr || args.stream == nullptr) {
        return INVALID_ARGUMENT;
    }


    if (args.llr_qam == args.llr_nr) return INVALID_ARGUMENT;
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    const Status status = BuildCurrentProfile(*args.config, args.num_slots,
                                               &expected, &metadata);
    if (status != OK) return status;
    return SameLayout(expected, *args.layout) ? OK : LAYOUT_MISMATCH;
}

Status BuildGoldSign(const PuschMimoConfig &config,
                     const PuschMimoLayout &layout,
                     uint32_t num_slots,
                     int16_t *sign,
                     size_t sign_elems)
{
    if (sign == nullptr) return INVALID_ARGUMENT;
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    const Status status = BuildCurrentProfile(config, num_slots, &expected, &metadata);
    if (status != OK) return status;
    if (!SameLayout(expected, layout)) return LAYOUT_MISMATCH;
    const size_t required = BufferElems(num_slots, config.num_layers);
    if (sign_elems < required) return RESOURCE_TOO_SMALL;

    std::fill(sign, sign + required, int16_t{1});
    const uint32_t c_init =
        (static_cast<uint32_t>(config.rnti) << 15) |
        (static_cast<uint32_t>(config.codeword_index & 1u) << 14) |
        static_cast<uint32_t>(config.data_scrambling_id);
    GoldGenerator gold(c_init);
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        const size_t slot_base = static_cast<size_t>(slot) * Q_M * layout.codeword_stride;
        for (uint32_t symbol = 0; symbol < layout.codeword_symbols; ++symbol) {
            for (uint32_t bit = 0; bit < Q_M; ++bit) {
                sign[slot_base + static_cast<size_t>(bit) * layout.codeword_stride + symbol] =
                    gold.Next() == 0 ? int16_t{1} : int16_t{-1};
            }
        }
    }
    return OK;
}

Status ReferenceDescramble(const int16_t *llr_qam,
                           const int16_t *sign,
                           const PuschMimoConfig &config,
                           const PuschMimoLayout &layout,
                           uint32_t num_slots,
                           int16_t *llr_nr)
{
    if (llr_qam == nullptr || sign == nullptr || llr_nr == nullptr) {
        return INVALID_ARGUMENT;
    }
    if (llr_qam == llr_nr || sign == llr_nr) return INVALID_ARGUMENT;
    PuschMimoLayout expected {};
    KernelMetadata metadata {};
    const Status status = BuildCurrentProfile(config, num_slots, &expected, &metadata);
    if (status != OK) return status;
    if (!SameLayout(expected, layout)) return LAYOUT_MISMATCH;

    const size_t total = BufferElems(num_slots, config.num_layers);
    std::fill(llr_nr, llr_nr + total, int16_t{0});
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        const size_t slot_base = static_cast<size_t>(slot) * Q_M * layout.codeword_stride;
        for (uint32_t bit = 0; bit < Q_M; ++bit) {
            const uint32_t qam_stream = NR_BIT_TO_QAM_STREAM[bit];
            const size_t input_base = slot_base +
                static_cast<size_t>(qam_stream) * layout.codeword_stride;
            const size_t output_base = slot_base +
                static_cast<size_t>(bit) * layout.codeword_stride;
            for (uint32_t symbol = 0; symbol < layout.codeword_symbols; ++symbol) {
                llr_nr[output_base + symbol] = SaturatingProduct(
                    llr_qam[input_base + symbol], sign[output_base + symbol]);
            }
        }
    }
    return OK;
}

Status Enqueue(const DescrambleMimoOpArgsV1 &args,
               const void *sign,
               size_t sign_bytes,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes)
{
    const Status status = ValidateOpArgs(args);
    if (status != OK) return status;
    const size_t required_sign =
        BufferElems(args.num_slots, args.config->num_layers) * sizeof(int16_t);
    if (sign == nullptr || workspace == nullptr || tiling == nullptr ||
        sign == args.llr_qam || sign == args.llr_nr ||
        sign_bytes < required_sign || workspace_bytes < WORKSPACE_BYTES ||
        tiling_bytes < TILING_BYTES) {
        return RESOURCE_TOO_SMALL;
    }

    const uint32_t launch_status = ACLRT_LAUNCH_KERNEL(descramble_mimo_kernel)(
        BLOCK_DIM, static_cast<aclrtStream>(args.stream),
        const_cast<void *>(args.llr_qam), const_cast<void *>(sign), args.llr_nr,
        workspace, const_cast<void *>(tiling));
    return launch_status == ACL_ERROR_NONE ? OK : LAUNCH_FAILED;
}

}
