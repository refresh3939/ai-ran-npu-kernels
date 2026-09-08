#include "qam_mod_256_mimo.h"

#if __has_include("../layer_map/layer_map.h")
#include "../layer_map/layer_map.h"
#elif __has_include("../layer_map_mimo/layer_map.h")
#include "../layer_map_mimo/layer_map.h"
#else
#error "layer_map profile header not found"
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace qmm = airan::qam_mod_256_mimo;
namespace lm = airan::layer_map;

namespace {

qmm::PuschMimoConfig MakeConfig(uint16_t layers)
{
    qmm::PuschMimoConfig config {};
    config.abi_version = qmm::ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = layers;
    config.num_tx_ports = layers == 1 ? 1 : (layers == 2 ? 2 : 4);
    config.num_rx_antennas = 64;
    config.qm = qmm::Q_M;
    config.num_symbols = 14;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = 14;
    config.used_subcarriers = qmm::N_SC_USED;
    config.padded_subcarriers = 1664;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    for (uint16_t layer = 0; layer < layers; ++layer) {
        config.dmrs_ports[layer] = static_cast<uint16_t>(1000 + layer);
    }
    return config;
}

template <typename Destination, typename Source>
Destination AbiCopy(const Source &source)
{
    static_assert(sizeof(Destination) == sizeof(Source),
                  "shared PUSCH config ABI sizes differ");
    static_assert(std::is_trivially_copyable<Destination>::value,
                  "destination config must be trivially copyable");
    Destination destination {};
    std::memcpy(&destination, &source, sizeof(destination));
    destination.struct_size = sizeof(destination);
    return destination;
}

bool SameLayout(const qmm::PuschMimoLayout &qam,
                const lm::PuschMimoLayout &layer)
{
    return qam.num_dmrs_symbols == layer.num_dmrs_symbols &&
           qam.num_data_symbols == layer.num_data_symbols &&
           qam.num_data_re == layer.num_data_re &&
           qam.data_stride == layer.data_stride &&
           qam.codeword_symbols == layer.codeword_symbols &&
           qam.codeword_stride == layer.codeword_stride;
}

int PamLevel(int16_t c0, int16_t c1, int16_t c2, int16_t c3)
{
    const int d0 = 2 * c0 - 1;
    const int d1 = 2 * c1 - 1;
    const int d2 = 2 * c2 - 1;
    return d0 * (8 - d1 * (4 - d2 * (3 - 2 * c3)));
}

uint16_t ExpectedFp16(int level)
{
    constexpr uint16_t by_level[16] = {
        0xbc9a, 0xbbfb, 0xbac0, 0xb986,
        0xb84c, 0xb623, 0xb35e, 0xace9,
        0x2ce9, 0x335e, 0x3623, 0x384c,
        0x3986, 0x3ac0, 0x3bfb, 0x3c9a,
    };
    return by_level[(level + 15) / 2];
}

bool CheckHostReference(const qmm::PuschMimoConfig &config,
                        const qmm::PuschMimoLayout &layout)
{
    const size_t stride = layout.codeword_stride;
    std::vector<int16_t> bits(qmm::Q_M * stride, static_cast<int16_t>(0x5a5a));
    for (uint32_t stream = 0; stream < qmm::Q_M; ++stream) {
        for (uint32_t symbol = 0; symbol < layout.codeword_symbols; ++symbol) {


            const uint32_t pattern = symbol < 256
                ? symbol
                : symbol * 4051u + stream * 7919u + config.num_layers * 1237u;
            bits[static_cast<size_t>(stream) * stride + symbol] =
                static_cast<int16_t>((pattern >> stream) & 1u);
        }
    }
    std::vector<uint16_t> d_re(stride, uint16_t{0x5a5a});
    std::vector<uint16_t> d_im(stride, uint16_t{0xa5a5});
    if (qmm::ReferenceModulate(bits.data(), config, layout,
                               d_re.data(), d_im.data()) != qmm::OK) {
        return false;
    }
    for (uint32_t symbol = 0; symbol < layout.codeword_symbols; ++symbol) {
        const size_t offset = symbol;
        const int i_level = PamLevel(
            bits[0 * stride + offset], bits[1 * stride + offset],
            bits[2 * stride + offset], bits[3 * stride + offset]);
        const int q_level = PamLevel(
            bits[4 * stride + offset], bits[5 * stride + offset],
            bits[6 * stride + offset], bits[7 * stride + offset]);
        if (d_re[symbol] != ExpectedFp16(i_level) ||
            d_im[symbol] != ExpectedFp16(q_level)) {
            return false;
        }
    }
    for (uint32_t symbol = layout.codeword_symbols;
         symbol < layout.codeword_stride; ++symbol) {
        if (d_re[symbol] != 0 || d_im[symbol] != 0) return false;
    }
    return true;
}

bool RunRank(uint16_t layers)
{
    const qmm::PuschMimoConfig qam_config = MakeConfig(layers);
    const lm::PuschMimoConfig layer_config =
        AbiCopy<lm::PuschMimoConfig>(qam_config);
    qmm::PuschMimoLayout qam_layout {};
    lm::PuschMimoLayout layer_layout {};
    qmm::KernelMetadata qam_metadata {};
    lm::KernelMetadata layer_metadata {};
    std::vector<uint32_t> gather_index(lm::MAX_INDEX_ELEMS);

    const bool built =
        qmm::BuildCurrentProfile(qam_config, &qam_layout, &qam_metadata) == qmm::OK &&
        lm::BuildCurrentProfile(layer_config, &layer_layout, &layer_metadata,
                                gather_index.data()) == lm::OK;
    const bool layout_match = built && SameLayout(qam_layout, layer_layout);
    const bool compact_contract = layout_match &&
        qam_layout.num_data_re == qmm::N_DATA_RE &&
        qam_layout.data_stride == qmm::N_DATA_PAD &&
        qam_layout.codeword_symbols == layers * qmm::N_DATA_RE &&
        qam_layout.codeword_stride == layers * qmm::N_DATA_PAD;
    const bool host_reference = compact_contract &&
        CheckHostReference(qam_config, qam_layout);
    const uint32_t expected_order[qmm::Q_M] = {0, 2, 4, 6, 1, 3, 5, 7};
    bool stream_order = built;
    for (uint32_t stream = 0; stream < qmm::Q_M; ++stream) {
        stream_order &= qam_metadata.qam_stream_to_nr_bit[stream] ==
                        expected_order[stream];
    }
    const bool ok = built && layout_match && compact_contract && stream_order &&
                    host_reference;
    std::printf("rank%u qam_mod->layer_map profile=%s compact=%s stream_order=%s "
                "host_reference=%s %s\n",
                layers, layout_match ? "MATCH" : "MISMATCH",
                compact_contract ? "PASS" : "FAIL",
                stream_order ? "PASS" : "FAIL",
                host_reference ? "PASS" : "FAIL", ok ? "PASS" : "FAIL");
    return ok;
}

}

int main()
{
    bool all_ok = true;
    for (uint16_t layers = 1; layers <= qmm::MAX_LAYERS; ++layers) {
        all_ok &= RunRank(layers);
    }
    return all_ok ? 0 : 1;
}
