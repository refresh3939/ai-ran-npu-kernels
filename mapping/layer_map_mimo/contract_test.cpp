#include "layer_map.h"

#if __has_include("../layer_demap_mimo/layer_demap.h")
#include "../layer_demap_mimo/layer_demap.h"
#else
#include "../../rx/layer_demap/layer_demap.h"
#endif
#include "../mimo_resource_grid_map/mimo_resource_grid_map.h"
#include "../qam_mod_256_mimo/qam_mod_256_mimo.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace lm = airan::layer_map;
namespace ldm = airan::layer_demap;
namespace rgm = airan::mimo_resource_grid_map;
namespace qmm = airan::qam_mod_256_mimo;

namespace {

template <typename Config>
Config MakeConfig(uint16_t layers)
{
    Config config {};
    config.abi_version = lm::ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = layers;
    config.num_tx_ports = layers <= 2 ? layers : 4;
    config.num_rx_antennas = 64;
    config.qm = lm::Q_M;
    config.num_symbols = 14;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = 14;
    config.used_subcarriers = 1596;
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

template <typename Left, typename Right>
bool SameLayout(const Left &left, const Right &right)
{
    return left.num_dmrs_symbols == right.num_dmrs_symbols &&
           left.num_data_symbols == right.num_data_symbols &&
           left.num_data_re == right.num_data_re &&
           left.data_stride == right.data_stride &&
           left.codeword_symbols == right.codeword_symbols &&
           left.codeword_stride == right.codeword_stride;
}

bool VerifyQamAndGridContracts(
    uint16_t layers,
    const lm::PuschMimoConfig &config,
    const lm::PuschMimoLayout &layout,
    const std::vector<uint32_t> &dataOffset,
    const std::vector<uint32_t> &dmrsOffset)
{
    std::vector<int16_t> bits(qmm::InputElems(layers));
    for (uint32_t q = 0; q < lm::Q_M; ++q) {
        const size_t base = static_cast<size_t>(q) * layout.codeword_stride;
        for (uint32_t symbol = 0; symbol < layout.codeword_symbols; ++symbol) {
            bits[base + symbol] = static_cast<int16_t>((symbol + 3u * q + layers) & 1u);
        }

        std::fill(bits.begin() + base + layout.codeword_symbols,
                  bits.begin() + base + layout.codeword_stride,
                  static_cast<int16_t>(7));
    }

    std::vector<uint16_t> dRe(qmm::OutputElems(layers));
    std::vector<uint16_t> dIm(qmm::OutputElems(layers));
    if (qmm::ReferenceModulate(bits.data(), config, layout,
                               dRe.data(), dIm.data()) != qmm::OK) {
        return false;
    }

    std::vector<uint16_t> layerRe(lm::LayerElems(layers));
    std::vector<uint16_t> layerIm(lm::LayerElems(layers));
    if (lm::ReferenceMap(dRe.data(), dIm.data(), config, layout,
                         layerRe.data(), layerIm.data()) != lm::OK) {
        return false;
    }

    std::vector<uint16_t> dmrsRe(rgm::DmrsElems(layers), uint16_t{0});
    std::vector<uint16_t> dmrsIm(rgm::DmrsElems(layers), uint16_t{0});
    std::vector<uint16_t> gridRe(rgm::GridElems(layers), uint16_t{0xffff});
    std::vector<uint16_t> gridIm(rgm::GridElems(layers), uint16_t{0xffff});
    if (rgm::ReferenceMap(layerRe.data(), layerIm.data(),
                          dmrsRe.data(), dmrsIm.data(), config, layout,
                          dataOffset.data(), dmrsOffset.data(),
                          gridRe.data(), gridIm.data()) != rgm::OK) {
        return false;
    }

    for (uint32_t layer = 0; layer < layers; ++layer) {
        const size_t layerBase = static_cast<size_t>(layer) * layout.data_stride;
        const size_t gridBase = static_cast<size_t>(layer) * rgm::N_GRID;
        for (uint32_t dataRe = 0; dataRe < layout.num_data_re; ++dataRe) {
            const size_t gridIndex =
                gridBase + dataOffset[dataRe] / sizeof(uint16_t);
            if (gridRe[gridIndex] != layerRe[layerBase + dataRe] ||
                gridIm[gridIndex] != layerIm[layerBase + dataRe]) {
                return false;
            }
        }
        if (std::any_of(layerRe.begin() + layerBase + layout.num_data_re,
                        layerRe.begin() + layerBase + layout.data_stride,
                        [](uint16_t value) { return value != 0; }) ||
            std::any_of(layerIm.begin() + layerBase + layout.num_data_re,
                        layerIm.begin() + layerBase + layout.data_stride,
                        [](uint16_t value) { return value != 0; })) {
            return false;
        }
    }
    return true;
}

bool RunRank(uint16_t layers)
{
    const auto config = MakeConfig<lm::PuschMimoConfig>(layers);
    const auto demapConfig = MakeConfig<ldm::PuschMimoConfig>(layers);
    lm::PuschMimoLayout mapLayout {};
    ldm::PuschMimoLayout demapLayout {};
    rgm::PuschMimoLayout gridLayout {};
    qmm::PuschMimoLayout qamLayout {};
    lm::KernelMetadata mapMetadata {};
    ldm::KernelMetadata demapMetadata {};
    rgm::KernelMetadata gridMetadata {};
    qmm::KernelMetadata qamMetadata {};
    std::vector<uint32_t> mapIndex(lm::MAX_INDEX_ELEMS);
    std::vector<uint32_t> demapIndex(ldm::MAX_INDEX_ELEMS);
    std::vector<uint32_t> dataOffset(rgm::N_DATA_PAD);
    std::vector<uint32_t> dmrsOffset(rgm::DmrsOffsetElems());

    const bool tensorSizesOk =
        qmm::OutputElems(layers) == lm::CodewordElems(layers) &&
        lm::LayerElems(layers) == rgm::DataElems(layers) &&
        ldm::INPUT_LAYER_Q_STRIDE == lm::N_DATA_PAD;
    const bool profilesOk =
        lm::BuildCurrentProfile(config, &mapLayout, &mapMetadata,
                                mapIndex.data()) == lm::OK &&
        ldm::BuildCurrentProfile(demapConfig, &demapLayout, &demapMetadata,
                                 demapIndex.data()) == ldm::OK &&
        rgm::BuildCurrentProfile(config, &gridLayout, &gridMetadata,
                                 dataOffset.data(), dmrsOffset.data()) == rgm::OK &&
        qmm::BuildCurrentProfile(config, &qamLayout, &qamMetadata) == qmm::OK;
    bool qamStreamOrderOk = true;
    for (uint32_t stream = 0; stream < lm::Q_M; ++stream) {
        qamStreamOrderOk &= qamMetadata.qam_stream_to_nr_bit[stream] ==
                            qmm::QAM_STREAM_TO_NR_BIT[stream];
    }
    if (!tensorSizesOk || !profilesOk ||
        !SameLayout(mapLayout, demapLayout) ||
        !SameLayout(mapLayout, gridLayout) ||
        !SameLayout(mapLayout, qamLayout) || !qamStreamOrderOk) {
        return false;
    }

    if (!VerifyQamAndGridContracts(layers, config, mapLayout,
                                   dataOffset, dmrsOffset)) {
        return false;
    }

    const size_t cwStride = mapLayout.codeword_stride;
    const size_t layerPlaneElems = lm::LayerElems(layers);
    std::vector<int16_t> cw(static_cast<size_t>(lm::Q_M) * cwStride);
    for (uint32_t q = 0; q < lm::Q_M; ++q) {
        const size_t base = static_cast<size_t>(q) * cwStride;
        for (uint32_t index = 0; index < mapLayout.codeword_symbols; ++index) {
            cw[base + index] = static_cast<int16_t>(
                (index * 4051u + q * 7919u + layers * 1237u) & 0xffffu);
        }
        std::fill(cw.begin() + base + mapLayout.codeword_symbols,
                  cw.begin() + base + cwStride, static_cast<int16_t>(0x5a5a));
    }

    std::vector<int16_t> layerLlr(ldm::LayerElems(layers),
                                  static_cast<int16_t>(0x6b6b));
    std::vector<uint16_t> mapped0(layerPlaneElems);
    std::vector<uint16_t> mapped1(layerPlaneElems);
    for (uint32_t q = 0; q < lm::Q_M; q += 2) {
        const auto *source0 = reinterpret_cast<const uint16_t *>(
            cw.data() + static_cast<size_t>(q) * cwStride);
        const auto *source1 = reinterpret_cast<const uint16_t *>(
            cw.data() + static_cast<size_t>(q + 1) * cwStride);
        if (lm::ReferenceMap(source0, source1, config, mapLayout,
                             mapped0.data(), mapped1.data()) != lm::OK) {
            return false;
        }
        for (uint32_t layer = 0; layer < layers; ++layer) {
            const auto *compact0 = reinterpret_cast<const int16_t *>(mapped0.data()) +
                                   static_cast<size_t>(layer) * lm::N_DATA_PAD;
            const auto *compact1 = reinterpret_cast<const int16_t *>(mapped1.data()) +
                                   static_cast<size_t>(layer) * lm::N_DATA_PAD;
            const size_t physical0 =
                (static_cast<size_t>(layer) * lm::Q_M + q) *
                ldm::INPUT_LAYER_Q_STRIDE;
            const size_t physical1 = physical0 + ldm::INPUT_LAYER_Q_STRIDE;



            for (uint32_t symbol = 0; symbol < ldm::N_DATA_SYMBOLS; ++symbol) {
                const size_t compactRow = static_cast<size_t>(symbol) * ldm::N_SC_USED;
                const size_t physicalRow = static_cast<size_t>(symbol) * ldm::N_SC_LLR_PAD;
                std::copy_n(compact0 + compactRow, ldm::N_SC_USED,
                            layerLlr.data() + physical0 + physicalRow);
                std::copy_n(compact1 + compactRow, ldm::N_SC_USED,
                            layerLlr.data() + physical1 + physicalRow);
            }
        }
    }

    std::vector<int16_t> recovered(static_cast<size_t>(lm::Q_M) * cwStride,
                                   static_cast<int16_t>(0x7c7c));
    if (ldm::ReferenceDemap(layerLlr.data(), demapConfig, demapLayout,
                            recovered.data()) != ldm::OK) {
        return false;
    }
    for (uint32_t q = 0; q < lm::Q_M; ++q) {
        const size_t base = static_cast<size_t>(q) * cwStride;
        if (!std::equal(cw.begin() + base,
                        cw.begin() + base + mapLayout.codeword_symbols,
                        recovered.begin() + base)) {
            return false;
        }
        if (std::any_of(recovered.begin() + base + mapLayout.codeword_symbols,
                        recovered.begin() + base + cwStride,
                        [](int16_t value) { return value != 0; })) {
            return false;
        }
    }
    return true;
}

}

int main()
{
    bool allOk = true;
    for (uint16_t layers = 1; layers <= lm::MAX_LAYERS; ++layers) {
        const bool ok = RunRank(layers);
        std::printf("rank%u qam_mod->layer_map->grid + physical_llr->demap %s\n",
                    layers, ok ? "PASS" : "FAIL");
        allOk &= ok;
    }
    return allOk ? 0 : 1;
}
