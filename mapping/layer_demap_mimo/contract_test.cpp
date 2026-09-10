#include "layer_demap.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace ldm = airan::layer_demap;

namespace {

ldm::PuschMimoConfig MakeConfig(uint16_t layers)
{
    ldm::PuschMimoConfig config {};
    config.abi_version = ldm::ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = layers;
    config.num_tx_ports = layers <= 2 ? layers : 4;
    config.num_rx_antennas = 64;
    config.qm = ldm::Q_M;
    config.num_symbols = 14;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = 14;
    config.used_subcarriers = ldm::N_SC_USED;
    config.padded_subcarriers = 1664;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    for (uint16_t layer = 0; layer < std::min<uint16_t>(layers, 4); ++layer) {
        config.dmrs_ports[layer] = static_cast<uint16_t>(1000 + layer);
    }
    return config;
}

int16_t Value(uint32_t layer, uint32_t q, uint32_t dataSymbol, uint32_t sc)
{
    return static_cast<int16_t>(
        (layer * 7919u + q * 4051u + dataSymbol * 1237u + sc * 251u) & 0xffffu);
}

bool RunRank(uint16_t layers)
{
    const auto config = MakeConfig(layers);
    ldm::PuschMimoLayout layout {};
    ldm::KernelMetadata metadata {};
    std::vector<uint32_t> gather(ldm::MAX_INDEX_ELEMS, 0xdeadbeefu);
    if (ldm::BuildCurrentProfile(config, &layout, &metadata, gather.data()) != ldm::OK) {
        return false;
    }

    const bool shapeOk =
        layout.num_data_symbols == ldm::N_DATA_SYMBOLS &&
        layout.num_data_re == ldm::N_DATA_RE &&
        layout.data_stride == ldm::N_DATA_PAD &&
        layout.codeword_symbols == layers * ldm::N_DATA_RE &&
        layout.codeword_stride == layers * ldm::N_DATA_PAD &&
        metadata.input_symbol_stride == ldm::N_SC_LLR_PAD &&
        metadata.gather_index_elems == layers * ldm::GATHER_CHUNK;
    if (!shapeOk) return false;

    for (uint32_t re = 0; re < ldm::GATHER_CHUNK; ++re) {
        for (uint32_t layer = 0; layer < layers; ++layer) {
            const uint32_t expected = sizeof(int16_t) *
                                      (layer * ldm::GROUP_DATA_RE + re);
            if (gather[re * layers + layer] != expected) return false;
        }
    }
    if (std::any_of(gather.begin() + layers * ldm::GATHER_CHUNK, gather.end(),
                    [](uint32_t value) { return value != 0; })) {
        return false;
    }

    std::vector<int16_t> physical(ldm::LayerElems(layers));
    for (uint32_t layer = 0; layer < layers; ++layer) {
        for (uint32_t q = 0; q < ldm::Q_M; ++q) {
            const size_t plane = (static_cast<size_t>(layer) * ldm::Q_M + q) *
                                 ldm::INPUT_LAYER_Q_STRIDE;
            for (uint32_t symbol = 0; symbol < ldm::N_DATA_SYMBOLS; ++symbol) {
                const size_t row = plane + static_cast<size_t>(symbol) * ldm::N_SC_LLR_PAD;
                for (uint32_t sc = 0; sc < ldm::N_SC_USED; ++sc) {
                    physical[row + sc] = Value(layer, q, symbol, sc);
                }
                std::fill_n(physical.begin() + row + ldm::N_SC_USED,
                            ldm::N_SC_LLR_PAD - ldm::N_SC_USED,
                            static_cast<int16_t>(0x5a5a));
            }
        }
    }

    std::vector<int16_t> output(ldm::CodewordElems(layers),
                                static_cast<int16_t>(0x6b6b));
    if (ldm::ReferenceDemap(physical.data(), config, layout, output.data()) != ldm::OK) {
        return false;
    }
    for (uint32_t q = 0; q < ldm::Q_M; ++q) {
        const size_t plane = static_cast<size_t>(q) * layout.codeword_stride;
        for (uint32_t symbol = 0; symbol < ldm::N_DATA_SYMBOLS; ++symbol) {
            for (uint32_t sc = 0; sc < ldm::N_SC_USED; ++sc) {
                const size_t re = static_cast<size_t>(symbol) * ldm::N_SC_USED + sc;
                for (uint32_t layer = 0; layer < layers; ++layer) {
                    if (output[plane + re * layers + layer] !=
                        Value(layer, q, symbol, sc)) {
                        return false;
                    }
                }
            }
        }
        if (std::any_of(output.begin() + plane + layout.codeword_symbols,
                        output.begin() + plane + layout.codeword_stride,
                        [](int16_t value) { return value != 0; })) {
            return false;
        }
    }

    const std::vector<int16_t> first = output;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        for (uint32_t q = 0; q < ldm::Q_M; ++q) {
            const size_t plane = (static_cast<size_t>(layer) * ldm::Q_M + q) *
                                 ldm::INPUT_LAYER_Q_STRIDE;
            for (uint32_t symbol = 0; symbol < ldm::N_DATA_SYMBOLS; ++symbol) {
                const size_t pad = plane + static_cast<size_t>(symbol) *
                                             ldm::N_SC_LLR_PAD + ldm::N_SC_USED;
                for (uint32_t index = 0;
                     index < ldm::N_SC_LLR_PAD - ldm::N_SC_USED; ++index) {
                    physical[pad + index] = static_cast<int16_t>(
                        0x8000u + layer * 97u + q * 17u + symbol * 5u + index);
                }
            }
        }
    }
    std::fill(output.begin(), output.end(), static_cast<int16_t>(0x7c7c));
    if (ldm::ReferenceDemap(physical.data(), config, layout, output.data()) != ldm::OK ||
        output != first) {
        return false;
    }

    ldm::LayerDemapOpArgsV1 args {};
    args.abi_version = ldm::ABI_VERSION;
    args.struct_size = sizeof(args);
    args.layer_llr = physical.data();
    args.cw_llr = output.data();
    args.config = &config;
    args.layout = &layout;
    args.stream = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
    if (ldm::ValidateOpArgs(args) != ldm::OK) return false;
    auto badLayout = layout;
    ++badLayout.codeword_stride;
    args.layout = &badLayout;
    return ldm::ValidateOpArgs(args) == ldm::LAYOUT_MISMATCH;
}

}  // namespace

int main()
{
    bool allOk = true;
    for (uint16_t layers = 1; layers <= ldm::MAX_LAYERS; ++layers) {
        const bool ok = RunRank(layers);
        std::printf("rank%u host contract: physical [L,8,12,1600] -> compact "
                    "[8,L*19200] %s\n", layers, ok ? "PASS" : "FAIL");
        allOk &= ok;
    }

    auto unsupported = MakeConfig(ldm::MAX_LAYERS + 1);
    ldm::PuschMimoLayout layout {};
    ldm::KernelMetadata metadata {};
    std::vector<uint32_t> gather(ldm::MAX_INDEX_ELEMS);
    const bool rejectsRank5 =
        ldm::BuildCurrentProfile(unsupported, &layout, &metadata, gather.data()) ==
        ldm::UNSUPPORTED_PROFILE;
    std::printf("rank5 rejection %s\n", rejectsRank5 ? "PASS" : "FAIL");
    return allOk && rejectsRank5 ? 0 : 1;
}
