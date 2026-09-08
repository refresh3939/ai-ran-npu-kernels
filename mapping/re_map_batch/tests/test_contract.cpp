#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "re_map_batch.h"

namespace rmb = airan::re_map_batch;

namespace {

airan::PuschMimoConfig MakeConfig(uint16_t layers, uint16_t ports, bool codebook)
{
    airan::PuschMimoConfig config {};
    config.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = layers;
    config.num_tx_ports = ports;
    config.num_rx_antennas = 64;
    config.qm = 8;
    config.num_symbols = rmb::N_SYMBOLS;
    config.fft_size = rmb::N_FFT;
    config.num_rb = 133;
    config.num_allocated_symbols = rmb::N_SYMBOLS;
    config.used_subcarriers = rmb::N_SC_USED;
    config.padded_subcarriers = rmb::N_SC_PAD;
    config.dmrs_symbol_mask = (1u << 2) | (1u << 11);
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    config.codebook_enabled = codebook ? 1 : 0;
    return config;
}

}

int main()
{
    auto config = MakeConfig(3, 4, true);
    airan::PuschMimoLayout layout {};
    std::vector<uint32_t> index(rmb::SCATTER_INDEX_ELEMS);
    assert(rmb::BuildCurrentProfile(config, &layout, index.data(), index.size()) == rmb::OK);
    assert(layout.num_dmrs_symbols == 2);
    assert(layout.num_data_symbols == 12);
    assert(layout.num_data_re == 19152);
    assert(layout.data_stride == 19200);
    assert(layout.codeword_symbols == 57456);
    assert(layout.codeword_stride == 57600);
    assert(std::count(index.begin(), index.end(),
                      rmb::ZERO_SLOT * sizeof(uint16_t)) ==
           static_cast<ptrdiff_t>(rmb::N_FFT - rmb::N_SC_USED));

    std::vector<bool> source_seen(rmb::N_SC_USED, false);
    for (uint32_t offset : index) {
        if (offset == rmb::ZERO_SLOT * sizeof(uint16_t)) continue;
        assert((offset & 1u) == 0);
        assert(offset / sizeof(uint16_t) < rmb::N_SC_USED);
        assert(!source_seen[offset / sizeof(uint16_t)]);
        source_seen[offset / sizeof(uint16_t)] = true;
    }
    assert(std::all_of(source_seen.begin(), source_seen.end(),
                       [](bool seen) { return seen; }));

    const uint32_t ports = config.num_tx_ports;
    std::vector<uint16_t> input_re(rmb::PortGridElems(ports));
    std::vector<uint16_t> input_im(rmb::PortGridElems(ports));
    for (size_t element = 0; element < input_re.size(); ++element) {
        input_re[element] = static_cast<uint16_t>((element % 65534u) + 1u);
        input_im[element] = static_cast<uint16_t>(0xffffu - (element % 65534u));
    }
    for (uint32_t port = 0; port < ports; ++port) {
        for (uint32_t symbol = 0; symbol < rmb::N_SYMBOLS; ++symbol) {
            const size_t base = static_cast<size_t>(port) * rmb::GRID_PORT_STRIDE +
                                static_cast<size_t>(symbol) * rmb::N_SC_PAD;
            std::fill(input_re.begin() + base + rmb::N_SC_USED,
                      input_re.begin() + base + rmb::N_SC_PAD, uint16_t{0x7b00});
            std::fill(input_im.begin() + base + rmb::N_SC_USED,
                      input_im.begin() + base + rmb::N_SC_PAD, uint16_t{0xfb00});
        }
    }
    std::vector<uint16_t> output_re(rmb::FftGridElems(ports), 0xa5a5u);
    std::vector<uint16_t> output_im(rmb::FftGridElems(ports), 0x5a5au);
    assert(rmb::ReferenceMap(input_re.data(), input_im.data(), config, layout,
                             index.data(), index.size(), output_re.data(),
                             output_im.data()) == rmb::OK);
    for (uint32_t port = 0; port < ports; ++port) {
        for (uint32_t symbol = 0; symbol < rmb::N_SYMBOLS; ++symbol) {
            const size_t input_base = static_cast<size_t>(port) * rmb::GRID_PORT_STRIDE +
                                      static_cast<size_t>(symbol) * rmb::N_SC_PAD;
            const size_t output_base = static_cast<size_t>(port) * rmb::FFT_PORT_STRIDE +
                                       static_cast<size_t>(symbol) * rmb::N_FFT;
            for (uint32_t destination = 0; destination < rmb::N_FFT; ++destination) {
                const uint32_t offset = index[destination];
                if (offset == rmb::ZERO_SLOT * sizeof(uint16_t)) {
                    assert(output_re[output_base + destination] == 0);
                    assert(output_im[output_base + destination] == 0);
                } else {
                    assert(output_re[output_base + destination] ==
                           input_re[input_base + offset / sizeof(uint16_t)]);
                    assert(output_im[output_base + destination] ==
                           input_im[input_base + offset / sizeof(uint16_t)]);
                }
            }
        }
    }

    rmb::ReMapBatchOpArgsV1 args {};
    args.abi_version = rmb::ABI_VERSION;
    args.struct_size = sizeof(args);
    args.port_grid_re = input_re.data();
    args.port_grid_im = input_im.data();
    args.fft_grid_re = output_re.data();
    args.fft_grid_im = output_im.data();
    args.config = &config;
    args.layout = &layout;
    args.stream = reinterpret_cast<void *>(uintptr_t{1});
    assert(rmb::ValidateOpArgs(args) == rmb::OK);

    auto wrong_layout = layout;
    ++wrong_layout.data_stride;
    args.layout = &wrong_layout;
    assert(rmb::ValidateOpArgs(args) == rmb::LAYOUT_MISMATCH);
    args.layout = &layout;

    config.codebook_enabled = 0;
    assert(rmb::ValidateOpArgs(args) == rmb::UNSUPPORTED_PROFILE);
    config = MakeConfig(4, 4, false);
    assert(rmb::BuildCurrentProfile(config, &layout, index.data(), index.size()) == rmb::OK);
    return 0;
}
