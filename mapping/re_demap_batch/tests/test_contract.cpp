#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "re_demap_batch.h"

namespace rdb = airan::re_demap_batch;

namespace {

airan::PuschMimoConfig MakeConfig(uint16_t numRx)
{
    airan::PuschMimoConfig config {};
    config.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = 2;
    config.num_tx_ports = 2;
    config.num_rx_antennas = numRx;
    config.qm = 8;
    config.num_symbols = rdb::N_SYMBOLS;
    config.fft_size = rdb::N_FFT;
    config.num_rb = rdb::N_RB;
    config.num_allocated_symbols = rdb::N_SYMBOLS;
    config.used_subcarriers = rdb::N_SC_USED;
    config.padded_subcarriers = rdb::N_SC_PAD;
    config.dmrs_symbol_mask = (1u << 2) | (1u << 11);
    return config;
}

}  // namespace

int main()
{
    constexpr uint32_t numRx = 2;
    auto config = MakeConfig(numRx);
    airan::PuschMimoLayout layout {};
    rdb::KernelMetadata metadata {};
    assert(rdb::BuildCurrentProfile(config, &layout, &metadata) == rdb::OK);
    assert(layout.num_dmrs_symbols == 2);
    assert(layout.num_data_symbols == 12);
    assert(layout.num_data_re == 19152);
    assert(layout.data_stride == 19200);
    assert(layout.codeword_symbols == 38304);
    assert(layout.codeword_stride == 38400);
    assert(metadata.magic == rdb::META_MAGIC);
    assert(metadata.num_rx_antennas == numRx);

    std::vector<uint32_t> index(rdb::N_SC_PAD, 0xffffffffu);
    assert(rdb::BuildGatherIndex(index.data(), index.size()) == rdb::OK);
    assert(std::all_of(index.begin(), index.begin() + rdb::N_SC_USED,
                       [](uint32_t value) {
                           return (value & 1u) == 0 &&
                                  value < rdb::N_FFT * sizeof(uint16_t);
                       }));
    assert(std::all_of(index.begin() + rdb::N_SC_USED, index.end(),
                       [](uint32_t value) { return value == 0; }));

    std::vector<uint16_t> inputRe(rdb::InputElems(numRx));
    std::vector<uint16_t> inputIm(rdb::InputElems(numRx));
    for (size_t element = 0; element < inputRe.size(); ++element) {
        inputRe[element] = static_cast<uint16_t>((element % 65534u) + 1u);
        inputIm[element] = static_cast<uint16_t>(0xffffu - (element % 65534u));
    }
    std::vector<uint16_t> outputRe(rdb::OutputElems(numRx), 0xa5a5u);
    std::vector<uint16_t> outputIm(rdb::OutputElems(numRx), 0xa5a5u);
    assert(rdb::Reference(inputRe.data(), inputIm.data(), numRx, index.data(),
                          outputRe.data(), outputIm.data()) == rdb::OK);
    for (uint32_t rx = 0; rx < numRx; ++rx) {
        for (uint32_t symbol = 0; symbol < rdb::N_SYMBOLS; ++symbol) {
            const size_t inputBase = static_cast<size_t>(rx) * rdb::GRID_IN_ELEMS +
                                     static_cast<size_t>(symbol) * rdb::N_FFT;
            const size_t outputBase = static_cast<size_t>(rx) * rdb::GRID_OUT_ELEMS +
                                      static_cast<size_t>(symbol) * rdb::N_SC_PAD;
            assert(outputRe[outputBase] == inputRe[inputBase + index[0] / 2]);
            assert(outputIm[outputBase] == inputIm[inputBase + index[0] / 2]);
            for (uint32_t sc = rdb::N_SC_USED; sc < rdb::N_SC_PAD; ++sc) {
                assert(outputRe[outputBase + sc] == 0);
                assert(outputIm[outputBase + sc] == 0);
            }
        }
    }

    rdb::ReDemapBatchOpArgsV1 args {};
    args.abi_version = rdb::ABI_VERSION;
    args.struct_size = sizeof(args);
    args.fft_grid_re = inputRe.data();
    args.fft_grid_im = inputIm.data();
    args.rx_grid_re = outputRe.data();
    args.rx_grid_im = outputIm.data();
    args.config = &config;
    args.layout = &layout;
    args.stream = reinterpret_cast<void *>(uintptr_t{1});
    assert(rdb::ValidateOpArgs(args) == rdb::OK);

    auto wrongLayout = layout;
    ++wrongLayout.data_stride;
    args.layout = &wrongLayout;
    assert(rdb::ValidateOpArgs(args) == rdb::LAYOUT_MISMATCH);
    args.layout = &layout;

    args.rx_grid_re = const_cast<void *>(args.fft_grid_im);
    assert(rdb::ValidateOpArgs(args) == rdb::INVALID_ARGUMENT);
    args.rx_grid_re = outputRe.data();

    config.num_rx_antennas = rdb::MAX_RX_ANTENNAS + 1;
    assert(rdb::ValidateOpArgs(args) == rdb::UNSUPPORTED_PROFILE);
    return 0;
}
