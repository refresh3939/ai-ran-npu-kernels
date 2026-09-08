#include "rate_match_mimo.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace airan::rate_match_mimo;

namespace {

PuschMimoConfig MakeConfig(uint16_t layers)
{
    PuschMimoConfig config {};
    config.abi_version = ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = layers;
    config.num_tx_ports = layers == 1 ? 1 : (layers == 2 ? 2 : 4);
    config.num_rx_antennas = 64;
    config.qm = Q_M;
    config.num_symbols = 14;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = 14;
    config.used_subcarriers = N_SC_USED;
    config.padded_subcarriers = 1664;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    return config;
}

RateMatchFecConfigV1 MakeFec()
{
    RateMatchFecConfigV1 fec {};
    fec.abi_version = ABI_VERSION;
    fec.struct_size = sizeof(fec);
    fec.num_code_blocks = C_NUM;
    fec.encoded_stride = N_CB_BUF;
    fec.base_graph = 1;
    fec.lifting_size = LDPC_Z;
    return fec;
}

}

int main()
{
    const RateMatchFecConfigV1 fec = MakeFec();
    std::vector<int8_t> code_blocks(InputElems(fec));
    for (size_t i = 0; i < code_blocks.size(); ++i) {
        code_blocks[i] = static_cast<int8_t>((i * 17u + i / N_CB_BUF) & 1u);
    }

    for (uint16_t layers = 1; layers <= MAX_LAYERS; ++layers) {
        const PuschMimoConfig config = MakeConfig(layers);
        PuschMimoLayout layout {};
        KernelMetadata metadata {};
        assert(BuildCurrentProfile(config, fec, MAX_SLOTS, &layout, &metadata) == OK);
        assert(layout.codeword_symbols == layers * N_DATA_RE);
        assert(layout.codeword_stride == layers * N_DATA_PAD);
        assert(metadata.num_slots == MAX_SLOTS);

        std::vector<RateMatchDescriptor> desc(C_NUM);
        assert(BuildRateMatchDescriptors(config, fec, MAX_SLOTS, desc.data(),
                                         desc.size()) == OK);
        uint64_t sum_e = 0;
        for (uint32_t cb = 0; cb < C_NUM; ++cb) {
            sum_e += desc[cb].e;
            assert(desc[cb].k0 == 0);
            assert(desc[cb].ncb == N_CB_BUF);
            assert(desc[cb].cw_bit_offset == cb * N_CB_BUF);
        }
        assert(sum_e == static_cast<uint64_t>(MAX_SLOTS) * Q_M * layers *
                            N_DATA_RE);
        if (layers == 1) {
            for (uint32_t cb = 0; cb < 87; ++cb) assert(desc[cb].e == 24640);
            for (uint32_t cb = 87; cb < C_NUM; ++cb) assert(desc[cb].e == 24648);
        }

        const char *data_root = std::getenv("AIRAN_DATA_DIR");
        std::vector<int8_t> generated_code_blocks;
        const int8_t *input_bits = code_blocks.data();
        if (data_root != nullptr) {
            const std::string desc_path = std::string(data_root) +
                "/golden/rank" + std::to_string(layers) + "/rm_desc.bin";
            std::ifstream desc_input(desc_path, std::ios::binary | std::ios::ate);
            assert(desc_input);
            assert(static_cast<size_t>(desc_input.tellg()) ==
                   desc.size() * sizeof(RateMatchDescriptor));
            std::vector<RateMatchDescriptor> python_desc(desc.size());
            desc_input.seekg(0);
            desc_input.read(reinterpret_cast<char *>(python_desc.data()),
                            static_cast<std::streamsize>(
                                python_desc.size() * sizeof(RateMatchDescriptor)));
            assert(desc_input);
            for (uint32_t cb = 0; cb < C_NUM; ++cb) {
                assert(python_desc[cb].e == desc[cb].e);
                assert(python_desc[cb].k0 == desc[cb].k0);
                assert(python_desc[cb].ncb == desc[cb].ncb);
                assert(python_desc[cb].cw_bit_offset == desc[cb].cw_bit_offset);
            }

            const std::string input_path = std::string(data_root) +
                "/golden/rank" + std::to_string(layers) + "/code_blocks.bin";
            std::ifstream input(input_path, std::ios::binary | std::ios::ate);
            assert(input);
            assert(static_cast<size_t>(input.tellg()) == code_blocks.size());
            generated_code_blocks.resize(code_blocks.size());
            input.seekg(0);
            input.read(reinterpret_cast<char *>(generated_code_blocks.data()),
                       static_cast<std::streamsize>(generated_code_blocks.size()));
            assert(input);
            input_bits = generated_code_blocks.data();
        }

        std::vector<int16_t> output(OutputElems(MAX_SLOTS, layers), -1);
        assert(ReferenceRateMatch(input_bits, config, layout, fec,
                                  MAX_SLOTS, desc.data(), output.data()) == OK);
        for (uint32_t slot = 0; slot < MAX_SLOTS; ++slot) {
            for (uint32_t bit = 0; bit < Q_M; ++bit) {
                const size_t base = static_cast<size_t>(slot) * Q_M *
                                        layout.codeword_stride +
                                    static_cast<size_t>(bit) *
                                        layout.codeword_stride;
                for (uint32_t i = layout.codeword_symbols;
                     i < layout.codeword_stride; ++i) {
                    assert(output[base + i] == 0);
                }
            }
        }

        if (data_root != nullptr) {
            const std::string path = std::string(data_root) + "/golden/rank" +
                                     std::to_string(layers) + "/bits_nr.bin";
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            assert(input);
            assert(static_cast<size_t>(input.tellg()) ==
                   output.size() * sizeof(int16_t));
            std::vector<int16_t> python(output.size());
            input.seekg(0);
            input.read(reinterpret_cast<char *>(python.data()),
                       static_cast<std::streamsize>(python.size() * sizeof(int16_t)));
            assert(input);
            assert(python == output);
        }
    }

    PuschMimoConfig invalid = MakeConfig(1);
    invalid.num_layers = 0;
    PuschMimoLayout layout {};
    KernelMetadata metadata {};
    assert(BuildCurrentProfile(invalid, fec, 1, &layout, &metadata) ==
           UNSUPPORTED_PROFILE);
    assert(BuildCurrentProfile(MakeConfig(1), fec, MAX_SLOTS + 1, &layout,
                               &metadata) == INVALID_ARGUMENT);
    return 0;
}
