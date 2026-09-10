#include "mimo_detect_bri_host_contract.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Require(bool value, const char *message)
{
    if (!value) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main()
{
    using namespace airan;
    std::string reason;
    MimoDetectBriHostContract contract{
        MIMO_IO_ABI_VERSION, MIMO_IO_NR, MIMO_IO_PHYSICAL_LAYERS, 4,
        MIMO_IO_SYMBOLS, MIMO_IO_SUBCARRIERS, 2, 0};
    Require(ValidateMimoDetectBriHostContract(contract, &reason),
            "valid L=4/B=2 contract rejected");
    Require(MimoIoOutputElements() * sizeof(uint16_t) == 745472,
            "output ABI is not [16,23296] fp16");

    MimoDetectBriHostContract bad = contract;
    bad.nr = MIMO_IO_NR == 64 ? 32 : 64;
    Require(!ValidateMimoDetectBriHostContract(bad, &reason),
            "wrong compiled NR was accepted");
    bad = contract;
    bad.active_layers = 16;
    bad.bri_block_size = 4;
    Require(!ValidateMimoDetectBriHostContract(bad, &reason),
            "non-convergent L=16/B=4 was accepted");

    // One shared all-zero matrix is a valid hrm plane and repeated-y yvpad
    // plane. This keeps the test's peak allocation near one physical tensor.
    std::vector<uint16_t> matrix(MimoIoMatrixElements(), 0);
    std::vector<uint16_t> noise(MimoIoNoiseElements(), 0);
    Require(ValidateMimoDetectBriPhysicalInputs(
                contract,
                matrix.data(), matrix.size(), matrix.data(), matrix.size(),
                matrix.data(), matrix.size(), matrix.data(), matrix.size(),
                noise.data(), noise.size(), &reason),
            "valid physical buffers rejected");

    Require(!ValidateMimoDetectBriPhysicalInputs(
                contract,
                matrix.data(), matrix.size(), matrix.data(), matrix.size(),
                matrix.data(), matrix.size() / 8, matrix.data(), matrix.size(),
                noise.data(), noise.size(), &reason),
            "legacy N_RE/8 y_group-sized input was accepted");

    matrix[4] = 0x3c00u;  // 1.0 in first inactive H column.
    Require(!ValidateMimoDetectBriPhysicalInputs(
                contract,
                matrix.data(), matrix.size(), matrix.data(), matrix.size(),
                matrix.data(), matrix.size(), matrix.data(), matrix.size(),
                noise.data(), noise.size(), &reason),
            "non-zero inactive H layer was accepted");
    matrix[4] = 0;

    matrix[1] = 0x3c00u;  // layer 1 differs from layer 0 in yvpad.
    Require(!ValidateMimoDetectBriPhysicalInputs(
                contract,
                matrix.data(), matrix.size(), matrix.data(), matrix.size(),
                matrix.data(), matrix.size(), matrix.data(), matrix.size(),
                noise.data(), noise.size(), &reason),
            "non-repeated yvpad was accepted");
    matrix[1] = 0;

    noise[0] = 0xbc00u;  // -1.0
    Require(!ValidateMimoDetectBriPhysicalInputs(
                contract,
                matrix.data(), matrix.size(), matrix.data(), matrix.size(),
                matrix.data(), matrix.size(), matrix.data(), matrix.size(),
                noise.data(), noise.size(), &reason),
            "negative noise was accepted");

    std::cout << "PASS: host contract and physical ABI positive/negative cases\n";
    return EXIT_SUCCESS;
}
