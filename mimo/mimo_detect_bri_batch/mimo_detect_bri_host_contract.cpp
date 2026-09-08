#include "mimo_detect_bri_host_contract.h"

#include <sstream>

namespace airan {
namespace {

bool Fail(std::string *reason, const std::string &message)
{
    if (reason != nullptr) {
        *reason = message;
    }
    return false;
}

bool HalfIsFinite(uint16_t value)
{
    return (value & 0x7c00u) != 0x7c00u;
}

bool HalfIsZero(uint16_t value)
{

    return (value & 0x7fffu) == 0;
}

bool CheckElements(
    const char *name, const uint16_t *data, size_t actual, size_t expected,
    std::string *reason)
{
    if (data == nullptr) {
        return Fail(reason, std::string(name) + " is null");
    }
    if (actual != expected) {
        std::ostringstream out;
        out << name << " has " << actual << " fp16 elements; expected " << expected;
        return Fail(reason, out.str());
    }
    return true;
}

bool ValidateHrmPlane(
    const char *name, const uint16_t *data, uint32_t active_layers,
    std::string *reason)
{
    const size_t rows = static_cast<size_t>(MIMO_IO_RE) * MIMO_IO_NR;
    for (size_t row = 0; row < rows; ++row) {
        const size_t base = row * MIMO_IO_PHYSICAL_LAYERS;
        for (uint32_t layer = 0; layer < MIMO_IO_PHYSICAL_LAYERS; ++layer) {
            const uint16_t value = data[base + layer];
            if (!HalfIsFinite(value)) {
                std::ostringstream out;
                out << name << " contains non-finite fp16 at flat index " << base + layer;
                return Fail(reason, out.str());
            }
            if (layer >= active_layers && !HalfIsZero(value)) {
                std::ostringstream out;
                out << name << " inactive layer " << layer
                    << " is not zero at [re=" << row / MIMO_IO_NR
                    << ",rx=" << row % MIMO_IO_NR << "]";
                return Fail(reason, out.str());
            }
        }
    }
    return true;
}

bool ValidateYvpadPlane(const char *name, const uint16_t *data, std::string *reason)
{
    const size_t rows = static_cast<size_t>(MIMO_IO_RE) * MIMO_IO_NR;
    for (size_t row = 0; row < rows; ++row) {
        const size_t base = row * MIMO_IO_PHYSICAL_LAYERS;
        const uint16_t expected = data[base];
        if (!HalfIsFinite(expected)) {
            std::ostringstream out;
            out << name << " contains non-finite fp16 at flat index " << base;
            return Fail(reason, out.str());
        }
        for (uint32_t layer = 1; layer < MIMO_IO_PHYSICAL_LAYERS; ++layer) {
            const uint16_t actual = data[base + layer];
            if (actual != expected) {
                std::ostringstream out;
                out << name << " is not repeated-y layout at [re="
                    << row / MIMO_IO_NR << ",rx=" << row % MIMO_IO_NR
                    << ",layer=" << layer << "]";
                return Fail(reason, out.str());
            }
        }
    }
    return true;
}

}

bool ValidateMimoDetectBriHostContract(
    const MimoDetectBriHostContract &contract, std::string *reason)
{
    if (contract.abi_version != MIMO_IO_ABI_VERSION) {
        return Fail(reason, "unsupported mimo_detect_io_pack ABI version");
    }
    if (contract.nr != MIMO_IO_NR ||
        contract.physical_layers != MIMO_IO_PHYSICAL_LAYERS ||
        contract.symbols != MIMO_IO_SYMBOLS ||
        contract.subcarriers != MIMO_IO_SUBCARRIERS ||
        contract.pack_mode != 0) {
        return Fail(reason,
            "io_pack ABI dimensions do not match the compiled detector "
            "variant (symbols=14, subcarriers=1664, PACK=0 required)");
    }
    if (contract.active_layers < 1 || contract.active_layers > contract.physical_layers) {
        return Fail(reason, "active L must be in [1,16]");
    }
    if (contract.bri_block_size < 1 ||
        contract.physical_layers % contract.bri_block_size != 0) {
        return Fail(reason, "BRI_B must be a positive divisor of physical NL=16");
    }
    const uint32_t min_block_or_active =
        contract.bri_block_size < contract.active_layers
            ? contract.bri_block_size : contract.active_layers;
    if (contract.nr < 8u * min_block_or_active) {
        return Fail(reason, "Neumann constraint violated: NR >= 8*min(BRI_B, active L)");
    }
    if (2u * contract.bri_block_size < contract.active_layers) {
        return Fail(reason, "BRI constraint violated: 2*BRI_B >= active L");
    }
    if (reason != nullptr) {
        reason->clear();
    }
    return true;
}

bool ValidateMimoDetectBriPhysicalInputs(
    const MimoDetectBriHostContract &contract,
    const uint16_t *hrm_re, size_t hrm_re_elements,
    const uint16_t *hrm_im, size_t hrm_im_elements,
    const uint16_t *yvpad_re, size_t yvpad_re_elements,
    const uint16_t *yvpad_im, size_t yvpad_im_elements,
    const uint16_t *noise, size_t noise_elements,
    std::string *reason)
{
    if (!ValidateMimoDetectBriHostContract(contract, reason)) {
        return false;
    }
    const size_t matrix_elements = MimoIoMatrixElements();
    if (!CheckElements("hrm_re", hrm_re, hrm_re_elements, matrix_elements, reason) ||
        !CheckElements("hrm_im", hrm_im, hrm_im_elements, matrix_elements, reason) ||
        !CheckElements("yvpad_re", yvpad_re, yvpad_re_elements, matrix_elements, reason) ||
        !CheckElements("yvpad_im", yvpad_im, yvpad_im_elements, matrix_elements, reason) ||
        !CheckElements("no", noise, noise_elements, MimoIoNoiseElements(), reason)) {
        return false;
    }
    if (!ValidateHrmPlane("hrm_re", hrm_re, contract.active_layers, reason) ||
        !ValidateHrmPlane("hrm_im", hrm_im, contract.active_layers, reason) ||
        !ValidateYvpadPlane("yvpad_re", yvpad_re, reason) ||
        !ValidateYvpadPlane("yvpad_im", yvpad_im, reason)) {
        return false;
    }
    for (size_t i = 0; i < noise_elements; ++i) {
        const uint16_t value = noise[i];
        if (!HalfIsFinite(value) ||
            ((value & 0x8000u) != 0 && !HalfIsZero(value))) {
            std::ostringstream out;
            out << "no contains negative or non-finite fp16 at RE " << i;
            return Fail(reason, out.str());
        }
    }
    if (reason != nullptr) {
        reason->clear();
    }
    return true;
}

}
