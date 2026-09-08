#include <acl/acl.h>
#include "aclrtlaunch_mimo_detect_bri_kernel.h"
#include "mimo_detect_bri.h"
#include "tiling/platform/platform_ascendc.h"
#include "../bri_grouped_rhs_adapter/bri_grouped_rhs_adapter.h"
#include "pusch_mimo_runtime_config.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

extern "C" void GenerateTiling(const char *soc_version, uint8_t *buffer);

namespace fs = std::filesystem;
using namespace airan;

#define ACL_OK(call) do { \
    const aclError status_ = (call); \
    if (status_ != ACL_ERROR_NONE) { \
        std::fprintf(stderr, "[HARD_FAIL] %s=%d\n", #call, status_); \
        return 1; \
    } \
} while (0)

static bool ReadExact(const fs::path &path, std::vector<uint16_t> *value,
                      size_t elements)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || static_cast<size_t>(stream.tellg()) != elements * sizeof(uint16_t)) {
        std::fprintf(stderr, "[HARD_FAIL] %s: expected exactly %zu bytes\n",
                     path.c_str(), elements * sizeof(uint16_t));
        return false;
    }
    value->resize(elements);
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(value->data()),
                static_cast<std::streamsize>(elements * sizeof(uint16_t)));
    return stream.good();
}

static bool WriteExact(const fs::path &path, const std::vector<uint16_t> &value)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char *>(value.data()),
                 static_cast<std::streamsize>(value.size() * sizeof(uint16_t)));
    return stream.good();
}

static bool IsFiniteHalf(uint16_t bits)
{
    return (bits & 0x7c00u) != 0x7c00u;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s IO_PACK_DIR OUTPUT_DIR ACTIVE_LAYERS\n", argv[0]);
        return 2;
    }
    namespace adapter = airan::bri_grouped_rhs_adapter;
    constexpr uint32_t group_batch = adapter::GROUP_SIZE;
    constexpr size_t matrix_elements = static_cast<size_t>(N_RE) * NR * NL;
    constexpr size_t output_elements = static_cast<size_t>(NL) * N_RE;
    constexpr size_t grouped_y_elements =
        static_cast<size_t>(N_RE / group_batch) * NR * NL;

    const fs::path input(argv[1]);
    const fs::path output(argv[2]);
    const uint32_t active_layers = static_cast<uint32_t>(std::stoul(argv[3]));
    if (active_layers == 0 || active_layers > 4 || active_layers != MIMO_KR) {
        std::fprintf(stderr, "[HARD_FAIL] active layer/build mismatch: arg=%u build=%u\n",
                     active_layers, static_cast<unsigned>(MIMO_KR));
        return 2;
    }
    PuschMimoRuntimeConfig runtime{};
    bool runtime_enabled = false;
    std::string why;
    if (LoadMimoRuntimeConfigFromEnv(active_layers, &runtime, &runtime_enabled,
                                     &why) != MimoRuntimeConfigStatus::kSuccess) {
        std::fprintf(stderr, "[HARD_FAIL] runtime config: %s\n", why.c_str());
        return 1;
    }
    if (runtime_enabled &&
        (runtime.rx_bucket != NR || runtime.layer_bucket != NL ||
         runtime.num_symbols != 14 || runtime.padded_subcarriers != 1664 ||
         runtime.grid_re_per_port != N_RE)) {
        std::fprintf(stderr,
                     "[HARD_FAIL] runtime config exceeds current BRI kernel profile\n");
        return 1;
    }
    fs::create_directories(output);
    std::vector<uint16_t> hr, hi, canonical_yr, canonical_yi, noise;
    if (!ReadExact(input / "hrm_re.bin", &hr, matrix_elements) ||
        !ReadExact(input / "hrm_im.bin", &hi, matrix_elements) ||
        !ReadExact(input / "yvpad_re.bin", &canonical_yr, matrix_elements) ||
        !ReadExact(input / "yvpad_im.bin", &canonical_yi, matrix_elements) ||
        !ReadExact(input / "no.bin", &noise, N_RE)) {
        return 1;
    }

    std::vector<uint16_t> grouped_yr(grouped_y_elements, 0);
    std::vector<uint16_t> grouped_yi(grouped_y_elements, 0);
    bool any_h = false;
    bool any_y = false;
    for (uint32_t re = 0; re < N_RE; ++re) {
        if (!IsFiniteHalf(noise[re]) || (noise[re] & 0x8000u) != 0) {
            std::fprintf(stderr, "[HARD_FAIL] invalid detector noise at RE %u\n", re);
            return 1;
        }
        for (uint32_t rx = 0; rx < NR; ++rx) {
            const size_t base = adapter::CanonicalIndex(re,rx,0);
            if (!IsFiniteHalf(hr[base]) || !IsFiniteHalf(hi[base])) {
                std::fprintf(stderr, "[HARD_FAIL] non-finite active H at RE %u RX %u\n", re, rx);
                return 1;
            }
            any_h = any_h || hr[base] != 0 || hi[base] != 0;
            for (uint32_t layer = active_layers; layer < NL; ++layer) {
                if (hr[base + layer] != 0 || hi[base + layer] != 0) {
                    std::fprintf(stderr, "[HARD_FAIL] inactive H is nonzero\n");
                    return 1;
                }
                if (canonical_yr[base + layer] != canonical_yr[base] ||
                    canonical_yi[base + layer] != canonical_yi[base]) {
                    std::fprintf(stderr, "[HARD_FAIL] io_pack y is not repeated over 16 columns\n");
                    return 1;
                }
            }
            if (!IsFiniteHalf(canonical_yr[base]) ||
                !IsFiniteHalf(canonical_yi[base])) {
                std::fprintf(stderr, "[HARD_FAIL] non-finite y at RE %u RX %u\n", re, rx);
                return 1;
            }
            any_y = any_y || canonical_yr[base] != 0 || canonical_yi[base] != 0;
            const size_t grouped_index = adapter::GroupedIndex(re,rx);
            grouped_yr[grouped_index] = canonical_yr[base];
            grouped_yi[grouped_index] = canonical_yi[base];
        }
    }
    if (!any_h || !any_y) {
        std::fprintf(stderr, "[HARD_FAIL] all-zero H or y at detector boundary\n");
        return 1;
    }

    ACL_OK(aclInit(nullptr));
    ACL_OK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_OK(aclrtCreateStream(&stream));
    void *dhr = nullptr, *dhi = nullptr, *dyr = nullptr, *dyi = nullptr;
    void *dnoise = nullptr, *dxr = nullptr, *dxi = nullptr, *dne = nullptr;
    void *ddummy = nullptr, *dworkspace = nullptr, *dtiling = nullptr;
    const size_t matrix_bytes = matrix_elements * sizeof(uint16_t);
    const size_t grouped_y_bytes = grouped_y_elements * sizeof(uint16_t);
    const size_t output_bytes = output_elements * sizeof(uint16_t);
    const size_t noise_bytes = static_cast<size_t>(N_RE) * sizeof(uint16_t);
    const size_t workspace_bytes =
        platform_ascendc::PlatformAscendCManager::GetInstance("Ascend310P1")
            ->GetLibApiWorkSpaceSize();
    ACL_OK(aclrtMalloc(&dhr, matrix_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dhi, matrix_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dyr, grouped_y_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dyi, grouped_y_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dnoise, noise_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dxr, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dxi, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dne, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&ddummy, 64, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dworkspace, workspace_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dtiling, TILING_TOTAL_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMemcpy(dhr, matrix_bytes, hr.data(), matrix_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(dhi, matrix_bytes, hi.data(), matrix_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(dyr, grouped_y_bytes, grouped_yr.data(), grouped_y_bytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(dyi, grouped_y_bytes, grouped_yi.data(), grouped_y_bytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(dnoise, noise_bytes, noise.data(), noise_bytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemset(ddummy, 64, 0, 64));
    std::vector<uint8_t> tiling(TILING_TOTAL_SIZE, 0);
    GenerateTiling("Ascend310P1", tiling.data());
    ACL_OK(aclrtMemcpy(dtiling, tiling.size(), tiling.data(), tiling.size(),
                       ACL_MEMCPY_HOST_TO_DEVICE));

    const uint32_t launch_status = ACLRT_LAUNCH_KERNEL(mimo_detect_bri_kernel)(
        BLOCK_DIM, stream, dhr, dhi, ddummy, ddummy, dnoise, ddummy,
        dxr, dxi, dne, dyr, dyi, dworkspace, dtiling);
    if (launch_status != ACL_ERROR_NONE) {
        std::fprintf(stderr, "[HARD_FAIL] BRI launch=%u\n", launch_status);
        return 1;
    }
    ACL_OK(aclrtSynchronizeStream(stream));
    std::vector<uint16_t> xr(output_elements), xi(output_elements), ne(output_elements);
    ACL_OK(aclrtMemcpy(xr.data(), output_bytes, dxr, output_bytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_OK(aclrtMemcpy(xi.data(), output_bytes, dxi, output_bytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_OK(aclrtMemcpy(ne.data(), output_bytes, dne, output_bytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
    if (!std::any_of(xr.begin(), xr.begin() + active_layers * N_RE, [](uint16_t v) { return v != 0; }) ||
        !std::any_of(ne.begin(), ne.begin() + active_layers * N_RE, [](uint16_t v) { return v != 0; }) ||
        !WriteExact(output / "xhat_re.bin", xr) ||
        !WriteExact(output / "xhat_im.bin", xi) ||
        !WriteExact(output / "no_eff.bin", ne)) {
        std::fprintf(stderr, "[HARD_FAIL] invalid/all-zero detector output\n");
        return 1;
    }

    for (void *pointer : {dtiling, dworkspace, ddummy, dne, dxi, dxr,
                          dnoise, dyi, dyr, dhi, dhr}) {
        aclrtFree(pointer);
    }
    ACL_OK(aclrtDestroyStream(stream));
    ACL_OK(aclrtResetDevice(0));
    ACL_OK(aclFinalize());
    std::printf("[PASS] BRI grouped-RHS ABI adapter activeL=%u capacity=%ux%u "
                "runtime=%s: canonical io_pack validated, actual kernel launched\n",
                active_layers, static_cast<unsigned>(NR), static_cast<unsigned>(NL),
                runtime_enabled ? "profile" : "legacy");
    return 0;
}
