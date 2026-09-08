






#include "ldpc_encode.h"
#include "data_utils.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_ldpc_encode_kernel.h"
#endif

namespace airan {
extern "C" uint8_t* GenerateTiling  (const char*, uint32_t);
extern "C" size_t   GetTilingSize   ();
extern "C" size_t   GetWorkspaceSize();
}

constexpr int N_BENCH_RUNS = 20;
using Clock = std::chrono::high_resolution_clock;

static const char* GetDataRoot() {
    const char* r = std::getenv("AIRAN_DATA_DIR");
    return r ? r : "../../data";
}

static bool ReadBinExact(const std::string& path, void* buf, size_t expect) {
    size_t actual = 0;
    if (!ReadFile(path, actual, buf, expect)) {
        fprintf(stderr, "[err] ReadFile failed: %s\n", path.c_str());
        return false;
    }
    if (actual != expect) {
        fprintf(stderr, "[err] %s size %zu != %zu\n", path.c_str(), actual, expect);
        return false;
    }
    return true;
}

static size_t CompareInt8(const int8_t* a, const int8_t* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) ++d;
    return d;
}

int main() {
    const std::string root = GetDataRoot();

    const std::string pInfo   = root + "/golden/input.bin";
    const std::string pGolden = root + "/golden/output.bin";
    const std::string pOut    = root + "/ascend_output/output.bin";
    const std::string wRoot   = root + "/weights/ldpc_bg1_z384_shifts";

    const size_t szInfo   = airan::INFO_BYTES;
    const size_t szOut    = airan::OUTPUT_BYTES;
    const size_t szSA     = airan::SHIFT_A_BYTES;
    const size_t szSBi    = airan::SHIFT_BI_BYTES;
    const size_t szSC     = airan::SHIFT_C_BYTES;
    const size_t szSD     = airan::SHIFT_D_BYTES;
    const size_t szDbg    = airan::LDPC_C_NUM * airan::DBG_WORDS_PER_CB * sizeof(int32_t);


    const size_t sz4Info = 4 * airan::LDPC_K;
    const size_t sz4Out  = 4 * airan::LDPC_N_RAW;

    auto pad16 = [](size_t e) { return ((e + 15) / 16) * 16; };
    const size_t szSA_pad  = pad16(airan::SHIFT_A_ELEMS)  * sizeof(int16_t);
    const size_t szSBi_pad = pad16(airan::SHIFT_BI_ELEMS) * sizeof(int16_t);
    const size_t szSC_pad  = pad16(airan::SHIFT_C_ELEMS)  * sizeof(int16_t);
    const size_t szSD_pad  = pad16(airan::SHIFT_D_ELEMS)  * sizeof(int16_t);

    printf("=== LDPC Encoder (N_CB=%u) ===\n", airan::LDPC_C_NUM);
    printf("  info bytes:  %zu (%.1f KB)\n", szInfo, szInfo / 1024.0);
    printf("  out bytes:   %zu (%.1f KB)\n", szOut,  szOut  / 1024.0);

    auto* hInfo   = (int8_t*)  std::malloc(szInfo);
    auto* hOut    = (int8_t*)  std::malloc(szOut);
    auto* hGolden = (int8_t*)  std::malloc(szOut);
    auto* hInfo4  = (int8_t*)  std::malloc(sz4Info);
    auto* hGold4  = (int8_t*)  std::malloc(sz4Out);
    auto* hSA     = (int16_t*) std::malloc(szSA);
    auto* hSBi    = (int16_t*) std::malloc(szSBi);
    auto* hSC     = (int16_t*) std::malloc(szSC);
    auto* hSD     = (int16_t*) std::malloc(szSD);
    auto* hDbg    = (int32_t*) std::malloc(szDbg);

    if (!ReadBinExact(pInfo,   hInfo4, sz4Info)) return 1;
    if (!ReadBinExact(pGolden, hGold4, sz4Out))  return 1;
    if (!ReadBinExact(wRoot + "/shift_A.bin",  hSA,  szSA))  return 1;
    if (!ReadBinExact(wRoot + "/shift_Bi.bin", hSBi, szSBi)) return 1;
    if (!ReadBinExact(wRoot + "/shift_C.bin",  hSC,  szSC))  return 1;
    if (!ReadBinExact(wRoot + "/shift_D.bin",  hSD,  szSD))  return 1;


    for (uint32_t cb = 0; cb < airan::LDPC_C_NUM; ++cb) {
        uint32_t src_cb = cb % 4;
        std::memcpy(hInfo   + cb * airan::LDPC_K,     hInfo4 + src_cb * airan::LDPC_K,     airan::LDPC_K);
        std::memcpy(hGolden + cb * airan::LDPC_N_RAW, hGold4 + src_cb * airan::LDPC_N_RAW, airan::LDPC_N_RAW);
    }

#ifndef ASCENDC_CPU_DEBUG
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *dInfo, *dOut, *dSA, *dSBi, *dSC, *dSD, *dDbg;
    CHECK_ACL(aclrtMalloc(&dInfo, szInfo,    ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dOut,  szOut,     ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dSA,   szSA_pad,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dSBi,  szSBi_pad, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dSC,   szSC_pad,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dSD,   szSD_pad,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dDbg,  szDbg,     ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemset(dSA,  szSA_pad,  0, szSA_pad));
    CHECK_ACL(aclrtMemset(dSBi, szSBi_pad, 0, szSBi_pad));
    CHECK_ACL(aclrtMemset(dSC,  szSC_pad,  0, szSC_pad));
    CHECK_ACL(aclrtMemset(dSD,  szSD_pad,  0, szSD_pad));

    CHECK_ACL(aclrtMemcpy(dSA,  szSA,  hSA,  szSA,  ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dSBi, szSBi, hSBi, szSBi, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dSC,  szSC,  hSC,  szSC,  ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dSD,  szSD,  hSD,  szSD,  ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dInfo, szInfo, hInfo, szInfo, ACL_MEMCPY_HOST_TO_DEVICE));

    const uint32_t blockDim = 4;


    printf("\n[warmup] correctness check (blockDim=%u, N_CB=%u)...\n", blockDim, airan::LDPC_C_NUM);
    std::vector<int8_t> init_pattern(szOut, (int8_t)0xAA);
    CHECK_ACL(aclrtMemcpy(dOut, szOut, init_pattern.data(), szOut, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(dDbg, szDbg, 0, szDbg));

    ACLRT_LAUNCH_KERNEL(ldpc_encode_kernel)(blockDim, stream,
        dInfo, dSA, dSBi, dSC, dSD, dDbg, dOut);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(hOut, szOut, dOut, szOut, ACL_MEMCPY_DEVICE_TO_HOST));
    size_t diff = CompareInt8(hOut, hGolden, szOut);
    printf("[warmup] output diff = %zu / %zu  %s\n",
           diff, szOut, diff == 0 ? "PASS" : "FAIL");

    if (diff != 0) {
        printf("\n[dbg] per-CB diff (first 8 CBs):\n");
        for (uint32_t cb = 0; cb < std::min(airan::LDPC_C_NUM, 8u); ++cb) {
            const int8_t* out_cb  = hOut    + cb * airan::LDPC_N_RAW;
            const int8_t* gold_cb = hGolden + cb * airan::LDPC_N_RAW;
            size_t d = CompareInt8(out_cb, gold_cb, airan::LDPC_N_RAW);
            printf("  CB %u: diff=%zu / %u\n", cb, d, airan::LDPC_N_RAW);
        }
        return 1;
    }


    printf("\n[bench] running %d iterations (10 launches per sync)...\n", N_BENCH_RUNS);


    for (int i = 0; i < 3; ++i) {
        ACLRT_LAUNCH_KERNEL(ldpc_encode_kernel)(blockDim, stream,
            dInfo, dSA, dSBi, dSC, dSD, dDbg, dOut);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<double> times_us;
    const int launches_per_sync = 10;
    for (int i = 0; i < N_BENCH_RUNS; ++i) {
        auto t0 = Clock::now();
        for (int j = 0; j < launches_per_sync; ++j) {
            ACLRT_LAUNCH_KERNEL(ldpc_encode_kernel)(blockDim, stream,
                dInfo, dSA, dSBi, dSC, dSD, dDbg, dOut);
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = Clock::now();
        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        times_us.push_back(total_us / launches_per_sync);
    }
    std::sort(times_us.begin(), times_us.end());
    double median_us = times_us[times_us.size() / 2];
    double p10_us    = times_us[times_us.size() / 10];
    double p90_us    = times_us[times_us.size() * 9 / 10];

    double per_cb_us = median_us / airan::LDPC_C_NUM;
    double gbps      = (double)airan::LDPC_C_NUM * airan::LDPC_N_RAW / (median_us * 1e-6) / 1e9;

    printf("\n=== Results for N_CB=%u ===\n", airan::LDPC_C_NUM);
    printf("  kernel time:   median=%.2f us  p10=%.2f  p90=%.2f\n", median_us, p10_us, p90_us);
    printf("  per-CB time:   %.3f us\n", per_cb_us);
    printf("  throughput:    %.3f Gbit/s\n", gbps);


    if (airan::LDPC_C_NUM >= 4) {
        double img_us = per_cb_us * 143.0;
        printf("\n  [ImageNet 224x224x3 estimate]\n");
        printf("    143 CBs × %.2f us/CB = %.2f us = %.3f ms per image\n",
               per_cb_us, img_us, img_us / 1000.0);
        printf("    images/sec: %.0f\n", 1e6 / img_us);
    }

    WriteFile(pOut, hOut, szOut);

    aclrtFree(dInfo); aclrtFree(dOut);
    aclrtFree(dSA); aclrtFree(dSBi); aclrtFree(dSC); aclrtFree(dSD);
    aclrtFree(dDbg);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
#endif

    std::free(hInfo); std::free(hOut); std::free(hGolden);
    std::free(hInfo4); std::free(hGold4);
    std::free(hSA); std::free(hSBi); std::free(hSC); std::free(hSD);
    std::free(hDbg);
    return 0;
}