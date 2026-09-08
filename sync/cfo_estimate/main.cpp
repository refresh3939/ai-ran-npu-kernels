












#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cfo_estimate.h"
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_cfo_estimate_kernel.h"
#else
#include "tikicpulib.h"
extern "C" void cfo_estimate_kernel(uint8_t *, uint8_t *, uint8_t *, uint8_t *);
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

namespace {

constexpr int N_WARMUP = 10;
constexpr int N_TIMED  = 50;


struct TestCase {
    const char *dir_name;
    float       injected_cfo_hz;
};
constexpr TestCase TEST_CASES[] = {
    {"case_0_cfo_+0hz",      0.0f},
    {"case_1_cfo_+500hz",   +500.0f},
    {"case_2_cfo_-1200hz", -1200.0f},
    {"case_3_cfo_+5000hz", +5000.0f},
    {"case_4_cfo_-10000hz",-10000.0f},
};
constexpr int N_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);


std::string DataDir() {
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}
std::string CaseInputPath(int i) {
    return DataDir() + "/data/golden/" + TEST_CASES[i].dir_name + "/input.bin";
}
std::string AscendOutputPath(int i) {
    return DataDir() + "/data/ascend_output/" + TEST_CASES[i].dir_name + ".bin";
}


float TolHz(float truth) {
    float t = std::abs(truth) * 0.05f;
    return t > 50.0f ? t : 50.0f;
}

}


int32_t main(int32_t  , char *  [])
{
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    const size_t inputBytes   = static_cast<size_t>(airan::N_SAMPLE_PER_SLOT) * 2 * sizeof(int16_t);
    const size_t outputBytes  = airan::OUT_GM_BYTES;
    const size_t scratchBytes = airan::SCR_GM_BYTES;
    const size_t tilingBytes  = airan::TILING_TOTAL_SIZE;
    const size_t wsBytes      = static_cast<size_t>(plat->GetLibApiWorkSpaceSize());

    uint8_t *tilingBuf = (uint8_t *)malloc(tilingBytes);
    GenerateTiling(socVersion, tilingBuf);

    uint32_t blockDim = airan::N_BLOCKS;

    printf("[cfo_estimate] SOC=%s blockDim=%u\n", socVersion, blockDim);
    printf("[cfo_estimate] input %zu B, output %zu B, scratch %zu B, ws %zu B\n",
           inputBytes, outputBytes, scratchBytes, wsBytes);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *inp = (uint8_t *)AscendC::GmAlloc(inputBytes);
    uint8_t *scr = (uint8_t *)AscendC::GmAlloc(scratchBytes);
    uint8_t *ws  = (uint8_t *)AscendC::GmAlloc(wsBytes);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outputBytes);

    int n_pass = 0;
    for (int ci = 0; ci < N_CASES; ++ci) {
        size_t bytes_io = inputBytes;
        ReadFile(CaseInputPath(ci).c_str(), bytes_io, inp, inputBytes);
        memset(scr, 0, scratchBytes);
        memset(out, 0xAA, outputBytes);
        ICPU_RUN_KF(cfo_estimate_kernel, blockDim, inp, scr, ws, out);

        float R_re   = ((float *)out)[0];
        float R_im   = ((float *)out)[1];
        float est_hz = ((float *)out)[2];
        float truth  = TEST_CASES[ci].injected_cfo_hz;
        float err    = std::abs(est_hz - truth);
        float tol    = TolHz(truth);
        bool ok      = err < tol;
        printf("[cpu] %-22s truth=%+9.2f Hz  est=%+9.2f Hz  err=%7.4f Hz  R=(%+.3f, %+.3f)  %s\n",
               TEST_CASES[ci].dir_name, truth, est_hz, err, R_re, R_im, ok ? "PASS" : "FAIL");
        if (ok) n_pass++;

        WriteFile(AscendOutputPath(ci).c_str(), out, outputBytes);
    }
    printf("[cpu] result: %d / %d PASS\n", n_pass, N_CASES);

    AscendC::GmFree(inp);
    AscendC::GmFree(scr);
    AscendC::GmFree(ws);
    AscendC::GmFree(out);
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));


    uint8_t *inpH, *inpD;
    uint8_t *outH, *outD;
    uint8_t *scrD, *wsD;
    CHECK_ACL(aclrtMallocHost((void **)&inpH, inputBytes));
    CHECK_ACL(aclrtMalloc((void **)&inpD, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void **)&outH, outputBytes));
    CHECK_ACL(aclrtMalloc((void **)&outD, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&scrD, scratchBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsD,  wsBytes,      ACL_MEM_MALLOC_HUGE_FIRST));


    printf("\n[cfo_estimate] correctness pass (%d cases):\n", N_CASES);
    int n_pass = 0;
    std::vector<float> est_results(N_CASES);
    for (int ci = 0; ci < N_CASES; ++ci) {
        size_t bytes_io = inputBytes;
        ReadFile(CaseInputPath(ci).c_str(), bytes_io, inpH, inputBytes);
        CHECK_ACL(aclrtMemcpy(inpD, inputBytes, inpH, inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));

        memset(outH, 0xAA, outputBytes);
        CHECK_ACL(aclrtMemcpy(outD, outputBytes, outH, outputBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        ACLRT_LAUNCH_KERNEL(cfo_estimate_kernel)
        (blockDim, stream, inpD, scrD, wsD, outD);
        CHECK_ACL(aclrtSynchronizeStream(stream));

        CHECK_ACL(aclrtMemcpy(outH, outputBytes, outD, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        float R_re   = ((float *)outH)[0];
        float R_im   = ((float *)outH)[1];
        float est_hz = ((float *)outH)[2];
        float truth  = TEST_CASES[ci].injected_cfo_hz;
        float err    = std::abs(est_hz - truth);
        float tol    = TolHz(truth);
        bool ok      = err < tol;
        est_results[ci] = est_hz;
        printf("  %-22s truth=%+9.2f Hz  est=%+9.4f Hz  err=%7.4f Hz  R=(%+.3f, %+.3f)  %s\n",
               TEST_CASES[ci].dir_name, truth, est_hz, err, R_re, R_im, ok ? "PASS" : "FAIL");
        if (ok) n_pass++;

        WriteFile(AscendOutputPath(ci).c_str(), outH, outputBytes);
    }
    printf("[cfo_estimate] correctness: %d / %d PASS\n", n_pass, N_CASES);


    {
        size_t bytes_io = inputBytes;
        ReadFile(CaseInputPath(1).c_str(), bytes_io, inpH, inputBytes);
    }
    CHECK_ACL(aclrtMemcpy(inpD, inputBytes, inpH, inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("\n[cfo_estimate] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
        ACLRT_LAUNCH_KERNEL(cfo_estimate_kernel)
        (blockDim, stream, inpD, scrD, wsD, outD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    printf("[cfo_estimate] timed:   %d runs\n", N_TIMED);
    std::vector<double> us_list; us_list.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(cfo_estimate_kernel)
        (blockDim, stream, inpD, scrD, wsD, outD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        us_list.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    double sum = 0, mn = us_list[0], mx = us_list[0];
    for (double v : us_list) { sum += v; if (v < mn) mn = v; if (v > mx) mx = v; }
    double avg = sum / N_TIMED;
    std::vector<double> sorted = us_list; std::sort(sorted.begin(), sorted.end());
    double p50 = sorted[N_TIMED / 2];
    double p99 = sorted[(N_TIMED * 99) / 100];
    printf("[cfo_estimate] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           mn, avg, p50, p99, mx);
    printf("[cfo_estimate] real-time headroom (vs 1ms slot): %.1fx\n", 1000.0 / avg);

    CHECK_ACL(aclrtFree(inpD));   CHECK_ACL(aclrtFreeHost(inpH));
    CHECK_ACL(aclrtFree(outD));   CHECK_ACL(aclrtFreeHost(outH));
    CHECK_ACL(aclrtFree(scrD));
    CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif

    free(tilingBuf);
    return 0;
}