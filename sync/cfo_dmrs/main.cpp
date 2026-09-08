












#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <string>
#include <vector>

#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "cfo_dmrs.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_cfo_dmrs_kernel.h"
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);


namespace {

constexpr int N_WARMUP = 10;
constexpr int N_TIMED  = 50;


constexpr size_t N_SC_PAD       = 1664;
constexpr size_t N_DMRS_PAD     = 896;
constexpr size_t N_SYMBOL_HOST  = 14;

constexpr size_t Y_PLANE_HALF   = N_SYMBOL_HOST * N_SC_PAD;
constexpr size_t X_PLANE_HALF   = 2 * N_DMRS_PAD;
constexpr size_t OUT_F_LEN      = 8;
constexpr size_t TILING_BYTES   = 128;
constexpr size_t SCR_BYTES      = 128;

struct TestCase {
    const char *dir_name;
    const char *description;
};
constexpr TestCase TEST_CASES[] = {
    {"case_0_no_cfo",          "δf=0, H=1, no noise (sanity)"},
    {"case_1_small_positive",  "δf=+200 Hz, H=1"},
    {"case_2_large_negative",  "δf=-1000 Hz, H=1 (near unambig)"},
    {"case_3_doppler_only",    "δf=0, freq+time_var (Doppler test)"},
    {"case_4_cfo_plus_noise",  "δf=+500 Hz + AWGN 10 dB"},
};
constexpr int N_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);


constexpr float DELTA_F_HZ_THRESH[N_CASES] = {
    1.0f,
    5.0f,
    5.0f,
    200.0f,
    50.0f,
};

std::string DataDir() {
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}
std::string CasePath(int i, const char *fname) {
    return DataDir() + "/data/golden/" + TEST_CASES[i].dir_name + "/" + fname;
}
std::string AscendOutputPath(int i) {
    return DataDir() + "/data/ascend_output/cfo_dmrs_" + TEST_CASES[i].dir_name + ".bin";
}

}


int32_t main(int32_t  , char *  [])
{
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    const size_t yBytes  = Y_PLANE_HALF * sizeof(uint16_t);
    const size_t xBytes  = X_PLANE_HALF * sizeof(uint16_t);
    const size_t outBytes = OUT_F_LEN * sizeof(float);
    const size_t wsBytes = (size_t)plat->GetLibApiWorkSpaceSize();

    uint8_t *tilingBuf = (uint8_t *)malloc(TILING_BYTES);
    GenerateTiling(socVersion, tilingBuf);

    uint32_t blockDim = airan::BLOCK_DIM;

    printf("[cfo_dmrs v2] SOC=%s blockDim=%u\n", socVersion, blockDim);
    printf("[cfo_dmrs v2] y=%zuB/plane (x2) x=%zuB/plane (x2) out=%zuB ws=%zuB\n",
           yBytes, xBytes, outBytes, wsBytes);

#ifdef ASCENDC_CPU_DEBUG
    fprintf(stderr, "[cfo_dmrs] CPU debug mode unsupported here\n");
    return 1;
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));


    uint8_t *yReH = nullptr, *yImH = nullptr, *xReH = nullptr, *xImH = nullptr, *outH = nullptr;
    uint8_t *yReD = nullptr, *yImD = nullptr, *xReD = nullptr, *xImD = nullptr, *outD = nullptr;
    uint8_t *scrD = nullptr, *wsD  = nullptr, *tilH = nullptr, *tilD = nullptr;

    CHECK_ACL(aclrtMallocHost((void **)&yReH, yBytes));
    CHECK_ACL(aclrtMallocHost((void **)&yImH, yBytes));
    CHECK_ACL(aclrtMallocHost((void **)&xReH, xBytes));
    CHECK_ACL(aclrtMallocHost((void **)&xImH, xBytes));
    CHECK_ACL(aclrtMallocHost((void **)&outH, outBytes));
    CHECK_ACL(aclrtMalloc((void **)&yReD, yBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yImD, yBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&xReD, xBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&xImD, xBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&outD, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&scrD, SCR_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsD,  wsBytes,   ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&tilH, TILING_BYTES));
    CHECK_ACL(aclrtMalloc((void **)&tilD, TILING_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    std::memcpy(tilH, tilingBuf, TILING_BYTES);
    CHECK_ACL(aclrtMemcpy(tilD, TILING_BYTES, tilH, TILING_BYTES, ACL_MEMCPY_HOST_TO_DEVICE));


    int n_pass = 0;
    for (int ci = 0; ci < N_CASES; ++ci) {
        size_t got = 0;
        bool ok_in = true;
        ok_in &= ReadFile(CasePath(ci, "y_re.bin").c_str(), got, yReH, yBytes) && got == yBytes;
        ok_in &= ReadFile(CasePath(ci, "y_im.bin").c_str(), got, yImH, yBytes) && got == yBytes;
        ok_in &= ReadFile(CasePath(ci, "x_re.bin").c_str(), got, xReH, xBytes) && got == xBytes;
        ok_in &= ReadFile(CasePath(ci, "x_im.bin").c_str(), got, xImH, xBytes) && got == xBytes;
        if (!ok_in) {
            ERROR_LOG("case %d: missing or wrong-size input bins", ci);
            continue;
        }
        CHECK_ACL(aclrtMemcpy(yReD, yBytes, yReH, yBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(yImD, yBytes, yImH, yBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(xReD, xBytes, xReH, xBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(xImD, xBytes, xImH, xBytes, ACL_MEMCPY_HOST_TO_DEVICE));


        std::memset(outH, 0xff, outBytes);
        CHECK_ACL(aclrtMemcpy(outD, outBytes, outH, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));


        ACLRT_LAUNCH_KERNEL(cfo_dmrs_kernel)
            (blockDim, stream, yReD, yImD, xReD, xImD, scrD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));

        CHECK_ACL(aclrtMemcpy(outH, outBytes, outD, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));


        WriteFile(AscendOutputPath(ci).c_str(), outH, outBytes);


        float truth_df = NAN;
        if (!ReadFile(CasePath(ci, "truth.bin").c_str(), got, &truth_df, sizeof(float))
            || got < sizeof(float)) {
            ERROR_LOG("case %d: missing truth.bin", ci);
            continue;
        }

        const float *out_f   = reinterpret_cast<const float *>(outH);
        float kernel_df = out_f[0];
        float c_re      = out_f[1];
        float c_im      = out_f[2];
        float sentinel  = out_f[7];

        float df_err = kernel_df - truth_df;
        bool sentinel_ok = (sentinel == 7.0f);
        bool err_ok = fabsf(df_err) < DELTA_F_HZ_THRESH[ci];
        bool ok = sentinel_ok && err_ok;

        printf("  case %d %-22s δf_kernel=%+8.2f truth=%+8.2f err=%+7.3f Hz "
               "thresh=%.1f c=(%+8.2f,%+8.2f) %s\n",
               ci, TEST_CASES[ci].dir_name,
               kernel_df, truth_df, df_err, DELTA_F_HZ_THRESH[ci],
               c_re, c_im,
               ok ? "[PASS]" : (sentinel_ok ? "[FAIL]" : "[NO-SENTINEL]"));

        if (ok) ++n_pass;
    }

    printf("[cfo_dmrs v2] %d / %d PASS\n", n_pass, N_CASES);


    {
        size_t got = 0;
        ReadFile(CasePath(2, "y_re.bin").c_str(), got, yReH, yBytes);
        ReadFile(CasePath(2, "y_im.bin").c_str(), got, yImH, yBytes);
        ReadFile(CasePath(2, "x_re.bin").c_str(), got, xReH, xBytes);
        ReadFile(CasePath(2, "x_im.bin").c_str(), got, xImH, xBytes);
        CHECK_ACL(aclrtMemcpy(yReD, yBytes, yReH, yBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(yImD, yBytes, yImH, yBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(xReD, xBytes, xReH, xBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(xImD, xBytes, xImH, xBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    printf("[cfo_dmrs v2] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        ACLRT_LAUNCH_KERNEL(cfo_dmrs_kernel)
            (blockDim, stream, yReD, yImD, xReD, xImD, scrD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }
    printf("[cfo_dmrs v2] timed:   %d runs\n", N_TIMED);
    std::vector<double> us_list;
    us_list.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(cfo_dmrs_kernel)
            (blockDim, stream, yReD, yImD, xReD, xImD, scrD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        us_list.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    double sum_us = 0, min_us = us_list[0], max_us = us_list[0];
    for (double v : us_list) { sum_us += v; if (v < min_us) min_us = v; if (v > max_us) max_us = v; }
    std::vector<double> sorted = us_list;
    std::sort(sorted.begin(), sorted.end());
    double p50 = sorted[N_TIMED / 2];
    double p99 = sorted[(N_TIMED * 99) / 100];
    printf("[cfo_dmrs v2] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           min_us, sum_us / N_TIMED, p50, p99, max_us);


    CHECK_ACL(aclrtFreeHost(yReH)); CHECK_ACL(aclrtFree(yReD));
    CHECK_ACL(aclrtFreeHost(yImH)); CHECK_ACL(aclrtFree(yImD));
    CHECK_ACL(aclrtFreeHost(xReH)); CHECK_ACL(aclrtFree(xReD));
    CHECK_ACL(aclrtFreeHost(xImH)); CHECK_ACL(aclrtFree(xImD));
    CHECK_ACL(aclrtFreeHost(outH)); CHECK_ACL(aclrtFree(outD));
    CHECK_ACL(aclrtFree(scrD));     CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtFreeHost(tilH)); CHECK_ACL(aclrtFree(tilD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif

    free(tilingBuf);
    return 0;
}