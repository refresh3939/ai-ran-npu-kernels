













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
#include "pss_cfo_estimator.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_pss_cfo_estimator_kernel.h"
#else
#include "tikicpulib.h"
extern "C" void pss_cfo_estimator_kernel(uint8_t *, uint8_t *, uint8_t *,
                                          uint8_t *, uint8_t *, uint8_t *);
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);


namespace {

constexpr int N_WARMUP             = 10;
constexpr int N_TIMED              = 50;
constexpr int LAUNCHES_PER_SYNC    = 10;


struct TestCase {
    const char *dir_name;
    const char *description;
};
constexpr TestCase TEST_CASES[] = {
    {"case_0_no_cfo",         "Δf=0, clean PSS (sanity)"},
    {"case_1_small_positive", "Δf=+500 Hz, clean"},
    {"case_2_mid_positive",   "Δf=+5000 Hz, clean (benchmarked)"},
    {"case_3_negative",       "Δf=-10000 Hz, clean"},
    {"case_4_cfo_plus_noise", "Δf=+5000 Hz + AWGN 10 dB"},
};
constexpr int N_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);









constexpr float HZ_THRESHOLD[N_CASES] = {
    50.0f,
    50.0f,
    100.0f,
    100.0f,
    300.0f,
};

std::string DataDir() {
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}
std::string CaseInputPath(int i) {
    return DataDir() + "/data/golden/" + TEST_CASES[i].dir_name + "/input.bin";
}
std::string CasePilotPath(int i) {
    return DataDir() + "/data/golden/" + TEST_CASES[i].dir_name + "/pilot.bin";
}
std::string CaseTruthPath(int i) {
    return DataDir() + "/data/golden/" + TEST_CASES[i].dir_name + "/truth.bin";
}
std::string AscendOutputPath(int i) {
    return DataDir() + "/data/ascend_output/pss_cfo_" + TEST_CASES[i].dir_name + ".bin";
}






constexpr int TRUTH_FLOAT_COUNT = 2;

bool VerifyCase(int case_idx, const uint8_t *out_buf, size_t out_bytes)
{
    const size_t kernel_min_bytes = 4 * sizeof(float);
    if (out_bytes < kernel_min_bytes) {
        ERROR_LOG("output buffer too small (%zu < %zu)", out_bytes, kernel_min_bytes);
        return false;
    }


    const size_t truth_bytes = TRUTH_FLOAT_COUNT * sizeof(float);
    std::vector<uint8_t> truth(truth_bytes);
    size_t got = 0;
    if (!ReadFile(CaseTruthPath(case_idx).c_str(), got, truth.data(), truth_bytes)
        || got != truth_bytes) {
        ERROR_LOG("failed to read %s (got %zu / %zu)",
                  CaseTruthPath(case_idx).c_str(), got, truth_bytes);
        return false;
    }

    const float *out_f   = reinterpret_cast<const float *>(out_buf);
    const float *truth_f = reinterpret_cast<const float *>(truth.data());

    float df_est       = out_f[0];
    float theta_est    = out_f[1];
    float mag2_c0      = out_f[2];
    float mag2_c1      = out_f[3];
    float df_true      = truth_f[0];
    float theta_true   = truth_f[1];

    float df_err    = fabsf(df_est - df_true);
    float theta_err = fabsf(theta_est - theta_true);

    bool ok = df_err < HZ_THRESHOLD[case_idx];
    printf("  case %d %-24s Δf_est=%+9.2f Hz (truth %+9.2f, err %6.2f, thresh %.0f)  "
           "θ_err=%.4f rad  |C_0|²=%7.2f |C_1|²=%7.2f  %s\n",
           case_idx, TEST_CASES[case_idx].dir_name,
           df_est, df_true, df_err, HZ_THRESHOLD[case_idx],
           theta_err, mag2_c0, mag2_c1,
           ok ? "[PASS]" : "[FAIL]");
    return ok;
}

}


int32_t main(int32_t  , char *  [])
{
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    const size_t inputBytes  = airan::INPUT_GM_INT16_LEN * sizeof(int16_t);
    const size_t pilotBytes  = airan::PILOT_GM_INT16_LEN * sizeof(int16_t);
    const size_t outputBytes = airan::OUT_GM_BYTES;
    const size_t scrBytes    = airan::SCR_GM_BYTES;
    const size_t tilingBytes = airan::TILING_TOTAL_SIZE;
    const size_t wsBytes     = (size_t)plat->GetLibApiWorkSpaceSize();

    uint8_t *tilingBuf = (uint8_t *)malloc(tilingBytes);
    GenerateTiling(socVersion, tilingBuf);

    uint32_t blockDim = airan::BLOCK_DIM;

    printf("[pss_cfo_estimator] SOC=%s blockDim=%u\n", socVersion, blockDim);
    printf("[pss_cfo_estimator] inputBytes=%zu pilotBytes=%zu outBytes=%zu "
           "tilingBytes=%zu wsBytes=%zu\n",
           inputBytes, pilotBytes, outputBytes, tilingBytes, wsBytes);

#ifdef ASCENDC_CPU_DEBUG
    fprintf(stderr, "[pss_cfo_estimator] CPU debug mode unsupported in this minimal main\n");
    return 1;
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *yH = nullptr, *xH = nullptr, *outH = nullptr;
    uint8_t *yD = nullptr, *xD = nullptr, *outD = nullptr;
    uint8_t *scrD = nullptr, *wsD = nullptr;
    uint8_t *tilH = nullptr, *tilD = nullptr;

    CHECK_ACL(aclrtMallocHost((void **)&yH,   inputBytes));
    CHECK_ACL(aclrtMallocHost((void **)&xH,   pilotBytes));
    CHECK_ACL(aclrtMallocHost((void **)&outH, outputBytes));
    CHECK_ACL(aclrtMalloc((void **)&yD,   inputBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&xD,   pilotBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&outD, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&scrD, scrBytes,    ACL_MEM_MALLOC_HUGE_FIRST));
    {
        constexpr int Ntw=256; std::vector<uint16_t> tw(2*Ntw);
        for(int n=0;n<Ntw;++n){ tw[n]=0x3C00; tw[Ntw+n]=0x0000; }
        uint8_t* twH=nullptr; CHECK_ACL(aclrtMallocHost((void**)&twH,2*Ntw*2));
        std::memcpy(twH,tw.data(),2*Ntw*2);
        CHECK_ACL(aclrtMemcpy(scrD,2*Ntw*2,twH,2*Ntw*2,ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtFreeHost(twH));
    }
    CHECK_ACL(aclrtMalloc((void **)&wsD,  wsBytes,     ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&tilH, tilingBytes));
    CHECK_ACL(aclrtMalloc((void **)&tilD, tilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::memcpy(tilH, tilingBuf, tilingBytes);
    CHECK_ACL(aclrtMemcpy(tilD, tilingBytes, tilH, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));


    int n_pass = 0;
    for (int ci = 0; ci < N_CASES; ++ci) {
        size_t got = 0;
        if (!ReadFile(CaseInputPath(ci).c_str(), got, yH, inputBytes)) {
            ERROR_LOG("missing %s", CaseInputPath(ci).c_str());
            continue;
        }
        if (!ReadFile(CasePilotPath(ci).c_str(), got, xH, pilotBytes)) {
            ERROR_LOG("missing %s", CasePilotPath(ci).c_str());
            continue;
        }
        CHECK_ACL(aclrtMemcpy(yD, inputBytes, yH, inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(xD, pilotBytes, xH, pilotBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        std::memset(outH, 0xaa, outputBytes);
        CHECK_ACL(aclrtMemcpy(outD, outputBytes, outH, outputBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        ACLRT_LAUNCH_KERNEL(pss_cfo_estimator_kernel)
            (blockDim, stream, yD, xD, scrD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));

        CHECK_ACL(aclrtMemcpy(outH, outputBytes, outD, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        WriteFile(AscendOutputPath(ci).c_str(), outH,
                  airan::OUT_FLOAT_LEN * sizeof(float));

        if (VerifyCase(ci, outH, outputBytes)) ++n_pass;
    }

    printf("[pss_cfo_estimator] %d / %d PASS\n", n_pass, N_CASES);


    {
        size_t got = 0;
        ReadFile(CaseInputPath(2).c_str(), got, yH, inputBytes);
        ReadFile(CasePilotPath(2).c_str(), got, xH, pilotBytes);
        CHECK_ACL(aclrtMemcpy(yD, inputBytes, yH, inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(xD, pilotBytes, xH, pilotBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    printf("[pss_cfo_estimator] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        ACLRT_LAUNCH_KERNEL(pss_cfo_estimator_kernel)
            (blockDim, stream, yD, xD, scrD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }


    printf("[pss_cfo_estimator] timed (single-launch):   %d runs\n", N_TIMED);
    std::vector<double> us_list;
    us_list.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(pss_cfo_estimator_kernel)
            (blockDim, stream, yD, xD, scrD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        us_list.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    {
        double sum_us = 0, min_us = us_list[0], max_us = us_list[0];
        for (double v : us_list) {
            sum_us += v;
            if (v < min_us) min_us = v;
            if (v > max_us) max_us = v;
        }
        std::vector<double> sorted = us_list;
        std::sort(sorted.begin(), sorted.end());
        double p50 = sorted[N_TIMED / 2];
        double p99 = sorted[(N_TIMED * 99) / 100];
        printf("[pss_cfo_estimator] latency single: min %.1f us, avg %.1f us, p50 %.1f us, "
               "p99 %.1f us, max %.1f us  (per launch, incl ACL sync)\n",
               min_us, sum_us / N_TIMED, p50, p99, max_us);
    }




    printf("[pss_cfo_estimator] timed (batched, %d launches/sync): %d samples\n",
           LAUNCHES_PER_SYNC, N_TIMED);
    std::vector<double> us_per_launch;
    us_per_launch.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < LAUNCHES_PER_SYNC; ++k) {
            ACLRT_LAUNCH_KERNEL(pss_cfo_estimator_kernel)
                (blockDim, stream, yD, xD, scrD, outD, wsD, tilD);
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        us_per_launch.push_back(elapsed_us / (double)LAUNCHES_PER_SYNC);
    }
    {
        double sum_us = 0, min_us = us_per_launch[0], max_us = us_per_launch[0];
        for (double v : us_per_launch) {
            sum_us += v;
            if (v < min_us) min_us = v;
            if (v > max_us) max_us = v;
        }
        std::vector<double> sorted = us_per_launch;
        std::sort(sorted.begin(), sorted.end());
        double p50 = sorted[N_TIMED / 2];
        double p99 = sorted[(N_TIMED * 99) / 100];
        printf("[pss_cfo_estimator] latency batch:  min %.1f us, avg %.1f us, p50 %.1f us, "
               "p99 %.1f us, max %.1f us  (per launch, amortized over %d)\n",
               min_us, sum_us / N_TIMED, p50, p99, max_us, LAUNCHES_PER_SYNC);
    }


    CHECK_ACL(aclrtFreeHost(yH));    CHECK_ACL(aclrtFree(yD));
    CHECK_ACL(aclrtFreeHost(xH));    CHECK_ACL(aclrtFree(xD));
    CHECK_ACL(aclrtFreeHost(outH));  CHECK_ACL(aclrtFree(outD));
    CHECK_ACL(aclrtFree(scrD));      CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtFreeHost(tilH));  CHECK_ACL(aclrtFree(tilD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif

    free(tilingBuf);
    return 0;
}
