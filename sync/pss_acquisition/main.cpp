








#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "pss_acquisition.h"
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_pss_acquisition_kernel.h"
#else
#include "tikicpulib.h"
extern "C" void pss_acquisition_kernel(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

namespace {

constexpr int N_WARMUP = 10;
constexpr int N_TIMED  = 50;



struct TestCase {
    const char *dir_name;
    float       injected_cfo_hz;
    int         injected_pss_id;
    int         injected_time_shift;
};
constexpr TestCase TEST_CASES[] = {
    {"case_0_pss0_cfo+0_t+0",        0.0f,    0,    0},
    {"case_1_pss0_cfo+500_t+0",     +500.0f,  0,    0},
    {"case_2_pss1_cfo+25000_t+50",  +25000.0f, 1,  +50},
    {"case_3_pss2_cfo-45000_t-80",  -45000.0f, 2,  -80},
    {"case_4_pss0_cfo+12500_t+130", +12500.0f, 0, +130},
};
constexpr int N_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);


std::string DataDir() {
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}
std::string CaseInputPath(int i) {
    return DataDir() + "/data/golden/rx/pss_acquisition/" + TEST_CASES[i].dir_name + "/input.bin";
}
std::string PssTemplatePath() {
    return DataDir() + "/data/golden/rx/pss_acquisition/pss_templates.bin";
}
std::string AscendOutputPath(int i) {
    return DataDir() + "/data/ascend_output/" + TEST_CASES[i].dir_name + ".bin";
}




struct PssResult {
    int     best_hyp;
    int     best_pss_id;
    int     best_t_idx;
    int     best_t_shift;
    float   best_freq_hz;
    float   peak_metric;
};

PssResult HostArgmax(const float *gm) {
    float peak = -1.0f;
    int best_b = 0, best_h_in_b = 0, best_p = 0, best_t = 0;
    for (uint32_t b = 0; b < airan::N_HYP_BATCHES; ++b) {
        uint32_t h_cnt = airan::HYP_BATCH;
        if (b * airan::HYP_BATCH + h_cnt > airan::N_FREQ_HYP)
            h_cnt = airan::N_FREQ_HYP - b * airan::HYP_BATCH;
        for (uint32_t t = 0; t < airan::N_TIME_OFFSETS; ++t) {
            for (uint32_t h_in_b = 0; h_in_b < h_cnt; ++h_in_b) {
                for (uint32_t p = 0; p < airan::N_PSS; ++p) {
                    uint32_t off = b * airan::OUT_FP32_PER_BATCH
                                 + t * airan::OUT_FP32_PER_BLOCK
                                 + h_in_b * airan::OUT_PSS_PAD + p;
                    float m = gm[off];
                    if (m > peak) {
                        peak = m;
                        best_b = b; best_h_in_b = h_in_b; best_p = p; best_t = t;
                    }
                }
            }
        }
    }
    PssResult r;
    r.best_hyp     = best_b * airan::HYP_BATCH + best_h_in_b;
    r.best_pss_id  = best_p;
    r.best_t_idx   = best_t;
    r.best_t_shift = best_t - (int)airan::TIME_SEARCH_HALF;
    r.best_freq_hz = airan::FREQ_HYP_MIN_HZ +
                     (float)r.best_hyp * airan::FREQ_HYP_STEP_HZ;
    r.peak_metric  = peak;
    return r;
}



constexpr float CFO_TOL_HZ        = 2500.0f;
constexpr int   TIME_TOL_SAMPLES  = 2;

}


int32_t main(int32_t  , char *  [])
{
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    const size_t inputBytes    = static_cast<size_t>(airan::N_SAMPLE_PER_SLOT) * 2 * sizeof(int16_t);
    const size_t pssTmplBytes  = airan::PSS_TMPL_GM_BYTES;
    const size_t outputBytes   = airan::OUT_GM_BYTES;
    const size_t scratchBytes  = airan::SCR_GM_BYTES;
    const size_t tilingBytes   = airan::TILING_TOTAL_SIZE;
    const size_t wsBytes       = static_cast<size_t>(plat->GetLibApiWorkSpaceSize());

    uint8_t *tilingBuf = (uint8_t *)malloc(tilingBytes);
    GenerateTiling(socVersion, tilingBuf);

    uint32_t blockDim = airan::N_BLOCKS;

    printf("[pss_acq] SOC=%s blockDim=%u\n", socVersion, blockDim);
    printf("[pss_acq] input %zu B, pss_tmpl %zu B, output %zu B, scratch %zu B, ws %zu B\n",
           inputBytes, pssTmplBytes, outputBytes, scratchBytes, wsBytes);
    printf("[pss_acq] search: %u hyp × %u pss × %u t = %u correlations\n",
           airan::N_FREQ_HYP, airan::N_PSS, airan::N_TIME_OFFSETS,
           airan::N_FREQ_HYP * airan::N_PSS * airan::N_TIME_OFFSETS);

#ifdef ASCENDC_CPU_DEBUG

    uint8_t *inp = (uint8_t *)AscendC::GmAlloc(inputBytes);
    uint8_t *tmpl = (uint8_t *)AscendC::GmAlloc(pssTmplBytes);
    uint8_t *scr = (uint8_t *)AscendC::GmAlloc(scratchBytes);
    uint8_t *ws  = (uint8_t *)AscendC::GmAlloc(wsBytes);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outputBytes);

    {
        size_t bytes_io = pssTmplBytes;
        if (!ReadFile(PssTemplatePath().c_str(), bytes_io, tmpl, pssTmplBytes)) {
            ERROR_LOG("Failed to read pss_templates.bin — 先跑 python3 scripts/gen_pss_templates.py");
            return 1;
        }
    }

    int n_pass = 0;
    for (int ci = 0; ci < N_CASES; ++ci) {
        size_t bytes_io = inputBytes;
        ReadFile(CaseInputPath(ci).c_str(), bytes_io, inp, inputBytes);
        memset(scr, 0, scratchBytes);
        memset(out, 0xAA, outputBytes);
        ICPU_RUN_KF(pss_acquisition_kernel, blockDim, inp, tmpl, scr, ws, out);

        PssResult r = HostArgmax((const float *)out);
        const TestCase &c = TEST_CASES[ci];
        bool cfo_ok  = std::abs(r.best_freq_hz - c.injected_cfo_hz) <= CFO_TOL_HZ;
        bool pss_ok  = r.best_pss_id == c.injected_pss_id;
        bool time_ok = std::abs(r.best_t_shift - c.injected_time_shift) <= TIME_TOL_SAMPLES;
        bool ok      = cfo_ok && pss_ok && time_ok;
        printf("[cpu] %-30s inj(cfo=%+8.0f,pss=%d,t=%+4d) est(cfo=%+8.0f,pss=%d,t=%+4d) "
               "peak=%.2e %s\n",
               c.dir_name, c.injected_cfo_hz, c.injected_pss_id, c.injected_time_shift,
               r.best_freq_hz, r.best_pss_id, r.best_t_shift,
               r.peak_metric, ok ? "PASS" : "FAIL");
        if (ok) n_pass++;

        WriteFile(AscendOutputPath(ci).c_str(), out, outputBytes);
    }
    printf("[cpu] result: %d / %d PASS\n", n_pass, N_CASES);

    AscendC::GmFree(inp);
    AscendC::GmFree(tmpl);
    AscendC::GmFree(scr);
    AscendC::GmFree(ws);
    AscendC::GmFree(out);
#else

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inpH, *inpD;
    uint8_t *tmplH, *tmplD;
    uint8_t *outH, *outD;
    uint8_t *scrD, *wsD;
    CHECK_ACL(aclrtMallocHost((void **)&inpH, inputBytes));
    CHECK_ACL(aclrtMalloc((void **)&inpD, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void **)&tmplH, pssTmplBytes));
    CHECK_ACL(aclrtMalloc((void **)&tmplD, pssTmplBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void **)&outH, outputBytes));
    CHECK_ACL(aclrtMalloc((void **)&outD, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&scrD, scratchBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsD,  wsBytes,      ACL_MEM_MALLOC_HUGE_FIRST));


    {
        size_t bytes_io = pssTmplBytes;
        if (!ReadFile(PssTemplatePath().c_str(), bytes_io, tmplH, pssTmplBytes)) {
            ERROR_LOG("Failed to read pss_templates.bin — 先跑 python3 scripts/gen_pss_templates.py");
            return 1;
        }
        CHECK_ACL(aclrtMemcpy(tmplD, pssTmplBytes, tmplH, pssTmplBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        printf("[pss_acq] loaded pss_templates.bin (%zu B)\n", pssTmplBytes);
    }


    printf("\n[pss_acq] correctness pass (%d cases):\n", N_CASES);
    int n_pass = 0;
    for (int ci = 0; ci < N_CASES; ++ci) {
        size_t bytes_io = inputBytes;
        ReadFile(CaseInputPath(ci).c_str(), bytes_io, inpH, inputBytes);
        CHECK_ACL(aclrtMemcpy(inpD, inputBytes, inpH, inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));

        memset(outH, 0xAA, outputBytes);
        CHECK_ACL(aclrtMemcpy(outD, outputBytes, outH, outputBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        ACLRT_LAUNCH_KERNEL(pss_acquisition_kernel)
        (blockDim, stream, inpD, tmplD, scrD, wsD, outD);
        CHECK_ACL(aclrtSynchronizeStream(stream));

        CHECK_ACL(aclrtMemcpy(outH, outputBytes, outD, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        PssResult r = HostArgmax((const float *)outH);
        const TestCase &c = TEST_CASES[ci];
        bool cfo_ok  = std::abs(r.best_freq_hz - c.injected_cfo_hz) <= CFO_TOL_HZ;
        bool pss_ok  = r.best_pss_id == c.injected_pss_id;
        bool time_ok = std::abs(r.best_t_shift - c.injected_time_shift) <= TIME_TOL_SAMPLES;
        bool ok      = cfo_ok && pss_ok && time_ok;
        printf("  %-30s inj(cfo=%+8.0f,pss=%d,t=%+4d) est(cfo=%+8.0f,pss=%d,t=%+4d) "
               "peak=%.2e %s%s%s%s\n",
               c.dir_name, c.injected_cfo_hz, c.injected_pss_id, c.injected_time_shift,
               r.best_freq_hz, r.best_pss_id, r.best_t_shift, r.peak_metric,
               ok ? "PASS" : "FAIL",
               cfo_ok ? "" : " [cfo]",
               pss_ok ? "" : " [pss]",
               time_ok ? "" : " [t]");
        if (ok) n_pass++;

        WriteFile(AscendOutputPath(ci).c_str(), outH, outputBytes);
    }
    printf("[pss_acq] correctness: %d / %d PASS\n", n_pass, N_CASES);


    {
        size_t bytes_io = inputBytes;
        ReadFile(CaseInputPath(0).c_str(), bytes_io, inpH, inputBytes);
    }
    CHECK_ACL(aclrtMemcpy(inpD, inputBytes, inpH, inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("\n[pss_acq] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
        ACLRT_LAUNCH_KERNEL(pss_acquisition_kernel)
        (blockDim, stream, inpD, tmplD, scrD, wsD, outD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    printf("[pss_acq] timed:   %d runs\n", N_TIMED);
    std::vector<double> us_list; us_list.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(pss_acquisition_kernel)
        (blockDim, stream, inpD, tmplD, scrD, wsD, outD);
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
    printf("[pss_acq] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           mn, avg, p50, p99, mx);
    printf("[pss_acq] real-time headroom (vs 1ms slot): %.2fx\n", 1000.0 / avg);

    CHECK_ACL(aclrtFree(inpD));    CHECK_ACL(aclrtFreeHost(inpH));
    CHECK_ACL(aclrtFree(tmplD));   CHECK_ACL(aclrtFreeHost(tmplH));
    CHECK_ACL(aclrtFree(outD));    CHECK_ACL(aclrtFreeHost(outH));
    CHECK_ACL(aclrtFree(scrD));
    CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif

    free(tilingBuf);
    return 0;
}
