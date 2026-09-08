



















#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "data_utils.h"
#include "sss_correlator.h"
#include "sss_tiling.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_sss_correlator_kernel.h"
#endif

extern "C" void GenerateSssTiling(const char *socVersion,
                                  uint8_t *buf,
                                  int32_t mu_t_pss,
                                  int32_t n_id_2,
                                  int32_t g_hat);

namespace {

constexpr int N_WARMUP = 5;
constexpr int N_TIMED  = 20;

std::string DataDir()
{
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}

std::string CaseDir()
{
    const char *env = std::getenv("SSS_CASE_DIR");
    std::string cd = env ? std::string(env) : std::string("case_0_clean_min_n_id_1");
    return DataDir() + "/data/golden/rx/sss_correlator/" + cd;
}

std::string GoldenRoot()
{
    return DataDir() + "/data/golden/rx/sss_correlator";
}

std::string OutputDir()
{
    return DataDir() + "/data/ascend_output";
}


int64_t ParseJsonInt(const std::string &path, const std::string &field, int64_t defv)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        ERROR_LOG("failed to open %s", path.c_str());
        return defv;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    std::string key = "\"" + field + "\"";
    size_t pos = s.find(key);
    if (pos == std::string::npos) return defv;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return defv;
    pos++;
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
    int64_t v = std::strtoll(s.c_str() + pos, nullptr, 10);
    return v;
}

#ifndef ASCENDC_CPU_DEBUG




void LoadCInt16TableSplitT(const std::string &path,
                           size_t n_pairs,
                           size_t outer, size_t N, size_t K,
                           uint8_t **devRe,
                           uint8_t **devIm,
                           bool divQ,
                           bool transpose_nk)
{
    size_t bytes_pairs = n_pairs * 2u * sizeof(int16_t);
    size_t bytes_split = n_pairs * sizeof(uint16_t);


    if (outer * N * K != n_pairs) {
        ERROR_LOG("LoadCInt16TableSplitT: outer*N*K (%zu*%zu*%zu=%zu) != n_pairs (%zu)",
                  outer, N, K, outer*N*K, n_pairs);
        return;
    }

    std::vector<int16_t> hostPairs(n_pairs * 2u);
    size_t got = 0;
    if (!ReadFile(path, got, hostPairs.data(), bytes_pairs)) {
        ERROR_LOG("failed to read %s", path.c_str());
        return;
    }
    (void)got;

    std::vector<aclFloat16> hostRe(n_pairs), hostIm(n_pairs);
    float scale = divQ ? (1.0f / static_cast<float>(Q_SCALE)) : 1.0f;

    if (!transpose_nk) {

        for (size_t i = 0; i < n_pairs; ++i) {
            float re_v = static_cast<float>(hostPairs[i * 2u + 0u]) * scale;
            float im_v = static_cast<float>(hostPairs[i * 2u + 1u]) * scale;
            hostRe[i] = aclFloatToFloat16(re_v);
            hostIm[i] = aclFloatToFloat16(im_v);
        }
    } else {

        for (size_t o = 0; o < outer; ++o) {
            for (size_t n = 0; n < N; ++n) {
                for (size_t k = 0; k < K; ++k) {
                    size_t src_idx = o * N * K + n * K + k;
                    size_t dst_idx = o * K * N + k * N + n;
                    float re_v = static_cast<float>(hostPairs[src_idx * 2u + 0u]) * scale;
                    float im_v = static_cast<float>(hostPairs[src_idx * 2u + 1u]) * scale;
                    hostRe[dst_idx] = aclFloatToFloat16(re_v);
                    hostIm[dst_idx] = aclFloatToFloat16(im_v);
                }
            }
        }
    }

    uint8_t *devReP, *devImP;
    CHECK_ACL(aclrtMalloc((void **)&devReP, bytes_split, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&devImP, bytes_split, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(devReP, bytes_split, hostRe.data(), bytes_split, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devImP, bytes_split, hostIm.data(), bytes_split, ACL_MEMCPY_HOST_TO_DEVICE));
    *devRe = devReP;
    *devIm = devImP;
}

void LoadCInt16TableSplit(const std::string &path,
                          size_t n_pairs,
                          uint8_t **devRe,
                          uint8_t **devIm,
                          bool divQ)
{
    size_t bytes_pairs = n_pairs * 2u * sizeof(int16_t);
    size_t bytes_split = n_pairs * sizeof(uint16_t);

    std::vector<int16_t> hostPairs(n_pairs * 2u);
    size_t got = 0;
    if (!ReadFile(path, got, hostPairs.data(), bytes_pairs)) {
        ERROR_LOG("failed to read %s", path.c_str());
        return;
    }
    (void)got;

    std::vector<aclFloat16> hostRe(n_pairs), hostIm(n_pairs);
    float scale = divQ ? (1.0f / static_cast<float>(Q_SCALE)) : 1.0f;
    for (size_t i = 0; i < n_pairs; ++i) {
        float re_v = static_cast<float>(hostPairs[i * 2u + 0u]) * scale;
        float im_v = static_cast<float>(hostPairs[i * 2u + 1u]) * scale;
        hostRe[i] = aclFloatToFloat16(re_v);
        hostIm[i] = aclFloatToFloat16(im_v);
    }

    uint8_t *devReP, *devImP;
    CHECK_ACL(aclrtMalloc((void **)&devReP, bytes_split, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&devImP, bytes_split, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(devReP, bytes_split, hostRe.data(), bytes_split, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devImP, bytes_split, hostIm.data(), bytes_split, ACL_MEMCPY_HOST_TO_DEVICE));
    *devRe = devReP;
    *devIm = devImP;
}


void LoadInt16Device(const std::string &path, size_t bytes, uint8_t **devOut)
{
    std::vector<int16_t> host(bytes / sizeof(int16_t));
    size_t got = 0;
    if (!ReadFile(path, got, host.data(), bytes)) {
        ERROR_LOG("failed to read %s", path.c_str());
        return;
    }
    (void)got;

    uint8_t *devPtr;
    CHECK_ACL(aclrtMalloc((void **)&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(devPtr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    *devOut = devPtr;
}
#endif

}


int32_t main(int32_t  , char *  [])
{
    const char *socVersion = SOC_VERSION;

    std::string caseDir = CaseDir();
    std::string goldenRoot = GoldenRoot();
    std::string outputDir = OutputDir();
    (void)outputDir;

    printf("[sss_correlator] case_dir   = %s\n", caseDir.c_str());
    printf("[sss_correlator] golden_root= %s\n", goldenRoot.c_str());


    std::string truthPath = caseDir + "/truth.json";
    int64_t mu_t_pss = ParseJsonInt(truthPath, "mu_t_pss", -1);
    int64_t n_id_2   = ParseJsonInt(truthPath, "n_id_2",   -1);
    int64_t g_hat    = ParseJsonInt(truthPath, "g_hat",    -2);
    if (mu_t_pss < 0 || n_id_2 < 0 || g_hat < -1) {
        ERROR_LOG("failed to parse truth.json (mu_t_pss=%ld n_id_2=%ld g_hat=%ld)",
                  mu_t_pss, n_id_2, g_hat);
        return 1;
    }
    printf("[sss_correlator] truth: mu_t_pss=%ld n_id_2=%ld g_hat=%ld\n",
           mu_t_pss, n_id_2, g_hat);


    const size_t tilingBytes = sizeof(SssTilingV1);
    uint8_t *tilingBuf = (uint8_t *)malloc(tilingBytes);
    GenerateSssTiling(socVersion, tilingBuf,
                       static_cast<int32_t>(mu_t_pss),
                       static_cast<int32_t>(n_id_2),
                       static_cast<int32_t>(g_hat));

    uint32_t blockDim = 4u;


    const size_t inputBytes        = N_SEARCH * 2u * sizeof(int16_t);
    const size_t tableHalfElems    = N_ID_2_COUNT * N_ID_1_COUNT * N_FFT;
    const size_t tableSplitBytes   = tableHalfElems * sizeof(uint16_t);
    const size_t twidPairsTotal    = N_ID_2_COUNT * SEG_LEN;
    const size_t twidSplitBytes    = twidPairsTotal * sizeof(uint16_t);
    const size_t scratchBytes      = 4u * 2u * 8u * 1024u
                                   + 2u * 22u * 1024u
                                   + 2u * 1024u;
    const size_t outputBytes       = 32u;

    printf("[sss_correlator] SOC=%s blockDim=%u\n", socVersion, blockDim);
    printf("[sss_correlator] input=%zu KB  table_split=%zu KB×2  twid=%zu B×2  scratch=%zu KB  output=%zu B\n",
           inputBytes / 1024, tableSplitBytes / 1024, twidSplitBytes, scratchBytes / 1024, outputBytes);
    (void)tableSplitBytes; (void)twidSplitBytes; (void)outputBytes;

#ifdef ASCENDC_CPU_DEBUG
    (void)blockDim;
    ERROR_LOG("CPU debug mode not implemented for SSS Correlator yet");
    free(tilingBuf);
    return 1;
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));


    uint8_t *rxD;
    LoadInt16Device(caseDir + "/input.bin", inputBytes, &rxD);





    uint8_t *sssReD, *sssImD;
    LoadCInt16TableSplitT(goldenRoot + "/sss_time_table.bin",
                          tableHalfElems,
                           static_cast<size_t>(N_ID_2_COUNT),
                           static_cast<size_t>(N_ID_1_COUNT),
                           static_cast<size_t>(N_FFT),
                          &sssReD, &sssImD,
                           true,
                           true);


    uint8_t *twidReD, *twidImD;
    LoadCInt16TableSplit(goldenRoot + "/twiddle_g.bin",
                         twidPairsTotal, &twidReD, &twidImD,  true);


    uint8_t *scrD;
    CHECK_ACL(aclrtMalloc((void **)&scrD, scratchBytes, ACL_MEM_MALLOC_HUGE_FIRST));


    uint8_t *outH, *outD;
    CHECK_ACL(aclrtMallocHost((void **)&outH, outputBytes));
    CHECK_ACL(aclrtMalloc((void **)&outD, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    memset(outH, 0xAA, outputBytes);
    CHECK_ACL(aclrtMemcpy(outD, outputBytes, outH, outputBytes, ACL_MEMCPY_HOST_TO_DEVICE));


    uint8_t *wsD;
    const size_t wsBytes = 1024;
    CHECK_ACL(aclrtMalloc((void **)&wsD, wsBytes, ACL_MEM_MALLOC_HUGE_FIRST));


    uint8_t *tilH, *tilD;
    CHECK_ACL(aclrtMallocHost((void **)&tilH, tilingBytes));
    CHECK_ACL(aclrtMalloc((void **)&tilD, tilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    memcpy(tilH, tilingBuf, tilingBytes);
    CHECK_ACL(aclrtMemcpy(tilD, tilingBytes, tilH, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("[sss_correlator] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
        ACLRT_LAUNCH_KERNEL(sss_correlator_kernel)
            (blockDim, stream,
             rxD, sssReD, sssImD, twidReD, twidImD,
             scrD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    printf("[sss_correlator] timed:   %d runs\n", N_TIMED);
    std::vector<double> us_list;
    us_list.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(sss_correlator_kernel)
            (blockDim, stream,
             rxD, sssReD, sssImD, twidReD, twidImD,
             scrD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        us_list.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us_list.begin(), us_list.end());
    double min_us = us_list.front();
    double p50 = us_list[N_TIMED / 2];
    double p99 = us_list[(N_TIMED * 99) / 100];
    double max_us = us_list.back();
    double sum_us = 0.0;
    for (double v : us_list) sum_us += v;
    double avg_us = sum_us / N_TIMED;
    printf("[sss_correlator] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           min_us, avg_us, p50, p99, max_us);


    CHECK_ACL(aclrtMemcpy(outH, outputBytes, outD, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));


    float *outF32 = reinterpret_cast<float*>(outH);
    printf("[sss_correlator] output (8 fp32):\n");
    printf("  [0] n_id_1              = %.1f\n", outF32[0]);
    printf("  [1] pcid                = %.1f\n", outF32[1]);
    printf("  [2] peak                = %.3e\n", outF32[2]);
    printf("  [3] second              = %.3e\n", outF32[3]);
    printf("  [4] tau_star            = %.1f\n", outF32[4]);
    printf("  [5] DBG corr_re[t0,n0]  = %.3e\n", outF32[5]);
    printf("  [6] reserved            = %.3e\n", outF32[6]);
    printf("  [7] sentinel = %.1f  %s\n", outF32[7],
           (outF32[7] == 7.0f) ? "✓ KERNEL_RAN" : "✗ STALE!");


    std::string outPath = outputDir + "/output.bin";
    WriteFile(outPath.c_str(), outH, outputBytes);
    printf("[sss_correlator] wrote %s\n", outPath.c_str());


    CHECK_ACL(aclrtFree(rxD));
    CHECK_ACL(aclrtFree(sssReD)); CHECK_ACL(aclrtFree(sssImD));
    CHECK_ACL(aclrtFree(twidReD)); CHECK_ACL(aclrtFree(twidImD));
    CHECK_ACL(aclrtFree(scrD));
    CHECK_ACL(aclrtFree(outD)); CHECK_ACL(aclrtFreeHost(outH));
    CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtFree(tilD)); CHECK_ACL(aclrtFreeHost(tilH));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif

    free(tilingBuf);
    return 0;
}