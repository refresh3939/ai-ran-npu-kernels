












#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "pss_correlator.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_pss_correlator_kernel.h"
#else
#include "tikicpulib.h"
extern "C" void pss_correlator_kernel(uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                       uint8_t *, uint8_t *, uint8_t *);
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

namespace {

using namespace airan_pss;

constexpr int N_WARMUP = 5;
constexpr int N_TIMED  = 20;



constexpr size_t S23_TOTAL_TILES   = 300;
constexpr size_t S23_PER_TILE_FP32 = 16;
constexpr size_t S23_METRIC_TOTAL  = (size_t)N_G * 4 * S23_TOTAL_TILES * S23_PER_TILE_FP32;
constexpr size_t S23_METRIC_BYTES  = S23_METRIC_TOTAL * sizeof(float);
constexpr size_t STAGE1_BYTES      = 6ull * N_SEARCH * sizeof(int16_t);
constexpr size_t SCRATCH_BYTES     = STAGE1_BYTES + S23_METRIC_BYTES;

std::string DataDir()
{
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}

std::string GoldenRoot()  { return DataDir() + "/data/golden/rx/pss_correlator"; }
std::string GInput() {
    const char *env = std::getenv("PSS_CASE_DIR");
    if (env != nullptr && env[0] != '\0') {
        return GoldenRoot() + "/" + env + "/input.bin";
    }
    return GoldenRoot() + "/case_0_pss0_cfop0_t12345_snr30/input.bin";
}
std::string GPssRef()     { return GoldenRoot() + "/pss_ref.bin"; }
std::string GTwiddle()    { return GoldenRoot() + "/twiddle.bin"; }
std::string GOutput()     { return DataDir() + "/data/ascend_output/output.bin"; }
std::string GStage1Out()  { return DataDir() + "/data/ascend_output/stage1_yG.bin"; }
std::string GStage23Out() { return DataDir() + "/data/ascend_output/stage23_metric.bin"; }

#ifndef ASCENDC_CPU_DEBUG
void LoadInputBin(const std::string &path, size_t bytes,
                  uint8_t **hostOut, uint8_t **devOut)
{
    CHECK_ACL(aclrtMallocHost((void **)hostOut, bytes));
    CHECK_ACL(aclrtMalloc((void **)devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    size_t fsz = 0;
    if (!ReadFile(path.c_str(), fsz, *hostOut, bytes)) {
        ERROR_LOG("Failed to read %s (expected %zu bytes)", path.c_str(), bytes);
        std::exit(1);
    }
    if (fsz != bytes) {
        ERROR_LOG("Size mismatch %s: read %zu, expected %zu", path.c_str(), fsz, bytes);
        std::exit(1);
    }
    CHECK_ACL(aclrtMemcpy(*devOut, bytes, *hostOut, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}
#endif

}


int32_t main(int32_t  , char *  [])
{
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);


    const size_t tilingBytes = sizeof(TCubeTiling);
    const size_t wsBytes     = static_cast<size_t>(plat->GetLibApiWorkSpaceSize());

    uint8_t *tilingBuf = (uint8_t *)malloc(tilingBytes);
    if (!tilingBuf) { ERROR_LOG("malloc tilingBuf fail"); return 1; }
    std::memset(tilingBuf, 0, tilingBytes);
    GenerateTiling(socVersion, tilingBuf);

    const uint32_t blockDim = BLOCK_DIM_SKELETON;

    printf("[pss_correlator] SOC=%s blockDim=%u\n", socVersion, blockDim);
    printf("[pss_correlator] N_SEARCH=%u N_MF=%u M_OUT=%u N_G=%u N_PSS=%u\n",
           N_SEARCH, N_MF, M_OUT, N_G, N_PSS);
    printf("[pss_correlator] input=%zu KB  pss_ref=%zu B  twiddle=%zu KB  output=%zu B\n",
           INPUT_BYTES / 1024, PSS_REF_BYTES,
           TWIDDLE_BYTES / 1024, (size_t)OUTPUT_BYTES);

#ifdef ASCENDC_CPU_DEBUG

    uint8_t *inp = (uint8_t *)AscendC::GmAlloc(INPUT_BYTES);
    uint8_t *prf = (uint8_t *)AscendC::GmAlloc(PSS_REF_BYTES);
    uint8_t *twd = (uint8_t *)AscendC::GmAlloc(TWIDDLE_BYTES);
    uint8_t *scr = (uint8_t *)AscendC::GmAlloc(SCRATCH_BYTES);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(OUTPUT_BYTES);
    uint8_t *ws  = (uint8_t *)AscendC::GmAlloc(wsBytes ? wsBytes : 1024);
    uint8_t *til = (uint8_t *)AscendC::GmAlloc(tilingBytes);

    size_t fsz = 0;
    ReadFile(GInput().c_str(),   fsz, inp, INPUT_BYTES);
    ReadFile(GPssRef().c_str(),  fsz, prf, PSS_REF_BYTES);
    ReadFile(GTwiddle().c_str(), fsz, twd, TWIDDLE_BYTES);
    std::memcpy(til, tilingBuf, tilingBytes);

    ICPU_RUN_KF(pss_correlator_kernel, blockDim,
                inp, prf, twd, scr, out, ws, til);

    WriteFile(GOutput().c_str(), out, OUTPUT_BYTES);

    AscendC::GmFree(inp); AscendC::GmFree(prf); AscendC::GmFree(twd);
    AscendC::GmFree(scr); AscendC::GmFree(out); AscendC::GmFree(ws);
    AscendC::GmFree(til);
#else

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inpH = nullptr, *inpD = nullptr;
    uint8_t *prfH = nullptr, *prfD = nullptr;
    uint8_t *twdH = nullptr, *twdD = nullptr;
    uint8_t *outH = nullptr, *outD = nullptr;
    uint8_t *scrD = nullptr;
    uint8_t *wsD  = nullptr;
    uint8_t *tilH = nullptr, *tilD = nullptr;

    LoadInputBin(GInput(),   INPUT_BYTES,   &inpH, &inpD);
    LoadInputBin(GPssRef(),  PSS_REF_BYTES, &prfH, &prfD);
    LoadInputBin(GTwiddle(), TWIDDLE_BYTES, &twdH, &twdD);


    CHECK_ACL(aclrtMallocHost((void **)&outH, OUTPUT_BYTES));
    CHECK_ACL(aclrtMalloc((void **)&outD, OUTPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));

    std::memset(outH, 0xAA, OUTPUT_BYTES);
    CHECK_ACL(aclrtMemcpy(outD, OUTPUT_BYTES, outH, OUTPUT_BYTES, ACL_MEMCPY_HOST_TO_DEVICE));


    CHECK_ACL(aclrtMalloc((void **)&scrD, SCRATCH_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(scrD, SCRATCH_BYTES, 0, SCRATCH_BYTES));


    const size_t wsBytesAlloc = wsBytes ? wsBytes : 1024;
    CHECK_ACL(aclrtMalloc((void **)&wsD, wsBytesAlloc, ACL_MEM_MALLOC_HUGE_FIRST));


    CHECK_ACL(aclrtMallocHost((void **)&tilH, tilingBytes));
    CHECK_ACL(aclrtMalloc((void **)&tilD, tilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::memcpy(tilH, tilingBuf, tilingBytes);
    CHECK_ACL(aclrtMemcpy(tilD, tilingBytes, tilH, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));


    printf("[pss_correlator] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        ACLRT_LAUNCH_KERNEL(pss_correlator_kernel)
        (blockDim, stream,
         inpD, prfD, twdD, scrD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }


    printf("[pss_correlator] timed:   %d runs\n", N_TIMED);
    std::vector<double> us_list;
    us_list.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(pss_correlator_kernel)
        (blockDim, stream,
         inpD, prfD, twdD, scrD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        us_list.push_back(us);
    }

    double sum_us = 0.0;
    double min_us = us_list[0], max_us = us_list[0];
    for (double v : us_list) {
        sum_us += v;
        if (v < min_us) min_us = v;
        if (v > max_us) max_us = v;
    }
    double avg_us = sum_us / N_TIMED;
    std::vector<double> sorted = us_list;
    std::sort(sorted.begin(), sorted.end());
    double p50 = sorted[N_TIMED / 2];
    double p99 = sorted[(N_TIMED * 99) / 100];

    printf("[pss_correlator] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           min_us, avg_us, p50, p99, max_us);


    CHECK_ACL(aclrtMemcpy(outH, OUTPUT_BYTES, outD, OUTPUT_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));



    uint16_t *out16 = reinterpret_cast<uint16_t *>(outH);
    bool kernel_wrote = (out16[15] == 0x4700);
    printf("[pss_correlator] sentinel out[15] = 0x%04X  (expect 0x4700 = half 7.0)  %s\n",
           out16[15], kernel_wrote ? "✓ KERNEL_RAN" : "✗ KERNEL_NOT_WRITTEN");


    printf("[pss_correlator] output dump (16 × uint16):");
    for (int i = 0; i < 16; ++i) printf(" %04X", out16[i]);
    printf("\n");

    WriteFile(GOutput().c_str(), outH, OUTPUT_BYTES);


    uint8_t *scrH = nullptr;
    CHECK_ACL(aclrtMallocHost((void **)&scrH, SCRATCH_BYTES));
    CHECK_ACL(aclrtMemcpy(scrH, SCRATCH_BYTES, scrD, SCRATCH_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));

    WriteFile(GStage1Out().c_str(),  scrH,                STAGE1_BYTES);
    WriteFile(GStage23Out().c_str(), scrH + STAGE1_BYTES, S23_METRIC_BYTES);

    uint16_t *scr16 = reinterpret_cast<uint16_t *>(scrH);
    const size_t G0_RE_OFFSET = 2u * N_SEARCH;
    printf("[pss_correlator] stage1 yG[G=0][re][0..3] (uint16): %04X %04X %04X %04X\n",
           scr16[G0_RE_OFFSET + 0], scr16[G0_RE_OFFSET + 1],
           scr16[G0_RE_OFFSET + 2], scr16[G0_RE_OFFSET + 3]);

    float *metric_fp32 = reinterpret_cast<float *>(scrH + STAGE1_BYTES);


    const size_t g_off = 1 * (4 * S23_TOTAL_TILES * S23_PER_TILE_FP32);
    for (int l = 0; l < 3; ++l) {
        float *base = metric_fp32 + g_off + 0 + l * 4;
        printf("[pss_correlator] reduce[g=1, aiv=0, tile=0, l=%d]: peak=%.4e idx=%.0f sum=%.4e count=%.0f\n",
               l, base[0], base[1], base[2], base[3]);
    }


    CHECK_ACL(aclrtFreeHost(scrH));


    CHECK_ACL(aclrtFree(inpD));   CHECK_ACL(aclrtFreeHost(inpH));
    CHECK_ACL(aclrtFree(prfD));   CHECK_ACL(aclrtFreeHost(prfH));
    CHECK_ACL(aclrtFree(twdD));   CHECK_ACL(aclrtFreeHost(twdH));
    CHECK_ACL(aclrtFree(outD));   CHECK_ACL(aclrtFreeHost(outH));
    CHECK_ACL(aclrtFree(scrD));
    CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtFree(tilD));   CHECK_ACL(aclrtFreeHost(tilH));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif

    free(tilingBuf);
    return 0;
}