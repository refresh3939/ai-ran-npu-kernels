


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
#include "ofdm_demod.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_ofdm_demod_kernel.h"
#else
#include "tikicpulib.h"
extern "C" void ofdm_demod_kernel(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                   uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                   uint8_t *, uint8_t *, uint8_t *);
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

using namespace ofdm_demod;

namespace {

constexpr int N_WARMUP = 10;
constexpr int N_TIMED  = 50;

std::string DataDir()
{

    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}
std::string GInput()  { return DataDir() + "/data/golden/input.bin"; }
std::string GWeight(const char *name) { return DataDir() + "/weights/" + name; }
std::string GOutRe()  { return DataDir() + "/data/ascend_output/output_re.bin"; }
std::string GOutIm()  { return DataDir() + "/data/ascend_output/output_im.bin"; }

#ifndef ASCENDC_CPU_DEBUG
void LoadInput(const std::string &path, size_t bytes, uint8_t **hostOut, uint8_t **devOut)
{
    CHECK_ACL(aclrtMallocHost((void **)hostOut, bytes));
    CHECK_ACL(aclrtMalloc((void **)devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile(path.c_str(), bytes, *hostOut, bytes);
    CHECK_ACL(aclrtMemcpy(*devOut, bytes, *hostOut, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}
#endif

}


int32_t main(int32_t  , char *  [])
{
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    const size_t inputBytes      = static_cast<size_t>(N_SAMPLE_PER_SLOT) * 2 * sizeof(int16_t);
    const size_t outputHalfBytes = static_cast<size_t>(N_SYMBOL) * N_FFT * sizeof(int16_t);
    const size_t w32Bytes        = static_cast<size_t>(P) * P * sizeof(int16_t);
    const size_t w64Bytes        = static_cast<size_t>(Q) * Q * sizeof(int16_t);
    const size_t twBytes         = static_cast<size_t>(P) * Q * sizeof(int16_t);


    const size_t scratchPerCoreBytes = 2 * BATCH_X_ELEMS * sizeof(int16_t);
    const size_t scratchBytes    = BLOCK_DIM * scratchPerCoreBytes;
    const size_t tilingBytes     = 2 * sizeof(TCubeTiling);
    const size_t wsBytes         = static_cast<size_t>(plat->GetLibApiWorkSpaceSize());

    uint8_t *tilingBuf = (uint8_t *)malloc(tilingBytes);
    GenerateTiling(socVersion, tilingBuf);

    uint32_t blockDim = BLOCK_DIM;

    printf("[ofdm_demod] SOC=%s blockDim=%u N_FFT=%u symbols=%u\n",
           socVersion, blockDim, N_FFT, N_SYMBOL);
    printf("[ofdm_demod] scratch: %zu KB per core\n", scratchPerCoreBytes / 1024);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *inp    = (uint8_t *)AscendC::GmAlloc(inputBytes);
    uint8_t *w32r   = (uint8_t *)AscendC::GmAlloc(w32Bytes);
    uint8_t *w32i   = (uint8_t *)AscendC::GmAlloc(w32Bytes);
    uint8_t *w64r   = (uint8_t *)AscendC::GmAlloc(w64Bytes);
    uint8_t *w64i   = (uint8_t *)AscendC::GmAlloc(w64Bytes);
    uint8_t *twr    = (uint8_t *)AscendC::GmAlloc(twBytes);
    uint8_t *twi    = (uint8_t *)AscendC::GmAlloc(twBytes);
    uint8_t *scr    = (uint8_t *)AscendC::GmAlloc(scratchBytes);
    uint8_t *outRe  = (uint8_t *)AscendC::GmAlloc(outputHalfBytes);
    uint8_t *outIm  = (uint8_t *)AscendC::GmAlloc(outputHalfBytes);
    uint8_t *ws     = (uint8_t *)AscendC::GmAlloc(wsBytes);
    uint8_t *til    = (uint8_t *)AscendC::GmAlloc(tilingBytes);

    ReadFile(GInput().c_str(),                     inputBytes, inp, inputBytes);
    ReadFile(GWeight("w_dft32_re.bin").c_str(),    w32Bytes,   w32r, w32Bytes);
    ReadFile(GWeight("w_dft32_im.bin").c_str(),    w32Bytes,   w32i, w32Bytes);
    ReadFile(GWeight("w_dft64_re_T.bin").c_str(),  w64Bytes,   w64r, w64Bytes);
    ReadFile(GWeight("w_dft64_im_T.bin").c_str(),  w64Bytes,   w64i, w64Bytes);
    ReadFile(GWeight("twiddle_pq_re.bin").c_str(), twBytes,    twr, twBytes);
    ReadFile(GWeight("twiddle_pq_im.bin").c_str(), twBytes,    twi, twBytes);
    memcpy(til, tilingBuf, tilingBytes);

    ICPU_RUN_KF(ofdm_demod_kernel, blockDim,
                inp, w32r, w32i, w64r, w64i, twr, twi, scr, outRe, outIm, ws, til);

    WriteFile(GOutRe().c_str(), outRe, outputHalfBytes);
    WriteFile(GOutIm().c_str(), outIm, outputHalfBytes);

    AscendC::GmFree(inp);  AscendC::GmFree(w32r); AscendC::GmFree(w32i);
    AscendC::GmFree(w64r); AscendC::GmFree(w64i); AscendC::GmFree(twr);
    AscendC::GmFree(twi);  AscendC::GmFree(scr);
    AscendC::GmFree(outRe); AscendC::GmFree(outIm);
    AscendC::GmFree(ws);   AscendC::GmFree(til);
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inpH, *inpD;
    uint8_t *w32rH, *w32rD, *w32iH, *w32iD;
    uint8_t *w64rH, *w64rD, *w64iH, *w64iD;
    uint8_t *twrH,  *twrD,  *twiH,  *twiD;
    uint8_t *outReH, *outReD, *outImH, *outImD;
    uint8_t *scrD,  *wsD;
    uint8_t *tilH,  *tilD;

    LoadInput(GInput(),                      inputBytes, &inpH,  &inpD);
    LoadInput(GWeight("w_dft32_re.bin"),     w32Bytes,   &w32rH, &w32rD);
    LoadInput(GWeight("w_dft32_im.bin"),     w32Bytes,   &w32iH, &w32iD);
    LoadInput(GWeight("w_dft64_re_T.bin"),   w64Bytes,   &w64rH, &w64rD);
    LoadInput(GWeight("w_dft64_im_T.bin"),   w64Bytes,   &w64iH, &w64iD);
    LoadInput(GWeight("twiddle_pq_re.bin"),  twBytes,    &twrH,  &twrD);
    LoadInput(GWeight("twiddle_pq_im.bin"),  twBytes,    &twiH,  &twiD);

    CHECK_ACL(aclrtMallocHost((void **)&outReH, outputHalfBytes));
    CHECK_ACL(aclrtMallocHost((void **)&outImH, outputHalfBytes));
    CHECK_ACL(aclrtMalloc((void **)&outReD, outputHalfBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&outImD, outputHalfBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    memset(outReH, 0, outputHalfBytes);
    memset(outImH, 0, outputHalfBytes);
    CHECK_ACL(aclrtMemcpy(outReD, outputHalfBytes, outReH, outputHalfBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(outImD, outputHalfBytes, outImH, outputHalfBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(aclrtMalloc((void **)&scrD, scratchBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsD,  wsBytes,      ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&tilH, tilingBytes));
    CHECK_ACL(aclrtMalloc((void **)&tilD, tilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    memcpy(tilH, tilingBuf, tilingBytes);
    CHECK_ACL(aclrtMemcpy(tilD, tilingBytes, tilH, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("[ofdm_demod] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
        ACLRT_LAUNCH_KERNEL(ofdm_demod_kernel)
        (blockDim, stream,
         inpD, w32rD, w32iD, w64rD, w64iD, twrD, twiD, scrD, outReD, outImD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    printf("[ofdm_demod] timed:   %d runs\n", N_TIMED);
    std::vector<double> us_list;
    us_list.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(ofdm_demod_kernel)
        (blockDim, stream,
         inpD, w32rD, w32iD, w64rD, w64iD, twrD, twiD, scrD, outReD, outImD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        us_list.push_back(us);
    }

    double sum_us = 0.0;
    double min_us = us_list[0], max_us = us_list[0];
    for (double v : us_list) { sum_us += v; if (v < min_us) min_us = v; if (v > max_us) max_us = v; }
    double avg_us = sum_us / N_TIMED;
    std::vector<double> sorted = us_list;
    std::sort(sorted.begin(), sorted.end());
    double p50 = sorted[N_TIMED / 2];
    double p99 = sorted[(N_TIMED * 99) / 100];

    printf("[ofdm_demod] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           min_us, avg_us, p50, p99, max_us);
    printf("[ofdm_demod] real-time headroom: %.1fx\n", SLOT_DURATION_US / avg_us);

    CHECK_ACL(aclrtMemcpy(outReH, outputHalfBytes, outReD, outputHalfBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(outImH, outputHalfBytes, outImD, outputHalfBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    WriteFile(GOutRe().c_str(), outReH, outputHalfBytes);
    WriteFile(GOutIm().c_str(), outImH, outputHalfBytes);

    CHECK_ACL(aclrtFree(inpD));   CHECK_ACL(aclrtFreeHost(inpH));
    CHECK_ACL(aclrtFree(w32rD));  CHECK_ACL(aclrtFreeHost(w32rH));
    CHECK_ACL(aclrtFree(w32iD));  CHECK_ACL(aclrtFreeHost(w32iH));
    CHECK_ACL(aclrtFree(w64rD));  CHECK_ACL(aclrtFreeHost(w64rH));
    CHECK_ACL(aclrtFree(w64iD));  CHECK_ACL(aclrtFreeHost(w64iH));
    CHECK_ACL(aclrtFree(twrD));   CHECK_ACL(aclrtFreeHost(twrH));
    CHECK_ACL(aclrtFree(twiD));   CHECK_ACL(aclrtFreeHost(twiH));
    CHECK_ACL(aclrtFree(outReD)); CHECK_ACL(aclrtFreeHost(outReH));
    CHECK_ACL(aclrtFree(outImD)); CHECK_ACL(aclrtFreeHost(outImH));
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
