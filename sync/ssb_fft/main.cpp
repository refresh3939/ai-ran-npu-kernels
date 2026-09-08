




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
#include "ssb_fft.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_ssb_fft_kernel.h"
#else
#include "tikicpulib.h"
extern "C" void ssb_fft_kernel(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                               uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                               uint8_t *, uint8_t *, uint8_t *, uint8_t *);
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

using namespace ssb_fft;

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
std::string GOutRe()  { return DataDir() + "/data/ascend_output/R_re.bin"; }
std::string GOutIm()  { return DataDir() + "/data/ascend_output/R_im.bin"; }

#ifndef ASCENDC_CPU_DEBUG
void LoadInput(const std::string &path, size_t allocBytes, size_t realBytes,
               uint8_t **hostOut, uint8_t **devOut)
{
    CHECK_ACL(aclrtMallocHost((void **)hostOut, allocBytes));
    CHECK_ACL(aclrtMalloc((void **)devOut, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    memset(*hostOut, 0, allocBytes);
    ReadFile(path.c_str(), realBytes, *hostOut, realBytes);
    CHECK_ACL(aclrtMemcpy(*devOut, allocBytes, *hostOut, allocBytes, ACL_MEMCPY_HOST_TO_DEVICE));
}
#endif

}


int32_t main(int32_t  , char *  [])
{
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);


    const size_t inputAlloc   = static_cast<size_t>(INPUT_GM_INT16_LEN) * sizeof(int16_t);
    const size_t inputReal    = static_cast<size_t>(N_SYMBOL) * SYM_STRIDE * 2 * sizeof(int16_t);
    const size_t w16Bytes     = static_cast<size_t>(P) * P * sizeof(int16_t);
    const size_t twBytes      = static_cast<size_t>(TILE_PQ) * sizeof(int16_t);
    const size_t gidxBytes    = static_cast<size_t>(N_DMRS_RE) * sizeof(int32_t);
    const size_t derotBytes   = static_cast<size_t>(N_DMRS_RE) * sizeof(int16_t);
    const size_t outBytes     = static_cast<size_t>(N_DMRS_RE) * sizeof(int16_t);
    const size_t scratchBytes = static_cast<size_t>(4) * BATCH_X_ELEMS * sizeof(int16_t);
    const size_t tilingBytes  = 2 * sizeof(TCubeTiling);
    const size_t wsBytes      = static_cast<size_t>(plat->GetLibApiWorkSpaceSize());

    uint8_t *tilingBuf = (uint8_t *)malloc(tilingBytes);
    GenerateTiling(socVersion, tilingBuf);

    uint32_t blockDim = BLOCK_DIM;
    printf("[ssb_fft] SOC=%s blockDim=%u N_FFT=%u symbols=%u DMRS=%u\n",
           socVersion, blockDim, N_FFT, N_SYMBOL, N_DMRS_RE);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *inp   = (uint8_t *)AscendC::GmAlloc(inputAlloc);
    uint8_t *w16r  = (uint8_t *)AscendC::GmAlloc(w16Bytes);
    uint8_t *w16i  = (uint8_t *)AscendC::GmAlloc(w16Bytes);
    uint8_t *twr   = (uint8_t *)AscendC::GmAlloc(twBytes);
    uint8_t *twi   = (uint8_t *)AscendC::GmAlloc(twBytes);
    uint8_t *gidx  = (uint8_t *)AscendC::GmAlloc(gidxBytes);
    uint8_t *dre   = (uint8_t *)AscendC::GmAlloc(derotBytes);
    uint8_t *dim   = (uint8_t *)AscendC::GmAlloc(derotBytes);
    uint8_t *scr   = (uint8_t *)AscendC::GmAlloc(scratchBytes);
    uint8_t *outRe = (uint8_t *)AscendC::GmAlloc(outBytes);
    uint8_t *outIm = (uint8_t *)AscendC::GmAlloc(outBytes);
    uint8_t *ws    = (uint8_t *)AscendC::GmAlloc(wsBytes);
    uint8_t *til   = (uint8_t *)AscendC::GmAlloc(tilingBytes);

    memset(inp, 0, inputAlloc);
    ReadFile(GInput().c_str(),                     inputReal, inp, inputReal);
    ReadFile(GWeight("w16_re.bin").c_str(),        w16Bytes,  w16r, w16Bytes);
    ReadFile(GWeight("w16_im.bin").c_str(),        w16Bytes,  w16i, w16Bytes);
    ReadFile(GWeight("twiddle_pq_re.bin").c_str(), twBytes,   twr, twBytes);
    ReadFile(GWeight("twiddle_pq_im.bin").c_str(), twBytes,   twi, twBytes);
    ReadFile(GWeight("gather_idx.bin").c_str(),    gidxBytes, gidx, gidxBytes);
    ReadFile(GWeight("derot_re.bin").c_str(),      derotBytes, dre, derotBytes);
    ReadFile(GWeight("derot_im.bin").c_str(),      derotBytes, dim, derotBytes);
    memcpy(til, tilingBuf, tilingBytes);

    ICPU_RUN_KF(ssb_fft_kernel, blockDim,
                inp, w16r, w16i, twr, twi, gidx, dre, dim, scr, outRe, outIm, ws, til);

    WriteFile(GOutRe().c_str(), outRe, outBytes);
    WriteFile(GOutIm().c_str(), outIm, outBytes);

    AscendC::GmFree(inp);  AscendC::GmFree(w16r); AscendC::GmFree(w16i);
    AscendC::GmFree(twr);  AscendC::GmFree(twi);  AscendC::GmFree(gidx);
    AscendC::GmFree(dre);  AscendC::GmFree(dim);  AscendC::GmFree(scr);
    AscendC::GmFree(outRe); AscendC::GmFree(outIm);
    AscendC::GmFree(ws);   AscendC::GmFree(til);
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inpH, *inpD;
    uint8_t *w16rH, *w16rD, *w16iH, *w16iD;
    uint8_t *twrH,  *twrD,  *twiH,  *twiD;
    uint8_t *gidxH, *gidxD;
    uint8_t *dreH,  *dreD,  *dimH,  *dimD;
    uint8_t *outReH, *outReD, *outImH, *outImD;
    uint8_t *scrD,  *wsD;
    uint8_t *tilH,  *tilD;

    LoadInput(GInput(),                      inputAlloc, inputReal, &inpH,  &inpD);
    LoadInput(GWeight("w16_re.bin"),         w16Bytes,   w16Bytes,  &w16rH, &w16rD);
    LoadInput(GWeight("w16_im.bin"),         w16Bytes,   w16Bytes,  &w16iH, &w16iD);
    LoadInput(GWeight("twiddle_pq_re.bin"),  twBytes,    twBytes,   &twrH,  &twrD);
    LoadInput(GWeight("twiddle_pq_im.bin"),  twBytes,    twBytes,   &twiH,  &twiD);
    LoadInput(GWeight("gather_idx.bin"),     gidxBytes,  gidxBytes, &gidxH, &gidxD);
    LoadInput(GWeight("derot_re.bin"),       derotBytes, derotBytes,&dreH,  &dreD);
    LoadInput(GWeight("derot_im.bin"),       derotBytes, derotBytes,&dimH,  &dimD);

    CHECK_ACL(aclrtMallocHost((void **)&outReH, outBytes));
    CHECK_ACL(aclrtMallocHost((void **)&outImH, outBytes));
    CHECK_ACL(aclrtMalloc((void **)&outReD, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&outImD, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    memset(outReH, 0, outBytes); memset(outImH, 0, outBytes);
    CHECK_ACL(aclrtMemcpy(outReD, outBytes, outReH, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(outImD, outBytes, outImH, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(aclrtMalloc((void **)&scrD, scratchBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsD,  wsBytes,      ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&tilH, tilingBytes));
    CHECK_ACL(aclrtMalloc((void **)&tilD, tilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    memcpy(tilH, tilingBuf, tilingBytes);
    CHECK_ACL(aclrtMemcpy(tilD, tilingBytes, tilH, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("[ssb_fft] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
        ACLRT_LAUNCH_KERNEL(ssb_fft_kernel)
        (blockDim, stream,
         inpD, w16rD, w16iD, twrD, twiD, gidxD, dreD, dimD, scrD, outReD, outImD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    printf("[ssb_fft] timed:   %d runs\n", N_TIMED);
    std::vector<double> us_list; us_list.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(ssb_fft_kernel)
        (blockDim, stream,
         inpD, w16rD, w16iD, twrD, twiD, gidxD, dreD, dimD, scrD, outReD, outImD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        us_list.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    double sum = 0, mn = us_list[0], mx = us_list[0];
    for (double v : us_list) { sum += v; if (v < mn) mn = v; if (v > mx) mx = v; }
    std::vector<double> sorted = us_list; std::sort(sorted.begin(), sorted.end());
    printf("[ssb_fft] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           mn, sum / N_TIMED, sorted[N_TIMED / 2], sorted[(N_TIMED * 99) / 100], mx);

    CHECK_ACL(aclrtMemcpy(outReH, outBytes, outReD, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(outImH, outBytes, outImD, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(GOutRe().c_str(), outReH, outBytes);
    WriteFile(GOutIm().c_str(), outImH, outBytes);

    CHECK_ACL(aclrtFree(inpD));   CHECK_ACL(aclrtFreeHost(inpH));
    CHECK_ACL(aclrtFree(w16rD));  CHECK_ACL(aclrtFreeHost(w16rH));
    CHECK_ACL(aclrtFree(w16iD));  CHECK_ACL(aclrtFreeHost(w16iH));
    CHECK_ACL(aclrtFree(twrD));   CHECK_ACL(aclrtFreeHost(twrH));
    CHECK_ACL(aclrtFree(twiD));   CHECK_ACL(aclrtFreeHost(twiH));
    CHECK_ACL(aclrtFree(gidxD));  CHECK_ACL(aclrtFreeHost(gidxH));
    CHECK_ACL(aclrtFree(dreD));   CHECK_ACL(aclrtFreeHost(dreH));
    CHECK_ACL(aclrtFree(dimD));   CHECK_ACL(aclrtFreeHost(dimH));
    CHECK_ACL(aclrtFree(outReD)); CHECK_ACL(aclrtFreeHost(outReH));
    CHECK_ACL(aclrtFree(outImD)); CHECK_ACL(aclrtFreeHost(outImH));
    CHECK_ACL(aclrtFree(scrD));   CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtFree(tilD));   CHECK_ACL(aclrtFreeHost(tilH));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBuf);
    return 0;
}
