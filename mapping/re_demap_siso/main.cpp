


#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "data_utils.h"
#include "re_demap.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_re_demap_kernel.h"
#else
#include "tikicpulib.h"
extern "C" void re_demap_kernel(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                 uint8_t *, uint8_t *);
#endif

namespace {
using namespace re_demap;

constexpr int N_WARMUP = 10;
constexpr int N_TIMED  = 50;

std::string DataDir()
{
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}
std::string GInRe()  { return DataDir() + "/data/golden/input_re.bin"; }
std::string GInIm()  { return DataDir() + "/data/golden/input_im.bin"; }
std::string GIdx()   { return DataDir() + "/weights/gather_idx.bin"; }
std::string GOutRe() { return DataDir() + "/data/ascend_output/yused_re.bin"; }
std::string GOutIm() { return DataDir() + "/data/ascend_output/yused_im.bin"; }

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
    (void)platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    const size_t inBytes  = static_cast<size_t>(N_SYMBOL) * N_FFT   * sizeof(int16_t);
    const size_t outBytes = static_cast<size_t>(N_SYMBOL) * N_SC_PAD * sizeof(int16_t);
    const size_t idxBytes = static_cast<size_t>(N_SC_PAD) * sizeof(uint32_t);
    const size_t dummyBytes = 1024;

    uint32_t blockDim = BLOCK_DIM;
    printf("[re_demap] SOC=%s blockDim=%u N_FFT=%u N_SC_PAD=%u symbols=%u\n",
           socVersion, blockDim, N_FFT, N_SC_PAD, N_SYMBOL);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *inRe  = (uint8_t *)AscendC::GmAlloc(inBytes);
    uint8_t *inIm  = (uint8_t *)AscendC::GmAlloc(inBytes);
    uint8_t *idx   = (uint8_t *)AscendC::GmAlloc(idxBytes);
    uint8_t *outRe = (uint8_t *)AscendC::GmAlloc(outBytes);
    uint8_t *outIm = (uint8_t *)AscendC::GmAlloc(outBytes);
    uint8_t *ws    = (uint8_t *)AscendC::GmAlloc(dummyBytes);
    uint8_t *til   = (uint8_t *)AscendC::GmAlloc(dummyBytes);

    ReadFile(GInRe().c_str(), inBytes,  inRe, inBytes);
    ReadFile(GInIm().c_str(), inBytes,  inIm, inBytes);
    ReadFile(GIdx().c_str(),  idxBytes, idx,  idxBytes);

    ICPU_RUN_KF(re_demap_kernel, blockDim, inRe, inIm, idx, outRe, outIm, ws, til);

    WriteFile(GOutRe().c_str(), outRe, outBytes);
    WriteFile(GOutIm().c_str(), outIm, outBytes);

    AscendC::GmFree(inRe); AscendC::GmFree(inIm); AscendC::GmFree(idx);
    AscendC::GmFree(outRe); AscendC::GmFree(outIm);
    AscendC::GmFree(ws); AscendC::GmFree(til);
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inReH, *inReD, *inImH, *inImD, *idxH, *idxD;
    uint8_t *outReH, *outReD, *outImH, *outImD;

    LoadInput(GInRe(), inBytes,  &inReH, &inReD);
    LoadInput(GInIm(), inBytes,  &inImH, &inImD);
    LoadInput(GIdx(),  idxBytes, &idxH,  &idxD);

    CHECK_ACL(aclrtMallocHost((void **)&outReH, outBytes));
    CHECK_ACL(aclrtMallocHost((void **)&outImH, outBytes));
    CHECK_ACL(aclrtMalloc((void **)&outReD, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&outImD, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *wsD, *tilD;
    CHECK_ACL(aclrtMalloc((void **)&wsD,  dummyBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&tilD, dummyBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    memset(outReH, 0xAA, outBytes); memset(outImH, 0xAA, outBytes);
    CHECK_ACL(aclrtMemcpy(outReD, outBytes, outReH, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(outImD, outBytes, outImH, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("[re_demap] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        ACLRT_LAUNCH_KERNEL(re_demap_kernel)(blockDim, stream, inReD, inImD, idxD, outReD, outImD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    printf("[re_demap] timed:   %d runs\n", N_TIMED);
    std::vector<double> us_list;
    us_list.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(re_demap_kernel)(blockDim, stream, inReD, inImD, idxD, outReD, outImD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        us_list.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    double sum = 0, mn = us_list[0], mx = us_list[0];
    for (double v : us_list) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    std::vector<double> sorted = us_list;
    std::sort(sorted.begin(), sorted.end());
    printf("[re_demap] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           mn, sum / N_TIMED, sorted[N_TIMED / 2], sorted[(N_TIMED * 99) / 100], mx);

    CHECK_ACL(aclrtMemcpy(outReH, outBytes, outReD, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(outImH, outBytes, outImD, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(GOutRe().c_str(), outReH, outBytes);
    WriteFile(GOutIm().c_str(), outImH, outBytes);

    CHECK_ACL(aclrtFree(inReD));  CHECK_ACL(aclrtFreeHost(inReH));
    CHECK_ACL(aclrtFree(inImD));  CHECK_ACL(aclrtFreeHost(inImH));
    CHECK_ACL(aclrtFree(idxD));   CHECK_ACL(aclrtFreeHost(idxH));
    CHECK_ACL(aclrtFree(outReD)); CHECK_ACL(aclrtFreeHost(outReH));
    CHECK_ACL(aclrtFree(outImD)); CHECK_ACL(aclrtFreeHost(outImH));
    CHECK_ACL(aclrtFree(wsD)); CHECK_ACL(aclrtFree(tilD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}