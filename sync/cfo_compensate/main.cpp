




#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "data_utils.h"
#include "cfo_compensate.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_cfo_compensate_kernel.h"
#else
#include "tikicpulib.h"
extern "C" void cfo_compensate_kernel(uint8_t *, uint8_t *, uint8_t *,
                                      uint8_t *, uint8_t *, uint8_t *, uint8_t *);
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

using namespace cfo_compensate;

namespace {
constexpr int N_WARMUP = 10;
constexpr int N_TIMED  = 50;

std::string DataDir()
{
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}
std::string GInput() { return DataDir() + "/data/golden/input.bin"; }
std::string GSwap()  { return DataDir() + "/data/golden/swap_idx.bin"; }
std::string GA2()    { return DataDir() + "/data/golden/a2.bin"; }
std::string GB2()    { return DataDir() + "/data/golden/b2.bin"; }
std::string GOutIQ() { return DataDir() + "/data/ascend_output/out_iq.bin"; }

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

    const size_t iqBytes   = static_cast<size_t>(IN_INT16_LEN) * sizeof(int16_t);
    const size_t tblBytes  = static_cast<size_t>(OUT_INT16_LEN) * sizeof(int16_t);
    const size_t outBytes  = static_cast<size_t>(OUT_INT16_LEN) * sizeof(int16_t);
    const size_t tilingBytes = TILING_TOTAL_SIZE;
    const size_t wsBytes     = WS_TOTAL;

    uint8_t *tilingBuf = (uint8_t *)malloc(tilingBytes);
    GenerateTiling(socVersion, tilingBuf);

    uint32_t blockDim = BLOCK_DIM;
    printf("[cfo_compensate] SOC=%s blockDim=%u N_SLOT=%u SUB_TILE=%u (swap-trick)\n",
           socVersion, blockDim, N_SAMPLE_PER_SLOT, SUB_TILE);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *inp  = (uint8_t *)AscendC::GmAlloc(iqBytes);
    const size_t swapIdxBytes = 4096u * sizeof(uint32_t);
    uint8_t *swp  = (uint8_t *)AscendC::GmAlloc(swapIdxBytes);
    uint8_t *a2B  = (uint8_t *)AscendC::GmAlloc(tblBytes);
    uint8_t *b2B  = (uint8_t *)AscendC::GmAlloc(tblBytes);
    uint8_t *outIQ= (uint8_t *)AscendC::GmAlloc(outBytes);
    uint8_t *ws   = (uint8_t *)AscendC::GmAlloc(wsBytes);
    uint8_t *til  = (uint8_t *)AscendC::GmAlloc(tilingBytes);

    ReadFile(GInput().c_str(), iqBytes,  inp, iqBytes);
    ReadFile(GSwap().c_str(),  swapIdxBytes,  swp, swapIdxBytes);
    ReadFile(GA2().c_str(),    tblBytes, a2B, tblBytes);
    ReadFile(GB2().c_str(),    tblBytes, b2B, tblBytes);
    memcpy(til, tilingBuf, tilingBytes);

    ICPU_RUN_KF(cfo_compensate_kernel, blockDim, inp, swp, a2B, b2B, outIQ, ws, til);

    WriteFile(GOutIQ().c_str(), outIQ, outBytes);

    AscendC::GmFree(inp); AscendC::GmFree(swp); AscendC::GmFree(a2B);
    AscendC::GmFree(b2B); AscendC::GmFree(outIQ);
    AscendC::GmFree(ws);  AscendC::GmFree(til);
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inpH, *inpD, *swpH, *swpD, *a2H, *a2D, *b2H, *b2D;
    uint8_t *outH, *outD, *wsD, *tilH, *tilD;

    LoadInput(GInput(), iqBytes,  &inpH, &inpD);
    const size_t swapIdxBytes = 4096u * sizeof(uint32_t);
    LoadInput(GSwap(),  swapIdxBytes,  &swpH, &swpD);
    LoadInput(GA2(),    tblBytes, &a2H,  &a2D);
    LoadInput(GB2(),    tblBytes, &b2H,  &b2D);

    CHECK_ACL(aclrtMallocHost((void **)&outH, outBytes));
    CHECK_ACL(aclrtMalloc((void **)&outD, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    memset(outH, 0xAA, outBytes);
    CHECK_ACL(aclrtMemcpy(outD, outBytes, outH, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(aclrtMalloc((void **)&wsD, wsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void **)&tilH, tilingBytes));
    CHECK_ACL(aclrtMalloc((void **)&tilD, tilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    memcpy(tilH, tilingBuf, tilingBytes);
    CHECK_ACL(aclrtMemcpy(tilD, tilingBytes, tilH, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("[cfo_compensate] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        ACLRT_LAUNCH_KERNEL(cfo_compensate_kernel)
        (blockDim, stream, inpD, swpD, a2D, b2D, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    printf("[cfo_compensate] timed:   %d runs\n", N_TIMED);
    std::vector<double> us_list; us_list.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(cfo_compensate_kernel)
        (blockDim, stream, inpD, swpD, a2D, b2D, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        us_list.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    double sum = 0, mn = us_list[0], mx = us_list[0];
    for (double v : us_list) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    std::vector<double> s = us_list; std::sort(s.begin(), s.end());
    printf("[cfo_compensate] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           mn, sum / N_TIMED, s[N_TIMED / 2], s[(N_TIMED * 99) / 100], mx);

    CHECK_ACL(aclrtMemcpy(outH, outBytes, outD, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(GOutIQ().c_str(), outH, outBytes);

    CHECK_ACL(aclrtFree(inpD)); CHECK_ACL(aclrtFreeHost(inpH));
    CHECK_ACL(aclrtFree(swpD)); CHECK_ACL(aclrtFreeHost(swpH));
    CHECK_ACL(aclrtFree(a2D));  CHECK_ACL(aclrtFreeHost(a2H));
    CHECK_ACL(aclrtFree(b2D));  CHECK_ACL(aclrtFreeHost(b2H));
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
