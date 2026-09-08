


#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "data_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "decimate.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_decimate_kernel.h"
#else
#include "tikicpulib.h"
extern "C" void decimate_kernel(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

using namespace decimate;

namespace {
constexpr int N_WARMUP = 5;
constexpr int N_TIMED  = 20;

std::string DataDir()
{
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}
std::string GInput()  { return DataDir() + "/data/golden/input.bin"; }
std::string GTaps()   { return DataDir() + "/weights/fir_taps.bin"; }
std::string GOut()    { return DataDir() + "/data/ascend_output/output.bin"; }

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


int32_t main(int32_t, char *[])
{
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    const size_t inputBytes  = static_cast<size_t>(IN_GM_I16_LEN) * sizeof(int16_t);
    const size_t tapsBytes   = static_cast<size_t>(L_TAP) * sizeof(uint16_t);
    const size_t outBytes    = static_cast<size_t>(OUT_GM_I16_LEN) * sizeof(int16_t);
    const size_t tilingBytes = 256;
    const size_t wsBytes     = static_cast<size_t>(plat->GetLibApiWorkSpaceSize());

    uint8_t *tilingBuf = (uint8_t *)malloc(tilingBytes);
    GenerateTiling(socVersion, tilingBuf);

    uint32_t blockDim = BLOCK_DIM;
    printf("[decimate] SOC=%s blockDim=%u  N_IN=%u N_OUT=%u  L_TAP=%u\n",
           socVersion, blockDim, N_IN, N_OUT, L_TAP);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *inp  = (uint8_t *)AscendC::GmAlloc(inputBytes);
    uint8_t *taps = (uint8_t *)AscendC::GmAlloc(tapsBytes);
    uint8_t *outIQ = (uint8_t *)AscendC::GmAlloc(outBytes);
    uint8_t *ws   = (uint8_t *)AscendC::GmAlloc(wsBytes ? wsBytes : 16);
    uint8_t *til  = (uint8_t *)AscendC::GmAlloc(tilingBytes);

    ReadFile(GInput().c_str(), inputBytes, inp,  inputBytes);
    ReadFile(GTaps().c_str(),  tapsBytes,  taps, tapsBytes);
    memcpy(til, tilingBuf, tilingBytes);

    ICPU_RUN_KF(decimate_kernel, blockDim, inp, taps, outIQ, ws, til);

    WriteFile(GOut().c_str(), outIQ, outBytes);

    AscendC::GmFree(inp);   AscendC::GmFree(taps); AscendC::GmFree(outIQ);
    AscendC::GmFree(ws);    AscendC::GmFree(til);
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inpH, *inpD, *tapsH, *tapsD;
    uint8_t *outH, *outD;
    uint8_t *wsD, *tilH, *tilD;

    LoadInput(GInput(), inputBytes, &inpH,  &inpD);
    LoadInput(GTaps(),  tapsBytes,  &tapsH, &tapsD);

    CHECK_ACL(aclrtMallocHost((void **)&outH, outBytes));
    CHECK_ACL(aclrtMalloc((void **)&outD, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsD, wsBytes ? wsBytes : 16, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&tilH, tilingBytes));
    CHECK_ACL(aclrtMalloc((void **)&tilD, tilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    memcpy(tilH, tilingBuf, tilingBytes);
    CHECK_ACL(aclrtMemcpy(tilD, tilingBytes, tilH, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("[decimate] warm-up: %d\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        ACLRT_LAUNCH_KERNEL(decimate_kernel)
        (blockDim, stream, inpD, tapsD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    printf("[decimate] timed: %d\n", N_TIMED);
    std::vector<double> us;
    us.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(decimate_kernel)
        (blockDim, stream, inpD, tapsD, outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());
    double sum = 0; for (double v : us) sum += v;
    printf("[decimate] latency: min %.1f us, avg %.1f us, p50 %.1f us, max %.1f us\n",
           us.front(), sum / N_TIMED, us[N_TIMED / 2], us.back());

    CHECK_ACL(aclrtMemcpy(outH, outBytes, outD, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(GOut().c_str(), outH, outBytes);

    CHECK_ACL(aclrtFree(inpD));  CHECK_ACL(aclrtFreeHost(inpH));
    CHECK_ACL(aclrtFree(tapsD)); CHECK_ACL(aclrtFreeHost(tapsH));
    CHECK_ACL(aclrtFree(outD));  CHECK_ACL(aclrtFreeHost(outH));
    CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtFree(tilD));  CHECK_ACL(aclrtFreeHost(tilH));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBuf);
    return 0;
}