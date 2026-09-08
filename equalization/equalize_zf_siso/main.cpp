













#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_equalize_kernel.h"
#else
#include "tikicpulib.h"

extern "C" void equalize_kernel(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                uint8_t *, uint8_t *, uint8_t *, uint8_t *);
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

namespace {

constexpr uint32_t N_SYMBOL  = 14;
constexpr uint32_t N_BLOCKS  = 4;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_SC_PAD  = 1664;

constexpr size_t ROW_LOG  = N_SC_USED * sizeof(uint16_t);
constexpr size_t ROW_PAD  = N_SC_PAD  * sizeof(uint16_t);
constexpr size_t S_LOG    = N_SYMBOL * ROW_LOG;
constexpr size_t S_PAD    = N_SYMBOL * ROW_PAD;
constexpr uint32_t S_LOG_HALF = N_SYMBOL * N_SC_USED;

constexpr size_t TILING_TOTAL = 128;
constexpr int N_WARMUP = 10;
constexpr int N_TIMED  = 50;

struct TestCase { const char *dir_name; };
constexpr TestCase TEST_CASES[] = {
    {"case_0_awgn_only"}, {"case_1_flat_fading"}, {"case_2_freq_selective"},
    {"case_3_freq_time_var"}, {"case_4_low_snr"},
};
constexpr int N_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);

std::string DataDir() {
    const char *e = std::getenv("AIRAN_DATA_DIR");
    return e ? std::string(e) : std::string(".");
}
std::string GoldenDir(int i) { return DataDir() + "/data/golden/" + TEST_CASES[i].dir_name; }
std::string OutDir(int i)    { return DataDir() + "/data/ascend_output/" + TEST_CASES[i].dir_name; }
void MakeDir(const std::string &p) { int rc = std::system(("mkdir -p " + p).c_str()); (void)rc; }

void PadRows(uint8_t *dst, const uint8_t *src) {
    memset(dst, 0, S_PAD);
    for (size_t r = 0; r < N_SYMBOL; ++r) memcpy(dst + r * ROW_PAD, src + r * ROW_LOG, ROW_LOG);
}
void UnpadRows(uint8_t *dst, const uint8_t *src) {
    for (size_t r = 0; r < N_SYMBOL; ++r) memcpy(dst + r * ROW_LOG, src + r * ROW_PAD, ROW_LOG);
}

void SanityCheck(const char *label, const uint16_t *buf, size_t n_half) {
    int zero = 0, sat = 0, nan = 0;
    for (size_t i = 0; i < n_half; ++i) {
        uint16_t b = buf[i];
        if (b == 0) zero++;
        uint16_t exp = (b >> 10) & 0x1f, mant = b & 0x3ff;
        if (exp == 0x1f) { if (mant == 0) sat++; else nan++; }
    }
    printf("  [%s] %zu half: zero=%d sat=%d nan=%d\n", label, n_half, zero, sat, nan);
}

const char *IN[5]  = {"in_y_re.bin", "in_y_im.bin", "in_h_re.bin", "in_h_im.bin", "in_n0.bin"};
const char *OUT[3] = {"out_xhat_re.bin", "out_xhat_im.bin", "out_no_eff.bin"};

}


int32_t main(int32_t, char *[])
{
    const char *soc = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    const size_t wsBytes = static_cast<size_t>(plat->GetLibApiWorkSpaceSize());

    uint8_t *tilingBuf = (uint8_t *)malloc(TILING_TOTAL);
    GenerateTiling(soc, tilingBuf);

    uint32_t blockDim = N_BLOCKS;
    printf("[equalize] SOC=%s blockDim=%u | full-grid fp16, %u SC pad, stream pad %zu B, ws %zu B\n",
           soc, blockDim, N_SC_PAD, S_PAD, wsBytes);

    uint8_t *sLog = (uint8_t *)malloc(S_LOG);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *iB[5]; for (int k = 0; k < 5; ++k) iB[k] = (uint8_t *)AscendC::GmAlloc(S_PAD);
    uint8_t *oB[3]; for (int k = 0; k < 3; ++k) oB[k] = (uint8_t *)AscendC::GmAlloc(S_PAD);
    uint8_t *ws = (uint8_t *)AscendC::GmAlloc(wsBytes);

    for (int ci = 0; ci < N_CASES; ++ci) {
        size_t bio;
        for (int k = 0; k < 5; ++k) {
            ReadFile((GoldenDir(ci) + "/" + IN[k]).c_str(), bio, sLog, S_LOG);
            PadRows(iB[k], sLog);
        }
        for (int k = 0; k < 3; ++k) memset(oB[k], 0, S_PAD);

        ICPU_RUN_KF(equalize_kernel, blockDim,
                    iB[0], iB[1], iB[2], iB[3], iB[4], oB[0], oB[1], oB[2], ws);

        printf("[cpu] %s\n", TEST_CASES[ci].dir_name);
        MakeDir(OutDir(ci));
        for (int k = 0; k < 3; ++k) {
            UnpadRows(sLog, oB[k]);
            SanityCheck(OUT[k], (const uint16_t *)sLog, S_LOG_HALF);
            WriteFile((OutDir(ci) + "/" + OUT[k]).c_str(), sLog, S_LOG);
        }
    }
    for (int k = 0; k < 5; ++k) AscendC::GmFree(iB[k]);
    for (int k = 0; k < 3; ++k) AscendC::GmFree(oB[k]);
    AscendC::GmFree(ws);
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *iH[5], *iD[5], *oH[3], *oD[3], *wsD, *tilH, *tilD;
    for (int k = 0; k < 5; ++k) {
        CHECK_ACL(aclrtMallocHost((void **)&iH[k], S_PAD));
        CHECK_ACL(aclrtMalloc((void **)&iD[k], S_PAD, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    for (int k = 0; k < 3; ++k) {
        CHECK_ACL(aclrtMallocHost((void **)&oH[k], S_PAD));
        CHECK_ACL(aclrtMalloc((void **)&oD[k], S_PAD, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclrtMalloc((void **)&wsD, wsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void **)&tilH, TILING_TOTAL));
    CHECK_ACL(aclrtMalloc((void **)&tilD, TILING_TOTAL, ACL_MEM_MALLOC_HUGE_FIRST));
    memcpy(tilH, tilingBuf, TILING_TOTAL);
    CHECK_ACL(aclrtMemcpy(tilD, TILING_TOTAL, tilH, TILING_TOTAL, ACL_MEMCPY_HOST_TO_DEVICE));

    auto load_case = [&](int ci) {
        size_t bio;
        for (int k = 0; k < 5; ++k) {
            ReadFile((GoldenDir(ci) + "/" + IN[k]).c_str(), bio, sLog, S_LOG);
            PadRows(iH[k], sLog);
            CHECK_ACL(aclrtMemcpy(iD[k], S_PAD, iH[k], S_PAD, ACL_MEMCPY_HOST_TO_DEVICE));
        }
        for (int k = 0; k < 3; ++k) {
            memset(oH[k], 0, S_PAD);
            CHECK_ACL(aclrtMemcpy(oD[k], S_PAD, oH[k], S_PAD, ACL_MEMCPY_HOST_TO_DEVICE));
        }
    };
    auto launch = [&]() {
        ACLRT_LAUNCH_KERNEL(equalize_kernel)
        (blockDim, stream, iD[0], iD[1], iD[2], iD[3], iD[4],
         oD[0], oD[1], oD[2], wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    };

    printf("\n[equalize] running %d cases:\n", N_CASES);
    for (int ci = 0; ci < N_CASES; ++ci) {
        load_case(ci);
        launch();
        for (int k = 0; k < 3; ++k)
            CHECK_ACL(aclrtMemcpy(oH[k], S_PAD, oD[k], S_PAD, ACL_MEMCPY_DEVICE_TO_HOST));
        printf("  %-25s\n", TEST_CASES[ci].dir_name);
        MakeDir(OutDir(ci));
        for (int k = 0; k < 3; ++k) {
            UnpadRows(sLog, oH[k]);
            SanityCheck(OUT[k], (const uint16_t *)sLog, S_LOG_HALF);
            WriteFile((OutDir(ci) + "/" + OUT[k]).c_str(), sLog, S_LOG);
        }
    }

    load_case(2);
    printf("\n[equalize] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) launch();
    printf("[equalize] timed: %d runs\n", N_TIMED);
    std::vector<double> us; us.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        launch();
        auto t1 = std::chrono::high_resolution_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    double sum = 0, mn = us[0], mx = us[0];
    for (double v : us) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    std::vector<double> sd = us; std::sort(sd.begin(), sd.end());
    printf("[equalize] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           mn, sum / N_TIMED, sd[N_TIMED / 2], sd[(N_TIMED * 99) / 100], mx);
    printf("[equalize] real-time headroom (1ms slot): %.1fx\n", 1000.0 / (sum / N_TIMED));

    for (int k = 0; k < 5; ++k) { CHECK_ACL(aclrtFree(iD[k])); CHECK_ACL(aclrtFreeHost(iH[k])); }
    for (int k = 0; k < 3; ++k) { CHECK_ACL(aclrtFree(oD[k])); CHECK_ACL(aclrtFreeHost(oH[k])); }
    CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtFree(tilD)); CHECK_ACL(aclrtFreeHost(tilH));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif

    free(sLog); free(tilingBuf);
    return 0;
}