






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
#include "dmrs_gen.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_dmrs_gen_kernel.h"
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

namespace {

constexpr int N_WARMUP = 10;
constexpr int N_TIMED  = 50;

constexpr size_t N_RE     = 798;
constexpr size_t N_PAD    = 896;
constexpr size_t PLANE    = 1792;
constexpr size_t NBITS    = 31;
constexpr size_t NC       = 1600;
constexpr size_t M        = 2 * N_RE;
constexpr size_t MAT_LEN  = NBITS * PLANE;
constexpr size_t GRID_HALF = 2 * N_PAD;
constexpr size_t CINIT_PAD = 16;
constexpr size_t OUT_DBG_LEN = 8;
constexpr size_t TILING_BYTES = 128;
constexpr size_t SCR_BYTES    = 128;

constexpr uint16_t H_ONE  = 0x3C00;
constexpr uint16_t H_ZERO = 0x0000;
constexpr float ERR_THRESH = 1e-3f;

struct TestCase { const char *dir_name; };
constexpr TestCase TEST_CASES[] = {
    {"case_0_baseline"}, {"case_1_nid0"}, {"case_2_slot7"},
    {"case_3_nscid1"},   {"case_4_nid_large"},
};
constexpr int N_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);

std::string DataDir() {
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}
std::string CasePath(int i, const char *f) {
    return DataDir() + "/data/golden/" + TEST_CASES[i].dir_name + "/" + f;
}
std::string AscendReOut(int i) {
    return DataDir() + "/data/ascend_output/dmrs_gen_" + TEST_CASES[i].dir_name + "_re.bin";
}
std::string AscendImOut(int i) {
    return DataDir() + "/data/ascend_output/dmrs_gen_" + TEST_CASES[i].dir_name + "_im.bin";
}




void BuildDmrsMatrix(uint16_t *gmat, uint16_t *g1) {

    std::vector<int8_t> x1(M + NC + 31, 0);
    x1[0] = 1;
    for (size_t n = 0; n < M + NC; ++n) x1[n + 31] = (x1[n + 3] + x1[n]) & 1;

    auto put = [&](uint16_t *dst, const int8_t *seq) {

        for (size_t k = 0; k < N_RE; ++k) {
            dst[k]         = seq[2 * k]     ? H_ONE : H_ZERO;
            dst[N_PAD + k] = seq[2 * k + 1] ? H_ONE : H_ZERO;
        }
        for (size_t p = N_RE; p < N_PAD; ++p) { dst[p] = H_ZERO; dst[N_PAD + p] = H_ZERO; }
    };


    put(g1, &x1[NC]);


    std::vector<int8_t> x2(M + NC + 31, 0);
    for (size_t i = 0; i < NBITS; ++i) {
        std::fill(x2.begin(), x2.end(), 0);
        for (size_t j = 0; j < 31; ++j) x2[j] = ((1u << i) >> j) & 1;
        for (size_t n = 0; n < M + NC; ++n)
            x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) & 1;
        put(&gmat[i * PLANE], &x2[NC]);
    }
}

inline float half2float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else { exp = 113; while ((man & 0x400) == 0) { man <<= 1; --exp; } man &= 0x3FF;
               f = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) { f = sign | 0x7F800000 | (man << 13); }
    else { f = sign | ((exp + 112) << 23) | (man << 13); }
    float out; std::memcpy(&out, &f, 4); return out;
}
float MaxAbsErr(const uint16_t *a, const uint16_t *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(half2float(a[i]) - half2float(b[i]));
        if (d > m) m = d;
    }
    return m;
}

}


int32_t main(int32_t, char *[])
{
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    const size_t cinitBytes = CINIT_PAD * sizeof(int32_t);
    const size_t gmatBytes  = MAT_LEN   * sizeof(uint16_t);
    const size_t g1Bytes    = PLANE     * sizeof(uint16_t);
    const size_t gridBytes  = GRID_HALF * sizeof(uint16_t);
    const size_t dbgBytes   = OUT_DBG_LEN * sizeof(float);
    const size_t wsBytes    = (size_t)plat->GetLibApiWorkSpaceSize();

    uint8_t *tilingBuf = (uint8_t *)malloc(TILING_BYTES);
    GenerateTiling(socVersion, tilingBuf);

    uint32_t blockDim = airan::BLOCK_DIM;
    printf("[dmrs_gen v2] SOC=%s blockDim=%u\n", socVersion, blockDim);
    printf("[dmrs_gen v2] gmat=%zuB g1=%zuB grid=%zuB/plane(x2) ws=%zuB\n",
           gmatBytes, g1Bytes, gridBytes, wsBytes);

#ifdef ASCENDC_CPU_DEBUG
    fprintf(stderr, "[dmrs_gen] CPU debug mode unsupported here\n");
    return 1;
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *ciH=nullptr, *gmatH=nullptr, *g1H=nullptr, *reH=nullptr, *imH=nullptr, *dbgH=nullptr;
    uint8_t *goldReH=nullptr, *goldImH=nullptr;
    uint8_t *ciD=nullptr, *gmatD=nullptr, *g1D=nullptr, *reD=nullptr, *imD=nullptr, *dbgD=nullptr;
    uint8_t *scrD=nullptr, *wsD=nullptr, *tilH=nullptr, *tilD=nullptr;

    CHECK_ACL(aclrtMallocHost((void **)&ciH,   cinitBytes));
    CHECK_ACL(aclrtMallocHost((void **)&gmatH, gmatBytes));
    CHECK_ACL(aclrtMallocHost((void **)&g1H,   g1Bytes));
    CHECK_ACL(aclrtMallocHost((void **)&reH,   gridBytes));
    CHECK_ACL(aclrtMallocHost((void **)&imH,   gridBytes));
    CHECK_ACL(aclrtMallocHost((void **)&dbgH,  dbgBytes));
    CHECK_ACL(aclrtMallocHost((void **)&goldReH, gridBytes));
    CHECK_ACL(aclrtMallocHost((void **)&goldImH, gridBytes));
    CHECK_ACL(aclrtMalloc((void **)&ciD,   cinitBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&gmatD, gmatBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&g1D,   g1Bytes,    ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&reD,   gridBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&imD,   gridBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&dbgD,  dbgBytes,   ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&scrD,  SCR_BYTES,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsD,   wsBytes,    ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&tilH, TILING_BYTES));
    CHECK_ACL(aclrtMalloc((void **)&tilD, TILING_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    std::memcpy(tilH, tilingBuf, TILING_BYTES);
    CHECK_ACL(aclrtMemcpy(tilD, TILING_BYTES, tilH, TILING_BYTES, ACL_MEMCPY_HOST_TO_DEVICE));


    BuildDmrsMatrix((uint16_t *)gmatH, (uint16_t *)g1H);
    CHECK_ACL(aclrtMemcpy(gmatD, gmatBytes, gmatH, gmatBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(g1D,   g1Bytes,   g1H,   g1Bytes,   ACL_MEMCPY_HOST_TO_DEVICE));

    int n_pass = 0;
    for (int ci = 0; ci < N_CASES; ++ci) {
        size_t got = 0; bool ok_in = true;
        std::memset(ciH, 0, cinitBytes);
        ok_in &= ReadFile(CasePath(ci, "cinit.bin").c_str(), got, ciH, 2 * sizeof(int32_t))
                 && got == 2 * sizeof(int32_t);
        ok_in &= ReadFile(CasePath(ci, "x_re.bin").c_str(), got, goldReH, gridBytes) && got == gridBytes;
        ok_in &= ReadFile(CasePath(ci, "x_im.bin").c_str(), got, goldImH, gridBytes) && got == gridBytes;
        if (!ok_in) { ERROR_LOG("case %d: missing/wrong-size bins", ci); continue; }

        CHECK_ACL(aclrtMemcpy(ciD, cinitBytes, ciH, cinitBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        std::memset(reH, 0xff, gridBytes); std::memset(imH, 0xff, gridBytes); std::memset(dbgH, 0xff, dbgBytes);
        CHECK_ACL(aclrtMemcpy(reD,  gridBytes, reH,  gridBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(imD,  gridBytes, imH,  gridBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(dbgD, dbgBytes,  dbgH, dbgBytes,  ACL_MEMCPY_HOST_TO_DEVICE));


        ACLRT_LAUNCH_KERNEL(dmrs_gen_kernel)
            (blockDim, stream, ciD, gmatD, g1D, scrD, reD, imD, dbgD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));

        CHECK_ACL(aclrtMemcpy(reH,  gridBytes, reD,  gridBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(imH,  gridBytes, imD,  gridBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(dbgH, dbgBytes,  dbgD, dbgBytes,  ACL_MEMCPY_DEVICE_TO_HOST));

        WriteFile(AscendReOut(ci).c_str(), reH, gridBytes);
        WriteFile(AscendImOut(ci).c_str(), imH, gridBytes);

        float re_err = MaxAbsErr((uint16_t *)reH, (uint16_t *)goldReH, GRID_HALF);
        float im_err = MaxAbsErr((uint16_t *)imH, (uint16_t *)goldImH, GRID_HALF);
        float sentinel = ((float *)dbgH)[7];
        bool sentinel_ok = (sentinel == 7.0f);
        bool err_ok = (re_err < ERR_THRESH) && (im_err < ERR_THRESH);
        bool ok = sentinel_ok && err_ok;
        printf("  case %d %-18s re_err=%.5f im_err=%.5f  c_init=[%.0f,%.0f]  %s\n",
               ci, TEST_CASES[ci].dir_name, re_err, im_err,
               ((float *)dbgH)[0], ((float *)dbgH)[1],
               ok ? "[PASS]" : (sentinel_ok ? "[FAIL]" : "[NO-SENTINEL]"));
        if (ok) ++n_pass;
    }
    printf("[dmrs_gen v2] %d / %d PASS\n", n_pass, N_CASES);


    { size_t got = 0; std::memset(ciH, 0, cinitBytes);
      ReadFile(CasePath(0, "cinit.bin").c_str(), got, ciH, 2 * sizeof(int32_t));
      CHECK_ACL(aclrtMemcpy(ciD, cinitBytes, ciH, cinitBytes, ACL_MEMCPY_HOST_TO_DEVICE)); }
    for (int i = 0; i < N_WARMUP; ++i) {
        ACLRT_LAUNCH_KERNEL(dmrs_gen_kernel)(blockDim, stream, ciD, gmatD, g1D, scrD, reD, imD, dbgD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }
    std::vector<double> us; us.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(dmrs_gen_kernel)(blockDim, stream, ciD, gmatD, g1D, scrD, reD, imD, dbgD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());
    double sum = 0; for (double v : us) sum += v;
    printf("[dmrs_gen v2] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           us.front(), sum / N_TIMED, us[N_TIMED / 2], us[(N_TIMED * 99) / 100], us.back());

    CHECK_ACL(aclrtFreeHost(ciH));   CHECK_ACL(aclrtFree(ciD));
    CHECK_ACL(aclrtFreeHost(gmatH)); CHECK_ACL(aclrtFree(gmatD));
    CHECK_ACL(aclrtFreeHost(g1H));   CHECK_ACL(aclrtFree(g1D));
    CHECK_ACL(aclrtFreeHost(reH));   CHECK_ACL(aclrtFree(reD));
    CHECK_ACL(aclrtFreeHost(imH));   CHECK_ACL(aclrtFree(imD));
    CHECK_ACL(aclrtFreeHost(dbgH));  CHECK_ACL(aclrtFree(dbgD));
    CHECK_ACL(aclrtFreeHost(goldReH)); CHECK_ACL(aclrtFreeHost(goldImH));
    CHECK_ACL(aclrtFree(scrD));      CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtFreeHost(tilH));  CHECK_ACL(aclrtFree(tilD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBuf);
    return 0;
}