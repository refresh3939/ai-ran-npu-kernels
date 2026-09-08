





#include "rate_dematch.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_rate_dematch_kernel.h"
#endif

namespace airan {
extern "C" uint8_t* GenerateTiling   (const char*, uint32_t);
extern "C" size_t   GetTilingSize    ();
extern "C" size_t   GetWorkspaceSize ();
}

#define CHECK_ACL(call) do { \
    auto _e = (call); \
    if (_e != ACL_SUCCESS) { \
        fprintf(stderr, "ACL error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

static const char* GetDataRoot() {
    const char* r = std::getenv("AIRAN_DATA_DIR");
    return r ? r : "../../..";
}
static bool ReadBin(const std::string& path, void* buf, size_t expect) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) { fprintf(stderr, "[err] open %s failed\n", path.c_str()); return false; }
    size_t sz = (size_t)f.tellg();
    if (sz != expect) { fprintf(stderr, "[err] %s size %zu != %zu\n", path.c_str(), sz, expect); return false; }
    f.seekg(0); f.read((char*)buf, (std::streamsize)sz);
    return true;
}

int main() {
    using namespace airan;
    const std::string dir = std::string(GetDataRoot()) + "/data/golden/rx/rate_dematch";

    const size_t DESCRAM_B = (size_t)DESCRAM_LEN * sizeof(int16_t);
    const size_t LAM_B     = (size_t)C_NUM * LDPC_N * sizeof(int16_t);

    std::vector<int16_t>  hDescram(DESCRAM_B / sizeof(int16_t));
    std::vector<int16_t>  hRef    (LAM_B     / sizeof(int16_t));
    std::vector<int16_t>  hLam    (LAM_B     / sizeof(int16_t));

    if (!ReadBin(dir + "/descram_in.bin", hDescram.data(), DESCRAM_B)) return 1;
    if (!ReadBin(dir + "/lam_ref.bin",    hRef.data(),     LAM_B))     return 1;

    printf("=== rate_dematch (38.212 §5.4.2, rv0; in-kernel descriptor) ===\n");
    printf("  C_NUM=%u  LDPC_N=%u  blockDim=%u\n", C_NUM, LDPC_N, BLOCK_DIM);

#ifndef ASCENDC_CPU_DEBUG
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *dDescram = nullptr, *dLam = nullptr;
    CHECK_ACL(aclrtMalloc(&dDescram, DESCRAM_B, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dLam,     LAM_B,     ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(dDescram, DESCRAM_B, hDescram.data(), DESCRAM_B, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(dLam, LAM_B, 0, LAM_B));

    const uint32_t blockDim = BLOCK_DIM;
    printf("\n[launch] blockDim=%u...\n", blockDim);
    ACLRT_LAUNCH_KERNEL(rate_dematch_kernel)(blockDim, stream,
        (uint8_t*)dDescram, (uint8_t*)dDescram, (uint8_t*)dDescram, (uint8_t*)dLam);
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        fprintf(stderr, "[err] sync after launch failed\n"); return 1;
    }
    CHECK_ACL(aclrtMemcpy(hLam.data(), LAM_B, dLam, LAM_B, ACL_MEMCPY_DEVICE_TO_HOST));

    size_t bad = 0, first = hLam.size();
    for (size_t i = 0; i < hLam.size(); ++i)
        if (hLam[i] != hRef[i]) { if (bad == 0) first = i; ++bad; }
    printf("\n[verify] %zu / %zu mismatches -> %s\n",
           bad, hLam.size(), bad == 0 ? "PASS (bit-exact, TOL=0)" : "FAIL");
    if (bad) printf("  first @ %zu (cb=%zu col=%zu): got=%d ref=%d\n",
                    first, first / LDPC_N, first % LDPC_N, (int)hLam[first], (int)hRef[first]);
    if (bad) {
        printf("  cb0 payload got:");
        for (size_t i = 0; i < 16; ++i) printf(" %d", (int)hLam[N_2Z + i]);
        printf("\n  cb0 payload ref:");
        for (size_t i = 0; i < 16; ++i) printf(" %d", (int)hRef[N_2Z + i]);
        printf("\n");
    }
    printf("  [sanity] lam[0,768..771] got/ref = "
           "%d/%d %d/%d %d/%d %d/%d\n",
           (int)hLam[768], (int)hRef[768], (int)hLam[769], (int)hRef[769],
           (int)hLam[770], (int)hRef[770], (int)hLam[771], (int)hRef[771]);

    if (bad == 0) {
        const int LPS = 10, REPS = 50;
        std::vector<double> us; us.reserve(REPS);
        auto launch = [&]() {
            ACLRT_LAUNCH_KERNEL(rate_dematch_kernel)(blockDim, stream,
                (uint8_t*)dDescram, (uint8_t*)dDescram, (uint8_t*)dDescram, (uint8_t*)dLam);
        };
        for (int w = 0; w < 5; ++w) { launch(); }
        CHECK_ACL(aclrtSynchronizeStream(stream));
        for (int r = 0; r < REPS; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < LPS; ++k) launch();
            CHECK_ACL(aclrtSynchronizeStream(stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / LPS);
        }
        std::sort(us.begin(), us.end());

        constexpr double N_SLOT_TB = static_cast<double>(N_SLOT_MAX);
        printf("\n=== Performance ===\n");
        printf("  %d runs x %d launches/sync\n", REPS, LPS);
        printf("  Per TB:   p50=%.1f us  min=%.1f us  p90=%.1f us\n",
               us[REPS/2], us.front(), us[(REPS*9)/10]);
        printf("  Per slot: p50=%.2f us  min=%.2f us  p90=%.2f us  (TB/%.0f slots)\n",
               us[REPS/2] / N_SLOT_TB,
               us.front()  / N_SLOT_TB,
               us[(REPS*9)/10] / N_SLOT_TB,
               N_SLOT_TB);
    }

    CHECK_ACL(aclrtFree(dDescram));
    CHECK_ACL(aclrtFree(dLam));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return bad ? 1 : 0;
#else
    return 0;
#endif
}
