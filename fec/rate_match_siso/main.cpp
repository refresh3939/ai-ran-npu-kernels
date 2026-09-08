






#include "rate_match.h"
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
#include "aclrtlaunch_rate_match_kernel.h"
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
    const std::string dir = std::string(GetDataRoot()) + "/data/golden/tx/rate_match";

    const size_t CW_B     = (size_t)CODEWORD_LEN * sizeof(int8_t);
    const size_t LAYOUT_B = (size_t)LAYOUT_LEN   * sizeof(int16_t);

    std::vector<int8_t>   hCw    (CW_B     / sizeof(int8_t));
    std::vector<int16_t>  hRef   (LAYOUT_B / sizeof(int16_t));
    std::vector<int16_t>  hLayout(LAYOUT_B / sizeof(int16_t));

    if (!ReadBin(dir + "/codeword_in.bin", hCw.data(),  CW_B))     return 1;
    if (!ReadBin(dir + "/layout_ref.bin",  hRef.data(), LAYOUT_B)) return 1;

    printf("=== rate_match v5c (stream-owned slot batch; 4-CB input ring) ===\n");
    printf("  C_NUM=%u  N_CB_BUF=%u  blockDim=%u\n", C_NUM, N_CB_BUF, BLOCK_DIM);

#ifndef ASCENDC_CPU_DEBUG
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *dCw = nullptr, *dLayout = nullptr;
    CHECK_ACL(aclrtMalloc(&dCw,     CW_B,     ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dLayout, LAYOUT_B, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(dCw, CW_B, hCw.data(), CW_B, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(dLayout, LAYOUT_B, 0, LAYOUT_B));

    const uint32_t blockDim = BLOCK_DIM;
    printf("\n[launch] blockDim=%u...\n", blockDim);
    ACLRT_LAUNCH_KERNEL(rate_match_kernel)(blockDim, stream,
        (uint8_t*)dCw, (uint8_t*)dCw, (uint8_t*)dCw, (uint8_t*)dLayout);
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        fprintf(stderr, "[err] sync after launch failed\n"); return 1;
    }
    CHECK_ACL(aclrtMemcpy(hLayout.data(), LAYOUT_B, dLayout, LAYOUT_B, ACL_MEMCPY_DEVICE_TO_HOST));

    size_t bad = 0, first = hLayout.size();
    for (size_t i = 0; i < hLayout.size(); ++i)
        if (hLayout[i] != hRef[i]) { if (bad == 0) first = i; ++bad; }
    printf("\n[verify] %zu / %zu mismatches -> %s\n",
           bad, hLayout.size(), bad == 0 ? "PASS (bit-exact, TOL=0)" : "FAIL");
    if (bad) {
        const size_t slot = first / SLOT_STRIDE;
        const size_t rem  = first % SLOT_STRIDE;
        printf("  first @ %zu (slot=%zu stream=%zu re=%zu): got=%d ref=%d\n",
               first, slot, rem / STREAM_STRIDE, rem % STREAM_STRIDE,
               (int)hLayout[first], (int)hRef[first]);
        for (size_t b = 0; b < N_STREAMS; ++b) {
            size_t streamBad = 0, gotOnes = 0, refOnes = 0;
            for (size_t s = 0; s < N_SLOT_MAX; ++s) {
                const size_t base = s * SLOT_STRIDE + b * STREAM_STRIDE;
                for (size_t re = 0; re < N_SYM_PAD; ++re) {
                    const int16_t got = hLayout[base + re];
                    const int16_t ref = hRef[base + re];
                    streamBad += got != ref;
                    gotOnes += got != 0;
                    refOnes += ref != 0;
                }
            }
            printf("  stream %zu: bad=%zu got_nonzero=%zu ref_nonzero=%zu\n",
                   b, streamBad, gotOnes, refOnes);
        }
    }

    if (bad == 0) {
        const int LPS = 10, REPS = 50;
        std::vector<double> us; us.reserve(REPS);
        auto launch = [&]() {
            ACLRT_LAUNCH_KERNEL(rate_match_kernel)(blockDim, stream,
                (uint8_t*)dCw, (uint8_t*)dCw, (uint8_t*)dCw, (uint8_t*)dLayout);
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

        constexpr double N_SLOT_TB = 23.0;
        printf("\n=== Performance ===\n");
        printf("  %d runs x %d launches/sync\n", REPS, LPS);
        printf("  Per TB:   p50=%.1f us  min=%.1f us  p90=%.1f us\n",
               us[REPS/2], us.front(), us[(REPS*9)/10]);
        printf("  Per slot: p50=%.2f us  min=%.2f us  p90=%.2f us  (TB/%.0f slots)\n",
               us[REPS/2] / N_SLOT_TB, us.front() / N_SLOT_TB,
               us[(REPS*9)/10] / N_SLOT_TB, N_SLOT_TB);
    }

    CHECK_ACL(aclrtFree(dCw));
    CHECK_ACL(aclrtFree(dLayout));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return bad ? 1 : 0;
#else
    return 0;
#endif
}
