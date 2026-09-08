







#include "llr_assemble.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_llr_assemble_kernel.h"
#endif

namespace airan {
extern "C" uint8_t* GenerateTiling   (const char*, uint32_t);
extern "C" size_t   GetTilingSize    ();
extern "C" size_t   GetWorkspaceSize ();
}

#define CHECK_ACL(call) do { \
    auto _e = (call); \
    if (_e != ACL_SUCCESS) { \
        fprintf(stderr, "ACL error %d at %s:%d\n", _e, __FILE__, __LINE__); std::exit(1); } \
} while (0)

using Clock = std::chrono::high_resolution_clock;

static const char* GetDataRoot() {
    const char* r = std::getenv("AIRAN_DATA_DIR");
    return r ? r : "../../..";
}
static bool ReadBin(const std::string& path, void* buf, size_t expect) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) { fprintf(stderr, "[err] open %s failed\n", path.c_str()); return false; }
    size_t sz = (size_t)f.tellg();
    if (sz != expect) { fprintf(stderr, "[err] %s size %zu != %zu\n", path.c_str(), sz, expect); return false; }
    f.seekg(0); f.read((char*)buf, sz); return true;
}
static bool WriteBin(const std::string& path, const void* buf, size_t sz) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.write((const char*)buf, sz); return true;
}

int main() {
    using namespace airan;
    const std::string root = GetDataRoot();
    const std::string base = root + "/data/golden/rx/llr_assemble";

    printf("=== llr_assemble (per-slot -> codeword rope-copy) ===\n");
    printf("  In   : [%u, %u, %u] int16 (qam per-slot stack)\n", N_SLOT_IN, N_STREAMS, SLOT_PAD);
    printf("  Out  : [%u, %u, %u] int16 (descramble codeword), valid/tile=%u pad/tile=%u\n",
           N_TILE, N_STREAMS, TILE_PAD, TILE_VALID, TILE_PAD - TILE_VALID);
    printf("  CW_LEN/stream=%u  (= %u tiles x %u)\n", CW_LEN, N_TILE, TILE_VALID);

    std::vector<int16_t> hIn  (IN_BYTES  / sizeof(int16_t));
    std::vector<int16_t> hGold(OUT_BYTES / sizeof(int16_t));
    if (!ReadBin(base + "/llr_in.bin",   hIn.data(),   IN_BYTES))  return 1;
    if (!ReadBin(base + "/codeword.bin", hGold.data(), OUT_BYTES)) return 1;
    printf("  Loaded golden files\n");

#ifndef ASCENDC_CPU_DEBUG
    const size_t szIn = IN_BYTES, szOut = OUT_BYTES;
    const size_t szTil = TILING_TOTAL_SIZE, szWs = WS_TOTAL;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *dIn, *dOut, *dWs, *dTil;
    CHECK_ACL(aclrtMalloc(&dIn,  szIn,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dOut, szOut, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dWs,  szWs,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dTil, szTil, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* hTil = GenerateTiling("llr_assemble", BLOCK_DIM);
    CHECK_ACL(aclrtMemcpy(dTil, szTil, hTil, szTil, ACL_MEMCPY_HOST_TO_DEVICE));
    std::free(hTil);

    CHECK_ACL(aclrtMemcpy(dIn, szIn, hIn.data(), szIn, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(dOut, szOut, 0, szOut));

    const uint32_t blockDim = BLOCK_DIM;
    printf("\n[launch] blockDim=%u ...\n", blockDim);
    ACLRT_LAUNCH_KERNEL(llr_assemble_kernel)(blockDim, stream,
        (uint8_t*)dIn, (uint8_t*)dOut, (uint8_t*)dWs, (uint8_t*)dTil);
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) { fprintf(stderr, "[err] sync\n"); return 1; }

    std::vector<int16_t> hOut(OUT_BYTES / sizeof(int16_t));
    CHECK_ACL(aclrtMemcpy(hOut.data(), szOut, dOut, szOut, ACL_MEMCPY_DEVICE_TO_HOST));
    {
        const std::string outDir = root + "/data/ascend_output/rx/llr_assemble";
        WriteBin(outDir + "/codeword.bin", hOut.data(), szOut);
    }


    printf("\n========== Verify ==========\n");
    auto IDX = [](uint32_t t, uint32_t b, uint32_t q) {
        return (size_t)t * N_STREAMS * TILE_PAD + (size_t)b * TILE_PAD + q;
    };
    size_t diff = 0; int max_err = 0; long argmax = -1;
    for (uint32_t t = 0; t < N_TILE; ++t)
        for (uint32_t b = 0; b < N_STREAMS; ++b)
            for (uint32_t q = 0; q < TILE_VALID; ++q) {
                size_t idx = IDX(t, b, q);
                int d = std::abs((int)hOut[idx] - (int)hGold[idx]);
                if (d > 0) ++diff;
                if (d > max_err) { max_err = d; argmax = (long)idx; }
            }
    size_t pad_nonzero = 0;
    for (uint32_t t = 0; t < N_TILE; ++t)
        for (uint32_t b = 0; b < N_STREAMS; ++b)
            for (uint32_t q = TILE_VALID; q < TILE_PAD; ++q)
                if (hOut[IDX(t, b, q)] != 0) ++pad_nonzero;

    printf("  valid diff vs golden: %zu  max_err=%d", diff, max_err);
    if (argmax >= 0 && max_err > 0) {
        long off = argmax % TILE_PAD; int t = argmax / (N_STREAMS * TILE_PAD);
        int b = (argmax / TILE_PAD) % N_STREAMS;
        printf("  (tile=%d stream=%d q=%ld kernel=%d gold=%d)", t, b, off,
               (int)hOut[argmax], (int)hGold[argmax]);
    }
    printf("\n  pad [%u,%u) nonzero: %zu (should be 0)\n", TILE_VALID, TILE_PAD, pad_nonzero);

    bool ok = (diff == 0 && pad_nonzero == 0);
    printf("  %s\n", ok ? "PASS (bit-exact codeword, zero pad)" : "FAIL");


    constexpr int N_BATCH = 100, N_WARMUP = 8, N_RUNS = 10;
    for (int i = 0; i < N_WARMUP; ++i)
        ACLRT_LAUNCH_KERNEL(llr_assemble_kernel)(blockDim, stream,
            (uint8_t*)dIn, (uint8_t*)dOut, (uint8_t*)dWs, (uint8_t*)dTil);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    std::vector<double> tus;
    for (int i = 0; i < N_RUNS; ++i) {
        auto t0 = Clock::now();
        for (int j = 0; j < N_BATCH; ++j)
            ACLRT_LAUNCH_KERNEL(llr_assemble_kernel)(blockDim, stream,
                (uint8_t*)dIn, (uint8_t*)dOut, (uint8_t*)dWs, (uint8_t*)dTil);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = Clock::now();
        tus.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / N_BATCH);
    }
    std::sort(tus.begin(), tus.end());
    printf("\n=== Results ===\n  verify: %s\n  [bench] min %.2f us / median %.2f us\n",
           ok ? "PASS" : "FAIL", tus.front(), tus[tus.size() / 2]);

    CHECK_ACL(aclrtFree(dIn));  CHECK_ACL(aclrtFree(dOut));
    CHECK_ACL(aclrtFree(dWs));  CHECK_ACL(aclrtFree(dTil));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0)); CHECK_ACL(aclFinalize());
    return ok ? 0 : 1;
#else
    return 0;
#endif
}
