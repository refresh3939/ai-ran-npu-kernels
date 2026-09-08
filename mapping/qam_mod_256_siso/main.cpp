







#include "qam256_mod.h"

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
#include "aclrtlaunch_qam256_mod_kernel.h"
#endif

namespace airan {
extern "C" uint8_t* GenerateTiling   (const char*, uint32_t);
extern "C" size_t   GetTilingSize    ();
extern "C" size_t   GetWorkspaceSize ();
}

#define CHECK_ACL(call) do { \
    auto _e = (call); \
    if (_e != ACL_SUCCESS) { \
        fprintf(stderr, "ACL error %d at %s:%d\n", _e, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

constexpr int N_BENCH_RUNS = 10;
using Clock = std::chrono::high_resolution_clock;


static const char* CASE_NAMES[] = {
    "case_0_random_a",
    "case_1_all_zero",
    "case_2_all_one",
    "case_3_exhaustive",
    "case_4_random_b",
};
constexpr int N_CASES = (int)(sizeof(CASE_NAMES) / sizeof(CASE_NAMES[0]));

static std::string GetDataDir() {
    const char* r = std::getenv("AIRAN_DATA_DIR");
    return r ? std::string(r) : std::string("../../data");
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
    if (!f.is_open()) { fprintf(stderr, "[err] write %s failed\n", path.c_str()); return false; }
    f.write((const char*)buf, sz); return true;
}

static uint32_t DataSymToPhys(uint32_t ds) {
    if (ds < 2) return ds;
    if (ds < 10) return ds + 1;
    return ds + 2;
}


static bool VerifyCase(const char* name,
                       const std::vector<uint16_t>& hRe, const std::vector<uint16_t>& hIm,
                       const std::vector<uint16_t>& gRe, const std::vector<uint16_t>& gIm) {
    auto ROW = [](uint32_t phys, uint32_t sc) { return (size_t)phys * airan::N_SC_PAD + sc; };
    bool is_dmrs[airan::N_SYMBOL_MAX] = {false};
    is_dmrs[airan::DMRS_SYM_0] = true; is_dmrs[airan::DMRS_SYM_1] = true;

    size_t sentinel = 0, tail_nz = 0, dmrs_nz = 0;
    size_t diff = 0; int max_err = 0; long argmax = -1;

    for (uint32_t ds = 0; ds < airan::N_DATA_SYM; ++ds) {
        uint32_t phys = DataSymToPhys(ds);
        for (uint32_t sc = 0; sc < airan::N_SC_USED; ++sc) {
            size_t idx = ROW(phys, sc);
            if (hRe[idx] == 0xAAAA || hIm[idx] == 0xAAAA) ++sentinel;
            int dr = (hRe[idx] != gRe[idx]); int di2 = (hIm[idx] != gIm[idx]);
            if (dr || di2) ++diff;
            int e = dr ? 1 : 0; if (e > max_err) { max_err = e; argmax = (long)idx; }
        }
        for (uint32_t sc = airan::N_SC_USED; sc < airan::N_SC_PAD; ++sc) {
            size_t idx = ROW(phys, sc);
            if (hRe[idx] != 0 || hIm[idx] != 0) ++tail_nz;
        }
    }
    for (uint32_t s = 0; s < airan::N_SYMBOL_MAX; ++s) {
        if (!is_dmrs[s]) continue;
        for (uint32_t sc = 0; sc < airan::N_SC_PAD; ++sc) {
            size_t idx = ROW(s, sc);
            if (hRe[idx] != 0 || hIm[idx] != 0) ++dmrs_nz;
        }
    }

    bool ok = (diff == 0 && sentinel == 0 && tail_nz == 0 && dmrs_nz == 0);
    printf("  [%-18s] sentinel=%zu tail_nz=%zu dmrs_nz=%zu  vs golden: diff=%zu",
           name, sentinel, tail_nz, dmrs_nz, diff);
    if (diff != 0 && argmax >= 0) {
        int phys = (int)(argmax / airan::N_SC_PAD), sc = (int)(argmax % airan::N_SC_PAD);
        printf("\n      worst: phys=%d sc=%d  re kernel=0x%04X gold=0x%04X",
               phys, sc, hRe[argmax], gRe[argmax]);
    }
    printf("  => %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    const std::string dataDir = GetDataDir();
    constexpr const char* inputFile = "input_bits_padded.bin";
    constexpr const char* inputLayout = "[8,12,1600] int16 symbol-padded bits";
    constexpr const char* implName = "vector_batch3_padded";

    printf("=== qam256_mod (对称 qam256_demod, multi-case) ===\n");
    printf("  data_dir : %s\n", dataDir.c_str());
    printf("  impl     : %s\n", implName);
    printf("  Input    : %s\n", inputLayout);
    printf("  Output   : [%u, %u] fp16 x_re/x_im (DMRS{%u,%u}/pad=0)\n",
           airan::N_SYMBOL, airan::N_SC_PAD, airan::DMRS_SYM_0, airan::DMRS_SYM_1);
    printf("  cases    : %d\n", N_CASES);

#ifndef ASCENDC_CPU_DEBUG
    const size_t szIn = airan::IN_BYTES, szOut = airan::OUT_BYTES;
    const size_t szTil = airan::TILING_TOTAL_SIZE, szWs = airan::WS_TOTAL;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *dBits, *dXre, *dXim, *dWs, *dTil;
    CHECK_ACL(aclrtMalloc(&dBits, szIn,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dXre,  szOut, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dXim,  szOut, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dWs,   szWs,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dTil,  szTil, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* hTil = airan::GenerateTiling("qam256_mod", airan::BLOCK_DIM);
    CHECK_ACL(aclrtMemcpy(dTil, szTil, hTil, szTil, ACL_MEMCPY_HOST_TO_DEVICE));
    std::free(hTil);

    const uint32_t blockDim = airan::BLOCK_DIM;
    const size_t   grid_elems = (size_t)airan::N_SYMBOL_MAX * airan::N_SC_PAD;

    std::vector<int16_t>  hBits(airan::Q_M * airan::N_SYM_PAD);
    std::vector<uint16_t> hRe(grid_elems), hIm(grid_elems);
    std::vector<uint16_t> gRe(grid_elems), gIm(grid_elems);

    printf("\n========== per-case verify (bit-exact x_re/x_im, zero tail/DMRS) ==========\n");
    int n_pass = 0;
    for (int c = 0; c < N_CASES; ++c) {
        const std::string g = dataDir + "/golden/"        + CASE_NAMES[c];
        const std::string o = dataDir + "/ascend_output/" + CASE_NAMES[c];
        if (!ReadBin(g + "/" + inputFile, hBits.data(), szIn))  return 1;
        if (!ReadBin(g + "/x_re.bin",       gRe.data(),  szOut)) return 1;
        if (!ReadBin(g + "/x_im.bin",       gIm.data(),  szOut)) return 1;

        CHECK_ACL(aclrtMemcpy(dBits, szIn, hBits.data(), szIn, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemset(dXre, szOut, 0xAA, szOut));
        CHECK_ACL(aclrtMemset(dXim, szOut, 0xAA, szOut));

        ACLRT_LAUNCH_KERNEL(qam256_mod_kernel)(blockDim, stream,
            (uint8_t*)dBits, (uint8_t*)dXre, (uint8_t*)dXim,
            (uint8_t*)dWs, (uint8_t*)dTil);
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) { fprintf(stderr, "[err] sync\n"); return 1; }

        CHECK_ACL(aclrtMemcpy(hRe.data(), szOut, dXre, szOut, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(hIm.data(), szOut, dXim, szOut, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteBin(o + "/x_re.bin", hRe.data(), szOut);
        WriteBin(o + "/x_im.bin", hIm.data(), szOut);

        if (VerifyCase(CASE_NAMES[c], hRe, hIm, gRe, gIm)) ++n_pass;
    }
    bool all_ok = (n_pass == N_CASES);
    printf("  -------------------------------------------------------------\n");
    printf("  cases passed: %d / %d  => %s\n", n_pass, N_CASES, all_ok ? "ALL PASS" : "FAIL");


    constexpr int N_BATCH = 100, N_WARMUP = 8;
    for (int i = 0; i < N_WARMUP; ++i)
        ACLRT_LAUNCH_KERNEL(qam256_mod_kernel)(blockDim, stream,
            (uint8_t*)dBits,(uint8_t*)dXre,(uint8_t*)dXim,(uint8_t*)dWs,(uint8_t*)dTil);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    std::vector<double> tus;
    for (int i = 0; i < N_BENCH_RUNS; ++i) {
        auto t0 = Clock::now();
        for (int j = 0; j < N_BATCH; ++j)
            ACLRT_LAUNCH_KERNEL(qam256_mod_kernel)(blockDim, stream,
                (uint8_t*)dBits,(uint8_t*)dXre,(uint8_t*)dXim,(uint8_t*)dWs,(uint8_t*)dTil);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = Clock::now();
        tus.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / N_BATCH);
    }
    std::sort(tus.begin(), tus.end());
    printf("\n=== Results ===\n  verify: %s (%d/%d)\n  [bench] min %.2f us / median %.2f us per launch\n",
           all_ok ? "PASS" : "FAIL", n_pass, N_CASES, tus.front(), tus[tus.size()/2]);

    CHECK_ACL(aclrtFree(dBits)); CHECK_ACL(aclrtFree(dXre)); CHECK_ACL(aclrtFree(dXim));
    CHECK_ACL(aclrtFree(dWs));   CHECK_ACL(aclrtFree(dTil));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0)); CHECK_ACL(aclFinalize());
    return all_ok ? 0 : 1;
#else
    return 0;
#endif
}
