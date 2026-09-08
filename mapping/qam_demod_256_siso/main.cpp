






#include "qam256_demod.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_qam256_demod_kernel.h"
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
    "case_0_awgn_only",
    "case_1_flat_fading",
    "case_2_freq_select",
    "case_3_low_snr",
    "case_4_high_snr",
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


static bool VerifyCase(const char* name,
                       const std::vector<int16_t>& hOut,
                       const std::vector<int16_t>& hGold,
                       const std::vector<int16_t>& hGoldSionna) {
    auto GOLD_IDX = [](uint32_t b, uint32_t ds, uint32_t sc) {
        return (size_t)b * airan::N_SYM_PAD + (size_t)ds * airan::N_SC_USED + sc;
    };
    auto OUT_IDX = [](uint32_t b, uint32_t ds, uint32_t sc) {
#ifdef QAM256_VECTOR_BATCH3_PAD
        return (size_t)b * airan::N_SYM_PAD + (size_t)ds * 1600u + sc;
#else
        return (size_t)b * airan::N_SYM_PAD + (size_t)ds * airan::N_SC_USED + sc;
#endif
    };
    size_t sentinel = 0;
    for (uint32_t b = 0; b < airan::Q_M; ++b)
        for (uint32_t ds = 0; ds < airan::N_DATA_SYM; ++ds)
            for (uint32_t sc = 0; sc < airan::N_SC_USED; ++sc)
                if ((uint16_t)hOut[OUT_IDX(b,ds,sc)] == 0xAAAA) ++sentinel;

    size_t tail_nonzero = 0;
#ifdef QAM256_VECTOR_BATCH3_PAD
    for (uint32_t b = 0; b < airan::Q_M; ++b)
        for (uint32_t ds = 0; ds < airan::N_DATA_SYM; ++ds)
            for (uint32_t sc = airan::N_SC_USED; sc < 1600u; ++sc)
                if (hOut[(size_t)b * airan::N_SYM_PAD + (size_t)ds * 1600u + sc] != 0)
                    ++tail_nonzero;
#else
    for (uint32_t b = 0; b < airan::Q_M; ++b)
        for (uint32_t i = airan::N_RE_DATA; i < airan::N_SYM_PAD; ++i)
            if (hOut[(size_t)b * airan::N_SYM_PAD + i] != 0) ++tail_nonzero;
#endif

    size_t diff = 0; int max_err = 0; long argmax = -1; long argmaxGold = -1;
    for (uint32_t b = 0; b < airan::Q_M; ++b)
        for (uint32_t ds = 0; ds < airan::N_DATA_SYM; ++ds)
            for (uint32_t sc = 0; sc < airan::N_SC_USED; ++sc) {
                size_t oi = OUT_IDX(b,ds,sc), gi = GOLD_IDX(b,ds,sc);
                int d = std::abs((int)hOut[oi] - (int)hGold[gi]);
                if (d > 0) ++diff;
                if (d > max_err) { max_err = d; argmax = (long)oi; argmaxGold = (long)gi; }
            }

    bool ok = (diff == 0 && sentinel == 0 && tail_nonzero == 0);
    printf("  [%-18s] sentinel=%zu tail_nz=%zu  vs pipeline: diff=%zu max_err=%d",
           name, sentinel, tail_nonzero, diff, max_err);

    if (diff != 0) {
        const char* bn[8] = {"I_b3","I_b2","I_b1","I_b0","Q_b3","Q_b2","Q_b1","Q_b0"};
        long off = argmax % airan::N_SYM_PAD;
        int b = (int)(argmax / airan::N_SYM_PAD);
#ifdef QAM256_VECTOR_BATCH3_PAD
        int ds = (int)(off / 1600), sc = (int)(off % 1600);
#else
        int ds = (int)(off / airan::N_SC_USED), sc = (int)(off % airan::N_SC_USED);
#endif
        printf("\n      worst: stream %d (%s) ds=%d sc=%d  kernel=%d gold=%d",
               b, bn[b], ds, sc, (int)hOut[argmax], (int)hGold[argmaxGold]);
    } else {
        size_t gt2 = 0, gt30 = 0; int mse = 0;
        for (uint32_t b = 0; b < airan::Q_M; ++b)
            for (uint32_t ds = 0; ds < airan::N_DATA_SYM; ++ds)
                for (uint32_t sc = 0; sc < airan::N_SC_USED; ++sc) {
                    size_t oi = OUT_IDX(b,ds,sc), gi = GOLD_IDX(b,ds,sc);
                    int d = std::abs((int)hOut[oi] - (int)hGoldSionna[gi]);
                    if (d > 2)  ++gt2;
                    if (d > 30) ++gt30;
                    if (d > mse) mse = d;
                }
        size_t tot = (size_t)airan::Q_M * airan::N_RE_DATA;
        printf("  |  vs Sionna: max=%d >2:%.1f%% >30:%.2f%%", mse, 100.0*gt2/tot, 100.0*gt30/tot);
    }
    printf("  => %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    const std::string dataDir = GetDataDir();

#ifdef QAM256_HINGE_CUBE
    printf("=== qam256_demod (EXPERIMENT: 16-hinge + Cube, 4 AIC, multi-case) ===\n");
#elif defined(QAM256_VECTOR_FUSED)
    printf("=== qam256_demod (EXPERIMENT: instruction-reduced Vector, 4 AIV, multi-case) ===\n");
#elif defined(QAM256_VECTOR_ABS)
    printf("=== qam256_demod (EXPERIMENT: abs-reuse Vector, 4 AIV, multi-case) ===\n");
#elif defined(QAM256_VECTOR_BATCH3)
    printf("=== qam256_demod (EXPERIMENT: batch-3 Vector, 4 AIV, multi-case) ===\n");
#elif defined(QAM256_VECTOR_BATCH3_PAIRWISE)
    printf("=== qam256_demod (OPTIMIZED: batch-3 pairwise fp16 padded-output, 4 AIV) ===\n");
#elif defined(QAM256_VECTOR_BATCH3_PAD)
    printf("=== qam256_demod (EXPERIMENT: batch-3 padded-output, 4 AIV, multi-case) ===\n");
#else
    printf("=== qam256_demod (V5.1 4-AIV staged compact, multi-case) ===\n");
#endif
    printf("  data_dir : %s\n", dataDir.c_str());
    printf("  Input    : [%u, %u] fp16 × 3\n", airan::N_SYMBOL, airan::N_SC_PAD);
#ifdef QAM256_VECTOR_BATCH3_PAD
    printf("  Output   : [%u,12,1600] int16 padded, first 1596 valid per symbol\n",
           airan::Q_M);
#else
    printf("  Output   : [%u, %u] int16 compact valid [0,%u), tail=0\n",
           airan::Q_M, airan::N_SYM_PAD, airan::N_RE_DATA);
#endif
    printf("  cases    : %d\n", N_CASES);

#ifndef ASCENDC_CPU_DEBUG
    const size_t szIn = airan::IN_BYTES, szOut = airan::OUT_BYTES;
    const size_t szTil = airan::TILING_TOTAL_SIZE, szWs = airan::WS_TOTAL;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *dXre, *dXim, *dNe, *dWs, *dTil, *dOut;
    CHECK_ACL(aclrtMalloc(&dXre, szIn,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dXim, szIn,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dNe,  szIn,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dWs,  szWs,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dTil, szTil, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dOut, szOut, ACL_MEM_MALLOC_HUGE_FIRST));


    CHECK_ACL(aclrtMemset(dWs, szWs, 0, szWs));

    uint8_t* hTil = airan::GenerateTiling("qam256_demod", airan::BLOCK_DIM);
    CHECK_ACL(aclrtMemcpy(dTil, szTil, hTil, szTil, ACL_MEMCPY_HOST_TO_DEVICE));
    std::free(hTil);

    const uint32_t blockDim = airan::BLOCK_DIM;

    std::vector<uint16_t> hXre(airan::N_SYMBOL * airan::N_SC_PAD);
    std::vector<uint16_t> hXim(airan::N_SYMBOL * airan::N_SC_PAD);
    std::vector<uint16_t> hNe (airan::N_SYMBOL * airan::N_SC_PAD);
    std::vector<int16_t>  hGold      (airan::Q_M * airan::N_SYM_PAD);
    std::vector<int16_t>  hGoldSionna(airan::Q_M * airan::N_SYM_PAD);
    std::vector<int16_t>  hOut       (airan::Q_M * airan::N_SYM_PAD);

#ifdef QAM256_VECTOR_BATCH3_PAD
    printf("\n========== per-case verify (12x1600 padded, first 1596 valid) ==========\n");
#else
    printf("\n========== per-case verify (strict contiguous %u, zero tail) ==========\n",
           airan::N_RE_DATA);
#endif
    int n_pass = 0;
    for (int c = 0; c < N_CASES; ++c) {
        const std::string g = dataDir + "/golden/"        + CASE_NAMES[c];
        const std::string o = dataDir + "/ascend_output/" + CASE_NAMES[c];
        if (!ReadBin(g + "/x_re.bin",   hXre.data(), airan::IN_BYTES)) return 1;
        if (!ReadBin(g + "/x_im.bin",   hXim.data(), airan::IN_BYTES)) return 1;
        if (!ReadBin(g + "/no_eff.bin", hNe .data(), airan::IN_BYTES)) return 1;
        if (!ReadBin(g + "/output_llr.bin", hGold.data(), airan::OUT_BYTES)) return 1;
        std::fill(hGoldSionna.begin(), hGoldSionna.end(), (int16_t)0);
        ReadBin(g + "/output_llr_sionna.bin", hGoldSionna.data(), airan::OUT_BYTES);

        CHECK_ACL(aclrtMemcpy(dXre, szIn, hXre.data(), szIn, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(dXim, szIn, hXim.data(), szIn, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(dNe,  szIn, hNe .data(), szIn, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemset(dOut, szOut, 0xAA, szOut));

        ACLRT_LAUNCH_KERNEL(qam256_demod_kernel)(blockDim, stream,
            (uint8_t*)dXre, (uint8_t*)dXim, (uint8_t*)dNe,
            (uint8_t*)dOut, (uint8_t*)dWs, (uint8_t*)dTil);
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) { fprintf(stderr, "[err] sync\n"); return 1; }

        CHECK_ACL(aclrtMemcpy(hOut.data(), szOut, dOut, szOut, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteBin(o + "/output_llr.bin", hOut.data(), szOut);

        if (VerifyCase(CASE_NAMES[c], hOut, hGold, hGoldSionna)) ++n_pass;
    }
    bool all_ok = (n_pass == N_CASES);
    printf("  -------------------------------------------------------------\n");
    printf("  cases passed: %d / %d  => %s\n", n_pass, N_CASES, all_ok ? "ALL PASS" : "FAIL");


    constexpr int N_BATCH = 100, N_WARMUP = 8;
    for (int i = 0; i < N_WARMUP; ++i)
        ACLRT_LAUNCH_KERNEL(qam256_demod_kernel)(blockDim, stream,
            (uint8_t*)dXre,(uint8_t*)dXim,(uint8_t*)dNe,(uint8_t*)dOut,(uint8_t*)dWs,(uint8_t*)dTil);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    std::vector<double> tus;
    for (int i = 0; i < N_BENCH_RUNS; ++i) {
        auto t0 = Clock::now();
        for (int j = 0; j < N_BATCH; ++j)
            ACLRT_LAUNCH_KERNEL(qam256_demod_kernel)(blockDim, stream,
                (uint8_t*)dXre,(uint8_t*)dXim,(uint8_t*)dNe,(uint8_t*)dOut,(uint8_t*)dWs,(uint8_t*)dTil);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = Clock::now();
        tus.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / N_BATCH);
    }
    std::sort(tus.begin(), tus.end());
    printf("\n=== Results ===\n  verify: %s (%d/%d)\n  [bench] min %.2f us / median %.2f us per launch\n",
           all_ok ? "PASS" : "FAIL", n_pass, N_CASES, tus.front(), tus[tus.size()/2]);

    CHECK_ACL(aclrtFree(dXre)); CHECK_ACL(aclrtFree(dXim)); CHECK_ACL(aclrtFree(dNe));
    CHECK_ACL(aclrtFree(dWs));  CHECK_ACL(aclrtFree(dTil)); CHECK_ACL(aclrtFree(dOut));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0)); CHECK_ACL(aclFinalize());
    return all_ok ? 0 : 1;
#else
    return 0;
#endif
}
