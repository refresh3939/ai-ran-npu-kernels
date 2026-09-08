













#include "ldpc_decode.h"

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
#include "aclrtlaunch_ldpc_decode_kernel.h"
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




static size_t pad16(size_t e) { return ((e + 15) / 16) * 16; }

static const char* GetDataRoot() {
    const char* r = std::getenv("AIRAN_DATA_DIR");
    return r ? r : "../../data/snr_5db";
}

static const char* GetWeightRoot() {
    const char* r = std::getenv("AIRAN_WEIGHT_DIR");
    return r ? r : "../../data/weights/ldpc_bg1_z384_shifts";
}

static const char* GetOutputRoot() {
    const char* r = std::getenv("AIRAN_OUTPUT_DIR");
    return r ? r : "../../data/ascend_output";
}

static bool ReadBin(const std::string& path, void* buf, size_t expect) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        fprintf(stderr, "[err] open %s failed\n", path.c_str());
        return false;
    }
    size_t sz = (size_t)f.tellg();
    if (sz != expect) {
        fprintf(stderr, "[err] %s size %zu != %zu\n", path.c_str(), sz, expect);
        return false;
    }
    f.seekg(0);
    f.read((char*)buf, sz);
    return true;
}

static bool WriteBin(const std::string& path, const void* buf, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "[err] open %s for write failed\n", path.c_str());
        return false;
    }
    f.write((const char*)buf, (std::streamsize)bytes);
    return f.good();
}

#ifndef ASCENDC_CPU_DEBUG

static void* AclMalloc(size_t bytes) {
    void* p = nullptr;
    CHECK_ACL(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return p;
}

static void AclMemcpyHtoD(void* dst, const void* src, size_t bytes) {
    CHECK_ACL(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}

static void AclMemcpyDtoH(void* dst, const void* src, size_t bytes) {
    CHECK_ACL(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
}
#endif




int main() {
    const std::string root = GetDataRoot();
    const std::string weightRoot = GetWeightRoot();
    const std::string outputRoot = GetOutputRoot();
    bool correctnessOk = false;

    const std::string pLamIn   = root + "/lam_in.bin";
    const std::string pBits    = root + "/info_bits.bin";
    const std::string pShift   = weightRoot + "/shift_table.bin";
    const std::string pDeg     = root + "/degrees.bin";
    const std::string pEOff    = root + "/edge_offsets.bin";
    const std::string pKBits   = outputRoot + "/decoded_bits.bin";
    const std::string pKLam    = outputRoot + "/lam_out.bin";

    printf("=== ldpc_decode S5+S6 (iter loop + HardDecide) ===\n");
    printf("  NFULL=%u  MB=%u  MAX_DEG=%u  TOTAL_EDGES=%u  Z=%u  MAX_ITER=%u\n",
           airan::LDPC_NFULL, airan::LDPC_MB, airan::LDPC_MAX_DEG,
           airan::LDPC_TOTAL_EDGES, airan::LDPC_Z, airan::LDPC_MAX_ITER);

    const size_t lam_bytes  = airan::LAM_BYTES;
    const size_t prev_bytes = airan::PREV_BYTES;
    const size_t bits_bytes = airan::BITS_BYTES;




    std::vector<int16_t> hLamIn  (airan::LDPC_C_NUM * airan::LAM_ELEMS_PER_CB);
    std::vector<int8_t>  hBitsGd (airan::LDPC_C_NUM * airan::LDPC_K);
    std::vector<int16_t> hShift  (airan::SHIFT_ELEMS);
    std::vector<int16_t> hDeg    (airan::LDPC_MB);
    std::vector<int32_t> hEOff   (airan::LDPC_MB + 1);

    if (!ReadBin(pLamIn,   hLamIn  .data(), lam_bytes))                 return 1;
    if (!ReadBin(pBits,    hBitsGd .data(), bits_bytes))                return 1;
    if (!ReadBin(pShift,   hShift  .data(), airan::SHIFT_ELEMS * 2))    return 1;
    if (!ReadBin(pDeg,     hDeg    .data(), airan::LDPC_MB * 2))        return 1;
    if (!ReadBin(pEOff,    hEOff   .data(), (airan::LDPC_MB + 1) * 4))  return 1;




    std::vector<int16_t> hPackedBc(airan::PACKED_ELEMS_TOTAL, 0);
    std::vector<int16_t> hPackedS (airan::PACKED_ELEMS_TOTAL, 0);
    for (uint32_t br = 0; br < airan::LDPC_MB; ++br) {
        uint32_t k = 0;
        for (uint32_t bc_n = 0; bc_n < airan::LDPC_NFULL; ++bc_n) {
            int16_t s = hShift[br * airan::LDPC_NFULL + bc_n];
            if (s < 0) continue;
            hPackedBc[br * airan::LDPC_MAX_DEG + k] = (int16_t)bc_n;
            hPackedS [br * airan::LDPC_MAX_DEG + k] = s;
            ++k;
        }
    }

    printf("  GM: lam=%zu MB  prev=%zu MB  bits=%zu KB\n",
           lam_bytes / (1024 * 1024), prev_bytes / (1024 * 1024),
           bits_bytes / 1024);

    printf("  data:    %s\n", root.c_str());
    printf("  weights: %s\n", weightRoot.c_str());
    printf("  output:  %s\n", outputRoot.c_str());

#ifndef ASCENDC_CPU_DEBUG
    const size_t szLam       = lam_bytes;
    const size_t szPrev      = prev_bytes;
    const size_t szBits      = bits_bytes;
    const size_t szPackedPad = pad16(airan::PACKED_ELEMS_TOTAL) * sizeof(int16_t);
    const size_t szDegPad    = pad16(airan::LDPC_MB)        * sizeof(int16_t);
    const size_t szEOffPad   = pad16(airan::LDPC_MB + 1)    * sizeof(int32_t);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* dLamIn      = AclMalloc(szLam);
    void* dPackedBc   = AclMalloc(szPackedPad);
    void* dPackedS    = AclMalloc(szPackedPad);
    void* dDeg        = AclMalloc(szDegPad);
    void* dEOff       = AclMalloc(szEOffPad);
    void* dPrev       = AclMalloc(szPrev);
    void* dBits       = AclMalloc(szBits);
    void* dLamOut     = AclMalloc(szLam);

    void* dLamScratch = AclMalloc(airan::LAM_SCRATCH_BYTES);
    CHECK_ACL(aclrtMemset(dLamScratch, airan::LAM_SCRATCH_BYTES, 0, airan::LAM_SCRATCH_BYTES));

    CHECK_ACL(aclrtMemset(dPrev,     szPrev,      0,    szPrev));
    CHECK_ACL(aclrtMemset(dPackedBc, szPackedPad, 0,    szPackedPad));
    CHECK_ACL(aclrtMemset(dPackedS,  szPackedPad, 0,    szPackedPad));
    CHECK_ACL(aclrtMemset(dDeg,      szDegPad,    0,    szDegPad));
    CHECK_ACL(aclrtMemset(dEOff,     szEOffPad,   0,    szEOffPad));
    CHECK_ACL(aclrtMemset(dBits,     szBits,      0xFF, szBits));

    AclMemcpyHtoD(dLamIn,    hLamIn   .data(),  szLam);
    AclMemcpyHtoD(dPackedBc, hPackedBc.data(),  airan::PACKED_BYTES);
    AclMemcpyHtoD(dPackedS,  hPackedS .data(),  airan::PACKED_BYTES);
    AclMemcpyHtoD(dDeg,      hDeg     .data(),  airan::LDPC_MB * 2);
    AclMemcpyHtoD(dEOff,     hEOff    .data(),  (airan::LDPC_MB + 1) * 4);

    {
        std::vector<int16_t> init_pat(airan::LDPC_C_NUM * airan::LAM_ELEMS_PER_CB,
                                      (int16_t)0xAAAA);
        AclMemcpyHtoD(dLamOut, init_pat.data(), szLam);
    }




    const uint32_t blockDim = 4;
    printf("\n[launch] blockDim=%u...\n", blockDim);

    ACLRT_LAUNCH_KERNEL(ldpc_decode_kernel)(blockDim, stream,
        (uint8_t*)dLamIn,
        (uint8_t*)dPackedBc, (uint8_t*)dPackedS,
        (uint8_t*)dDeg, (uint8_t*)dEOff,
        (uint8_t*)dPrev,
        (uint8_t*)dBits,
        (uint8_t*)dLamOut,
        (uint8_t*)dLamScratch);

    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        fprintf(stderr, "[err] sync after launch failed\n");
        return 1;
    }

    std::vector<int16_t> hLamOut(airan::LDPC_C_NUM * airan::LAM_ELEMS_PER_CB);
    std::vector<int8_t>  hBits  (airan::LDPC_C_NUM * airan::LDPC_K);
    AclMemcpyDtoH(hLamOut.data(), dLamOut, szLam);
    AclMemcpyDtoH(hBits  .data(), dBits,   szBits);


    if (WriteBin(pKBits, hBits.data(), hBits.size())) {
        printf("[dump] kernel decoded bits → %s (%zu bytes)\n",
               pKBits.c_str(), hBits.size());
    }
    if (WriteBin(pKLam, hLamOut.data(), hLamOut.size() * sizeof(int16_t))) {
        printf("[dump] kernel posterior LLR → %s (%zu bytes)\n",
               pKLam.c_str(), hLamOut.size() * sizeof(int16_t));
    }




    {
        constexpr double BIT_ERR_TOL = 0.0005;

        printf("\n[debug] cb=0 first 32 bits:\n  kernel:");
        for (int i = 0; i < 32; ++i) printf(" %d", (int)hBits[i]);
        printf("\n  gold  :");
        for (int i = 0; i < 32; ++i) printf(" %d", (int)hBitsGd[i]);
        printf("\n  lam_out[0..7]: ");
        for (int i = 0; i < 8; ++i) printf("%d ", hLamOut[i]);
        printf("\n");

        size_t bits_diff = 0, cb_fail = 0;
        int first_cb_fail = -1;
        for (uint32_t cb = 0; cb < airan::LDPC_C_NUM; ++cb) {
            size_t cb_diff = 0;
            for (uint32_t i = 0; i < airan::LDPC_K; ++i) {
                size_t idx = (size_t)cb * airan::LDPC_K + i;
                if (hBits[idx] != hBitsGd[idx]) { ++cb_diff; ++bits_diff; }
            }
            if (cb_diff > 0) {
                ++cb_fail;
                if (first_cb_fail < 0) first_cb_fail = (int)cb;
            }
        }
        double ber = (double)bits_diff
                   / ((double)airan::LDPC_C_NUM * airan::LDPC_K);
        printf("\n[verify/bits] total_bit_diff=%zu  cb_fail=%zu/%u",
               bits_diff, cb_fail, airan::LDPC_C_NUM);
        if (first_cb_fail >= 0) printf("  first_cb_fail=%d", first_cb_fail);
        printf("\n[verify/bits] BER=%.4f%%  tol=%.2f%%\n",
               ber * 100.0, BIT_ERR_TOL * 100.0);
        printf("[verify/bits] %s\n",
               (ber > BIT_ERR_TOL) ? "FAIL (BER exceeds tolerance)"
                                   : "PASS (BER within tolerance)");
        correctnessOk = ber <= BIT_ERR_TOL;
    }




    printf("\n[bench] %d iterations (3 launches per sync)...\n", N_BENCH_RUNS);

    auto launch_kernel = [&]() {
        ACLRT_LAUNCH_KERNEL(ldpc_decode_kernel)(blockDim, stream,
            (uint8_t*)dLamIn,
            (uint8_t*)dPackedBc, (uint8_t*)dPackedS,
            (uint8_t*)dDeg, (uint8_t*)dEOff,
            (uint8_t*)dPrev,  (uint8_t*)dBits,
            (uint8_t*)dLamOut, (uint8_t*)dLamScratch);
    };

    for (int i = 0; i < 2; ++i) {
        CHECK_ACL(aclrtMemset(dPrev, szPrev, 0, szPrev));
        launch_kernel();
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<double> times_us;
    times_us.reserve(N_BENCH_RUNS);
    for (int i = 0; i < N_BENCH_RUNS; ++i) {
        auto t0 = Clock::now();
        for (int j = 0; j < 3; ++j) {
            CHECK_ACL(aclrtMemset(dPrev, szPrev, 0, szPrev));
            launch_kernel();
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = Clock::now();
        times_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count() / 3.0);
    }
    std::sort(times_us.begin(), times_us.end());
    const double median = times_us[times_us.size() / 2];

    printf("\n=== Results ===\n");
    printf("  kernel median: %.1f us (%.3f us/CB)\n",
           median, median / airan::LDPC_C_NUM);
    printf("  per-frame (worst MAX_ITER=%u): %.2f ms\n",
           airan::LDPC_MAX_ITER, median / 1000.0);

    CHECK_ACL(aclrtFree(dLamIn));
    CHECK_ACL(aclrtFree(dPackedBc));
    CHECK_ACL(aclrtFree(dPackedS));
    CHECK_ACL(aclrtFree(dDeg));
    CHECK_ACL(aclrtFree(dEOff));
    CHECK_ACL(aclrtFree(dPrev));
    CHECK_ACL(aclrtFree(dBits));
    CHECK_ACL(aclrtFree(dLamOut));
    CHECK_ACL(aclrtFree(dLamScratch));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif
    return correctnessOk ? 0 : 1;
}
