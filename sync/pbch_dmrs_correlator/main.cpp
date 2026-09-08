


















#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "data_utils.h"
#include "pbch_dmrs_correlator.h"
#include "pbch_dmrs_tiling.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_pbch_dmrs_correlator_kernel.h"
#endif

namespace {

constexpr int N_WARMUP = 5;
constexpr int N_TIMED  = 20;





std::string DataDir()
{
    const char *env = std::getenv("PBCH_DMRS_DATA_ROOT");
    return env ? std::string(env) : std::string("../..");
}

std::string CaseDir()
{
    const char *env = std::getenv("PBCH_DMRS_CASE_DIR");
    std::string cd = env ? std::string(env) : std::string("case_0_clean_L4_i0");
    return DataDir() + "/data/golden/" + cd;
}

std::string OutputDir()
{
    return DataDir() + "/data/ascend_output";
}


int64_t ParseJsonInt(const std::string &path, const std::string &field, int64_t defv)
{
    std::ifstream f(path);
    if (!f.is_open()) return defv;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    std::string key = "\"" + field + "\"";
    size_t pos = s.find(key);
    if (pos == std::string::npos) return defv;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return defv;
    pos++;
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
    return std::strtoll(s.c_str() + pos, nullptr, 10);
}

double ParseJsonFloat(const std::string &path, const std::string &field, double defv)
{
    std::ifstream f(path);
    if (!f.is_open()) return defv;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    std::string key = "\"" + field + "\"";
    size_t pos = s.find(key);
    if (pos == std::string::npos) return defv;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return defv;
    pos++;
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
    return std::strtod(s.c_str() + pos, nullptr);
}

#ifndef ASCENDC_CPU_DEBUG

bool LoadBinDevice(const std::string &path, size_t bytes, uint8_t **devOut)
{
    std::vector<uint8_t> host(bytes);
    size_t got = 0;
    if (!ReadFile(path, got, host.data(), bytes)) {
        ERROR_LOG("failed to read %s", path.c_str());
        return false;
    }
    if (got != bytes) {
        ERROR_LOG("size mismatch on %s: got %zu, expected %zu", path.c_str(), got, bytes);
        return false;
    }
    uint8_t *devPtr = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(devPtr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    *devOut = devPtr;
    return true;
}
#endif

}


int32_t main(int32_t  , char *  [])
{
    std::string caseDir   = CaseDir();
    std::string outputDir = OutputDir();
    (void)outputDir;

    printf("[pbch_dmrs_correlator] case_dir = %s\n", caseDir.c_str());


    std::string truthPath = caseDir + "/golden/truth.json";
    int64_t truth_i  = ParseJsonInt(truthPath, "true_i_ssb",   -1);
    int64_t truth_p  = ParseJsonInt(truthPath, "pcid",          -1);
    int64_t truth_L  = ParseJsonInt(truthPath, "l_max",         -1);
    double  fp16_p   = ParseJsonFloat(truthPath, "fp16_peak",   0.0);
    double  fp16_s   = ParseJsonFloat(truthPath, "fp16_second", 0.0);
    int64_t fp16_i   = ParseJsonInt(truthPath, "fp16_i_ssb",   -1);
    printf("[pbch_dmrs_correlator] truth: pcid=%ld true_i=%ld L_max=%ld  "
           "fp16_ref: i=%ld peak=%.3f second=%.3f\n",
           truth_p, truth_i, truth_L, fp16_i, fp16_p, fp16_s);

    const uint32_t blockDim = 4u;


    const size_t r_re_bytes      = IN_R_RE_BYTES;
    const size_t r_im_bytes      = IN_R_IM_BYTES;
    const size_t d_re_bytes      = IN_D_RE_BYTES;
    const size_t d_im_bytes      = IN_D_IM_BYTES;
    const size_t d_im_neg_bytes  = IN_D_IM_NEG_BYTES;
    const size_t outputBytes     = OUT_BYTES;
    const size_t tilingBytes     = sizeof(PbchDmrsTilingV1);
    const size_t wsBytes         = 1024;

    printf("[pbch_dmrs_correlator] R: %zu+%zu B  D: %zu+%zu+%zu B  out: %zu B  tiling: %zu B\n",
           r_re_bytes, r_im_bytes, d_re_bytes, d_im_bytes, d_im_neg_bytes,
           outputBytes, tilingBytes);

#ifdef ASCENDC_CPU_DEBUG
    ERROR_LOG("CPU debug mode not implemented");
    return 1;
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));


    uint8_t *rReD = nullptr, *rImD = nullptr;
    uint8_t *dReD = nullptr, *dImD = nullptr, *dImNegD = nullptr;
    if (!LoadBinDevice(caseDir + "/inputs/r_re.bin",     r_re_bytes,     &rReD))    return 1;
    if (!LoadBinDevice(caseDir + "/inputs/r_im.bin",     r_im_bytes,     &rImD))    return 1;
    if (!LoadBinDevice(caseDir + "/inputs/d_re.bin",     d_re_bytes,     &dReD))    return 1;
    if (!LoadBinDevice(caseDir + "/inputs/d_im.bin",     d_im_bytes,     &dImD))    return 1;
    if (!LoadBinDevice(caseDir + "/inputs/d_im_neg.bin", d_im_neg_bytes, &dImNegD)) return 1;


    uint8_t *tilD = nullptr;
    if (!LoadBinDevice(caseDir + "/inputs/tiling.bin", tilingBytes, &tilD)) return 1;


    uint8_t *outH = nullptr, *outD = nullptr;
    CHECK_ACL(aclrtMallocHost((void **)&outH, outputBytes));
    CHECK_ACL(aclrtMalloc((void **)&outD, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    memset(outH, 0xAA, outputBytes);
    CHECK_ACL(aclrtMemcpy(outD, outputBytes, outH, outputBytes, ACL_MEMCPY_HOST_TO_DEVICE));


    uint8_t *wsD = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&wsD, wsBytes, ACL_MEM_MALLOC_HUGE_FIRST));


    printf("[pbch_dmrs_correlator] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        ACLRT_LAUNCH_KERNEL(pbch_dmrs_correlator_kernel)
            (blockDim, stream,
             rReD, rImD, dReD, dImD, dImNegD,
             outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }


    printf("[pbch_dmrs_correlator] timed:   %d runs\n", N_TIMED);
    std::vector<double> us_list;
    us_list.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(pbch_dmrs_correlator_kernel)
            (blockDim, stream,
             rReD, rImD, dReD, dImD, dImNegD,
             outD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        us_list.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us_list.begin(), us_list.end());
    double min_us = us_list.front();
    double p50    = us_list[N_TIMED / 2];
    double p99    = us_list[(N_TIMED * 99) / 100];
    double max_us = us_list.back();
    double sum = 0.0; for (double v : us_list) sum += v;
    double avg_us = sum / N_TIMED;
    printf("[pbch_dmrs_correlator] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           min_us, avg_us, p50, p99, max_us);


    CHECK_ACL(aclrtMemcpy(outH, outputBytes, outD, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    float *outF = reinterpret_cast<float *>(outH);

    printf("[pbch_dmrs_correlator] output (24 fp32):\n");
    printf("  [ 0] i_ssb           = %.1f       (truth %ld, fp16_ref %ld)\n",
           outF[0], truth_i, fp16_i);
    printf("  [ 1] peak            = %.3f       (fp16_ref %.3f)\n", outF[1], fp16_p);
    printf("  [ 2] second          = %.3f       (fp16_ref %.3f)\n", outF[2], fp16_s);
    printf("  [ 3] l_max (echo)    = %.1f       (truth %ld)\n", outF[3], truth_L);
    printf("  [ 4..11] corr_re[0..7] = ");
    for (int i = 0; i < 8; ++i) printf("%.2f ", outF[4 + i]);
    printf("\n");
    printf("  [12..19] corr_im[0..7] = ");
    for (int i = 0; i < 8; ++i) printf("%.2f ", outF[12 + i]);
    printf("\n");
    printf("  [20] dbg_R_re[0]     = %.4f\n", outF[20]);
    printf("  [21] dbg_D_re[0,0]   = %.4f\n", outF[21]);
    printf("  [22] metric_max      = %.3f\n", outF[22]);
    printf("  [23] sentinel        = %.1f  %s\n", outF[23],
           (outF[23] == SENTINEL_F32) ? "✓ KERNEL_RAN" : "✗ STALE!");

    bool pass = (static_cast<int64_t>(outF[0]) == truth_i)
             && (outF[23] == SENTINEL_F32);
    printf("[pbch_dmrs_correlator] argmax %s   (npu=%.0f, truth=%ld)\n",
           pass ? "PASS ✓" : "FAIL ✗",
           outF[0], truth_i);


    std::string outPath = outputDir + "/output.bin";
    WriteFile(outPath.c_str(), outH, outputBytes);
    printf("[pbch_dmrs_correlator] wrote %s\n", outPath.c_str());


    CHECK_ACL(aclrtFree(rReD)); CHECK_ACL(aclrtFree(rImD));
    CHECK_ACL(aclrtFree(dReD)); CHECK_ACL(aclrtFree(dImD)); CHECK_ACL(aclrtFree(dImNegD));
    CHECK_ACL(aclrtFree(tilD));
    CHECK_ACL(aclrtFree(outD)); CHECK_ACL(aclrtFreeHost(outH));
    CHECK_ACL(aclrtFree(wsD));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());

    return pass ? 0 : 1;
#endif
}
