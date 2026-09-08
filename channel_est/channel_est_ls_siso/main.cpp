













#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <sys/stat.h>

#include "acl/acl.h"
#include "aclrtlaunch_channel_est_ls_kernel.h"
#include "data_utils.h"

extern "C" void GenerateTiling(const char* socVersion, uint8_t* buf);



constexpr uint32_t N_SC_RAW          = 1596;
constexpr uint32_t N_SC_USED         = 1664;
constexpr uint32_t N_SYMBOL          = 14;
constexpr uint32_t N_BLOCKS          = 4;
constexpr uint32_t N_DMRS_SYM        = 2;
constexpr uint32_t N_DMRS_RE_RAW     = 798;
constexpr uint32_t N_DMRS_RE_PADDED  = 832;

constexpr uint32_t DMRS_RE_WRITE_PAIR  = 208;
constexpr uint32_t WEAVE_IDX_COUNT     = 2 * DMRS_RE_WRITE_PAIR;


constexpr size_t INPUT_BYTES_RAW   = N_SYMBOL * N_SC_RAW * 2 * sizeof(int16_t);
constexpr size_t PILOT_BYTES_RAW   = N_DMRS_SYM * N_DMRS_RE_RAW * 2 * sizeof(int16_t);


constexpr uint32_t Y_HALF_LEN      = N_SYMBOL * N_SC_USED;
constexpr uint32_t X_HALF_LEN      = N_DMRS_SYM * N_DMRS_RE_PADDED;
constexpr uint32_t Y_SAFETY_PAD    = 256;
constexpr uint32_t X_SAFETY_PAD    = 256;
constexpr size_t   Y_BYTES         = (Y_HALF_LEN + Y_SAFETY_PAD) * sizeof(uint16_t);
constexpr size_t   X_BYTES         = (X_HALF_LEN + X_SAFETY_PAD) * sizeof(uint16_t);
constexpr size_t   OUT_H_BYTES     = Y_HALF_LEN * sizeof(uint16_t);
constexpr size_t   OUT_ERRVAR_BYTES = Y_HALF_LEN * sizeof(uint16_t);
constexpr size_t   WEAVE_IDX_BYTES = WEAVE_IDX_COUNT * sizeof(uint32_t);

constexpr size_t TILING_TOTAL_SIZE = 128;
constexpr int    Q_BITS            = 14;
constexpr float  Q_SCALE_INV       = 1.0f / (float)(1 << Q_BITS);



struct TestCase {
    const char* dir_name;
};
static const TestCase TEST_CASES[] = {
    {"case_0_awgn_only"},
    {"case_1_flat_fading"},
    {"case_2_freq_selective"},
    {"case_3_freq_time_var"},
    {"case_4_low_snr"},
};
constexpr int N_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);


static std::string ProjectRoot()
{
    if (const char *env = std::getenv("AIRAN_DATA_DIR")) return std::string(env);
    return std::string(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") + "/AI-RAN-NPU/kernels/rx/channel_est_ls";
}
static std::string GoldenRoot()    { return ProjectRoot() + "/data/golden"; }
static std::string AscendOutRoot() { return ProjectRoot() + "/data/ascend_output"; }
static std::string CaseInputPath(int ci)  { return GoldenRoot() + "/" + TEST_CASES[ci].dir_name + "/input.bin"; }
static std::string CasePilotPath(int ci)  { return GoldenRoot() + "/" + TEST_CASES[ci].dir_name + "/pilot.bin"; }
static std::string CaseEvarPath(int ci)   { return GoldenRoot() + "/" + TEST_CASES[ci].dir_name + "/truth_err_var.bin"; }
static std::string AscendOutDir(int ci)   { return AscendOutRoot() + "/" + TEST_CASES[ci].dir_name; }

static inline void MakeDir(const std::string& path)
{
    std::string cmd = "mkdir -p " + path; (void)std::system(cmd.c_str());
}




static inline uint16_t FloatToHalf(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000);
    int32_t  expn = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;

    if (expn >= 31) {

        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    } else if (expn <= 0) {

        if (expn < -10) return sign;

        mant |= 0x00800000;
        uint32_t shift = (uint32_t)(14 - expn);
        uint32_t rounded = (mant + (1u << (shift - 1))) >> shift;
        return (uint16_t)(sign | rounded);
    } else {

        uint32_t round_bias = 0x1000 + ((mant >> 13) & 0x1);
        uint32_t rounded = (mant + round_bias) >> 13;
        if (rounded & 0x400) {

            rounded = 0;
            expn += 1;
            if (expn >= 31) return (uint16_t)(sign | 0x7C00);
        }
        return (uint16_t)(sign | (uint16_t)(expn << 10) | (uint16_t)(rounded & 0x3FF));
    }
}





static void ConvertCint16ToFp16Separate(
    const int16_t* src_cint16, uint32_t n_rows, uint32_t m_raw,
    uint16_t* dst_re_fp16, uint16_t* dst_im_fp16, uint32_t m_padded)
{
    for (uint32_t r = 0; r < n_rows; ++r) {
        const int16_t* row = src_cint16 + r * m_raw * 2;
        uint16_t* dst_re = dst_re_fp16 + r * m_padded;
        uint16_t* dst_im = dst_im_fp16 + r * m_padded;
        for (uint32_t k = 0; k < m_raw; ++k) {
            float re_f = (float)row[2*k + 0] * Q_SCALE_INV;
            float im_f = (float)row[2*k + 1] * Q_SCALE_INV;
            dst_re[k] = FloatToHalf(re_f);
            dst_im[k] = FloatToHalf(im_f);
        }

    }
}












static void GenerateWeaveIdx(uint32_t* weave_idx, uint32_t N)
{
    const uint32_t SZ = sizeof(uint16_t);
    for (uint32_t i = 0; i < N; ++i) {
        weave_idx[2*i]     = i * SZ;
        weave_idx[2*i + 1] = (N + i) * SZ;
    }
}


int32_t main(int32_t  , char*  [])
{
#ifdef ASCENDC_CPU_DEBUG
    fprintf(stderr, "[v6.2] CPU debug mode not yet ported to new signature. Use NPU build.\n");
    return 1;
#else
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    CHECK_ACL(aclrtSetDevice(0));
    CHECK_ACL(aclrtCreateContext(&context, 0));
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint32_t blockDim = N_BLOCKS;

    uint8_t  tilingBuf[TILING_TOTAL_SIZE];
    GenerateTiling("Ascend310P3", tilingBuf);
    size_t   tilingBytes = TILING_TOTAL_SIZE;
    size_t   yBytes      = Y_BYTES;
    size_t   xBytes      = X_BYTES;
    size_t   outHBytes   = OUT_H_BYTES;
    size_t   outEvBytes  = OUT_ERRVAR_BYTES;
    size_t   weaveBytes  = WEAVE_IDX_BYTES;
    size_t   wsBytes     = 4096;

    printf("[channel_est_ls v6.2] grid [14, 1664] padded, fp16 separate IO + natural-order Gather\n");
    printf("  Y per stream %zu B, X per stream %zu B, weave_idx %zu B, H per stream %zu B\n",
           yBytes, xBytes, weaveBytes, outHBytes);


    uint8_t *yReH, *yImH, *xReH, *xImH;
    uint8_t *yReD, *yImD, *xReD, *xImD;
    uint8_t *oHReh, *oHRed, *oHImh, *oHImd;
    uint8_t *weaveH, *weaveD;
    uint8_t *wsD;
    uint8_t *tilH, *tilD;

    CHECK_ACL(aclrtMallocHost((void **)&yReH, yBytes));
    CHECK_ACL(aclrtMallocHost((void **)&yImH, yBytes));
    CHECK_ACL(aclrtMallocHost((void **)&xReH, xBytes));
    CHECK_ACL(aclrtMallocHost((void **)&xImH, xBytes));
    CHECK_ACL(aclrtMalloc((void **)&yReD, yBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yImD, yBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&xReD, xBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&xImD, xBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&oHReh, outHBytes));
    CHECK_ACL(aclrtMallocHost((void **)&oHImh, outHBytes));
    CHECK_ACL(aclrtMalloc((void **)&oHRed, outHBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&oHImd, outHBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&weaveH, weaveBytes));
    CHECK_ACL(aclrtMalloc((void **)&weaveD, weaveBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsD, wsBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&tilH, tilingBytes));
    CHECK_ACL(aclrtMalloc((void **)&tilD, tilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::memcpy(tilH, tilingBuf, tilingBytes);
    CHECK_ACL(aclrtMemcpy(tilD, tilingBytes, tilH, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));


    GenerateWeaveIdx((uint32_t*)weaveH, DMRS_RE_WRITE_PAIR);
    CHECK_ACL(aclrtMemcpy(weaveD, weaveBytes, weaveH, weaveBytes, ACL_MEMCPY_HOST_TO_DEVICE));


    std::vector<int16_t> inputRaw(INPUT_BYTES_RAW / sizeof(int16_t));
    std::vector<int16_t> pilotRaw(PILOT_BYTES_RAW / sizeof(int16_t));

    std::vector<uint8_t> errvarBuf(outEvBytes);

    printf("\n[channel_est_ls v6.2] running %d cases:\n", N_CASES);
    for (int ci = 0; ci < N_CASES; ++ci) {
        size_t bytes_io;


        std::memset(inputRaw.data(), 0, INPUT_BYTES_RAW);
        ReadFile(CaseInputPath(ci).c_str(), bytes_io, inputRaw.data(), INPUT_BYTES_RAW);
        std::memset(yReH, 0, yBytes);
        std::memset(yImH, 0, yBytes);
        ConvertCint16ToFp16Separate(
            inputRaw.data(), N_SYMBOL, N_SC_RAW,
            (uint16_t*)yReH, (uint16_t*)yImH, N_SC_USED);
        CHECK_ACL(aclrtMemcpy(yReD, yBytes, yReH, yBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(yImD, yBytes, yImH, yBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        std::memset(pilotRaw.data(), 0, PILOT_BYTES_RAW);
        ReadFile(CasePilotPath(ci).c_str(), bytes_io, pilotRaw.data(), PILOT_BYTES_RAW);
        std::memset(xReH, 0, xBytes);
        std::memset(xImH, 0, xBytes);
        ConvertCint16ToFp16Separate(
            pilotRaw.data(), N_DMRS_SYM, N_DMRS_RE_RAW,
            (uint16_t*)xReH, (uint16_t*)xImH, N_DMRS_RE_PADDED);
        CHECK_ACL(aclrtMemcpy(xReD, xBytes, xReH, xBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(xImD, xBytes, xImH, xBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        std::memset(oHReh, 0, outHBytes);
        std::memset(oHImh, 0, outHBytes);
        CHECK_ACL(aclrtMemcpy(oHRed, outHBytes, oHReh, outHBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(oHImd, outHBytes, oHImh, outHBytes, ACL_MEMCPY_HOST_TO_DEVICE));


        ACLRT_LAUNCH_KERNEL(channel_est_ls_kernel)
        (blockDim, stream,
         yReD, yImD, xReD, xImD,
         oHRed, oHImd, weaveD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));


        CHECK_ACL(aclrtMemcpy(oHReh, outHBytes, oHRed, outHBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(oHImh, outHBytes, oHImd, outHBytes, ACL_MEMCPY_DEVICE_TO_HOST));


        std::memset(errvarBuf.data(), 0, outEvBytes);
        ReadFile(CaseEvarPath(ci).c_str(), bytes_io, errvarBuf.data(), outEvBytes);


        MakeDir(AscendOutDir(ci));
        WriteFile((AscendOutDir(ci) + "/out_h_re.bin").c_str(), oHReh, outHBytes);
        WriteFile((AscendOutDir(ci) + "/out_h_im.bin").c_str(), oHImh, outHBytes);
        WriteFile((AscendOutDir(ci) + "/out_err_var.bin").c_str(), errvarBuf.data(), outEvBytes);
    }


    const int NUM_WARMUP = 10;
    const int NUM_TIMED  = 50;
    printf("\n[channel_est_ls v6.2] warm-up: %d runs\n", NUM_WARMUP);
    for (int i = 0; i < NUM_WARMUP; ++i) {
        ACLRT_LAUNCH_KERNEL(channel_est_ls_kernel)
        (blockDim, stream, yReD, yImD, xReD, xImD,
         oHRed, oHImd, weaveD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }
    printf("[channel_est_ls v6.2] timed: %d runs\n", NUM_TIMED);
    std::vector<double> times(NUM_TIMED);
    for (int i = 0; i < NUM_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(channel_est_ls_kernel)
        (blockDim, stream, yReD, yImD, xReD, xImD,
         oHRed, oHImd, weaveD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    double minT = times.front();
    double avgT = 0; for (double t : times) avgT += t; avgT /= NUM_TIMED;
    double p50T = times[NUM_TIMED / 2];
    double p99T = times[(int)(NUM_TIMED * 0.99)];
    double maxT = times.back();
    printf("[channel_est_ls v6.2] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
           minT, avgT, p50T, p99T, maxT);
    printf("[channel_est_ls v6.2] real-time headroom (1ms slot): %.1fx\n", 1000.0 / p50T);


    CHECK_ACL(aclrtFreeHost(yReH)); CHECK_ACL(aclrtFreeHost(yImH));
    CHECK_ACL(aclrtFreeHost(xReH)); CHECK_ACL(aclrtFreeHost(xImH));
    CHECK_ACL(aclrtFreeHost(oHReh)); CHECK_ACL(aclrtFreeHost(oHImh));
    CHECK_ACL(aclrtFreeHost(weaveH)); CHECK_ACL(aclrtFreeHost(tilH));
    CHECK_ACL(aclrtFree(yReD)); CHECK_ACL(aclrtFree(yImD));
    CHECK_ACL(aclrtFree(xReD)); CHECK_ACL(aclrtFree(xImD));
    CHECK_ACL(aclrtFree(oHRed)); CHECK_ACL(aclrtFree(oHImd));
    CHECK_ACL(aclrtFree(weaveD)); CHECK_ACL(aclrtFree(wsD)); CHECK_ACL(aclrtFree(tilD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
#endif
}