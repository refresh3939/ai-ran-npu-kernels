/**
 * @file main.cpp — batched OFDM demodulator host runner
 *
 * Input is interleaved int16 IQ [B, 30720, 2].  A single persistent device
 * launch produces separated fp16 S4 grids [B, 14, 32, 64].
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ofdm_demod.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_ofdm_demod_batch_kernel.h"
#else
#include "tikicpulib.h"
extern "C" void ofdm_demod_batch_kernel(
    uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, uint8_t *, uint8_t *, uint32_t, uint8_t *, uint8_t *);
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

using namespace ofdm_demod_batch;

namespace {

constexpr int N_WARMUP = 10;
constexpr int N_TIMED = 50;

std::string DataDir()
{
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}

std::string GInput() { return DataDir() + "/data/golden/input.bin"; }
std::string GWeight(const char *name) { return DataDir() + "/weights/" + name; }
std::string GOutRe() { return DataDir() + "/data/ascend_output/output_re.bin"; }
std::string GOutIm() { return DataDir() + "/data/ascend_output/output_im.bin"; }

uint32_t BatchSize()
{
    const char *env = std::getenv("OFDM_BATCH_SIZE");
    if (env == nullptr) return DEFAULT_BATCH_SIZE;
    char *end = nullptr;
    const unsigned long value = std::strtoul(env, &end, 10);
    if (*env == '\0' || *end != '\0' || value == 0 || value > UINT32_MAX) {
        std::fprintf(stderr, "[error] invalid OFDM_BATCH_SIZE=%s\n", env);
        std::exit(2);
    }
    return static_cast<uint32_t>(value);
}

#ifndef ASCENDC_CPU_DEBUG
void LoadInput(const std::string &path, size_t bytes, uint8_t **hostOut, uint8_t **devOut)
{
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(hostOut), bytes));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(devOut), bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile(path.c_str(), bytes, *hostOut, bytes);
    CHECK_ACL(aclrtMemcpy(*devOut, bytes, *hostOut, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}
#endif

}  // namespace

int32_t main(int32_t, char **)
{
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    const uint32_t batchSize = BatchSize();

    const size_t inputBytes = static_cast<size_t>(batchSize) * INPUT_INT16_PER_BATCH * sizeof(int16_t);
    const size_t outputBytes = static_cast<size_t>(batchSize) * OUTPUT_ELEMS_PER_BATCH * sizeof(int16_t);
    const size_t w32Bytes = static_cast<size_t>(P) * P * sizeof(int16_t);
    const size_t w64Bytes = static_cast<size_t>(Q) * Q * sizeof(int16_t);
    const size_t twBytes = static_cast<size_t>(P) * Q * sizeof(int16_t);
    const size_t tilingBytes = 2 * sizeof(TCubeTiling);
    const size_t wsBytes = static_cast<size_t>(plat->GetLibApiWorkSpaceSize());

    auto *tilingBuf = static_cast<uint8_t *>(std::malloc(tilingBytes));
    if (tilingBuf == nullptr) {
        std::fprintf(stderr, "[error] failed to allocate tiling buffer\n");
        return 2;
    }
    GenerateTiling(socVersion, tilingBuf);

    const uint32_t blockDim = BLOCK_DIM;
    std::printf("[ofdm_demod_batch] SOC=%s blockDim=%u batch=%u N_FFT=%u symbols=%u\n",
                socVersion, blockDim, batchSize, N_FFT, N_SYMBOL);
    std::printf("[ofdm_demod_batch] one persistent launch, L1 phase hand-off, no GM scratch\n");

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *inp = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(inputBytes));
    uint8_t *w32r = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(w32Bytes));
    uint8_t *w32i = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(w32Bytes));
    uint8_t *w64r = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(w64Bytes));
    uint8_t *w64i = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(w64Bytes));
    uint8_t *twr = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(twBytes));
    uint8_t *twi = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(twBytes));
    uint8_t *outRe = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(outputBytes));
    uint8_t *outIm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(outputBytes));
    uint8_t *ws = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(wsBytes));
    uint8_t *til = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(tilingBytes));

    ReadFile(GInput().c_str(), inputBytes, inp, inputBytes);
    ReadFile(GWeight("w_dft32_re.bin").c_str(), w32Bytes, w32r, w32Bytes);
    ReadFile(GWeight("w_dft32_im.bin").c_str(), w32Bytes, w32i, w32Bytes);
    ReadFile(GWeight("w_dft64_re_T.bin").c_str(), w64Bytes, w64r, w64Bytes);
    ReadFile(GWeight("w_dft64_im_T.bin").c_str(), w64Bytes, w64i, w64Bytes);
    ReadFile(GWeight("twiddle_pq_re.bin").c_str(), twBytes, twr, twBytes);
    ReadFile(GWeight("twiddle_pq_im.bin").c_str(), twBytes, twi, twBytes);
    std::memcpy(til, tilingBuf, tilingBytes);

    ICPU_RUN_KF(ofdm_demod_batch_kernel, blockDim,
                inp, w32r, w32i, w64r, w64i, twr, twi,
                outRe, outIm, batchSize, ws, til);

    WriteFile(GOutRe().c_str(), outRe, outputBytes);
    WriteFile(GOutIm().c_str(), outIm, outputBytes);
    AscendC::GmFree(inp); AscendC::GmFree(w32r); AscendC::GmFree(w32i);
    AscendC::GmFree(w64r); AscendC::GmFree(w64i); AscendC::GmFree(twr);
    AscendC::GmFree(twi); AscendC::GmFree(outRe);
    AscendC::GmFree(outIm); AscendC::GmFree(ws); AscendC::GmFree(til);
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inpH, *inpD, *w32rH, *w32rD, *w32iH, *w32iD;
    uint8_t *w64rH, *w64rD, *w64iH, *w64iD, *twrH, *twrD, *twiH, *twiD;
    uint8_t *outReH, *outReD, *outImH, *outImD, *wsD, *tilH, *tilD;

    LoadInput(GInput(), inputBytes, &inpH, &inpD);
    LoadInput(GWeight("w_dft32_re.bin"), w32Bytes, &w32rH, &w32rD);
    LoadInput(GWeight("w_dft32_im.bin"), w32Bytes, &w32iH, &w32iD);
    LoadInput(GWeight("w_dft64_re_T.bin"), w64Bytes, &w64rH, &w64rD);
    LoadInput(GWeight("w_dft64_im_T.bin"), w64Bytes, &w64iH, &w64iD);
    LoadInput(GWeight("twiddle_pq_re.bin"), twBytes, &twrH, &twrD);
    LoadInput(GWeight("twiddle_pq_im.bin"), twBytes, &twiH, &twiD);

    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&outReH), outputBytes));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&outImH), outputBytes));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&outReD), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&outImD), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(outReD, outputBytes, 0, outputBytes));
    CHECK_ACL(aclrtMemset(outImD, outputBytes, 0, outputBytes));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&wsD), wsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&tilH), tilingBytes));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&tilD), tilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::memcpy(tilH, tilingBuf, tilingBytes);
    CHECK_ACL(aclrtMemcpy(tilD, tilingBytes, tilH, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    std::printf("[ofdm_demod_batch] warm-up: %d runs\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; ++i) {
        ACLRT_LAUNCH_KERNEL(ofdm_demod_batch_kernel)
        (blockDim, stream, inpD, w32rD, w32iD, w64rD, w64iD, twrD, twiD,
         outReD, outImD, batchSize, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    std::printf("[ofdm_demod_batch] timed: %d runs\n", N_TIMED);
    std::vector<double> timings;
    timings.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        const auto begin = std::chrono::high_resolution_clock::now();
        ACLRT_LAUNCH_KERNEL(ofdm_demod_batch_kernel)
        (blockDim, stream, inpD, w32rD, w32iD, w64rD, w64iD, twrD, twiD,
         outReD, outImD, batchSize, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        const auto end = std::chrono::high_resolution_clock::now();
        timings.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }
    std::vector<double> sorted = timings;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double value : timings) sum += value;
    const double average = sum / timings.size();
    std::printf("[ofdm_demod_batch] latency: min %.1f us, avg %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
                sorted.front(), average, sorted[N_TIMED / 2], sorted[(N_TIMED * 99) / 100], sorted.back());
    std::printf("[ofdm_demod_batch] throughput: %.1f slots/s, p50 %.1f us/slot\n",
                batchSize * 1.0e6 / average, sorted[N_TIMED / 2] / batchSize);

    CHECK_ACL(aclrtMemcpy(outReH, outputBytes, outReD, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(outImH, outputBytes, outImD, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(GOutRe().c_str(), outReH, outputBytes);
    WriteFile(GOutIm().c_str(), outImH, outputBytes);

    CHECK_ACL(aclrtFree(inpD)); CHECK_ACL(aclrtFreeHost(inpH));
    CHECK_ACL(aclrtFree(w32rD)); CHECK_ACL(aclrtFreeHost(w32rH));
    CHECK_ACL(aclrtFree(w32iD)); CHECK_ACL(aclrtFreeHost(w32iH));
    CHECK_ACL(aclrtFree(w64rD)); CHECK_ACL(aclrtFreeHost(w64rH));
    CHECK_ACL(aclrtFree(w64iD)); CHECK_ACL(aclrtFreeHost(w64iH));
    CHECK_ACL(aclrtFree(twrD)); CHECK_ACL(aclrtFreeHost(twrH));
    CHECK_ACL(aclrtFree(twiD)); CHECK_ACL(aclrtFreeHost(twiH));
    CHECK_ACL(aclrtFree(outReD)); CHECK_ACL(aclrtFreeHost(outReH));
    CHECK_ACL(aclrtFree(outImD)); CHECK_ACL(aclrtFreeHost(outImH));
    CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtFree(tilD)); CHECK_ACL(aclrtFreeHost(tilH));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif

    std::free(tilingBuf);
    return 0;
}
