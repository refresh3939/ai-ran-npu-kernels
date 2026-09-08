



#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "re_demap_batch.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#else
#include "tikicpulib.h"
extern "C" void re_demap_batch_kernel(
    uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint32_t,
    uint8_t *, uint8_t *);
#endif

namespace rdb = airan::re_demap_batch;

namespace {

constexpr int N_WARMUP = 10;
constexpr int N_TIMED = 50;

std::string DataDir()
{
    const char *path = std::getenv("AIRAN_DATA_DIR");
    return path == nullptr ? std::string(".") : std::string(path);
}

std::string Golden(const char *name) { return DataDir() + "/data/golden/" + name; }
std::string Weight(const char *name) { return DataDir() + "/weights/" + name; }
std::string Output(const char *name) { return DataDir() + "/data/ascend_output/" + name; }

uint32_t BatchSize()
{
    const char *text = std::getenv("RE_DEMAP_BATCH_SIZE");
    if (text == nullptr) return rdb::DEFAULT_BATCH_SIZE;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value == 0 ||
        value > rdb::MAX_RX_ANTENNAS) {
        std::fprintf(stderr, "[error] RE_DEMAP_BATCH_SIZE must be in [1,%u]\n",
                     rdb::MAX_RX_ANTENNAS);
        std::exit(2);
    }
    return static_cast<uint32_t>(value);
}

bool ReadFile(const std::string &path, void *buffer, size_t bytes)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || static_cast<size_t>(file.tellg()) != bytes) {
        std::fprintf(stderr, "[error] invalid input file: %s (expected %zu bytes)\n",
                     path.c_str(), bytes);
        return false;
    }
    file.seekg(0);
    return static_cast<bool>(file.read(static_cast<char *>(buffer), bytes));
}

bool WriteFile(const std::string &path, const void *buffer, size_t bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || !file.write(static_cast<const char *>(buffer), bytes)) {
        std::fprintf(stderr, "[error] failed to write %s\n", path.c_str());
        return false;
    }
    return true;
}

airan::PuschMimoConfig MakeConfig(uint32_t batchSize)
{
    airan::PuschMimoConfig config {};
    config.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = 1;
    config.num_tx_ports = 1;
    config.num_rx_antennas = static_cast<uint16_t>(batchSize);
    config.qm = 8;
    config.num_symbols = rdb::N_SYMBOLS;
    config.fft_size = rdb::N_FFT;
    config.num_rb = rdb::N_RB;
    config.start_symbol = 0;
    config.num_allocated_symbols = rdb::N_SYMBOLS;
    config.used_subcarriers = rdb::N_SC_USED;
    config.padded_subcarriers = rdb::N_SC_PAD;
    config.dmrs_symbol_mask = (1u << 2) | (1u << 11);
    config.dmrs_ports[0] = 1000;
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    config.prg_size_rb = rdb::N_RB;
    return config;
}

#ifndef ASCENDC_CPU_DEBUG
bool CheckAcl(aclError status, const char *expression, int line)
{
    if (status == ACL_ERROR_NONE) return true;
    std::fprintf(stderr, "[acl] line %d: %s failed with %d\n",
                 line, expression, status);
    return false;
}

#define CHECK_ACL(call) \
    do { if (!CheckAcl((call), #call, __LINE__)) return 2; } while (0)

bool LoadDeviceFile(const std::string &path, size_t bytes,
                    uint8_t **host, uint8_t **device)
{
    if (aclrtMallocHost(reinterpret_cast<void **>(host), bytes) != ACL_ERROR_NONE ||
        aclrtMalloc(reinterpret_cast<void **>(device), bytes,
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_ERROR_NONE ||
        !ReadFile(path, *host, bytes) ||
        aclrtMemcpy(*device, bytes, *host, bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
        return false;
    }
    return true;
}
#endif

}

int32_t main(int32_t, char **)
{
    const char *socVersion = SOC_VERSION;
    (void)platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    const uint32_t batchSize = BatchSize();
    const size_t inputBytes = rdb::InputElems(batchSize) * sizeof(uint16_t);
    const size_t outputBytes = rdb::OutputElems(batchSize) * sizeof(uint16_t);
    const size_t indexBytes = rdb::N_SC_PAD * sizeof(uint32_t);

    airan::PuschMimoConfig config = MakeConfig(batchSize);
    airan::PuschMimoLayout layout {};
    rdb::KernelMetadata metadata {};
    if (rdb::BuildCurrentProfile(config, &layout, &metadata) != rdb::OK) {
        std::fprintf(stderr, "[error] failed to build current PUSCH profile\n");
        return 2;
    }

    std::printf("[re_demap_batch] SOC=%s blockDim=%u NR=%u\n",
                socVersion, rdb::BLOCK_DIM, batchSize);
    std::printf("[re_demap_batch] input=[%u,14,32,64], output=[%u,14,1664], padding=zero\n",
                batchSize, batchSize);

#ifdef ASCENDC_CPU_DEBUG
    auto *inputRe = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(inputBytes));
    auto *inputIm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(inputBytes));
    auto *index = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(indexBytes));
    auto *outputRe = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(outputBytes));
    auto *outputIm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(outputBytes));
    auto *workspace = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(rdb::WORKSPACE_BYTES));
    auto *tiling = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(rdb::TILING_BYTES));
    if (!ReadFile(Golden("input_re.bin"), inputRe, inputBytes) ||
        !ReadFile(Golden("input_im.bin"), inputIm, inputBytes) ||
        !ReadFile(Weight("gather_idx.bin"), index, indexBytes)) {
        return 2;
    }
    std::memcpy(tiling, &metadata, sizeof(metadata));
    ICPU_RUN_KF(re_demap_batch_kernel, rdb::BLOCK_DIM,
                inputRe, inputIm, index, outputRe, outputIm, batchSize,
                workspace, tiling);
    if (!WriteFile(Output("rx_grid_re.bin"), outputRe, outputBytes) ||
        !WriteFile(Output("rx_grid_im.bin"), outputIm, outputBytes)) {
        return 2;
    }
    AscendC::GmFree(inputRe); AscendC::GmFree(inputIm); AscendC::GmFree(index);
    AscendC::GmFree(outputRe); AscendC::GmFree(outputIm);
    AscendC::GmFree(workspace); AscendC::GmFree(tiling);
#else
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inputReHost = nullptr, *inputReDevice = nullptr;
    uint8_t *inputImHost = nullptr, *inputImDevice = nullptr;
    uint8_t *indexHost = nullptr, *indexDevice = nullptr;
    if (!LoadDeviceFile(Golden("input_re.bin"), inputBytes,
                        &inputReHost, &inputReDevice) ||
        !LoadDeviceFile(Golden("input_im.bin"), inputBytes,
                        &inputImHost, &inputImDevice) ||
        !LoadDeviceFile(Weight("gather_idx.bin"), indexBytes,
                        &indexHost, &indexDevice)) {
        std::fprintf(stderr, "[error] failed to load device inputs\n");
        return 2;
    }

    uint8_t *outputReHost = nullptr, *outputImHost = nullptr;
    uint8_t *outputReDevice = nullptr, *outputImDevice = nullptr;
    uint8_t *workspace = nullptr, *tilingHost = nullptr, *tilingDevice = nullptr;
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&outputReHost), outputBytes));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&outputImHost), outputBytes));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&outputReDevice), outputBytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&outputImDevice), outputBytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(outputReDevice, outputBytes, 0xa5, outputBytes));
    CHECK_ACL(aclrtMemset(outputImDevice, outputBytes, 0xa5, outputBytes));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&workspace), rdb::WORKSPACE_BYTES,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&tilingHost), rdb::TILING_BYTES));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&tilingDevice), rdb::TILING_BYTES,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    std::memcpy(tilingHost, &metadata, sizeof(metadata));
    CHECK_ACL(aclrtMemcpy(tilingDevice, rdb::TILING_BYTES, tilingHost,
                          rdb::TILING_BYTES, ACL_MEMCPY_HOST_TO_DEVICE));

    rdb::ReDemapBatchOpArgsV1 args {};
    args.abi_version = rdb::ABI_VERSION;
    args.struct_size = sizeof(args);
    args.fft_grid_re = inputReDevice;
    args.fft_grid_im = inputImDevice;
    args.rx_grid_re = outputReDevice;
    args.rx_grid_im = outputImDevice;
    args.config = &config;
    args.layout = &layout;
    args.stream = stream;

    for (int run = 0; run < N_WARMUP; ++run) {
        if (rdb::Enqueue(args, indexDevice, indexBytes, workspace,
                         rdb::WORKSPACE_BYTES, tilingDevice,
                         rdb::TILING_BYTES) != rdb::OK) {
            std::fprintf(stderr, "[error] warm-up launch failed\n");
            return 2;
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    std::vector<double> timings;
    timings.reserve(N_TIMED);
    for (int run = 0; run < N_TIMED; ++run) {
        const auto begin = std::chrono::high_resolution_clock::now();
        if (rdb::Enqueue(args, indexDevice, indexBytes, workspace,
                         rdb::WORKSPACE_BYTES, tilingDevice,
                         rdb::TILING_BYTES) != rdb::OK) {
            std::fprintf(stderr, "[error] timed launch failed\n");
            return 2;
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
        const auto end = std::chrono::high_resolution_clock::now();
        timings.push_back(
            std::chrono::duration<double, std::micro>(end - begin).count());
    }
    std::sort(timings.begin(), timings.end());
    double sum = 0.0;
    for (double value : timings) sum += value;
    std::printf("[re_demap_batch] p50=%.1f us, avg=%.1f us, amortized=%.1f us/antenna\n",
                timings[N_TIMED / 2], sum / N_TIMED,
                timings[N_TIMED / 2] / batchSize);

    CHECK_ACL(aclrtMemcpy(outputReHost, outputBytes, outputReDevice, outputBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(outputImHost, outputBytes, outputImDevice, outputBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteFile(Output("rx_grid_re.bin"), outputReHost, outputBytes) ||
        !WriteFile(Output("rx_grid_im.bin"), outputImHost, outputBytes)) {
        return 2;
    }

    CHECK_ACL(aclrtFree(inputReDevice)); CHECK_ACL(aclrtFreeHost(inputReHost));
    CHECK_ACL(aclrtFree(inputImDevice)); CHECK_ACL(aclrtFreeHost(inputImHost));
    CHECK_ACL(aclrtFree(indexDevice)); CHECK_ACL(aclrtFreeHost(indexHost));
    CHECK_ACL(aclrtFree(outputReDevice)); CHECK_ACL(aclrtFreeHost(outputReHost));
    CHECK_ACL(aclrtFree(outputImDevice)); CHECK_ACL(aclrtFreeHost(outputImHost));
    CHECK_ACL(aclrtFree(workspace));
    CHECK_ACL(aclrtFree(tilingDevice)); CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif

    return 0;
}
