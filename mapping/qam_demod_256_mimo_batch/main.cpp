#include "qam256_demod_batch.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "acl/acl.h"

using namespace airan::qam256_demod_batch;

namespace {

#define ACL_CHECK(expr) do { \
    const aclError status_ = (expr); \
    if (status_ != ACL_ERROR_NONE) { \
        std::fprintf(stderr, "[ACL] %s:%d status=%d\n", __FILE__, __LINE__, status_); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

struct Buffer {
    void *device = nullptr;
    void *host = nullptr;
    size_t bytes = 0;

    void Allocate(size_t count) {
        bytes = count;
        ACL_CHECK(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost(&host, bytes));
    }
    void Load(const std::string &path) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) {
            std::fprintf(stderr, "[FAIL] cannot open %s\n", path.c_str());
            std::exit(EXIT_FAILURE);
        }
        const auto end = input.tellg();
        if (end <= 0) {
            std::fprintf(stderr, "[FAIL] empty input %s\n", path.c_str());
            std::exit(EXIT_FAILURE);
        }
        Allocate(static_cast<size_t>(end));
        input.seekg(0);
        input.read(static_cast<char *>(host), static_cast<std::streamsize>(bytes));
        if (!input) {
            std::fprintf(stderr, "[FAIL] short read %s\n", path.c_str());
            std::exit(EXIT_FAILURE);
        }
        ACL_CHECK(aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    void Save(const std::string &path) {
        ACL_CHECK(aclrtMemcpy(host, bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(static_cast<const char *>(host), static_cast<std::streamsize>(bytes));
        if (!output) {
            std::fprintf(stderr, "[FAIL] cannot write %s\n", path.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    ~Buffer() {
        if (device != nullptr) ACL_CHECK(aclrtFree(device));
        if (host != nullptr) ACL_CHECK(aclrtFreeHost(host));
    }
};

std::string DataRoot()
{
    const char *root = std::getenv("AIRAN_DATA_DIR");
    return root == nullptr ? "data" : root;
}

void EnsureDir(const std::string &path)
{
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        std::perror(path.c_str());
        std::exit(EXIT_FAILURE);
    }
}

int EnvInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    return value == nullptr ? fallback : std::max(0, std::atoi(value));
}

PuschMimoConfig MakeConfig(uint16_t layers)
{
    PuschMimoConfig config {};
    config.abi_version = ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = layers;
    config.num_tx_ports = layers == 3 ? 4 : layers;
    config.num_rx_antennas = 64;
    config.qm = Q_M;
    config.num_symbols = N_SYMBOLS;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = N_SYMBOLS;
    config.used_subcarriers = N_SC_USED;
    config.padded_subcarriers = N_SC_GRID_PAD;
    config.dmrs_symbol_mask = DMRS_SYMBOL_MASK;
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    for (uint16_t layer = 0; layer < layers; ++layer) {
        config.dmrs_ports[layer] = static_cast<uint16_t>(1000 + layer);
    }
    return config;
}

double RunBatch(const QamDemod256BatchOpArgsV1 &args,
                Buffer &workspace, Buffer &tiling)
{
    auto enqueue = [&]() {
        const Status status = Enqueue(args, workspace.device, workspace.bytes,
                                      tiling.device, tiling.bytes);
        if (status != OK) {
            std::fprintf(stderr, "[FAIL] batch enqueue status=%d\n", status);
            std::exit(EXIT_FAILURE);
        }
    };
    for (int i = 0; i < EnvInt("WARMUP", 3); ++i) enqueue();
    ACL_CHECK(aclrtSynchronizeStream(static_cast<aclrtStream>(args.stream)));

    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    ACL_CHECK(aclrtCreateEvent(&start));
    ACL_CHECK(aclrtCreateEvent(&end));
    std::vector<double> latency;
    for (int i = 0; i < EnvInt("TIMED", 20); ++i) {
        ACL_CHECK(aclrtRecordEvent(start, static_cast<aclrtStream>(args.stream)));
        enqueue();
        ACL_CHECK(aclrtRecordEvent(end, static_cast<aclrtStream>(args.stream)));
        ACL_CHECK(aclrtSynchronizeEvent(end));
        float elapsed_ms = 0.0f;
        ACL_CHECK(aclrtEventElapsedTime(&elapsed_ms, start, end));
        latency.push_back(static_cast<double>(elapsed_ms) * 1000.0);
    }
    ACL_CHECK(aclrtDestroyEvent(start));
    ACL_CHECK(aclrtDestroyEvent(end));
    std::sort(latency.begin(), latency.end());
    return latency.empty() ? 0.0 : latency[latency.size() / 2];
}

void RunCase(aclrtStream stream, uint16_t layers, Buffer &workspace)
{
    const std::string name = "rank" + std::to_string(layers);
    const std::string golden = DataRoot() + "/golden/" + name;
    const std::string output_dir = DataRoot() + "/ascend_output/" + name;
    EnsureDir(DataRoot() + "/ascend_output");
    EnsureDir(output_dir);

    Buffer x_re, x_im, no_eff;
    x_re.Load(golden + "/x_re.bin");
    x_im.Load(golden + "/x_im.bin");
    no_eff.Load(golden + "/no_eff.bin");
    const size_t expected_input = GridElems(layers) * sizeof(uint16_t);
    if (x_re.bytes != expected_input || x_im.bytes != expected_input ||
        no_eff.bytes != expected_input) {
        std::fprintf(stderr, "[FAIL] %s grid input byte count mismatch\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    Buffer output;
    output.Allocate(LlrElems(layers) * sizeof(int16_t));
    ACL_CHECK(aclrtMemset(output.device, output.bytes, 0xa5, output.bytes));

    const PuschMimoConfig config = MakeConfig(layers);
    PuschMimoLayout layout {};
    BatchTilingData host_tiling {};
    if (BuildCurrentProfile(config, &layout, &host_tiling) != OK) {
        std::fprintf(stderr, "[FAIL] cannot derive %s profile\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }
    Buffer tiling;
    tiling.Allocate(sizeof(host_tiling));
    std::memcpy(tiling.host, &host_tiling, sizeof(host_tiling));
    ACL_CHECK(aclrtMemcpy(tiling.device, tiling.bytes, tiling.host, tiling.bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));

    QamDemod256BatchOpArgsV1 args {};
    args.abi_version = ABI_VERSION;
    args.struct_size = sizeof(args);
    args.x_re = x_re.device;
    args.x_im = x_im.device;
    args.no_eff = no_eff.device;
    args.layer_llr = output.device;
    args.config = &config;
    args.layout = &layout;
    args.stream = stream;
    if (ValidateOpArgs(args) != OK) {
        std::fprintf(stderr, "[FAIL] public OpArgs validation failed for %s\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    const double latency_us = RunBatch(args, workspace, tiling);
    output.Save(output_dir + "/layer_llr.bin");
    constexpr uint32_t launches = 1;
    constexpr const char *implementation = "native";
    constexpr uint32_t cores = BLOCK_DIM;
    std::printf("[case] %-5s impl=%-7s input=[%u,14,1664] "
                "output=[%u,8,12,1600] launches=%u cores=%u p50=%.1f us\n",
                name.c_str(), implementation, layers, layers, launches, cores,
                latency_us);
}

}

int main()
{
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    {
        Buffer workspace;
        workspace.Allocate(WORKSPACE_BYTES);
        ACL_CHECK(aclrtMemset(workspace.device, workspace.bytes, 0, workspace.bytes));
        for (uint16_t layers = 1; layers <= MAX_LAYERS; ++layers) {
            RunCase(stream, layers, workspace);
        }
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
