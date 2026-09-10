#include "layer_demap.h"

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
#include "aclrtlaunch_layer_demap_kernel.h"

using namespace airan::layer_demap;

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
    config.num_symbols = 14;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = 14;
    config.used_subcarriers = 1596;
    config.padded_subcarriers = 1664;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    for (uint16_t layer = 0; layer < layers; ++layer) {
        config.dmrs_ports[layer] = static_cast<uint16_t>(1000 + layer);
    }
    return config;
}

double RunKernel(aclrtStream stream, const Buffer &input, Buffer &output,
                 const Buffer &tiling)
{
    auto enqueue = [&]() {
        ACLRT_LAUNCH_KERNEL(layer_demap_kernel)
            (BLOCK_DIM, stream, input.device, output.device, nullptr, tiling.device);
    };
    for (int i = 0; i < EnvInt("WARMUP", 3); ++i) enqueue();
    ACL_CHECK(aclrtSynchronizeStream(stream));
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    ACL_CHECK(aclrtCreateEvent(&start));
    ACL_CHECK(aclrtCreateEvent(&end));
    std::vector<double> latency;
    for (int i = 0; i < EnvInt("TIMED", 20); ++i) {
        ACL_CHECK(aclrtRecordEvent(start, stream));
        enqueue();
        ACL_CHECK(aclrtRecordEvent(end, stream));
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

void RunCase(aclrtStream stream, uint16_t layers)
{
    const std::string name = "rank" + std::to_string(layers);
    const std::string golden = DataRoot() + "/golden/" + name;
    const std::string output_dir = DataRoot() + "/ascend_output/" + name;
    EnsureDir(DataRoot() + "/ascend_output");
    EnsureDir(output_dir);

    Buffer input;
    input.Load(golden + "/layer_llr.bin");
    const size_t expected_input = LayerElems(layers) * sizeof(int16_t);
    if (input.bytes != expected_input) {
        std::fprintf(stderr, "[FAIL] %s input bytes=%zu expected=%zu\n",
                     name.c_str(), input.bytes, expected_input);
        std::exit(EXIT_FAILURE);
    }

    Buffer output;
    output.Allocate(CodewordElems(layers) * sizeof(int16_t));
    ACL_CHECK(aclrtMemset(output.device, output.bytes, 0x5a, output.bytes));

    const PuschMimoConfig config = MakeConfig(layers);
    PuschMimoLayout layout {};
    Buffer tiling;
    tiling.Allocate(TILING_BYTES);
    std::memset(tiling.host, 0, tiling.bytes);
    auto *metadata = static_cast<KernelMetadata *>(tiling.host);
    auto *gather_index = reinterpret_cast<uint32_t *>(
        static_cast<uint8_t *>(tiling.host) + sizeof(KernelMetadata));
    if (BuildCurrentProfile(config, &layout, metadata, gather_index) != OK) {
        std::fprintf(stderr, "[FAIL] cannot derive %s profile\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    LayerDemapOpArgsV1 args {};
    args.abi_version = ABI_VERSION;
    args.struct_size = sizeof(args);
    args.layer_llr = input.device;
    args.cw_llr = output.device;
    args.config = &config;
    args.layout = &layout;
    args.stream = stream;
    if (ValidateOpArgs(args) != OK) {
        std::fprintf(stderr, "[FAIL] public OpArgs validation failed for %s\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    std::vector<int16_t> reference(CodewordElems(layers));
    if (ReferenceDemap(static_cast<const int16_t *>(input.host), config, layout,
                       reference.data()) != OK) {
        std::fprintf(stderr, "[FAIL] host reference failed for %s\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }
    std::ifstream expected_file(golden + "/cw_llr.bin", std::ios::binary | std::ios::ate);
    if (!expected_file || static_cast<size_t>(expected_file.tellg()) != output.bytes) {
        std::fprintf(stderr, "[FAIL] invalid golden for %s\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }
    expected_file.seekg(0);
    std::vector<int16_t> expected(reference.size());
    expected_file.read(reinterpret_cast<char *>(expected.data()),
                       static_cast<std::streamsize>(output.bytes));
    if (!expected_file || reference != expected) {
        std::fprintf(stderr, "[FAIL] C++ reference disagrees with independent golden for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    ACL_CHECK(aclrtMemcpy(tiling.device, tiling.bytes, tiling.host, tiling.bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    const double latency_us = RunKernel(stream, input, output, tiling);
    output.Save(output_dir + "/cw_llr.bin");
    std::printf("[case] %-5s input=[%u,8,12,1600] output=[8,%u] p50=%.1f us\n",
                name.c_str(), layers, layers * N_DATA_PAD, latency_us);
}

}  // namespace

int main()
{
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    {
        for (uint16_t layers = 1; layers <= MAX_LAYERS; ++layers) {
            RunCase(stream, layers);
        }
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
