#include "layer_map.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_layer_map_kernel.h"
#include "pusch_mimo_runtime_config.h"

namespace lm = airan::layer_map;

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

lm::PuschMimoConfig MakeConfig(uint16_t layers)
{
    lm::PuschMimoConfig config {};
    config.abi_version = lm::ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = layers;
    config.num_tx_ports = layers == 1 ? 1 : (layers == 2 ? 2 : 4);
    config.num_rx_antennas = 64;
    config.qm = lm::Q_M;
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

std::vector<uint16_t> ReadWords(const std::string &path, size_t count)
{
    std::vector<uint16_t> result(count);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(uint16_t)) {
        std::fprintf(stderr, "[FAIL] invalid file %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    input.seekg(0);
    input.read(reinterpret_cast<char *>(result.data()),
               static_cast<std::streamsize>(count * sizeof(uint16_t)));
    if (!input) {
        std::fprintf(stderr, "[FAIL] short read %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    return result;
}

double RunKernel(aclrtStream stream, const Buffer &dRe, const Buffer &dIm,
                 Buffer &layerRe, Buffer &layerIm, const Buffer &tiling)
{
    auto enqueue = [&]() {
        ACLRT_LAUNCH_KERNEL(layer_map_kernel)(
            lm::BLOCK_DIM, stream, dRe.device, dIm.device,
            layerRe.device, layerIm.device, nullptr, tiling.device);
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
        float elapsedMs = 0.0f;
        ACL_CHECK(aclrtEventElapsedTime(&elapsedMs, start, end));
        latency.push_back(static_cast<double>(elapsedMs) * 1000.0);
    }
    ACL_CHECK(aclrtDestroyEvent(start));
    ACL_CHECK(aclrtDestroyEvent(end));
    std::sort(latency.begin(), latency.end());
    return latency.empty() ? 0.0 : latency[latency.size() / 2];
}

void RunCase(aclrtStream stream, uint16_t layers,
             const lm::PuschMimoConfig *profileConfig = nullptr)
{
    const std::string name = "rank" + std::to_string(layers);
    const std::string golden = DataRoot() + "/golden/" + name;
    const std::string outputDir = DataRoot() + "/ascend_output/" + name;
    EnsureDir(DataRoot() + "/ascend_output");
    EnsureDir(outputDir);

    Buffer dRe;
    Buffer dIm;
    dRe.Load(golden + "/d_re.bin");
    dIm.Load(golden + "/d_im.bin");
    const size_t inputBytes = lm::CodewordElems(layers) * sizeof(uint16_t);
    if (dRe.bytes != inputBytes || dIm.bytes != inputBytes) {
        std::fprintf(stderr, "[FAIL] %s input shape mismatch\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    Buffer layerRe;
    Buffer layerIm;
    const size_t outputBytes = lm::LayerElems(layers) * sizeof(uint16_t);
    layerRe.Allocate(outputBytes);
    layerIm.Allocate(outputBytes);
    ACL_CHECK(aclrtMemset(layerRe.device, outputBytes, 0x5a, outputBytes));
    ACL_CHECK(aclrtMemset(layerIm.device, outputBytes, 0xa5, outputBytes));

    const lm::PuschMimoConfig config =
        profileConfig == nullptr ? MakeConfig(layers) : *profileConfig;
    lm::PuschMimoLayout layout {};
    Buffer tiling;
    tiling.Allocate(lm::TILING_BYTES);
    std::memset(tiling.host, 0, tiling.bytes);
    auto *metadata = static_cast<lm::KernelMetadata *>(tiling.host);
    auto *gatherIndex = reinterpret_cast<uint32_t *>(
        static_cast<uint8_t *>(tiling.host) + sizeof(lm::KernelMetadata));
    if (lm::BuildCurrentProfile(config, &layout, metadata, gatherIndex) != lm::OK) {
        std::fprintf(stderr, "[FAIL] cannot derive %s profile\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    lm::LayerMapOpArgsV1 args {};
    args.abi_version = lm::ABI_VERSION;
    args.struct_size = sizeof(args);
    args.d_re = dRe.device;
    args.d_im = dIm.device;
    args.layer_re = layerRe.device;
    args.layer_im = layerIm.device;
    args.config = &config;
    args.layout = &layout;
    args.stream = stream;
    if (lm::ValidateOpArgs(args) != lm::OK) {
        std::fprintf(stderr, "[FAIL] public OpArgs validation failed for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    std::vector<uint16_t> referenceRe(lm::LayerElems(layers));
    std::vector<uint16_t> referenceIm(lm::LayerElems(layers));
    if (lm::ReferenceMap(static_cast<const uint16_t *>(dRe.host),
                         static_cast<const uint16_t *>(dIm.host),
                         config, layout, referenceRe.data(), referenceIm.data()) != lm::OK) {
        std::fprintf(stderr, "[FAIL] host reference failed for %s\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }
    const auto expectedRe = ReadWords(golden + "/layer_re.bin", referenceRe.size());
    const auto expectedIm = ReadWords(golden + "/layer_im.bin", referenceIm.size());
    if (referenceRe != expectedRe || referenceIm != expectedIm) {
        std::fprintf(stderr,
                     "[FAIL] C++ reference disagrees with independent golden for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    ACL_CHECK(aclrtMemcpy(tiling.device, tiling.bytes, tiling.host, tiling.bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    const double latencyUs = RunKernel(stream, dRe, dIm, layerRe, layerIm, tiling);
    layerRe.Save(outputDir + "/layer_re.bin");
    layerIm.Save(outputDir + "/layer_im.bin");
    std::printf("[case] %-5s input=[%u] output=[%u,19200] p50=%.1f us "
                "runtime=%s\n", name.c_str(), layers * lm::N_DATA_PAD,
                layers, latencyUs, profileConfig == nullptr ? "legacy" : "profile");
}

bool ContractRejectionChecks(aclrtStream stream)
{
    lm::PuschMimoConfig config = MakeConfig(1);
    lm::PuschMimoLayout layout {};
    lm::KernelMetadata metadata {};
    uint32_t gatherIndex[lm::MAX_INDEX_ELEMS] = {};
    config.num_layers = 0;
    bool ok = lm::BuildCurrentProfile(config, &layout, &metadata, gatherIndex) ==
              lm::UNSUPPORTED_PROFILE;
    config = MakeConfig(1);
    config.reserved[0] = 1;
    ok &= lm::BuildCurrentProfile(config, &layout, &metadata, gatherIndex) ==
          lm::UNSUPPORTED_PROFILE;
    config = MakeConfig(1);
    ok &= lm::BuildCurrentProfile(config, &layout, &metadata, gatherIndex) == lm::OK;
    lm::PuschMimoLayout badLayout = layout;
    ++badLayout.data_stride;
    lm::LayerMapOpArgsV1 args {};
    args.abi_version = lm::ABI_VERSION;
    args.struct_size = sizeof(args);
    args.d_re = &args;
    args.d_im = &args;
    args.layer_re = &args;
    args.layer_im = &args;
    args.config = &config;
    args.layout = &badLayout;
    args.stream = stream;
    ok &= lm::ValidateOpArgs(args) == lm::LAYOUT_MISMATCH;
    return ok;
}

}

int main()
{
    airan::PuschMimoRuntimeConfig runtime {};
    bool runtimeEnabled = false;
    std::string runtimeWhy;
    if (airan::LoadMimoRuntimeConfigFromEnv(
            0, &runtime, &runtimeEnabled, &runtimeWhy) !=
        airan::MimoRuntimeConfigStatus::kSuccess) {
        std::fprintf(stderr, "[FAIL] runtime config: %s\n", runtimeWhy.c_str());
        return 1;
    }
    lm::PuschMimoConfig profileConfig {};
    if (runtimeEnabled) {
        if (runtime.num_layers > lm::MAX_LAYERS || runtime.max_rx_antennas != 64 ||
            runtime.max_layers != 16 || runtime.qm != lm::Q_M ||
            runtime.data_re_per_layer != lm::N_DATA_RE ||
            runtime.data_stride_per_layer != lm::N_DATA_PAD) {
            std::fprintf(stderr,
                         "[FAIL] runtime config exceeds current layer-map profile\n");
            return 1;
        }
        if (airan::DerivePuschMimoConfig(runtime, runtime.max_rx_antennas,
                                         &profileConfig, &runtimeWhy) !=
            airan::MimoRuntimeConfigStatus::kSuccess) {
            std::fprintf(stderr, "[FAIL] runtime config derivation: %s\n",
                         runtimeWhy.c_str());
            return 1;
        }
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    bool contractOk = false;
    {
        if (runtimeEnabled) {
            RunCase(stream, runtime.num_layers, &profileConfig);
        } else {
            for (uint16_t layers = 1; layers <= lm::MAX_LAYERS; ++layers) {
                RunCase(stream, layers);
            }
        }
        contractOk = ContractRejectionChecks(stream);
        std::printf("[contract] invalid profile/layout rejection %s\n",
                    contractOk ? "PASS" : "FAIL");
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return contractOk ? 0 : 1;
}
