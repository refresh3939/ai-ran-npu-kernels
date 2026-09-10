#include "descramble_mimo.h"

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

using namespace airan::descramble_mimo;

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
    config.used_subcarriers = N_SC_USED;
    config.padded_subcarriers = 1664;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    config.data_scrambling_id = 321;
    config.rnti = 12345;
    config.codeword_index = 0;
    for (uint16_t layer = 0; layer < layers; ++layer) {
        config.dmrs_ports[layer] = static_cast<uint16_t>(1000 + layer);
    }
    return config;
}

std::vector<int16_t> LoadHostI16(const std::string &path, size_t elements)
{
    std::ifstream input(path, std::ios::binary);
    std::vector<int16_t> values(elements);
    input.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(elements * sizeof(int16_t)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        std::fprintf(stderr, "[FAIL] invalid int16 file %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    return values;
}

double RunKernel(const DescrambleMimoOpArgsV1 &args, const Buffer &sign,
                 Buffer &workspace, const Buffer &tiling)
{
    auto enqueue = [&]() {
        const Status status = Enqueue(args, sign.device, sign.bytes,
                                      workspace.device, workspace.bytes,
                                      tiling.device, tiling.bytes);
        if (status != OK) {
            std::fprintf(stderr, "[FAIL] enqueue status=%d\n", status);
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

void RunCase(aclrtStream stream, uint16_t layers, uint32_t num_slots,
             Buffer &workspace)
{
    const std::string name = "rank" + std::to_string(layers);
    const std::string golden_dir = DataRoot() + "/golden/" + name;
    const std::string output_dir = DataRoot() + "/ascend_output/" + name;
    EnsureDir(DataRoot() + "/ascend_output");
    EnsureDir(output_dir);

    const PuschMimoConfig config = MakeConfig(layers);
    PuschMimoLayout layout {};
    KernelMetadata host_tiling {};
    if (BuildCurrentProfile(config, num_slots, &layout, &host_tiling) != OK) {
        std::fprintf(stderr, "[FAIL] cannot derive %s profile\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }
    const size_t elements = BufferElems(num_slots, layers);
    const size_t bytes = elements * sizeof(int16_t);

    Buffer input;
    input.Load(golden_dir + "/llr_qam.bin");
    if (input.bytes != bytes) {
        std::fprintf(stderr, "[FAIL] %s input bytes=%zu expected=%zu\n",
                     name.c_str(), input.bytes, bytes);
        std::exit(EXIT_FAILURE);
    }

    Buffer sign;
    sign.Allocate(bytes);
    const Status sign_status = BuildGoldSign(
        config, layout, num_slots, static_cast<int16_t *>(sign.host), elements);
    if (sign_status != OK) {
        std::fprintf(stderr, "[FAIL] BuildGoldSign status=%d\n", sign_status);
        std::exit(EXIT_FAILURE);
    }
    const auto expected_sign = LoadHostI16(golden_dir + "/sign.bin", elements);
    if (!std::equal(expected_sign.begin(), expected_sign.end(),
                    static_cast<const int16_t *>(sign.host))) {
        std::fprintf(stderr, "[FAIL] C++ Gold sign disagrees with independent golden for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclrtMemcpy(sign.device, sign.bytes, sign.host, sign.bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));

    const auto expected = LoadHostI16(golden_dir + "/llr_nr.bin", elements);
    std::vector<int16_t> reference(elements);
    const Status ref_status = ReferenceDescramble(
        static_cast<const int16_t *>(input.host),
        static_cast<const int16_t *>(sign.host), config, layout, num_slots,
        reference.data());
    if (ref_status != OK || reference != expected) {
        std::fprintf(stderr, "[FAIL] C++ reference disagrees with independent golden for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    Buffer output;
    output.Allocate(bytes);
    ACL_CHECK(aclrtMemset(output.device, output.bytes, 0x5a, output.bytes));

    Buffer tiling;
    tiling.Allocate(sizeof(host_tiling));
    std::memcpy(tiling.host, &host_tiling, sizeof(host_tiling));
    ACL_CHECK(aclrtMemcpy(tiling.device, tiling.bytes, tiling.host, tiling.bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));

    DescrambleMimoOpArgsV1 args {};
    args.abi_version = ABI_VERSION;
    args.struct_size = sizeof(args);
    args.llr_qam = input.device;
    args.llr_nr = output.device;
    args.config = &config;
    args.layout = &layout;
    args.num_slots = num_slots;
    args.stream = stream;
    if (ValidateOpArgs(args) != OK) {
        std::fprintf(stderr, "[FAIL] public OpArgs validation failed for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    const double latency_us = RunKernel(args, sign, workspace, tiling);
    output.Save(output_dir + "/llr_nr.bin");
    std::printf("[case] %-5s slots=%u input=[%u,8,%u] output=[%u,8,%u] "
                "launches=1 cores=%u p50=%.1f us (%.2f us/slot)\n",
                name.c_str(), num_slots, num_slots, layout.codeword_stride,
                num_slots, layout.codeword_stride, BLOCK_DIM, latency_us,
                latency_us / num_slots);
}

}  // namespace

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
        constexpr uint32_t slots_by_rank[MAX_LAYERS] = {23, 7, 3, 2};
        for (uint16_t layers = 1; layers <= MAX_LAYERS; ++layers) {
            RunCase(stream, layers, slots_by_rank[layers - 1], workspace);
        }
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
