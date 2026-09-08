#include "rate_match_mimo.h"

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

using namespace airan::rate_match_mimo;

namespace {

#define ACL_CHECK(expr) do { \
    const aclError status_ = (expr); \
    if (status_ != ACL_ERROR_NONE) { \
        std::fprintf(stderr, "[ACL] %s:%d status=%d\n", \
                     __FILE__, __LINE__, status_); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

struct Buffer {
    void *device = nullptr;
    void *host = nullptr;
    size_t bytes = 0;

    void Allocate(size_t count)
    {
        bytes = count;
        ACL_CHECK(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost(&host, bytes));
    }

    void Load(const std::string &path)
    {
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
        ACL_CHECK(aclrtMemcpy(device, bytes, host, bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void Save(const std::string &path)
    {
        ACL_CHECK(aclrtMemcpy(host, bytes, device, bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST));
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(static_cast<const char *>(host),
                     static_cast<std::streamsize>(bytes));
        if (!output) {
            std::fprintf(stderr, "[FAIL] cannot write %s\n", path.c_str());
            std::exit(EXIT_FAILURE);
        }
    }

    ~Buffer()
    {
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
    config.num_tx_ports = layers == 1 ? 1 : (layers == 2 ? 2 : 4);
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
    return config;
}

RateMatchFecConfigV1 MakeFecConfig()
{
    RateMatchFecConfigV1 fec {};
    fec.abi_version = ABI_VERSION;
    fec.struct_size = sizeof(fec);
    fec.num_code_blocks = C_NUM;
    fec.encoded_stride = N_CB_BUF;
    fec.rv_index = 0;
    fec.base_graph = 1;
    fec.lifting_size = LDPC_Z;
    return fec;
}

std::vector<int16_t> LoadI16(const std::string &path, size_t elements)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) !=
                      elements * sizeof(int16_t)) {
        std::fprintf(stderr, "[FAIL] invalid int16 file %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    std::vector<int16_t> values(elements);
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(elements * sizeof(int16_t)));
    if (!input) {
        std::fprintf(stderr, "[FAIL] short read %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    return values;
}

double RunKernel(const RateMatchMimoOpArgsV1 &args, const Buffer &descriptors,
                 Buffer &workspace, const Buffer &tiling)
{
    auto enqueue = [&]() {
        const Status status = Enqueue(args, descriptors.device, descriptors.bytes,
                                      workspace.device, workspace.bytes,
                                      tiling.device, tiling.bytes);
        if (status != OK) {
            std::fprintf(stderr, "[FAIL] enqueue status=%d\n", status);
            std::exit(EXIT_FAILURE);
        }
    };
    for (int i = 0; i < EnvInt("WARMUP", 2); ++i) enqueue();
    ACL_CHECK(aclrtSynchronizeStream(static_cast<aclrtStream>(args.stream)));

    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    ACL_CHECK(aclrtCreateEvent(&start));
    ACL_CHECK(aclrtCreateEvent(&end));
    std::vector<double> latency;
    for (int i = 0; i < EnvInt("TIMED", 5); ++i) {
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
    const std::string output_root = DataRoot() + "/ascend_output";
    const std::string output_dir = output_root + "/" + name;
    EnsureDir(output_root);
    EnsureDir(output_dir);

    const PuschMimoConfig config = MakeConfig(layers);
    const RateMatchFecConfigV1 fec = MakeFecConfig();
    PuschMimoLayout layout {};
    KernelMetadata metadata {};
    if (BuildCurrentProfile(config, fec, num_slots, &layout, &metadata) != OK) {
        std::fprintf(stderr, "[FAIL] cannot derive %s profile\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }
    if (layout.codeword_symbols != layers * N_DATA_RE ||
        layout.codeword_stride != layers * N_DATA_PAD) {
        std::fprintf(stderr, "[FAIL] scramble_mimo layout contract mismatch\n");
        std::exit(EXIT_FAILURE);
    }

    std::vector<RateMatchDescriptor> host_descriptors(C_NUM);
    if (BuildRateMatchDescriptors(config, fec, num_slots,
                                  host_descriptors.data(),
                                  host_descriptors.size()) != OK) {
        std::fprintf(stderr, "[FAIL] descriptor build failed for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }
    uint64_t total_e = 0;
    for (const auto &desc : host_descriptors) total_e += desc.e;
    const uint64_t expected_g = static_cast<uint64_t>(num_slots) * Q_M *
                                layers * N_DATA_RE;
    if (total_e != expected_g) {
        std::fprintf(stderr, "[FAIL] descriptor E sum mismatch for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    Buffer input;
    input.Load(golden_dir + "/code_blocks.bin");
    const size_t input_bytes = InputElems(fec) * sizeof(int8_t);
    if (input.bytes != input_bytes) {
        std::fprintf(stderr, "[FAIL] %s LDPC input size mismatch\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }
    const size_t output_elems = OutputElems(num_slots, layers);
    const size_t output_bytes = output_elems * sizeof(int16_t);
    const auto expected = LoadI16(golden_dir + "/bits_nr.bin", output_elems);
    std::vector<int16_t> reference(output_elems);
    if (ReferenceRateMatch(static_cast<const int8_t *>(input.host), config,
                           layout, fec, num_slots, host_descriptors.data(),
                           reference.data()) != OK || reference != expected) {
        std::fprintf(stderr,
                     "[FAIL] C++ reference disagrees with Python for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    Buffer descriptors;
    descriptors.Allocate(host_descriptors.size() * sizeof(RateMatchDescriptor));
    std::memcpy(descriptors.host, host_descriptors.data(), descriptors.bytes);
    ACL_CHECK(aclrtMemcpy(descriptors.device, descriptors.bytes, descriptors.host,
                         descriptors.bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    Buffer tiling;
    tiling.Allocate(sizeof(metadata));
    std::memcpy(tiling.host, &metadata, sizeof(metadata));
    ACL_CHECK(aclrtMemcpy(tiling.device, tiling.bytes, tiling.host, tiling.bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));
    Buffer output;
    output.Allocate(output_bytes);
    ACL_CHECK(aclrtMemset(output.device, output.bytes, 0x5a, output.bytes));

    RateMatchMimoOpArgsV1 args {};
    args.abi_version = ABI_VERSION;
    args.struct_size = sizeof(args);
    args.code_blocks = input.device;
    args.bits_nr = output.device;
    args.config = &config;
    args.layout = &layout;
    args.fec = &fec;
    args.num_slots = num_slots;
    args.stream = stream;
    if (ValidateOpArgs(args) != OK) {
        std::fprintf(stderr, "[FAIL] OpArgs validation failed for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }
    const double latency_us = RunKernel(args, descriptors, workspace, tiling);
    output.Save(output_dir + "/bits_nr.bin");
    std::printf("[case] %-5s slots=%u G=%llu shape=[%u,8,%u] "
                "E=[%u,%u] p50=%.1f us (%.2f us/slot)\n",
                name.c_str(), num_slots,
                static_cast<unsigned long long>(expected_g), num_slots,
                layout.codeword_stride, host_descriptors.front().e,
                host_descriptors.back().e, latency_us,
                latency_us / num_slots);
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
        ACL_CHECK(aclrtMemset(workspace.device, workspace.bytes, 0,
                             workspace.bytes));
        for (uint16_t layers = 1; layers <= MAX_LAYERS; ++layers) {
            RunCase(stream, layers, MAX_SLOTS, workspace);
        }
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
