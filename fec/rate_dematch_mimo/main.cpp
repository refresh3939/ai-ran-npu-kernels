#include "rate_dematch_mimo.h"

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

using namespace airan::rate_dematch_mimo;

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
        std::memset(host, 0, bytes);
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
    void CopyToDevice() {
        ACL_CHECK(aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    void CopyToHost() {
        ACL_CHECK(aclrtMemcpy(host, bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    void Save(const std::string &path) const {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(static_cast<const char *>(host),
                     static_cast<std::streamsize>(bytes));
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

int EnvInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    return value == nullptr ? fallback : std::max(0, std::atoi(value));
}

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
    config.codeword_index = 0;
    return config;
}

std::vector<int16_t> LoadI16(const std::string &path, size_t elements)
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

double LaunchTimed(const RateDematchMimoOpArgsV1 &args,
                   const Buffer &descriptors,
                   Buffer &workspace,
                   const Buffer &tiling)
{
    auto enqueue = [&]() {
        const Status status = Enqueue(args, descriptors.device, descriptors.bytes,
                                      workspace.device, workspace.bytes,
                                      tiling.device, tiling.bytes);
        if (status != OK) {
            std::fprintf(stderr, "[FAIL] Enqueue status=%d\n", status);
            std::exit(EXIT_FAILURE);
        }
    };
    for (int i = 0; i < EnvInt("WARMUP", 1); ++i) enqueue();
    ACL_CHECK(aclrtSynchronizeStream(static_cast<aclrtStream>(args.stream)));

    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    ACL_CHECK(aclrtCreateEvent(&start));
    ACL_CHECK(aclrtCreateEvent(&end));
    std::vector<double> latency;
    for (int i = 0; i < EnvInt("TIMED", 3); ++i) {
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
    const std::string case_dir = DataRoot() + "/golden/" + name;
    const std::string output_root = DataRoot() + "/ascend_output";
    const std::string output_dir = output_root + "/" + name;
    EnsureDir(output_root);
    EnsureDir(output_dir);
    const PuschMimoConfig config = MakeConfig(layers);
    PuschMimoLayout layout {};
    KernelMetadata metadata {};
    RateMatchDescriptor host_desc[C_NUM] {};
    const Status profile_status = BuildCurrentProfile(
        config, num_slots, &layout, &metadata, host_desc, C_NUM);
    if (profile_status != OK) {
        std::fprintf(stderr, "[FAIL] BuildCurrentProfile %s status=%d\n",
                     name.c_str(), profile_status);
        std::exit(EXIT_FAILURE);
    }

    Buffer input;
    input.Load(case_dir + "/cw_llr.bin");
    const size_t input_bytes = InputElems(num_slots, layers) * sizeof(int16_t);
    if (input.bytes != input_bytes) {
        std::fprintf(stderr, "[FAIL] %s input bytes=%zu expected=%zu\n",
                     name.c_str(), input.bytes, input_bytes);
        std::exit(EXIT_FAILURE);
    }
    const auto expected = LoadI16(case_dir + "/ldpc_llr.bin", OutputElems());
    std::vector<int16_t> reference(OutputElems());
    const Status reference_status = ReferenceRateDematch(
        static_cast<const int16_t *>(input.host), config, layout, num_slots,
        host_desc, C_NUM, reference.data());
    if (reference_status != OK || reference != expected) {
        std::fprintf(stderr,
                     "[FAIL] C++ reference disagrees with independent Python for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    Buffer descriptors;
    descriptors.Allocate(DESCRIPTOR_BYTES);
    std::memcpy(descriptors.host, host_desc, sizeof(host_desc));
    descriptors.CopyToDevice();
    Buffer tiling;
    tiling.Allocate(TILING_BYTES);
    std::memcpy(tiling.host, &metadata, sizeof(metadata));
    tiling.CopyToDevice();
    Buffer output;
    output.Allocate(OutputElems() * sizeof(int16_t));
    ACL_CHECK(aclrtMemset(output.device, output.bytes, 0x5a, output.bytes));

    RateDematchMimoOpArgsV1 args {};
    args.abi_version = ABI_VERSION;
    args.struct_size = sizeof(args);
    args.cw_llr = input.device;
    args.ldpc_llr = output.device;
    args.config = &config;
    args.layout = &layout;
    args.num_slots = num_slots;
    args.stream = stream;
    if (ValidateOpArgs(args) != OK) {
        std::fprintf(stderr, "[FAIL] public ABI validation failed for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    const double latency_us = LaunchTimed(args, descriptors, workspace, tiling);
    output.CopyToHost();
    const auto *actual = static_cast<const int16_t *>(output.host);
    size_t first_bad = OutputElems();
    for (size_t i = 0; i < OutputElems(); ++i) {
        if (actual[i] != expected[i]) {
            first_bad = i;
            break;
        }
    }
    if (first_bad != OutputElems()) {
        std::fprintf(stderr,
                     "[FAIL] %s mismatch index=%zu cb=%zu col=%zu got=%d expected=%d\n",
                     name.c_str(), first_bad, first_bad / LDPC_N,
                     first_bad % LDPC_N, actual[first_bad], expected[first_bad]);
        std::exit(EXIT_FAILURE);
    }
    bool prefix_zero = true;
    for (uint32_t cb = 0; cb < C_NUM && prefix_zero; ++cb) {
        const size_t row = static_cast<size_t>(cb) * LDPC_N;
        prefix_zero = std::all_of(actual + row, actual + row + N_2Z,
                                  [](int16_t value) { return value == 0; });
    }
    if (!prefix_zero) {
        std::fprintf(stderr, "[FAIL] %s punctured 2Z prefix is not zero\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }
    output.Save(output_dir + "/ldpc_llr.bin");
    const uint32_t e_low = host_desc[0].e;
    const uint32_t e_high = host_desc[C_NUM - 1].e;
    std::printf("[PASS] %-5s slots=%u G=%u E={%u,%u} repetition=%s "
                "shape=[%u,8,%u]->[%u,%u] p50=%.1f us\n",
                name.c_str(), num_slots, metadata.total_coded_bits,
                e_low, e_high, e_high > N_CB_BUF ? "yes" : "no",
                num_slots, layout.codeword_stride, C_NUM, LDPC_N, latency_us);
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
        workspace.CopyToDevice();
        for (uint16_t layers = 1; layers <= MAX_LAYERS; ++layers) {
            RunCase(stream, layers, MAX_SLOTS, workspace);
        }
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
