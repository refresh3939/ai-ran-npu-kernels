#include "re_map_batch.h"

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

using namespace airan::re_map_batch;

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
        ACL_CHECK(aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void CopyFrom(const void *source, size_t count)
    {
        Allocate(count);
        std::memcpy(host, source, count);
        ACL_CHECK(aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void Save(const std::string &path)
    {
        ACL_CHECK(aclrtMemcpy(host, bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(static_cast<const char *>(host), static_cast<std::streamsize>(bytes));
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

PuschMimoConfig MakeConfig(uint16_t ports)
{
    PuschMimoConfig config {};
    config.abi_version = ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = ports == 4 ? 3 : ports;
    config.num_tx_ports = ports;
    config.num_rx_antennas = 64;
    config.qm = 8;
    config.num_symbols = N_SYMBOLS;
    config.fft_size = N_FFT;
    config.num_rb = 133;
    config.num_allocated_symbols = N_SYMBOLS;
    config.used_subcarriers = N_SC_USED;
    config.padded_subcarriers = N_SC_PAD;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    config.codebook_enabled = ports == 4 ? 1 : 0;
    for (uint16_t layer = 0; layer < config.num_layers; ++layer) {
        config.dmrs_ports[layer] = static_cast<uint16_t>(1000 + layer);
    }
    return config;
}

void CheckContractValidation(aclrtStream stream)
{
    PuschMimoConfig config = MakeConfig(4);
    PuschMimoLayout layout {};
    uint32_t scatter[SCATTER_INDEX_ELEMS] = {};
    if (BuildCurrentProfile(config, &layout, scatter, SCATTER_INDEX_ELEMS) != OK) {
        std::fprintf(stderr, "[FAIL] current profile construction failed\n");
        std::exit(EXIT_FAILURE);
    }
    ReMapBatchOpArgsV1 args {};
    args.abi_version = ABI_VERSION;
    args.struct_size = sizeof(args);
    args.port_grid_re = &args;
    args.port_grid_im = &args;
    args.fft_grid_re = &args;
    args.fft_grid_im = &args;
    args.config = &config;
    args.layout = &layout;
    args.stream = stream;
    if (ValidateOpArgs(args) != OK) {
        std::fprintf(stderr, "[FAIL] valid public contract rejected\n");
        std::exit(EXIT_FAILURE);
    }
    ++layout.data_stride;
    if (ValidateOpArgs(args) != LAYOUT_MISMATCH) {
        std::fprintf(stderr, "[FAIL] upstream layout mismatch was not rejected\n");
        std::exit(EXIT_FAILURE);
    }
    --layout.data_stride;
    config.codebook_enabled = 0;
    if (ValidateOpArgs(args) != UNSUPPORTED_PROFILE) {
        std::fprintf(stderr, "[FAIL] illegal rank-3 precoder bypass was not rejected\n");
        std::exit(EXIT_FAILURE);
    }
}

double RunBatch(const ReMapBatchOpArgsV1 &args,
                const Buffer &scatter,
                Buffer &workspace,
                Buffer &tiling)
{
    auto enqueue = [&]() {
        const Status status = Enqueue(args, scatter.device, scatter.bytes,
                                      workspace.device, workspace.bytes,
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

void RunCase(aclrtStream stream, uint16_t ports, Buffer &workspace, Buffer &tiling)
{
    const std::string name = "port" + std::to_string(ports);
    const std::string golden = DataRoot() + "/golden/" + name;
    const std::string output_dir = DataRoot() + "/ascend_output/" + name;
    EnsureDir(DataRoot() + "/ascend_output");
    EnsureDir(output_dir);

    Buffer input_re, input_im;
    input_re.Load(golden + "/port_grid_re.bin");
    input_im.Load(golden + "/port_grid_im.bin");
    const size_t expected_input = PortGridElems(ports) * sizeof(uint16_t);
    if (input_re.bytes != expected_input || input_im.bytes != expected_input) {
        std::fprintf(stderr, "[FAIL] %s input byte count mismatch\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    Buffer output_re, output_im;
    const size_t output_bytes = FftGridElems(ports) * sizeof(uint16_t);
    output_re.Allocate(output_bytes);
    output_im.Allocate(output_bytes);
    ACL_CHECK(aclrtMemset(output_re.device, output_re.bytes, 0xa5, output_re.bytes));
    ACL_CHECK(aclrtMemset(output_im.device, output_im.bytes, 0x5a, output_im.bytes));

    const PuschMimoConfig config = MakeConfig(ports);
    PuschMimoLayout layout {};
    uint32_t host_scatter[SCATTER_INDEX_ELEMS] = {};
    if (BuildCurrentProfile(config, &layout, host_scatter, SCATTER_INDEX_ELEMS) != OK) {
        std::fprintf(stderr, "[FAIL] cannot derive %s profile\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }
    Buffer scatter;
    scatter.CopyFrom(host_scatter, sizeof(host_scatter));

    ReMapBatchOpArgsV1 args {};
    args.abi_version = ABI_VERSION;
    args.struct_size = sizeof(args);
    args.port_grid_re = input_re.device;
    args.port_grid_im = input_im.device;
    args.fft_grid_re = output_re.device;
    args.fft_grid_im = output_im.device;
    args.config = &config;
    args.layout = &layout;
    args.stream = stream;
    if (ValidateOpArgs(args) != OK) {
        std::fprintf(stderr, "[FAIL] public OpArgs validation failed for %s\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    const double latency_us = RunBatch(args, scatter, workspace, tiling);
    output_re.Save(output_dir + "/fft_grid_re.bin");
    output_im.Save(output_dir + "/fft_grid_im.bin");
    std::printf("[case] %-5s input=[%u,14,1664] output=[%u,14,2048] "
                "launches=%u cores/launch=%u p50=%.1f us\n",
                name.c_str(), ports, ports, ports, SISO_BLOCK_DIM, latency_us);
}

}

int main()
{
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    {
        CheckContractValidation(stream);
        Buffer workspace, tiling;
        workspace.Allocate(DUMMY_WORKSPACE_BYTES);
        tiling.Allocate(DUMMY_TILING_BYTES);
        ACL_CHECK(aclrtMemset(workspace.device, workspace.bytes, 0, workspace.bytes));
        ACL_CHECK(aclrtMemset(tiling.device, tiling.bytes, 0, tiling.bytes));
        for (uint16_t ports : {uint16_t{1}, uint16_t{2}, uint16_t{4}}) {
            RunCase(stream, ports, workspace, tiling);
        }
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
