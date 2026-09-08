#include "mimo_data_extract.h"

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
#include "aclrtlaunch_mimo_data_extract_kernel.h"

using namespace airan::mimo_data_extract;

namespace {

#define ACL_CHECK(expr) do { \
    const aclError status_ = (expr); \
    if (status_ != ACL_ERROR_NONE) { \
        std::fprintf(stderr, "[ACL] %s:%d status=%d\n", __FILE__, __LINE__, status_); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

struct TestCase {
    const char *name;
    uint16_t layers;
    uint16_t dmrs_mask;
};

constexpr TestCase CASES[] = {
    {"rank1_default", 1, static_cast<uint16_t>((1u << 2) | (1u << 11))},
    {"rank2_default", 2, static_cast<uint16_t>((1u << 2) | (1u << 11))},
    {"rank3_edge_dmrs", 3, static_cast<uint16_t>((1u << 0) | (1u << 13))},
    {"rank4_alt_dmrs", 4, static_cast<uint16_t>((1u << 1) | (1u << 12))},
};

struct Buffer {
    void *device = nullptr;
    void *host = nullptr;
    size_t bytes = 0;

    void Allocate(size_t count) {
        bytes = count;
        ACL_CHECK(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost(&host, bytes));
    }

    void Load(const std::string &path, size_t expected_bytes) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input || input.tellg() < 0 ||
            static_cast<size_t>(input.tellg()) != expected_bytes) {
            std::fprintf(stderr, "[FAIL] %s must contain %zu bytes\n",
                         path.c_str(), expected_bytes);
            std::exit(EXIT_FAILURE);
        }
        Allocate(expected_bytes);
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

PuschMimoConfig MakeConfig(const TestCase &test)
{
    PuschMimoConfig config {};
    config.abi_version = ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = test.layers;
    config.num_tx_ports = test.layers == 3 ? 4 : test.layers;
    config.num_rx_antennas = 64;
    config.qm = 8;
    config.num_symbols = N_SYMBOLS;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = N_SYMBOLS;
    config.used_subcarriers = N_SC_USED;
    config.padded_subcarriers = N_SC_PAD;
    config.dmrs_symbol_mask = test.dmrs_mask;
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    for (uint16_t layer = 0; layer < test.layers; ++layer) {
        config.dmrs_ports[layer] = static_cast<uint16_t>(1000 + layer);
    }
    return config;
}

void VerifyHostContract()
{
    const PuschMimoConfig config = MakeConfig(CASES[0]);
    PuschMimoLayout layout {};
    KernelMetadata metadata {};
    if (BuildCurrentProfile(config, &layout, &metadata) != OK ||
        layout.num_dmrs_symbols != 2 || layout.num_data_symbols != 12 ||
        layout.num_data_re != N_DATA_RE || layout.data_stride != N_DATA_PAD) {
        std::fprintf(stderr, "[FAIL] host profile/layout contract\n");
        std::exit(EXIT_FAILURE);
    }

    constexpr uint32_t expected_symbols[N_DATA_SYMBOLS] = {
        0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13,
    };
    if (std::memcmp(metadata.data_symbol_to_grid, expected_symbols,
                    sizeof(expected_symbols)) != 0) {
        std::fprintf(stderr, "[FAIL] compact symbol ordering contract\n");
        std::exit(EXIT_FAILURE);
    }

    void *const sentinel = reinterpret_cast<void *>(uintptr_t{1});
    MimoDataExtractOpArgsV1 args {};
    args.abi_version = ABI_VERSION;
    args.struct_size = sizeof(args);
    args.xhat_re = sentinel;
    args.xhat_im = sentinel;
    args.no_eff = sentinel;
    args.data_re = sentinel;
    args.data_im = sentinel;
    args.data_no_eff = sentinel;
    args.config = &config;
    args.layout = &layout;
    args.stream = sentinel;
    args.input_layout = TENSOR_LAYOUT_FULL_GRID_FP16;
    args.output_layout = TENSOR_LAYOUT_COMPACT_DATA_FP16;
    if (ValidateOpArgs(args) != OK) {
        std::fprintf(stderr, "[FAIL] valid tensor-layout contract rejected\n");
        std::exit(EXIT_FAILURE);
    }

    args.input_layout = TENSOR_LAYOUT_COMPACT_DATA_FP16;
    if (ValidateOpArgs(args) != TENSOR_LAYOUT_MISMATCH) {
        std::fprintf(stderr, "[FAIL] compact input miswire was not rejected\n");
        std::exit(EXIT_FAILURE);
    }
    args.input_layout = TENSOR_LAYOUT_FULL_GRID_FP16;
    args.output_layout = TENSOR_LAYOUT_FULL_GRID_FP16;
    if (ValidateOpArgs(args) != TENSOR_LAYOUT_MISMATCH) {
        std::fprintf(stderr, "[FAIL] full-grid output miswire was not rejected\n");
        std::exit(EXIT_FAILURE);
    }



    if (ValidateConsumerLayout(TENSOR_LAYOUT_FULL_GRID_FP16) !=
            TENSOR_LAYOUT_MISMATCH ||
        ValidateConsumerLayout(TENSOR_LAYOUT_COMPACT_DATA_FP16) != OK) {
        std::fprintf(stderr, "[FAIL] downstream compatibility guard\n");
        std::exit(EXIT_FAILURE);
    }

    PuschMimoLayout bad_layout = layout;
    ++bad_layout.data_stride;
    args.output_layout = TENSOR_LAYOUT_COMPACT_DATA_FP16;
    args.layout = &bad_layout;
    if (ValidateOpArgs(args) != LAYOUT_MISMATCH) {
        std::fprintf(stderr, "[FAIL] mismatched derived layout was not rejected\n");
        std::exit(EXIT_FAILURE);
    }

    PuschMimoConfig bad_config = config;
    bad_config.dmrs_symbol_mask = static_cast<uint16_t>(1u << 2);
    args.config = &bad_config;
    args.layout = &layout;
    if (ValidateOpArgs(args) != UNSUPPORTED_PROFILE) {
        std::fprintf(stderr, "[FAIL] unsupported DMRS profile was not rejected\n");
        std::exit(EXIT_FAILURE);
    }

    std::vector<uint16_t> grid_re(GridElems(1));
    std::vector<uint16_t> grid_im(GridElems(1));
    std::vector<uint16_t> grid_no_eff(GridElems(1));
    for (size_t index = 0; index < grid_re.size(); ++index) {
        grid_re[index] = static_cast<uint16_t>(index * 17u + 1u);
        grid_im[index] = static_cast<uint16_t>(index * 29u + 3u);
        grid_no_eff[index] = static_cast<uint16_t>(index * 43u + 5u);
    }
    std::vector<uint16_t> data_re(DataElems(1), uint16_t{0x5a5a});
    std::vector<uint16_t> data_im(DataElems(1), uint16_t{0x5a5a});
    std::vector<uint16_t> data_no_eff(DataElems(1), uint16_t{0x5a5a});
    if (ReferenceExtract(grid_re.data(), grid_im.data(), grid_no_eff.data(),
                         config, layout, data_re.data(), data_im.data(),
                         data_no_eff.data()) != OK) {
        std::fprintf(stderr, "[FAIL] host reference rejected valid inputs\n");
        std::exit(EXIT_FAILURE);
    }
    for (uint32_t ordinal = 0; ordinal < N_DATA_SYMBOLS; ++ordinal) {
        const size_t source =
            static_cast<size_t>(expected_symbols[ordinal]) * N_SC_PAD;
        const size_t destination = static_cast<size_t>(ordinal) * N_SC_USED;
        if (std::memcmp(grid_re.data() + source, data_re.data() + destination,
                        N_SC_USED * sizeof(uint16_t)) != 0 ||
            std::memcmp(grid_im.data() + source, data_im.data() + destination,
                        N_SC_USED * sizeof(uint16_t)) != 0 ||
            std::memcmp(grid_no_eff.data() + source,
                        data_no_eff.data() + destination,
                        N_SC_USED * sizeof(uint16_t)) != 0) {
            std::fprintf(stderr, "[FAIL] host reference data ordering\n");
            std::exit(EXIT_FAILURE);
        }
    }
    for (uint32_t index = N_DATA_RE; index < N_DATA_PAD; ++index) {
        if (data_re[index] != 0 || data_im[index] != 0 ||
            data_no_eff[index] != 0) {
            std::fprintf(stderr, "[FAIL] host reference output tail\n");
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("[host] profile/layout/order/miswire guards PASS\n");
}

bool FileEquals(const std::string &path, const uint16_t *actual, size_t elements)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const size_t expected_bytes = elements * sizeof(uint16_t);
    if (!input || input.tellg() < 0 ||
        static_cast<size_t>(input.tellg()) != expected_bytes) {
        return false;
    }
    input.seekg(0);
    std::vector<uint16_t> expected(elements);
    input.read(reinterpret_cast<char *>(expected.data()),
               static_cast<std::streamsize>(expected_bytes));
    return input.good() &&
           std::memcmp(expected.data(), actual, expected_bytes) == 0;
}

double RunKernel(aclrtStream stream,
                 const Buffer &xhat_re, const Buffer &xhat_im, const Buffer &no_eff,
                 Buffer &data_re, Buffer &data_im, Buffer &data_no_eff,
                 const Buffer &tiling)
{
    auto enqueue = [&]() {
        ACLRT_LAUNCH_KERNEL(mimo_data_extract_kernel)(
            BLOCK_DIM, stream, xhat_re.device, xhat_im.device, no_eff.device,
            data_re.device, data_im.device, data_no_eff.device, nullptr, tiling.device);
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

void RunCase(aclrtStream stream, const TestCase &test)
{
    const std::string golden = DataRoot() + "/golden/" + test.name;
    const std::string output = DataRoot() + "/ascend_output/" + test.name;
    EnsureDir(DataRoot() + "/ascend_output");
    EnsureDir(output);
    const size_t grid_bytes = GridElems(test.layers) * sizeof(uint16_t);
    const size_t data_bytes = DataElems(test.layers) * sizeof(uint16_t);

    Buffer xhat_re, xhat_im, no_eff;
    xhat_re.Load(golden + "/xhat_re.bin", grid_bytes);
    xhat_im.Load(golden + "/xhat_im.bin", grid_bytes);
    no_eff.Load(golden + "/no_eff.bin", grid_bytes);
    Buffer data_re, data_im, data_no_eff;
    data_re.Allocate(data_bytes);
    data_im.Allocate(data_bytes);
    data_no_eff.Allocate(data_bytes);
    ACL_CHECK(aclrtMemset(data_re.device, data_bytes, 0x5a, data_bytes));
    ACL_CHECK(aclrtMemset(data_im.device, data_bytes, 0x5a, data_bytes));
    ACL_CHECK(aclrtMemset(data_no_eff.device, data_bytes, 0x5a, data_bytes));

    const PuschMimoConfig config = MakeConfig(test);
    PuschMimoLayout layout {};
    Buffer tiling;
    tiling.Allocate(TILING_BYTES);
    std::memset(tiling.host, 0, tiling.bytes);
    if (BuildCurrentProfile(config, &layout,
                            static_cast<KernelMetadata *>(tiling.host)) != OK) {
        std::fprintf(stderr, "[FAIL] cannot derive %s profile\n", test.name);
        std::exit(EXIT_FAILURE);
    }

    MimoDataExtractOpArgsV1 args {};
    args.abi_version = ABI_VERSION;
    args.struct_size = sizeof(args);
    args.xhat_re = xhat_re.device;
    args.xhat_im = xhat_im.device;
    args.no_eff = no_eff.device;
    args.data_re = data_re.device;
    args.data_im = data_im.device;
    args.data_no_eff = data_no_eff.device;
    args.config = &config;
    args.layout = &layout;
    args.stream = stream;
    args.input_layout = TENSOR_LAYOUT_FULL_GRID_FP16;
    args.output_layout = TENSOR_LAYOUT_COMPACT_DATA_FP16;
    if (ValidateOpArgs(args) != OK) {
        std::fprintf(stderr, "[FAIL] public OpArgs validation failed for %s\n", test.name);
        std::exit(EXIT_FAILURE);
    }

    std::vector<uint16_t> reference_re(DataElems(test.layers));
    std::vector<uint16_t> reference_im(DataElems(test.layers));
    std::vector<uint16_t> reference_no_eff(DataElems(test.layers));
    if (ReferenceExtract(static_cast<const uint16_t *>(xhat_re.host),
                         static_cast<const uint16_t *>(xhat_im.host),
                         static_cast<const uint16_t *>(no_eff.host), config, layout,
                         reference_re.data(), reference_im.data(),
                         reference_no_eff.data()) != OK ||
        !FileEquals(golden + "/data_re.bin", reference_re.data(), reference_re.size()) ||
        !FileEquals(golden + "/data_im.bin", reference_im.data(), reference_im.size()) ||
        !FileEquals(golden + "/data_no_eff.bin", reference_no_eff.data(),
                    reference_no_eff.size())) {
        std::fprintf(stderr,
                     "[FAIL] C++ reference disagrees with independent golden for %s\n",
                     test.name);
        std::exit(EXIT_FAILURE);
    }

    ACL_CHECK(aclrtMemcpy(tiling.device, tiling.bytes, tiling.host, tiling.bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    const double latency_us = RunKernel(stream, xhat_re, xhat_im, no_eff,
                                        data_re, data_im, data_no_eff, tiling);
    data_re.Save(output + "/data_re.bin");
    data_im.Save(output + "/data_im.bin");
    data_no_eff.Save(output + "/data_no_eff.bin");
    std::printf("[case] %-16s L=%u mask=0x%04x p50=%.1f us\n",
                test.name, test.layers, test.dmrs_mask, latency_us);
}

}

int main()
{
    VerifyHostContract();
    if (std::getenv("HOST_CONTRACT_ONLY") != nullptr) return 0;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    {
        for (const TestCase &test : CASES) RunCase(stream, test);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
