#include "qam_mod_256_mimo.h"

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

namespace qmm = airan::qam_mod_256_mimo;

namespace {

#define ACL_CHECK(expr) do {                                                    \
    const aclError status_ = (expr);                                            \
    if (status_ != ACL_ERROR_NONE) {                                            \
        std::fprintf(stderr, "[ACL] %s:%d status=%d\n",                       \
                     __FILE__, __LINE__, status_);                              \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                          \
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
        ACL_CHECK(aclrtMemcpy(device, bytes, host, bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void Save(const std::string &path) {
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

qmm::PuschMimoConfig MakeConfig(uint16_t layers)
{
    qmm::PuschMimoConfig config {};
    config.abi_version = qmm::ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = layers;
    config.num_tx_ports = layers == 1 ? 1 : (layers == 2 ? 2 : 4);
    config.num_rx_antennas = 64;
    config.qm = qmm::Q_M;
    config.num_symbols = 14;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = 14;
    config.used_subcarriers = qmm::N_SC_USED;
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
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(uint16_t)) {
        std::fprintf(stderr, "[FAIL] invalid fp16 file %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    std::vector<uint16_t> values(count);
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(count * sizeof(uint16_t)));
    if (!input) {
        std::fprintf(stderr, "[FAIL] short read %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    return values;
}

double RunKernel(const qmm::QamMod256MimoOpArgsV1 &args,
                 Buffer &workspace, const Buffer &tiling)
{
    auto enqueue = [&]() {
        const qmm::Status status = qmm::Enqueue(
            args, workspace.device, workspace.bytes, tiling.device, tiling.bytes);
        if (status != qmm::OK) {
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

void RequireExact(const char *name, const uint16_t *actual,
                  const std::vector<uint16_t> &expected)
{
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr,
                         "[FAIL] %s mismatch at %zu: npu=0x%04x golden=0x%04x\n",
                         name, i, actual[i], expected[i]);
            std::exit(EXIT_FAILURE);
        }
    }
}

void RunCase(aclrtStream stream, uint16_t layers, Buffer &workspace)
{
    const std::string name = "rank" + std::to_string(layers);
    const std::string golden = DataRoot() + "/golden/" + name;
    const std::string output = DataRoot() + "/ascend_output/" + name;
    EnsureDir(DataRoot() + "/ascend_output");
    EnsureDir(output);

    const qmm::PuschMimoConfig config = MakeConfig(layers);
    qmm::PuschMimoLayout layout {};
    qmm::KernelMetadata metadata {};
    if (qmm::BuildCurrentProfile(config, &layout, &metadata) != qmm::OK) {
        std::fprintf(stderr, "[FAIL] cannot derive %s profile\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    Buffer bits;
    bits.Load(golden + "/bits_qam.bin");
    const size_t input_bytes = qmm::InputElems(layers) * sizeof(int16_t);
    const size_t output_elems = qmm::OutputElems(layers);
    const size_t output_bytes = output_elems * sizeof(uint16_t);
    if (bits.bytes != input_bytes) {
        std::fprintf(stderr, "[FAIL] %s input shape mismatch\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    const auto expected_re = ReadWords(golden + "/d_re.bin", output_elems);
    const auto expected_im = ReadWords(golden + "/d_im.bin", output_elems);
    std::vector<uint16_t> reference_re(output_elems);
    std::vector<uint16_t> reference_im(output_elems);
    if (qmm::ReferenceModulate(static_cast<const int16_t *>(bits.host),
                               config, layout, reference_re.data(),
                               reference_im.data()) != qmm::OK ||
        reference_re != expected_re || reference_im != expected_im) {
        std::fprintf(stderr,
                     "[FAIL] C++ SISO-math reference disagrees with Python for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    Buffer d_re;
    Buffer d_im;
    d_re.Allocate(output_bytes);
    d_im.Allocate(output_bytes);
    ACL_CHECK(aclrtMemset(d_re.device, d_re.bytes, 0x5a, d_re.bytes));
    ACL_CHECK(aclrtMemset(d_im.device, d_im.bytes, 0xa5, d_im.bytes));

    Buffer tiling;
    tiling.Allocate(qmm::TILING_BYTES);
    std::memcpy(tiling.host, &metadata, sizeof(metadata));
    ACL_CHECK(aclrtMemcpy(tiling.device, tiling.bytes, tiling.host, tiling.bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    qmm::QamMod256MimoOpArgsV1 args {};
    args.abi_version = qmm::ABI_VERSION;
    args.struct_size = sizeof(args);
    args.bits_qam = bits.device;
    args.d_re = d_re.device;
    args.d_im = d_im.device;
    args.config = &config;
    args.layout = &layout;
    args.stream = stream;
    if (qmm::ValidateOpArgs(args) != qmm::OK) {
        std::fprintf(stderr, "[FAIL] public OpArgs validation failed for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    const double latency_us = RunKernel(args, workspace, tiling);
    d_re.Save(output + "/d_re.bin");
    d_im.Save(output + "/d_im.bin");
    RequireExact((name + " d_re").c_str(),
                 static_cast<const uint16_t *>(d_re.host), expected_re);
    RequireExact((name + " d_im").c_str(),
                 static_cast<const uint16_t *>(d_im.host), expected_im);
    std::printf("[case] %-5s input=[8,%u] output=[%u] valid=%u tail=%u "
                "elementwise=PASS p50=%.1f us\n",
                name.c_str(), layout.codeword_stride, layout.codeword_stride,
                layout.codeword_symbols,
                layout.codeword_stride - layout.codeword_symbols, latency_us);
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
        workspace.Allocate(qmm::WORKSPACE_BYTES);
        ACL_CHECK(aclrtMemset(workspace.device, workspace.bytes, 0,
                              workspace.bytes));
        for (uint16_t layers = 1; layers <= qmm::MAX_LAYERS; ++layers) {
            RunCase(stream, layers, workspace);
        }
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
