#include "mimo_detect_io_pack.h"

#include <algorithm>
#include <array>
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

using namespace airan::mimo_detect_io_pack;
extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

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

    void Load(const std::string &path, size_t expected)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input || input.tellg() < 0 || static_cast<size_t>(input.tellg()) != expected) {
            std::fprintf(stderr, "[FAIL] input size mismatch: %s expected=%zu\n",
                         path.c_str(), expected);
            std::exit(EXIT_FAILURE);
        }
        Allocate(expected);
        input.seekg(0);
        input.read(static_cast<char *>(host), static_cast<std::streamsize>(bytes));
        if (!input) {
            std::fprintf(stderr, "[FAIL] short read: %s\n", path.c_str());
            std::exit(EXIT_FAILURE);
        }
        ACL_CHECK(aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void Save(const std::string &path)
    {
        ACL_CHECK(aclrtMemcpy(host, bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(static_cast<const char *>(host), static_cast<std::streamsize>(bytes));
        if (!output) {
            std::fprintf(stderr, "[FAIL] write failed: %s\n", path.c_str());
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
    const char *value = std::getenv("AIRAN_DATA_DIR");
    return value == nullptr ? "data" : value;
}

int EnvInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    return value == nullptr ? fallback : std::max(0, std::atoi(value));
}

void EnsureDir(const std::string &path)
{
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        std::perror(path.c_str());
        std::exit(EXIT_FAILURE);
    }
}

PuschMimoConfig MakeConfig(uint16_t layers, uint16_t ports)
{
    PuschMimoConfig config {};
    config.abi_version = ABI_VERSION;
    config.struct_size = static_cast<uint16_t>(sizeof(config));
    config.num_layers = layers;
    config.num_tx_ports = ports;
    config.num_rx_antennas = NR;
    config.qm = 8;
    config.num_symbols = N_SYMBOL;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.rb_start = 0;
    config.slot_number = 7;
    config.start_symbol = 0;
    config.num_allocated_symbols = N_SYMBOL;
    config.used_subcarriers = 1596;
    config.padded_subcarriers = N_SC_PAD;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
    for (uint16_t layer = 0; layer < MAX_PUSCH_LAYERS; ++layer) {
        config.dmrs_ports[layer] = static_cast<uint16_t>(1000 + layer);
    }
    config.dmrs_scrambling_id = 37;
    config.data_scrambling_id = 37;
    config.rnti = 0x1234;
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.mapping_type = 0;
    config.num_cdm_groups_without_data = 2;
    config.codebook_enabled = 1;
    config.tpmi = 0;
    config.prg_size_rb = 133;
    return config;
}

MimoDetectLayerPlan MakeLayerPlan(
    const std::vector<PuschMimoConfig> &configs)
{
    MimoDetectLayerPlan plan {};
    plan.abi_version = ABI_VERSION;
    plan.struct_size = static_cast<uint16_t>(sizeof(plan));
    plan.num_rx_antennas = NR;
    plan.layer_capacity = NL;
    plan.num_allocations = static_cast<uint16_t>(configs.size());
    uint16_t offset = 0;
    for (size_t i = 0; i < configs.size(); ++i) {
        plan.allocations[i].pusch_index = static_cast<uint16_t>(i);
        plan.allocations[i].layer_offset = offset;
        plan.allocations[i].num_layers = configs[i].num_layers;
        plan.allocations[i].num_tx_ports = configs[i].num_tx_ports;
        offset = static_cast<uint16_t>(offset + configs[i].num_layers);
    }
    plan.total_layers = offset;
    return plan;
}

}

int main()
{
    constexpr size_t half_bytes = sizeof(uint16_t);
    const std::string root = DataRoot();
    const std::string golden = root + "/golden";
    const std::string output_dir = root + "/ascend_output";
    EnsureDir(root);
    EnsureDir(output_dir);

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    {
        std::vector<PuschMimoConfig> configs;
        configs.push_back(MakeConfig(4, 4));
        if (NR == 32) {
            configs.push_back(MakeConfig(4, 4));
        } else if (NR == 64) {
            configs.push_back(MakeConfig(3, 4));
            configs.push_back(MakeConfig(2, 2));
        }
        const MimoDetectLayerPlan layer_plan = MakeLayerPlan(configs);
        KernelMetadata metadata {};
        if (BuildCurrentProfile(configs.data(), configs.size(), layer_plan,
                                &metadata) != OK) {
            std::fprintf(stderr, "[FAIL] valid common layer plan was rejected\n");
            std::exit(EXIT_FAILURE);
        }
        MimoDetectLayerPlan invalid_plan = layer_plan;
        ++invalid_plan.allocations[1].layer_offset;
        if (BuildCurrentProfile(configs.data(), configs.size(), invalid_plan,
                                &metadata) != PLAN_MISMATCH) {
            std::fprintf(stderr, "[FAIL] non-contiguous layer plan was accepted\n");
            std::exit(EXIT_FAILURE);
        }
        if (BuildCurrentProfile(configs.data(), configs.size(), layer_plan,
                                &metadata) != OK) {
            std::fprintf(stderr, "[FAIL] failed to restore valid kernel metadata\n");
            std::exit(EXIT_FAILURE);
        }

        Buffer rx_re, rx_im, h_re, h_im, noise;
        rx_re.Load(golden + "/rx_grid_re.bin", RX_ELEMS * half_bytes);
        rx_im.Load(golden + "/rx_grid_im.bin", RX_ELEMS * half_bytes);
        h_re.Load(golden + "/h_grid_re.bin", H_ELEMS * half_bytes);
        h_im.Load(golden + "/h_grid_im.bin", H_ELEMS * half_bytes);
        noise.Load(golden + "/noise_var_rx.bin", NOISE_ELEMS * half_bytes);

        Buffer hrm_re, hrm_im, yvpad_re, yvpad_im, no, tiling;
        hrm_re.Allocate(PACKED_ELEMS * half_bytes);
        hrm_im.Allocate(PACKED_ELEMS * half_bytes);
        yvpad_re.Allocate(PACKED_ELEMS * half_bytes);
        yvpad_im.Allocate(PACKED_ELEMS * half_bytes);
        no.Allocate(NO_ELEMS * half_bytes);
        tiling.Allocate(TILING_BYTES);
        GenerateTiling(SOC_VERSION, static_cast<uint8_t *>(tiling.host));
        std::memcpy(tiling.host, &metadata, sizeof(metadata));
        ACL_CHECK(aclrtMemcpy(tiling.device, tiling.bytes, tiling.host, tiling.bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE));
        for (Buffer *buffer : {&hrm_re, &hrm_im, &yvpad_re, &yvpad_im, &no}) {
            ACL_CHECK(aclrtMemset(buffer->device, buffer->bytes, 0x5a, buffer->bytes));
        }

        MimoDetectIoPackOpArgsV1 args {};
        args.abi_version = ABI_VERSION;
        args.struct_size = sizeof(args);
        args.rx_grid_re = rx_re.device;
        args.rx_grid_im = rx_im.device;
        args.h_grid_re = h_re.device;
        args.h_grid_im = h_im.device;
        args.noise_var_rx = noise.device;
        args.hrm_re = hrm_re.device;
        args.hrm_im = hrm_im.device;
        args.yvpad_re = yvpad_re.device;
        args.yvpad_im = yvpad_im.device;
        args.no = no.device;
        args.configs = configs.data();
        args.num_configs = static_cast<uint16_t>(configs.size());
        args.layer_plan = &layer_plan;
        args.stream = stream;
        if (ValidateOpArgs(args) != OK) {
            std::fprintf(stderr, "[FAIL] public OpArgs validation failed\n");
            std::exit(EXIT_FAILURE);
        }

        auto enqueue = [&]() {
            const Status status = Enqueue(args, tiling.device, tiling.bytes);
            if (status != OK) {
                std::fprintf(stderr, "[FAIL] enqueue status=%d\n",
                             static_cast<int>(status));
                std::exit(EXIT_FAILURE);
            }
        };
        for (int i = 0; i < EnvInt("WARMUP", 3); ++i) enqueue();
        ACL_CHECK(aclrtSynchronizeStream(stream));

        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        ACL_CHECK(aclrtCreateEvent(&start));
        ACL_CHECK(aclrtCreateEvent(&end));
        std::vector<double> latency;
        const int timed = EnvInt("TIMED", 20);
        for (int i = 0; i < timed; ++i) {
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
        const double p50 = latency.empty() ? 0.0 : latency[latency.size() / 2];
        std::printf("[case] NR=%u physicalNL=%u activeL=%u allocations=%zu "
                    "grid=[%u,%u] blockDim=%u RE/core=%u p50=%.1f us\n",
                    NR, NL, metadata.active_layers, configs.size(), N_SYMBOL, N_SC_PAD,
                    BLOCK_DIM, RE_PER_CORE, p50);

        hrm_re.Save(output_dir + "/hrm_re.bin");
        hrm_im.Save(output_dir + "/hrm_im.bin");
        yvpad_re.Save(output_dir + "/yvpad_re.bin");
        yvpad_im.Save(output_dir + "/yvpad_im.bin");
        no.Save(output_dir + "/no.bin");
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
