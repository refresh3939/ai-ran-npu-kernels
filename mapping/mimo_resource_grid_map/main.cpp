#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_mimo_resource_grid_map_kernel.h"
#include "mimo_resource_grid_map.h"
#include "tiling/platform/platform_ascendc.h"

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

namespace {
namespace rgm = airan::mimo_resource_grid_map;
constexpr uint32_t N_WARMUP = 10;
constexpr uint32_t N_TIMED = 30;
constexpr uint32_t N_BENCH_BATCH = 100;

#define CHECK_ACL(call) do { \
    const aclError error = (call); \
    if (error != ACL_SUCCESS) { \
        std::fprintf(stderr, "ACL error %d at %s:%d\n", static_cast<int>(error), __FILE__, __LINE__); \
        std::exit(2); \
    } \
} while (0)

struct TestCase {
    const char *name;
    uint16_t layers;
    uint16_t ports[4];
    uint16_t dmrs_mask;
};

constexpr TestCase CASES[] = {
    {"case_0_rank1_port1000", 1, {1000, 0, 0, 0}, static_cast<uint16_t>((1u << 2) | (1u << 11))},
    {"case_1_rank2_shared", 2, {1000, 1001, 0, 0}, static_cast<uint16_t>((1u << 2) | (1u << 11))},
    {"case_2_rank2_disjoint", 2, {1000, 1002, 0, 0}, static_cast<uint16_t>((1u << 2) | (1u << 11))},
    {"case_3_rank3_reordered", 3, {1003, 1000, 1002, 0}, static_cast<uint16_t>((1u << 2) | (1u << 11))},
    {"case_4_rank4_alt_mask", 4, {1003, 1002, 1001, 1000}, static_cast<uint16_t>((1u << 1) | (1u << 12))},
};

std::string Root()
{
    const char *root = std::getenv("AIRAN_DATA_DIR");
    return root == nullptr ? "." : root;
}

bool ReadExact(const std::string &path, void *destination, size_t bytes)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.is_open() || static_cast<size_t>(stream.tellg()) != bytes) {
        std::fprintf(stderr, "[error] %s: expected %zu bytes\n", path.c_str(), bytes);
        return false;
    }
    stream.seekg(0);
    stream.read(static_cast<char *>(destination), static_cast<std::streamsize>(bytes));
    return stream.good();
}

bool WriteExact(const std::string &path, const void *source, size_t bytes)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(static_cast<const char *>(source), static_cast<std::streamsize>(bytes));
    return stream.good();
}

rgm::PuschMimoConfig MakeConfig(const TestCase &test)
{
    rgm::PuschMimoConfig config {};
    config.abi_version = rgm::ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = test.layers;
    config.num_tx_ports = test.layers == 1 ? 1 : (test.layers == 2 ? 2 : 4);
    config.num_rx_antennas = 64;
    config.qm = 8;
    config.num_symbols = rgm::N_SYMBOLS;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = rgm::N_SYMBOLS;
    config.used_subcarriers = rgm::N_SC_USED;
    config.padded_subcarriers = rgm::N_SC_PAD;
    config.dmrs_symbol_mask = test.dmrs_mask;
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    for (uint32_t layer = 0; layer < test.layers; ++layer) config.dmrs_ports[layer] = test.ports[layer];
    return config;
}

size_t CountDifferences(const uint16_t *left, const uint16_t *right, size_t count)
{
    size_t differences = 0;
    for (size_t i = 0; i < count; ++i) differences += left[i] != right[i] ? 1u : 0u;
    return differences;
}

}

int main()
{
    const char *socVersion = SOC_VERSION;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    const size_t workspaceBytes = static_cast<size_t>(platform->GetLibApiWorkSpaceSize());

    const size_t maxDataBytes = rgm::DataElems(rgm::MAX_LAYERS) * sizeof(uint16_t);
    const size_t maxDmrsBytes = rgm::DmrsElems(rgm::MAX_LAYERS) * sizeof(uint16_t);
    const size_t maxGridBytes = rgm::GridElems(rgm::MAX_LAYERS) * sizeof(uint16_t);
    const size_t dataOffsetBytes = rgm::N_DATA_PAD * sizeof(uint32_t);
    const size_t dmrsOffsetBytes = rgm::DmrsOffsetElems() * sizeof(uint32_t);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *layerReD = nullptr, *layerImD = nullptr, *dmrsReD = nullptr, *dmrsImD = nullptr;
    void *gridReD = nullptr, *gridImD = nullptr, *dataOffsetD = nullptr, *dmrsOffsetD = nullptr;
    void *tilingD = nullptr, *workspaceD = nullptr;
    CHECK_ACL(aclrtMalloc(&layerReD, maxDataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&layerImD, maxDataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dmrsReD, maxDmrsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dmrsImD, maxDmrsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&gridReD, maxGridBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&gridImD, maxGridBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dataOffsetD, dataOffsetBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dmrsOffsetD, dmrsOffsetBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&tilingD, rgm::TILING_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&workspaceD, std::max<size_t>(workspaceBytes, 32), ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<uint16_t> layerRe(rgm::DataElems(rgm::MAX_LAYERS));
    std::vector<uint16_t> layerIm(rgm::DataElems(rgm::MAX_LAYERS));
    std::vector<uint16_t> dmrsRe(rgm::DmrsElems(rgm::MAX_LAYERS));
    std::vector<uint16_t> dmrsIm(rgm::DmrsElems(rgm::MAX_LAYERS));
    std::vector<uint16_t> gridRe(rgm::GridElems(rgm::MAX_LAYERS));
    std::vector<uint16_t> gridIm(rgm::GridElems(rgm::MAX_LAYERS));
    std::vector<uint16_t> goldenRe(rgm::GridElems(rgm::MAX_LAYERS));
    std::vector<uint16_t> goldenIm(rgm::GridElems(rgm::MAX_LAYERS));
    std::vector<uint16_t> referenceRe(rgm::GridElems(rgm::MAX_LAYERS));
    std::vector<uint16_t> referenceIm(rgm::GridElems(rgm::MAX_LAYERS));
    std::vector<uint32_t> dataOffset(rgm::N_DATA_PAD);
    std::vector<uint32_t> dmrsOffset(rgm::DmrsOffsetElems());
    std::vector<uint32_t> goldenDataOffset(rgm::N_DATA_PAD);
    std::vector<uint32_t> goldenDmrsOffset(rgm::DmrsOffsetElems());
    std::vector<uint8_t> tiling(rgm::TILING_BYTES);
    GenerateTiling(socVersion, tiling.data());

    uint32_t passed = 0;
    for (const TestCase &test : CASES) {
        const size_t dataBytes = rgm::DataElems(test.layers) * sizeof(uint16_t);
        const size_t dmrsBytes = rgm::DmrsElems(test.layers) * sizeof(uint16_t);
        const size_t gridBytes = rgm::GridElems(test.layers) * sizeof(uint16_t);
        const std::string base = Root() + "/data/golden/" + test.name + "/";
        std::fill(layerRe.begin(), layerRe.end(), uint16_t{0});
        std::fill(layerIm.begin(), layerIm.end(), uint16_t{0});
        std::fill(dmrsRe.begin(), dmrsRe.end(), uint16_t{0});
        std::fill(dmrsIm.begin(), dmrsIm.end(), uint16_t{0});
        bool filesOk = ReadExact(base + "layer_re.bin", layerRe.data(), dataBytes) &&
                       ReadExact(base + "layer_im.bin", layerIm.data(), dataBytes) &&
                       ReadExact(base + "dmrs_re.bin", dmrsRe.data(), dmrsBytes) &&
                       ReadExact(base + "dmrs_im.bin", dmrsIm.data(), dmrsBytes) &&
                       ReadExact(base + "grid_re.bin", goldenRe.data(), gridBytes) &&
                       ReadExact(base + "grid_im.bin", goldenIm.data(), gridBytes) &&
                       ReadExact(base + "data_dst_offset.bin", goldenDataOffset.data(), dataOffsetBytes) &&
                       ReadExact(base + "dmrs_dst_offset.bin", goldenDmrsOffset.data(), dmrsOffsetBytes);
        if (!filesOk) continue;

        const rgm::PuschMimoConfig config = MakeConfig(test);
        rgm::PuschMimoLayout layout {};
        rgm::KernelMetadata metadata {};
        if (rgm::BuildCurrentProfile(config, &layout, &metadata,
                                     dataOffset.data(), dmrsOffset.data()) != rgm::OK) {
            std::printf("  %-26s [FAIL: profile]\n", test.name);
            continue;
        }
        const size_t offsetDiff =
            std::memcmp(dataOffset.data(), goldenDataOffset.data(), dataOffsetBytes) != 0 ||
            std::memcmp(dmrsOffset.data(), goldenDmrsOffset.data(), dmrsOffsetBytes) != 0;
        if (offsetDiff != 0 || rgm::ReferenceMap(layerRe.data(), layerIm.data(), dmrsRe.data(),
                                                 dmrsIm.data(), config, layout, dataOffset.data(),
                                                 dmrsOffset.data(), referenceRe.data(),
                                                 referenceIm.data()) != rgm::OK) {
            std::printf("  %-26s [FAIL: resources/reference]\n", test.name);
            continue;
        }
        const size_t hostDiff = CountDifferences(referenceRe.data(), goldenRe.data(), gridBytes / 2) +
                                CountDifferences(referenceIm.data(), goldenIm.data(), gridBytes / 2);

        rgm::MimoResourceGridMapOpArgsV1 args {};
        args.abi_version = rgm::ABI_VERSION;
        args.struct_size = sizeof(args);
        args.layer_re = layerReD;
        args.layer_im = layerImD;
        args.dmrs_re = dmrsReD;
        args.dmrs_im = dmrsImD;
        args.layer_grid_re = gridReD;
        args.layer_grid_im = gridImD;
        args.config = &config;
        args.layout = &layout;
        args.stream = stream;
        if (rgm::ValidateOpArgs(args) != rgm::OK) {
            std::printf("  %-26s [FAIL: public args]\n", test.name);
            continue;
        }

        CHECK_ACL(aclrtMemcpy(layerReD, maxDataBytes, layerRe.data(), maxDataBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(layerImD, maxDataBytes, layerIm.data(), maxDataBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(dmrsReD, maxDmrsBytes, dmrsRe.data(), maxDmrsBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(dmrsImD, maxDmrsBytes, dmrsIm.data(), maxDmrsBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemset(gridReD, maxGridBytes, 0xff, maxGridBytes));
        CHECK_ACL(aclrtMemset(gridImD, maxGridBytes, 0xff, maxGridBytes));
        CHECK_ACL(aclrtMemcpy(dataOffsetD, dataOffsetBytes, dataOffset.data(), dataOffsetBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(dmrsOffsetD, dmrsOffsetBytes, dmrsOffset.data(), dmrsOffsetBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        std::memcpy(tiling.data(), &metadata, sizeof(metadata));
        CHECK_ACL(aclrtMemcpy(tilingD, rgm::TILING_BYTES, tiling.data(), rgm::TILING_BYTES,
                              ACL_MEMCPY_HOST_TO_DEVICE));

        ACLRT_LAUNCH_KERNEL(mimo_resource_grid_map_kernel)(
            rgm::BLOCK_DIM, stream, layerReD, layerImD, dmrsReD, dmrsImD,
            dataOffsetD, dmrsOffsetD, gridReD, gridImD, workspaceD, tilingD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        CHECK_ACL(aclrtMemcpy(gridRe.data(), maxGridBytes, gridReD, maxGridBytes,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(gridIm.data(), maxGridBytes, gridImD, maxGridBytes,
                              ACL_MEMCPY_DEVICE_TO_HOST));

        const size_t deviceDiff = CountDifferences(gridRe.data(), goldenRe.data(), gridBytes / 2) +
                                  CountDifferences(gridIm.data(), goldenIm.data(), gridBytes / 2);
        const bool ok = hostDiff == 0 && deviceDiff == 0;
        std::printf("  %-26s L=%u host_diff=%zu npu_diff=%zu %s\n", test.name,
                    test.layers, hostDiff, deviceDiff, ok ? "[PASS]" : "[FAIL]");
        const std::string output = Root() + "/data/ascend_output/" + test.name;
        WriteExact(output + "_re.bin", gridRe.data(), gridBytes);
        WriteExact(output + "_im.bin", gridIm.data(), gridBytes);
        passed += ok ? 1u : 0u;
    }


    rgm::PuschMimoConfig invalid = MakeConfig(CASES[0]);
    rgm::PuschMimoLayout ignoredLayout {};
    rgm::KernelMetadata ignoredMetadata {};
    invalid.num_layers = 0;
    bool rejects = rgm::BuildCurrentProfile(invalid, &ignoredLayout, &ignoredMetadata,
                                             dataOffset.data(), dmrsOffset.data()) == rgm::UNSUPPORTED_PROFILE;
    invalid = MakeConfig(CASES[0]);
    invalid.reserved[0] = 1;
    rejects &= rgm::BuildCurrentProfile(invalid, &ignoredLayout, &ignoredMetadata,
                                         dataOffset.data(), dmrsOffset.data()) == rgm::UNSUPPORTED_PROFILE;
    std::printf("  contract rejection checks: %s\n", rejects ? "[PASS]" : "[FAIL]");
    std::printf("[mimo_resource_grid_map] %u/%zu device cases PASS\n",
                passed, sizeof(CASES) / sizeof(CASES[0]));


    for (uint32_t iteration = 0; iteration < N_WARMUP; ++iteration) {
        ACLRT_LAUNCH_KERNEL(mimo_resource_grid_map_kernel)(
            rgm::BLOCK_DIM, stream, layerReD, layerImD, dmrsReD, dmrsImD,
            dataOffsetD, dmrsOffsetD, gridReD, gridImD, workspaceD, tilingD);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    CHECK_ACL(aclrtCreateEvent(&start));
    CHECK_ACL(aclrtCreateEvent(&end));
    std::vector<float> latencyUs;
    latencyUs.reserve(N_TIMED);
    for (uint32_t sample = 0; sample < N_TIMED; ++sample) {
        CHECK_ACL(aclrtRecordEvent(start, stream));
        for (uint32_t batch = 0; batch < N_BENCH_BATCH; ++batch) {
            ACLRT_LAUNCH_KERNEL(mimo_resource_grid_map_kernel)(
                rgm::BLOCK_DIM, stream, layerReD, layerImD, dmrsReD, dmrsImD,
                dataOffsetD, dmrsOffsetD, gridReD, gridImD, workspaceD, tilingD);
        }
        CHECK_ACL(aclrtRecordEvent(end, stream));
        CHECK_ACL(aclrtSynchronizeEvent(end));
        float elapsedMs = 0.0f;
        CHECK_ACL(aclrtEventElapsedTime(&elapsedMs, start, end));
        latencyUs.push_back(elapsedMs * 1000.0f / static_cast<float>(N_BENCH_BATCH));
    }
    CHECK_ACL(aclrtDestroyEvent(start));
    CHECK_ACL(aclrtDestroyEvent(end));
    std::sort(latencyUs.begin(), latencyUs.end());
    float latencySum = 0.0f;
    for (float value : latencyUs) latencySum += value;
    std::printf("[benchmark] Rank-4 min=%.2f avg=%.2f p50=%.2f p99=%.2f max=%.2f us\n",
                latencyUs.front(), latencySum / latencyUs.size(), latencyUs[N_TIMED / 2],
                latencyUs[(N_TIMED * 99) / 100], latencyUs.back());

    CHECK_ACL(aclrtFree(layerReD)); CHECK_ACL(aclrtFree(layerImD));
    CHECK_ACL(aclrtFree(dmrsReD)); CHECK_ACL(aclrtFree(dmrsImD));
    CHECK_ACL(aclrtFree(gridReD)); CHECK_ACL(aclrtFree(gridImD));
    CHECK_ACL(aclrtFree(dataOffsetD)); CHECK_ACL(aclrtFree(dmrsOffsetD));
    CHECK_ACL(aclrtFree(tilingD)); CHECK_ACL(aclrtFree(workspaceD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return passed == sizeof(CASES) / sizeof(CASES[0]) && rejects ? 0 : 1;
}
