#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_mimo_dmrs_ls_kernel.h"
#include "aclrtlaunch_mimo_dmrs_ls_pack_ce_kernel.h"
#include "mimo_dmrs_ls.h"

using namespace airan::mimo_dmrs_ls;

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

    void Allocate(size_t n, bool withHost = true) {
        bytes = n;
        ACL_CHECK(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        if (withHost) ACL_CHECK(aclrtMallocHost(&host, bytes));
    }
    void Load(const std::string &path) {
        Allocate(FileSize(path));
        std::ifstream input(path, std::ios::binary);
        input.read(static_cast<char *>(host), static_cast<std::streamsize>(bytes));
        if (!input || static_cast<size_t>(input.gcount()) != bytes) {
            std::fprintf(stderr, "[FAIL] cannot read %s\n", path.c_str());
            std::exit(EXIT_FAILURE);
        }
        ACL_CHECK(aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    void ZeroDevice() { ACL_CHECK(aclrtMemset(device, bytes, 0, bytes)); }
    void Save(const std::string &path) {
        ACL_CHECK(aclrtMemcpy(host, bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(static_cast<const char *>(host), static_cast<std::streamsize>(bytes));
        if (!output) {
            std::fprintf(stderr, "[FAIL] cannot write %s\n", path.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    static size_t FileSize(const std::string &path) {
        struct stat info {};
        if (::stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode)) {
            std::fprintf(stderr, "[FAIL] missing input %s\n", path.c_str());
            std::exit(EXIT_FAILURE);
        }
        return static_cast<size_t>(info.st_size);
    }
    ~Buffer() {
        if (device) ACL_CHECK(aclrtFree(device));
        if (host) ACL_CHECK(aclrtFreeHost(host));
    }
};

void EnsureDir(const std::string &path)
{
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        std::perror(path.c_str());
        std::exit(EXIT_FAILURE);
    }
}

std::string DataRoot()
{
    const char *root = std::getenv("AIRAN_DATA_DIR");
    return root ? root : "data";
}

int EnvInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    return value ? std::max(0, std::atoi(value)) : fallback;
}

bool LegacyPackCompatible(const uint16_t *counts, uint32_t numLayers,
                          uint32_t numDmrsSymbols)
{
    return ValidateLmmse798Compatibility(counts, numLayers,
                                         numDmrsSymbols) == OK;
}

double RunLs(aclrtStream stream, Buffer &rxRe, Buffer &rxIm, Buffer &refRe, Buffer &refIm,
             Buffer &outRe, Buffer &outIm, Buffer &pilotSc, Buffer &pilotCount,
             Buffer &noise, Buffer &workspace, Buffer &metadata)
{
    auto enqueue = [&]() {
        ACLRT_LAUNCH_KERNEL(mimo_dmrs_ls_kernel)
            (BLOCK_DIM, stream, rxRe.device, rxIm.device, refRe.device, refIm.device,
             outRe.device, outIm.device, pilotSc.device, pilotCount.device,
             noise.device, workspace.device, metadata.device);
    };
    for (int i = 0; i < EnvInt("WARMUP", 3); ++i) enqueue();
    ACL_CHECK(aclrtSynchronizeStream(stream));
    aclrtEvent start = nullptr, end = nullptr;
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

void RunCase(aclrtStream stream, const std::string &name, Buffer &packIndex)
{
    const std::string root = DataRoot();
    const std::string gold = root + "/golden/" + name;
    const std::string output = root + "/ascend_output/" + name;
    EnsureDir(root + "/ascend_output");
    EnsureDir(output);

    Buffer rxRe, rxIm, refRe, refIm, metadata;
    rxRe.Load(gold + "/rx_re.bin");
    rxIm.Load(gold + "/rx_im.bin");
    refRe.Load(gold + "/dmrs_ref_re.bin");
    refIm.Load(gold + "/dmrs_ref_im.bin");
    metadata.Load(gold + "/metadata.bin");
    KernelMetadata meta {};
    std::memcpy(&meta, metadata.host, sizeof(meta));
    if (meta.magic != META_MAGIC || meta.num_layers == 0 || meta.num_layers > MAX_LAYERS) {
        std::fprintf(stderr, "[FAIL] invalid metadata for %s\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }
    PuschMimoConfig config {};
    config.abi_version = ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = static_cast<uint16_t>(meta.num_layers);
    config.num_tx_ports = static_cast<uint16_t>(meta.num_layers);
    config.num_rx_antennas = NR_CURRENT;
    config.qm = 8;
    config.num_symbols = N_SYMBOLS;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = N_SYMBOLS;
    config.used_subcarriers = N_SC_USED;
    config.padded_subcarriers = N_SC_PAD;
    config.dmrs_type = 1;
    config.num_cdm_groups_without_data = 2;
    for (uint32_t dmrs = 0; dmrs < meta.num_dmrs_symbols; ++dmrs) {
        config.dmrs_symbol_mask |= static_cast<uint16_t>(1u << meta.dmrs_symbols[dmrs]);
    }
    for (uint32_t layer = 0; layer < meta.num_layers; ++layer) {
        config.dmrs_ports[layer] = static_cast<uint16_t>(meta.ports[layer]);
    }
    PuschMimoLayout layout {};
    KernelMetadata derivedMeta {};
    uint16_t derivedCount[COUNT_PAD] = {};
    uint16_t derivedSc[MAX_LAYERS * CURRENT_DMRS_SYMBOLS * N_PILOT_PAD] = {};
    if (BuildCurrentProfile(config, &layout, &derivedMeta, derivedCount, derivedSc) != OK ||
        std::memcmp(&derivedMeta, &meta, sizeof(meta)) != 0) {
        std::fprintf(stderr, "[FAIL] public config did not reproduce kernel metadata for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }

    constexpr size_t halfBytes = sizeof(uint16_t);
    Buffer outRe, outIm, pilotSc, pilotCount, noise, workspace;
    outRe.Allocate(NaturalElems(meta.num_layers) * halfBytes);
    outIm.Allocate(NaturalElems(meta.num_layers) * halfBytes);
    pilotSc.Allocate(static_cast<size_t>(meta.num_layers) * CURRENT_DMRS_SYMBOLS * N_PILOT_PAD * halfBytes);
    pilotCount.Allocate(COUNT_PAD * halfBytes);
    noise.Allocate(NR_CURRENT * halfBytes);
    workspace.Allocate(128, false);
    outRe.ZeroDevice(); outIm.ZeroDevice(); pilotSc.ZeroDevice();
    pilotCount.ZeroDevice(); noise.ZeroDevice(); workspace.ZeroDevice();

    MimoDmrsLsOpArgsV1 publicArgs {};
    publicArgs.abi_version = ABI_VERSION;
    publicArgs.struct_size = sizeof(publicArgs);
    publicArgs.rx_grid_re = rxRe.device;
    publicArgs.rx_grid_im = rxIm.device;
    publicArgs.dmrs_ref_re = refRe.device;
    publicArgs.dmrs_ref_im = refIm.device;
    publicArgs.h_ls_re = outRe.device;
    publicArgs.h_ls_im = outIm.device;
    publicArgs.pilot_sc = pilotSc.device;
    publicArgs.pilot_count = pilotCount.device;
    publicArgs.noise_var_rx = noise.device;
    publicArgs.config = &config;
    publicArgs.layout = &layout;
    publicArgs.stream = stream;
    if (ValidateOpArgs(publicArgs) != OK) {
        std::fprintf(stderr, "[FAIL] public OpArgs validation failed for %s\n", name.c_str());
        std::exit(EXIT_FAILURE);
    }

    const double us = RunLs(stream, rxRe, rxIm, refRe, refIm, outRe, outIm,
                            pilotSc, pilotCount, noise, workspace, metadata);
    std::printf("[case] %-24s L=%u LS_p50=%.1f us", name.c_str(), meta.num_layers, us);
    outRe.Save(output + "/h_ls_re.bin");
    outIm.Save(output + "/h_ls_im.bin");
    pilotSc.Save(output + "/pilot_sc.bin");
    pilotCount.Save(output + "/pilot_count.bin");
    noise.Save(output + "/noise_var_rx.bin");

    const auto *actualCounts = static_cast<const uint16_t *>(pilotCount.host);
    ObservationModel models[MAX_LAYERS] = {};
    const auto *actualSc = static_cast<const uint16_t *>(pilotSc.host);
    if (DescribeObservationModels(actualCounts, meta.num_layers,
                                  meta.num_dmrs_symbols, models) != OK ||
        ValidateNaturalLmmseContract(actualCounts, actualSc, meta.num_layers,
                                     meta.num_dmrs_symbols) != OK) {
        std::fprintf(stderr, "[FAIL] invalid natural CE observation model for %s\n",
                     name.c_str());
        std::exit(EXIT_FAILURE);
    }
    {
        std::ofstream marker(output + "/ce_natural_contract.txt");
        marker << "PASS";
        for (uint32_t layer = 0; layer < meta.num_layers; ++layer) {
            marker << (models[layer] == FD_OCC2_399 ? " FD_OCC2_399" : " COMB2_798");
        }
        marker << '\n';
    }

    if (!LegacyPackCompatible(actualCounts, meta.num_layers, meta.num_dmrs_symbols)) {
        std::ofstream(output + "/legacy_ce_pack_skipped.txt")
            << "legacy 798-only pack skipped; canonical natural CE contract is valid\n";
        std::printf(" CE-natural=PASS legacy-pack=skipped\n");
        return;
    }

    Buffer ceRe, ceIm, ceNegIm;
    ceRe.Allocate(CePackedElems() * halfBytes);
    ceIm.Allocate(CePackedElems() * halfBytes);
    ceNegIm.Allocate(CePackedElems() * halfBytes);
    ceRe.ZeroDevice(); ceIm.ZeroDevice(); ceNegIm.ZeroDevice();
    ACLRT_LAUNCH_KERNEL(mimo_dmrs_ls_pack_ce_kernel)
        (MAX_LAYERS, stream, outRe.device, outIm.device, packIndex.device,
         ceRe.device, ceIm.device, ceNegIm.device, workspace.device, metadata.device);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ceRe.Save(output + "/ce_hls_re.bin");
    ceIm.Save(output + "/ce_hls_im.bin");
    ceNegIm.Save(output + "/ce_hls_neg_im.bin");
    std::printf(" CE-natural=PASS legacy-pack=PASS\n");
}

}  // namespace

int main()
{
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    {
        Buffer packIndex;
        packIndex.Load(DataRoot() + "/weights/ce_pack_gather_index.bin");
        for (const char *name : {"case_rank1", "case_rank2_disjoint",
                                 "case_rank3_mixed", "case_rank4_occ"}) {
            RunCase(stream, name, packIndex);
        }
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
