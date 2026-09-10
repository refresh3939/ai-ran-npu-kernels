#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_channel_est_lmmse_pack_natural_kernel.h"
#include "aclrtlaunch_channel_est_lmmse_pad_layers_kernel.h"
#include "aclrtlaunch_channel_est_lmmse_kernel.h"
#include "data_utils.h"
#include "channel_est_lmmse.h"
#include "pusch_mimo_runtime_config.h"

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);
using namespace airan::channel_est_lmmse;

namespace {

struct DeviceFile {
    uint8_t *host = nullptr;
    uint8_t *device = nullptr;
    size_t bytes = 0;
};

int EnvInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    return value ? std::max(0, std::atoi(value)) : fallback;
}

std::string DataRoot()
{
    const char *value = std::getenv("AIRAN_DATA_DIR");
    return value ? std::string(value) : std::string("data");
}

void LoadInput(const std::string &path, size_t bytes, DeviceFile &file)
{
    file.bytes = bytes;
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&file.host), bytes));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&file.device), bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    size_t actual = 0;
    if (!ReadFile(path, actual, file.host, bytes) || actual != bytes) {
        std::fprintf(stderr, "[FAIL] input size mismatch: %s expected=%zu actual=%zu\n",
                     path.c_str(), bytes, actual);
        std::exit(EXIT_FAILURE);
    }
    CHECK_ACL(aclrtMemcpy(file.device, bytes, file.host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}

void Release(DeviceFile &file)
{
    if (file.device) CHECK_ACL(aclrtFree(file.device));
    if (file.host) CHECK_ACL(aclrtFreeHost(file.host));
}

void EnsureDirectory(const std::string &path)
{
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        std::perror(path.c_str());
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main()
{
    static_assert(sizeof(uint16_t) == sizeof(aclFloat16), "host half size mismatch");
    constexpr size_t halfBytes = sizeof(uint16_t);
    const std::string root = DataRoot();
    const std::string gold = root + "/golden/" + CASE_NAME;
    const std::string output = root + "/ascend_output/" + CASE_NAME;
    airan::PuschMimoRuntimeConfig runtime {};
    bool runtimeEnabled = false;
    std::string runtimeWhy;
    if (airan::LoadMimoRuntimeConfigFromEnv(
            NL, &runtime, &runtimeEnabled, &runtimeWhy) !=
        airan::MimoRuntimeConfigStatus::kSuccess) {
        std::fprintf(stderr, "[FAIL] runtime config: %s\n", runtimeWhy.c_str());
        return EXIT_FAILURE;
    }
    if (runtimeEnabled &&
        (runtime.rx_bucket != NR ||
         runtime.layer_bucket != DETECTOR_LAYERS ||
         runtime.num_dmrs_symbols != N_DMRS_SYMBOL ||
         runtime.dmrs_port_count != NL || runtime.num_symbols != N_SYMBOL ||
         runtime.used_subcarriers != N_SC_USED ||
         runtime.padded_subcarriers != N_SC_PAD ||
         runtime.grid_re_per_port != N_SYMBOL * N_SC_PAD)) {
        std::fprintf(stderr,
                     "[FAIL] runtime config exceeds compiled CE profile "
                     "NR=%u NL=%u rank=%u\n", NR, NL, RANK);
        return EXIT_FAILURE;
    }
    PuschMimoConfig config {};
    if (runtimeEnabled) {
        if (airan::DerivePuschMimoConfig(runtime, NR, &config, &runtimeWhy) !=
            airan::MimoRuntimeConfigStatus::kSuccess) {
            std::fprintf(stderr, "[FAIL] runtime config derivation: %s\n",
                         runtimeWhy.c_str());
            return EXIT_FAILURE;
        }
    } else {
        config.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
        config.struct_size = sizeof(config);
        config.num_layers = NL;
        config.num_tx_ports = NL;
        config.num_rx_antennas = NR;
        config.qm = 8;
        config.num_symbols = N_SYMBOL;
        config.used_subcarriers = N_SC_USED;
        config.padded_subcarriers = N_SC_PAD;
        config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2u) | (1u << 11u));
        config.dmrs_type = 1;
        config.dmrs_ports[0] = 1000;
        if (NL == 2) {
            config.dmrs_ports[1] = 1002;
        } else if (NL >= 3) {
            config.dmrs_ports[1] = 1001;
            config.dmrs_ports[2] = 1002;
            if (NL == 4) config.dmrs_ports[3] = 1003;
        }
    }
    EnsureDirectory(root);
    EnsureDirectory(root + "/ascend_output");
    EnsureDirectory(output);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    DeviceFile br, bi, naturalRe, naturalIm, pilotCount, pilotSc, packIndex, weightModelFile;
    DeviceFile ar, ai, ani, wr, wi;
    LoadInput(gold + "/factor_b_re.bin", B_ELEMS * halfBytes, br);
    LoadInput(gold + "/factor_b_im.bin", B_ELEMS * halfBytes, bi);
    LoadInput(gold + "/h_ls_re.bin", NATURAL_HLS_ELEMS * halfBytes, naturalRe);
    LoadInput(gold + "/h_ls_im.bin", NATURAL_HLS_ELEMS * halfBytes, naturalIm);
    LoadInput(gold + "/pilot_count.bin", PILOT_COUNT_PAD * sizeof(uint16_t), pilotCount);
    LoadInput(gold + "/pilot_sc.bin",
              static_cast<size_t>(NL) * N_DMRS_SYMBOL * N_PILOT_PAD * sizeof(uint16_t), pilotSc);
    LoadInput(gold + "/ce_pack_gather_index.bin", PACK_INDEX_ELEMS * sizeof(uint32_t), packIndex);
    LoadInput(gold + "/weight_model.bin", sizeof(LmmseWeightModelV1), weightModelFile);
#if defined(CE_CUBE_TIME_POST_GEMM) && CE_CUBE_TIME_POST_GEMM
    LoadInput(gold + "/factor_a_re.bin", A_ELEMS * halfBytes, ar);
    LoadInput(gold + "/factor_a_im.bin", A_ELEMS * halfBytes, ai);
    LoadInput(gold + "/factor_a_neg_im.bin", A_ELEMS * halfBytes, ani);
    LoadInput(gold + "/cube_time_post_matrix.bin", WT_ELEMS * halfBytes, wr);
#elif defined(CE_CUBE_TIME_FUSED) && CE_CUBE_TIME_FUSED
    LoadInput(gold + "/cube_time_fused_d0_re.bin", A_ELEMS * halfBytes, ar);
    LoadInput(gold + "/cube_time_fused_d0_im.bin", A_ELEMS * halfBytes, ai);
    LoadInput(gold + "/cube_time_fused_d1_re.bin", A_ELEMS * halfBytes, ani);
    LoadInput(gold + "/cube_time_fused_d1_im.bin", WT_ELEMS * halfBytes, wr);
    LoadInput(gold + "/cube_time_fused_d1_im.bin", WT_ELEMS * halfBytes, wi);
#else
    LoadInput(gold + "/factor_a_re.bin", A_ELEMS * halfBytes, ar);
    LoadInput(gold + "/factor_a_im.bin", A_ELEMS * halfBytes, ai);
    LoadInput(gold + "/factor_a_neg_im.bin", A_ELEMS * halfBytes, ani);
    LoadInput(gold + "/wt_re.bin", WT_ELEMS * halfBytes, wr);
    LoadInput(gold + "/wt_im.bin", WT_ELEMS * halfBytes, wi);
#endif

    uint8_t *hr = nullptr, *hi = nullptr, *hn = nullptr;
    uint8_t *tRe = nullptr, *tIm = nullptr, *outRe = nullptr, *outIm = nullptr;
    uint8_t *paddedRe = nullptr, *paddedIm = nullptr;
    uint8_t *workspace = nullptr, *tilingDevice = nullptr, *outReHost = nullptr, *outImHost = nullptr;
    const size_t hlsBytes = HLS_ELEMS * halfBytes;
    const size_t tReBytes = T_ELEMS * halfBytes;
    const size_t tImBytes = T_ELEMS * halfBytes;
    const size_t activeOutputBytes = OUT_ELEMS * halfBytes;
    const size_t paddedOutputBytes = PADDED_OUT_ELEMS * halfBytes;
    // The kernel only uses workspace for four-core software SyncAll.  Keep the
    // same conservative 4 MiB allocation as the proven predecessor instead of
    // querying PlatformAscendCManager at runtime (that query is not stable on
    // all 310P driver/toolkit combinations).
    constexpr size_t workspaceBytes = 4 * 1024 * 1024;

    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&hr), hlsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&hi), hlsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&hn), hlsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&tRe), tReBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&tIm), tImBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&outRe), activeOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&outIm), activeOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&paddedRe), paddedOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&paddedIm), paddedOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&workspace), workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&tilingDevice), TILING_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&outReHost), paddedOutputBytes));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&outImHost), paddedOutputBytes));

    std::vector<uint8_t> tiling(TILING_BYTES);
    GenerateTiling(SOC_VERSION, tiling.data());
    CHECK_ACL(aclrtMemcpy(tilingDevice, TILING_BYTES, tiling.data(), TILING_BYTES,
                         ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(hr, hlsBytes, 0, hlsBytes));
    CHECK_ACL(aclrtMemset(hi, hlsBytes, 0, hlsBytes));
    CHECK_ACL(aclrtMemset(hn, hlsBytes, 0, hlsBytes));
    CHECK_ACL(aclrtMemset(outRe, activeOutputBytes, 0, activeOutputBytes));
    CHECK_ACL(aclrtMemset(outIm, activeOutputBytes, 0, activeOutputBytes));
    CHECK_ACL(aclrtMemset(paddedRe, paddedOutputBytes, 0, paddedOutputBytes));
    CHECK_ACL(aclrtMemset(paddedIm, paddedOutputBytes, 0, paddedOutputBytes));

    PuschMimoLayout layout {};
    layout.num_dmrs_symbols = N_DMRS_SYMBOL;
    layout.num_data_symbols = N_SYMBOL - N_DMRS_SYMBOL;
    layout.num_data_re = layout.num_data_symbols * N_SC_USED;
    layout.data_stride = (layout.num_data_re + 127u) / 128u * 128u;
    layout.codeword_symbols = NL * layout.num_data_re;
    layout.codeword_stride = NL * layout.data_stride;
    auto *countHost = reinterpret_cast<const uint16_t *>(pilotCount.host);
    auto *scHost = reinterpret_cast<const uint16_t *>(pilotSc.host);
    auto *weightModel = reinterpret_cast<const LmmseWeightModelV1 *>(weightModelFile.host);
    ChannelEstLmmseOpArgsV1 publicArgs {};
    publicArgs.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
    publicArgs.struct_size = sizeof(publicArgs);
    publicArgs.h_ls_re = naturalRe.device;
    publicArgs.h_ls_im = naturalIm.device;
    publicArgs.pilot_count = pilotCount.device;
    publicArgs.pilot_sc = pilotSc.device;
    publicArgs.pilot_count_host = countHost;
    publicArgs.pilot_sc_host = scHost;
    publicArgs.weight_model = weightModel;
    publicArgs.h_grid_re = paddedRe;
    publicArgs.h_grid_im = paddedIm;
    publicArgs.config = &config;
    publicArgs.layout = &layout;
    publicArgs.stream = stream;
    const Status contractStatus = ValidateOpArgs(publicArgs);
    if (contractStatus != OK) {
        std::fprintf(stderr, "[FAIL] public MIMO contract rejected status=%d\n",
                     static_cast<int>(contractStatus));
        return EXIT_FAILURE;
    }
    ACLRT_LAUNCH_KERNEL(channel_est_lmmse_pack_natural_kernel)
        (BLOCK_DIM, stream, naturalRe.device, naturalIm.device, packIndex.device,
         hr, hi, hn, workspace, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    auto prepare = [&]() {
        // Phase-A uses atomic add so only its small scratch and SyncAll workspace
        // must be cleared per launch; the output is fully overwritten.
        CHECK_ACL(aclrtMemset(tRe, tReBytes, 0, tReBytes));
        CHECK_ACL(aclrtMemset(tIm, tImBytes, 0, tImBytes));
        CHECK_ACL(aclrtMemset(workspace, workspaceBytes, 0, workspaceBytes));
    };
    auto enqueueKernel = [&]() {
        ACLRT_LAUNCH_KERNEL(channel_est_lmmse_kernel)
        (BLOCK_DIM, stream,
         br.device, bi.device, hr, hi, hn,
         ar.device, ai.device, ani.device, wr.device,
#if defined(CE_CUBE_TIME_POST_GEMM) && CE_CUBE_TIME_POST_GEMM
         wr.device,
#else
         wi.device,
#endif
         tRe, tIm, outRe, outIm, workspace, tilingDevice);
    };
    auto launch = [&]() {
        prepare();
        enqueueKernel();
        CHECK_ACL(aclrtSynchronizeStream(stream));
    };

    const int warmup = EnvInt("WARMUP", 3);
    const int timed = EnvInt("TIMED", 20);
    std::printf("[case] %s | NR=%u NL=%u rank=%u blockDim=%u runtime=%s\n",
                CASE_NAME, NR, NL, RANK, BLOCK_DIM,
                runtimeEnabled ? "profile" : "legacy");
#if defined(CE_CUBE_TIME_POST_GEMM) && CE_CUBE_TIME_POST_GEMM
    std::printf("[variant] cube_time_post_gemm=ON; output layout=[rx,layer,symbol,sc]\n");
#elif defined(CE_CUBE_TIME_FUSED) && CE_CUBE_TIME_FUSED
    std::printf("[variant] cube_time_fused=ON; output layout=[rx,layer,symbol,sc]\n");
#else
    std::printf("[variant] cube_time_fused=OFF; output layout=[rx,layer,symbol,sc]\n");
#endif
    std::printf("[contract] natural h_ls=[%u,%u,2,832], pilots checked; "
                "public H=[%u,16,14,1664]\n", NR, NL, NR);
    for (int i = 0; i < warmup; ++i) launch();

    aclrtEvent kernelStart = nullptr, kernelEnd = nullptr;
    CHECK_ACL(aclrtCreateEvent(&kernelStart));
    CHECK_ACL(aclrtCreateEvent(&kernelEnd));
    std::vector<double> latency, kernelLatency;
    latency.reserve(timed);
    kernelLatency.reserve(timed);
    for (int i = 0; i < timed; ++i) {
        const auto start = std::chrono::steady_clock::now();
        prepare();
        CHECK_ACL(aclrtRecordEvent(kernelStart, stream));
        enqueueKernel();
        CHECK_ACL(aclrtRecordEvent(kernelEnd, stream));
        CHECK_ACL(aclrtSynchronizeEvent(kernelEnd));
        const auto end = std::chrono::steady_clock::now();
        latency.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        float kernelMs = 0.0f;
        CHECK_ACL(aclrtEventElapsedTime(&kernelMs, kernelStart, kernelEnd));
        kernelLatency.push_back(static_cast<double>(kernelMs) * 1000.0);
    }
    if (!latency.empty()) {
        std::sort(latency.begin(), latency.end());
        double sum = 0.0;
        for (double value : latency) sum += value;
        std::printf("[latency] host wall incl. scratch memset: min=%.1f avg=%.1f p50=%.1f us\n",
                    latency.front(), sum / latency.size(), latency[latency.size() / 2]);
        std::sort(kernelLatency.begin(), kernelLatency.end());
        sum = 0.0;
        for (double value : kernelLatency) sum += value;
        std::printf("[latency] device kernel only: min=%.1f avg=%.1f p50=%.1f us\n",
                    kernelLatency.front(), sum / kernelLatency.size(),
                    kernelLatency[kernelLatency.size() / 2]);
    }

    CHECK_ACL(aclrtMemset(paddedRe, paddedOutputBytes, 0, paddedOutputBytes));
    CHECK_ACL(aclrtMemset(paddedIm, paddedOutputBytes, 0, paddedOutputBytes));
    ACLRT_LAUNCH_KERNEL(channel_est_lmmse_pad_layers_kernel)
        (BLOCK_DIM, stream, outRe, outIm, paddedRe, paddedIm, workspace, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(outReHost, paddedOutputBytes, paddedRe, paddedOutputBytes,
                         ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(outImHost, paddedOutputBytes, paddedIm, paddedOutputBytes,
                         ACL_MEMCPY_DEVICE_TO_HOST));
#if defined(CE_CUBE_TIME_POST_GEMM) && CE_CUBE_TIME_POST_GEMM
    const char *outReName = "/h_cube_time_post_re.bin";
    const char *outImName = "/h_cube_time_post_im.bin";
#elif defined(CE_CUBE_TIME_FUSED) && CE_CUBE_TIME_FUSED
    const char *outReName = "/h_cube_time_fused_re.bin";
    const char *outImName = "/h_cube_time_fused_im.bin";
#else
    const char *outReName = "/h_re.bin";
    const char *outImName = "/h_im.bin";
#endif
    if (!WriteFile(output + outReName, outReHost, paddedOutputBytes) ||
        !WriteFile(output + outImName, outImHost, paddedOutputBytes)) {
        std::fprintf(stderr, "[FAIL] failed to write output under %s\n", output.c_str());
        return EXIT_FAILURE;
    }

    for (DeviceFile *file : {&br, &bi, &naturalRe, &naturalIm, &pilotCount, &pilotSc,
                             &packIndex, &weightModelFile, &ar, &ai, &ani, &wr, &wi}) Release(*file);
    CHECK_ACL(aclrtFree(hr));
    CHECK_ACL(aclrtFree(hi));
    CHECK_ACL(aclrtFree(hn));
    CHECK_ACL(aclrtFree(tRe));
    CHECK_ACL(aclrtFree(tIm));
    CHECK_ACL(aclrtFree(outRe));
    CHECK_ACL(aclrtFree(outIm));
    CHECK_ACL(aclrtFree(paddedRe));
    CHECK_ACL(aclrtFree(paddedIm));
    CHECK_ACL(aclrtFree(workspace));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(outReHost));
    CHECK_ACL(aclrtFreeHost(outImHost));
    CHECK_ACL(aclrtDestroyEvent(kernelStart));
    CHECK_ACL(aclrtDestroyEvent(kernelEnd));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}
