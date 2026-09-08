






#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "mimo_dmrs_gen.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_mimo_dmrs_gen_kernel.h"
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

namespace {

constexpr int N_WARMUP = 10;
constexpr int N_TIMED  = 50;
constexpr int N_BENCH_BATCH = 100;

constexpr size_t N_PAD    = 896;
constexpr size_t PLANE    = 1792;
constexpr size_t NBITS    = 31;
constexpr size_t MAT_LEN  = NBITS * PLANE;
constexpr size_t N_SYM    = 2;
constexpr size_t NL       = airan::mimo_dmrs_gen::MAX_LAYERS;
constexpr size_t CINIT_PAD = 16;
constexpr size_t OUT_DBG_LEN = 8;
constexpr size_t TILING_BYTES = 128;
constexpr size_t SCR_BYTES    = 128;


constexpr size_t OUT_GRID  = NL * N_SYM * N_PAD;

constexpr float ERR_THRESH = 1e-3f;

struct TestCase {
    const char *dir_name;
    uint16_t slot;
    uint16_t n_id;
    uint8_t n_scid;
    uint16_t num_layers;
    uint16_t ports[4];
};
constexpr TestCase TEST_CASES[] = {
    {"case_0_baseline", 0, 1, 0, 1, {1000, 0, 0, 0}},
    {"case_1_nid0", 0, 0, 0, 2, {1000, 1002, 0, 0}},
    {"case_2_slot7", 7, 1, 0, 2, {1001, 1000, 0, 0}},
    {"case_3_nscid1", 0, 1, 1, 3, {1002, 1000, 1003, 0}},
    {"case_4_nid_large", 0, 300, 0, 4, {1003, 1002, 1001, 1000}},
};
constexpr int N_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);

std::string DataDir() {
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}
std::string CasePath(int i, const char *f) {
    return DataDir() + "/data/golden/" + TEST_CASES[i].dir_name + "/" + f;
}
std::string AscendReOut(int i) {
    return DataDir() + "/data/ascend_output/mimo_dmrs_" + TEST_CASES[i].dir_name + "_re.bin";
}
std::string AscendImOut(int i) {
    return DataDir() + "/data/ascend_output/mimo_dmrs_" + TEST_CASES[i].dir_name + "_im.bin";
}

inline float half2float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else { exp = 113; while ((man & 0x400) == 0) { man <<= 1; --exp; } man &= 0x3FF;
               f = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) { f = sign | 0x7F800000 | (man << 13); }
    else { f = sign | ((exp + 112) << 23) | (man << 13); }
    float out; std::memcpy(&out, &f, 4); return out;
}
float MaxAbsErr(const uint16_t *a, const uint16_t *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(half2float(a[i]) - half2float(b[i]));
        if (d > m) m = d;
    }
    return m;
}

float LayerErr(const uint16_t *a, const uint16_t *b, size_t layer) {
    return MaxAbsErr(a + layer*N_SYM*N_PAD, b + layer*N_SYM*N_PAD, N_SYM*N_PAD);
}

airan::mimo_dmrs_gen::PuschMimoConfig MakeConfig(const TestCase &test)
{
    using namespace airan::mimo_dmrs_gen;
    PuschMimoConfig config {};
    config.abi_version = ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = test.num_layers;
    config.num_tx_ports = test.num_layers == 1 ? 1 : (test.num_layers == 2 ? 2 : 4);
    config.num_rx_antennas = 64;
    config.qm = 8;
    config.num_symbols = N_SYMBOLS;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.slot_number = test.slot;
    config.num_allocated_symbols = N_SYMBOLS;
    config.used_subcarriers = N_SC_USED;
    config.padded_subcarriers = N_SC_PAD;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
    config.dmrs_scrambling_id = test.n_id;
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    config.n_scid = test.n_scid;
    for (uint32_t layer = 0; layer < test.num_layers; ++layer) config.dmrs_ports[layer] = test.ports[layer];
    return config;
}

}


int32_t main(int32_t, char *[])
{
    int exitCode = 0;
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    const size_t cinitBytes = CINIT_PAD * sizeof(int32_t);
    const size_t gmatBytes  = MAT_LEN   * sizeof(uint16_t);
    const size_t g1Bytes    = PLANE     * sizeof(uint16_t);
    const size_t gridBytes  = OUT_GRID  * sizeof(uint16_t);
    const size_t dbgBytes   = OUT_DBG_LEN * sizeof(float);
    const size_t wsBytes    = (size_t)plat->GetLibApiWorkSpaceSize();

    uint8_t *tilingBuf = (uint8_t *)malloc(TILING_BYTES);
    GenerateTiling(socVersion, tilingBuf);

    uint32_t blockDim = airan::mimo_dmrs_gen::BLOCK_DIM;
    printf("[mimo_dmrs] SOC=%s blockDim=%u NL=%zu\n", socVersion, blockDim, NL);
    printf("[mimo_dmrs] gmat=%zuB g1=%zuB grid=%zuB/plane ws=%zuB\n",
           gmatBytes, g1Bytes, gridBytes, wsBytes);

#ifdef ASCENDC_CPU_DEBUG
    fprintf(stderr, "[mimo_dmrs] CPU debug mode unsupported here\n");
    return 1;
#else
    const aclError initStatus = aclInit(nullptr);
    if (initStatus != ACL_ERROR_NONE) {
        ERROR_LOG("aclInit failed: %d (device may be busy)", static_cast<int>(initStatus));
        free(tilingBuf);
        return 2;
    }
    const aclError deviceStatus = aclrtSetDevice(0);
    if (deviceStatus != ACL_ERROR_NONE) {
        ERROR_LOG("aclrtSetDevice failed: %d", static_cast<int>(deviceStatus));
        CHECK_ACL(aclFinalize());
        free(tilingBuf);
        return 2;
    }
    aclrtStream stream = nullptr;
    const aclError streamStatus = aclrtCreateStream(&stream);
    if (streamStatus != ACL_ERROR_NONE) {
        ERROR_LOG("aclrtCreateStream failed: %d", static_cast<int>(streamStatus));
        CHECK_ACL(aclrtResetDevice(0));
        CHECK_ACL(aclFinalize());
        free(tilingBuf);
        return 2;
    }

    uint8_t *ciH=nullptr, *gmatH=nullptr, *g1H=nullptr, *reH=nullptr, *imH=nullptr, *dbgH=nullptr;
    uint8_t *goldReH=nullptr, *goldImH=nullptr;
    uint8_t *ciD=nullptr, *gmatD=nullptr, *g1D=nullptr, *reD=nullptr, *imD=nullptr, *dbgD=nullptr;
    uint8_t *scrD=nullptr, *wsD=nullptr, *tilH=nullptr, *tilD=nullptr;

    CHECK_ACL(aclrtMallocHost((void **)&ciH,   cinitBytes));
    CHECK_ACL(aclrtMallocHost((void **)&gmatH, gmatBytes));
    CHECK_ACL(aclrtMallocHost((void **)&g1H,   g1Bytes));
    CHECK_ACL(aclrtMallocHost((void **)&reH,   gridBytes));
    CHECK_ACL(aclrtMallocHost((void **)&imH,   gridBytes));
    CHECK_ACL(aclrtMallocHost((void **)&dbgH,  dbgBytes));
    CHECK_ACL(aclrtMallocHost((void **)&goldReH, gridBytes));
    CHECK_ACL(aclrtMallocHost((void **)&goldImH, gridBytes));
    CHECK_ACL(aclrtMalloc((void **)&ciD,   cinitBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&gmatD, gmatBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&g1D,   g1Bytes,    ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&reD,   gridBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&imD,   gridBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&dbgD,  dbgBytes,   ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&scrD,  SCR_BYTES,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsD,   wsBytes,    ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&tilH, TILING_BYTES));
    CHECK_ACL(aclrtMalloc((void **)&tilD, TILING_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    std::memcpy(tilH, tilingBuf, TILING_BYTES);
    CHECK_ACL(aclrtMemcpy(tilD, TILING_BYTES, tilH, TILING_BYTES, ACL_MEMCPY_HOST_TO_DEVICE));

    airan::mimo_dmrs_gen::BuildGoldBasis(reinterpret_cast<uint16_t *>(gmatH),
                                          reinterpret_cast<uint16_t *>(g1H));
    CHECK_ACL(aclrtMemcpy(gmatD, gmatBytes, gmatH, gmatBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(g1D,   g1Bytes,   g1H,   g1Bytes,   ACL_MEMCPY_HOST_TO_DEVICE));

    airan::mimo_dmrs_gen::MimoDmrsGenRuntimeV1 *runtime = nullptr;
    if (airan::mimo_dmrs_gen::CreateRuntime(&runtime) != airan::mimo_dmrs_gen::OK) {
        ERROR_LOG("cannot create public runtime adapter");
        return 1;
    }

    int n_pass = 0;
    for (int ci = 0; ci < N_CASES; ++ci) {
        size_t got = 0; bool ok_in = true;
        using namespace airan::mimo_dmrs_gen;
        const PuschMimoConfig config = MakeConfig(TEST_CASES[ci]);
        PuschMimoLayout layout {};
        KernelMetadata metadata {};
        int32_t expectedCinit[airan::mimo_dmrs_gen::CINIT_PAD] = {};
        if (BuildCurrentProfile(config, &layout, &metadata, expectedCinit) != OK) {
            ERROR_LOG("case %d: public config rejected", ci);
            continue;
        }
        std::memcpy(ciH, expectedCinit, cinitBytes);
        int32_t goldenCinit[airan::mimo_dmrs_gen::CINIT_PAD] = {};
        KernelMetadata goldenMetadata {};
        ok_in &= ReadFile(CasePath(ci, "cinit.bin").c_str(), got,
                          goldenCinit, sizeof(goldenCinit)) && got == sizeof(goldenCinit);
        ok_in &= ReadFile(CasePath(ci, "metadata.bin").c_str(), got,
                          &goldenMetadata, sizeof(goldenMetadata)) && got == sizeof(goldenMetadata);
        ok_in &= ReadFile(CasePath(ci, "x_re.bin").c_str(), got, goldReH, gridBytes) && got == gridBytes;
        ok_in &= ReadFile(CasePath(ci, "x_im.bin").c_str(), got, goldImH, gridBytes) && got == gridBytes;
        if (!ok_in) { ERROR_LOG("case %d: missing/wrong-size bins", ci); continue; }
        if (std::memcmp(expectedCinit, goldenCinit, sizeof(goldenCinit)) != 0 ||
            std::memcmp(&metadata, &goldenMetadata, sizeof(metadata)) != 0) {
            ERROR_LOG("case %d: derived c_init/metadata disagrees with independent reference", ci);
            continue;
        }

        const size_t logicalElems = LogicalOutputElems(config.num_layers, layout.num_dmrs_symbols);
        const size_t logicalBytes = logicalElems * sizeof(uint16_t);
        void *logicalReD = nullptr;
        void *logicalImD = nullptr;
        CHECK_ACL(aclrtMalloc(&logicalReD, logicalBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(&logicalImD, logicalBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        MimoDmrsGenOpArgsV1 publicArgs {};
        publicArgs.abi_version = ABI_VERSION;
        publicArgs.struct_size = sizeof(publicArgs);
        publicArgs.dmrs_re = logicalReD;
        publicArgs.dmrs_im = logicalImD;
        publicArgs.config = &config;
        publicArgs.layout = &layout;
        publicArgs.stream = stream;
        if (ValidateOpArgs(publicArgs) != OK) {
            ERROR_LOG("case %d: public OpArgs validation failed", ci);
            CHECK_ACL(aclrtFree(logicalReD));
            CHECK_ACL(aclrtFree(logicalImD));
            continue;
        }

        const Status enqueueStatus = Enqueue(runtime, publicArgs);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        std::vector<uint16_t> logicalRe(logicalElems);
        std::vector<uint16_t> logicalIm(logicalElems);
        CHECK_ACL(aclrtMemcpy(logicalRe.data(), logicalBytes, logicalReD, logicalBytes,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(logicalIm.data(), logicalBytes, logicalImD, logicalBytes,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtFree(logicalReD));
        CHECK_ACL(aclrtFree(logicalImD));
        const float abiReErr = MaxAbsErr(logicalRe.data(),
                                         reinterpret_cast<uint16_t *>(goldReH), logicalElems);
        const float abiImErr = MaxAbsErr(logicalIm.data(),
                                         reinterpret_cast<uint16_t *>(goldImH), logicalElems);
        const bool abiOk = enqueueStatus == OK && abiReErr < ERR_THRESH && abiImErr < ERR_THRESH;

        CHECK_ACL(aclrtMemcpy(ciD, cinitBytes, ciH, cinitBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        std::memcpy(tilH, &metadata, sizeof(metadata));
        CHECK_ACL(aclrtMemcpy(tilD, ::TILING_BYTES, tilH, ::TILING_BYTES, ACL_MEMCPY_HOST_TO_DEVICE));
        std::memset(reH, 0xff, gridBytes); std::memset(imH, 0xff, gridBytes); std::memset(dbgH, 0xff, dbgBytes);
        CHECK_ACL(aclrtMemcpy(reD,  gridBytes, reH,  gridBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(imD,  gridBytes, imH,  gridBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(dbgD, dbgBytes,  dbgH, dbgBytes,  ACL_MEMCPY_HOST_TO_DEVICE));

        ACLRT_LAUNCH_KERNEL(mimo_dmrs_gen_kernel)
            (blockDim, stream, ciD, gmatD, g1D, scrD, reD, imD, dbgD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));

        CHECK_ACL(aclrtMemcpy(reH,  gridBytes, reD,  gridBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(imH,  gridBytes, imD,  gridBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(dbgH, dbgBytes,  dbgD, dbgBytes,  ACL_MEMCPY_DEVICE_TO_HOST));

        WriteFile(AscendReOut(ci).c_str(), reH, gridBytes);
        WriteFile(AscendImOut(ci).c_str(), imH, gridBytes);

        float re_err = MaxAbsErr((uint16_t *)reH, (uint16_t *)goldReH, OUT_GRID);
        float im_err = MaxAbsErr((uint16_t *)imH, (uint16_t *)goldImH, OUT_GRID);
        float sentinel = ((float *)dbgH)[7];
        bool sentinel_ok = (sentinel == 7.0f);
        bool err_ok = (re_err < ERR_THRESH) && (im_err < ERR_THRESH);
        bool ok = sentinel_ok && err_ok && abiOk;
        printf("  case %d %-18s L=%u basis_rows=%.0f re_err=%.5f im_err=%.5f abi_err=(%.5f,%.5f) kernel_L=%.0f  %s\n",
               ci, TEST_CASES[ci].dir_name, TEST_CASES[ci].num_layers,
               ((float *)dbgH)[3], re_err, im_err, abiReErr, abiImErr, ((float *)dbgH)[2],
               ok ? "[PASS]" : (sentinel_ok ? "[FAIL]" : "[NO-SENTINEL]"));

        if (!ok) {
            for (size_t l = 0; l < NL; ++l) {
                float le = LayerErr((uint16_t*)reH, (uint16_t*)goldReH, l);
                float li = LayerErr((uint16_t*)imH, (uint16_t*)goldImH, l);
                printf("      layer%zu re_err=%.5f im_err=%.5f\n", l, le, li);
            }
        }
        if (ok) ++n_pass;
    }
    printf("[mimo_dmrs] %d / %d PASS\n", n_pass, N_CASES);

    bool validationOk = true;
    {
        using namespace airan::mimo_dmrs_gen;
        const PuschMimoConfig valid = MakeConfig(TEST_CASES[4]);
        PuschMimoLayout layout {};
        KernelMetadata metadata {};
        int32_t cinit[airan::mimo_dmrs_gen::CINIT_PAD] = {};
        validationOk &= BuildCurrentProfile(valid, &layout, &metadata, cinit) == OK;
        for (uint32_t index = 0; index < MAX_LAYERS; ++index) {
            PortOccSemantics semantics {};
            validationOk &= GetCurrentPortOccSemantics(
                                static_cast<uint16_t>(1000u + index), &semantics) == OK &&
                            semantics.comb_delta == index / 2u &&
                            semantics.wf_odd_negative == (index & 1u) &&
                            semantics.wt_negative[0] == 0 && semantics.wt_negative[1] == 0;
        }
        PuschMimoConfig duplicate = valid;
        duplicate.dmrs_ports[1] = duplicate.dmrs_ports[0];
        validationOk &= BuildCurrentProfile(duplicate, &layout, &metadata, cinit) ==
                        UNSUPPORTED_PROFILE;
        PuschMimoConfig badPort = valid;
        badPort.dmrs_ports[0] = 1004;
        validationOk &= BuildCurrentProfile(badPort, &layout, &metadata, cinit) ==
                        UNSUPPORTED_PROFILE;
        PuschMimoConfig badRank = valid;
        badRank.num_layers = 0;
        validationOk &= BuildCurrentProfile(badRank, &layout, &metadata, cinit) ==
                        UNSUPPORTED_PROFILE;

        PuschMimoLayout correctLayout {};
        validationOk &= BuildCurrentProfile(valid, &correctLayout, &metadata, cinit) == OK;
        PuschMimoLayout badLayout = correctLayout;
        --badLayout.data_stride;
        MimoDmrsGenOpArgsV1 badArgs {ABI_VERSION,
                                     static_cast<uint16_t>(sizeof(MimoDmrsGenOpArgsV1)),
                                     reD, imD, &valid, &badLayout, stream};
        validationOk &= ValidateOpArgs(badArgs) == INVALID_ARGUMENT;
    }
    printf("[mimo_dmrs] ABI/port/OCC negative validation %s\n",
           validationOk ? "[PASS]" : "[FAIL]");

    printf("[mimo_dmrs] kernel latency by profile (device event, batch=%d):\n", N_BENCH_BATCH);
    for (int ci = 0; ci < N_CASES; ++ci) {
        using namespace airan::mimo_dmrs_gen;
        const PuschMimoConfig config = MakeConfig(TEST_CASES[ci]);
        PuschMimoLayout layout {};
        KernelMetadata metadata {};
        int32_t cinit[airan::mimo_dmrs_gen::CINIT_PAD] = {};
        if (BuildCurrentProfile(config, &layout, &metadata, cinit) != OK) return 1;
        std::memcpy(ciH, cinit, cinitBytes);
        std::memcpy(tilH, &metadata, sizeof(metadata));
        CHECK_ACL(aclrtMemcpy(ciD, cinitBytes, ciH, cinitBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(tilD, ::TILING_BYTES, tilH, ::TILING_BYTES, ACL_MEMCPY_HOST_TO_DEVICE));

        for (int i = 0; i < N_WARMUP; ++i) {
            ACLRT_LAUNCH_KERNEL(mimo_dmrs_gen_kernel)
                (blockDim, stream, ciD, gmatD, g1D, scrD, reD, imD, dbgD, wsD, tilD);
            CHECK_ACL(aclrtSynchronizeStream(stream));
        }
        std::vector<double> us;
        us.reserve(N_TIMED);
        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        CHECK_ACL(aclrtCreateEvent(&start));
        CHECK_ACL(aclrtCreateEvent(&end));
        for (int i = 0; i < N_TIMED; ++i) {
            CHECK_ACL(aclrtRecordEvent(start, stream));
            for (int batch = 0; batch < N_BENCH_BATCH; ++batch) {
                ACLRT_LAUNCH_KERNEL(mimo_dmrs_gen_kernel)
                    (blockDim, stream, ciD, gmatD, g1D, scrD, reD, imD, dbgD, wsD, tilD);
            }
            CHECK_ACL(aclrtRecordEvent(end, stream));
            CHECK_ACL(aclrtSynchronizeEvent(end));
            float elapsedMs = 0.0f;
            CHECK_ACL(aclrtEventElapsedTime(&elapsedMs, start, end));
            us.push_back(static_cast<double>(elapsedMs) * 1000.0 / N_BENCH_BATCH);
        }
        CHECK_ACL(aclrtDestroyEvent(start));
        CHECK_ACL(aclrtDestroyEvent(end));
        std::sort(us.begin(), us.end());
        double sum = 0;
        for (double value : us) sum += value;
        printf("  %-18s L=%u min=%.1f avg=%.1f p50=%.1f p99=%.1f max=%.1f us\n",
               TEST_CASES[ci].dir_name, TEST_CASES[ci].num_layers, us.front(),
               sum / N_TIMED, us[N_TIMED / 2], us[(N_TIMED * 99) / 100], us.back());
    }

    airan::mimo_dmrs_gen::DestroyRuntime(runtime);
    CHECK_ACL(aclrtFreeHost(ciH));   CHECK_ACL(aclrtFree(ciD));
    CHECK_ACL(aclrtFreeHost(gmatH)); CHECK_ACL(aclrtFree(gmatD));
    CHECK_ACL(aclrtFreeHost(g1H));   CHECK_ACL(aclrtFree(g1D));
    CHECK_ACL(aclrtFreeHost(reH));   CHECK_ACL(aclrtFree(reD));
    CHECK_ACL(aclrtFreeHost(imH));   CHECK_ACL(aclrtFree(imD));
    CHECK_ACL(aclrtFreeHost(dbgH));  CHECK_ACL(aclrtFree(dbgD));
    CHECK_ACL(aclrtFreeHost(goldReH)); CHECK_ACL(aclrtFreeHost(goldImH));
    CHECK_ACL(aclrtFree(scrD));      CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtFreeHost(tilH));  CHECK_ACL(aclrtFree(tilD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    exitCode = n_pass == N_CASES && validationOk ? 0 : 1;
#endif
    free(tilingBuf);
    return exitCode;
}
