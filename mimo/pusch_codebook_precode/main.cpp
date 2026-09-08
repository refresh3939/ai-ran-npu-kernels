
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_pusch_codebook_precode_kernel.h"
#include "pusch_codebook_precode.h"
#include "pusch_codebook_precode_runtime.h"

namespace pc = airan::pusch_precode;

#define ACL_CHECK(expr) do { \
    const aclError error__ = (expr); \
    if (error__ != ACL_SUCCESS) { \
        std::fprintf(stderr, "[ACL] %s failed: %d at %s:%d\n", #expr, error__, __FILE__, __LINE__); \
        std::exit(2); \
    } \
} while (0)

namespace {

struct Case {
    const char* name;
    uint16_t ports;
    uint16_t layers;
    uint16_t tpmi;
    uint16_t prg;
    bool codebook;
};

constexpr Case CASES[] = {
    {"p1_l1_identity", 1, 1, 0, 133, true},
    {"p2_l1_tpmi4", 2, 1, 4, 1, true},
    {"p2_l2_tpmi2", 2, 2, 2, 8, true},
    {"p4_l1_tpmi17", 4, 1, 17, 16, true},
    {"p4_l2_tpmi21", 4, 2, 21, 5, true},
    {"p4_l3_tpmi6", 4, 3, 6, 4, true},
    {"p4_l4_tpmi4", 4, 4, 4, 133, true},
    {"p4_l4_bypass", 4, 4, 0, 7, false},
};

constexpr float ERROR_LIMIT = 6.0e-3f;
constexpr int WARMUP = 5;
constexpr int TIMED = 20;

float InputRe(uint32_t layer, uint32_t symbol, uint32_t subcarrier) {
    return 0.31f * std::sin(0.013f * static_cast<float>(1 + subcarrier) +
                            0.17f * static_cast<float>(symbol) + 0.29f * static_cast<float>(layer));
}

float InputIm(uint32_t layer, uint32_t symbol, uint32_t subcarrier) {
    return 0.27f * std::cos(0.009f * static_cast<float>(3 + subcarrier) +
                            0.11f * static_cast<float>(symbol) - 0.23f * static_cast<float>(layer));
}

void* DeviceAlloc(size_t bytes) {
    void* ptr = nullptr;
    ACL_CHECK(aclrtMalloc(&ptr, std::max<size_t>(bytes, 1), ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

float MaxError(const aclFloat16* actual, const std::vector<float>& golden, size_t count) {
    float result = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        result = std::max(result, std::fabs(aclFloat16ToFloat(actual[i]) - golden[i]));
    }
    return result;
}

bool PaddingIsZero(const aclFloat16* values, uint32_t ports) {
    for (uint32_t p = 0; p < ports; ++p) {
        for (uint32_t s = 0; s < pc::N_SYMBOLS; ++s) {
            const size_t row = (static_cast<size_t>(p) * pc::N_SYMBOLS + s) * pc::N_SC_PAD;
            for (uint32_t k = pc::N_SC_USED; k < pc::N_SC_PAD; ++k) {
                if (aclFloat16ToFloat(values[row + k]) != 0.0f) return false;
            }
        }
    }
    return true;
}

}

int main() {
    const size_t max_layer_count = static_cast<size_t>(pc::MAX_LAYERS) * pc::N_RE_GRID;
    const size_t max_port_count = static_cast<size_t>(pc::MAX_PORTS) * pc::N_RE_GRID;
    const size_t max_grid_bytes = max_layer_count * sizeof(aclFloat16);
    const size_t max_output_bytes = max_port_count * sizeof(aclFloat16);
    const size_t weight_bytes = pc::MAX_WEIGHT_ELEMS * sizeof(aclFloat16);
    const size_t prg_bytes = pc::PRG_MAP_PAD * sizeof(uint16_t);

    std::vector<aclFloat16> layer_re(max_layer_count), layer_im(max_layer_count);
    std::vector<float> layer_re_f(max_layer_count), layer_im_f(max_layer_count);
    for (uint32_t l = 0; l < pc::MAX_LAYERS; ++l) {
        for (uint32_t s = 0; s < pc::N_SYMBOLS; ++s) {
            for (uint32_t k = 0; k < pc::N_SC_PAD; ++k) {
                const size_t i = (static_cast<size_t>(l) * pc::N_SYMBOLS + s) * pc::N_SC_PAD + k;

                const float r = k < pc::N_SC_USED ? InputRe(l, s, k) : 3.0f;
                const float q = k < pc::N_SC_USED ? InputIm(l, s, k) : -2.0f;
                layer_re[i] = aclFloatToFloat16(r);
                layer_im[i] = aclFloatToFloat16(q);
                layer_re_f[i] = aclFloat16ToFloat(layer_re[i]);
                layer_im_f[i] = aclFloat16ToFloat(layer_im[i]);
            }
        }
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));



    const size_t workspace_bytes = 1;

    void* layer_re_d = DeviceAlloc(max_grid_bytes);
    void* layer_im_d = DeviceAlloc(max_grid_bytes);
    void* weight_re_d = DeviceAlloc(weight_bytes);
    void* weight_im_d = DeviceAlloc(weight_bytes);
    void* prg_d = DeviceAlloc(prg_bytes);
    void* port_re_d = DeviceAlloc(max_output_bytes);
    void* port_im_d = DeviceAlloc(max_output_bytes);
    void* workspace_d = DeviceAlloc(workspace_bytes);
    void* tiling_d = DeviceAlloc(pc::TILING_BYTES);

    ACL_CHECK(aclrtMemcpy(layer_re_d, max_grid_bytes, layer_re.data(), max_grid_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(layer_im_d, max_grid_bytes, layer_im.data(), max_grid_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    int passed = 0;
    for (const Case& test : CASES) {
        airan::PuschMimoConfig config = pc::MakeDefaultConfig(test.layers, test.ports);
        config.tpmi = test.tpmi;
        config.prg_size_rb = test.prg;
        config.codebook_enabled = test.codebook ? 1 : 0;

        pc::CodebookPlan plan;
        std::string why;
        if (pc::BuildCodebookPlan(config, &plan, &why) != pc::Status::kSuccess) {
            std::fprintf(stderr, "[%s] config failed: %s\n", test.name, why.c_str());
            continue;
        }

        std::vector<aclFloat16> weight_re(pc::MAX_WEIGHT_ELEMS);
        std::vector<aclFloat16> weight_im(pc::MAX_WEIGHT_ELEMS);
        for (uint32_t i = 0; i < plan.weight_count_padded; ++i) {
            weight_re[i] = aclFloatToFloat16(plan.weight_re[i]);
            weight_im[i] = aclFloatToFloat16(plan.weight_im[i]);
        }
        ACL_CHECK(aclrtMemcpy(weight_re_d, weight_bytes, weight_re.data(), weight_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(weight_im_d, weight_bytes, weight_im.data(), weight_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(prg_d, prg_bytes, plan.prg_of_rb.data(), prg_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(tiling_d, pc::TILING_BYTES, plan.tiling.data(), pc::TILING_BYTES,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemset(port_re_d, max_output_bytes, 0xff, max_output_bytes));
        ACL_CHECK(aclrtMemset(port_im_d, max_output_bytes, 0xff, max_output_bytes));

        ACLRT_LAUNCH_KERNEL(pusch_codebook_precode_kernel)(
            pc::BLOCK_DIM, stream, layer_re_d, layer_im_d, weight_re_d, weight_im_d,
            prg_d, port_re_d, port_im_d, workspace_d, tiling_d);
        ACL_CHECK(aclrtSynchronizeStream(stream));

        const size_t output_count = static_cast<size_t>(test.ports) * pc::N_RE_GRID;
        const size_t output_bytes = output_count * sizeof(aclFloat16);
        std::vector<aclFloat16> actual_re(output_count), actual_im(output_count);
        std::vector<float> golden_re(output_count), golden_im(output_count);
        ACL_CHECK(aclrtMemcpy(actual_re.data(), output_bytes, port_re_d, output_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(actual_im.data(), output_bytes, port_im_d, output_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        if (pc::Reference(layer_re_f.data(), layer_im_f.data(), config,
                          golden_re.data(), golden_im.data(), &why) != pc::Status::kSuccess) {
            std::fprintf(stderr, "[%s] reference failed: %s\n", test.name, why.c_str());
            continue;
        }
        const float re_error = MaxError(actual_re.data(), golden_re, output_count);
        const float im_error = MaxError(actual_im.data(), golden_im, output_count);
        const bool padding_ok = PaddingIsZero(actual_re.data(), test.ports) &&
                                PaddingIsZero(actual_im.data(), test.ports);
        const bool ok = re_error <= ERROR_LIMIT && im_error <= ERROR_LIMIT && padding_ok;
        std::printf("%-18s P=%u L=%u TPMI=%u PRG=%u re=%.6f im=%.6f pad=%s %s\n",
                    test.name, test.ports, test.layers, test.tpmi, test.prg,
                    re_error, im_error, padding_ok ? "zero" : "BAD", ok ? "[PASS]" : "[FAIL]");
        if (!ok) {
            std::printf("  sample actual=(%.6f,%.6f) golden=(%.6f,%.6f)\n",
                        aclFloat16ToFloat(actual_re[0]), aclFloat16ToFloat(actual_im[0]),
                        golden_re[0], golden_im[0]);
            if (test.layers == 1 && test.ports == 1) {
                for (size_t i : {size_t{0}, size_t{1}, size_t{7}, size_t{11}, size_t{12},
                                 size_t{15}, size_t{16}, size_t{127}, size_t{1595}, size_t{1664}}) {
                    std::printf("    i=%zu a=(%.4f,%.4f) g=(%.4f,%.4f)\n", i,
                                aclFloat16ToFloat(actual_re[i]), aclFloat16ToFloat(actual_im[i]),
                                golden_re[i], golden_im[i]);
                }
            }
        }
        if (ok) ++passed;
    }


    airan::PuschMimoConfig bench = pc::MakeDefaultConfig(4, 4);
    bench.tpmi = 4;
    std::string why;
    pc::RuntimeContext runtime;
    if (pc::RuntimeInit(&runtime) != pc::Status::kSuccess) {
        std::fprintf(stderr, "runtime facade initialization failed\n");
        return 2;
    }
    std::array<airan::PuschMimoConfig, 5> mixed_configs{{
        pc::MakeDefaultConfig(4, 4), pc::MakeDefaultConfig(3, 4),
        pc::MakeDefaultConfig(2, 2), pc::MakeDefaultConfig(1, 1),
        pc::MakeDefaultConfig(4, 4)}};
    mixed_configs[0].tpmi = 4;
    mixed_configs[1].tpmi = 6;
    mixed_configs[2].tpmi = 2;
    mixed_configs[4].codebook_enabled = 0;
    mixed_configs[4].tpmi = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (const auto& config : mixed_configs) {
            if (pc::Launch(&runtime, layer_re_d, layer_im_d, config,
                           port_re_d, port_im_d, workspace_d, stream, &why) !=
                pc::Status::kSuccess) {
                std::fprintf(stderr, "mixed-config runtime launch failed: %s\n", why.c_str());
                return 2;
            }
        }
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (runtime.cache_entries != mixed_configs.size() || runtime.cache_misses != mixed_configs.size() ||
        runtime.cache_hits < mixed_configs.size()) {
        std::fprintf(stderr, "runtime cache failed: entries=%u misses=%u hits=%u\n",
                     runtime.cache_entries, runtime.cache_misses, runtime.cache_hits);
        return 2;
    }
    std::printf("runtime config cache: entries=%u misses=%u hits=%u [PASS]\n",
                runtime.cache_entries, runtime.cache_misses, runtime.cache_hits);
    for (int i = 0; i < WARMUP; ++i) {
        if (pc::Launch(&runtime, layer_re_d, layer_im_d, bench, port_re_d, port_im_d,
                       workspace_d, stream, &why) != pc::Status::kSuccess) {
            std::fprintf(stderr, "runtime facade launch failed: %s\n", why.c_str());
            return 2;
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }
    std::vector<double> times;
    for (int i = 0; i < TIMED; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (pc::Launch(&runtime, layer_re_d, layer_im_d, bench, port_re_d, port_im_d,
                       workspace_d, stream, &why) != pc::Status::kSuccess) {
            std::fprintf(stderr, "runtime facade launch failed: %s\n", why.c_str());
            return 2;
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        const auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    std::sort(times.begin(), times.end());
    double sum = 0.0;
    for (double t : times) sum += t;
    std::printf("P4/L4/NR=64 latency: "
                "min=%.1fus avg=%.1fus p50=%.1fus p99=%.1fus\n",
                times.front(), sum / times.size(), times[times.size() / 2], times.back());
    std::printf("correctness: %d/%zu PASS\n", passed, sizeof(CASES) / sizeof(CASES[0]));
    pc::RuntimeDestroy(&runtime);

    ACL_CHECK(aclrtFree(layer_re_d));
    ACL_CHECK(aclrtFree(layer_im_d));
    ACL_CHECK(aclrtFree(weight_re_d));
    ACL_CHECK(aclrtFree(weight_im_d));
    ACL_CHECK(aclrtFree(prg_d));
    ACL_CHECK(aclrtFree(port_re_d));
    ACL_CHECK(aclrtFree(port_im_d));
    ACL_CHECK(aclrtFree(workspace_d));
    ACL_CHECK(aclrtFree(tiling_d));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return passed == static_cast<int>(sizeof(CASES) / sizeof(CASES[0])) ? 0 : 1;
}
