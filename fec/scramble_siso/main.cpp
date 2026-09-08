










#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

#include "acl/acl.h"
#include "aclrtlaunch_scramble_kernel.h"
#include "scramble.h"

#define ACL_CHECK(expr)                                                        \
    do {                                                                       \
        aclError __e = (expr);                                                 \
        if (__e != ACL_SUCCESS) {                                              \
            fprintf(stderr, "[ACL] error %d at %s:%d\n", __e, __FILE__, __LINE__); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static std::string data_dir() {
    const char* e = std::getenv("AIRAN_DATA_DIR");
    std::string base = e ? std::string(e) : "./data";
    return base + "/golden";
}

template<typename T>
static std::vector<T> load_bin(const std::string& path, size_t n_elems) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[load] Cannot open: %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<T> buf(n_elems);
    size_t got = std::fread(buf.data(), sizeof(T), n_elems, f);
    std::fclose(f);
    if (got != n_elems) {
        fprintf(stderr, "[load] Short read %s: got %zu wanted %zu\n", path.c_str(), got, n_elems);
        std::exit(1);
    }
    return buf;
}

int main() {
    using namespace airan_scr;

    const std::string ddir = data_dir();

    constexpr uint32_t N_SLOT    = N_SLOT_MAX;
    constexpr uint32_t IN_ELEMS  = N_SLOT * N_STREAMS * N_SYM_PAD;
    constexpr uint32_t OUT_ELEMS = N_SLOT * N_STREAMS * N_DATA_SYM * QAM_SYM_STRIDE;
    constexpr size_t   IN_BYTES  = IN_ELEMS  * sizeof(int16_t);
    constexpr size_t   OUT_BYTES = OUT_ELEMS * sizeof(int16_t);


    constexpr uint32_t NSLOT_ELEMS = 16;
    constexpr size_t   NSLOT_BYTES = NSLOT_ELEMS * sizeof(int32_t);

    printf("=== scramble test ===\n");
    printf("input=[%u,%u,%u]  output=[%u,%u,%u,%u]\n", N_SLOT, N_STREAMS,
           N_SYM_PAD, N_SLOT, N_STREAMS, N_DATA_SYM, QAM_SYM_STRIDE);
    printf("IN_BYTES=%.2f MB  OUT_BYTES=%.2f MB\n", IN_BYTES/1e6, OUT_BYTES/1e6);
    printf("data dir: %s\n", ddir.c_str());


    auto h_input  = load_bin<int16_t>(ddir + "/input.bin",     IN_ELEMS);
    auto h_gold   = load_bin<int16_t>(ddir + "/gold_flat.bin", IN_ELEMS);
    auto h_golden = load_bin<int16_t>(ddir + "/golden.bin",    OUT_ELEMS);
    printf("Loaded: input=%zu  gold=%zu  golden=%zu  elements\n",
           h_input.size(), h_gold.size(), h_golden.size());


    std::vector<int32_t> h_nslot(NSLOT_ELEMS, 0);
    h_nslot[0] = (int32_t)N_SLOT;


    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));


    void *d_input, *d_gold, *d_nslot, *d_output;
    ACL_CHECK(aclrtMalloc(&d_input,  IN_BYTES,    ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_gold,   IN_BYTES,    ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_nslot,  NSLOT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_output, OUT_BYTES,   ACL_MEM_MALLOC_HUGE_FIRST));



    void *d_ws, *d_tiling;
    ACL_CHECK(aclrtMalloc(&d_ws,     WS_TOTAL,          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_tiling, TILING_TOTAL_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(d_ws,     WS_TOTAL,          0, WS_TOTAL));
    ACL_CHECK(aclrtMemset(d_tiling, TILING_TOTAL_SIZE, 0, TILING_TOTAL_SIZE));


    ACL_CHECK(aclrtMemset(d_output, OUT_BYTES, 0xAA, OUT_BYTES));


    ACL_CHECK(aclrtMemcpy(d_input,  IN_BYTES,    h_input.data(), IN_BYTES,  ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_gold,   IN_BYTES,    h_gold.data(),  IN_BYTES,  ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_nslot,  NSLOT_BYTES, h_nslot.data(), NSLOT_BYTES, ACL_MEMCPY_HOST_TO_DEVICE));


    ACLRT_LAUNCH_KERNEL(scramble_kernel)(BLOCK_DIM, stream, d_input, d_gold, d_nslot, d_output, d_ws, d_tiling);
    ACL_CHECK(aclrtSynchronizeStream(stream));


    constexpr int LAUNCHES = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < LAUNCHES; ++i) {
        ACLRT_LAUNCH_KERNEL(scramble_kernel)(BLOCK_DIM, stream, d_input, d_gold, d_nslot, d_output, d_ws, d_tiling);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_tb_ms = total_ms / LAUNCHES;
    double per_slot_us = per_tb_ms * 1000.0 / N_SLOT;

    printf("\n=== Performance ===\n");
    printf("  %d launches  total=%.2f ms\n", LAUNCHES, total_ms);
    printf("  Per TB:   %.3f ms\n", per_tb_ms);
    printf("  Per slot: %.2f us  (target <50 us)\n", per_slot_us);


    std::vector<int16_t> h_output(OUT_ELEMS);
    ACL_CHECK(aclrtMemcpy(h_output.data(), OUT_BYTES, d_output, OUT_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));


    {
        const char* e = std::getenv("AIRAN_DATA_DIR");
        std::string out_path = (e ? std::string(e) : std::string("./data"))
                             + "/ascend_output/output.bin";
        FILE* fo = std::fopen(out_path.c_str(), "wb");
        if (fo) {
            std::fwrite(h_output.data(), sizeof(int16_t), OUT_ELEMS, fo);
            std::fclose(fo);
            printf("Dumped NPU output -> %s\n", out_path.c_str());
        } else {
            fprintf(stderr, "[warn] cannot write %s\n", out_path.c_str());
        }
    }


    int sentinel_count = 0;
    for (auto v : h_output) if (v == (int16_t)0xAAAA) ++sentinel_count;
    if (sentinel_count > 0)
        printf("\nWARN: %d elements still sentinel 0xAA\n", sentinel_count);


    int max_err = 0, err_count = 0, first_err = -1;
    for (int i = 0; i < (int)OUT_ELEMS; ++i) {
        int e = std::abs((int)h_output[i] - (int)h_golden[i]);
        if (e > 0) {
            if (first_err < 0) first_err = i;
            ++err_count;
            max_err = std::max(max_err, e);
        }
    }

    printf("\n=== Accuracy ===\n");
    printf("  max_err=%d  err_count=%d / %u  (TOL=0)\n", max_err, err_count, OUT_ELEMS);
    if (first_err >= 0) {

        int slot = first_err / (N_STREAMS * N_SYM_PAD);
        int b    = (first_err / N_SYM_PAD) % N_STREAMS;
        int rem  = first_err % N_SYM_PAD;
        int sym  = rem / QAM_SYM_STRIDE;
        int sc   = rem % QAM_SYM_STRIDE;
        printf("  First error: flat=%d slot=%d qam_stream=%d sym=%d sc=%d got=%d golden=%d\n",
               first_err, slot, b, sym, sc, (int)h_output[first_err], (int)h_golden[first_err]);
    }

    if (max_err == 0)
        printf("\n[PASS] bit-exact match\n");
    else
        printf("\n[FAIL] errors detected\n");


    aclrtFree(d_input); aclrtFree(d_gold); aclrtFree(d_nslot); aclrtFree(d_output);
    aclrtFree(d_ws); aclrtFree(d_tiling);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return (max_err == 0) ? 0 : 1;
}
