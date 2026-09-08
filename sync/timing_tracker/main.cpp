





#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <string>
#include <vector>

#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_timing_tracker_kernel.h"
#else
#include "tikicpulib.h"



extern "C" void timing_tracker_kernel(uint8_t *, uint8_t *, uint8_t *,
                                       uint8_t *, uint8_t *, uint8_t *,
                                       uint8_t *, uint8_t *, uint8_t *,
                                       uint8_t *, uint8_t *, uint8_t *,
                                       uint8_t *);
#endif

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf);

namespace {
constexpr uint32_t N_FFT      = 4096;
constexpr uint32_t K_DMRS     = 220;
constexpr uint32_t N_SYM      = 2;

constexpr int N_WARMUP = 10;
constexpr int N_TIMED  = 50;

std::string DataDir()
{
    const char *env = std::getenv("AIRAN_DATA_DIR");
    return env ? std::string(env) : std::string(".");
}
std::string GWeight(const char *name) { return DataDir() + "/weights/" + name; }

bool DirExists(const std::string &path) __attribute__((unused));
bool DirExists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
void MakeDir(const std::string &path) {
    mkdir(path.c_str(), 0755);
}

std::vector<int> DiscoverCases() {
    std::vector<int> cases;
    std::string base = DataDir() + "/data/golden";
    DIR *d = opendir(base.c_str());
    if (!d) return cases;
    dirent *e;
    while ((e = readdir(d))) {
        std::string name = e->d_name;
        if (name.substr(0, 5) == "case_") {
            int id = atoi(name.c_str() + 5);
            cases.push_back(id);
        }
    }
    closedir(d);
    std::sort(cases.begin(), cases.end());
    return cases;
}


void ReadFileC(const std::string &path, size_t bytes, void *buf)
{
    size_t tmp = bytes;
    ReadFile(path, tmp, buf, bytes);
}

#ifndef ASCENDC_CPU_DEBUG
void HostToDev(uint8_t *devPtr, const void *hostBuf, size_t bytes)
{
    CHECK_ACL(aclrtMemcpy(devPtr, bytes, hostBuf, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}
#endif
}


int32_t main(int32_t  , char *  [])
{
    const char *socVersion = SOC_VERSION;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);



    constexpr uint32_t N_SYM_TOTAL = 14;
    constexpr uint32_t N_SC_PAD    = 1664;
    const size_t hPlaneBytes  = N_SYM_TOTAL * N_SC_PAD * sizeof(uint16_t);
    const size_t w64Bytes     = 64 * 64 * sizeof(int16_t);
    const size_t twBytes      = 64 * 64 * sizeof(int16_t);
    const size_t cirBytes     = N_FFT * sizeof(int16_t);
    const size_t dtBytes      = 1 * sizeof(float);
    const size_t dtScaleBytes = 1 * sizeof(float);
    const size_t scratchBytes = 6 * N_FFT * sizeof(int16_t);
    const size_t tilingBytes  = 2 * sizeof(TCubeTiling);
    const size_t wsBytes      = static_cast<size_t>(plat->GetLibApiWorkSpaceSize());

    uint8_t *tilingBuf = (uint8_t *)malloc(tilingBytes);
    GenerateTiling(socVersion, tilingBuf);

    uint32_t blockDim = 4;


    std::vector<int> case_ids = DiscoverCases();
    bool multi_case = !case_ids.empty();

    printf("[timing_tracker] SOC=%s blockDim=%u N_FFT=%u K_DMRS=%u\n",
           socVersion, blockDim, N_FFT, K_DMRS);
    if (multi_case) {
        printf("[timing_tracker] multi-case mode: %zu cases (", case_ids.size());
        for (size_t i = 0; i < case_ids.size(); ++i)
            printf("%d%s", case_ids[i], i + 1 < case_ids.size() ? "," : "");
        printf(")\n");
    } else {
        printf("[timing_tracker] single-case mode (no case_N subdirs)\n");
        case_ids.push_back(-1);
    }

#ifndef ASCENDC_CPU_DEBUG
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));


    uint8_t *hReH;     uint8_t *hReD;
    uint8_t *hImH;     uint8_t *hImD;
    uint8_t *wreH;     uint8_t *wreD;
    uint8_t *wimH;     uint8_t *wimD;
    uint8_t *twreH;    uint8_t *twreD;
    uint8_t *twimH;    uint8_t *twimD;
    uint8_t *cirReH;   uint8_t *cirReD;
    uint8_t *cirImH;   uint8_t *cirImD;
    uint8_t *dtH;      uint8_t *dtD;
    uint8_t *dtScaleH; uint8_t *dtScaleD;
    uint8_t *scrD, *wsD, *tilH, *tilD;

    CHECK_ACL(aclrtMallocHost((void **)&hReH, hPlaneBytes));
    CHECK_ACL(aclrtMalloc((void **)&hReD, hPlaneBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void **)&hImH, hPlaneBytes));
    CHECK_ACL(aclrtMalloc((void **)&hImD, hPlaneBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    auto load_weight = [&](uint8_t **hostP, uint8_t **devP, size_t bytes, const std::string &path) {
        CHECK_ACL(aclrtMallocHost((void **)hostP, bytes));
        CHECK_ACL(aclrtMalloc((void **)devP, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ReadFileC(path, bytes, *hostP);
        CHECK_ACL(aclrtMemcpy(*devP, bytes, *hostP, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    };
    load_weight(&wreH,  &wreD,  w64Bytes, GWeight("w_idft64_re.bin"));
    load_weight(&wimH,  &wimD,  w64Bytes, GWeight("w_idft64_im.bin"));
    load_weight(&twreH, &twreD, twBytes,  GWeight("twiddle_inv_re.bin"));
    load_weight(&twimH, &twimD, twBytes,  GWeight("twiddle_inv_im.bin"));

    CHECK_ACL(aclrtMallocHost((void **)&cirReH, cirBytes));
    CHECK_ACL(aclrtMallocHost((void **)&cirImH, cirBytes));
    CHECK_ACL(aclrtMalloc((void **)&cirReD, cirBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&cirImD, cirBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void **)&dtH, dtBytes));
    CHECK_ACL(aclrtMalloc((void **)&dtD, dtBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&dtScaleH, dtScaleBytes));
    CHECK_ACL(aclrtMalloc((void **)&dtScaleD, dtScaleBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    *reinterpret_cast<float *>(dtScaleH) = 1.0f;
    CHECK_ACL(aclrtMemcpy(dtScaleD, dtScaleBytes, dtScaleH, dtScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMalloc((void **)&scrD, scratchBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsD,  wsBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMallocHost((void **)&tilH, tilingBytes));
    CHECK_ACL(aclrtMalloc((void **)&tilD, tilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    memcpy(tilH, tilingBuf, tilingBytes);
    CHECK_ACL(aclrtMemcpy(tilD, tilingBytes, tilH, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_kernel = [&]() {
        ACLRT_LAUNCH_KERNEL(timing_tracker_kernel)
        (blockDim, stream,
         hReD, hImD, wreD, wimD, twreD, twimD, dtScaleD, scrD, cirReD, cirImD, dtD, wsD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    };


    for (size_t ci = 0; ci < case_ids.size(); ++ci) {
        int case_id = case_ids[ci];
        std::string hRePath, hImPath, outDir;
        std::string caseLabel;
        if (case_id < 0) {
            hRePath   = DataDir() + "/data/golden/h_re.bin";
            hImPath   = DataDir() + "/data/golden/h_im.bin";
            outDir    = DataDir() + "/data/ascend_output";
            caseLabel = "(root)";
        } else {
            char buf[32]; snprintf(buf, sizeof(buf), "case_%d", case_id);
            hRePath   = DataDir() + "/data/golden/" + buf + "/h_re.bin";
            hImPath   = DataDir() + "/data/golden/" + buf + "/h_im.bin";
            outDir    = DataDir() + "/data/ascend_output/" + buf;
            caseLabel = std::string("case_") + std::to_string(case_id);
        }
        MakeDir(DataDir() + "/data/ascend_output");
        MakeDir(outDir);

        ReadFileC(hRePath, hPlaneBytes, hReH);
        ReadFileC(hImPath, hPlaneBytes, hImH);
        HostToDev(hReD, hReH, hPlaneBytes);
        HostToDev(hImD, hImH, hPlaneBytes);

        bool is_timed = (case_id <= 0);
        if (is_timed) {

            for (int i = 0; i < N_WARMUP; ++i) {
                CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
                run_kernel();
            }
            std::vector<double> us_list; us_list.reserve(N_TIMED);
            for (int i = 0; i < N_TIMED; ++i) {
                CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
                auto t0 = std::chrono::high_resolution_clock::now();
                run_kernel();
                auto t1 = std::chrono::high_resolution_clock::now();
                us_list.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            double sum = 0, mn = us_list[0], mx = us_list[0];
            for (double v : us_list) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
            std::vector<double> srt = us_list; std::sort(srt.begin(), srt.end());
            printf("[timing_tracker] %s  latency: min %.1f us  p50 %.1f us  p99 %.1f us  max %.1f us  avg %.1f us\n",
                   caseLabel.c_str(), mn, srt[N_TIMED/2], srt[(N_TIMED*99)/100], mx, sum/N_TIMED);
        } else {
            CHECK_ACL(aclrtMemset(scrD, scratchBytes, 0, scratchBytes));
            run_kernel();
            printf("[timing_tracker] %s  (single run)\n", caseLabel.c_str());
        }


        CHECK_ACL(aclrtMemcpy(cirReH, cirBytes, cirReD, cirBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(cirImH, cirBytes, cirImD, cirBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(dtH,    dtBytes,  dtD,    dtBytes,  ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile((outDir + "/cir_re.bin").c_str(), cirReH, cirBytes);
        WriteFile((outDir + "/cir_im.bin").c_str(), cirImH, cirBytes);
        WriteFile((outDir + "/delta_T.bin").c_str(), dtH, dtBytes);


        float dt_val = *reinterpret_cast<float *>(dtH);
        printf("[timing_tracker] %s  delta_T = %+.4f samples\n", caseLabel.c_str(), dt_val);
    }


    CHECK_ACL(aclrtFree(hReD));    CHECK_ACL(aclrtFreeHost(hReH));
    CHECK_ACL(aclrtFree(hImD));    CHECK_ACL(aclrtFreeHost(hImH));
    CHECK_ACL(aclrtFree(wreD));    CHECK_ACL(aclrtFreeHost(wreH));
    CHECK_ACL(aclrtFree(wimD));    CHECK_ACL(aclrtFreeHost(wimH));
    CHECK_ACL(aclrtFree(twreD));   CHECK_ACL(aclrtFreeHost(twreH));
    CHECK_ACL(aclrtFree(twimD));   CHECK_ACL(aclrtFreeHost(twimH));
    CHECK_ACL(aclrtFree(cirReD));  CHECK_ACL(aclrtFreeHost(cirReH));
    CHECK_ACL(aclrtFree(cirImD));  CHECK_ACL(aclrtFreeHost(cirImH));
    CHECK_ACL(aclrtFree(dtD));     CHECK_ACL(aclrtFreeHost(dtH));
    CHECK_ACL(aclrtFree(dtScaleD));CHECK_ACL(aclrtFreeHost(dtScaleH));
    CHECK_ACL(aclrtFree(scrD));
    CHECK_ACL(aclrtFree(wsD));
    CHECK_ACL(aclrtFree(tilD));    CHECK_ACL(aclrtFreeHost(tilH));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#else

    uint8_t *hRe   = (uint8_t *)AscendC::GmAlloc(hPlaneBytes);
    uint8_t *hIm   = (uint8_t *)AscendC::GmAlloc(hPlaneBytes);
    uint8_t *wre   = (uint8_t *)AscendC::GmAlloc(w64Bytes);
    uint8_t *wim   = (uint8_t *)AscendC::GmAlloc(w64Bytes);
    uint8_t *twre  = (uint8_t *)AscendC::GmAlloc(twBytes);
    uint8_t *twim  = (uint8_t *)AscendC::GmAlloc(twBytes);
    uint8_t *scr   = (uint8_t *)AscendC::GmAlloc(scratchBytes);
    uint8_t *cirRe = (uint8_t *)AscendC::GmAlloc(cirBytes);
    uint8_t *cirIm = (uint8_t *)AscendC::GmAlloc(cirBytes);
    uint8_t *dt    = (uint8_t *)AscendC::GmAlloc(dtBytes);
    uint8_t *dtScale = (uint8_t *)AscendC::GmAlloc(dtScaleBytes);
    uint8_t *ws    = (uint8_t *)AscendC::GmAlloc(wsBytes);
    uint8_t *til   = (uint8_t *)AscendC::GmAlloc(tilingBytes);

    ReadFileC(DataDir() + "/data/golden/h_re.bin",    hPlaneBytes, hRe);
    ReadFileC(DataDir() + "/data/golden/h_im.bin",    hPlaneBytes, hIm);
    ReadFileC(GWeight("w_idft64_re.bin"),             w64Bytes,    wre);
    ReadFileC(GWeight("w_idft64_im.bin"),             w64Bytes,    wim);
    ReadFileC(GWeight("twiddle_inv_re.bin"),          twBytes,     twre);
    ReadFileC(GWeight("twiddle_inv_im.bin"),          twBytes,     twim);
    *reinterpret_cast<float *>(dtScale) = 1.0f;
    memcpy(til, tilingBuf, tilingBytes);

    ICPU_RUN_KF(timing_tracker_kernel, blockDim,
                hRe, hIm, wre, wim, twre, twim, dtScale, scr, cirRe, cirIm, dt, ws, til);

    MakeDir(DataDir() + "/data/ascend_output");
    WriteFile((DataDir() + "/data/ascend_output/cir_re.bin").c_str(), cirRe, cirBytes);
    WriteFile((DataDir() + "/data/ascend_output/cir_im.bin").c_str(), cirIm, cirBytes);
    WriteFile((DataDir() + "/data/ascend_output/delta_T.bin").c_str(), dt, dtBytes);

    AscendC::GmFree(hRe); AscendC::GmFree(hIm);
    AscendC::GmFree(wre); AscendC::GmFree(wim);
    AscendC::GmFree(twre); AscendC::GmFree(twim); AscendC::GmFree(scr);
    AscendC::GmFree(cirRe); AscendC::GmFree(cirIm); AscendC::GmFree(dt);
    AscendC::GmFree(dtScale);
    AscendC::GmFree(ws); AscendC::GmFree(til);
#endif
    free(tilingBuf);
    return 0;
}