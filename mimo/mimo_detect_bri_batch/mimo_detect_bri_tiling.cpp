/**
 * @file mimo_detect_bri_tiling.cpp — 生成合法 Cube tiling (BRI 用 base Mmad, 但 Cube 单元需合法 tiling).
 *   Gram: 16×16×64 (k=M=64); BRI 迭代: 16×16×16. 取最大 k=64.
 */
#include <cstdio>
#include <cstring>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "mimo_detect_bri.h"
using namespace matmul_tiling;

extern "C" void GenerateTiling(const char* socVersion, uint8_t* buf)
{
    optiling::TCubeTiling td;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling api(*plat);
    api.SetDim(1);
    api.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);    // Aᵀ (Hᴴ 转置载)
    api.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    api.SetCType(TPosition::VECCALC, CubeFormat::ND, DataType::DT_FLOAT);
    api.SetShape(16, 16, 16);
    api.SetOrgShape(16, 16, 16);
    api.SetBias(false);
    api.SetTraverse(MatrixTraverse::FIRSTM);
    api.SetFixSplit(16, 16, -1);
    api.SetBufferSpace(-1, -1, -1);
    if (api.GetTiling(td) == -1) { printf("[tiling] FAIL\n"); std::abort(); }
    td.set_stepM(1); td.set_stepN(1);
    td.SaveToBuffer(buf, sizeof(optiling::TCubeTiling));
    printf("[tiling] Cube 16x16x16 tiling OK\n");
}
