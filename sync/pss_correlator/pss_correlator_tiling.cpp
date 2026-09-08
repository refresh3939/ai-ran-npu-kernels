








#include <cassert>
#include <cstdio>
#include <cstring>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace matmul_tiling;

extern "C" void GenerateTiling(const char *socVersion, uint8_t *buf)
{
    optiling::TCubeTiling td;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling api(*plat);


    api.SetDim(1);
    api.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    api.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    api.SetCType(TPosition::VECCALC, CubeFormat::ND, DataType::DT_FLOAT);
    api.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    api.SetOrgShape(16, 16, 256);
    api.SetShape(16, 16, 256);
    api.SetBias(false);
    api.SetTraverse(MatrixTraverse::FIRSTM);
    api.SetFixSplit(16, 16, -1);
    api.SetBufferSpace(-1, -1, -1);

    if (api.GetTiling(td) == -1) {
        printf("[pss_tiling] FAIL\n");
        std::abort();
    }
    td.set_stepM(1);
    td.set_stepN(1);
    td.SaveToBuffer(buf, sizeof(optiling::TCubeTiling));

    printf("[pss_tiling] Mmad mode (placeholder tiling for compat)\n");
}
