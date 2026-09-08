



#include <cassert>
#include <cstdio>
#include <iostream>
#include <cstring>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "ofdm_demod.h"

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
    api.SetOrgShape(ofdm_demod_batch::M_MMAD, ofdm_demod_batch::Q, ofdm_demod_batch::K_PHASE1);
    api.SetShape(ofdm_demod_batch::M_MMAD, ofdm_demod_batch::Q, ofdm_demod_batch::K_PHASE1);
    api.SetBias(false);
    api.SetTraverse(MatrixTraverse::FIRSTM);
    api.SetFixSplit(ofdm_demod_batch::M_MMAD, ofdm_demod_batch::Q, -1);
    api.SetBufferSpace(-1, -1, -1);
    if (api.GetTiling(td) == -1) { printf("[tiling] FAIL\n"); std::abort(); }
    td.set_stepM(1); td.set_stepN(1);
    td.SaveToBuffer(buf, sizeof(optiling::TCubeTiling));
    printf("[tiling] Mmad mode (placeholder tiling for compat)\n");
}
