





#pragma once
#include <cstddef>
#include <cstdint>

namespace airan {


constexpr uint32_t N_FFT             = 2048;
constexpr uint32_t N_SYMBOL          = 14;
constexpr uint32_t SYMBOLS_PER_CORE  = 4;
constexpr uint32_t N_BLOCKS          = 4;
constexpr uint32_t N_SAMPLE_PER_SLOT = 30720;
constexpr uint32_t INPUT_GM_INT16_LEN = N_SAMPLE_PER_SLOT * 2;

constexpr uint32_t CP_LEN_FIRST      = 176;
constexpr uint32_t CP_LEN_OTHER      = 144;
constexpr uint32_t SYM_STRIDE        = N_FFT + CP_LEN_OTHER;


constexpr float    Q_SCALE_INV       = 1.0f / 256.0f;




constexpr uint32_t OUT_FP32_COUNT    = 8;
constexpr uint32_t OUT_GM_BYTES      = OUT_FP32_COUNT * sizeof(float);




constexpr uint32_t SCR_FP32_PER_CORE = 8;
constexpr uint32_t SCR_FP32_TOTAL    = N_BLOCKS * SCR_FP32_PER_CORE;
constexpr uint32_t SCR_GM_BYTES      = SCR_FP32_TOTAL * sizeof(float);


constexpr size_t TILING_TOTAL_SIZE = 128;
constexpr size_t WS_TOTAL          = 1 * 1024 * 1024;





constexpr uint32_t USE_XOR_AIV_MAP   = 0;

}
