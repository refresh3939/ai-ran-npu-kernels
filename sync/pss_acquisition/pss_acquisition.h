



















#pragma once
#include <cstddef>
#include <cstdint>

namespace airan {


constexpr uint32_t N_FFT             = 2048;
constexpr uint32_t N_SYMBOL          = 14;
constexpr uint32_t N_SAMPLE_PER_SLOT = 30720;
constexpr uint32_t INPUT_GM_INT16_LEN = N_SAMPLE_PER_SLOT * 2;

constexpr uint32_t CP_LEN_FIRST      = 176;
constexpr uint32_t CP_LEN_OTHER      = 144;
constexpr uint32_t SYM_STRIDE        = N_FFT + CP_LEN_OTHER;


constexpr float    Q_SCALE_INV       = 1.0f / 256.0f;


constexpr uint32_t SAMPLE_RATE_HZ    = 61440000u;
constexpr float    SAMPLE_RATE_HZ_F  = 61440000.0f;
constexpr uint32_t N_PSS             = 3;


constexpr uint32_t N_FREQ_HYP        = 21;
constexpr float    FREQ_HYP_MIN_HZ   = -50000.0f;
constexpr float    FREQ_HYP_STEP_HZ  =   5000.0f;


constexpr uint32_t TIME_SEARCH_HALF  = 200;
constexpr uint32_t N_TIME_OFFSETS    = 2 * TIME_SEARCH_HALF + 1;
constexpr int32_t  EXPECTED_SSB_START = 176;





constexpr uint32_t HYP_BATCH         = 6;
constexpr uint32_t N_HYP_BATCHES     = 4;


constexpr uint32_t N_BLOCKS          = 4;
constexpr uint32_t TIME_PER_CORE_MAX = (N_TIME_OFFSETS + N_BLOCKS - 1) / N_BLOCKS;



constexpr uint32_t PSS_TMPL_INT16_LEN = N_PSS * N_FFT * 2;
constexpr uint32_t PSS_TMPL_GM_BYTES  = PSS_TMPL_INT16_LEN * sizeof(int16_t);






constexpr uint32_t OUT_PSS_PAD       = 4;
constexpr uint32_t OUT_HYP_PAD       = 8;
constexpr uint32_t OUT_FP32_PER_BLOCK = OUT_HYP_PAD * OUT_PSS_PAD;
constexpr uint32_t OUT_FP32_PER_BATCH = N_TIME_OFFSETS * OUT_FP32_PER_BLOCK;
constexpr uint32_t OUT_FP32_TOTAL    = N_HYP_BATCHES * OUT_FP32_PER_BATCH;
constexpr uint32_t OUT_GM_BYTES      = OUT_FP32_TOTAL * sizeof(float);


constexpr uint32_t SCR_FP32_TOTAL    = 8;
constexpr uint32_t SCR_GM_BYTES      = SCR_FP32_TOTAL * sizeof(float);


constexpr size_t TILING_TOTAL_SIZE = 128;
constexpr size_t WS_TOTAL          = 1 * 1024 * 1024;


constexpr uint32_t USE_XOR_AIV_MAP   = 0;

}
