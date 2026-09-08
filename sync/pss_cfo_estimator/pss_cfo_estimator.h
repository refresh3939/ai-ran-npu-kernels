

























#pragma once
#include <stddef.h>
#include <stdint.h>

namespace airan {


constexpr uint32_t N_PSS              = 256;
constexpr uint32_t N_HALF             = 128;
constexpr float    SAMPLE_RATE_HZ_F   = 7.68e6f;
constexpr uint32_t N_FFT_SSB          = 256;




constexpr float    Q_SCALE_INV        = 1.0f / 32768.0f;













constexpr float    FP16_MAX_NORMAL    = 65504.0f;
constexpr float    Q15_PRODUCT_MAX    = 2.0f;
constexpr float    Q15_REDUCE_MAX     = Q15_PRODUCT_MAX * (float)N_HALF;
static_assert(Q15_REDUCE_MAX < FP16_MAX_NORMAL * 0.5f,
              "fp16 reduce safety margin shrunk — check quantization or N_HALF change");






constexpr float    HZ_PER_RAD         = SAMPLE_RATE_HZ_F /
                                        (3.14159265358979323846f * (float)N_FFT_SSB);


constexpr uint32_t INPUT_GM_INT16_LEN = N_PSS  * 2;
constexpr uint32_t PILOT_GM_INT16_LEN = N_PSS  * 2;







constexpr uint32_t OUT_FLOAT_LEN      = 4;
constexpr uint32_t OUT_FLOAT_PADDED   = 8;
constexpr uint32_t OUT_GM_BYTES       = OUT_FLOAT_PADDED * sizeof(float);




constexpr size_t   TILING_TOTAL_SIZE  = 128;
constexpr size_t   WS_TOTAL           = 1 * 1024 * 1024;



constexpr uint32_t USE_XOR_AIV_MAP    = 0;
constexpr uint32_t BLOCK_DIM          = 1;


constexpr uint32_t SCR_GM_BYTES       = 1024;

}
