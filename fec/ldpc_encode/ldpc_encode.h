








#pragma once
#include <cstddef>
#include <cstdint>

namespace airan {




constexpr uint32_t LDPC_C_NUM     = 143;




constexpr uint32_t LDPC_Z         = 384;
constexpr uint32_t LDPC_KB        = 22;
constexpr uint32_t LDPC_MB        = 46;
constexpr uint32_t LDPC_G         = 4;
constexpr uint32_t LDPC_MBMG      = LDPC_MB - LDPC_G;

constexpr uint32_t LDPC_K         = LDPC_KB * LDPC_Z;
constexpr uint32_t LDPC_FZ        = LDPC_G  * LDPC_Z;
constexpr uint32_t LDPC_MB4Z      = LDPC_MBMG * LDPC_Z;
constexpr uint32_t LDPC_INFO_OUT  = LDPC_K - 2 * LDPC_Z;
constexpr uint32_t LDPC_N_RAW     = LDPC_INFO_OUT + LDPC_FZ + LDPC_MB4Z;




constexpr uint32_t PAD            = LDPC_Z;
constexpr uint32_t Z_PAD          = LDPC_Z + PAD;


constexpr uint32_t SHIFT_A_ELEMS   = LDPC_G    * LDPC_KB;
constexpr uint32_t SHIFT_BI_MAXW   = 4;
constexpr uint32_t SHIFT_BI_ELEMS  = LDPC_G    * LDPC_G * SHIFT_BI_MAXW;
constexpr uint32_t SHIFT_C_ELEMS   = LDPC_MBMG * LDPC_KB;
constexpr uint32_t SHIFT_D_ELEMS   = LDPC_MBMG * LDPC_G;

constexpr size_t SHIFT_A_BYTES  = SHIFT_A_ELEMS  * sizeof(int16_t);
constexpr size_t SHIFT_BI_BYTES = SHIFT_BI_ELEMS * sizeof(int16_t);
constexpr size_t SHIFT_C_BYTES  = SHIFT_C_ELEMS  * sizeof(int16_t);
constexpr size_t SHIFT_D_BYTES  = SHIFT_D_ELEMS  * sizeof(int16_t);


constexpr size_t INFO_BYTES   = LDPC_C_NUM * LDPC_K;
constexpr size_t OUTPUT_BYTES = LDPC_C_NUM * LDPC_N_RAW;


constexpr size_t TILING_TOTAL_SIZE = 128;
constexpr size_t WS_TOTAL          = 1 * 1024 * 1024;
constexpr size_t DBG_WORDS_PER_CB  = 64;

}
