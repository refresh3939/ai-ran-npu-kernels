









#pragma once
#include <cstdint>
#include <cstddef>

namespace airan_pss {


constexpr uint32_t FS_HZ          = 7680000u;
constexpr uint32_t SCS_HZ         = 30000u;
constexpr uint32_t N_FFT          = 256u;
constexpr uint32_t N_SC_SSB       = 240u;
constexpr uint32_t N_SC_PSS       = 127u;


constexpr uint32_t N_SEARCH       = 153600u;
constexpr uint32_t N_MF           = 256u;
constexpr uint32_t M_OUT          = N_SEARCH - N_MF + 1;


constexpr uint32_t N_G            = 3u;
constexpr uint32_t N_PSS          = 3u;



constexpr uint32_t Q_FRAC_BITS    = 13u;
constexpr int32_t  Q_SCALE        = 1 << Q_FRAC_BITS;



constexpr uint32_t CP_FIRST_768   = 44u;
constexpr uint32_t CP_OTHER_768   = 36u;










constexpr uint32_t OUTPUT_FP16_LEN = 16u;
constexpr uint32_t OUTPUT_BYTES    = OUTPUT_FP16_LEN * 2u;


constexpr size_t INPUT_BYTES       = static_cast<size_t>(N_SEARCH) * 2u * 2u;
constexpr size_t PSS_REF_BYTES = 16384u;
constexpr size_t TWIDDLE_BYTES     = static_cast<size_t>(N_G) * N_SEARCH * 2u * 2u;




constexpr uint32_t M_TILE_PLACEHOLDER = 16384u;




constexpr uint32_t BLOCK_DIM_SKELETON = 4u;


constexpr size_t TILING_TOTAL_SIZE = 128u;
constexpr size_t WS_TOTAL          = 1ull * 1024ull * 1024ull;

}