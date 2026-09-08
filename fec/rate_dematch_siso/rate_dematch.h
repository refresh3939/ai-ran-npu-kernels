#pragma once
#include <cstddef>
#include <cstdint>
























namespace airan {

constexpr uint32_t Q_M          = 8;
constexpr uint32_t N_STREAMS    = 8;
constexpr uint32_t N_SYM        = 19152;
constexpr uint32_t N_SYM_PAD    = 19200;
constexpr uint32_t N_SLOT_MAX   = 23;
constexpr uint32_t BLOCK_DIM    = 4;

constexpr uint32_t C_NUM        = 143;
constexpr uint32_t LDPC_N       = 26112;
constexpr uint32_t N_2Z         = 768;

constexpr int16_t  SCALE        = 8;
constexpr int16_t  LLR_CLIP     = 5120;


constexpr uint32_t STREAM_STRIDE = N_SYM_PAD;
constexpr uint32_t SLOT_STRIDE   = N_STREAMS * N_SYM_PAD;
constexpr uint32_t DESCRAM_LEN   = (uint32_t)N_SLOT_MAX * N_STREAMS * N_SYM_PAD;


constexpr uint32_t UB_SRC_MAX   = 25344;
constexpr uint32_t UB_E_MAX     = 24656;



constexpr uint32_t NFLOOR       = 87;
constexpr uint32_t EQ_LO        = 3080;
constexpr uint32_t EQ_HI        = 3081;
constexpr uint32_t E_LO         = 24640;
constexpr uint32_t E_HI         = 24648;
constexpr uint32_t EPAD_LO      = 24640;
constexpr uint32_t EPAD_HI      = 24656;


constexpr uint32_t LPM          = 3168;

constexpr uint32_t INPUT_RING_DEPTH = 2;

constexpr size_t TILING_TOTAL_SIZE = 128;
constexpr size_t WS_TOTAL          = 4 * 1024 * 1024;

}
