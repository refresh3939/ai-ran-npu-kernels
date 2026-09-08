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
constexpr uint32_t N_CB_BUF     = 25344;


constexpr uint32_t STREAM_STRIDE = N_SYM_PAD;
constexpr uint32_t SLOT_STRIDE   = N_STREAMS * N_SYM_PAD;
constexpr uint32_t LAYOUT_LEN    = (uint32_t)N_SLOT_MAX * N_STREAMS * N_SYM_PAD;
constexpr uint32_t CODEWORD_LEN  = C_NUM * N_CB_BUF;


constexpr uint32_t NFLOOR       = 87;
constexpr uint32_t EQ_LO        = 3080;
constexpr uint32_t EQ_HI        = 3081;
constexpr uint32_t E_LO         = 24640;
constexpr uint32_t E_HI         = 24648;
constexpr uint32_t EPAD_LO      = 24640;
constexpr uint32_t EPAD_HI      = 24656;
constexpr uint32_t EPAD32       = 24672;


constexpr uint32_t STREAMS_PER_AIV = N_STREAMS / BLOCK_DIM;
constexpr uint32_t SLOT_PAIR_ELEMS = STREAMS_PER_AIV * N_SYM;
constexpr uint32_t SLOT_RING_DEPTH = 1;
constexpr uint32_t PAIR_INPUT_PAD  = 6208;
constexpr uint32_t PAIR_RING_DEPTH = 4;

constexpr size_t TILING_TOTAL_SIZE = 128;
constexpr size_t WS_TOTAL          = 4 * 1024 * 1024;

}
