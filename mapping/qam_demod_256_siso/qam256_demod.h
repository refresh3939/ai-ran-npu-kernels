















#pragma once
#include <cstddef>
#include <cstdint>

namespace airan {


constexpr uint32_t Q_M           = 8;
constexpr uint32_t N_SYMBOL_MAX  = 14;


constexpr uint32_t N_SC_PAD_MAX  = 1664;
constexpr uint32_t N_DATA_SYM_MAX= 14;


constexpr uint32_t N_SC_USED_DEF = 1596;
constexpr uint32_t N_SC_PAD_DEF  = 1664;
constexpr uint32_t N_DATA_SYM_DEF= 12;
constexpr uint32_t N_RE_DATA_DEF = N_DATA_SYM_DEF * N_SC_USED_DEF;
constexpr uint32_t N_SYM_PAD_DEF = 19200;
constexpr uint32_t BLOCK_DIM     = 4;
constexpr uint32_t DMRS_SYM_0    = 2;
constexpr uint32_t DMRS_SYM_1    = 11;


constexpr uint32_t N_SYMBOL      = N_SYMBOL_MAX;
constexpr uint32_t N_DATA_SYM    = N_DATA_SYM_DEF;
constexpr uint32_t N_SC_USED     = N_SC_USED_DEF;
constexpr uint32_t N_SC_PAD      = N_SC_PAD_DEF;
constexpr uint32_t N_RE_DATA     = N_RE_DATA_DEF;
constexpr uint32_t N_SYM_PAD     = N_SYM_PAD_DEF;


constexpr int16_t  Q_SCALE        = 32;
constexpr int16_t  LLR_CLIP_FX    = 2560;
constexpr int16_t  Y_SCALED_CLIP  = 1000;

constexpr float D_256QAM        = 0.07669649888473704f;
constexpr float D_X_QSCALE      = 0.07669649888473704f * 32.0f;
constexpr float D2_X_QSCALE     = (0.07669649888473704f *
                                   0.07669649888473704f) * 32.0f;






struct TilingData {
    int32_t n_sc_used;
    int32_t n_sc_pad;
    int32_t n_data_sym;
    int32_t n_re_data;
    int32_t n_sym_pad;
    int32_t block_dim;
    int32_t llr_clip;
    int32_t y_scaled_clip;
    int32_t phys_sym[N_SYMBOL_MAX];
    int32_t group_sz;
    int32_t reserved[32 - 8 - N_SYMBOL_MAX - 1];
};
static_assert(sizeof(TilingData) == 128, "TilingData must be exactly 128 bytes");


namespace til {
constexpr uint32_t N_SC_USED   = 0;
constexpr uint32_t N_SC_PAD    = 1;
constexpr uint32_t N_DATA_SYM  = 2;
constexpr uint32_t N_RE_DATA   = 3;
constexpr uint32_t N_SYM_PAD   = 4;
constexpr uint32_t BLOCK_DIM   = 5;
constexpr uint32_t LLR_CLIP    = 6;
constexpr uint32_t Y_CLIP      = 7;
constexpr uint32_t PHYS_SYM0   = 8;
constexpr uint32_t GROUP_SZ    = 22;
constexpr uint32_t N_INT32     = 32;
}

constexpr size_t TILING_TOTAL_SIZE = 128;



constexpr size_t WS_SYSTEM_RESERVED = 2 * 1024 * 1024;
constexpr size_t WS_USER_BYTES      = 1 * 1024 * 1024;
constexpr size_t WS_TOTAL           = WS_SYSTEM_RESERVED + WS_USER_BYTES;


constexpr size_t IN_BYTES        = (size_t)N_SYMBOL_MAX * N_SC_PAD_DEF * sizeof(uint16_t);
constexpr size_t OUT_BYTES       = (size_t)Q_M * N_SYM_PAD_DEF * sizeof(int16_t);

}
