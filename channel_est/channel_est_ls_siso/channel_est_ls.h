














#pragma once
#include <cstddef>
#include <cstdint>

namespace airan {


constexpr uint32_t N_FFT             = 2048;
constexpr uint32_t N_SC_USED         = 1596;
constexpr uint32_t N_SYMBOL          = 14;
constexpr uint32_t N_BLOCKS          = 4;


constexpr uint32_t N_DMRS_SYM        = 2;
constexpr uint32_t DMRS_SYM_0        = 2;
constexpr uint32_t DMRS_SYM_1        = 11;
constexpr uint32_t N_DMRS_RE_PER_SYM = N_SC_USED / 2;




























constexpr uint32_t SC_PER_CORE_NORM       = 400;
constexpr uint32_t SC_PER_CORE_LAST       = 396;
constexpr uint32_t DMRS_RE_PER_CORE_NORM  = SC_PER_CORE_NORM / 2;
constexpr uint32_t DMRS_RE_PER_CORE_LAST  = SC_PER_CORE_LAST / 2;


constexpr uint32_t DMRS_RE_PER_CORE_NORM_WITH_HALO = DMRS_RE_PER_CORE_NORM + 1;
constexpr uint32_t DMRS_RE_PER_CORE_LAST_WITH_HALO = DMRS_RE_PER_CORE_LAST;
constexpr uint32_t DMRS_RE_PER_CORE_MAX            = DMRS_RE_PER_CORE_NORM_WITH_HALO;

constexpr uint32_t SC_PER_CORE_MAX   = SC_PER_CORE_NORM;


static_assert(3 * SC_PER_CORE_NORM + SC_PER_CORE_LAST == N_SC_USED,
              "SC partition mismatch");



constexpr uint32_t INPUT_GM_INT16_LEN  = N_SYMBOL * N_SC_USED * 2;
constexpr uint32_t PILOT_GM_INT16_LEN  = N_DMRS_SYM * N_DMRS_RE_PER_SYM * 2;
constexpr uint32_t OUTPUT_GM_INT16_LEN = N_SYMBOL * N_SC_USED * 2;




constexpr uint32_t PILOT_GM_PADDING_INT16 = 32;

constexpr size_t   INPUT_GM_BYTES  = INPUT_GM_INT16_LEN  * sizeof(int16_t);
constexpr size_t   PILOT_GM_BYTES  = (PILOT_GM_INT16_LEN + PILOT_GM_PADDING_INT16) * sizeof(int16_t);
constexpr size_t   OUTPUT_GM_BYTES = OUTPUT_GM_INT16_LEN * sizeof(int16_t);



constexpr uint32_t Q_BITS            = 15;
constexpr float    Q_SCALE           = 32768.0f;
constexpr float    Q_SCALE_INV       = 1.0f / 32768.0f;


constexpr size_t TILING_TOTAL_SIZE   = 128;
constexpr size_t WS_TOTAL            = 1 * 1024 * 1024;




constexpr uint32_t USE_XOR_AIV_MAP   = 0;






__attribute__((always_inline)) inline
void GetCoreScRange(uint32_t aiv_id,
                    uint32_t &sc_start,
                    uint32_t &sc_count,
                    uint32_t &dmrs_re_count_with_halo) {
    if (aiv_id < N_BLOCKS - 1) {
        sc_start                = aiv_id * SC_PER_CORE_NORM;
        sc_count                = SC_PER_CORE_NORM;
        dmrs_re_count_with_halo = DMRS_RE_PER_CORE_NORM_WITH_HALO;
    } else {
        sc_start                = (N_BLOCKS - 1) * SC_PER_CORE_NORM;
        sc_count                = SC_PER_CORE_LAST;
        dmrs_re_count_with_halo = DMRS_RE_PER_CORE_LAST_WITH_HALO;
    }
}

}
