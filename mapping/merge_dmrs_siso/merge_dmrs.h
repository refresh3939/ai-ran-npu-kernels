










#pragma once
#include <cstdint>
#include <cstddef>

namespace airan {

constexpr uint32_t N_SYMBOL    = 14;
constexpr uint32_t N_SC_USED   = 1596;
constexpr uint32_t N_SC_PAD    = 1664;
constexpr uint32_t N_DMRS_RE   = 798;
constexpr uint32_t N_DMRS_PAD  = 896;
constexpr uint32_t N_DMRS_SYM  = 2;
constexpr uint32_t DMRS_SYM_0  = 2;
constexpr uint32_t DMRS_SYM_1  = 11;
constexpr uint32_t COMB        = 2;
constexpr uint32_t DELTA       = 0;

constexpr size_t   GU_BYTES    = (size_t)N_SYMBOL  * N_SC_PAD  * sizeof(uint16_t);
constexpr size_t   DMRS_BYTES  = (size_t)N_DMRS_SYM * N_DMRS_PAD * sizeof(uint16_t);




void merge_dmrs_insert(uint8_t* gu_re, uint8_t* gu_im,
                       uint8_t* dmrs_re, uint8_t* dmrs_im);

}
