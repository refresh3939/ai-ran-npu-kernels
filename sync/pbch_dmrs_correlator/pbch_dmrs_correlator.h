













#ifndef PBCH_DMRS_CORRELATOR_H_
#define PBCH_DMRS_CORRELATOR_H_

#include <cstdint>


constexpr uint32_t N_DMRS_RE      = 144u;
constexpr uint32_t L_MAX_HW       = 8u;
constexpr uint32_t N_PCID         = 1008u;


constexpr uint16_t M_MMAD         = 16u;
constexpr uint16_t K_SUB          = 144u;
constexpr uint16_t N_SUB          = 16u;
constexpr uint16_t N_SPLITS       = 1u;
constexpr uint16_t K_SPLITS       = 1u;



















constexpr uint32_t IN_R_RE_BYTES      = M_MMAD * K_SUB * 2u;
constexpr uint32_t IN_R_IM_BYTES      = M_MMAD * K_SUB * 2u;
constexpr uint32_t IN_D_RE_BYTES      = K_SUB * N_SUB * 2u;
constexpr uint32_t IN_D_IM_BYTES      = K_SUB * N_SUB * 2u;
constexpr uint32_t IN_D_IM_NEG_BYTES  = K_SUB * N_SUB * 2u;












constexpr uint32_t OUT_FP32_COUNT     = 24u;
constexpr uint32_t OUT_BYTES          = OUT_FP32_COUNT * 4u;


constexpr float SENTINEL_F32 = 7.0f;

#endif
