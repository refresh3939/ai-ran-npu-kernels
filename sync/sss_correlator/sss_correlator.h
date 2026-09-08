


#ifndef SSS_CORRELATOR_H_
#define SSS_CORRELATOR_H_

#include <cstdint>


constexpr uint32_t FS_HZ          = 7680000u;
constexpr uint32_t SCS_HZ         = 30000u;
constexpr uint32_t N_FFT          = 256u;
constexpr uint32_t N_SC_SSB       = 240u;
constexpr uint32_t N_SC_PSS       = 127u;


constexpr uint32_t CP_FIRST_768   = 44u;
constexpr uint32_t CP_OTHER_768   = 36u;


constexpr uint32_t SYM0_LEN       = CP_FIRST_768 + N_FFT;
constexpr uint32_t SYM_OTHER_LEN  = CP_OTHER_768 + N_FFT;
constexpr uint32_t N_SSB_SAMPLES  = SYM0_LEN + 3u * SYM_OTHER_LEN;


constexpr uint32_t N_SEARCH       = 153600u;
constexpr uint32_t N_ID_1_COUNT   = 336u;
constexpr uint32_t N_ID_2_COUNT   = 3u;
constexpr uint32_t T_HALF         = 4u;
constexpr uint32_t N_TAU          = 2u * T_HALF + 1u;


constexpr uint32_t N_TAU_PADDED   = 16u;
constexpr uint32_t K_FFT_PADDED   = N_FFT;
constexpr uint32_t N_ID_1_PADDED  = 352u;











constexpr uint16_t M_MMAD         = 16;
constexpr uint16_t N_SUB          = 16;
constexpr uint16_t K_SUB          = 256;
constexpr uint16_t N_SPLITS       = N_ID_1_PADDED / N_SUB;
constexpr uint16_t K_SPLITS       = N_FFT / K_SUB;


constexpr uint32_t Q_FRAC_BITS    = 13u;
constexpr int32_t  Q_SCALE        = 1 << Q_FRAC_BITS;






constexpr int32_t SYM2_OFFSET_FROM_MU_T =
      static_cast<int32_t>(N_FFT)
    + static_cast<int32_t>(SYM_OTHER_LEN)
    + static_cast<int32_t>(CP_OTHER_768);


constexpr uint32_t SEG_LEN        = 2u * T_HALF + N_FFT;


constexpr uint64_t IN_RX_BYTES         = static_cast<uint64_t>(N_SEARCH) * 4ull;
constexpr uint64_t IN_TABLE_HALF_BYTES = static_cast<uint64_t>(N_ID_2_COUNT)
                                       * static_cast<uint64_t>(N_ID_1_COUNT)
                                       * static_cast<uint64_t>(N_FFT) * 2ull;
constexpr uint64_t IN_TWID_HALF_BYTES  = static_cast<uint64_t>(N_ID_2_COUNT)
                                       * static_cast<uint64_t>(SEG_LEN) * 2ull;
constexpr uint32_t OUT_BYTES           = 16u * static_cast<uint32_t>(sizeof(uint16_t));



constexpr uint16_t SENTINEL_HALF_7 = 0x4700u;

#endif