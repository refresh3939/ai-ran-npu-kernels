

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "sss_correlator.h"
#include "sss_tiling.h"









extern "C" void GenerateSssTiling(const char *socVersion,
                                  uint8_t *buf,
                                  int32_t mu_t_pss,
                                  int32_t n_id_2,
                                  int32_t g_hat)
{
    (void)socVersion;

    SssTilingV1 td;
    memset(&td, 0, sizeof(td));

    td.mu_t_pss = mu_t_pss;
    td.n_id_2   = n_id_2;
    td.g_hat    = g_hat;

    td.sym2_start_nominal = mu_t_pss + SYM2_OFFSET_FROM_MU_T;
    td.seg_start          = td.sym2_start_nominal - static_cast<int32_t>(T_HALF);
    td.g_idx              = g_hat + 1;
    td.block_dim          = 4u;


    if (td.seg_start < 0 ||
        td.seg_start + static_cast<int32_t>(SEG_LEN) > static_cast<int32_t>(N_SEARCH)) {
        printf("[sss_tiling] WARN: seg_start=%d out of range [0, %u]\n",
               td.seg_start, N_SEARCH - SEG_LEN);
    }

    memcpy(buf, &td, sizeof(td));

    printf("[sss_tiling] mu_t_pss=%d n_id_2=%d g_hat=%d "
           "sym2_start=%d seg_start=%d g_idx=%d\n",
           td.mu_t_pss, td.n_id_2, td.g_hat,
           td.sym2_start_nominal, td.seg_start, td.g_idx);
}
