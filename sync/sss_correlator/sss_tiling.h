

#ifndef SSS_TILING_H_
#define SSS_TILING_H_

#include <cstdint>

struct SssTilingV1 {

    int32_t  mu_t_pss;
    int32_t  n_id_2;
    int32_t  g_hat;
    int32_t  _pad0;


    int32_t  sym2_start_nominal;
    int32_t  seg_start;
    int32_t  g_idx;
    uint32_t block_dim;


    uint32_t _pad[8];
};
static_assert(sizeof(SssTilingV1) == 64,
              "SssTilingV1 must be 64 bytes (DataCopy align)");

#endif
