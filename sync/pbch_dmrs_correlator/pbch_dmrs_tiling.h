

#ifndef PBCH_DMRS_TILING_H_
#define PBCH_DMRS_TILING_H_

#include <cstdint>

struct PbchDmrsTilingV1 {
    int32_t  l_max;
    int32_t  pcid;
    uint32_t block_dim;
    int32_t  _pad0;


    uint32_t _pad[12];
};
static_assert(sizeof(PbchDmrsTilingV1) == 64,
              "PbchDmrsTilingV1 must be 64 bytes");

#endif
