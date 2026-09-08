

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "pbch_dmrs_correlator.h"
#include "pbch_dmrs_tiling.h"

extern "C" void GeneratePbchDmrsTiling(const char *socVersion,
                                       uint8_t *buf,
                                       int32_t l_max,
                                       int32_t pcid)
{
    (void)socVersion;

    PbchDmrsTilingV1 td;
    memset(&td, 0, sizeof(td));

    if (l_max != 4 && l_max != 8) {
        printf("[pbch_dmrs_tiling] WARN: l_max=%d not in {4,8}, defaulting to 8\n", l_max);
        l_max = 8;
    }
    if (pcid < 0 || pcid >= static_cast<int32_t>(N_PCID)) {
        printf("[pbch_dmrs_tiling] WARN: pcid=%d out of [0, %u)\n", pcid, N_PCID);
    }

    td.l_max     = l_max;
    td.pcid      = pcid;
    td.block_dim = 4u;

    memcpy(buf, &td, sizeof(td));

    printf("[pbch_dmrs_tiling] l_max=%d pcid=%d block_dim=%u\n",
           td.l_max, td.pcid, td.block_dim);
}
