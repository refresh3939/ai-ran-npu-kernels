








#include "merge_dmrs.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace airan {

static inline void mchk(aclError e, const char* what) {
    if (e != ACL_SUCCESS) { std::fprintf(stderr, "[merge_dmrs] ACL err %d @ %s\n", (int)e, what); std::exit(1); }
}

void merge_dmrs_insert(uint8_t* gu_re, uint8_t* gu_im,
                       uint8_t* dmrs_re, uint8_t* dmrs_im)
{
    const int    DMRS_SYM[2] = { (int)DMRS_SYM_0, (int)DMRS_SYM_1 };
    const size_t H = sizeof(uint16_t);


    std::vector<uint16_t> dre((size_t)N_DMRS_SYM * N_DMRS_PAD);
    std::vector<uint16_t> dim((size_t)N_DMRS_SYM * N_DMRS_PAD);
    mchk(aclrtMemcpy(dre.data(), dre.size()*H, dmrs_re, dre.size()*H, ACL_MEMCPY_DEVICE_TO_HOST), "D2H dmrs_re");
    mchk(aclrtMemcpy(dim.data(), dim.size()*H, dmrs_im, dim.size()*H, ACL_MEMCPY_DEVICE_TO_HOST), "D2H dmrs_im");


    for (int s = 0; s < (int)N_DMRS_SYM; ++s) {
        const uint32_t l = (uint32_t)DMRS_SYM[s];
        std::vector<uint16_t> rr(N_SC_PAD, 0), ri(N_SC_PAD, 0);
        for (uint32_t k = 0; k < N_DMRS_RE; ++k) {
            rr[COMB*k + DELTA] = dre[(size_t)s*N_DMRS_PAD + k];
            ri[COMB*k + DELTA] = dim[(size_t)s*N_DMRS_PAD + k];
        }
        mchk(aclrtMemcpy(gu_re + (size_t)l*N_SC_PAD*H, N_SC_PAD*H, rr.data(), N_SC_PAD*H, ACL_MEMCPY_HOST_TO_DEVICE), "H2D gu_re row");
        mchk(aclrtMemcpy(gu_im + (size_t)l*N_SC_PAD*H, N_SC_PAD*H, ri.data(), N_SC_PAD*H, ACL_MEMCPY_HOST_TO_DEVICE), "H2D gu_im row");
    }
}

}
