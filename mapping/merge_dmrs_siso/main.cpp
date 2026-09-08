







#include "merge_dmrs.h"
#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CK(call) do { auto _e=(call); if(_e!=ACL_SUCCESS){ \
    std::fprintf(stderr,"ACL err %d @ %s:%d\n",(int)_e,__FILE__,__LINE__); std::exit(1);} } while(0)

using namespace airan;

static std::string Root() { const char* r=std::getenv("AIRAN_DATA_DIR"); return r?r:"."; }
static bool RD(const std::string& p, void* buf, size_t bytes) {
    std::ifstream f(p, std::ios::binary|std::ios::ate);
    if(!f.is_open()){ std::fprintf(stderr,"[err] open %s\n",p.c_str()); return false; }
    size_t sz=(size_t)f.tellg(); if(sz!=bytes){ std::fprintf(stderr,"[err] %s sz %zu!=%zu\n",p.c_str(),sz,bytes); return false; }
    f.seekg(0); f.read((char*)buf,(std::streamsize)bytes); return true;
}

int main() {
    const std::string g = Root() + "/data/golden/case_0";
    const size_t NG = GU_BYTES/sizeof(uint16_t), ND = DMRS_BYTES/sizeof(uint16_t);
    std::vector<uint16_t> guInRe(NG),guInIm(NG),dRe(ND),dIm(ND),mRe(NG),mIm(NG);
    if(!RD(g+"/gu_in_re.bin",guInRe.data(),GU_BYTES)) return 1;
    if(!RD(g+"/gu_in_im.bin",guInIm.data(),GU_BYTES)) return 1;
    if(!RD(g+"/dmrs_in_re.bin",dRe.data(),DMRS_BYTES)) return 1;
    if(!RD(g+"/dmrs_in_im.bin",dIm.data(),DMRS_BYTES)) return 1;
    if(!RD(g+"/gu_merged_re.bin",mRe.data(),GU_BYTES)) return 1;
    if(!RD(g+"/gu_merged_im.bin",mIm.data(),GU_BYTES)) return 1;

    printf("=== merge_dmrs (TX DMRS comb-2 insert) standalone ===\n");
    printf("  gu [%u,%u]  dmrs [%u,%u]  DMRS sym {%u,%u}  delta=%u\n",
           N_SYMBOL,N_SC_PAD,N_DMRS_SYM,N_DMRS_PAD,DMRS_SYM_0,DMRS_SYM_1,DELTA);

    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream st=nullptr; CK(aclrtCreateStream(&st));

    uint8_t *dGuRe,*dGuIm,*dDRe,*dDIm;
    CK(aclrtMalloc((void**)&dGuRe,GU_BYTES,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc((void**)&dGuIm,GU_BYTES,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc((void**)&dDRe,DMRS_BYTES,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc((void**)&dDIm,DMRS_BYTES,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMemcpy(dGuRe,GU_BYTES,guInRe.data(),GU_BYTES,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dGuIm,GU_BYTES,guInIm.data(),GU_BYTES,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dDRe,DMRS_BYTES,dRe.data(),DMRS_BYTES,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dDIm,DMRS_BYTES,dIm.data(),DMRS_BYTES,ACL_MEMCPY_HOST_TO_DEVICE));

    merge_dmrs_insert(dGuRe,dGuIm,dDRe,dDIm);

    std::vector<uint16_t> oRe(NG),oIm(NG);
    CK(aclrtMemcpy(oRe.data(),GU_BYTES,dGuRe,GU_BYTES,ACL_MEMCPY_DEVICE_TO_HOST));
    CK(aclrtMemcpy(oIm.data(),GU_BYTES,dGuIm,GU_BYTES,ACL_MEMCPY_DEVICE_TO_HOST));

    auto GU=[&](const std::vector<uint16_t>& v,uint32_t l,uint32_t c){ return v[(size_t)l*N_SC_PAD+c]; };

    size_t diff=0; for(size_t i=0;i<NG;++i){ if(oRe[i]!=mRe[i])++diff; if(oIm[i]!=mIm[i])++diff; }

    size_t rt=0; for(uint32_t s=0;s<N_DMRS_SYM;++s){ uint32_t l=s?DMRS_SYM_1:DMRS_SYM_0;
        for(uint32_t k=0;k<N_DMRS_RE;++k){ if(GU(oRe,l,COMB*k+DELTA)!=dRe[(size_t)s*N_DMRS_PAD+k])++rt;
                                           if(GU(oIm,l,COMB*k+DELTA)!=dIm[(size_t)s*N_DMRS_PAD+k])++rt; } }

    size_t nz=0; for(uint32_t s=0;s<N_DMRS_SYM;++s){ uint32_t l=s?DMRS_SYM_1:DMRS_SYM_0;
        for(uint32_t k=0;k<N_DMRS_RE;++k){ if(GU(oRe,l,COMB*k+1))++nz; if(GU(oIm,l,COMB*k+1))++nz; }
        for(uint32_t c=N_SC_USED;c<N_SC_PAD;++c){ if(GU(oRe,l,c))++nz; if(GU(oIm,l,c))++nz; } }

    size_t dch=0; for(uint32_t l=0;l<N_SYMBOL;++l){ if(l==DMRS_SYM_0||l==DMRS_SYM_1)continue;
        for(uint32_t c=0;c<N_SC_PAD;++c){ if(GU(oRe,l,c)!=GU(guInRe,l,c))++dch; if(GU(oIm,l,c)!=GU(guInIm,l,c))++dch; } }

    printf("\n  [1] bit-exact vs gu_merged : %zu diff\n",diff);
    printf("  [2] round-trip 偶数SC==dmrs: %zu err\n",rt);
    printf("  [3] 奇数SC+pad=0           : %zu nonzero\n",nz);
    printf("  [4] 数据符号行不变         : %zu changed\n",dch);
    bool ok=(diff==0&&rt==0&&nz==0&&dch==0);
    printf("\n  => %s\n", ok?"PASS (bit-exact, round-trip, zero-odd, data-intact)":"FAIL");

    CK(aclrtFree(dGuRe));CK(aclrtFree(dGuIm));CK(aclrtFree(dDRe));CK(aclrtFree(dDIm));
    CK(aclrtDestroyStream(st)); CK(aclrtResetDevice(0)); CK(aclFinalize());
    return ok?0:1;
}
