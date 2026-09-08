#include "../tx_chain.h"
#include "acl/acl.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

#define CK_T(x) do{ aclError _e=(x); if(_e!=ACL_SUCCESS){ \
  std::fprintf(stderr,"tiling acl fail %d @%d\n",_e,__LINE__); std::exit(1);} }while(0)

static constexpr size_t TIL_A = 65536;
static const char* SOC = SOC_VERSION;
static constexpr uint32_t BD = 4;
static size_t ws_bytes = 16u*1024*1024;

extern "C" uint8_t* gt_ldpc_encode(const char*, uint32_t);
extern "C" size_t   gts_ldpc_encode();
extern "C" size_t   gws_ldpc_encode();
extern "C" uint8_t* gt_rate_match(const char*, uint32_t);
extern "C" size_t   gts_rate_match();
extern "C" size_t   gws_rate_match();
extern "C" uint8_t* gt_scramble(const char*, uint32_t);
extern "C" size_t   gts_scramble();
extern "C" size_t   gws_scramble();
extern "C" uint8_t* gt_qam256_mod(const char*, uint32_t);
extern "C" size_t   gts_qam256_mod();
extern "C" size_t   gws_qam256_mod();
extern "C" void gt_dmrs_gen(const char*, uint8_t*);

static uint8_t* dmalloc_t(size_t b){ void* p=nullptr;
  CK_T(aclrtMalloc(&p,b?b:1,ACL_MEM_MALLOC_HUGE_FIRST)); CK_T(aclrtMemset(p,b?b:1,0,b?b:1));
  return (uint8_t*)p; }

namespace airan_tx {
void setup_tilings(TxArena& a){

    { uint8_t* h=gt_ldpc_encode(SOC, BD); size_t n=gts_ldpc_encode();
      if(n>TIL_A) n=TIL_A;
      a.tiling_bytes[OP_ldpc_encode]=n; a.tiling[OP_ldpc_encode]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_ldpc_encode],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE));
      size_t w=gws_ldpc_encode(); if(w>ws_bytes) ws_bytes=w;   }
    { uint8_t* h=gt_rate_match(SOC, BD); size_t n=gts_rate_match();
      if(n>TIL_A) n=TIL_A;
      a.tiling_bytes[OP_rate_match]=n; a.tiling[OP_rate_match]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_rate_match],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE));
      size_t w=gws_rate_match(); if(w>ws_bytes) ws_bytes=w;   }
    { uint8_t* h=gt_scramble(SOC, BD); size_t n=gts_scramble();
      if(n>TIL_A) n=TIL_A;
      a.tiling_bytes[OP_scramble]=n; a.tiling[OP_scramble]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_scramble],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE));
      size_t w=gws_scramble(); if(w>ws_bytes) ws_bytes=w;   }
    { uint8_t* h=gt_qam256_mod(SOC, BD); size_t n=gts_qam256_mod();
      if(n>TIL_A) n=TIL_A;
      a.tiling_bytes[OP_qam256_mod]=n; a.tiling[OP_qam256_mod]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_qam256_mod],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE));
      size_t w=gws_qam256_mod(); if(w>ws_bytes) ws_bytes=w;   }
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_dmrs_gen(SOC, h);
      a.tiling_bytes[OP_dmrs_gen]=n; a.tiling[OP_dmrs_gen]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_dmrs_gen],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE));   }
    a.tiling_bytes[OP_re_map]=256; a.tiling[OP_re_map]=dmalloc_t(256);
    a.tiling_bytes[OP_ofdm_mod]=256; a.tiling[OP_ofdm_mod]=dmalloc_t(256);
    a.ws_bytes = ws_bytes;
    a.ws = dmalloc_t(ws_bytes);
}
}
