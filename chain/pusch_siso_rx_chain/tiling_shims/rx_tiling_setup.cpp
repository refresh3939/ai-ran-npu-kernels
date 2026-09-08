
#include "../rx_chain.h"
#include "acl/acl.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

#define CK_T(x) do{ aclError _e=(x); if(_e!=ACL_SUCCESS){ \
  std::fprintf(stderr,"tiling acl fail %d @%d\n",_e,__LINE__); std::exit(1);} }while(0)

static constexpr size_t TIL_A = 4096;
static const char* SOC = SOC_VERSION;
static constexpr uint32_t BD = 4;
static size_t ws_bytes = 16u*1024*1024;

extern "C" void gt_decimate(const char*, uint8_t*);
extern "C" void gt_pss_correlator(const char*, uint8_t*);
extern "C" void gt_pss_cfo_estimator(const char*, uint8_t*);
extern "C" void gt_ssb_fft(const char*, uint8_t*);
extern "C" void gt_cfo_estimate(const char*, uint8_t*);
extern "C" void gt_cfo_compensate(const char*, uint8_t*);
extern "C" void gt_ofdm_demod(const char*, uint8_t*);
extern "C" void gt_channel_est_ls(const char*, uint8_t*);
extern "C" void gt_equalize(const char*, uint8_t*);
extern "C" uint8_t* gt_qam256_demod(const char*, uint32_t);
extern "C" size_t   gts_qam256_demod();
extern "C" size_t   gws_qam256_demod();
extern "C" void gt_cfo_dmrs(const char*, uint8_t*);
extern "C" void gt_timing_tracker(const char*, uint8_t*);
extern "C" uint8_t* gt_descramble(const char*, uint32_t);
extern "C" size_t   gts_descramble();
extern "C" size_t   gws_descramble();
extern "C" uint8_t* gt_rate_dematch(const char*, uint32_t);
extern "C" size_t   gts_rate_dematch();
extern "C" size_t   gws_rate_dematch();
extern "C" uint8_t* gt_ldpc_decode(const char*, uint32_t);
extern "C" size_t   gts_ldpc_decode();
extern "C" size_t   gws_ldpc_decode();

static uint8_t* dmalloc_t(size_t b){ void* p=nullptr;
  CK_T(aclrtMalloc(&p,b?b:1,ACL_MEM_MALLOC_HUGE_FIRST)); CK_T(aclrtMemset(p,b?b:1,0,b?b:1));
  return (uint8_t*)p; }

namespace airan_rx {
void setup_tilings(RxArena& a){

    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_decimate(SOC, h);
      a.tiling_bytes[OP_decimate]=n; a.tiling[OP_decimate]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_decimate],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_pss_correlator(SOC, h);
      a.tiling_bytes[OP_pss_correlator]=n; a.tiling[OP_pss_correlator]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_pss_correlator],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_pss_cfo_estimator(SOC, h);
      a.tiling_bytes[OP_pss_cfo_estimator]=n; a.tiling[OP_pss_cfo_estimator]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_pss_cfo_estimator],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }
    a.tiling_bytes[OP_sss_correlator]=256; a.tiling[OP_sss_correlator]=dmalloc_t(256);
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_ssb_fft(SOC, h);
      a.tiling_bytes[OP_ssb_fft]=n; a.tiling[OP_ssb_fft]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_ssb_fft],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }
    a.tiling_bytes[OP_pbch_dmrs_correlator]=256; a.tiling[OP_pbch_dmrs_correlator]=dmalloc_t(256);
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_cfo_estimate(SOC, h);
      a.tiling_bytes[OP_cfo_estimate]=n; a.tiling[OP_cfo_estimate]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_cfo_estimate],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_cfo_compensate(SOC, h);
      a.tiling_bytes[OP_cfo_compensate]=n; a.tiling[OP_cfo_compensate]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_cfo_compensate],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_ofdm_demod(SOC, h);
      a.tiling_bytes[OP_ofdm_demod]=n; a.tiling[OP_ofdm_demod]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_ofdm_demod],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }
    a.tiling_bytes[OP_re_demap]=256; a.tiling[OP_re_demap]=dmalloc_t(256);
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_channel_est_ls(SOC, h);
      a.tiling_bytes[OP_channel_est_ls]=n; a.tiling[OP_channel_est_ls]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_channel_est_ls],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_equalize(SOC, h);
      a.tiling_bytes[OP_equalize]=n; a.tiling[OP_equalize]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_equalize],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }
    { uint8_t* h=gt_qam256_demod(SOC, BD); size_t n=gts_qam256_demod();
      a.tiling_bytes[OP_qam256_demod]=n; a.tiling[OP_qam256_demod]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_qam256_demod],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE));
      size_t w=gws_qam256_demod(); if(w>ws_bytes) ws_bytes=w; free(h); }
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_cfo_dmrs(SOC, h);
      a.tiling_bytes[OP_cfo_dmrs]=n; a.tiling[OP_cfo_dmrs]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_cfo_dmrs],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_timing_tracker(SOC, h);
      a.tiling_bytes[OP_timing_tracker]=n; a.tiling[OP_timing_tracker]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_timing_tracker],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }
    { uint8_t* h=gt_descramble(SOC, BD); size_t n=gts_descramble();
      a.tiling_bytes[OP_descramble]=n; a.tiling[OP_descramble]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_descramble],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE));
      size_t w=gws_descramble(); if(w>ws_bytes) ws_bytes=w; free(h); }
    { uint8_t* h=gt_rate_dematch(SOC, BD); size_t n=gts_rate_dematch();
      a.tiling_bytes[OP_rate_dematch]=n; a.tiling[OP_rate_dematch]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_rate_dematch],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE));
      size_t w=gws_rate_dematch(); if(w>ws_bytes) ws_bytes=w; free(h); }
    { uint8_t* h=gt_ldpc_decode(SOC, BD); size_t n=gts_ldpc_decode();
      a.tiling_bytes[OP_ldpc_decode]=n; a.tiling[OP_ldpc_decode]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_ldpc_decode],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE));
      size_t w=gws_ldpc_decode(); if(w>ws_bytes) ws_bytes=w; free(h); }
    a.ws_bytes = ws_bytes;
    a.ws = dmalloc_t(ws_bytes);
}
}
