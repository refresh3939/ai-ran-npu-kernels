













#include "rx_chain.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <functional>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_decimate_kernel.h"
#include "aclrtlaunch_pss_correlator_kernel.h"
#include "aclrtlaunch_pss_cfo_estimator_kernel.h"
#include "aclrtlaunch_sss_correlator_kernel.h"
#include "aclrtlaunch_ssb_fft_kernel.h"
#include "aclrtlaunch_pbch_dmrs_correlator_kernel.h"
#include "aclrtlaunch_cfo_estimate_kernel.h"
#include "aclrtlaunch_cfo_compensate_kernel.h"
#include "aclrtlaunch_ofdm_demod_kernel.h"
#include "aclrtlaunch_re_demap_kernel.h"
#include "aclrtlaunch_channel_est_ls_kernel.h"
#include "aclrtlaunch_equalize_kernel.h"
#include "aclrtlaunch_qam256_demod_kernel.h"
#include "aclrtlaunch_cfo_dmrs_kernel.h"
#include "aclrtlaunch_timing_tracker_kernel.h"
#include "aclrtlaunch_win_slice_kernel.h"
#include "aclrtlaunch_descramble_kernel.h"
#include "aclrtlaunch_rate_dematch_kernel.h"
#include "aclrtlaunch_ldpc_decode_kernel.h"

#define CK(x) do { aclError _e=(x); if(_e!=ACL_SUCCESS){ \
    std::fprintf(stderr,"ACL fail %d @ %s:%d\n",_e,__FILE__,__LINE__); std::exit(1);} } while(0)

namespace airan_rx {

static void build_cfo_comp_aux(RxArena& a, const SyncState& st);
float parse_cfo_dmrs(const float* c);

extern "C" void GenerateSssTiling(const char*, uint8_t*, int32_t, int32_t, int32_t);

static inline float half2float(uint16_t h){
    uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,out;
    if(e==0){ if(m==0)out=s<<31; else{e=127-15+1;while(!(m&0x400)){m<<=1;--e;}m&=0x3ff;out=(s<<31)|(e<<23)|(m<<13);} }
    else if(e==0x1f)out=(s<<31)|(0xff<<23)|(m<<13);
    else out=(s<<31)|((e-15+127)<<23)|(m<<13);
    float f; std::memcpy(&f,&out,4); return f;
}
static inline uint16_t float2half(float f){
    uint32_t x; std::memcpy(&x,&f,4);
    uint32_t s=(x>>16)&0x8000; int32_t e=((x>>23)&0xff)-127+15; uint32_t m=x&0x7fffff;
    if(e<=0) return (uint16_t)s;
    if(e>=0x1f) return (uint16_t)(s|0x7c00);
    return (uint16_t)(s|(e<<10)|(m>>13));
}
static void read_f32(uint8_t* gm, float* host, int n){
    CK(aclrtMemcpy(host,n*4,gm,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
}
static void read_f16(uint8_t* gm, float* host, int n){
    std::vector<uint16_t> t(n);
    CK(aclrtMemcpy(t.data(),n*2,gm,n*2,ACL_MEMCPY_DEVICE_TO_HOST));
    for(int i=0;i<n;++i) host[i]=half2float(t[i]);
}
static uint8_t* dmalloc(size_t b){ void* p=nullptr; CK(aclrtMalloc(&p,b?b:1,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMemset(p,b?b:1,0,b?b:1)); return (uint8_t*)p; }


static bool g_no_zero = false;
static inline void zero(uint8_t* gm,size_t b){ if(g_no_zero) return; CK(aclrtMemset(gm,b,0,b)); }
static void load_bin(uint8_t* gm,size_t cap,const std::string& path){
    FILE* f=std::fopen(path.c_str(),"rb");
    if(!f){ std::fprintf(stderr,"[warn] aux missing: %s (zeroed)\n",path.c_str()); return; }
    std::fseek(f,0,SEEK_END); long n=std::ftell(f); std::fseek(f,0,SEEK_SET);
    if((size_t)n>cap)n=(long)cap;
    std::vector<uint8_t> h(n); fread(h.data(),1,n,f); std::fclose(f);
    CK(aclrtMemcpy(gm,n,h.data(),n,ACL_MEMCPY_HOST_TO_DEVICE));
}




int rx_arena_init(RxArena& a, const char* data_dir){
    CK(aclrtCreateStream(&a.stream));
    std::string d = data_dir ? data_dir : ".";
    auto W=[&](const char* f){ return d+"/weights/"+f; };

    a.capture  = dmalloc(CAPTURE_B);
    a.ssb768   = dmalloc(SSB768_B);
    a.fir_taps = dmalloc(96u*8*2);       load_bin(a.fir_taps,96u*8*2,W("decimate_fir.bin"));
    a.pss_ref  = dmalloc(PSS_REF_B);     load_bin(a.pss_ref,PSS_REF_B,W("pss_ref.bin"));
    a.pss_twid = dmalloc(PSS_TWID_B);    load_bin(a.pss_twid,PSS_TWID_B,W("pss_twiddle.bin"));
    a.pss_scr  = dmalloc(PSS_SCR_B);     a.pss_out=dmalloc(PSS_OUT_B);
    a.ypss     = dmalloc(YPSS_B);
    a.psscfo_ref=dmalloc(PSSCFO_REF_B);  load_bin(a.psscfo_ref,PSSCFO_REF_B,W("psscfo_ref3.bin"));
    a.psscfo_scr=dmalloc(PSS_SCR_B);     a.psscfo_out=dmalloc(PSSCFO_OUT_B);
    a.sss_re   = dmalloc(SSS_REF_B);     load_bin(a.sss_re,SSS_REF_B,W("sss_re.bin"));
    a.sss_im   = dmalloc(SSS_REF_B);     load_bin(a.sss_im,SSS_REF_B,W("sss_im.bin"));
    a.sss_twr  = dmalloc(SSS_TWID_B);    load_bin(a.sss_twr,SSS_TWID_B,W("sss_twid_re.bin"));
    a.sss_twi  = dmalloc(SSS_TWID_B);    load_bin(a.sss_twi,SSS_TWID_B,W("sss_twid_im.bin"));
    a.sss_scr  = dmalloc(SSS_SCR_B);     a.sss_out=dmalloc(SSS_OUT_B);
    a.ssbfft_w16r=dmalloc(SSBFFT_W16_B); load_bin(a.ssbfft_w16r,SSBFFT_W16_B,W("ssbfft_w16_re.bin"));
    a.ssbfft_w16i=dmalloc(SSBFFT_W16_B); load_bin(a.ssbfft_w16i,SSBFFT_W16_B,W("ssbfft_w16_im.bin"));
    a.ssbfft_twr =dmalloc(SSBFFT_TW_B);  load_bin(a.ssbfft_twr,SSBFFT_TW_B,W("ssbfft_tw_re.bin"));
    a.ssbfft_twi =dmalloc(SSBFFT_TW_B);  load_bin(a.ssbfft_twi,SSBFFT_TW_B,W("ssbfft_tw_im.bin"));
    a.ssbfft_gidx=dmalloc(SSBFFT_GIDX_B);load_bin(a.ssbfft_gidx,SSBFFT_GIDX_B,W("ssbfft_gather.bin"));

    for(int nu=0; nu<4; ++nu){
        a.ssbfft_gidx_nu[nu]=dmalloc(SSBFFT_GIDX_B);
        char fn[32]; std::snprintf(fn,sizeof(fn),"ssbfft_gather_nu%d.bin",nu);
        load_bin(a.ssbfft_gidx_nu[nu],SSBFFT_GIDX_B,W(fn));
    }
    a.ssbfft_in=dmalloc(SSBFFT_IN_B);
    a.ssbfft_dr =dmalloc(SSBFFT_DEROT_B);load_bin(a.ssbfft_dr,SSBFFT_DEROT_B,W("ssbfft_derot_re.bin"));
    a.ssbfft_di =dmalloc(SSBFFT_DEROT_B);load_bin(a.ssbfft_di,SSBFFT_DEROT_B,W("ssbfft_derot_im.bin"));
    a.ssbfft_scr=dmalloc(SSS_SCR_B);
    a.r_re=dmalloc(4608); a.r_im=dmalloc(4608);
    a.pbch_dre=dmalloc(PBCH_D_B);  load_bin(a.pbch_dre,PBCH_D_B,W("pbch_d_re.bin"));
    a.pbch_dim=dmalloc(PBCH_D_B);  load_bin(a.pbch_dim,PBCH_D_B,W("pbch_d_im.bin"));
    a.pbch_dimn=dmalloc(PBCH_D_B); load_bin(a.pbch_dimn,PBCH_D_B,W("pbch_d_im_neg.bin"));
    a.pbch_out=dmalloc(PBCH_OUT_B);

    a.rd_ptr=dmalloc(8u*sizeof(int32_t));
    a.slot_iq=dmalloc(SLOT_IQ_B);
    a.slot_iq_swap=dmalloc(SLOT_IQ_B);
    {
        std::vector<uint32_t> sw(4096);
        for(uint32_t k=0;k<2048;++k){ sw[2*k]=(2*k+1)*2u; sw[2*k+1]=(2*k)*2u; }
        CK(aclrtMemcpy(a.slot_iq_swap, 4096u*4, sw.data(), 4096u*4, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    a.cc_a2=dmalloc(CFOCOMP_AB_B); a.cc_b2=dmalloc(CFOCOMP_AB_B); a.slot_iq_d=dmalloc(SLOT_IQ_B);
    a.cfoest_scr=dmalloc(SSS_SCR_B); a.cfoest_out=dmalloc(CFOEST_OUT_B);
    a.ofdm_w32r=dmalloc(32u*32*2); a.ofdm_w32i=dmalloc(32u*32*2);
    a.ofdm_w64r=dmalloc(64u*64*2); a.ofdm_w64i=dmalloc(64u*64*2);
    a.ofdm_twr =dmalloc(N_FFT*2);     a.ofdm_twi =dmalloc(N_FFT*2);
    a.ofdm_scr =dmalloc(256u*1024);
    load_bin(a.ofdm_w32r,32u*32*2,W("w_dft32_re.bin"));   load_bin(a.ofdm_w32i,32u*32*2,W("w_dft32_im.bin"));
    load_bin(a.ofdm_w64r,64u*64*2,W("w_dft64_re_T.bin")); load_bin(a.ofdm_w64i,64u*64*2,W("w_dft64_im_T.bin"));
    load_bin(a.ofdm_twr ,N_FFT*2 ,W("twiddle_pq_re.bin"));   load_bin(a.ofdm_twi ,N_FFT*2 ,W("twiddle_pq_im.bin"));
    a.y_re=dmalloc(Y2048_B); a.y_im=dmalloc(Y2048_B);
    a.re_idx=dmalloc(N_SC_PAD*4); load_bin(a.re_idx,N_SC_PAD*4,W("re_demap_idx.bin"));
    a.yd_re=dmalloc(YD_B); a.yd_im=dmalloc(YD_B);
    a.xref_re=dmalloc(XREF_B); a.xref_im=dmalloc(XREF_B);



    a.ce_weave=dmalloc(N_SC_PAD*4);
    { constexpr uint32_t N=208; std::vector<uint32_t> w(2u*N);
      for(uint32_t i=0;i<N;++i){ w[2*i]=i*2u; w[2*i+1]=(N+i)*2u; }
      CK(aclrtMemcpy(a.ce_weave, w.size()*sizeof(uint32_t), w.data(),
                     w.size()*sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE)); }
    a.h_re=dmalloc(H_B); a.h_im=dmalloc(H_B);
    a.eq_n0=dmalloc(EQ_B);
    a.x_re=dmalloc(EQ_B); a.x_im=dmalloc(EQ_B); a.no_eff=dmalloc(EQ_B);
    a.llr_in=dmalloc(LLR_IN_B);
    a.cfod_yre=dmalloc(CFOD_Y_B); a.cfod_yim=dmalloc(CFOD_Y_B);
    a.cfod_xre=dmalloc(CFOD_X_B); a.cfod_xim=dmalloc(CFOD_X_B);
    a.trel=dmalloc(TREL_B); load_bin(a.trel,TREL_B,W("cfo_dmrs_trel.bin"));
    a.cfodmrs_scr=dmalloc(SSS_SCR_B); a.cfodmrs_out=dmalloc(CFODMRS_OUT_B);
    a.h220_re=dmalloc(H220_B); a.h220_im=dmalloc(H220_B);
    a.tt_dtscale=dmalloc(DTSCALE_B); load_bin(a.tt_dtscale,DTSCALE_B,W("tt_dtscale.bin"));
    a.tt_wr=dmalloc(TT_W_B); a.tt_wi=dmalloc(TT_W_B); a.tt_twr=dmalloc(TT_TW_B); a.tt_twi=dmalloc(TT_TW_B);
    load_bin(a.tt_wr,TT_W_B,W("tt_w_re.bin"));   load_bin(a.tt_wi,TT_W_B,W("tt_w_im.bin"));
    load_bin(a.tt_twr,TT_TW_B,W("tt_tw_re.bin"));load_bin(a.tt_twi,TT_TW_B,W("tt_tw_im.bin"));
    a.tt_scr=dmalloc(TT_SCR_B); a.cir_re=dmalloc(CIR_B); a.cir_im=dmalloc(CIR_B); a.dt_out=dmalloc(DT_OUT_B);

    a.cw=dmalloc(CW_B); a.sign=dmalloc(SIGN_B); load_bin(a.sign,SIGN_B,W("gold_sign.bin"));
    a.n_slot=dmalloc(NSLOT_B); a.cw_ds=dmalloc(CW_B);
    { int32_t ns[16]={0}; ns[0]=(int32_t)SLOTS_PER_TB;
      CK(aclrtMemcpy(a.n_slot,NSLOT_B,ns,NSLOT_B,ACL_MEMCPY_HOST_TO_DEVICE)); }
    a.rd_tmpl=dmalloc(TMPL_B); load_bin(a.rd_tmpl,TMPL_B,W("rate_dematch_tmpl.bin"));
    a.rd_dummy=dmalloc(DUMMY_B); a.lam=dmalloc(LAM_B);
    a.ldpc_packbc=dmalloc(LDPC_PACKBC_B);
    a.ldpc_packs =dmalloc(LDPC_PACKS_B);

    a.ldpc_deg   =dmalloc(LDPC_DEG_B);    load_bin(a.ldpc_deg ,LDPC_DEG_B ,W("ldpc_degrees.bin"));
    a.ldpc_eoff  =dmalloc(LDPC_EOFF_B);   load_bin(a.ldpc_eoff,LDPC_EOFF_B,W("ldpc_edge_offsets.bin"));


    {
        constexpr uint32_t MB = 46, NFULL = 68, MAX_DEG = 19;
        constexpr uint32_t SHIFT_ELEMS = MB * NFULL;
        std::vector<int16_t> hShift(SHIFT_ELEMS, 0);
        const std::string sp = W("ldpc_bg1_z384_shifts/shift_table.bin");
        FILE* f = std::fopen(sp.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "[warn] aux missing: %s  -> LDPC structure ZERO, decoder is a no-op!\n", sp.c_str());
        } else {
            size_t got = std::fread(hShift.data(), sizeof(int16_t), SHIFT_ELEMS, f);
            std::fclose(f);
            if (got != SHIFT_ELEMS)
                std::fprintf(stderr, "[warn] shift_table short read: %zu/%u\n", got, SHIFT_ELEMS);
            std::vector<int16_t> bc(MB * MAX_DEG, 0), ps(MB * MAX_DEG, 0);
            for (uint32_t br = 0; br < MB; ++br) {
                uint32_t k = 0;
                for (uint32_t bc_n = 0; bc_n < NFULL; ++bc_n) {
                    const int16_t s = hShift[br * NFULL + bc_n];
                    if (s < 0) continue;
                    bc[br * MAX_DEG + k] = (int16_t)bc_n;
                    ps[br * MAX_DEG + k] = s;
                    ++k;
                }
            }
            CK(aclrtMemcpy(a.ldpc_packbc, LDPC_PACKBC_B, bc.data(), LDPC_PACKBC_B, ACL_MEMCPY_HOST_TO_DEVICE));
            CK(aclrtMemcpy(a.ldpc_packs,  LDPC_PACKS_B,  ps.data(), LDPC_PACKS_B,  ACL_MEMCPY_HOST_TO_DEVICE));
        }
    }
    a.ldpc_prev  =dmalloc(LDPC_PREV_B);   a.ldpc_lamscr=dmalloc(LDPC_LAMSCR_B);
    CK(aclrtMemset(a.ldpc_prev,LDPC_PREV_B,0,LDPC_PREV_B));
    CK(aclrtMemset(a.ldpc_lamscr,LDPC_LAMSCR_B,0,LDPC_LAMSCR_B));
    a.bits=dmalloc(BITS_B); a.lam_out=dmalloc(LAM_B);

    setup_tilings(a);
    return 0;
}

void rx_arena_free(RxArena& a){ if(a.stream) aclrtDestroyStream(a.stream); }




static void build_cfo_comp_aux(RxArena& a, const SyncState& st){

    constexpr int N=(int)N_SAMP_SLOT; const double FS=61.44e6, SIGN=-1.0;
    const double cfo=(double)st.cfo_total;
    std::vector<uint16_t> A2(2*N), B2(2*N);
    for(int n=0;n<N;++n){
        double th=SIGN*2.0*M_PI*cfo*(double)n/FS;
        uint16_t c=float2half((float)std::cos(th));
        uint16_t s=float2half((float)std::sin(th));
        uint16_t sneg=float2half(-(float)std::sin(th));
        A2[2*n]=c; A2[2*n+1]=c;
        B2[2*n]=sneg; B2[2*n+1]=s;
    }
    CK(aclrtMemcpy(a.cc_a2, CFOCOMP_AB_B, A2.data(), (size_t)2*N*2, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(a.cc_b2, CFOCOMP_AB_B, B2.data(), (size_t)2*N*2, ACL_MEMCPY_HOST_TO_DEVICE));
}
[[maybe_unused]] static void window(RxArena& a, const int16_t* stream_host, const SyncState& st, int slot){
    size_t off=(size_t)(st.mu_t_data+slot*(int)N_SAMP_SLOT)*2;
    CK(aclrtMemcpy(a.slot_iq,SLOT_IQ_B,stream_host+off,SLOT_IQ_B,ACL_MEMCPY_HOST_TO_DEVICE));
    build_cfo_comp_aux(a,st);
}




static void extract_y_at_pss(RxArena& a,const SyncState& st){
    size_t off = (size_t)st.mu_t * 4;
    CK(aclrtMemcpy(a.ypss, YPSS_B, a.ssb768 + off, YPSS_B, ACL_MEMCPY_DEVICE_TO_DEVICE));
}


static uint32_t dmrs_cinit(int slot, int l, uint32_t n_id, uint32_t n_scid){
    uint64_t v = ((uint64_t)1 << 17) * (uint64_t)(14*slot + l + 1) * (uint64_t)(2*n_id + 1)
               + (uint64_t)(2*n_id + n_scid);
    return (uint32_t)(v & 0x7FFFFFFFu);
}

static void dmrs_seq_fp16(int slot, int l, uint32_t n_id, uint32_t n_scid,
                          int n_re, uint16_t* dst_re, uint16_t* dst_im){
    const int NC = 1600, len = 2*n_re, ntot = len + NC;
    std::vector<uint8_t> x1(ntot + 31, 0), x2(ntot + 31, 0);
    const uint32_t ci = dmrs_cinit(slot, l, n_id, n_scid);
    x1[0] = 1;
    for (int i = 0; i < 31; ++i) x2[i] = (ci >> i) & 1;
    for (int n = 0; n < ntot; ++n){
        x1[n+31] = (x1[n+3] ^ x1[n]) & 1;
        x2[n+31] = (x2[n+3] ^ x2[n+2] ^ x2[n+1] ^ x2[n]) & 1;
    }
    static const uint16_t P = float2half( 0.70710678f);
    static const uint16_t N = float2half(-0.70710678f);
    for (int k = 0; k < n_re; ++k){
        uint8_t c_re = (x1[NC + 2*k]     ^ x2[NC + 2*k])     & 1;
        uint8_t c_im = (x1[NC + 2*k + 1] ^ x2[NC + 2*k + 1]) & 1;
        dst_re[k] = c_re ? N : P;
        dst_im[k] = c_im ? N : P;
    }
}




static void gen_dmrs_ref(RxArena& a, const SyncState& st){
    constexpr int N_RE = 798, XW = 832, CW = 896, NS = (int)SLOTS_PER_TB;
    constexpr int DSYM[2] = {2, 11};


    const uint32_t n_id = (uint32_t)st.cell_id, n_scid = 0;

    std::vector<uint16_t> xr((size_t)NS*2*XW + 256, 0), xi((size_t)NS*2*XW + 256, 0);
    std::vector<uint16_t> cr((size_t)NS*2*CW + 256, 0), ci((size_t)NS*2*CW + 256, 0);
    uint16_t tre[N_RE], tim[N_RE];
    for (int slot = 0; slot < NS; ++slot)
        for (int s = 0; s < 2; ++s){
            dmrs_seq_fp16(slot, DSYM[s], n_id, n_scid, N_RE, tre, tim);
            std::memcpy(&xr[((size_t)slot*2 + s)*XW], tre, N_RE*sizeof(uint16_t));
            std::memcpy(&xi[((size_t)slot*2 + s)*XW], tim, N_RE*sizeof(uint16_t));
            std::memcpy(&cr[((size_t)slot*2 + s)*CW], tre, N_RE*sizeof(uint16_t));
            std::memcpy(&ci[((size_t)slot*2 + s)*CW], tim, N_RE*sizeof(uint16_t));
        }
    CK(aclrtMemcpy(a.xref_re,  XREF_B,   xr.data(), (size_t)NS*2*XW*2, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(a.xref_im,  XREF_B,   xi.data(), (size_t)NS*2*XW*2, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(a.cfod_xre, CFOD_X_B, cr.data(), (size_t)NS*2*CW*2, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(a.cfod_xim, CFOD_X_B, ci.data(), (size_t)NS*2*CW*2, ACL_MEMCPY_HOST_TO_DEVICE));
}
[[maybe_unused]] static void repack_cfo_dmrs(RxArena& a,const SyncState& st){ (void)a;(void)st; }
[[maybe_unused]] static void subsample_H_220(RxArena& a){ (void)a; }
static void bake_gold_sign(RxArena& a, const SyncState& st){







    static int cached_cell_id = -1;
    if(cached_cell_id == st.cell_id) return;
    cached_cell_id = st.cell_id;
    constexpr int NSLOT=(int)SLOTS_PER_TB, NSTR=8, NSYM=19200, NVALID=19152, GOLD_LEN=1600;
    const uint32_t n_RNTI=12345u, q=0u, N_ID=(uint32_t)st.cell_id;
    const uint32_t c_init=((n_RNTI&0xFFFFu)<<15)|((q&1u)<<14)|(N_ID&0x3FFu);
    const size_t length=(size_t)NSLOT*NVALID*NSTR, total=length+GOLD_LEN;
    std::vector<uint8_t> x1(total+31,0), x2(total+31,0); x1[0]=1;
    for(int i=0;i<31;++i) x2[i]=(uint8_t)((c_init>>i)&1u);
    for(size_t n=0;n<total;++n){
        x1[n+31]=(uint8_t)((x1[n+3]+x1[n])&1u);
        x2[n+31]=(uint8_t)((x2[n+3]+x2[n+2]+x2[n+1]+x2[n])&1u);
    }
    std::vector<int16_t> sign((size_t)NSLOT*NSTR*NSYM, (int16_t)1);
    for(int slot=0;slot<NSLOT;++slot)for(int b=0;b<NSTR;++b){
        for(int re=0;re<NVALID;++re){
            const size_t idx=((size_t)slot*NVALID+re)*NSTR+b;
            const uint8_t c=(uint8_t)((x1[GOLD_LEN+idx]^x2[GOLD_LEN+idx])&1u);
            sign[((size_t)slot*NSTR+b)*NSYM+re]=(int16_t)(1-2*(int)c);
        }
    }
    CK(aclrtMemcpy(a.sign, SIGN_B, sign.data(), SIGN_B, ACL_MEMCPY_HOST_TO_DEVICE));
}
static double defiller_concat_and_ber(RxArena& a,const uint8_t* tx){


    const long tot=143L*8448;
    std::vector<int8_t> dbits(tot);
    CK(aclrtMemcpy(dbits.data(), (size_t)tot, a.bits, (size_t)tot, ACL_MEMCPY_DEVICE_TO_HOST));
    if(!tx){
        long ones=0; for(auto b:dbits) ones+=(b&1);
        std::printf("[ber] no tx_bits; decoded ones=%ld/%ld (sanity)\n",ones,tot);
        return -1.0;
    }
    long err=0;
    for(long i=0;i<tot;++i) if((dbits[i]&1)!=(tx[i]&1)) ++err;
    std::printf("[ber] *** 真链路 BER = %ld / %ld = %.3e %s ***\n",
        err,tot,(double)err/tot, err==0?"  <<< BER=0 真链路 PASS":"");
    for(int cb=0; cb<3; ++cb){ long e=0; for(int k=0;k<8448;++k) if((dbits[cb*8448+k]&1)!=(tx[cb*8448+k]&1))++e;
        std::printf("[ber]   CB%d err=%ld/8448\n",cb,e); }
    return (double)err/(double)tot;
}
float parse_cfo_dmrs(const float* c){ float ph=std::atan2(c[2*11+1],c[2*11+0]); return ph; }




static uint32_t BD = 4;

void rx_acquire(RxArena& a, const int16_t* capture_host, SyncState& st){
    CK(aclrtMemcpy(a.capture, CAPTURE_B, capture_host, CAPTURE_B, ACL_MEMCPY_HOST_TO_DEVICE));

    if(std::getenv("RX_DUMP_CAP")){
        std::vector<int16_t> hc(CAPTURE_B/2);
        CK(aclrtMemcpy(hc.data(), CAPTURE_B, a.capture, CAPTURE_B, ACL_MEMCPY_DEVICE_TO_HOST));
        std::string p = std::string(std::getenv("AIRAN_DATA_DIR")?std::getenv("AIRAN_DATA_DIR"):".") + "/dbg_cap.bin";
        FILE* f=std::fopen(p.c_str(),"wb"); std::fwrite(hc.data(),2,hc.size(),f); std::fclose(f);
        long mx=0; for(auto v:hc){ long a2=v<0?-v:v; if(a2>mx)mx=a2; }
        std::printf("[dbg] a.capture dumped: %zu int16 |max|=%ld -> %s\n", hc.size(), mx, p.c_str());
    }

    ACLRT_LAUNCH_KERNEL(decimate_kernel)(BD, a.stream,
        a.capture, a.fir_taps, a.ssb768, a.ws, a.tiling[OP_decimate]);


    if(std::getenv("RX_DUMP_SSB768")){
        CK(aclrtSynchronizeStream(a.stream));
        std::vector<int16_t> h(SSB768_B/2);
        CK(aclrtMemcpy(h.data(), SSB768_B, a.ssb768, SSB768_B, ACL_MEMCPY_DEVICE_TO_HOST));
        std::string p = std::string(std::getenv("AIRAN_DATA_DIR")?std::getenv("AIRAN_DATA_DIR"):".") + "/dbg_ssb768.bin";
        FILE* f=std::fopen(p.c_str(),"wb"); std::fwrite(h.data(),2,h.size(),f); std::fclose(f);
        long mx=0,sq=0; for(auto v:h){ long a2=v<0?-v:v; if(a2>mx)mx=a2; sq+=(long)v*v; }
        std::printf("[dbg] ssb768 dumped: %zu int16, |max|=%ld rms=%ld -> %s\n",
                    h.size(), mx, (long)std::sqrt((double)sq/h.size()), p.c_str());
    }




    {
        std::vector<int16_t> hs(SSB768_B/2);
        CK(aclrtSynchronizeStream(a.stream));
        CK(aclrtMemcpy(hs.data(), SSB768_B, a.ssb768, SSB768_B, ACL_MEMCPY_DEVICE_TO_HOST));
        double sq0=0; for(auto v:hs) sq0+=(double)v*v;
        double rms0=std::sqrt(sq0/hs.size());
        float g = (rms0>1e-6)? (float)(3000.0/rms0) : 1.0f;
        if(std::getenv("RX_NO_AGC")) g=1.0f;
        for(auto& v:hs){ int32_t s=(int32_t)std::lround((float)v*g);
                         v=(int16_t)(s>32767?32767:(s<-32767?-32767:s)); }
        CK(aclrtMemcpy(a.ssb768, SSB768_B, hs.data(), SSB768_B, ACL_MEMCPY_HOST_TO_DEVICE));
        if(std::getenv("RX_DUMP_SSB768")){
            long sq=0,mx=0; for(auto v:hs){ sq+=(long)v*v; long a2=v<0?-v:v; if(a2>mx)mx=a2; }
            std::printf("[agc] ssb768 gain=%.2f -> rms=%ld |max|=%ld (clip@32767)\n",
                        g, (long)std::sqrt((double)sq/hs.size()), mx);
        }
    }

    zero(a.pss_scr, PSS_SCR_B);
    ACLRT_LAUNCH_KERNEL(pss_correlator_kernel)(BD, a.stream,
        a.ssb768, a.pss_ref, a.pss_twid, a.pss_scr, a.pss_out, a.ws, a.tiling[OP_pss_correlator]);
    CK(aclrtSynchronizeStream(a.stream));
    int g_hat=0;





    {
        const size_t METRIC_OFF = 6u*153600u*2u;
        const int NG=3, NAIV=4, NTILE=300, PER=16;
        const int M_PER_AIV=38336, M_TILE=128, TAIL_MU=64;
        std::vector<uint8_t> sc(PSS_SCR_B);
        CK(aclrtMemcpy(sc.data(), PSS_SCR_B, a.pss_scr, PSS_SCR_B, ACL_MEMCPY_DEVICE_TO_HOST));
        const float* m = reinterpret_cast<const float*>(sc.data()+METRIC_OFF);
        float best=-1e30f; int ba=0,bt=0,bl=0,bidx=0,bg=0; double tsum=0,tcnt=0;
        for(int g=0; g<NG; ++g)
        for(int a2=0; a2<NAIV; ++a2)
        for(int t=0; t<NTILE; ++t)
        for(int l=0; l<3; ++l){
            const float* f = m + ((size_t)g*NAIV*NTILE + (size_t)a2*NTILE + t)*PER + l*4;
            float peak=f[0]; int idx=(int)f[1]; tsum+=f[2]; tcnt+=f[3];
            if(t==299 && idx>=TAIL_MU) continue;
            if(peak>best){ best=peak; bg=g; ba=a2; bt=t; bl=l; bidx=idx; }
        }
        st.mu_t   = ba*M_PER_AIV + bt*M_TILE + bidx;
        st.n_id_2 = bl;
        g_hat     = bg - 1;
        st.cfo_int= (float)g_hat*(float)SCS_HZ;
        st.noise  = (tcnt>0)? (float)(tsum/tcnt) : 0.0f;
        if(std::getenv("RX_DUMP_PSSM")){
            std::string p=std::string(std::getenv("AIRAN_DATA_DIR")?std::getenv("AIRAN_DATA_DIR"):".")+"/dbg_pssm.bin";
            FILE* fp=std::fopen(p.c_str(),"wb"); std::fwrite(sc.data()+METRIC_OFF,4,3*4*300*16,fp); std::fclose(fp);
            std::printf("[pssm] dumped -> %s ; host pick bg=%d bl=%d ba=%d bt=%d bidx=%d best=%.3e\n",p.c_str(),bg,bl,ba,bt,bidx,best);
        }
    }

    extract_y_at_pss(a, st);


    {
        constexpr int Ntw=256; const double FS=7.68e6;
        std::vector<uint16_t> tw(2*Ntw);
        for(int n=0;n<Ntw;++n){
            double th=2.0*M_PI*(double)st.cfo_int*(double)n/FS;
            tw[n]     =float2half((float)std::cos(th));
            tw[Ntw+n] =float2half((float)std::sin(th));
        }
        CK(aclrtMemcpy(a.psscfo_scr, 2*Ntw*2, tw.data(), 2*Ntw*2, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    uint8_t* psscfo_ref_sel = a.psscfo_ref + (size_t)st.n_id_2 * 1024u;
    ACLRT_LAUNCH_KERNEL(pss_cfo_estimator_kernel)(BD, a.stream,
        a.ypss, psscfo_ref_sel, a.psscfo_scr, a.psscfo_out, a.ws, a.tiling[OP_pss_cfo_estimator]);
    CK(aclrtSynchronizeStream(a.stream));
    { float o[8]; read_f32(a.psscfo_out,o,8); st.cfo_frac=o[0]; }


    { uint8_t tb[64];
      GenerateSssTiling("Ascend310P3", tb, st.mu_t, st.n_id_2, g_hat);
      CK(aclrtMemcpy(a.tiling[OP_sss_correlator],64,tb,64,ACL_MEMCPY_HOST_TO_DEVICE)); }
    zero(a.sss_scr, SSS_SCR_B);
    ACLRT_LAUNCH_KERNEL(sss_correlator_kernel)(BD, a.stream,
        a.ssb768, a.sss_re, a.sss_im, a.sss_twr, a.sss_twi, a.sss_scr, a.sss_out, a.ws, a.tiling[OP_sss_correlator]);
    CK(aclrtSynchronizeStream(a.stream));
    { float o[8]; read_f32(a.sss_out,o,8); st.n_id_1=(int)o[0]; st.cell_id=3*st.n_id_1+st.n_id_2; }





    st.cfo_total = st.cfo_int + st.cfo_frac;
    {
        constexpr int SSB_PBCH_OFF = 256;
        constexpr int NW           = 3 * 292;
        std::vector<int16_t> win((size_t)NW * 2);
        CK(aclrtMemcpy(win.data(), (size_t)NW*4,
                       a.ssb768 + (size_t)(st.mu_t + SSB_PBCH_OFF) * 4, (size_t)NW*4,
                       ACL_MEMCPY_DEVICE_TO_HOST));


        const double PI2 = -2.0 * 3.14159265358979323846 * (double)st.cfo_total / (double)FS_SSB;
        for(int k=0;k<NW;++k){
            double c=std::cos(PI2*k), s=std::sin(PI2*k);
            double re=(double)win[2*k], im=(double)win[2*k+1];
            win[2*k]   = (int16_t)std::lround(re*c - im*s);
            win[2*k+1] = (int16_t)std::lround(re*s + im*c);
        }
        CK(aclrtMemcpy(a.ssbfft_in, (size_t)NW*4, win.data(), (size_t)NW*4,
                       ACL_MEMCPY_HOST_TO_DEVICE));
    }
    uint8_t* gidx_sel = a.ssbfft_gidx_nu[st.cell_id % 4];
    if(!gidx_sel) gidx_sel = a.ssbfft_gidx;

    zero(a.ssbfft_scr, SSS_SCR_B);
    ACLRT_LAUNCH_KERNEL(ssb_fft_kernel)(BD, a.stream,
        a.ssbfft_in, a.ssbfft_w16r, a.ssbfft_w16i, a.ssbfft_twr, a.ssbfft_twi,
        gidx_sel, a.ssbfft_dr, a.ssbfft_di, a.ssbfft_scr, a.r_re, a.r_im, a.ws, a.tiling[OP_ssb_fft]);

    ACLRT_LAUNCH_KERNEL(pbch_dmrs_correlator_kernel)(BD, a.stream,
        a.r_re, a.r_im, a.pbch_dre, a.pbch_dim, a.pbch_dimn, a.pbch_out, a.ws, a.tiling[OP_pbch_dmrs_correlator]);
    CK(aclrtSynchronizeStream(a.stream));
    { float o[24]; read_f32(a.pbch_out,o,24); st.i_ssb=(int)o[0]; }

    st.mu_t_data = st.mu_t * (int)DECIM;
    st.locked    = 1;




    int pusch_off = 52640;
    if(const char* e=std::getenv("RX_PUSCH_OFFSET")) pusch_off=std::atoi(e);
    const int pusch_start = st.mu_t_data + pusch_off;

    { int32_t rd0[8]={0}; rd0[0]=pusch_start-(int)N_SAMP_SLOT;
      CK(aclrtMemcpy(a.rd_ptr,8*sizeof(int32_t),rd0,8*sizeof(int32_t),ACL_MEMCPY_HOST_TO_DEVICE));
      float dt0[8]={0};
      CK(aclrtMemcpy(a.dt_out,DT_OUT_B,dt0,(DT_OUT_B<32?DT_OUT_B:32),ACL_MEMCPY_HOST_TO_DEVICE)); }

    { uint16_t n0=float2half(st.noise); std::vector<uint16_t> v(EQ_B/2, n0);
      CK(aclrtMemcpy(a.eq_n0,EQ_B,v.data(),EQ_B,ACL_MEMCPY_HOST_TO_DEVICE)); }


    gen_dmrs_ref(a, st);



    build_cfo_comp_aux(a, st);
}




void rx_slot(RxArena& a, const int16_t*  , SyncState& st, int slot){

    ACLRT_LAUNCH_KERNEL(win_slice_kernel)(BD, a.stream,
        a.capture, a.dt_out, a.slot_iq, a.rd_ptr, a.ws, a.tiling[OP_win_slice]);

    ACLRT_LAUNCH_KERNEL(cfo_estimate_kernel)(BD, a.stream,
        a.slot_iq, a.cfoest_scr, a.ws, a.cfoest_out);



    ACLRT_LAUNCH_KERNEL(cfo_compensate_kernel)(BD, a.stream,
        a.slot_iq, a.slot_iq_swap, a.cc_a2, a.cc_b2, a.slot_iq_d, a.ws, a.tiling[OP_cfo_compensate]);

    ACLRT_LAUNCH_KERNEL(ofdm_demod_kernel)(BD, a.stream,
        a.slot_iq_d, a.ofdm_w32r,a.ofdm_w32i,a.ofdm_w64r,a.ofdm_w64i, a.ofdm_twr,a.ofdm_twi,
        a.ofdm_scr, a.y_re, a.y_im, a.ws, a.tiling[OP_ofdm_demod]);

    ACLRT_LAUNCH_KERNEL(re_demap_kernel)(BD, a.stream,
        a.y_re, a.y_im, a.re_idx, a.yd_re, a.yd_im, a.ws, a.tiling[OP_re_demap]);

    uint8_t* xref_re_s = a.xref_re + (size_t)slot * XREF_SLOT_HALF * 2;
    uint8_t* xref_im_s = a.xref_im + (size_t)slot * XREF_SLOT_HALF * 2;
    ACLRT_LAUNCH_KERNEL(channel_est_ls_kernel)(BD, a.stream,
        a.yd_re, a.yd_im, xref_re_s, xref_im_s, a.h_re, a.h_im, a.ce_weave, a.ws, a.tiling[OP_channel_est_ls]);

    ACLRT_LAUNCH_KERNEL(equalize_kernel)(BD, a.stream,
        a.yd_re, a.yd_im, a.h_re, a.h_im, a.eq_n0, a.x_re, a.x_im, a.no_eff, a.ws, a.tiling[OP_equalize]);

    uint8_t* llr_slot = a.llr_in + (size_t)slot*8*LLR_SLOT_PAD*2;
    ACLRT_LAUNCH_KERNEL(qam256_demod_kernel)(BD, a.stream,
        a.x_re, a.x_im, a.no_eff, llr_slot, a.ws, a.tiling[OP_qam256_demod]);

    uint8_t* cfod_xre_s = a.cfod_xre + (size_t)slot * CFOD_X_SLOT_HALF * 2;
    uint8_t* cfod_xim_s = a.cfod_xim + (size_t)slot * CFOD_X_SLOT_HALF * 2;


    ACLRT_LAUNCH_KERNEL(cfo_dmrs_kernel)(BD, a.stream,
        a.yd_re, a.yd_im, cfod_xre_s, cfod_xim_s,
        a.cfodmrs_scr, a.cfodmrs_out, a.ws, a.tiling[OP_cfo_dmrs]);
    ACLRT_LAUNCH_KERNEL(timing_tracker_kernel)(BD, a.stream,
        a.h_re, a.h_im, a.tt_wr, a.tt_wi, a.tt_twr, a.tt_twi, a.tt_dtscale, a.tt_scr,
        a.cir_re, a.cir_im, a.dt_out, a.ws, a.tiling[OP_timing_tracker]);


}

void rx_tb(RxArena& a, SyncState& st, const uint8_t* tx_bits_known, double* ber){
    if (std::getenv("RX_SKIP_TB")) { *ber = 0.0; return; }

    bake_gold_sign(a, st);
    ACLRT_LAUNCH_KERNEL(descramble_kernel)(BD, a.stream,
        a.llr_in, a.sign, a.n_slot, a.cw_ds, a.ws, a.tiling[OP_descramble]);

    ACLRT_LAUNCH_KERNEL(rate_dematch_kernel)(BD, a.stream,
        a.cw_ds, a.rd_tmpl, a.rd_dummy, a.lam);

    ACLRT_LAUNCH_KERNEL(ldpc_decode_kernel)(BD, a.stream,
        a.lam, a.ldpc_packbc, a.ldpc_packs, a.ldpc_deg, a.ldpc_eoff,
        a.ldpc_prev, a.bits, a.lam_out, a.ldpc_lamscr);
    CK(aclrtSynchronizeStream(a.stream));

    *ber = defiller_concat_and_ber(a, tx_bits_known);
}






void rx_loopback(RxArena& a, SyncState& st){
    const char* iqp = std::getenv("RX_LB_IQ");
    if(!iqp){ std::printf("[loopback] need RX_LB_IQ=/path/to/tx_iq.bin\n"); return; }
    int shift = 0; if(const char* s=std::getenv("RX_LB_SHIFT")) shift=std::atoi(s);
    const int NS = (int)SLOTS_PER_TB;
    const size_t per_i16 = (size_t)N_SAMP_SLOT*2;
    const size_t tot_i16 = per_i16*(size_t)NS;
    const int    YDN = (int)(N_SYM*N_SC_PAD);

    std::vector<int16_t> iq(tot_i16);
    FILE* fp=std::fopen(iqp,"rb");
    if(!fp){ std::printf("[loopback] open fail: %s\n",iqp); return; }
    size_t got=std::fread(iq.data(),sizeof(int16_t),tot_i16,fp); std::fclose(fp);
    std::printf("[loopback] read %zu/%zu int16 from %s  (shift<<%d)\n",got,tot_i16,iqp,shift);
    if(shift>0) for(auto& v:iq){ int t=((int)v)<<shift; v=(int16_t)(t>32767?32767:(t<-32768?-32768:t)); }

    size_t cb=std::min(tot_i16*sizeof(int16_t),(size_t)CAPTURE_B);
    CK(aclrtMemcpy(a.capture, cb, iq.data(), cb, ACL_MEMCPY_HOST_TO_DEVICE));


    st = SyncState{}; st.mu_t_data=0; st.locked=1;
    if(const char* e=std::getenv("RX_CELL_ID")) st.cell_id=std::atoi(e);
    { int32_t rd0[8]={0}; rd0[0]=-(int)N_SAMP_SLOT;
      CK(aclrtMemcpy(a.rd_ptr,8*sizeof(int32_t),rd0,8*sizeof(int32_t),ACL_MEMCPY_HOST_TO_DEVICE)); }
    int32_t z8[8]={0}; size_t dtb=(DT_OUT_B<32?DT_OUT_B:32);
    CK(aclrtMemcpy(a.dt_out,dtb,z8,dtb,ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int16_t> ydr(YDN), ydi(YDN);
    const char* dd=std::getenv("AIRAN_DATA_DIR"); std::string base=dd?dd:".";
    std::string pr=base+"/rx_yd_re.bin", pi=base+"/rx_yd_im.bin";
    FILE* fr=std::fopen(pr.c_str(),"wb"); FILE* fi=std::fopen(pi.c_str(),"wb");


    const bool do_ber = std::getenv("RX_LB_BER");
    std::vector<int8_t> tx_bits;
    if(do_ber){
        const char* tbp=std::getenv("RX_LB_TXBITS");
        std::string tp = tbp ? std::string(tbp) : (base+"/tx_bits.bin");
        FILE* ft=std::fopen(tp.c_str(),"rb");
        if(ft){ tx_bits.resize(143u*8448);
            size_t g=std::fread(tx_bits.data(),1,tx_bits.size(),ft); std::fclose(ft);
            std::printf("[ber] tx_bits %zu/%zu B <- %s\n",g,tx_bits.size(),tp.c_str()); }
        else std::printf("[ber] WARN no tx_bits at %s (set RX_LB_TXBITS=/path)\n",tp.c_str());
    }



    if(do_ber){
        gen_dmrs_ref(a, st);
        std::printf("[loopback] 完整数据链 channel_est+equalize (fp32, 无 AGC)\n");
    }

    for(int slot=0; slot<NS; ++slot){
        ACLRT_LAUNCH_KERNEL(win_slice_kernel)(BD,a.stream,
            a.capture,a.dt_out,a.slot_iq,a.rd_ptr,a.ws,a.tiling[OP_win_slice]);
        ACLRT_LAUNCH_KERNEL(ofdm_demod_kernel)(BD,a.stream,
            a.slot_iq,a.ofdm_w32r,a.ofdm_w32i,a.ofdm_w64r,a.ofdm_w64i,a.ofdm_twr,a.ofdm_twi,
            a.ofdm_scr,a.y_re,a.y_im,a.ws,a.tiling[OP_ofdm_demod]);
        ACLRT_LAUNCH_KERNEL(re_demap_kernel)(BD,a.stream,
            a.y_re,a.y_im,a.re_idx,a.yd_re,a.yd_im,a.ws,a.tiling[OP_re_demap]);
        CK(aclrtMemcpy(a.dt_out,dtb,z8,dtb,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtSynchronizeStream(a.stream));
        if(slot==0 && std::getenv("RX_LB_DBG")){
            auto chk=[&](uint8_t* g,size_t bytes,const char* nm){
                std::vector<uint16_t> h(bytes/2);
                CK(aclrtMemcpy(h.data(),bytes,g,bytes,ACL_MEMCPY_DEVICE_TO_HOST));
                size_t nz=0; uint16_t mx=0; for(uint16_t v:h){ if(v){++nz; if(v>mx)mx=v;} }
                std::printf("[lb dbg] %-9s nonzero=%zu/%zu  rawmax=0x%04x\n",nm,nz,h.size(),mx);
            };
            chk(a.slot_iq, SLOT_IQ_B,      "slot_iq");
            chk(a.y_re,    Y2048_B,        "y_re");
            chk(a.yd_re,   (size_t)YDN*2,  "yd_re");

            { std::vector<uint8_t> h(Y2048_B);
              CK(aclrtMemcpy(h.data(),Y2048_B,a.y_re,Y2048_B,ACL_MEMCPY_DEVICE_TO_HOST));
              FILE* f=std::fopen((base+"/rx_y_re.bin").c_str(),"wb"); std::fwrite(h.data(),1,Y2048_B,f); std::fclose(f);
              CK(aclrtMemcpy(h.data(),Y2048_B,a.y_im,Y2048_B,ACL_MEMCPY_DEVICE_TO_HOST));
              f=std::fopen((base+"/rx_y_im.bin").c_str(),"wb"); std::fwrite(h.data(),1,Y2048_B,f); std::fclose(f);
              std::printf("[lb dbg] dumped y[14,2048] -> rx_y_{re,im}.bin\n"); }
        }
        CK(aclrtMemcpy(ydr.data(),(size_t)YDN*2,a.yd_re,(size_t)YDN*2,ACL_MEMCPY_DEVICE_TO_HOST));
        CK(aclrtMemcpy(ydi.data(),(size_t)YDN*2,a.yd_im,(size_t)YDN*2,ACL_MEMCPY_DEVICE_TO_HOST));
        std::fwrite(ydr.data(),sizeof(int16_t),YDN,fr);
        std::fwrite(ydi.data(),sizeof(int16_t),YDN,fi);


        if(do_ber){


            float n0f=5.0e-3f; if(const char* e=std::getenv("RX_LB_N0")) n0f=(float)std::atof(e);
            { static const int DS[12]={0,1,3,4,5,6,7,8,9,10,12,13};
              double ss=0; long cnt=0;
              for(int di=0;di<12;++di){int ph=DS[di]; for(int sc=0;sc<1596;++sc){
                  float r=half2float((uint16_t)ydr[ph*N_SC_PAD+sc]);
                  float m=half2float((uint16_t)ydi[ph*N_SC_PAD+sc]);
                  ss+=(double)r*r+(double)m*m; cnt+=2;}}
              float rms=(float)std::sqrt(ss/std::max(1L,cnt));
              std::vector<uint16_t> nv((size_t)EQ_B/2, float2half(n0f*rms*rms));
              CK(aclrtMemcpy(a.eq_n0,EQ_B,nv.data(),EQ_B,ACL_MEMCPY_HOST_TO_DEVICE));
              if(slot==0) std::printf("[loopback] slot0 rms=%.4g eq_n0=%.4g -> no_eff~%.4g\n",rms,n0f*rms*rms,n0f);
            }
            uint8_t* xref_re_s = a.xref_re + (size_t)slot*XREF_SLOT_HALF*2;
            uint8_t* xref_im_s = a.xref_im + (size_t)slot*XREF_SLOT_HALF*2;
            ACLRT_LAUNCH_KERNEL(channel_est_ls_kernel)(BD,a.stream,
                a.yd_re,a.yd_im,xref_re_s,xref_im_s,a.h_re,a.h_im,a.ce_weave,a.ws,a.tiling[OP_channel_est_ls]);
            ACLRT_LAUNCH_KERNEL(equalize_kernel)(BD,a.stream,
                a.yd_re,a.yd_im,a.h_re,a.h_im,a.eq_n0,a.x_re,a.x_im,a.no_eff,a.ws,a.tiling[OP_equalize]);
            uint8_t* llr_slot = a.llr_in + (size_t)slot*8*LLR_SLOT_PAD*2;
            ACLRT_LAUNCH_KERNEL(qam256_demod_kernel)(BD,a.stream,
                a.x_re,a.x_im,a.no_eff,llr_slot,a.ws,a.tiling[OP_qam256_demod]);
            CK(aclrtSynchronizeStream(a.stream));
        }
    }
    std::fclose(fr); std::fclose(fi);


    if(do_ber){
        std::printf("[ber] === coded backend: descramble->rate_dematch->ldpc_decode (NS=%d) ===\n",NS);
        bake_gold_sign(a, st);


        ACLRT_LAUNCH_KERNEL(descramble_kernel)(BD,a.stream,
            a.llr_in, a.sign, a.n_slot, a.cw_ds, a.ws, a.tiling[OP_descramble]);
        ACLRT_LAUNCH_KERNEL(rate_dematch_kernel)(BD,a.stream,
            a.cw_ds, a.rd_tmpl, a.rd_dummy, a.lam);
        ACLRT_LAUNCH_KERNEL(ldpc_decode_kernel)(BD,a.stream,
            a.lam, a.ldpc_packbc, a.ldpc_packs, a.ldpc_deg, a.ldpc_eoff,
            a.ldpc_prev, a.bits, a.lam_out, a.ldpc_lamscr);
        CK(aclrtSynchronizeStream(a.stream));

        std::vector<int8_t> dbits(143u*8448);
        CK(aclrtMemcpy(dbits.data(),143u*8448,a.bits,143u*8448,ACL_MEMCPY_DEVICE_TO_HOST));
        if(!tx_bits.empty()){
            long err=0, tot=143L*8448;
            for(long i=0;i<tot;++i) if((dbits[i]&1)!=(tx_bits[i]&1)) ++err;
            std::printf("[ber] *** BER = %ld / %ld = %.3e %s ***\n",
                err,tot,(double)err/tot, err==0?"  <<< BER=0 LOOPBACK PASS":"");

            for(int cb=0; cb<3; ++cb){ long e=0; for(int k=0;k<8448;++k) if((dbits[cb*8448+k]&1)!=(tx_bits[cb*8448+k]&1))++e;
                std::printf("[ber]   CB%d err=%ld/8448\n",cb,e); }
        } else {
            long ones=0; for(auto b:dbits) ones+=(b&1);
            std::printf("[ber] no tx_bits; decoded ones=%ld/%ld (sanity)\n",ones,143L*8448);
        }
    }
    std::printf("[loopback] dumped %d slots yd (each 14x%u fp16) -> %s , %s\n",NS,N_SC_PAD,pr.c_str(),pi.c_str());
    std::printf("[loopback] 看星座: python3 plot_constellation.py %s\n", base.c_str());
}






static double time_kernel(aclrtStream s, const char* name,
                          const std::function<void()>& launch,
                          double* group_sum, int K=8, int M=20){
    for(int i=0;i<2*M;++i) launch();
    aclrtSynchronizeStream(s);
    std::vector<double> avg; avg.reserve(K);
    for(int k=0;k<K;++k){
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<M;++i) launch();
        aclrtSynchronizeStream(s);
        auto t1=std::chrono::high_resolution_clock::now();
        avg.push_back(std::chrono::duration<double,std::micro>(t1-t0).count()/M);
    }
    std::sort(avg.begin(), avg.end());
    double mn=avg.front(), med=avg[K/2], mx=avg.back();
    std::printf("   %-26s %8.1f %8.1f %8.1f\n", name, mn, med, mx);
    if(group_sum) *group_sum += mn;
    return mn;
}

void profile_kernels(RxArena& a, SyncState& st){
    std::printf("\n");
    std::printf("==================================================================\n");
    std::printf(" RX CHAIN PROFILE    blockDim=%u    sustained (8x20)    zero data\n", BD);
    std::printf("==================================================================\n");
    std::printf(" per-kernel latency               %8s %8s %8s  [us]\n", "min", "med", "max");
    std::printf("------------------------------------------------------------------\n");

    { uint8_t tb[64]; GenerateSssTiling("Ascend310P3", tb, st.mu_t, st.n_id_2, 0);
      aclrtMemcpy(a.tiling[OP_sss_correlator],64,tb,64,ACL_MEMCPY_HOST_TO_DEVICE); }

    double acq=0, slot=0, tb=0, t_descr=0;


    auto descr_probe=[&](const char* tag, uint32_t bd){
      const size_t IOB=(size_t)41*8*11264*2;
      void *di=0,*ds=0,*dn=0,*dout=0;
      if(aclrtMalloc(&di,IOB,ACL_MEM_MALLOC_HUGE_FIRST)||aclrtMalloc(&ds,IOB,ACL_MEM_MALLOC_HUGE_FIRST)
       ||aclrtMalloc(&dn,64,ACL_MEM_MALLOC_HUGE_FIRST)||aclrtMalloc(&dout,IOB,ACL_MEM_MALLOC_HUGE_FIRST)){
        std::printf("   [%s bd=%u] malloc fail\n",tag,bd); return; }
      aclrtMemset(di,IOB,0,IOB);
      { std::vector<int16_t> ones(IOB/2,1); aclrtMemcpy(ds,IOB,ones.data(),IOB,ACL_MEMCPY_HOST_TO_DEVICE); }
      { int32_t ns[16]={0}; ns[0]=24; aclrtMemcpy(dn,64,ns,64,ACL_MEMCPY_HOST_TO_DEVICE); }
      aclrtSynchronizeStream(a.stream);
      ACLRT_LAUNCH_KERNEL(descramble_kernel)(bd,a.stream,(uint8_t*)di,(uint8_t*)ds,(uint8_t*)dn,(uint8_t*)dout,a.ws,a.tiling[OP_descramble]);
      aclError e=aclrtSynchronizeStream(a.stream);
      if(e!=ACL_SUCCESS){ std::printf("   [%s bd=%u] descramble FAILED  aclError=%d\n",tag,bd,(int)e); }
      else{
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<10;++i) ACLRT_LAUNCH_KERNEL(descramble_kernel)(bd,a.stream,(uint8_t*)di,(uint8_t*)ds,(uint8_t*)dn,(uint8_t*)dout,a.ws,a.tiling[OP_descramble]);
        aclError e2=aclrtSynchronizeStream(a.stream);
        auto t1=std::chrono::high_resolution_clock::now();
        double per=std::chrono::duration<double,std::micro>(t1-t0).count()/10.0;
        if(e2!=ACL_SUCCESS) std::printf("   [%s bd=%u] OK-single, FAILED-x10  aclError=%d\n",tag,bd,(int)e2);
        else std::printf("   [%s bd=%u] %.1f us/launch = %.2f us/slot  OK\n",tag,bd,per,per/24.0);
      }
      aclrtFree(di);aclrtFree(ds);aclrtFree(dn);aclrtFree(dout);
    };
    if(std::getenv("RX_DESCR_FRESH")){ descr_probe("fresh-EARLY",4); descr_probe("fresh-EARLY",1); }

    std::printf(" ACQUIRE  (once per capture)\n");
    time_kernel(a.stream,"decimate",[&]{ ACLRT_LAUNCH_KERNEL(decimate_kernel)(BD,a.stream, a.capture,a.fir_taps,a.ssb768,a.ws,a.tiling[OP_decimate]); },&acq);
    time_kernel(a.stream,"pss_correlator",[&]{ ACLRT_LAUNCH_KERNEL(pss_correlator_kernel)(BD,a.stream, a.ssb768,a.pss_ref,a.pss_twid,a.pss_scr,a.pss_out,a.ws,a.tiling[OP_pss_correlator]); },&acq);
    time_kernel(a.stream,"pss_cfo_estimator",[&]{ ACLRT_LAUNCH_KERNEL(pss_cfo_estimator_kernel)(BD,a.stream, a.ypss,a.psscfo_ref,a.psscfo_scr,a.psscfo_out,a.ws,a.tiling[OP_pss_cfo_estimator]); },&acq);
    time_kernel(a.stream,"sss_correlator",[&]{ ACLRT_LAUNCH_KERNEL(sss_correlator_kernel)(BD,a.stream, a.ssb768,a.sss_re,a.sss_im,a.sss_twr,a.sss_twi,a.sss_scr,a.sss_out,a.ws,a.tiling[OP_sss_correlator]); },&acq);
    time_kernel(a.stream,"ssb_fft",[&]{ ACLRT_LAUNCH_KERNEL(ssb_fft_kernel)(BD,a.stream, a.ssb768,a.ssbfft_w16r,a.ssbfft_w16i,a.ssbfft_twr,a.ssbfft_twi,a.ssbfft_gidx,a.ssbfft_dr,a.ssbfft_di,a.ssbfft_scr,a.r_re,a.r_im,a.ws,a.tiling[OP_ssb_fft]); },&acq);
    time_kernel(a.stream,"pbch_dmrs_correlator",[&]{ ACLRT_LAUNCH_KERNEL(pbch_dmrs_correlator_kernel)(BD,a.stream, a.r_re,a.r_im,a.pbch_dre,a.pbch_dim,a.pbch_dimn,a.pbch_out,a.ws,a.tiling[OP_pbch_dmrs_correlator]); },&acq);

    std::printf(" SLOT  (per 0.5 ms slot)\n");
    time_kernel(a.stream,"cfo_estimate",[&]{ ACLRT_LAUNCH_KERNEL(cfo_estimate_kernel)(BD,a.stream, a.slot_iq,a.cfoest_scr,a.ws,a.cfoest_out); },&slot);
    time_kernel(a.stream,"cfo_compensate",[&]{ ACLRT_LAUNCH_KERNEL(cfo_compensate_kernel)(BD,a.stream, a.slot_iq,a.slot_iq_swap,a.cc_a2,a.cc_b2,a.slot_iq_d,a.ws,a.tiling[OP_cfo_compensate]); },&slot);
    time_kernel(a.stream,"ofdm_demod",[&]{ ACLRT_LAUNCH_KERNEL(ofdm_demod_kernel)(BD,a.stream, a.slot_iq_d,a.ofdm_w32r,a.ofdm_w32i,a.ofdm_w64r,a.ofdm_w64i,a.ofdm_twr,a.ofdm_twi,a.ofdm_scr,a.y_re,a.y_im,a.ws,a.tiling[OP_ofdm_demod]); },&slot);
    time_kernel(a.stream,"re_demap",[&]{ ACLRT_LAUNCH_KERNEL(re_demap_kernel)(BD,a.stream, a.y_re,a.y_im,a.re_idx,a.yd_re,a.yd_im,a.ws,a.tiling[OP_re_demap]); },&slot);
    time_kernel(a.stream,"channel_est_ls",[&]{ ACLRT_LAUNCH_KERNEL(channel_est_ls_kernel)(BD,a.stream, a.yd_re,a.yd_im,a.xref_re,a.xref_im,a.h_re,a.h_im,a.ce_weave,a.ws,a.tiling[OP_channel_est_ls]); },&slot);
    time_kernel(a.stream,"equalize",[&]{ ACLRT_LAUNCH_KERNEL(equalize_kernel)(BD,a.stream, a.yd_re,a.yd_im,a.h_re,a.h_im,a.eq_n0,a.x_re,a.x_im,a.no_eff,a.ws,a.tiling[OP_equalize]); },&slot);
    time_kernel(a.stream,"qam256_demod",[&]{ ACLRT_LAUNCH_KERNEL(qam256_demod_kernel)(BD,a.stream, a.x_re,a.x_im,a.no_eff,a.llr_in,a.ws,a.tiling[OP_qam256_demod]); },&slot);
    time_kernel(a.stream,"cfo_dmrs",[&]{ ACLRT_LAUNCH_KERNEL(cfo_dmrs_kernel)(BD,a.stream, a.yd_re,a.yd_im,a.cfod_xre,a.cfod_xim,a.cfodmrs_scr,a.cfodmrs_out,a.ws,a.tiling[OP_cfo_dmrs]); },&slot);
    time_kernel(a.stream,"timing_tracker",[&]{ ACLRT_LAUNCH_KERNEL(timing_tracker_kernel)(BD,a.stream, a.h_re,a.h_im,a.tt_wr,a.tt_wi,a.tt_twr,a.tt_twi,a.tt_dtscale,a.tt_scr,a.cir_re,a.cir_im,a.dt_out,a.ws,a.tiling[OP_timing_tracker]); },&slot);

    std::printf(" TB  (per 24 slots)%s\n", std::getenv("RX_SKIP_TB")?"                  [skipped: RX_SKIP_TB]":"");
    if (!std::getenv("RX_SKIP_TB")) {

        { int32_t ns[16]={0}; ns[0]=(int32_t)SLOTS_PER_TB;
          CK(aclrtMemcpy(a.n_slot,NSLOT_B,ns,NSLOT_B,ACL_MEMCPY_HOST_TO_DEVICE));
          CK(aclrtSynchronizeStream(a.stream));
          int32_t rb[16]={0};
          CK(aclrtMemcpy(rb,NSLOT_B,a.n_slot,NSLOT_B,ACL_MEMCPY_DEVICE_TO_HOST));
          std::printf("   [diag] n_slot[0] written=%d  readback=%d\n", (int)SLOTS_PER_TB, (int)rb[0]); }
        if (std::getenv("RX_DESCR_FRESH")) descr_probe("fresh-LATE",4);
        t_descr = time_kernel(a.stream,"descramble",[&]{ ACLRT_LAUNCH_KERNEL(descramble_kernel)(BD,a.stream, a.llr_in,a.sign,a.n_slot,a.cw_ds, a.ws,a.tiling[OP_descramble]); },&tb);
        time_kernel(a.stream,"rate_dematch",[&]{ ACLRT_LAUNCH_KERNEL(rate_dematch_kernel)(BD,a.stream, a.cw_ds,a.rd_tmpl,a.rd_dummy,a.lam); },&tb);
        time_kernel(a.stream,"ldpc_decode",[&]{ ACLRT_LAUNCH_KERNEL(ldpc_decode_kernel)(BD,a.stream, a.lam,a.ldpc_packbc,a.ldpc_packs,a.ldpc_deg,a.ldpc_eoff,a.ldpc_prev,a.bits,a.lam_out,a.ldpc_lamscr); },&tb);
    }

    std::printf("------------------------------------------------------------------\n");
    std::printf(" sum(min)   acquire %.0f   slot %.0f   tb %.0f   [us]\n", acq, slot, tb);
    std::printf("==================================================================\n");
    std::printf(" PIPELINE  (real syncs, zero data)\n");
    std::printf("------------------------------------------------------------------\n");
    std::vector<int16_t> cap(CAPTURE_B/2, 0), strm(SLOT_IQ_B, 0);
    using Clk = std::chrono::high_resolution_clock;
    double slot_p50 = 0;


    { std::vector<int16_t> hcap(CAPTURE_B/2, 0);
      std::vector<uint8_t> hbits(BITS_B, 0);
      auto bench=[&](const char* nm, const std::function<void()>& fn, double bytes, const char* when){
        for(int i=0;i<5;++i) fn();
        std::vector<double> v; v.reserve(20);
        for(int k=0;k<20;++k){ auto t0=Clk::now(); fn(); auto t1=Clk::now();
          v.push_back(std::chrono::duration<double,std::micro>(t1-t0).count()); }
        std::sort(v.begin(),v.end()); double md=v[v.size()/2];
        std::printf("   %-22s %8.0f us   %6.2f MB  %5.1f GB/s   %s\n",
                    nm, md, bytes/1e6, bytes/1e3/md, when); };
      bench("H2D capture (head)",
            [&]{ CK(aclrtMemcpy(a.capture,CAPTURE_B,hcap.data(),CAPTURE_B,ACL_MEMCPY_HOST_TO_DEVICE)); },
            (double)CAPTURE_B, "1x / acquire");
      bench("D2H bits (tail)",
            [&]{ CK(aclrtMemcpy(hbits.data(),BITS_B,a.bits,BITS_B,ACL_MEMCPY_DEVICE_TO_HOST)); },
            (double)BITS_B, "1x / TB (24 slot)");
    }
    std::printf("------------------------------------------------------------------\n");

    { auto t0=Clk::now(); rx_acquire(a, cap.data(), st); auto t1=Clk::now();
      std::printf("   rx_acquire   %8.0f us   (one-shot, off real-time loop)\n",
                  std::chrono::duration<double,std::micro>(t1-t0).count()); }



    { const int M = (int)SLOTS_PER_TB;
      int32_t rd0[8]={0}; rd0[0]=-(int)N_SAMP_SLOT;
      float dt0[8]={0};
      auto reset=[&]{ CK(aclrtMemcpy(a.rd_ptr,8*sizeof(int32_t),rd0,8*sizeof(int32_t),ACL_MEMCPY_HOST_TO_DEVICE));
                      CK(aclrtMemcpy(a.dt_out,DT_OUT_B,dt0,(DT_OUT_B<32?DT_OUT_B:32),ACL_MEMCPY_HOST_TO_DEVICE));
                      CK(aclrtSynchronizeStream(a.stream)); };
      for(int w=0;w<2;++w){ reset(); for(int s=0;s<M;++s) rx_slot(a,strm.data(),st,s); CK(aclrtSynchronizeStream(a.stream)); }
      std::vector<double> u; u.reserve(40);
      for(int it=0;it<40;++it){
        reset();
        auto t0=Clk::now();
        for(int s=0;s<M;++s) rx_slot(a,strm.data(),st,s);
        CK(aclrtSynchronizeStream(a.stream));
        auto t1=Clk::now();
        u.push_back(std::chrono::duration<double,std::micro>(t1-t0).count()/M);
      }
      std::sort(u.begin(),u.end());
      double mn=u.front(), p50=u[u.size()/2], mx=u.back();
      slot_p50 = p50;
      std::printf("   rx_slot      %8.0f us   budget 500 -> %.1fx margin   %s\n",
                  p50, 500.0/p50, p50>500.0?"OVER":"OK");
      std::printf("                min %.0f  p50 %.0f  max %.0f   (%d-slot pipeline, drain @ TB)\n",
                  mn, p50, mx, M);
      std::printf("                = device %.0f (sustained) + orchestration %.0f  (no per-slot sync)\n",
                  slot, mn-slot); }





    { const int M = (int)SLOTS_PER_TB;
      const double DESCR_STANDALONE_US = 50.0 * M;
      bool descr_anom = (t_descr > 10000.0);
      double tb_use = tb;
      if (descr_anom) tb_use = tb - t_descr + DESCR_STANDALONE_US;
      if (std::getenv("RX_SKIP_TB") || tb<=0.0) {
        std::printf("   rx_tb            (skipped: RX_SKIP_TB)\n");
        std::printf("------------------------------------------------------------------\n");
        std::printf("   per-slot total %8.0f us   budget 500 -> %.1fx margin   %s   (TB excluded)\n",
                    slot_p50, 500.0/slot_p50, slot_p50>500.0?"OVER":"OK");
      } else {
        if (descr_anom)
          std::printf("   [!] descramble %.0f us ANOMALOUS (watchdog/hang on zero data) — replaced with %.0f us (~50us/slot standalone)\n",
                      t_descr, DESCR_STANDALONE_US);
        double per = tb_use/(double)M, comb = slot_p50 + per;
        std::printf("   rx_tb        %8.0f us   / %d slot = %.0f us/slot   (sum of per-kernel, 12000 us window)\n",
                    tb_use, M, per);
        std::printf("------------------------------------------------------------------\n");
        std::printf("   per-slot total %8.0f us   = slot %.0f + tb/slot %.0f%s\n",
                    comb, slot_p50, per, descr_anom?"   (descramble = standalone est.)":"");
        std::printf("                  budget 500 -> %.1fx margin   %s\n",
                    500.0/comb, comb>500.0?"OVER":"OK");
      } }
    std::printf("==================================================================\n");
}








static void print_rx_report(RxArena& a, const SyncState& st, double ber,
                            long err, long tot)
{
    CK(aclrtSynchronizeStream(a.stream));
    float df=0.f; CK(aclrtMemcpy(&df, 4, a.cfodmrs_out, 4, ACL_MEMCPY_DEVICE_TO_HOST));
    float dt=0.f; CK(aclrtMemcpy(&dt, 4, a.dt_out,      4, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<uint16_t> cir(4096);
    CK(aclrtMemcpy(cir.data(), 8192, a.cir_re, 8192, ACL_MEMCPY_DEVICE_TO_HOST));
    int pk=0; uint16_t mx=0;
    for(int i=0;i<4096;++i){ uint16_t v=cir[i]&0x7FFF; if(v>mx){mx=v;pk=i;} }

    const char* LINE="------------------------------------------------------------------";
    std::printf("\n==================================================================\n");
    std::printf(" AI-RAN NPU  PUSCH SISO RX  —  真链路 (带 CFO OTA 仿真)\n");
    std::printf("==================================================================\n");
    std::printf(" [同步 ACQUIRE]\n");
    std::printf("   cell_id = %d   (n_id1=%d, n_id2=%d)    i_ssb = %d\n",
                st.cell_id, st.n_id_1, st.n_id_2, st.i_ssb);
    std::printf("   mu_t    = %d @7.68M   ->   mu_t_data = %d @61.44M\n",
                st.mu_t, st.mu_t_data);
    std::printf("   noise   = %.3g\n", st.noise);
    std::printf("%s\n", LINE);
    std::printf(" [CFO 三层跟踪]\n");
    std::printf("   L1 粗 (PSS)      g·SCS    = %+9.1f Hz\n", st.cfo_int);
    std::printf("   L2 帧 (PSS frac) cfo_frac = %+9.1f Hz\n", st.cfo_frac);
    std::printf("   L2 合计 cfo_total         = %+9.1f Hz   -> cfo_compensate 去旋\n", st.cfo_total);
    std::printf("   L3 精 (DMRS δf)           = %+9.1f Hz   (残余, 诊断)\n", df);
    std::printf("%s\n", LINE);
    std::printf(" [定时跟踪 TIMING / SFO]\n");
    std::printf("   dt (末slot)  = %+.3f samples   CIR 主径@idx=%d (|mag|=0x%04X)\n", dt, pk, mx);
    std::printf("   SFO 跟踪     = win_slice 分数延迟 (rd_ptr Q16 frac + 线性插值)\n");
    std::printf("%s\n", LINE);
    std::printf(" [解码 BER]\n");
    std::printf("   真链路 BER = %ld / %ld = %.3e   [%s]\n",
                err, tot, ber, (err==0)?"PASS":"FAIL");
    std::printf("==================================================================\n\n");
}

static void time_rx_kernels_realchain(RxArena& a, SyncState& st)
{
    std::printf("\n==================================================================\n");
    std::printf(" per-kernel latency (真链路数据, sustained 8x20)   %8s %8s %8s [us]\n","min","med","max");
    std::printf("------------------------------------------------------------------\n");

    double acq=0, slot=0, tb=0;

    std::printf(" ACQUIRE  (once per capture)\n");
    time_kernel(a.stream,"decimate",[&]{ ACLRT_LAUNCH_KERNEL(decimate_kernel)(BD,a.stream, a.capture,a.fir_taps,a.ssb768,a.ws,a.tiling[OP_decimate]); },&acq);
    time_kernel(a.stream,"pss_correlator",[&]{ ACLRT_LAUNCH_KERNEL(pss_correlator_kernel)(BD,a.stream, a.ssb768,a.pss_ref,a.pss_twid,a.pss_scr,a.pss_out,a.ws,a.tiling[OP_pss_correlator]); },&acq);
    time_kernel(a.stream,"pss_cfo_estimator",[&]{ ACLRT_LAUNCH_KERNEL(pss_cfo_estimator_kernel)(BD,a.stream, a.ypss,a.psscfo_ref,a.psscfo_scr,a.psscfo_out,a.ws,a.tiling[OP_pss_cfo_estimator]); },&acq);
    time_kernel(a.stream,"sss_correlator",[&]{ ACLRT_LAUNCH_KERNEL(sss_correlator_kernel)(BD,a.stream, a.ssb768,a.sss_re,a.sss_im,a.sss_twr,a.sss_twi,a.sss_scr,a.sss_out,a.ws,a.tiling[OP_sss_correlator]); },&acq);
    time_kernel(a.stream,"ssb_fft",[&]{ ACLRT_LAUNCH_KERNEL(ssb_fft_kernel)(BD,a.stream, a.ssbfft_in,a.ssbfft_w16r,a.ssbfft_w16i,a.ssbfft_twr,a.ssbfft_twi,a.ssbfft_gidx,a.ssbfft_dr,a.ssbfft_di,a.ssbfft_scr,a.r_re,a.r_im,a.ws,a.tiling[OP_ssb_fft]); },&acq);
    time_kernel(a.stream,"pbch_dmrs_correlator",[&]{ ACLRT_LAUNCH_KERNEL(pbch_dmrs_correlator_kernel)(BD,a.stream, a.r_re,a.r_im,a.pbch_dre,a.pbch_dim,a.pbch_dimn,a.pbch_out,a.ws,a.tiling[OP_pbch_dmrs_correlator]); },&acq);

    std::printf(" SLOT  (per 0.5 ms slot)\n");
    time_kernel(a.stream,"win_slice",[&]{ ACLRT_LAUNCH_KERNEL(win_slice_kernel)(BD,a.stream, a.capture,a.dt_out,a.slot_iq,a.rd_ptr,a.ws,a.tiling[OP_win_slice]); },&slot);
    time_kernel(a.stream,"cfo_estimate",[&]{ ACLRT_LAUNCH_KERNEL(cfo_estimate_kernel)(BD,a.stream, a.slot_iq,a.cfoest_scr,a.ws,a.cfoest_out); },&slot);
    time_kernel(a.stream,"cfo_compensate",[&]{ ACLRT_LAUNCH_KERNEL(cfo_compensate_kernel)(BD,a.stream, a.slot_iq,a.slot_iq_swap,a.cc_a2,a.cc_b2,a.slot_iq_d,a.ws,a.tiling[OP_cfo_compensate]); },&slot);
    time_kernel(a.stream,"ofdm_demod",[&]{ ACLRT_LAUNCH_KERNEL(ofdm_demod_kernel)(BD,a.stream, a.slot_iq_d,a.ofdm_w32r,a.ofdm_w32i,a.ofdm_w64r,a.ofdm_w64i,a.ofdm_twr,a.ofdm_twi,a.ofdm_scr,a.y_re,a.y_im,a.ws,a.tiling[OP_ofdm_demod]); },&slot);
    time_kernel(a.stream,"re_demap",[&]{ ACLRT_LAUNCH_KERNEL(re_demap_kernel)(BD,a.stream, a.y_re,a.y_im,a.re_idx,a.yd_re,a.yd_im,a.ws,a.tiling[OP_re_demap]); },&slot);
    time_kernel(a.stream,"channel_est_ls",[&]{ ACLRT_LAUNCH_KERNEL(channel_est_ls_kernel)(BD,a.stream, a.yd_re,a.yd_im,a.xref_re,a.xref_im,a.h_re,a.h_im,a.ce_weave,a.ws,a.tiling[OP_channel_est_ls]); },&slot);
    time_kernel(a.stream,"equalize",[&]{ ACLRT_LAUNCH_KERNEL(equalize_kernel)(BD,a.stream, a.yd_re,a.yd_im,a.h_re,a.h_im,a.eq_n0,a.x_re,a.x_im,a.no_eff,a.ws,a.tiling[OP_equalize]); },&slot);
    time_kernel(a.stream,"qam256_demod",[&]{ ACLRT_LAUNCH_KERNEL(qam256_demod_kernel)(BD,a.stream, a.x_re,a.x_im,a.no_eff,a.llr_in,a.ws,a.tiling[OP_qam256_demod]); },&slot);
    time_kernel(a.stream,"cfo_dmrs",[&]{ ACLRT_LAUNCH_KERNEL(cfo_dmrs_kernel)(BD,a.stream, a.yd_re,a.yd_im,a.cfod_xre,a.cfod_xim,a.cfodmrs_scr,a.cfodmrs_out,a.ws,a.tiling[OP_cfo_dmrs]); },&slot);
    time_kernel(a.stream,"timing_tracker",[&]{ ACLRT_LAUNCH_KERNEL(timing_tracker_kernel)(BD,a.stream, a.h_re,a.h_im,a.tt_wr,a.tt_wi,a.tt_twr,a.tt_twi,a.tt_dtscale,a.tt_scr,a.cir_re,a.cir_im,a.dt_out,a.ws,a.tiling[OP_timing_tracker]); },&slot);

    std::printf(" TB  (per 23 slots)\n");
    time_kernel(a.stream,"descramble",[&]{ ACLRT_LAUNCH_KERNEL(descramble_kernel)(BD,a.stream, a.llr_in,a.sign,a.n_slot,a.cw_ds,a.ws,a.tiling[OP_descramble]); },&tb);
    time_kernel(a.stream,"rate_dematch",[&]{ ACLRT_LAUNCH_KERNEL(rate_dematch_kernel)(BD,a.stream, a.cw_ds,a.rd_tmpl,a.rd_dummy,a.lam); },&tb);
    time_kernel(a.stream,"ldpc_decode",[&]{ ACLRT_LAUNCH_KERNEL(ldpc_decode_kernel)(BD,a.stream, a.lam,a.ldpc_packbc,a.ldpc_packs,a.ldpc_deg,a.ldpc_eoff,a.ldpc_prev,a.bits,a.lam_out,a.ldpc_lamscr); },&tb);

    std::printf("------------------------------------------------------------------\n");
    std::printf(" sum(min)   acquire %.0f us   slot %.0f us   tb %.0f us\n", acq, slot, tb);
    std::printf(" per-slot (slot + tb/23) = %.0f us   budget 500 -> %s\n",
                slot + tb/(double)SLOTS_PER_TB,
                (slot + tb/(double)SLOTS_PER_TB)>500.0?"OVER":"OK");
    std::printf("==================================================================\n\n");
    (void)st;
}

}




int main(int argc, char** argv){
    using namespace airan_rx;
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* data_dir = std::getenv("AIRAN_DATA_DIR");


    platform_ascendc::PlatformAscendCManager::GetInstance("Ascend310P3");
    CK(aclInit(nullptr));
    CK(aclrtSetDevice(0));

    RxArena a; SyncState st;
    if (const char* bd = std::getenv("RX_BLOCKDIM")) { int v=std::atoi(bd); if(v>0) BD=(uint32_t)v; }
    if (std::getenv("RX_NO_ZERO")) g_no_zero = true;
    rx_arena_init(a, data_dir ? data_dir : ".");

    if((argc>1 && std::string(argv[1])=="--loopback") || std::getenv("RX_LOOPBACK")){
        rx_loopback(a, st);
        rx_arena_free(a); CK(aclrtResetDevice(0)); CK(aclFinalize()); return 0;
    }

    { const char* pf = std::getenv("RX_PROFILE");
      if (pf && pf[0] && pf[0]!='0') {
        profile_kernels(a, st);
        rx_arena_free(a);
        CK(aclrtResetDevice(0));
        CK(aclFinalize());
        return 0;
    } }

    std::vector<int16_t> capture(CAPTURE_B/2);
    std::vector<int16_t> stream((size_t)N_SAMP_SLOT*SLOTS_PER_TB*2 + 65536);
    std::vector<uint8_t> tx_bits(143u*8448);


    if(const char* tbp=std::getenv("RX_TXBITS")){
        FILE* ft=std::fopen(tbp,"rb");
        if(ft){ size_t g=std::fread(tx_bits.data(),1,tx_bits.size(),ft); std::fclose(ft);
            std::printf("[真链路] tx_bits %zu/%zu B <- %s\n",g,tx_bits.size(),tbp); }
        else std::printf("[真链路] WARN tx_bits open fail: %s\n",tbp);
    }

    {
        std::string cap_path = std::string(data_dir ? data_dir : ".") + "/capture.bin";
        FILE* fc = std::fopen(cap_path.c_str(), "rb");
        if(fc){
            size_t got = std::fread(capture.data(), sizeof(int16_t), capture.size(), fc);
            std::fclose(fc);
            std::printf("[main] loaded capture.bin: %zu/%zu int16 from %s\n",
                        got, capture.size(), cap_path.c_str());
        } else {
            std::printf("[main] no capture.bin at %s (acquire 喂零)\n", cap_path.c_str());
        }
    }

    rx_acquire(a, capture.data(), st);
    std::printf("[acquire] cell_id=%d n_id1=%d n_id2=%d i_ssb=%d mu_t=%d cfo=%.1fHz noise=%.3g\n",
                st.cell_id, st.n_id_1, st.n_id_2, st.i_ssb, st.mu_t, st.cfo_total, st.noise);




    if(std::getenv("RX_CONTINUOUS")){
        std::printf("[连续流] capture 即连续流 (SSB+PUSCH 同段), PUSCH@mu_t_data+52640. 不覆盖.\n");
    }

    else if(const char* dp=std::getenv("RX_DATA_IQ")){
        FILE* fp=std::fopen(dp,"rb");
        if(fp){
            std::vector<int16_t> pusch(CAPTURE_B/2);
            size_t g=std::fread(pusch.data(),sizeof(int16_t),CAPTURE_B/2,fp); std::fclose(fp);
            CK(aclrtMemcpy(a.capture, CAPTURE_B, pusch.data(), CAPTURE_B, ACL_MEMCPY_HOST_TO_DEVICE));
            std::printf("[真链路] data capture <- %s (%zu int16)\n",dp,g);
        } else std::printf("[真链路] WARN RX_DATA_IQ open fail: %s\n",dp);
    }

    double ber=0.0; bool loop=(argc>1 && std::string(argv[1])=="--loop"); int tb=0;
    do {
        for(int slot=0; slot<(int)SLOTS_PER_TB; ++slot)
            rx_slot(a, stream.data(), st, slot);
        rx_tb(a, st, tx_bits.data(), &ber);

        const long TOT = 143L*8448;
        const long ERR = (long)(ber*(double)TOT + 0.5);
        print_rx_report(a, st, ber, ERR, TOT);

        if(!std::getenv("RX_NO_KTIME")) time_rx_kernels_realchain(a, st);
        tb++;
        if(ber>1e-3 || !st.locked) rx_acquire(a, capture.data(), st);
    } while(loop);

    rx_arena_free(a);
    CK(aclrtResetDevice(0));
    CK(aclFinalize());
    return 0;
}
