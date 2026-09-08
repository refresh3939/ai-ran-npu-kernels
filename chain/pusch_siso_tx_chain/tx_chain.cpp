



#include "tx_chain.h"
#include "data_utils.h"

#include <acl/acl.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>

#include "aclrtlaunch_ldpc_encode_kernel.h"
#include "aclrtlaunch_rate_match_kernel.h"
#include "aclrtlaunch_scramble_kernel.h"
#include "aclrtlaunch_qam256_mod_kernel.h"
#include "aclrtlaunch_dmrs_gen_kernel.h"
#include "aclrtlaunch_merge_dmrs_device_kernel.h"
#include "aclrtlaunch_re_map_kernel.h"
#include "aclrtlaunch_ofdm_mod_kernel.h"

#define ACL_CHECK(x) do{ aclError _e=(x); if(_e!=ACL_SUCCESS){ \
  std::fprintf(stderr,"[acl] fail %d @%s:%d\n",_e,__FILE__,__LINE__); std::exit(1);} }while(0)

namespace airan_tx {

static constexpr uint32_t BD = 4;


static std::string kernel_root(const char* data_dir){
    std::string d(data_dir);
    auto p = d.rfind("/chain/pusch_siso_tx_chain/");
    return (p==std::string::npos) ? d : d.substr(0,p);
}

static uint8_t* dmalloc(size_t b){
    void* p=nullptr;
    ACL_CHECK(aclrtMalloc(&p, b?b:1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(p, b?b:1, 0, b?b:1));
    return (uint8_t*)p;
}


static uint8_t* load_dev(const std::string& path, size_t expect=0){
    struct stat sb;
    if (stat(path.c_str(), &sb)!=0 || !S_ISREG(sb.st_mode) || sb.st_size==0){
        std::fprintf(stderr,"[aux] MISSING/empty %s  (先跑该 op 的 python ref 生成 weights)\n", path.c_str());
        std::exit(1);
    }
    size_t n=(size_t)sb.st_size;
    if (expect && n!=expect) std::fprintf(stderr,"[aux] WARN %s size=%zu expect=%zu\n",path.c_str(),n,expect);
    std::vector<uint8_t> h(n); size_t got=0;
    if(!ReadFile(path,got,h.data(),n)){ std::fprintf(stderr,"[aux] read fail %s\n",path.c_str()); std::exit(1);}
    uint8_t* d=dmalloc(n);
    ACL_CHECK(aclrtMemcpy(d,n,h.data(),n,ACL_MEMCPY_HOST_TO_DEVICE));
    return d;
}


static void fill_cinit(int slot, uint16_t n_id, uint16_t n_scid, int32_t out[14]){
    for(int l=0;l<14;++l){
        uint64_t v = ((uint64_t)1<<17)*(uint64_t)(14*slot+l+1)*(uint64_t)(2*n_id+1)
                   + (uint64_t)(2*n_id+n_scid);
        out[l]=(int32_t)(v & 0x7FFFFFFFu);
    }
}



static void BuildDmrsMatrix(uint16_t* gmat, uint16_t* g1){
    constexpr size_t N_RE=798, N_PAD=896, PLANE=1792, NBITS=31, NC=1600, M=1596;
    constexpr uint16_t H_ONE=0x3C00, H_ZERO=0x0000;
    std::vector<int8_t> x1(M+NC+31,0); x1[0]=1;
    for(size_t n=0;n<M+NC;++n) x1[n+31]=(x1[n+3]+x1[n])&1;
    auto put=[&](uint16_t* dst,const int8_t* seq){
        for(size_t k=0;k<N_RE;++k){ dst[k]=seq[2*k]?H_ONE:H_ZERO; dst[N_PAD+k]=seq[2*k+1]?H_ONE:H_ZERO; }
        for(size_t p=N_RE;p<N_PAD;++p){ dst[p]=H_ZERO; dst[N_PAD+p]=H_ZERO; }
    };
    put(g1,&x1[NC]);
    std::vector<int8_t> x2(M+NC+31,0);
    for(size_t i=0;i<NBITS;++i){
        std::fill(x2.begin(),x2.end(),0);
        for(size_t j=0;j<31;++j) x2[j]=((1u<<i)>>j)&1;
        for(size_t n=0;n<M+NC;++n) x2[n+31]=(x2[n+3]+x2[n+2]+x2[n+1]+x2[n])&1;
        put(&gmat[i*PLANE],&x2[NC]);
    }
}


static void upload_dmrs_cinit(TxArena& a, int slot, uint16_t n_id, uint16_t n_scid){
    int32_t ci14[14]; fill_cinit(slot, n_id, n_scid, ci14);
    int32_t ci[16]={0}; ci[0]=ci14[2]; ci[1]=ci14[11];
    ACL_CHECK(aclrtMemcpy(a.cinit, CINIT_B, ci, CINIT_B, ACL_MEMCPY_HOST_TO_DEVICE));
}




static void bake_tx_gold(TxArena& a, const TxState& st){
    constexpr int NSLOT=(int)SLOTS_PER_TB, NSTR=8, NVALID=19152, NSTRIDE=19200, GOLD_LEN=1600;
    const uint32_t n_RNTI=12345u, q=0u, N_ID=(uint32_t)st.cell_id;
    const uint32_t c_init=((n_RNTI&0xFFFFu)<<15)|((q&1u)<<14)|(N_ID&0x3FFu);
    const size_t length=(size_t)NSLOT*NVALID*NSTR, total=length+GOLD_LEN;
    std::vector<uint8_t> x1(total+31,0), x2(total+31,0); x1[0]=1;
    for(int i=0;i<31;++i) x2[i]=(uint8_t)((c_init>>i)&1u);
    for(size_t n=0;n<total;++n){
        x1[n+31]=(uint8_t)((x1[n+3]+x1[n])&1u);
        x2[n+31]=(uint8_t)((x2[n+3]+x2[n+2]+x2[n+1]+x2[n])&1u);
    }
    std::vector<int16_t> gold((size_t)NSLOT*NSTR*NSTRIDE, 0);
    for(int slot=0;slot<NSLOT;++slot)for(int b_nr=0;b_nr<NSTR;++b_nr){
        for(int re=0;re<NVALID;++re){
            const size_t idx=((size_t)slot*NVALID+re)*NSTR+b_nr;
            gold[((size_t)slot*NSTR+b_nr)*NSTRIDE+re]=
                (int16_t)((x1[GOLD_LEN+idx]^x2[GOLD_LEN+idx])&1u);
        }
    }
    ACL_CHECK(aclrtMemcpy(a.gold,(size_t)NSLOT*NSTR*NSTRIDE*2, gold.data(),
              (size_t)NSLOT*NSTR*NSTRIDE*2, ACL_MEMCPY_HOST_TO_DEVICE));
}
[[maybe_unused]] static void interleave_iq(uint8_t* d_re, uint8_t* d_im, uint8_t* d_iq,
                          int16_t* re, int16_t* im, int16_t* iq){
    ACL_CHECK(aclrtMemcpy(re, OFDM_PLN_B, d_re, OFDM_PLN_B, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(im, OFDM_PLN_B, d_im, OFDM_PLN_B, ACL_MEMCPY_DEVICE_TO_HOST));
    for(uint32_t i=0;i<N_SAMP_SLOT;++i){ iq[2*i]=re[i]; iq[2*i+1]=im[i]; }
    ACL_CHECK(aclrtMemcpy(d_iq, OFDM_IQ_B, iq, OFDM_IQ_B, ACL_MEMCPY_HOST_TO_DEVICE));
}

int tx_arena_init(TxArena& a, const char* data_dir){
    ACL_CHECK(aclrtCreateStream(&a.stream));
    const std::string R = kernel_root(data_dir);


    a.seg_bits = dmalloc(SEG_B);
    a.enc      = dmalloc(ENC_B);
    a.layout   = dmalloc(LAYOUT_B);
    a.scr      = dmalloc(SCR_B);
    a.ldpc_dbg = dmalloc(1u*1024*1024);


    a.gu_re=dmalloc(GRID_USED_B); a.gu_im=dmalloc(GRID_USED_B);
    a.dmrs_re=dmalloc(DMRS_B);    a.dmrs_im=dmalloc(DMRS_B);
    a.cinit=dmalloc(CINIT_B);
    a.dmrs_scr=dmalloc(64u*1024); a.dmrs_dbg=dmalloc(64u*1024);
    a.dmrs_gmat=dmalloc(DMRS_GMAT_B); a.dmrs_g1=dmalloc(DMRS_G1_B);
    { std::vector<uint16_t> gmat(DMRS_GMAT_B/2), g1(DMRS_G1_B/2);
      BuildDmrsMatrix(gmat.data(), g1.data());
      ACL_CHECK(aclrtMemcpy(a.dmrs_gmat, DMRS_GMAT_B, gmat.data(), DMRS_GMAT_B, ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(a.dmrs_g1,   DMRS_G1_B,   g1.data(),   DMRS_G1_B,   ACL_MEMCPY_HOST_TO_DEVICE)); }
    a.merge_idx=dmalloc(MERGE_IDX_B);
    { std::vector<uint32_t> idx(N_SC_PAD, 896u*sizeof(uint16_t));
      for(uint32_t sc=0; sc<N_SC_USED; sc+=2) idx[sc]=(sc/2)*sizeof(uint16_t);
      ACL_CHECK(aclrtMemcpy(a.merge_idx, MERGE_IDX_B, idx.data(), MERGE_IDX_B, ACL_MEMCPY_HOST_TO_DEVICE)); }
    a.gf_re=dmalloc(GRID_FFT_B);  a.gf_im=dmalloc(GRID_FFT_B);
    a.ofdm_scr=dmalloc(512u*1024);
    a.ofdm_re=dmalloc(OFDM_PLN_B);a.ofdm_im=dmalloc(OFDM_PLN_B);
    a.ofdm_iq=dmalloc(OFDM_IQ_B);


    const std::string LW = R + "/fec/ldpc_encode/data/weights/ldpc_bg1_z384_shifts/";
    a.sh_a  = load_dev(LW+"shift_A.bin");
    a.sh_bi = load_dev(LW+"shift_Bi.bin");
    a.sh_c  = load_dev(LW+"shift_C.bin");
    a.sh_d  = load_dev(LW+"shift_D.bin");


    a.gold  = dmalloc(SCR_B);
    a.scatter_idx = load_dev(R+"/mapping/re_map_siso/data/weights/scatter_idx.bin", IDX_B);
    const std::string OW = R + "/ofdm/ofdm_mod_siso/data/weights/";
    a.w32r=load_dev(OW+"iw_dft32_re.bin",   W32_B); a.w32i=load_dev(OW+"iw_dft32_im.bin",   W32_B);
    a.w64r=load_dev(OW+"iw_dft64_re_T.bin", W64_B); a.w64i=load_dev(OW+"iw_dft64_im_T.bin", W64_B);
    a.twr =load_dev(OW+"itwiddle_pq_re.bin",TW_B);  a.twi =load_dev(OW+"itwiddle_pq_im.bin",TW_B);


    a.n_slot = dmalloc(NSLOT_B);
    { int32_t ns[16]={0}; ns[0]=(int32_t)SLOTS_PER_TB;
      ACL_CHECK(aclrtMemcpy(a.n_slot, NSLOT_B, ns, NSLOT_B, ACL_MEMCPY_HOST_TO_DEVICE)); }

    setup_tilings(a);
    return 0;
}

void tx_arena_free(TxArena& a){
    uint8_t* all[]={a.ws,a.seg_bits,a.sh_a,a.sh_bi,a.sh_c,a.sh_d,a.ldpc_dbg,a.enc,a.layout,
        a.gold,a.n_slot,a.scr,a.gu_re,a.gu_im,a.dmrs_re,a.dmrs_im,a.cinit,a.dmrs_scr,a.dmrs_dbg,
        a.dmrs_gmat,a.dmrs_g1,a.merge_idx,
        a.scatter_idx,a.gf_re,a.gf_im,a.w32r,a.w32i,a.w64r,a.w64i,a.twr,a.twi,a.ofdm_scr,
        a.ofdm_re,a.ofdm_im,a.ofdm_iq};
    for(auto p:all) if(p) aclrtFree(p);
    for(int i=0;i<OP_N;++i) if(a.tiling[i]) aclrtFree(a.tiling[i]);
    if(a.stream) aclrtDestroyStream(a.stream);
}


void tx_tb(TxArena& a, const uint8_t* tx_bits_host, TxState& st){
    st.n_id = (uint16_t)st.cell_id;
    bake_tx_gold(a, st);
    ACL_CHECK(aclrtMemcpy(a.seg_bits, SEG_B, tx_bits_host, SEG_B, ACL_MEMCPY_HOST_TO_DEVICE));

    ACLRT_LAUNCH_KERNEL(ldpc_encode_kernel)(BD, a.stream,
        a.seg_bits, a.sh_a, a.sh_bi, a.sh_c, a.sh_d, a.ldpc_dbg, a.enc);





    ACLRT_LAUNCH_KERNEL(rate_match_kernel)(BD, a.stream,
        a.enc, a.enc, a.enc, a.layout);

    ACLRT_LAUNCH_KERNEL(scramble_kernel)(BD, a.stream,
        a.layout, a.gold, a.n_slot, a.scr, a.ws, a.tiling[OP_scramble]);

    (void)st;
}


void tx_slot(TxArena& a, TxState& st, int slot, int16_t* iq_out_host){

    uint8_t* bits_slot = a.scr + (size_t)slot * SLOT_BITS_B;


    ACLRT_LAUNCH_KERNEL(qam256_mod_kernel)(BD, a.stream,
        bits_slot, a.gu_re, a.gu_im, a.ws, a.tiling[OP_qam256_mod]);


    upload_dmrs_cinit(a, slot, st.n_id, st.n_scid);
    ACLRT_LAUNCH_KERNEL(dmrs_gen_kernel)(BD, a.stream,
        a.cinit, a.dmrs_gmat, a.dmrs_g1, a.dmrs_scr, a.dmrs_re, a.dmrs_im, a.dmrs_dbg, a.ws, a.tiling[OP_dmrs_gen]);
    ACLRT_LAUNCH_KERNEL(merge_dmrs_device_kernel)(BD, a.stream,
        a.gu_re, a.gu_im, a.dmrs_re, a.dmrs_im, a.merge_idx,
        a.ws, a.tiling[OP_re_map]);


    ACLRT_LAUNCH_KERNEL(re_map_kernel)(BD, a.stream,
        a.gu_re, a.gu_im, a.scatter_idx, a.gf_re, a.gf_im, a.ws, a.tiling[OP_re_map]);


    ACLRT_LAUNCH_KERNEL(ofdm_mod_kernel)(BD, a.stream,
        a.gf_re, a.gf_im, a.w32r, a.w32i, a.w64r, a.w64i, a.twr, a.twi, a.ofdm_scr,
        a.ofdm_re, a.ofdm_im, a.ofdm_iq, a.ws, a.tiling[OP_ofdm_mod]);

    if (iq_out_host) ACL_CHECK(aclrtMemcpy(iq_out_host, OFDM_IQ_B, a.ofdm_iq, OFDM_IQ_B, ACL_MEMCPY_DEVICE_TO_HOST));
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

void tx_profile_kernels(TxArena& a, TxState& st){

    upload_dmrs_cinit(a, 0, st.n_id, st.n_scid);
    uint8_t* bits_slot = a.scr;

    std::printf("\n==================================================================\n");
    std::printf(" TX CHAIN PROFILE    blockDim=%u    sustained (8x20)\n", BD);
    std::printf("   %-26s %8s %8s %8s\n", "kernel", "min", "med", "max");
    std::printf("------------------------------------------------------------- (us)\n");

    double tb_sum=0, slot_sum=0;
    std::printf(" -- per-TB (1x / 24 slots) --\n");
    time_kernel(a.stream,"ldpc_encode",[&]{ ACLRT_LAUNCH_KERNEL(ldpc_encode_kernel)(BD,a.stream,
        a.seg_bits,a.sh_a,a.sh_bi,a.sh_c,a.sh_d,a.ldpc_dbg,a.enc); },&tb_sum);
    time_kernel(a.stream,"rate_match",[&]{ ACLRT_LAUNCH_KERNEL(rate_match_kernel)(BD,a.stream,
        a.enc,a.enc,a.enc,a.layout); },&tb_sum);
    time_kernel(a.stream,"scramble",[&]{ ACLRT_LAUNCH_KERNEL(scramble_kernel)(BD,a.stream,
        a.layout,a.gold,a.n_slot,a.scr,a.ws,a.tiling[OP_scramble]); },&tb_sum);

    std::printf(" -- per-slot --\n");
    time_kernel(a.stream,"qam256_mod",[&]{ ACLRT_LAUNCH_KERNEL(qam256_mod_kernel)(BD,a.stream,
        bits_slot,a.gu_re,a.gu_im,a.ws,a.tiling[OP_qam256_mod]); },&slot_sum);
    time_kernel(a.stream,"dmrs_gen",[&]{ ACLRT_LAUNCH_KERNEL(dmrs_gen_kernel)(BD,a.stream,
        a.cinit,a.dmrs_gmat,a.dmrs_g1,a.dmrs_scr,a.dmrs_re,a.dmrs_im,a.dmrs_dbg,a.ws,a.tiling[OP_dmrs_gen]); },&slot_sum);
    time_kernel(a.stream,"re_map",[&]{ ACLRT_LAUNCH_KERNEL(re_map_kernel)(BD,a.stream,
        a.gu_re,a.gu_im,a.scatter_idx,a.gf_re,a.gf_im,a.ws,a.tiling[OP_re_map]); },&slot_sum);
    time_kernel(a.stream,"ofdm_mod",[&]{ ACLRT_LAUNCH_KERNEL(ofdm_mod_kernel)(BD,a.stream,
        a.gf_re,a.gf_im,a.w32r,a.w32i,a.w64r,a.w64i,a.twr,a.twi,a.ofdm_scr,
        a.ofdm_re,a.ofdm_im,a.ofdm_iq,a.ws,a.tiling[OP_ofdm_mod]); },&slot_sum);

    std::printf("------------------------------------------------------------------\n");
    std::printf(" tx_tb   sum(min) = %8.1f us\n", tb_sum);
    std::printf(" tx_slot sum(min) = %8.1f us   (budget 500 us)\n", slot_sum);
    std::printf("==================================================================\n");
}

}




using namespace airan_tx;
static double ms_since(std::chrono::high_resolution_clock::time_point t0){
    return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
}

int main(int, char**){
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));

    const char* dd = getenv("AIRAN_DATA_DIR"); if(!dd) dd=".";
    const bool prof = [](){ const char* e=getenv("TX_PROFILE"); return !e || e[0]!='0'; }();


    std::vector<uint8_t> tx_bits(SEG_B);
    { std::string p = std::string(dd)+"/tx_bits.bin"; size_t got=0;
      if(!ReadFile(p,got,tx_bits.data(),SEG_B)){
        std::fprintf(stderr,"[main] need %s  (ln -s ../ldpc_encode/data/golden/input.bin data/tx_bits.bin)\n",p.c_str());
        return 1; } }

    TxArena a; TxState st;
    if(const char* e=getenv("TX_CELL_ID")) st.cell_id=atoi(e);
    tx_arena_init(a, dd);


    std::printf("==================================================================\n");
    std::printf(" AI-RAN NPU  PUSCH SISO TX  —  基带生成\n");
    std::printf("==================================================================\n");
    std::printf(" cell_id   = %d   (驱动 DMRS + 扰码序列)\n", st.cell_id);
    std::printf(" TB 结构   = %u slots x %u CB   (LDPC BG, info %zu B)\n",
                SLOTS_PER_TB, 143u, (size_t)SEG_B);
    std::printf(" 采样率    = 61.44 MHz   调制 = QAM256\n");
    std::printf("------------------------------------------------------------------\n");

    std::vector<int16_t> iq((size_t)SLOTS_PER_TB * OFDM_IQ_B/2);

    auto t0=std::chrono::high_resolution_clock::now();
    tx_tb(a, tx_bits.data(), st);
    double tb_ms = ms_since(t0);

    std::vector<double> slot_ms; slot_ms.reserve(SLOTS_PER_TB);
    for(uint32_t s=0;s<SLOTS_PER_TB;++s){
        auto ts=std::chrono::high_resolution_clock::now();
        tx_slot(a, st, (int)s, iq.data()+(size_t)s*(OFDM_IQ_B/2));
        slot_ms.push_back(ms_since(ts));
    }

    std::string op=std::string(dd)+"/tx_iq.bin";
    WriteFile(op, iq.data(), (size_t)SLOTS_PER_TB*OFDM_IQ_B);
    std::printf("[tx] wrote %s  (%u slots x %zu B int16 IQ @61.44M)\n", op.c_str(), SLOTS_PER_TB, (size_t)OFDM_IQ_B);

    if(prof){
        tx_profile_kernels(a, st);
        double sum=0,mx=0; for(double m:slot_ms){ sum+=m; if(m>mx)mx=m; }
        std::printf("[prof] tx_tb      = %.3f ms\n", tb_ms);
        std::printf("[prof] tx_slot    mean=%.3f ms  max=%.3f ms  (budget 0.5ms)\n", sum/SLOTS_PER_TB, mx);
        std::printf("[prof] TB total   = %.3f ms\n", tb_ms+sum);
    }

    tx_arena_free(a);
    ACL_CHECK(aclrtResetDevice(0)); ACL_CHECK(aclFinalize());
    return 0;
}
