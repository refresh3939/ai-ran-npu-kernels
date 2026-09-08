#pragma once













#include <cstdint>
#include <cstddef>
#include <acl/acl.h>

namespace airan_rx {




constexpr uint32_t N_FFT            = 2048;
constexpr uint32_t SCS_HZ           = 30000;
constexpr uint32_t FS_DATA          = 61440000;
constexpr uint32_t FS_SSB           = 7680000;
constexpr uint32_t DECIM            = 8;
constexpr uint32_t N_SC_USED        = 1596;
constexpr uint32_t N_SC_PAD         = 1664;
constexpr uint32_t N_SYM            = 14;
constexpr uint32_t N_SAMP_SLOT      = 30720;
constexpr uint32_t RE_DATA_SLOT     = 19152;
constexpr uint32_t SLOTS_PER_TB     = 23;






constexpr size_t CAPTURE_B      = 2457600u * 2;
constexpr size_t SSB768_B       = 307200u  * 2;
constexpr size_t PSS_REF_B = 16384u;
constexpr size_t PSS_TWID_B     = 1800u * 1024;
constexpr size_t PSS_SCR_B      = 2208768u;
constexpr size_t PSS_OUT_B      = 16u * 2;
constexpr size_t YPSS_B         = 256u * 2 * 2;
constexpr size_t PSSCFO_REF_B   = 3u * 256u * 2 * 2;
constexpr size_t PSSCFO_OUT_B   = 8u * 4;
constexpr size_t SSS_REF_B      = 504u * 1024;
constexpr size_t SSS_TWID_B     = 1584u;
constexpr size_t SSS_SCR_B      = 110u * 1024;
constexpr size_t SSS_OUT_B      = 8u * 4;
constexpr size_t SSBFFT_W16_B   = 16u * 16 * 2;
constexpr size_t SSBFFT_TW_B    = 256u * 2;
constexpr size_t SSBFFT_GIDX_B  = 144u * 4;
constexpr size_t SSBFFT_IN_B    = 1760u * 2;
constexpr size_t SSBFFT_DEROT_B = 144u * 2;
constexpr size_t R_DMRS_B       = 144u * 2;
constexpr size_t PBCH_D_B       = 4608u;
constexpr size_t PBCH_OUT_B     = 24u * 4;


constexpr size_t SLOT_IQ_B      = 61440u * 2;
constexpr size_t CFOCOMP_AB_B   = 61440u * 2;
constexpr size_t CFOEST_OUT_B   = 8u * 4;
constexpr size_t Y2048_B        = 14u * 2048 * 2;
constexpr size_t YD_B           = 47104u;
constexpr size_t XREF_SLOT_HALF = 2u * 832u;
constexpr size_t XREF_B         = (SLOTS_PER_TB * XREF_SLOT_HALF + 256u) * 2u;

constexpr size_t H_B            = 14u * N_SC_PAD * 2;
constexpr size_t EQ_B           = 14u * N_SC_PAD * 2;
constexpr size_t LLR_SLOT_PAD   = 19200u;
constexpr size_t LLR_IN_B       = SLOTS_PER_TB * 8u * LLR_SLOT_PAD * 2;

constexpr size_t CFOD_Y_B       = 14u * 1664 * 2;
constexpr size_t CFOD_X_SLOT_HALF = 2u * 896u;
constexpr size_t CFOD_X_B       = (SLOTS_PER_TB * CFOD_X_SLOT_HALF + 256u) * 2u;

constexpr size_t TREL_B         = 14u * 2;
constexpr size_t CFODMRS_OUT_B  = 64u;

constexpr size_t H220_B         = 2u * 220 * 2;
constexpr size_t DTSCALE_B      = 64u;
constexpr size_t TT_W_B         = 4096u * 2;
constexpr size_t TT_TW_B        = 4096u * 2;
constexpr size_t TT_SCR_B       = 256u * 1024;
constexpr size_t CIR_B          = 4096u * 2;
constexpr size_t DT_OUT_B       = 4u;


constexpr size_t CW_B           = 23u * 8 * 19200 * 2;
constexpr size_t SIGN_B         = CW_B;
constexpr size_t NSLOT_B        = 16u * 4;
constexpr size_t LAM_B          = 143u * 26112 * 2;
constexpr size_t TMPL_B         = 1u * 1024 * 1024;
constexpr size_t DUMMY_B        = 64u * 1024;
constexpr size_t BITS_B         = 143u * 8448;


constexpr size_t LDPC_PACKBC_B  = 46u * 19 * 2;
constexpr size_t LDPC_PACKS_B   = 46u * 19 * 2;
constexpr size_t LDPC_DEG_B     = 46u * 4;
constexpr size_t LDPC_EOFF_B    = 47u * 4;
constexpr size_t LDPC_PREV_B    = 143u * 316u * 384u;
constexpr size_t LDPC_LAMSCR_B  = 4u * 26112u * 2u;


constexpr size_t WS_B           = 1u * 1024 * 1024;
constexpr size_t TILING_B       = 256u;




struct SyncState {

    int   mu_t = 0, mu_t_data = 0;
    int   n_id_2 = 0, n_id_1 = 0, cell_id = 0, i_ssb = 0;
    float noise = 0.f;

    float cfo_int = 0.f, cfo_frac = 0.f;
    float cfo_cp = 0.f;
    float cfo_resid = 0.f;
    float cfo_total = 0.f;

    int   delta_T = 0;
    int   locked = 0;
};




enum OpId {
    OP_decimate=0, OP_pss_correlator, OP_pss_cfo_estimator, OP_sss_correlator,
    OP_ssb_fft, OP_pbch_dmrs_correlator, OP_cfo_estimate, OP_cfo_compensate,
    OP_ofdm_demod, OP_re_demap, OP_channel_est_ls, OP_equalize, OP_qam256_demod,
    OP_cfo_dmrs, OP_timing_tracker, OP_descramble,
    OP_rate_dematch, OP_ldpc_decode, OP_win_slice, OP_N
};






struct RxArena {

    aclrtStream stream = nullptr;
    uint8_t *ws = 0;
    size_t   ws_bytes = 0;
    uint8_t *tiling[OP_N] = {0};
    size_t   tiling_bytes[OP_N] = {0};


    uint8_t *capture=0, *ssb768=0, *fir_taps=0;
    uint8_t *pss_ref=0, *pss_twid=0, *pss_scr=0, *pss_out=0;
    uint8_t *ypss=0, *psscfo_ref=0, *psscfo_scr=0, *psscfo_out=0;
    uint8_t *sss_re=0, *sss_im=0, *sss_twr=0, *sss_twi=0, *sss_scr=0, *sss_out=0;
    uint8_t *ssbfft_w16r=0, *ssbfft_w16i=0, *ssbfft_twr=0, *ssbfft_twi=0, *ssbfft_gidx=0, *ssbfft_dr=0, *ssbfft_di=0, *ssbfft_scr=0;
    uint8_t *ssbfft_gidx_nu[4]={0,0,0,0};
    uint8_t *ssbfft_in=0;
    uint8_t *r_re=0, *r_im=0;
    uint8_t *pbch_dre=0, *pbch_dim=0, *pbch_dimn=0, *pbch_out=0;


    uint8_t *rd_ptr=0;
    uint8_t *slot_iq=0, *slot_iq_swap=0, *cc_a2=0, *cc_b2=0, *slot_iq_d=0;
    uint8_t *cfoest_scr=0, *cfoest_out=0;
    uint8_t *ofdm_w32r=0,*ofdm_w32i=0,*ofdm_w64r=0,*ofdm_w64i=0,*ofdm_twr=0,*ofdm_twi=0,*ofdm_scr=0;
    uint8_t *y_re=0, *y_im=0;
    uint8_t *re_idx=0, *yd_re=0, *yd_im=0;
    uint8_t *xref_re=0, *xref_im=0, *ce_weave=0, *h_re=0, *h_im=0;
    uint8_t *eq_n0=0;
    uint8_t *x_re=0, *x_im=0, *no_eff=0;
    uint8_t *llr_in=0;

    uint8_t *cfod_yre=0, *cfod_yim=0, *cfod_xre=0, *cfod_xim=0, *trel=0, *cfodmrs_scr=0, *cfodmrs_out=0;

    uint8_t *h220_re=0, *h220_im=0, *tt_wr=0, *tt_wi=0, *tt_twr=0, *tt_twi=0, *tt_dtscale=0, *tt_scr=0, *cir_re=0, *cir_im=0, *dt_out=0;


    uint8_t *cw=0, *sign=0, *n_slot=0, *cw_ds=0;
    uint8_t *rd_tmpl=0, *rd_dummy=0, *lam=0;
    uint8_t *ldpc_packbc=0, *ldpc_packs=0, *ldpc_deg=0, *ldpc_eoff=0, *ldpc_prev=0, *ldpc_lamscr=0;
    uint8_t *bits=0, *lam_out=0;
};


void setup_tilings(RxArena& a);




int  rx_arena_init(RxArena& a, const char* data_dir);
void rx_arena_free(RxArena& a);




void rx_acquire(RxArena& a, const int16_t* capture_host, SyncState& st);
void rx_slot   (RxArena& a, const int16_t* stream_host, SyncState& st, int slot);
void rx_tb     (RxArena& a, SyncState& st, const uint8_t* tx_bits_known, double* ber);

}
