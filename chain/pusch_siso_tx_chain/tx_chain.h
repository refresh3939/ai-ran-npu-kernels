#pragma once
















#include <cstdint>
#include <cstddef>
#include <acl/acl.h>

namespace airan_tx {


constexpr uint32_t N_FFT        = 2048;
constexpr uint32_t N_SC_PAD     = 1664;
constexpr uint32_t N_SC_USED    = 1596;
constexpr uint32_t N_SYM        = 14;
constexpr uint32_t N_SAMP_SLOT  = 30720;
constexpr uint32_t SLOTS_PER_TB = 23;
constexpr uint32_t P_DFT        = 32;
constexpr uint32_t Q_DFT        = 64;

constexpr uint32_t N_CB         = 143;
constexpr uint32_t K_CB         = 8448;
constexpr uint32_t N_FULL       = 26112;
constexpr uint32_t N_CB_BUF     = 25344;


constexpr uint32_t N_STREAMS    = 8;
constexpr uint32_t N_SYM_PAD    = 19200;


constexpr size_t SEG_B        = (size_t)N_CB * K_CB;
constexpr size_t ENC_B        = (size_t)N_CB * N_CB_BUF;
constexpr size_t LAYOUT_B     = (size_t)SLOTS_PER_TB*N_STREAMS*N_SYM_PAD*2;
constexpr size_t SCR_B        = LAYOUT_B;
constexpr size_t SLOT_BITS_B  = (size_t)N_STREAMS*N_SYM_PAD*2;
constexpr size_t GRID_USED_B  = (size_t)N_SYM*N_SC_PAD*2;
constexpr size_t GRID_FFT_B   = (size_t)N_SYM*N_FFT*2;
constexpr size_t DMRS_B       = (size_t)2*896*2;
constexpr size_t DMRS_GMAT_B  = (size_t)31*1792*2;
constexpr size_t DMRS_G1_B    = (size_t)1792*2;
constexpr size_t OFDM_PLN_B   = (size_t)N_SAMP_SLOT*2;
constexpr size_t OFDM_IQ_B    = (size_t)N_SAMP_SLOT*2*2;
constexpr size_t W32_B        = (size_t)P_DFT*P_DFT*2;
constexpr size_t W64_B        = (size_t)Q_DFT*Q_DFT*2;
constexpr size_t TW_B         = (size_t)P_DFT*Q_DFT*2;
constexpr size_t IDX_B        = (size_t)N_FFT*4;
constexpr size_t MERGE_IDX_B  = (size_t)N_SC_PAD*4;
constexpr size_t CINIT_B      = (size_t)16*4;
constexpr size_t NSLOT_B      = 16u*4;


enum OpId {
    OP_ldpc_encode=0, OP_rate_match, OP_scramble, OP_qam256_mod,
    OP_dmrs_gen, OP_re_map, OP_ofdm_mod, OP_N
};


struct TxState {
    int      slot   = 0;
    uint16_t n_id   = 0;
    uint16_t n_scid = 0;
    int      cell_id = 1;
    int      n_cb   = (int)N_CB;
    int      locked = 1;
};


struct TxArena {
    aclrtStream stream = nullptr;
    uint8_t *ws = 0; size_t ws_bytes = 0;
    uint8_t *tiling[OP_N] = {0}; size_t tiling_bytes[OP_N] = {0};


    uint8_t *seg_bits=0;
    uint8_t *sh_a=0,*sh_bi=0,*sh_c=0,*sh_d=0,*ldpc_dbg=0;
    uint8_t *enc=0;
    uint8_t *layout=0;
    uint8_t *gold=0, *n_slot=0;
    uint8_t *scr=0;


    uint8_t *gu_re=0, *gu_im=0;
    uint8_t *dmrs_re=0, *dmrs_im=0;
    uint8_t *cinit=0, *dmrs_scr=0, *dmrs_dbg=0;
    uint8_t *dmrs_gmat=0, *dmrs_g1=0;
    uint8_t *merge_idx=0;
    uint8_t *scatter_idx=0;
    uint8_t *gf_re=0, *gf_im=0;
    uint8_t *w32r=0,*w32i=0,*w64r=0,*w64i=0,*twr=0,*twi=0,*ofdm_scr=0;
    uint8_t *ofdm_re=0, *ofdm_im=0;
    uint8_t *ofdm_iq=0;
};

void setup_tilings(TxArena& a);

int  tx_arena_init(TxArena& a, const char* data_dir);
void tx_arena_free(TxArena& a);

void tx_tb  (TxArena& a, const uint8_t* tx_bits_host, TxState& st);
void tx_slot(TxArena& a, TxState& st, int slot, int16_t* iq_out_host);

}
