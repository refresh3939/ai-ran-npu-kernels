#!/usr/bin/env python3
# ============================================================================
# qam256_demod_ref.py — 256-QAM Max-Log Soft Demapper golden 生成(V5.1-compatible)
#
# 自包含多 case:写到 <kernel_dir>/data/golden/case_N_xxx/(AIRAN_DATA_DIR 可覆盖)。
#   每个 case 一套 x_re/x_im/no_eff/output_llr/output_llr_sionna/N0。
#   同时建空的 data/ascend_output/case_N_xxx/ 供 kernel 落盘。
#
# 极性:标准 LLR>0 -> bit0(与 ldpc_decode lam>0<->bit0 一致, 去掉 RX_LB_NEG).
# 输出布局:严格 compact —— re_idx = data_sym_idx*N_SC_USED + sc,valid 紧排
#   [0, N_RE_DATA=19152),尾部 [N_RE_DATA, N_SYM_PAD) 全 0。bit-exact 目标不变。
# ============================================================================
import os, sys
import numpy as np

N_SYMBOL    = 14
DMRS_SYMS   = {2, 11}
N_DATA_SYM  = N_SYMBOL - len(DMRS_SYMS)  # 12
N_SC_USED   = 1596
N_SC_PAD    = 1664
N_RE_DATA   = N_DATA_SYM * N_SC_USED     # 19152
N_SYM_PAD   = ((N_RE_DATA + 127) // 128) * 128   # 19200
Q_M         = 8

DATA_SYM_TO_PHYS = [s for s in range(N_SYMBOL) if s not in DMRS_SYMS]
assert len(DATA_SYM_TO_PHYS) == N_DATA_SYM

Q_SCALE       = 32
LLR_CLIP_FX   = 2560
Y_SCALED_CLIP = 1000
D_256QAM      = 1.0 / np.sqrt(170.0)
PAM_LEVELS = np.array([-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15], dtype=np.float64)

def pam_bit_table():
    B = np.zeros((16, 4), dtype=np.int32)
    abs_s = np.abs(PAM_LEVELS)
    B[:, 0] = (PAM_LEVELS > 0).astype(np.int32)
    B[:, 1] = (abs_s < 8).astype(np.int32)
    B[:, 2] = (np.abs(abs_s - 8) < 4).astype(np.int32)
    B[:, 3] = (((abs_s % 8) >= 3) & ((abs_s % 8) <= 5)).astype(np.int32)
    return B
BIT_TABLE = pam_bit_table()

def brute_force_per_axis(u):
    dists = (u - PAM_LEVELS) ** 2
    L = np.zeros(4, dtype=np.float64)
    for b in range(4):
        m1 = BIT_TABLE[:, b] == 1
        L[b] = dists[m1].min() - dists[~m1].min()
    return L

def kernel_pipeline_per_re_8bit(x_re, x_im, no_eff):
    x_re_h=np.float16(x_re); x_im_h=np.float16(x_im); no_eff_h=np.float16(no_eff)
    D_X_QSCALE_h  = np.float16(D_256QAM * Q_SCALE)
    D2_X_QSCALE_h = np.float16(D_256QAM * D_256QAM * Q_SCALE)
    scale_y_h = D_X_QSCALE_h  / no_eff_h
    T_vec_h   = D2_X_QSCALE_h / no_eff_h
    y_I_h = x_re_h * scale_y_h; y_Q_h = x_im_h * scale_y_h
    def c16(v): return int(max(-32768, min(32767, np.rint(float(v)))))
    y_I_q=c16(y_I_h); y_Q_q=c16(y_Q_h); T_int=c16(T_vec_h)
    y_I_q=max(-Y_SCALED_CLIP,min(Y_SCALED_CLIP,y_I_q))
    y_Q_q=max(-Y_SCALED_CLIP,min(Y_SCALED_CLIP,y_Q_q))
    def piecewise(u_int, T):
        au=abs(u_int); LC=LLR_CLIP_FX
        def cs(x,l): return max(-l,min(l,x))
        sumC=0
        for k in (2,4,6,8,10,12,14): sumC += cs(u_int, k*T)
        L_b3=max(-LC,min(LC, 32*u_int - 4*sumC))
        L_b2=max(-LC,min(LC, 80*T-16*au-4*(min(au,2*T)+min(au,4*T)+min(au,6*T))
                 +4*(min(au,10*T)+min(au,12*T)+min(au,14*T))))
        L_b1=max(-LC,min(LC, -24*T-8*au+4*min(au,2*T)-4*min(au,6*T)+16*min(au,8*T)
                 -4*min(au,10*T)+4*min(au,14*T)))
        L_b0=max(-LC,min(LC, -8*T-4*au+8*min(au,4*T)-8*min(au,8*T)+8*min(au,12*T)))
        return [L_b3,L_b2,L_b1,L_b0]
    # 标准极性 LLR>0 -> bit0:对 piecewise(非标准 >0->bit1)整体取反(clip 对称, 精确)
    _v = np.array(piecewise(y_I_q,T_int)+piecewise(y_Q_q,T_int), dtype=np.int32)
    return (-_v).astype(np.int16)

def sionna_demap_per_re_8bit(x_re, x_im, no_eff):
    u_I=x_re/D_256QAM; u_Q=x_im/D_256QAM
    scale=(D_256QAM**2)/no_eff
    # 标准极性 LLR>0 -> bit0:brute_force = min_d(b1)-min_d(b0) 即标准, 不再取负
    return np.concatenate([brute_force_per_axis(u_I)*scale,
                           brute_force_per_axis(u_Q)*scale]).astype(np.float64)

# ---- 信道场景:返回 H_mag[N_SC_USED] ----
def H_flat(val):        return lambda: np.full(N_SC_USED, val, dtype=np.float64)
def H_freq_selective(): 
    def f():
        ph = 2*np.pi*np.arange(N_SC_USED)/200.0
        return np.clip(0.7 + 0.4*np.sin(ph) + 0.2*np.cos(ph*3), 0.15, 2.0)
    return f

# name, H_mag生成器, N0, seed —— 覆盖 AWGN/平坦衰落/频选 × 高低 SNR
CASES = [
    ("case_0_awgn_only",     H_flat(1.0),          0.0125, 0xA01),
    ("case_1_flat_fading",   H_flat(0.6),          0.0125, 0xA02),
    ("case_2_freq_select",   H_freq_selective(),   0.0125, 0xA03),
    ("case_3_low_snr",       H_freq_selective(),   0.0500, 0xA04),  # 高 N0 → 多裁剪
    ("case_4_high_snr",      H_freq_selective(),   0.0020, 0xA05),  # 低 N0 → 饱和 LLR
]

def gen_case(data_dir, name, H_gen, N0, seed):
    gdir = os.path.join(data_dir, "golden", name)
    odir = os.path.join(data_dir, "ascend_output", name)
    os.makedirs(gdir, exist_ok=True); os.makedirs(odir, exist_ok=True)
    rng = np.random.default_rng(seed)
    H_mag = H_gen()

    x_re   = np.zeros((N_SYMBOL, N_SC_PAD), dtype=np.float16)
    x_im   = np.zeros((N_SYMBOL, N_SC_PAD), dtype=np.float16)
    no_eff = np.ones((N_SYMBOL, N_SC_PAD), dtype=np.float16)   # padding/DMRS = 1.0
    for sym in range(N_SYMBOL):
        if sym in DMRS_SYMS: continue
        idx_I = rng.integers(0,16,size=N_SC_USED); idx_Q = rng.integers(0,16,size=N_SC_USED)
        s_I = PAM_LEVELS[idx_I]*D_256QAM; s_Q = PAM_LEVELS[idx_Q]*D_256QAM
        ne = N0/(H_mag**2)
        x_re[sym,:N_SC_USED]=(s_I+rng.normal(0,1,N_SC_USED)*np.sqrt(ne)).astype(np.float16)
        x_im[sym,:N_SC_USED]=(s_Q+rng.normal(0,1,N_SC_USED)*np.sqrt(ne)).astype(np.float16)
        no_eff[sym,:N_SC_USED]=ne.astype(np.float16)

    out_p = np.zeros((Q_M, N_SYM_PAD), dtype=np.int16)
    out_s = np.zeros((Q_M, N_SYM_PAD), dtype=np.float64)
    for di in range(N_DATA_SYM):
        sp = DATA_SYM_TO_PHYS[di]
        for sc in range(N_SC_USED):
            xr=float(x_re[sp,sc]); xi=float(x_im[sp,sc]); ne=float(no_eff[sp,sc])
            if ne<=0: ne=1e-6
            ri = di*N_SC_USED + sc
            out_p[:,ri]=kernel_pipeline_per_re_8bit(xr,xi,ne)
            out_s[:,ri]=sionna_demap_per_re_8bit(xr,xi,ne)
    out_sq = np.round(out_s*Q_SCALE).clip(-LLR_CLIP_FX,LLR_CLIP_FX).astype(np.int16)
    d = np.abs(out_p[:,:N_RE_DATA].astype(np.int32)-out_sq[:,:N_RE_DATA].astype(np.int32))

    x_re.tofile(os.path.join(gdir,"x_re.bin"))
    x_im.tofile(os.path.join(gdir,"x_im.bin"))
    no_eff.tofile(os.path.join(gdir,"no_eff.bin"))
    out_p.tofile(os.path.join(gdir,"output_llr.bin"))
    out_sq.tofile(os.path.join(gdir,"output_llr_sionna.bin"))
    np.array([N0],dtype=np.float32).tofile(os.path.join(gdir,"N0.bin"))
    print(f"  [{name}]  N0={N0:<7g} H={H_mag.min():.2f}~{H_mag.max():.2f}  "
          f"pipeline-vs-Sionna max_err={d.max()} LSB")

def main():
    kdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ref 在 scripts/ → 上一级=kernel 根
    data_dir = os.environ.get("AIRAN_DATA_DIR", os.path.join(kdir, "data"))
    print("=== qam256_demod golden (V5.1-compatible strict compact, multi-case) ===")
    print(f"  data_dir = {data_dir}")
    print(f"  layout   : golden/case_N_xxx/ + ascend_output/case_N_xxx/")
    print(f"  Output   : [{Q_M},{N_SYM_PAD}] int16, compact valid [0,{N_RE_DATA}), tail=0\n")
    for name,Hg,N0,seed in CASES:
        gen_case(data_dir, name, Hg, N0, seed)
    print(f"\n  {len(CASES)} cases written. Kernel must match output_llr.bin bit-exact per case.")

if __name__ == "__main__":
    main()
