"""
OFDMModulator reference for 5G NR (30 kHz SCS, 14-symbol slot).

ofdm_demod 的**严格逆变换** (exact structural inverse).

跑法:
    cd ~/AI-RAN-NPU/kernels/tx/ofdm_mod
    python3 scripts/ofdm_mod_ref.py

★ 对称设计 (layout 置换交给 re_map, 同 re_demap):
  · ofdm_demod:  时域(自然) → **原始 FFT** 3 阶段 → S4[b1,b2] (频域, 置换布局)
                 → re_demap 取自然子载波.  (demod 不归一化, 输出 ≈ 8000 量级)
  · ofdm_mod:    re_map 产出 S4-layout 频域 → **unitary 1/√N IDFT** (demod 三阶段反序 +
                 共轭权重, 每级带归一化) → 时域(自然) → CP 插入 → ×AIR_SCALE → int16.
                 与 demod 互为精确逆: 往返增益 N·(1/N)=1.

★ fp16 溢出纪律 (canonical 归一化, 不再靠输入幅度侥幸):
  · 1/√Q 折进 iw_dft64, 1/√P 折进 iw_dft32 ⇒ 合起来 = unitary 1/√N IDFT (保能量).
  · Phase A (IDFT-64) 峰 ≈ raw/Q, Phase C (IDFT-32) 峰 ≈ raw/N —— 中间永不溢出 fp16,
    单位功率输入下 cube 输出 ≈ 输入幅度, AIR_SCALE≈1600 远在 fp16 内.
  · 单一空口增益 AIR_SCALE: cube 出的 unitary IDFT (RMS~0.88) × AIR_SCALE → int16 tx_iq,
    填满 int16 且留 PAPR 余量. AIR_SCALE 是唯一带 magic 的常量, 物理意义明确.

Stage layout (demod 的反序; 所有指数 -j → +j; 归一化在权重):
  输入 S4[b1,b2] (频域)
  A. IDFT-64 over b2:  X1t[b1,a2] = Σ_b2 S4[b1,b2]·(conj(W64)/√Q)[b2,a2]  (cube, K=64)
  B. inv-twiddle:      X1[b1,a2]  = X1t[b1,a2]·conj(T)[b1,a2]              (vector)
  C. IDFT-32 over b1:  x_rs[a1,a2]= Σ_b1 (conj(W32)/√P)[a1,b1]·X1[b1,a2]  (cube, K=32)
  → x[64·a1 + a2] = x_rs[a1,a2]   (自然时序; 已是 unitary 1/√N IDFT, 无需再 scale)
  → × AIR_SCALE → int16 Q?  → CP 插入 → 30720 时域样点

W32 / W64 对称 (W[i,j]=W[j,i]) ⇒ conj(W)^T = conj(W), 权重加载与 demod 完全一致.
"""

import os
import inspect
import numpy as np
from pathlib import Path


_SCRIPT_DIR  = Path(__file__).resolve().parent
_KERNEL_DIR  = _SCRIPT_DIR.parent
DEFAULT_WEIGHTS_DIR = _KERNEL_DIR / "weights"
DEFAULT_GOLDEN_DIR  = _KERNEL_DIR / "data" / "golden"

N_FFT             = 2048
N_SYMBOL_PER_SLOT = 14
CP_LEN_FIRST      = 176
CP_LEN_OTHER      = 144
N_SAMPLE_PER_SLOT = 30720
L_MIN             = 0
Q_SCALE           = 256          # Q8.8 时域 scale (输入生成: demod 对 Q8.8 时域的输出)
AIR_SCALE         = 1600.0       # 单一空口增益: cube(unitary,RMS~0.88) × 此 → int16 tx_iq.
                                 # 按最坏 PAPR sym (26dB) 定: 0.88×~20(峰)×1600 ≈ 28k < 32767.
                                 # 1600<2048 ⇒ fp16 精确整数.
I16_MAX           =  32767
I16_MIN           = -32768

P = 32
Q = 64
assert P * Q == N_FFT

CP_LENS = np.array([CP_LEN_FIRST] + [CP_LEN_OTHER] * (N_SYMBOL_PER_SLOT - 1), dtype=np.int64)
assert int(CP_LENS.sum()) + N_SYMBOL_PER_SLOT * N_FFT == N_SAMPLE_PER_SLOT


def dft_matrix(n, dtype=np.complex64):
    k = np.arange(n)[:, None]; i = np.arange(n)[None, :]
    return np.exp(-2j * np.pi * k * i / n).astype(dtype)

def twiddle_pq(p, q, n_fft, dtype=np.complex64):
    k1 = np.arange(p)[:, None]; n2 = np.arange(q)[None, :]
    return np.exp(-2j * np.pi * k1 * n2 / n_fft).astype(dtype)

W_DFT_P  = dft_matrix(P)
W_DFT_Q  = dft_matrix(Q)
T_PQ     = twiddle_pq(P, Q, N_FFT)

# ★ IDFT 权重 = 共轭 + **unitary 归一化** (1/√P, 1/√Q). 合起来 = unitary 1/√N IDFT.
#   单位功率输入下 cube 输出 ≈ 输入幅度 (Parseval), AIR_SCALE 不需撑爆 fp16.
IW_DFT_P = (np.conj(W_DFT_P) / np.sqrt(P)).astype(np.complex64)   # 含 1/√P
IW_DFT_Q = (np.conj(W_DFT_Q) / np.sqrt(Q)).astype(np.complex64)   # 含 1/√Q
IT_PQ    = np.conj(T_PQ)                                  # twiddle 无归一化 (|T|=1)


def cp_insert(bodies):
    assert bodies.shape[-1] == N_FFT and bodies.shape[-2] == N_SYMBOL_PER_SLOT
    shape = bodies.shape[:-2]
    out = np.zeros((*shape, N_SAMPLE_PER_SLOT), dtype=np.complex64)
    offs = np.concatenate([[0], np.cumsum(CP_LENS + N_FFT)])
    for s in range(N_SYMBOL_PER_SLOT):
        start = int(offs[s] + CP_LENS[s]); cp = int(CP_LENS[s])
        out[..., start:start + N_FFT] = bodies[..., s, :]
        out[..., int(offs[s]):start]  = bodies[..., s, N_FFT - cp:]
    return out


def fft_s4_unitary(x_bodies):
    """自然时域 (..,14,N_FFT) → S4-layout **unitary FFT** (..,14,32,64), ×1/√N.
       生成单位功率频域栅格 (= re_map 喂给 mod 的 QAM 量级); 与 mod 的 unitary IDFT
       严格互逆, 往返 = 恒等 (能量守恒)."""
    shape = x_bodies.shape[:-1]
    x_rs = x_bodies.reshape(*shape, P, Q)
    X1   = np.einsum("ab,...bc->...ac", W_DFT_P, x_rs)       # DFT-32
    X1t  = X1 * T_PQ
    S4   = np.einsum("...ab,cb->...ac", X1t, W_DFT_Q)        # DFT-64 → [b1,b2]
    return (S4 / np.sqrt(N_FFT)).astype(np.complex64)        # unitary


def ifft_exact_inverse_complex(S4):
    """S4 (..,14,32,64) → (..,14,N_FFT) 自然时域. **unitary 1/√N IDFT** (归一化在权重)."""
    shape = S4.shape[:-2]
    X1t  = np.einsum("...ab,bc->...ac", S4, IW_DFT_Q)        # 含 1/√Q
    X1   = X1t * IT_PQ
    x_rs = np.einsum("ab,...bc->...ac", IW_DFT_P, X1)        # 含 1/√P → 总 1/√N
    return x_rs.reshape(*shape, N_FFT).astype(np.complex64)  # unitary (1/√N 已在权重)


def cmm(Ar, Ai, Br, Bi, ax):
    return (np.einsum(ax, Ar, Br) - np.einsum(ax, Ai, Bi),
            np.einsum(ax, Ar, Bi) + np.einsum(ax, Ai, Br))

def ifft_exact_inverse_4real(S4):
    shape = S4.shape[:-2]
    Sr = S4.real.astype(np.float32); Si = S4.imag.astype(np.float32)
    iW32r, iW32i = IW_DFT_P.real.astype(np.float32), IW_DFT_P.imag.astype(np.float32)
    iW64r, iW64i = IW_DFT_Q.real.astype(np.float32), IW_DFT_Q.imag.astype(np.float32)
    iTr,   iTi   = IT_PQ.real.astype(np.float32),    IT_PQ.imag.astype(np.float32)
    Ar, Ai = cmm(Sr, Si, iW64r, iW64i, "...ab,bc->...ac")    # 含 1/√Q
    Br = Ar * iTr - Ai * iTi
    Bi = Ar * iTi + Ai * iTr
    Cr, Ci = cmm(iW32r, iW32i, Br, Bi, "ab,...bc->...ac")    # 含 1/√P
    xr = Cr.reshape(*shape, N_FFT)
    xi = Ci.reshape(*shape, N_FFT)
    return (xr + 1j * xi).astype(np.complex64)               # unitary 1/√N IDFT


def ofdm_mod(S4_grid, use_4real_path=False, return_mid=False):
    mid = {}
    bodies = (ifft_exact_inverse_4real(S4_grid) if use_4real_path
              else ifft_exact_inverse_complex(S4_grid))
    mid["ifft_bodies"] = bodies
    x_time = cp_insert(bodies)
    mid["output"] = x_time
    if return_mid:
        return x_time, mid
    return x_time


def verify_vs_numpy_ifft(atol=1e-3):
    """mod 输出 = unitary IDFT = np.fft.ifft × √N."""
    rng = np.random.default_rng(1)
    S4 = ((rng.standard_normal((P, Q)) + 1j * rng.standard_normal((P, Q))) * 100).astype(np.complex64)
    x_mine = ifft_exact_inverse_complex(S4[None])[0]
    Xnat = np.zeros(N_FFT, np.complex64)
    for b1 in range(P):
        for b2 in range(Q):
            Xnat[b1 + 32 * b2] = S4[b1, b2]
    x_np = (np.fft.ifft(Xnat.astype(np.complex128)) * np.sqrt(N_FFT)).astype(np.complex64)  # unitary
    err = np.abs(x_mine - x_np).max()
    print(f"[exact-inv vs unitary ifft] max|err| = {err:.3e}  (tol={atol})")
    assert err < atol

def verify_4real_vs_complex(atol=1e-2):
    rng = np.random.default_rng(0)
    S4 = ((rng.standard_normal((3, P, Q)) + 1j * rng.standard_normal((3, P, Q))) * 5000).astype(np.complex64)
    err = np.abs(ifft_exact_inverse_complex(S4) - ifft_exact_inverse_4real(S4)).max()
    print(f"[4real vs complex        ] max|err| = {err:.3e}  (tol={atol})")
    assert err < atol

def verify_roundtrip(atol=1e-4):
    """unitary FFT → unitary IDFT = 恒等 (能量守恒, 单位功率)."""
    rng = np.random.default_rng(7)
    x = ((rng.standard_normal((N_SYMBOL_PER_SLOT, N_FFT)) +
          1j * rng.standard_normal((N_SYMBOL_PER_SLOT, N_FFT))) / np.sqrt(2)).astype(np.complex64)
    err = np.abs(x - ifft_exact_inverse_complex(fft_s4_unitary(x))).max()
    print(f"[roundtrip fft_u∘mod = I  ] max|err| = {err:.3e}  (tol={atol}, |x|RMS~{np.abs(x).std():.2f})")
    assert err < atol

def verify_against_sionna(S4_grid, x_bodies, atol=1.0):
    try:
        from sionna.phy.ofdm import OFDMModulator
    except ImportError as e:
        print(f"[sionna] not installed — skipping ({e})"); return None
    print(f"[sionna] signature: {inspect.signature(OFDMModulator.__init__)}")
    mod = OFDMModulator(cyclic_prefix_length=CP_LENS)
    Xnat = np.zeros((N_SYMBOL_PER_SLOT, N_FFT), np.complex64)
    for b1 in range(P):
        for b2 in range(Q):
            Xnat[:, b1 + 32 * b2] = S4_grid[:, b1, b2]
    Xc = np.fft.fftshift(Xnat, axes=-1).astype(np.complex64)
    try:
        import tensorflow as tf
        y = mod(tf.constant(Xc, dtype=tf.complex64)).numpy()
    except (ImportError, AttributeError, TypeError):
        try:
            import torch
            y = mod(torch.tensor(Xc, dtype=torch.complex64)).numpy()
        except ImportError as e:
            print(f"[sionna] no backend — skipping ({e})"); return None
    err = np.abs(y - ofdm_mod(S4_grid)).max()
    print(f"[sionna] max|err| vs ofdm_mod = {err:.3e}  (tol={atol})")
    assert err < atol; return y


def dump_goldens(S4_grid, x_bodies, out_root=None, weights_root=None):
    if out_root is None:     out_root = str(DEFAULT_GOLDEN_DIR)
    if weights_root is None: weights_root = str(DEFAULT_WEIGHTS_DIR)
    os.makedirs(out_root, exist_ok=True); os.makedirs(weights_root, exist_ok=True)

    # --- 归一化共轭 IDFT 权重 (fp16). W 对称 ⇒ conj(W)^T = conj(W) ---
    IW_DFT_P.real  .astype(np.float16).tofile(f"{weights_root}/iw_dft32_re.bin")   # conj(W32)/P
    IW_DFT_P.imag  .astype(np.float16).tofile(f"{weights_root}/iw_dft32_im.bin")
    IW_DFT_Q.real.T.astype(np.float16).tofile(f"{weights_root}/iw_dft64_re_T.bin") # conj(W64).T/Q
    IW_DFT_Q.imag.T.astype(np.float16).tofile(f"{weights_root}/iw_dft64_im_T.bin")
    IT_PQ   .real  .astype(np.float16).tofile(f"{weights_root}/itwiddle_pq_re.bin")
    IT_PQ   .imag  .astype(np.float16).tofile(f"{weights_root}/itwiddle_pq_im.bin")

    # --- 输入 = S4-layout 频域栅格 (separated re/im fp16) ---
    S4_flat = S4_grid.reshape(N_SYMBOL_PER_SLOT, N_FFT)
    re16 = S4_flat.real.astype(np.float16)
    im16 = S4_flat.imag.astype(np.float16)
    re16.tofile(f"{out_root}/in_re.bin")
    im16.tofile(f"{out_root}/in_im.bin")
    # golden 用 fp16 量化后的输入计算 (与 kernel 看到的输入一致 → 误差只剩 cube fp16 噪声)
    S4_q = (re16.astype(np.float32) + 1j * im16.astype(np.float32)).reshape(N_SYMBOL_PER_SLOT, P, Q)

    # --- 时域 golden: cube(1/N IDFT) × AIR_SCALE → int16 (含 CP, 自然时序) ---
    y, mid = ofdm_mod(S4_q, return_mid=True)
    re_i = np.clip(np.round(y.real * AIR_SCALE), I16_MIN, I16_MAX).astype(np.int16)
    im_i = np.clip(np.round(y.imag * AIR_SCALE), I16_MIN, I16_MAX).astype(np.int16)
    re_i.tofile(f"{out_root}/output_re.bin")
    im_i.tofile(f"{out_root}/output_im.bin")
    iq_i = np.empty(2 * re_i.size, dtype=np.int16); iq_i[0::2] = re_i; iq_i[1::2] = im_i
    iq_i.tofile(f"{out_root}/output_iq.bin")
    mid["ifft_bodies"].astype(np.complex64).tofile(f"{out_root}/ifft_bodies.bin")

    # Per-stage (kernel-aligned 4real, fp16 输入, 含归一化) — §6 板斧2
    Sr = S4_q.real.astype(np.float32); Si = S4_q.imag.astype(np.float32)
    iW32r, iW32i = IW_DFT_P.real.astype(np.float32), IW_DFT_P.imag.astype(np.float32)
    iW64r, iW64i = IW_DFT_Q.real.astype(np.float32), IW_DFT_Q.imag.astype(np.float32)
    iTr,   iTi   = IT_PQ.real.astype(np.float32),    IT_PQ.imag.astype(np.float32)
    Ar, Ai = cmm(Sr, Si, iW64r, iW64i, "nab,bc->nac")
    Br = Ar * iTr - Ai * iTi; Bi = Ar * iTi + Ai * iTr
    Cr, Ci = cmm(iW32r, iW32i, Br, Bi, "ab,nbc->nac")
    (Ar + 1j*Ai).astype(np.complex64).tofile(f"{out_root}/stageA_idft64.bin")
    (Br + 1j*Bi).astype(np.complex64).tofile(f"{out_root}/stageB_itwiddle.bin")
    (Cr + 1j*Ci).astype(np.complex64).tofile(f"{out_root}/stageC_idft32.bin")

    print(f"\n[dump] weights → {weights_root}/   goldens → {out_root}/")
    for name in ["iw_dft32_re","iw_dft32_im","iw_dft64_re_T","iw_dft64_im_T","itwiddle_pq_re","itwiddle_pq_im"]:
        print(f"  W  {name+'.bin':<22} {os.path.getsize(f'{weights_root}/{name}.bin')/1024:>6.1f} KiB")
    for name in ["in_re","in_im","output_re","output_im","output_iq","ifft_bodies","stageA_idft64","stageB_itwiddle","stageC_idft32"]:
        print(f"  G  {name+'.bin':<22} {os.path.getsize(f'{out_root}/{name}.bin')/1024:>6.1f} KiB")

    print(f"\n[dump] 量级 (fp16 max 65504, int16 max 32767):")
    t_rms = float(np.sqrt(np.mean(np.abs(y)**2))); t_peak = float(np.abs(y).max())
    papr  = 20*np.log10(t_peak/t_rms) if t_rms > 0 else 0.0
    rms_=lambda z: float(np.sqrt(np.mean(np.abs(z)**2)))
    print(f"  输入 S4 grid       RMS={rms_(S4_q):>8.3f}  peak={np.abs(S4_q).max():>8.3f}")
    print(f"  Phase A (IDFT64)                 peak={np.abs(Ar+1j*Ai).max():>8.3f}")
    print(f"  Phase C (cube,unitary IDFT) RMS={rms_(Cr+1j*Ci):>8.3f}  peak={np.abs(Cr+1j*Ci).max():>8.3f}")
    print(f"  时域 PAPR = {papr:.1f} dB  (峰/RMS = {t_peak/t_rms:.1f}x)")
    print(f"  tx_iq (×AIR_SCALE={AIR_SCALE:g})    RMS={rms_(re_i+1j*im_i):>8.0f}  peak={np.abs(iq_i).max():>8.0f}")
    sat = int(np.count_nonzero((np.abs(np.round(y.real*AIR_SCALE)) > I16_MAX) |
                               (np.abs(np.round(y.imag*AIR_SCALE)) > I16_MAX)))
    air_fit = (I16_MAX * 0.97) / t_peak
    print(f"  int16 饱和样点数 = {sat}  ({'OK' if sat == 0 else '⚠ 调小 AIR_SCALE'})")
    print(f"  本数据 PAPR 下填满 int16 的 AIR_SCALE ≈ {air_fit:.0f}  (当前 {AIR_SCALE:g})")


if __name__ == "__main__":
    print("=" * 60)
    print(" OFDM Modulator — 严格逆变换 (canonical unitary 1/√N) 数据生成 + 自检")
    print("=" * 60)
    print(f"  KERNEL_DIR  = {_KERNEL_DIR}\n")
    verify_vs_numpy_ifft()
    verify_4real_vs_complex()
    verify_roundtrip()
    # 单位功率时域 (RMS~0.88, 模拟 re_map QAM 占用 ~77% 子载波) → unitary FFT → 频域输入
    rng = np.random.default_rng(seed=54321)
    x_unit = ((rng.standard_normal((N_SYMBOL_PER_SLOT, N_FFT))
               + 1j * rng.standard_normal((N_SYMBOL_PER_SLOT, N_FFT)))
              * (0.88 / np.sqrt(2))).astype(np.complex64)
    S4_grid = fft_s4_unitary(x_unit)                     # 单位功率频域栅格 (= re_map 量级)
    verify_against_sionna(S4_grid, x_unit)
    dump_goldens(S4_grid, x_unit)
    print("\n" + "=" * 60)
    print(" ✅ 所有数据生成完成 — 跑 bash run.sh 即可测 kernel")
    print("=" * 60)
