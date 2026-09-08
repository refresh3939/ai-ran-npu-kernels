"""
timing_tracker_ref.py — numpy reference + weights/golden generator

算法: OAI CIR-argmax (依据 arXiv 2508.19736)
  1. H_avg[k]   = (H_ls[0,k] + H_ls[1,k]) / 2,   k ∈ [0, K=220)
  2. H_pad[n]   = H_avg[n] if n<K else 0,        n ∈ [0, N=4096)
  3. h[n]       = (1/N) · Σ_k H_pad[k] · exp(+j·2π·k·n/N)         ← IDFT
  4. p[n]       = |h[n]|²
  5. n_peak     = argmax(p[n])  在 tau ∈ [-W, +W) (W=72) 内
  6. n_fine     = n_peak + 0.5·(p[i-1]-p[i+1]) / (p[i-1]-2·p[i]+p[i+1])
  7. delta_T    = -(n_fine 映回 [-N/2, N/2))                       ← 符号约定

实现: 全 4096-pt Cooley-Tukey IDFT, mixed-radix 64×64
  - h_cir[64,64] = (W64 @ H_reshape[64,64] ⊙ T_inv) @ W64^T  (复 IDFT)
  - h[n] = h_cir 按 n = 64·k_p + k_q reshape

weights (host → device GM):
  - w_idft64_re.bin  [64,64] fp16  ← W64[a,b] = exp(+j·2π·ab/64) 的实部
  - w_idft64_im.bin  [64,64] fp16  ←                          的虚部
  - w_idft64_re_T.bin [64,64] fp16 ← W64^T (列主序, 给 Phase 3 用)
  - w_idft64_im_T.bin [64,64] fp16
  - twiddle_inv_re.bin [64,64] fp16 ← T_inv[k_p,q] = exp(+j·2π·k_p·q/4096) 的实部
  - twiddle_inv_im.bin [64,64] fp16

input/golden (host → device GM):
  - h_re.bin         fp16 [14, 1664]  channel_est 标准契约 symbol-major
  - h_im.bin         fp16 [14, 1664]  channel_est 标准契约 symbol-major
                     真信号写在 sym 2 / sym 11 的 SC [0:220], 其他 sym 全 0
  - golden_delta_T.bin fp32 [1]
  - golden_cir.bin     fp32 [4096, 2] (re, im)  仅 verify 用
"""
import numpy as np
from pathlib import Path
import sys


# === 系统常量 (跟 LAYER12_DEV_GUIDE.md / kernel header 对齐) ===
N_FFT    = 4096      # 系统 FFT 长度
K_DMRS   = 220       # DMRS 子载波数
N_SYM    = 2         # 2 个 DMRS 符号 (sym 2 + sym 11)
P        = 64        # Cooley-Tukey radix-1
Q        = 64        # Cooley-Tukey radix-2 (P*Q = N_FFT)
W_SEARCH = 72        # 搜索窗 ±W (CP/4)

# 输入量化 (与上游 cfo_dmrs 一致, cint16 表示 fp16)
INPUT_SCALE = 1.0 / 256.0     # cint16 → fp16


# ─────────────────────────────────────────────────────────────────
#  权重生成
# ─────────────────────────────────────────────────────────────────
def gen_w_idft64():
    """W64[a,b] = exp(+j·2π·a·b/64),  IDFT 矩阵.
    
    DFT: W_dft[a,b] = exp(-j·2π·ab/N)
    IDFT: W_idft = conj(W_dft),  即 exp(+j·2π·ab/N)
    """
    a = np.arange(P)[:, None]
    b = np.arange(P)[None, :]
    W = np.exp(+1j * 2*np.pi * a * b / P)
    return W.astype(np.complex128)


def gen_twiddle_inv():
    """T_inv[k_p, q] = exp(+j·2π·k_p·q/N_FFT),  IDFT twiddle."""
    k_p = np.arange(P)[:, None]
    q   = np.arange(Q)[None, :]
    T = np.exp(+1j * 2*np.pi * k_p * q / N_FFT)
    return T.astype(np.complex128)


# ─────────────────────────────────────────────────────────────────
#  numpy ref (3 种实现, 互相 cross-check)
# ─────────────────────────────────────────────────────────────────
def idft_direct(H_pad, dtype=np.complex128):
    """直接 IDFT (fp64), ground truth.
    
    h[n] = (1/N) · Σ_k H_pad[k] · exp(+j·2π·kn/N)
    """
    return np.fft.ifft(H_pad).astype(dtype)


def idft_mixed_radix(H_pad, P=P, Q=Q, dtype=np.complex128):
    """Cooley-Tukey IDFT,严格按经典 mixed-radix 推导.
    
    N = N_1 × N_2,在这里 N_1 = P, N_2 = Q.
    
    索引拆法 (严格按教科书 / Wikipedia Cooley-Tukey general radix):
        n = N_2·n_1 + n_2,    n_1 ∈ [0, N_1=P), n_2 ∈ [0, N_2=Q)  [n_1 慢变]
        k = N_1·k_2 + k_1,    k_1 ∈ [0, N_1=P), k_2 ∈ [0, N_2=Q)  [k_2 慢变]
    
    展开 W_N^(+nk) (IDFT, +j 符号):
        = W_{N_1}^(n_1·k_1) · W_N^(n_2·k_1) · W_{N_2}^(n_2·k_2)
    
    3-stage 算法:
        H_rs[n_1, n_2] = H_pad[N_2·n_1 + n_2]                ← 直接 reshape(P, Q)
        X1[k_1, n_2]   = Σ_{n_1} W_P^(n_1·k_1) · H_rs[n_1, n_2]    = W_P @ H_rs
        Tw[k_1, n_2]   = T[k_1, n_2] · X1[k_1, n_2],  T[a,b]=W_N^(a·b)
        h_cir[k_1, k_2] = Σ_{n_2} Tw[k_1, n_2] · W_Q^(n_2·k_2)    = Tw @ W_Q
    
    输出 flatten:
        h[k] = h_cir[k_1, k_2] / N, 其中 k = N_1·k_2 + k_1  [k_2 慢变!]
        ⇒ h = h_cir.T.flatten() / N      (即先转置, k_2 在外 k_1 在内)
    """
    N = P * Q
    # n = Q·n_1 + n_2  →  H_rs[n_1, n_2] = H_pad[Q·n_1 + n_2]
    # numpy 直接 reshape(P, Q) 就是这个语义 (P 是慢变维度)
    H_rs = H_pad.reshape(P, Q).astype(dtype)
    
    W_P = gen_w_idft64()                            # [P, P], W_P[a, b] = exp(+j·2π·ab/P)
    W_Q = gen_w_idft64()                            # [Q, Q]
    T   = gen_twiddle_inv()                         # [P, Q], T[k_1, n_2] = exp(+j·2π·k_1·n_2/N)
    
    # Stage 1: X1[k_1, n_2] = Σ_{n_1} W_P[k_1, n_1] · H_rs[n_1, n_2]  =  W_P @ H_rs
    X1 = W_P @ H_rs                                 # [P, Q]
    
    # Stage 2: Tw[k_1, n_2] = X1 ⊙ T
    Tw = X1 * T                                     # [P, Q]
    
    # Stage 3: h_cir[k_1, k_2] = Σ_{n_2} Tw[k_1, n_2] · W_Q[n_2, k_2]  =  Tw @ W_Q
    h_cir = Tw @ W_Q                                # [P, Q],  h_cir[k_1, k_2]
    
    # Flatten: k = P·k_2 + k_1,  k_2 慢变 ⇒ 先转置后 flatten
    h = h_cir.T.reshape(N) / N
    
    return h


def timing_tracker_ref(H_ls,
                       N_fft=N_FFT,
                       search_window=W_SEARCH,
                       use_mixed_radix=False):
    """完整 timing_tracker, 返回 (delta_T, h_cir).
    
    Args:
        H_ls:           [2, K_DMRS] complex (cfo_dmrs 上游输出)
        N_fft:          4096
        search_window:  ±W
        use_mixed_radix: True 用 Cooley-Tukey 64×64 (跟 kernel 一致),
                         False 用直接 IDFT (fp64 ground truth)
    
    Returns:
        delta_T:  float
        h_cir:    [N_fft] complex (供 debug)
    """
    # Step 1-2: 平均 + 零填充
    H_avg = H_ls.mean(axis=0)                          # [K]
    H_pad = np.zeros(N_fft, dtype=np.complex128)
    H_pad[:H_ls.shape[1]] = H_avg
    
    # Step 3: IDFT
    if use_mixed_radix:
        h = idft_mixed_radix(H_pad)
    else:
        h = idft_direct(H_pad)
    
    # Step 4: 功率
    p = (h.real**2 + h.imag**2)
    
    # Step 5: argmax 在搜索窗
    W = search_window
    tau_indices = np.concatenate([
        np.arange(N_fft - W, N_fft),    # 负 tau
        np.arange(0, W)                 # 正 tau
    ])
    p_window = p[tau_indices]
    tau_values = np.concatenate([
        np.arange(-W, 0),
        np.arange(0, W)
    ]).astype(np.float64)
    
    idx = int(np.argmax(p_window))
    
    # Step 6: 抛物线插值
    if 0 < idx < 2*W - 1:
        y_m = p_window[idx - 1]
        y_0 = p_window[idx]
        y_p = p_window[idx + 1]
        denom = (y_m - 2*y_0 + y_p)
        if abs(denom) > 1e-12 and y_0 > y_m and y_0 > y_p:
            offset = 0.5 * (y_m - y_p) / denom
        else:
            offset = 0.0
    else:
        offset = 0.0
    
    # Step 7: 符号 (IDFT 符号约定)
    delta_T = -(tau_values[idx] + offset)
    
    return float(delta_T), h


# ─────────────────────────────────────────────────────────────────
#  H_ls 测试输入生成
# ─────────────────────────────────────────────────────────────────
def gen_test_H_ls(delta_T_true=0.5, snr_db=30, K=K_DMRS, n_sym=N_SYM, seed=42):
    """生成 H_ls 测试 case: 单径平坦 + timing offset + AWGN.
    
    Returns:
        H_ls:        [n_sym, K] complex128
        H_ls_cint16: [n_sym, K, 2] int16 interleaved [I,Q] (上游 cint16 格式)
    """
    rng = np.random.default_rng(seed)
    
    # 单径平坦信道: 每符号一个全局相位 (模拟未补 CFO 残差)
    phase_per_sym = rng.uniform(-np.pi, np.pi, size=(n_sym, 1))
    H = np.ones((n_sym, K), dtype=np.complex128) * np.exp(1j * phase_per_sym)
    
    # 加 timing offset (频域线性相位)
    k = np.arange(K)
    H_off = H * np.exp(+1j * 2*np.pi * k * delta_T_true / N_FFT)
    
    # AWGN
    sig_pow = 1.0
    noise_pow = sig_pow / (10**(snr_db/10))
    noise = np.sqrt(noise_pow/2) * (
        rng.standard_normal(H_off.shape) + 1j*rng.standard_normal(H_off.shape)
    )
    H_noisy = H_off + noise
    
    # 量化到 cint16: 实部虚部各乘 256 取整, clip 到 int16 范围
    re_q = np.clip(np.round(H_noisy.real * 256), -32768, 32767).astype(np.int16)
    im_q = np.clip(np.round(H_noisy.imag * 256), -32768, 32767).astype(np.int16)
    
    # interleaved [I,Q,I,Q,...]: shape [n_sym, K, 2]
    H_ls_cint16 = np.stack([re_q, im_q], axis=-1).astype(np.int16)
    
    # 反量化得到 fp16-equivalent 的 H_ls (供 numpy ref 计算 golden)
    H_ls_fp16 = (re_q.astype(np.float64) + 1j*im_q.astype(np.float64)) * INPUT_SCALE
    H_ls_fp16 = (H_ls_fp16.real.astype(np.float16).astype(np.float64) +
                 1j*H_ls_fp16.imag.astype(np.float16).astype(np.float64))
    
    return H_ls_fp16, H_ls_cint16


def gen_test_H_ls_multipath(delta_T_true=0.0, snr_db=30,
                             K=K_DMRS, n_sym=N_SYM, seed=42,
                             taps=((0, 1.0), (8, 0.5), (15, 0.3))):
    """多径信道: 多个 tap 各自 delay + amplitude.
    
    频域响应: H[k] = Σ_t amp_t · exp(+j·2π·k·tap_t/N_FFT) · exp(+j·2π·k·ΔT/N_FFT)
    
    Args:
        delta_T_true: 全局 timing offset (samples)
        taps: ((delay_samples, complex_amp), ...) 各 tap 的延迟和幅度
              真信道质心 ≈ Σ |amp|²·delay / Σ |amp|²
    """
    rng = np.random.default_rng(seed)
    
    # 频域信道 = 各 tap 的频域响应相加
    k = np.arange(K)
    H_freq = np.zeros(K, dtype=np.complex128)
    for tap_delay, amp in taps:
        H_freq += amp * np.exp(+1j * 2*np.pi * k * tap_delay / N_FFT)
    
    # 加全局 ΔT (timing offset)
    H_freq *= np.exp(+1j * 2*np.pi * k * delta_T_true / N_FFT)
    
    # 每符号一个未知相位 (CFO 残差)
    phase_per_sym = rng.uniform(-np.pi, np.pi, size=(n_sym, 1))
    H = np.broadcast_to(H_freq, (n_sym, K)) * np.exp(1j * phase_per_sym)
    
    # AWGN
    sig_pow = np.mean(np.abs(H)**2)
    noise_pow = sig_pow / (10**(snr_db/10))
    noise = np.sqrt(noise_pow/2) * (
        rng.standard_normal(H.shape) + 1j*rng.standard_normal(H.shape)
    )
    H_noisy = H + noise
    
    # 量化到 cint16
    re_q = np.clip(np.round(H_noisy.real * 256), -32768, 32767).astype(np.int16)
    im_q = np.clip(np.round(H_noisy.imag * 256), -32768, 32767).astype(np.int16)
    H_ls_cint16 = np.stack([re_q, im_q], axis=-1).astype(np.int16)
    
    H_ls_fp16 = (re_q.astype(np.float64) + 1j*im_q.astype(np.float64)) * INPUT_SCALE
    H_ls_fp16 = (H_ls_fp16.real.astype(np.float16).astype(np.float64) +
                 1j*H_ls_fp16.imag.astype(np.float16).astype(np.float64))
    
    return H_ls_fp16, H_ls_cint16


# ─────────────────────────────────────────────────────────────────
#  Main: 生成所有文件
# ─────────────────────────────────────────────────────────────────
def write_fp16(path, arr):
    arr_fp16 = arr.astype(np.float16)
    arr_fp16.tofile(str(path))
    return arr_fp16


def write_fp32(path, arr):
    arr_fp32 = arr.astype(np.float32)
    arr_fp32.tofile(str(path))
    return arr_fp32


def write_int16(path, arr):
    arr_int16 = arr.astype(np.int16)
    arr_int16.tofile(str(path))
    return arr_int16


# ─────────────────────────────────────────────────────────────────
#  Pack H_ls [2, 220] complex → [14, 1664] symbol-major fp16 separate
#  ★ 标准 physical-grid 契约 (跟 channel_est_ls 输出对齐)
#  sym 2 / sym 11 的 SC [0:220] 是真信号 (REF_OFFSET=0), 其余 0
# ─────────────────────────────────────────────────────────────────
N_SYM_TOTAL = 14
N_SC_PAD    = 1664
DMRS_SYM_0  = 2
DMRS_SYM_1  = 11
REF_OFFSET  = 0

def pack_h_ls_to_physical_grid(H_ls_fp16):
    """[2, 220] complex → [14, 1664] re/im separate fp16
    Returns: (h_re_grid, h_im_grid) each [14, 1664] float16
    """
    n_sym, K = H_ls_fp16.shape  # [2, 220]
    assert n_sym == 2 and K == K_DMRS

    h_re = np.zeros((N_SYM_TOTAL, N_SC_PAD), dtype=np.float16)
    h_im = np.zeros((N_SYM_TOTAL, N_SC_PAD), dtype=np.float16)

    # H_ls[0] → sym 2, H_ls[1] → sym 11
    sym_indices = [DMRS_SYM_0, DMRS_SYM_1]
    for sym_idx, dmrs_sym in enumerate(sym_indices):
        h_re[dmrs_sym, REF_OFFSET:REF_OFFSET + K] = H_ls_fp16[sym_idx].real.astype(np.float16)
        h_im[dmrs_sym, REF_OFFSET:REF_OFFSET + K] = H_ls_fp16[sym_idx].imag.astype(np.float16)
    return h_re, h_im


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    weights_dir = project_dir / 'weights'
    golden_dir  = project_dir / 'data' / 'golden'
    weights_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[ref] project: {project_dir}")
    print(f"[ref] weights: {weights_dir}")
    print(f"[ref] golden:  {golden_dir}")
    
    # ── 1. 权重 ──────────────────────────────────────────────────
    W64 = gen_w_idft64()                            # [P, P] complex
    T_inv = gen_twiddle_inv()                       # [P, Q] complex
    
    write_fp16(weights_dir / 'w_idft64_re.bin', W64.real)
    write_fp16(weights_dir / 'w_idft64_im.bin', W64.imag)
    write_fp16(weights_dir / 'w_idft64_re_T.bin', W64.real.T)
    write_fp16(weights_dir / 'w_idft64_im_T.bin', W64.imag.T)
    write_fp16(weights_dir / 'twiddle_inv_re.bin', T_inv.real)
    write_fp16(weights_dir / 'twiddle_inv_im.bin', T_inv.imag)
    
    print(f"[ref] wrote 6 weight files (W64 64x64, twiddle 64x64)")
    
    # ── 2. 多 case 测试集 ─────────────────────────────────────────
    # case_N 子目录, 每个 case 一个 h_re.bin/h_im.bin + golden_cir.bin + golden_delta_T.bin
    # case_0 也仍然在 data/golden/ 根目录有备份 (向后兼容 MVP v1 verify)
    
    cases = [
        # (case_id, ΔT_true, SNR_dB, channel,    seed, description)
        (0, +0.5,  40, 'single',    42, "MVP v1 baseline (+0.5 sample, 40dB)"),
        (1,  0.0,  40, 'single',    43, "ΔT=0 (峰应在 tau=0)"),
        (2, -2.0,  40, 'single',    44, "负 ΔT (峰应在 tau=+2)"),
        (3, +5.0,  40, 'single',    45, "整数 ΔT"),
        (4, +0.5,  20, 'single',    46, "中 SNR (20 dB)"),
        (5, +0.5,  10, 'single',    47, "低 SNR (10 dB)"),
        (6,  0.0,  30, 'multipath', 48, "多径 (信道质心 delay 非零)"),
        (7, +50.0, 40, 'single',    49, "大整数 ΔT (50, 接近搜索窗 ±72)"),
    ]
    
    summary = []
    for case_id, dt_true, snr_db, ch, seed, desc in cases:
        case_dir = golden_dir / f'case_{case_id}'
        case_dir.mkdir(parents=True, exist_ok=True)
        
        H_ls_fp16, H_ls_cint16 = gen_test_H_ls(
            dt_true, snr_db, seed=seed,
        ) if ch == 'single' else gen_test_H_ls_multipath(
            dt_true, snr_db, seed=seed
        )
        
        # ★ 写 [14, 1664] fp16 separate (channel_est 标准契约)
        h_re_grid, h_im_grid = pack_h_ls_to_physical_grid(H_ls_fp16)
        write_fp16(case_dir / 'h_re.bin', h_re_grid)
        write_fp16(case_dir / 'h_im.bin', h_im_grid)
        
        # Golden: 用 fp16 量化后的 H_ls 跑 ref (kernel 内取 sym2/11 SC[0:220] 跟这里完全一致)
        # ★ 关键: kernel Phase 0 拷 h_re_grid[2/11, 0:220] + h_im_grid[2/11, 0:220]
        #    跟 ref 的 H_ls_fp16[0/1, 0:220] 完全相同的数据 (因为 pack_h_ls 把 H_ls_fp16 直接放进去)
        dt_fp64, h_cir_fp64 = timing_tracker_ref(H_ls_fp16, use_mixed_radix=False)
        
        # ★ Cross-check: case_0 验证 mixed-radix ≡ direct
        if case_id == 0:
            dt_mr, h_cir_mr = timing_tracker_ref(H_ls_fp16, use_mixed_radix=True)
            h_err = np.abs(h_cir_fp64 - h_cir_mr).max()
            dt_err = abs(dt_fp64 - dt_mr)
            assert h_err < 1e-10 and dt_err < 1e-10, f"mixed-radix mismatch! h={h_err}, dt={dt_err}"
            print(f"[ref] case_0 mixed-radix cross-check: CIR err {h_err:.1e}, ΔT err {dt_err:.1e} ✓")
        
        # 写 case golden
        write_fp32(case_dir / 'golden_delta_T.bin', np.array([dt_fp64]))
        cir_pairs = np.stack([h_cir_fp64.real, h_cir_fp64.imag], axis=-1)
        write_fp32(case_dir / 'golden_cir.bin', cir_pairs.flatten())
        
        # case_0 兼容根目录 (旧 verify_result.py 入口)
        if case_id == 0:
            write_fp16(golden_dir / 'h_re.bin', h_re_grid)
            write_fp16(golden_dir / 'h_im.bin', h_im_grid)
            write_fp32(golden_dir / 'golden_delta_T.bin', np.array([dt_fp64]))
            write_fp32(golden_dir / 'golden_cir.bin', cir_pairs.flatten())
        
        summary.append((case_id, dt_true, snr_db, ch, dt_fp64, desc))
        print(f"  case_{case_id}: ΔT_true={dt_true:+6.2f} SNR={snr_db:2d}dB {ch:<10s} "
              f"→ ΔT_golden={dt_fp64:+7.4f}   ({desc})")
    
    print(f"\n[ref] {len(cases)} cases generated.")
    print(f"[ref] DONE.")


if __name__ == '__main__':
    main()