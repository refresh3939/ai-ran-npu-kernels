"""
OFDMDemodulator reference for 5G NR (30 kHz SCS, 14-symbol slot).

跑这个脚本会生成所有 weights + per-stage goldens.

跑法:
    cd ~/AI-RAN-NPU/kernels/rx/ofdm_demod
    python3 scripts/ofdm_demod_ref.py

输出位置 (跟 kernel main.cpp 的 AIRAN_DATA_DIR 期望一致):
    ~/AI-RAN-NPU/weights/rx/ofdm_demod/          (DFT 矩阵 + twiddle, fp16)
    ~/AI-RAN-NPU/data/golden/rx/ofdm_demod/      (per-stage 中间结果)

Algorithm: Mixed-radix 32×64 Cooley-Tukey decomposition of FFT-2048.
This matches the planned Ascend kernel stage-by-stage so bit-exact
phase dumping (SKILL.md debug 板斧 2) is possible.

Stage layout:
  0. CP removal — symbol-wise gather over time samples
  1. reshape (2048,) → (32, 64) row-major: x_rs[n1, n2] = x[64·n1 + n2]
  2. Unitary DFT-32 along axis 0 (batched over 64 columns):
       X1 = (W32/sqrt(32)) @ x_rs
  3. Twiddle: X1_tw[k1, n2] = X1[k1, n2] · exp(-j·2π·n2·k1 / 2048)
  4. Unitary DFT-64 along axis 1 (batched over 32 rows):
       X2 = X1_tw @ (W64/sqrt(64)).T
  5. Reshape back: X[k1 + 32·k2] = X2[k1, k2]  (column-major flatten)
  6. No separate normalization op: 1/sqrt(2048) is already folded into
       the two matrix weights, matching ofdm_mod and Sionna's unitary FFT.
  7. Phase comp: X[k] *= exp(-j·2π·k·l_min/N)  (identity when l_min=0)
  8. fftshift (DC → index 1024, zero cost if output stride swaps halves)

Kernel-alignment: every complex matmul is lowered to 4 real matmuls.
Python provides BOTH paths and asserts they agree.
"""

import os
import inspect
import numpy as np
from pathlib import Path


# ============================================================
# 默认路径 — scripts/ 在 kernel 目录下, 所以 kernel 目录 = 此脚本上一层
# ============================================================
_SCRIPT_DIR  = Path(__file__).resolve().parent           # kernels/rx/ofdm_demod/scripts
_KERNEL_DIR  = _SCRIPT_DIR.parent                         # kernels/rx/ofdm_demod

DEFAULT_WEIGHTS_DIR = _KERNEL_DIR / "weights"
DEFAULT_GOLDEN_DIR  = _KERNEL_DIR / "data" / "golden"


# ============================================================
# 5G NR slot constants (LOCKED — don't edit without kernel update)
# ============================================================
N_FFT             = 2048
N_SYMBOL_PER_SLOT = 14
CP_LEN_FIRST      = 176
CP_LEN_OTHER      = 144
N_SAMPLE_PER_SLOT = 30720
L_MIN             = 0
Q_SCALE           = 256   # Q8.8

# Mixed-radix factorization
P = 32    # "inner" DFT size (fast axis of reshape)
Q = 64    # "outer" DFT size (slow axis of reshape)
assert P * Q == N_FFT

CP_LENS = np.array([CP_LEN_FIRST] + [CP_LEN_OTHER] * (N_SYMBOL_PER_SLOT - 1),
                   dtype=np.int32)
assert int(CP_LENS.sum()) + N_SYMBOL_PER_SLOT * N_FFT == N_SAMPLE_PER_SLOT


# ============================================================
# Twiddle / DFT matrices (kernel loads these from .bin)
# ============================================================
def dft_matrix(n, dtype=np.complex64):
    """Forward DFT matrix: W[k, i] = exp(-j·2π·k·i/n)."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    return np.exp(-2j * np.pi * k * i / n).astype(dtype)


def twiddle_pq(p, q, n_fft, dtype=np.complex64):
    """T[k1, n2] = exp(-j·2π·n2·k1 / n_fft), shape (p, q)."""
    k1 = np.arange(p)[:, None]
    n2 = np.arange(q)[None, :]
    return np.exp(-2j * np.pi * k1 * n2 / n_fft).astype(dtype)


W_DFT_P  = dft_matrix(P)                  # (32, 32) complex64
W_DFT_Q  = dft_matrix(Q)                  # (64, 64) complex64
U_DFT_P  = (W_DFT_P / np.sqrt(P)).astype(np.complex64)
U_DFT_Q  = (W_DFT_Q / np.sqrt(Q)).astype(np.complex64)
T_PQ     = twiddle_pq(P, Q, N_FFT)        # (32, 64) complex64


# ============================================================
# CP removal
# ============================================================
def cp_remove(x_time):
    """x_time: (..., N_SAMPLE_PER_SLOT) complex
       returns: (..., 14, N_FFT) complex"""
    assert x_time.shape[-1] == N_SAMPLE_PER_SLOT
    offs = np.concatenate([[0], np.cumsum(CP_LENS + N_FFT)])
    gather = np.empty((N_SYMBOL_PER_SLOT, N_FFT), dtype=np.int64)
    for s in range(N_SYMBOL_PER_SLOT):
        gather[s] = np.arange(offs[s] + CP_LENS[s],
                              offs[s] + CP_LENS[s] + N_FFT)
    return x_time[..., gather].astype(np.complex64)


# ============================================================
# Mixed-radix FFT — complex "reference" path
# ============================================================
def fft_2048_mixed_radix_complex(x):
    """x: (..., N_FFT) complex, returns (..., N_FFT) complex.
       Natural (non-shifted) output order."""
    assert x.shape[-1] == N_FFT
    shape = x.shape[:-1]

    # Stage 1: reshape (..., 2048) → (..., 32, 64) row-major
    x_rs = x.reshape(*shape, P, Q)                        # (..., 32, 64)

    # Stage 2: DFT-32 along axis -2  (batched over 64 cols)
    X1 = np.einsum("ab,...bc->...ac", U_DFT_P, x_rs)      # (..., 32, 64)

    # Stage 3: element-wise twiddle
    X1_tw = X1 * T_PQ                                     # (..., 32, 64)

    # Stage 4: DFT-64 along axis -1 (batched over 32 rows)
    X2 = np.einsum("...ab,cb->...ac", X1_tw, U_DFT_Q)     # (..., 32, 64)

    # Stage 5: reshape back — X[k1 + 32·k2] = X2[k1, k2]
    X_out = np.moveaxis(X2, -2, -1).reshape(*shape, N_FFT)

    return X_out


# ============================================================
# Mixed-radix FFT — kernel-aligned "4 real matmul" path
# ============================================================
def complex_matmul_as_4_real(A_re, A_im, B_re, B_im, axes):
    """(A_re + j·A_im) @ (B_re + j·B_im) via 4 real matmuls."""
    Y_re = np.einsum(axes, A_re, B_re) - np.einsum(axes, A_im, B_im)
    Y_im = np.einsum(axes, A_re, B_im) + np.einsum(axes, A_im, B_re)
    return Y_re, Y_im


def fft_2048_mixed_radix_4real(x):
    """Same output as fft_2048_mixed_radix_complex, but 4 real matmuls."""
    assert x.shape[-1] == N_FFT
    shape = x.shape[:-1]

    X_re = x.real.astype(np.float32).reshape(*shape, P, Q)
    X_im = x.imag.astype(np.float32).reshape(*shape, P, Q)

    W32_re = U_DFT_P.real.astype(np.float32)
    W32_im = U_DFT_P.imag.astype(np.float32)
    W64_re = U_DFT_Q.real.astype(np.float32)
    W64_im = U_DFT_Q.imag.astype(np.float32)
    T_re   = T_PQ.real.astype(np.float32)
    T_im   = T_PQ.imag.astype(np.float32)

    X1_re, X1_im = complex_matmul_as_4_real(
        W32_re, W32_im, X_re, X_im, "ab,...bc->...ac")

    X1_tw_re = X1_re * T_re - X1_im * T_im
    X1_tw_im = X1_re * T_im + X1_im * T_re

    X2_re, X2_im = complex_matmul_as_4_real(
        X1_tw_re, X1_tw_im, W64_re.T, W64_im.T,
        "...ab,bc->...ac")

    X_re_out = np.moveaxis(X2_re, -2, -1).reshape(*shape, N_FFT)
    X_im_out = np.moveaxis(X2_im, -2, -1).reshape(*shape, N_FFT)

    return (X_re_out + 1j * X_im_out).astype(np.complex64)


# ============================================================
# Full OFDMDemodulator pipeline
# ============================================================
def ofdm_demod(x_time, use_4real_path=False, return_mid=False):
    mid = {}
    x_cp   = cp_remove(x_time)
    mid["cp_removed"] = x_cp

    if use_4real_path:
        x_freq_natural = fft_2048_mixed_radix_4real(x_cp)
    else:
        x_freq_natural = fft_2048_mixed_radix_complex(x_cp)
    mid["fft_natural"] = x_freq_natural

    if L_MIN != 0:
        k = np.arange(N_FFT)
        phase = np.exp(-2j * np.pi * k * L_MIN / N_FFT).astype(np.complex64)
        x_freq_natural = x_freq_natural * phase

    x_freq = np.fft.fftshift(x_freq_natural, axes=-1).astype(np.complex64)
    mid["output"] = x_freq

    if return_mid:
        return x_freq, mid
    return x_freq


# ============================================================
# Verification
# ============================================================
def verify_complex_vs_numpy(atol=1e-4):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((3, N_FFT)) +
         1j * rng.standard_normal((3, N_FFT))).astype(np.complex64)
    y_mine = fft_2048_mixed_radix_complex(x)
    y_np = (np.fft.fft(x.astype(np.complex128), axis=-1)
            / np.sqrt(N_FFT)).astype(np.complex64)
    err = np.abs(y_mine - y_np).max()
    print(f"[complex  vs unitary np.fft] max|err| = {err:.3e}  (tol={atol})")
    assert err < atol, f"complex path mismatch: {err}"


def verify_4real_vs_complex(atol=1e-4):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((3, N_FFT)) +
         1j * rng.standard_normal((3, N_FFT))).astype(np.complex64)
    y_c = fft_2048_mixed_radix_complex(x)
    y_4 = fft_2048_mixed_radix_4real(x)
    err = np.abs(y_c - y_4).max()
    print(f"[4real    vs complex   ] max|err| = {err:.3e}  (tol={atol})")
    assert err < atol, f"4-real path mismatch: {err}"


def verify_against_sionna(x_time, atol=1e-3):
    try:
        from sionna.phy.ofdm import OFDMDemodulator
    except ImportError as e:
        print(f"[sionna] not installed — skipping ({e})")
        return None
    print(f"[sionna] signature: {inspect.signature(OFDMDemodulator.__init__)}")

    demod = OFDMDemodulator(fft_size=N_FFT, l_min=L_MIN,
                            cyclic_prefix_length=CP_LENS)

    try:
        import tensorflow as tf
        y_tensor = demod(tf.constant(x_time, dtype=tf.complex64))
        y_sionna = y_tensor.numpy()
        backend = "TensorFlow"
    except (ImportError, AttributeError, TypeError):
        try:
            import torch
            y_tensor = demod(torch.tensor(x_time, dtype=torch.complex64))
            y_sionna = y_tensor.numpy()
            backend = "PyTorch"
        except ImportError as e:
            print(f"[sionna] no backend (tf/torch) available — skipping ({e})")
            return None

    y_mine = ofdm_demod(x_time)
    err = np.abs(y_sionna - y_mine).max()
    print(f"[sionna/{backend:>10}] max|err| vs ofdm_demod = {err:.3e}  (tol={atol})")
    assert err < atol, f"sionna mismatch: {err}"
    return y_sionna


# ============================================================
# Golden dump
# ============================================================
def complex_to_q8_8_int16(z):
    """Interleaved int16 Q8.8."""
    re = np.round(z.real * Q_SCALE).astype(np.int16)
    im = np.round(z.imag * Q_SCALE).astype(np.int16)
    return np.stack([re, im], axis=-1).reshape(*z.shape[:-1], 2 * z.shape[-1])


def dump_goldens(x_time, out_root=None, weights_root=None):
    """生成所有 weights + goldens.
    out_root, weights_root: 不传则用默认 (项目根 weights/ + data/golden/)"""
    if out_root is None:
        out_root = str(DEFAULT_GOLDEN_DIR)
    if weights_root is None:
        weights_root = str(DEFAULT_WEIGHTS_DIR)

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(weights_root, exist_ok=True)

    # --- Kernel-ready weights (fp16) ---
    U_DFT_P.real.astype(np.float16).tofile(f"{weights_root}/w_dft32_re.bin")
    U_DFT_P.imag.astype(np.float16).tofile(f"{weights_root}/w_dft32_im.bin")
    U_DFT_Q.real.astype(np.float16).tofile(f"{weights_root}/w_dft64_re.bin")
    U_DFT_Q.imag.astype(np.float16).tofile(f"{weights_root}/w_dft64_im.bin")
    T_PQ   .real.astype(np.float16).tofile(f"{weights_root}/twiddle_pq_re.bin")
    T_PQ   .imag.astype(np.float16).tofile(f"{weights_root}/twiddle_pq_im.bin")

    # --- Kernel 需要 W64 转置版本 (Phase 3 用) ---
    U_DFT_Q.real.T.astype(np.float16).tofile(f"{weights_root}/w_dft64_re_T.bin")
    U_DFT_Q.imag.T.astype(np.float16).tofile(f"{weights_root}/w_dft64_im_T.bin")

    # Golden 必须从 kernel 实际读到的 Q8.8 输入反量化后计算，避免把输入
    # 量化误差错误归到 NPU kernel 上（与 ofdm_mod 的 golden 口径一致）。
    input_i16 = complex_to_q8_8_int16(x_time)
    input_iq = input_i16.reshape(*x_time.shape, 2)
    x_quantized = (input_iq[..., 0].astype(np.float32)
                   + 1j * input_iq[..., 1].astype(np.float32)) / Q_SCALE
    y, mid = ofdm_demod(x_quantized.astype(np.complex64), return_mid=True)

    input_i16.tofile(f"{out_root}/input.bin")
    complex_to_q8_8_int16(mid["cp_removed"]  ).tofile(f"{out_root}/cp_removed.bin")
    complex_to_q8_8_int16(mid["fft_natural"] ).tofile(f"{out_root}/fft_natural.bin")
    complex_to_q8_8_int16(mid["output"]      ).tofile(f"{out_root}/output.bin")

    # Per-stage intermediates as complex64
    X_re = mid["cp_removed"].real.astype(np.float32).reshape(-1, P, Q)
    X_im = mid["cp_removed"].imag.astype(np.float32).reshape(-1, P, Q)
    W32_re, W32_im = U_DFT_P.real.astype(np.float32), U_DFT_P.imag.astype(np.float32)
    W64_re, W64_im = U_DFT_Q.real.astype(np.float32), U_DFT_Q.imag.astype(np.float32)
    T_re,   T_im   = T_PQ.real.astype(np.float32),    T_PQ.imag.astype(np.float32)

    S2_re, S2_im = complex_matmul_as_4_real(W32_re, W32_im, X_re, X_im,
                                            "ab,nbc->nac")
    S3_re = S2_re * T_re - S2_im * T_im
    S3_im = S2_re * T_im + S2_im * T_re
    S4_re, S4_im = complex_matmul_as_4_real(S3_re, S3_im,
                                            W64_re.T, W64_im.T,
                                            "nab,bc->nac")

    (S2_re + 1j*S2_im).astype(np.complex64).tofile(f"{out_root}/stage2_dft32.bin")
    (S3_re + 1j*S3_im).astype(np.complex64).tofile(f"{out_root}/stage3_twiddle.bin")
    (S4_re + 1j*S4_im).astype(np.complex64).tofile(f"{out_root}/stage4_dft64.bin")

    print(f"\n[dump] weights → {weights_root}/")
    print(f"[dump] goldens → {out_root}/")
    print("\n[dump] weights:")
    for name in ["w_dft32_re", "w_dft32_im",
                 "w_dft64_re", "w_dft64_im",
                 "w_dft64_re_T", "w_dft64_im_T",
                 "twiddle_pq_re", "twiddle_pq_im"]:
        f = f"{weights_root}/{name}.bin"
        kib = os.path.getsize(f) / 1024.0
        print(f"  {name+'.bin':<22}  {kib:>6.1f} KiB")
    print("\n[dump] goldens:")
    for name in ["input", "cp_removed", "fft_natural", "output",
                 "stage2_dft32", "stage3_twiddle", "stage4_dft64"]:
        f = f"{out_root}/{name}.bin"
        kib = os.path.getsize(f) / 1024.0
        print(f"  {name+'.bin':<22}  {kib:>6.1f} KiB")


# ============================================================
# Driver
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" OFDM Demodulator — 数据生成 + 自检")
    print("=" * 60)
    print(f"  KERNEL_DIR  = {_KERNEL_DIR}")
    print(f"  WEIGHTS_DIR = {DEFAULT_WEIGHTS_DIR}")
    print(f"  GOLDEN_DIR  = {DEFAULT_GOLDEN_DIR}")
    print()

    # Self-tests (must pass before dump)
    verify_complex_vs_numpy()
    verify_4real_vs_complex()

    # Deterministic slot
    rng = np.random.default_rng(seed=12345)
    x_time = ((rng.standard_normal(N_SAMPLE_PER_SLOT)
               + 1j * rng.standard_normal(N_SAMPLE_PER_SLOT))
              * (1.0 / np.sqrt(2))).astype(np.complex64)

    verify_against_sionna(x_time)
    dump_goldens(x_time)

    print("\n" + "=" * 60)
    print(" ✅ 所有数据生成完成 — 跑 bash run.sh 即可测 kernel")
    print("=" * 60)
