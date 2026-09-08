#!/usr/bin/env python3
"""
equalize_ref.py — Equalizer golden generator (full-grid fp16 separated I/O).

Standardized PHYSICAL-DOMAIN format: fp16, re/im separate, FULL GRID natural SC
order [14, 1596] (host pads to N_SC_PAD=1664). No cint16/Q2.14, no even/odd split.

Pipeline: simulate channel + re_demap (-> fp16 Y), channel_est LS+interp (-> fp16
H full grid), then the SISO single-tap equalizer (golden the kernel matches).

Files per case (logical [14,1596] fp16; host pads to 1664):
  in_y_re.bin / in_y_im.bin     : Y           (= ofdm/re_demap output, physical)
  in_h_re.bin / in_h_im.bin     : H_hat       (= channel_est output, full grid)
  in_n0.bin                     : N0 per-RE   (pre-eq noise variance)
  truth_xhat_re.bin / truth_xhat_im.bin / truth_no_eff.bin
"""
import os
from pathlib import Path
import numpy as np

N_FFT, N_SC_USED, N_SYM = 2048, 1596, 14
DMRS_SYM_IDX = np.array([2, 11], dtype=np.int32)
N_DMRS_SYM   = len(DMRS_SYM_IDX)
N_DMRS_RE    = N_SC_USED // 2                    # 798
DMRS_SC_IDX  = np.arange(0, N_SC_USED, 2, dtype=np.int32)

SCS_HZ       = 30000.0
T_SYMBOL_SEC = 1.0 / SCS_HZ + 144.0 / (N_FFT * SCS_HZ)

G_FLOOR = 1.0e-4                                  # MUST match equalize.h
QAM16_NORM = np.sqrt(10.0)                         # unit-power 16-QAM (Sionna)


def f16c(x):
    """Round a complex array to fp16 re/im (the physical values the kernel reads)."""
    return (np.asarray(x).real.astype(np.float16).astype(np.float32) +
            1j * np.asarray(x).imag.astype(np.float16).astype(np.float32)).astype(np.complex64)


def gen_dmrs_qpsk(seed, length):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 4, size=length)
    phases = (bits * 0.5 + 0.25) * np.pi
    return np.exp(1j * phases).astype(np.complex64)


# ── channel models ─────────────────────────────────────────────────────────
def gen_channel_awgn(seed):
    return np.ones((N_SYM, N_SC_USED), dtype=np.complex64)

def gen_channel_flat(seed):
    rng = np.random.default_rng(seed)
    h0 = rng.uniform(-0.8, 0.8) + 1j * rng.uniform(-0.8, 0.8)
    return np.full((N_SYM, N_SC_USED), h0, dtype=np.complex64)

def gen_channel_freq_selective(seed):
    rng = np.random.default_rng(seed)
    path_delays_ns = np.array([0, 30, 70, 150], dtype=np.float64)
    path_gains_db  = np.array([0, -3, -6, -10], dtype=np.float64)
    amps   = 10.0 ** (path_gains_db / 20.0)
    phases = rng.uniform(0, 2 * np.pi, size=4)
    coeff  = amps * np.exp(1j * phases)
    sc_freqs = (np.arange(N_SC_USED) - N_SC_USED / 2) * SCS_HZ
    H = np.zeros(N_SC_USED, dtype=np.complex64)
    for p in range(4):
        H += coeff[p] * np.exp(-1j * 2 * np.pi * sc_freqs * (path_delays_ns[p] * 1e-9))
    H /= np.max(np.abs(H)) * 1.25
    return np.tile(H[np.newaxis, :], (N_SYM, 1)).astype(np.complex64)

def gen_channel_freq_time_var(seed):
    H = gen_channel_freq_selective(seed)
    rng = np.random.default_rng(seed + 1)
    dop = rng.uniform(50, 200)
    t = np.arange(N_SYM) * T_SYMBOL_SEC
    ph = np.exp(1j * 2 * np.pi * dop * t).astype(np.complex64)
    return H * ph[:, np.newaxis]

def gen_channel_low_snr(seed):
    return gen_channel_freq_selective(seed + 100)


# ── LS channel estimator (consumes fp16 Y; outputs full-grid H) ────────────
def channel_est_ls_numpy(Y_used, X_dmrs):
    H_ls = np.zeros((N_DMRS_SYM, N_DMRS_RE), dtype=np.complex64)
    for s in range(N_DMRS_SYM):
        H_ls[s] = Y_used[DMRS_SYM_IDX[s], DMRS_SC_IDX] * np.conj(X_dmrs[s])
    H_freq = np.zeros((N_DMRS_SYM, N_SC_USED), dtype=np.complex64)
    for s in range(N_DMRS_SYM):
        H_freq[s, 0::2]    = H_ls[s]
        H_freq[s, 1:-1:2]  = 0.5 * (H_ls[s, :-1] + H_ls[s, 1:])
        H_freq[s, -1]      = -0.5 * H_ls[s, -2] + 1.5 * H_ls[s, -1]
    H_hat = np.zeros((N_SYM, N_SC_USED), dtype=np.complex64)
    sym_a, sym_b = DMRS_SYM_IDX
    delta = sym_b - sym_a
    for sym in range(N_SYM):
        a = (sym - sym_a) / delta
        H_hat[sym] = (1.0 - a) * H_freq[0] + a * H_freq[1]
    return H_hat


# ── SISO single-tap equalizer (fp32-core golden; reads fp16 inputs) ────────
def equalize_numpy(Y, H_hat, N0):
    H  = H_hat.astype(np.complex64)
    g  = (H.real.astype(np.float32) ** 2 + H.imag.astype(np.float32) ** 2)
    g  = np.maximum(g, np.float32(G_FLOOR))
    inv = (np.float32(1.0) / g).astype(np.float32)
    N0_f = N0.astype(np.float16).astype(np.float32)
    x_hat  = (np.conj(H) * Y.astype(np.complex64)) * inv
    no_eff = (N0_f * inv).astype(np.float32)
    return x_hat.astype(np.complex64), no_eff


# ── per-case generation ─────────────────────────────────────────────────────
def gen_case(name, ch_fn, snr_db, seed, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    H_truth = ch_fn(seed)
    X_dmrs = np.zeros((N_DMRS_SYM, N_DMRS_RE), dtype=np.complex64)
    for s in range(N_DMRS_SYM):
        X_dmrs[s] = gen_dmrs_qpsk(seed * 10 + s, N_DMRS_RE)

    rng = np.random.default_rng(seed)
    data = ((rng.choice([-1, 1, -3, 3], size=(N_SYM, N_SC_USED)).astype(np.float32)
             + 1j * rng.choice([-1, 1, -3, 3], size=(N_SYM, N_SC_USED)).astype(np.float32))
            / QAM16_NORM).astype(np.complex64)
    Y = H_truth * data
    for s in range(N_DMRS_SYM):
        sym = DMRS_SYM_IDX[s]
        Y[sym, DMRS_SC_IDX] = H_truth[sym, DMRS_SC_IDX] * X_dmrs[s]

    if snr_db is not None:
        sig  = float(np.mean(np.abs(Y) ** 2))
        N0_c = sig / (10.0 ** (snr_db / 10.0))
        nstd = np.sqrt(N0_c / 2.0)
        Y = Y + (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape)
                 ).astype(np.complex64) * nstd
    else:
        N0_c = float(G_FLOOR)

    # PHYSICAL domain: Y is fp16 (no Q2.14). H estimated from the fp16 Y.
    Y_f16 = f16c(Y)
    H_hat = channel_est_ls_numpy(Y_f16, X_dmrs)
    H_f16 = f16c(H_hat)
    N0 = np.full((N_SYM, N_SC_USED), N0_c, dtype=np.float32)

    x_hat, no_eff = equalize_numpy(Y_f16, H_f16, N0)

    # Full-grid natural order, fp16, re/im separate (no even/odd split).
    Y_f16.real.astype(np.float16).tofile(out_dir / "in_y_re.bin")
    Y_f16.imag.astype(np.float16).tofile(out_dir / "in_y_im.bin")
    H_f16.real.astype(np.float16).tofile(out_dir / "in_h_re.bin")
    H_f16.imag.astype(np.float16).tofile(out_dir / "in_h_im.bin")
    N0.astype(np.float16).tofile(out_dir / "in_n0.bin")
    x_hat.real.astype(np.float16).tofile(out_dir / "truth_xhat_re.bin")
    x_hat.imag.astype(np.float16).tofile(out_dir / "truth_xhat_im.bin")
    no_eff.astype(np.float16).tofile(out_dir / "truth_no_eff.bin")

    print(f"[{name}]  snr={snr_db} dB  N0={N0_c:.4e}")
    print(f"  |H_hat| [{np.abs(H_hat).min():.4f}, {np.abs(H_hat).max():.4f}]"
          f"  |x_hat| [{np.abs(x_hat).min():.4f}, {np.abs(x_hat).max():.4f}]"
          f"  no_eff [{no_eff.min():.3e}, {no_eff.max():.3e}]")


def main():
    # kernel-self-contained: data lives under <kernel_dir>/data/golden/case_*
    root = os.environ.get("AIRAN_DATA_DIR",
                          str(Path(__file__).resolve().parent.parent))
    out_root = Path(root) / "data" / "golden"
    print(f"Output root: {out_root}")
    cases = [
        ("case_0_awgn_only",      gen_channel_awgn,           40.0, 0),
        ("case_1_flat_fading",    gen_channel_flat,           40.0, 1),
        ("case_2_freq_selective", gen_channel_freq_selective, 30.0, 2),
        ("case_3_freq_time_var",  gen_channel_freq_time_var,  25.0, 3),
        ("case_4_low_snr",        gen_channel_low_snr,         0.0, 4),
    ]
    for name, fn, snr, seed in cases:
        gen_case(name, fn, snr, seed, out_root / name)
    print(f"\nAll {len(cases)} cases generated under {out_root}")


if __name__ == "__main__":
    main()