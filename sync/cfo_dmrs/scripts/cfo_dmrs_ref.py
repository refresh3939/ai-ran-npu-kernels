#!/usr/bin/env python3
"""
cfo_dmrs_ref.py — DMRS-based CFO tracker (Layer 3) v2 OTA refactor

v2 changes (this file vs v1):
  - Input  format: 4 separate fp16 planes (Y_re, Y_im, X_re, X_im) matching
                   channel_est_ls output layout — no more cint16 interleaved
  - Output format: scalar δf (fp32 Hz) + debug fields (c.re, c.im) — no more
                   14-correction phasors (computed downstream by cfo_correct)

Algorithm (unchanged from v1):
    H_ls[s, k]  = Y[DMRS_sym[s], 2k] · conj(X_dmrs[s, k])    for s∈{0,1}, k∈[0,798)
    c           = Σ_k conj(H_ls[0, k]) · H_ls[1, k]
    δf_est_hz   = atan2(c.im, c.re) / (2π · ΔT)              ΔT = 9·T_sym_other

Test cases (5 cases, physically identical to v1, only output format changed):
  0  no_cfo            : δf=0, H=1, no noise
  1  small_positive    : δf=+200 Hz, H=1
  2  large_negative    : δf=-1000 Hz, H=1 (near unambig boundary)
  3  doppler_only      : δf=0, freq+time_var ch (Doppler must not be misread)
  4  cfo_plus_noise    : δf=+500 Hz, freq-sel ch, SNR=10dB

Output files per case (under ${AIRAN_DATA_DIR}/data/golden/case_N_xxx/):
  y_re.bin    : Y_re[14, 1664] fp16  (52416 B = 14·1664·2)
  y_im.bin    : Y_im[14, 1664] fp16  (52416 B)
  x_re.bin    : X_re[2, 896]   fp16  ( 3584 B = 2·896·2)
  x_im.bin    : X_im[2, 896]   fp16  ( 3584 B)
  truth.bin   : δf_est_hz scalar fp32 (4 B; first 4 bytes of 8-fp32 record)
  meta.txt    : δf_true, δf_est, error, debug
"""

import os
import numpy as np
from pathlib import Path

# ─── Constants (locked, matches kernel/h) ──────────────────────────────────
N_FFT          = 2048
SAMPLE_RATE_HZ = 61_440_000
N_SC_USED      = 1596
N_SC_PAD       = 1664           # channel_est_ls per-sym stride
N_SYMBOL       = 14
N_DMRS_RE      = 798
N_DMRS_PAD     = 896            # 7×128, X per-sym stride
DMRS_SYM_IDX   = np.array([2, 11], dtype=np.int32)

CP_LEN_FIRST   = 176
CP_LEN_OTHER   = 144
T_S            = 1.0 / SAMPLE_RATE_HZ

BASE_SEED      = 0xC503

# ─── DMRS 3GPP config (TS 38.211) — MUST match dmrs_gen_ref / dmrs_gen.h ────
# X golden produced here is byte-identical to dmrs_gen case_0_baseline output.
N_SYMB_SLOT    = 14
GOLD_NC        = 1600           # c(n) = (x1(n+Nc)+x2(n+Nc)) mod 2
DMRS_SLOT      = 0
DMRS_N_ID      = 1              # dmrs-ScramblingID (= N_ID^cell, skill n_id)
DMRS_N_SCID    = 0

# ─── Sym timing (for CFO injection only) ───────────────────────────────────
def sym_start_sample(n: int) -> int:
    s = 0
    for i in range(n):
        s += N_FFT + (CP_LEN_FIRST if i == 0 else CP_LEN_OTHER)
    return s

T_SYM     = np.array([sym_start_sample(n) * T_S for n in range(N_SYMBOL)])
T_REF     = T_SYM[DMRS_SYM_IDX[0]]
DELTA_T   = T_SYM[DMRS_SYM_IDX[1]] - T_SYM[DMRS_SYM_IDX[0]]
CFO_UNAMBIG_HZ = 0.5 / DELTA_T


# ─── DMRS QPSK sequence — real 3GPP TS 38.211 Gold (matches dmrs_gen) ───────
# Replaces the v1 np.random stub. Identical math to dmrs_gen_ref.dmrs_qpsk so
# the X golden here equals dmrs_gen's output bit-for-bit.
def dmrs_cinit(slot: int, l: int, n_id: int, n_scid: int) -> int:
    # TS 38.211 §6.4.1.1.1.1
    return (((1 << 17) * (N_SYMB_SLOT * slot + l + 1) * (2 * n_id + 1)
             + 2 * n_id + n_scid) % (1 << 31))


def gold_sequence(c_init: int, m: int, nc: int = GOLD_NC) -> np.ndarray:
    # TS 38.211 §5.2.1 length-31 Gold sequence
    n_total = m + nc
    x1 = np.zeros(n_total + 31, dtype=np.int8)
    x2 = np.zeros(n_total + 31, dtype=np.int8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for n in range(n_total):
        x1[n + 31] = (x1[n + 3] + x1[n]) & 1
        x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) & 1
    return ((x1[nc:nc + m] + x2[nc:nc + m]) & 1).astype(np.int8)


def dmrs_qpsk(c_init: int, n_re: int) -> np.ndarray:
    # TS 38.211 §6.4.1.1.1: r(n) = (1/√2)(1-2c(2n)) + j(1/√2)(1-2c(2n+1))
    c = gold_sequence(c_init, 2 * n_re)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    re = inv_sqrt2 * (1.0 - 2.0 * c[0::2])
    im = inv_sqrt2 * (1.0 - 2.0 * c[1::2])
    return (re + 1j * im).astype(np.complex128)


# ─── Channel models (unchanged from v1) ───────────────────────────────────
def channel_flat(n_sc, n_sym):
    return np.ones((n_sym, n_sc), dtype=np.complex128)

def channel_freq_selective(n_sc, n_sym, seed):
    rng = np.random.default_rng(seed)
    taps_t = np.array([0, 2, 5])
    taps_a = rng.standard_normal(3) + 1j * rng.standard_normal(3)
    taps_a /= np.linalg.norm(taps_a)
    sc_idx = np.arange(n_sc) - n_sc // 2
    H_freq = np.zeros(n_sc, dtype=np.complex128)
    for a, tau in zip(taps_a, taps_t):
        H_freq += a * np.exp(-1j * 2 * np.pi * sc_idx * tau / N_FFT)
    return np.tile(H_freq[None, :], (n_sym, 1))

def channel_freq_time_var(n_sc, n_sym, seed, doppler_hz=100.0):
    rng = np.random.default_rng(seed)
    taps_t = np.array([0, 2, 5])
    taps_a = rng.standard_normal(3) + 1j * rng.standard_normal(3)
    taps_a /= np.linalg.norm(taps_a)
    sc_idx = np.arange(n_sc) - n_sc // 2
    H = np.zeros((n_sym, n_sc), dtype=np.complex128)
    for sym in range(n_sym):
        tap_dopplers = rng.uniform(-doppler_hz, doppler_hz, size=3)
        for a, tau, fd in zip(taps_a, taps_t, tap_dopplers):
            phase = 2 * np.pi * fd * T_SYM[sym]
            H[sym] += a * np.exp(1j * phase) * \
                      np.exp(-1j * 2 * np.pi * sc_idx * tau / N_FFT)
    return H


def apply_cfo(Y, delta_f_hz):
    phases = 2 * np.pi * delta_f_hz * T_SYM
    rot = np.exp(1j * phases)
    return Y * rot[:, None]


def cfo_dmrs_compute(Y, X_dmrs):
    """Reference δf computation. Y, X_dmrs are complex128. Returns scalar δf (Hz)."""
    Y_dmrs = Y[DMRS_SYM_IDX, 0::2]
    H_ls   = Y_dmrs * np.conj(X_dmrs)
    c = np.sum(np.conj(H_ls[0]) * H_ls[1])
    delta_f_hz = np.angle(c) / (2 * np.pi * DELTA_T)
    return delta_f_hz, c, H_ls


# ─── Packers: complex → 4-plane fp16 with N_SC_PAD / N_DMRS_PAD stride ─────
def pack_Y_to_4plane_fp16(Y_complex: np.ndarray) -> tuple:
    """
    Y_complex: (14, 1596) complex128.
    Returns (Y_re, Y_im) each shape (14, 1664) fp16, with [:, 1596:1664] zero-padded.
    Layout matches channel_est_ls output.
    """
    Y_re_pad = np.zeros((N_SYMBOL, N_SC_PAD), dtype=np.float16)
    Y_im_pad = np.zeros((N_SYMBOL, N_SC_PAD), dtype=np.float16)
    Y_re_pad[:, :N_SC_USED] = np.real(Y_complex).astype(np.float16)
    Y_im_pad[:, :N_SC_USED] = np.imag(Y_complex).astype(np.float16)
    return Y_re_pad, Y_im_pad


def pack_X_to_4plane_fp16(X_complex: np.ndarray) -> tuple:
    """
    X_complex: (2, 798) complex128 — already comb-2 selected (1 RE per DMRS slot).
    Returns (X_re, X_im) each shape (2, 896) fp16, with [:, 798:896] zero-padded.
    """
    X_re_pad = np.zeros((2, N_DMRS_PAD), dtype=np.float16)
    X_im_pad = np.zeros((2, N_DMRS_PAD), dtype=np.float16)
    X_re_pad[:, :N_DMRS_RE] = np.real(X_complex).astype(np.float16)
    X_im_pad[:, :N_DMRS_RE] = np.imag(X_complex).astype(np.float16)
    return X_re_pad, X_im_pad


# ─── Case generation ──────────────────────────────────────────────────────
def gen_case(idx, name, delta_f_hz, channel_kind, snr_db, seed_offset):
    seed = BASE_SEED + seed_offset

    # DMRS QPSK pilots — real 3GPP Gold seq, fixed config (l=2,11), same for
    # all CFO cases (DMRS depends on slot/sym/N_ID only, not the CFO scenario).
    X_dmrs_complex = np.stack([
        dmrs_qpsk(dmrs_cinit(DMRS_SLOT, int(DMRS_SYM_IDX[s]), DMRS_N_ID, DMRS_N_SCID),
                  N_DMRS_RE)
        for s in range(2)
    ], axis=0)

    if channel_kind == 'flat':
        H_true = channel_flat(N_SC_USED, N_SYMBOL)
    elif channel_kind == 'freq_sel':
        H_true = channel_freq_selective(N_SC_USED, N_SYMBOL, seed=seed + 200)
    elif channel_kind == 'freq_time_var':
        H_true = channel_freq_time_var(N_SC_USED, N_SYMBOL, seed=seed + 200, doppler_hz=100.0)
    else:
        raise ValueError(channel_kind)

    # Build TX: zeros except DMRS comb-2 even SC on sym 2/11; random QPSK data elsewhere
    rng = np.random.default_rng(seed + 300)
    X_full = np.zeros((N_SYMBOL, N_SC_USED), dtype=np.complex128)
    for s_idx, sym in enumerate(DMRS_SYM_IDX):
        X_full[sym, 0::2] = X_dmrs_complex[s_idx]
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    data_table = np.array([
        ( inv_sqrt2 + 1j*inv_sqrt2), (-inv_sqrt2 + 1j*inv_sqrt2),
        ( inv_sqrt2 - 1j*inv_sqrt2), (-inv_sqrt2 - 1j*inv_sqrt2),
    ])
    for sym in [s for s in range(N_SYMBOL) if s not in DMRS_SYM_IDX]:
        X_full[sym] = data_table[rng.integers(0, 4, size=N_SC_USED)]

    Y_clean = X_full * H_true
    Y_cfo   = apply_cfo(Y_clean, delta_f_hz)
    if snr_db is None:
        Y_noisy = Y_cfo
    else:
        sig_power = np.mean(np.abs(Y_cfo) ** 2)
        noise_std = np.sqrt(sig_power / (10 ** (snr_db / 10)) / 2)
        noise = (rng.standard_normal(Y_cfo.shape) +
                 1j * rng.standard_normal(Y_cfo.shape)) * noise_std
        Y_noisy = Y_cfo + noise

    # ── Quantize to fp16 (kernel sees this), then run ref on the same fp16
    # values to get the "true" δf that fp16 hardware could possibly reproduce
    Y_re, Y_im = pack_Y_to_4plane_fp16(Y_noisy)
    X_re, X_im = pack_X_to_4plane_fp16(X_dmrs_complex)

    # Reconstruct what kernel actually sees (fp16 quantized)
    Y_kernel = (Y_re[:, :N_SC_USED].astype(np.float64) +
                1j * Y_im[:, :N_SC_USED].astype(np.float64))
    X_kernel = (X_re[:, :N_DMRS_RE].astype(np.float64) +
                1j * X_im[:, :N_DMRS_RE].astype(np.float64))

    delta_f_est, c, H_ls = cfo_dmrs_compute(Y_kernel, X_kernel)

    return {
        'idx': idx,
        'name': name,
        'delta_f_true_hz': delta_f_hz,
        'delta_f_est_hz': delta_f_est,
        'channel_kind': channel_kind,
        'snr_db': snr_db,
        'Y_re': Y_re,   # (14, 1664) fp16
        'Y_im': Y_im,
        'X_re': X_re,   # (2, 896)  fp16
        'X_im': X_im,
        'c': c,
    }


CASE_MANIFEST = [
    dict(idx=0, name='no_cfo',         delta_f_hz=   0.0, channel_kind='flat',          snr_db=None, seed_offset=0),
    dict(idx=1, name='small_positive', delta_f_hz= 200.0, channel_kind='flat',          snr_db=None, seed_offset=1),
    dict(idx=2, name='large_negative', delta_f_hz=-1000.0,channel_kind='flat',          snr_db=None, seed_offset=2),
    dict(idx=3, name='doppler_only',   delta_f_hz=   0.0, channel_kind='freq_time_var', snr_db=None, seed_offset=3),
    dict(idx=4, name='cfo_plus_noise', delta_f_hz= 500.0, channel_kind='freq_sel',      snr_db=10.0, seed_offset=4),
]

# Per-case Hz thresholds (for ref self-check; also used by verify_result.py)
DELTA_F_HZ_THRESH = {
    0: 1.0, 1: 5.0, 2: 5.0, 3: 200.0, 4: 50.0,
}


def main():
    out_dir = Path(os.environ.get('AIRAN_DATA_DIR', '.')) / 'data' / 'golden'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== cfo_dmrs v2 reference (4-plane fp16 input, scalar δf output) ===")
    print(f"  T_REF   = {T_REF*1e6:.4f} µs   (sym 2 start)")
    print(f"  ΔT      = {DELTA_T*1e6:.4f} µs (sym 11 - sym 2)")
    print(f"  unambig = ±{CFO_UNAMBIG_HZ:.2f} Hz")
    print(f"  N_SC_PAD={N_SC_PAD}  N_DMRS_PAD={N_DMRS_PAD}")
    print(f"  output  = {out_dir}")
    print()

    all_ok = True
    for cfg in CASE_MANIFEST:
        case = gen_case(**cfg)
        case_dir = out_dir / f"case_{case['idx']}_{case['name']}"
        case_dir.mkdir(parents=True, exist_ok=True)

        # 4-plane fp16 input bins
        case['Y_re'].tofile(case_dir / 'y_re.bin')
        case['Y_im'].tofile(case_dir / 'y_im.bin')
        case['X_re'].tofile(case_dir / 'x_re.bin')
        case['X_im'].tofile(case_dir / 'x_im.bin')

        # Scalar truth: δf_est in fp32 (first 4 bytes; rest of 8-fp32 record left to
        # be filled by kernel — verify only compares slot [0])
        truth_arr = np.array([case['delta_f_est_hz']], dtype=np.float32)
        truth_arr.tofile(case_dir / 'truth.bin')

        # Meta
        df_true = case['delta_f_true_hz']
        df_est  = case['delta_f_est_hz']
        df_err  = df_est - df_true
        with open(case_dir / 'meta.txt', 'w') as f:
            f.write(f"name           : {case['name']}\n")
            f.write(f"delta_f_true_hz: {df_true:+.4f}\n")
            f.write(f"delta_f_est_hz : {df_est:+.4f}\n")
            f.write(f"delta_f_err_hz : {df_err:+.4f}\n")
            f.write(f"channel        : {case['channel_kind']}\n")
            f.write(f"snr_db         : {case['snr_db']}\n")
            f.write(f"|c|            : {np.abs(case['c']):.4f}\n")
            f.write(f"arg(c) rad     : {np.angle(case['c']):+.6f}\n")
            f.write(f"thresh_hz      : {DELTA_F_HZ_THRESH[case['idx']]}\n")

        thresh = DELTA_F_HZ_THRESH[case['idx']]
        ok = abs(df_err) < thresh
        all_ok &= ok
        flag = "PASS" if ok else "FAIL"
        print(f"  case {case['idx']} {case['name']:18s} "
              f"δf_true={df_true:+8.2f}  δf_est={df_est:+8.2f}  "
              f"err={df_err:+7.3f} Hz   thresh<{thresh} Hz  [{flag}]")

    print()
    print(f"=== {'ALL CASES PASS' if all_ok else 'SOME CASES FAILED'} ===")
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
