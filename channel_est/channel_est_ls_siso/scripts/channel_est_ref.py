#!/usr/bin/env python3
"""
channel_est_ref.py — v6.2 Python golden generator.

Outputs natural-order H_re/H_im on 1664-padded grid:
  truth_h_re.bin / truth_h_im.bin : [14, 1664] fp16  (SC[0..1663] in natural order)
  truth_err_var.bin               : [14, 1664] fp16  (n0 placeholder)
"""
import os
import sys
import numpy as np
from pathlib import Path
from dmrs_ref import dmrs_seq          # single-source DMRS (== TX dmrs_gen)

# DMRS config — MUST match TX dmrs_gen launch params (n_id/n_scid/slot)
DMRS_N_ID   = 0
DMRS_N_SCID = 0
DMRS_SLOT   = 0

N_FFT          = 2048
N_SC_RAW       = 1596
N_SC_USED      = 1664
N_SYM          = 14
DMRS_SYM_IDX   = np.array([2, 11], dtype=np.int32)
N_DMRS_SYM     = len(DMRS_SYM_IDX)
N_DMRS_RE_RAW    = N_SC_RAW // 2
N_DMRS_RE_PADDED = N_SC_USED // 2
DMRS_SC_IDX    = np.arange(0, N_SC_RAW, 2, dtype=np.int32)

Q_BITS = 14
Q_SCALE = 1 << Q_BITS
Q_MAX = (1 << 15) - 1
Q_MIN = -(1 << 15)
SCS_HZ = 30000.0
T_SYMBOL_SEC = 1.0 / SCS_HZ + 144.0 / (N_FFT * SCS_HZ)


def complex_to_cint16(x):
    re = np.clip(np.round(x.real * Q_SCALE), Q_MIN, Q_MAX).astype(np.int16)
    im = np.clip(np.round(x.imag * Q_SCALE), Q_MIN, Q_MAX).astype(np.int16)
    out = np.empty(re.size * 2, dtype=np.int16)
    out[0::2] = re.flatten(); out[1::2] = im.flatten()
    return out


def cint16_to_complex(x, shape):
    re = x[0::2].astype(np.float32) / Q_SCALE
    im = x[1::2].astype(np.float32) / Q_SCALE
    return (re + 1j * im).reshape(shape)


# NOTE: DMRS pilots now come from dmrs_ref.dmrs_seq (TS 38.211 standard Gold
# sequence, bit-identical to TX dmrs_gen). The old gen_dmrs_qpsk used a test-only
# random QPSK seed that did NOT match the transmitted sequence -> RX X_ref != TX
# pilots -> LS estimate H = Y*conj(X) wrong on real OTA data. Removed.


def gen_channel_awgn(seed):
    return np.ones((N_SYM, N_SC_RAW), dtype=np.complex64)


def gen_channel_flat(seed):
    rng = np.random.default_rng(seed)
    h0 = rng.uniform(-0.8, 0.8) + 1j * rng.uniform(-0.8, 0.8)
    return np.full((N_SYM, N_SC_RAW), h0, dtype=np.complex64)


def gen_channel_freq_selective(seed):
    rng = np.random.default_rng(seed)
    delays_ns = np.array([0, 30, 70, 150], dtype=np.float64)
    gains_db  = np.array([0, -3, -6, -10], dtype=np.float64)
    amps = 10.0 ** (gains_db / 20.0)
    phases = rng.uniform(0, 2*np.pi, size=4)
    coeff = amps * np.exp(1j * phases)
    sc_freqs = (np.arange(N_SC_RAW) - N_SC_RAW / 2) * SCS_HZ
    H_freq = np.zeros(N_SC_RAW, dtype=np.complex64)
    for p in range(4):
        H_freq += coeff[p] * np.exp(-1j * 2 * np.pi * sc_freqs * (delays_ns[p] * 1e-9))
    H_freq /= np.max(np.abs(H_freq)) * 1.25
    return np.tile(H_freq[np.newaxis, :], (N_SYM, 1)).astype(np.complex64)


def gen_channel_freq_time_var(seed):
    H_base = gen_channel_freq_selective(seed)
    rng = np.random.default_rng(seed + 1)
    doppler = rng.uniform(50, 200)
    times_s = np.arange(N_SYM) * T_SYMBOL_SEC
    phase_per_sym = np.exp(1j * 2 * np.pi * doppler * times_s).astype(np.complex64)
    return H_base * phase_per_sym[:, np.newaxis]


def gen_channel_low_snr(seed):
    return gen_channel_freq_selective(seed + 100)


def channel_est_ls_numpy(Y_used, X_dmrs):
    """LS + zero-extended freq-interp on 1664-padded grid (kernel-aligned)."""
    H_ls_raw = np.zeros((N_DMRS_SYM, N_DMRS_RE_RAW), dtype=np.complex64)
    for s in range(N_DMRS_SYM):
        sym = DMRS_SYM_IDX[s]
        H_ls_raw[s] = Y_used[sym, DMRS_SC_IDX] * np.conj(X_dmrs[s])

    H_ls_padded = np.zeros((N_DMRS_SYM, N_DMRS_RE_PADDED + 1), dtype=np.complex64)
    H_ls_padded[:, :N_DMRS_RE_RAW] = H_ls_raw

    H_freq = np.zeros((N_DMRS_SYM, N_SC_USED), dtype=np.complex64)
    for s in range(N_DMRS_SYM):
        H_freq[s, 0::2] = H_ls_padded[s, :N_DMRS_RE_PADDED]
        H_freq[s, 1::2] = 0.5 * (H_ls_padded[s, :N_DMRS_RE_PADDED] +
                                  H_ls_padded[s, 1:N_DMRS_RE_PADDED + 1])

    H_hat = np.zeros((N_SYM, N_SC_USED), dtype=np.complex64)
    a, b = DMRS_SYM_IDX[0], DMRS_SYM_IDX[1]
    delta = b - a
    H_a, H_b = H_freq[0], H_freq[1]
    for sym in range(N_SYM):
        alpha = (sym - a) / delta
        H_hat[sym] = (1.0 - alpha) * H_a + alpha * H_b
    return H_hat


def gen_case(name, channel_fn, snr_db, seed, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    H_truth = channel_fn(seed)
    X_dmrs = np.zeros((N_DMRS_SYM, N_DMRS_RE_RAW), dtype=np.complex64)
    for s in range(N_DMRS_SYM):
        X_dmrs[s] = dmrs_seq(DMRS_SLOT, int(DMRS_SYM_IDX[s]),
                             DMRS_N_ID, DMRS_N_SCID, n_re=N_DMRS_RE_RAW)

    rng = np.random.default_rng(seed)
    data = (rng.choice([-1, 1, -3, 3], size=(N_SYM, N_SC_RAW)).astype(np.float32) / 3.0 +
            1j * rng.choice([-1, 1, -3, 3], size=(N_SYM, N_SC_RAW)).astype(np.float32) / 3.0).astype(np.complex64)
    Y_used = H_truth * data
    for s in range(N_DMRS_SYM):
        sym = DMRS_SYM_IDX[s]
        Y_used[sym, DMRS_SC_IDX] = H_truth[sym, DMRS_SC_IDX] * X_dmrs[s]

    sig_p = float(np.mean(np.abs(Y_used) ** 2))
    if snr_db is not None:
        noise_p = float(sig_p / (10.0 ** (snr_db / 10.0)))
        nstd = np.sqrt(noise_p / 2.0)
        noise = (rng.standard_normal(Y_used.shape) +
                 1j * rng.standard_normal(Y_used.shape)).astype(np.complex64) * nstd
        Y_used = Y_used + noise
    else:
        noise_p = 0.0

    Y_int16 = complex_to_cint16(Y_used)
    Y_requant = cint16_to_complex(Y_int16, (N_SYM, N_SC_RAW))
    X_int16 = complex_to_cint16(X_dmrs)
    X_requant = cint16_to_complex(X_int16, (N_DMRS_SYM, N_DMRS_RE_RAW))

    H_hat = channel_est_ls_numpy(Y_requant, X_requant)
    truth_h_re = H_hat.real.astype(np.float16)
    truth_h_im = H_hat.imag.astype(np.float16)
    err_var = np.full((N_SYM, N_SC_USED), noise_p, dtype=np.float16)

    Y_int16.tofile(out_dir / "input.bin")
    X_int16.tofile(out_dir / "pilot.bin")
    truth_h_re.tofile(out_dir / "truth_h_re.bin")
    truth_h_im.tofile(out_dir / "truth_h_im.bin")
    err_var   .tofile(out_dir / "truth_err_var.bin")
    complex_to_cint16(H_hat[:, :N_SC_RAW]).tofile(out_dir / "truth.bin")  # legacy

    H_ideal = channel_est_ls_numpy(Y_used, X_dmrs)
    mse = np.mean(np.abs(H_ideal[:, :N_SC_RAW] - H_truth) ** 2)
    snr_eff = -10 * np.log10(mse + 1e-30)
    print(f"\n[{name}]")
    print(f"  H_truth |.| range : [{np.abs(H_truth).min():.3f}, {np.abs(H_truth).max():.3f}]")
    print(f"  signal_p = {sig_p:.4e}  noise_p = {noise_p:.4e}")
    print(f"  ref estimator MSE vs ideal H : {mse:.4e}  ({snr_eff:.1f} dB)")


def main():
    # Default: kernel-local data/ (2 levels up from this script: scripts/ → channel_est_ls/)
    project_root = os.environ.get("AIRAN_DATA_DIR",
                                   str(Path(__file__).resolve().parent.parent))
    out_root = Path(project_root) / "data" / "golden"
    print(f"Output root: {out_root}")
    print(f"v6.2 layout: H_re/H_im [14,{N_SC_USED}] fp16 NATURAL ORDER")
    cases = [
        ("case_0_awgn_only",      gen_channel_awgn,             40.0, 0),
        ("case_1_flat_fading",    gen_channel_flat,             40.0, 1),
        ("case_2_freq_selective", gen_channel_freq_selective,   30.0, 2),
        ("case_3_freq_time_var",  gen_channel_freq_time_var,    25.0, 3),
        ("case_4_low_snr",        gen_channel_low_snr,           0.0, 4),
    ]
    for name, fn, snr, seed in cases:
        gen_case(name, fn, snr, seed, out_root / name)
    print(f"\n✓ All {len(cases)} cases generated under {out_root}")


if __name__ == "__main__":
    main()
