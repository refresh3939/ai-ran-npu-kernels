#!/usr/bin/env python3
"""
pss_cfo_estimator_ref.py — numpy reference implementation + golden generator

Algorithm: Tuninato 2023 Eq. 15-17, Wang-Berggren 2018 path.

Generates 5 test cases under data/golden/case_N_xxx/:
  - input.bin  : y_at_pss[256] cint16 Q1.15  (1024 B)
  - pilot.bin  : pss_ref_128[128] cint16 Q1.15  (512 B)
  - truth.bin  : [delta_f_true (fp32), phase_diff_true (fp32)]  (8 B)

PSS sequence generated from scratch via 3GPP TS 38.211 §7.4.2.2.1 m-sequence
recursion x(i+7) = (x(i+4)+x(i)) mod 2, x_init = [1,1,1,0,1,1,0].
No external py3gpp / OAI / srsRAN source consulted (clean room).
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

# ─── Parameters (must match pss_cfo_estimator.h) ───────────────────────────
N_PSS       = 256                  # matched-filter length
N_HALF      = 128
N_FFT       = 256                  # SSB FFT size
FS_HZ       = 7.68e6               # SSB rate
SCS_HZ      = 30000.0              # FR1, dummy here — Δf is in Hz
Q_SCALE     = 32768                # Q1.15
PSS_LEN_FD  = 127                  # frequency-domain PSS length
PSS_DC_OFFSET = 56                 # PSS occupies subcarriers [56..182] of 240-SC SSB

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "golden"


# ─── 3GPP TS 38.211 §7.4.2.2.1: PSS m-sequence generation ──────────────────
def gen_pss_freq(n_id_2: int) -> np.ndarray:
    """Generate frequency-domain PSS d_PSS(n) for n_id_2 ∈ {0,1,2}.

    3GPP TS 38.211 §7.4.2.2.1:
        x(i+7) = (x(i+4) + x(i)) mod 2,  x_init = [0,1,1,0,1,1,1]   # paper-order
        d_PSS(n) = 1 - 2*x((n + 43*n_id_2) mod 127)
    Output: real ±1, length 127.
    """
    # Generate base m-sequence of length 127
    x = np.zeros(127, dtype=np.int8)
    # 38.211 lists x(0..6) = [0, 1, 1, 0, 1, 1, 1] reading from x(6) down to x(0).
    # i.e. x(0)=0, x(1)=1, x(2)=1, x(3)=0, x(4)=1, x(5)=1, x(6)=1.
    x[0:7] = np.array([0, 1, 1, 0, 1, 1, 1], dtype=np.int8)
    for i in range(120):
        x[i + 7] = (x[i + 4] + x[i]) % 2

    n = np.arange(127)
    d = 1 - 2 * x[(n + 43 * n_id_2) % 127].astype(np.int32)
    return d.astype(np.float32)


def pss_time_domain(n_id_2: int, n_fft: int = N_FFT) -> np.ndarray:
    """Build time-domain PSS reference d^(l)_PSS(n) of length N_FFT.

    Steps:
      1. Generate frequency-domain BPSK sequence d_PSS (length 127).
      2. Map onto N_FFT-wide grid at DC-centred subcarriers (PSS in SSB
         occupies the middle 127 of 240 SCs; we centre them around DC).
      3. IFFT (so that low frequencies are around index 0 / N_FFT-1).
      4. Normalize to unit energy.
    """
    pss_fd = gen_pss_freq(n_id_2)              # length 127, ±1
    grid = np.zeros(n_fft, dtype=np.complex64)

    # Place 127 PSS subcarriers centred on DC.
    # IFFT-shift: positive freq → low index, negative freq → high index.
    half = PSS_LEN_FD // 2                     # = 63
    # subcarrier indices -63..-1 → high N_FFT indices, 0..+63 → low N_FFT indices
    # We use a layout: idx_FFT = (k + N_FFT) % N_FFT, for k = -63..+63
    for k_idx, k in enumerate(range(-half, half + 1)):
        grid[k % n_fft] = pss_fd[k_idx]

    # IFFT and normalise. Use np.fft.ifft with default 1/N scaling.
    td = np.fft.ifft(grid).astype(np.complex64)
    # Normalise to unit power (||td||^2 / N = 1)
    rms = np.sqrt(np.mean(np.abs(td) ** 2))
    td = td / max(rms, 1e-30)
    # Now scale down so that quantization to Q1.15 doesn't clip.
    # max |td| after RMS normalisation is sequence-dependent. Scale to 0.7.
    peak = np.max(np.abs(td))
    td = td * (0.7 / max(peak, 1e-30))
    return td.astype(np.complex64)


# ─── Q1.15 quantization ────────────────────────────────────────────────────
def to_cint16(c: np.ndarray) -> np.ndarray:
    """Quantize complex array to interleaved cint16 Q1.15 (real,imag,real,imag,...)."""
    re = np.round(c.real * Q_SCALE).clip(-32768, 32767).astype(np.int16)
    im = np.round(c.imag * Q_SCALE).clip(-32768, 32767).astype(np.int16)
    out = np.empty(2 * len(c), dtype=np.int16)
    out[0::2] = re
    out[1::2] = im
    return out


def from_cint16(buf: np.ndarray) -> np.ndarray:
    """Inverse of to_cint16."""
    re = buf[0::2].astype(np.float32) / Q_SCALE
    im = buf[1::2].astype(np.float32) / Q_SCALE
    return (re + 1j * im).astype(np.complex64)


# Matches kernel's MAG2_D_FLOOR — if |D|² is below this, atan2 returns
# undefined direction; we explicitly output (0, 0) to mirror kernel behavior.
MAG2_D_FLOOR = 1.0e-6


# ─── Reference algorithm (matches kernel exactly in float64) ───────────────
def pss_cfo_estimator_ref(y_at_pss: np.ndarray,
                          pss_ref: np.ndarray,
                          fs_hz: float = FS_HZ,
                          n_half: int = N_HALF) -> tuple[float, float, float, float]:
    """Reference float64 implementation of the algorithm.

    Per the corrected derivation (each y-half correlates with its OWN ref half):

        C_0 = Σ y[0:128]   · conj(ref[0:128])
        C_1 = Σ y[128:256] · conj(ref[128:256])
        Δθ  = arg(C_1 · conj(C_0))
        Δf  = Δθ · fs / (2π · N_HALF)

    Degenerate guard: if |D|² < MAG2_D_FLOOR (signal at noise floor), returns
    (0, 0, |C_0|², |C_1|²) — matches kernel's guard.

    Returns (delta_f_frac_Hz, phase_diff_rad, |C_0|², |C_1|²).

    Note: a normalized coherence |D|²/(|C_0|²·|C_1|²) is identically 1 for
    any two complex numbers (algebraic identity); we don't compute it.
    """
    assert y_at_pss.shape == (256,)
    assert pss_ref.shape == (256,)

    c0 = np.sum(y_at_pss[0:128]   * np.conj(pss_ref[0:128]))
    c1 = np.sum(y_at_pss[128:256] * np.conj(pss_ref[128:256]))

    D       = c1 * np.conj(c0)
    mag2_D  = float((D * np.conj(D)).real)
    mag2_c0 = float((c0 * np.conj(c0)).real)
    mag2_c1 = float((c1 * np.conj(c1)).real)

    if mag2_D < MAG2_D_FLOOR:
        return 0.0, 0.0, mag2_c0, mag2_c1

    delta_theta = np.angle(D)
    delta_f = delta_theta * fs_hz / (2.0 * np.pi * n_half)

    return float(delta_f), float(delta_theta), mag2_c0, mag2_c1


# ─── Test case generation ─────────────────────────────────────────────────
def build_case(name: str, n_id_2: int, delta_f_hz: float,
               snr_db: float | None = None,
               rng_seed: int = 0):
    """Generate y_at_pss (cint16) + pilot (cint16) + truth (fp32) for one case.

    Model:
      1. Take PSS time-domain reference d^(n_id_2)_PSS(n) (256 samples).
      2. Apply true CFO rotation: y(n) = d(n) · exp(j · 2π · Δf · n / fs)
      3. (Optional) add complex AWGN at given SNR.
      4. Quantize to cint16 Q1.15.
      5. truth = (delta_f_hz_after_quantization, expected_phase_diff).
         We compute these *from the post-quantization signal* so that the
         kernel-vs-truth comparison absorbs quantization noise.
    """
    rng = np.random.default_rng(rng_seed)

    # 1. Time-domain reference (full 256-pt)
    pss_td_full = pss_time_domain(n_id_2)        # complex64, length 256

    # 2. Apply CFO
    n = np.arange(N_PSS)
    rot = np.exp(1j * 2.0 * np.pi * delta_f_hz * n / FS_HZ).astype(np.complex64)
    y = (pss_td_full * rot).astype(np.complex64)

    # 3. AWGN (optional)
    if snr_db is not None:
        sig_pow = np.mean(np.abs(y) ** 2)
        noise_pow = sig_pow / (10 ** (snr_db / 10.0))
        sigma = np.sqrt(noise_pow / 2.0)  # per real/imag axis
        noise = (rng.standard_normal(N_PSS) + 1j * rng.standard_normal(N_PSS)).astype(np.complex64)
        y = y + (sigma * noise).astype(np.complex64)

    # 4. Quantize to cint16
    y_cint16     = to_cint16(y)
    pilot_cint16 = to_cint16(pss_td_full)        # FULL 256-pt PSS

    # 5. Compute truth from the *de-quantized* arrays (so the kernel sees
    #    the same bits as the reference algorithm). The kernel does
    #    cast-int16-then-Muls(1/32768), which is exactly from_cint16().
    y_dq     = from_cint16(y_cint16)
    pilot_dq = from_cint16(pilot_cint16)
    delta_f_true, theta_true, mag2_c0, mag2_c1 = pss_cfo_estimator_ref(y_dq, pilot_dq)

    # Write files. truth.bin remains 2 fp32 (Δf + Δθ); kernel-side mag2 values
    # are diagnostic and not part of the algorithm contract, so we only print them.
    case_dir = OUT_DIR / name
    case_dir.mkdir(parents=True, exist_ok=True)
    y_cint16.tofile(case_dir / "input.bin")
    pilot_cint16.tofile(case_dir / "pilot.bin")
    with open(case_dir / "truth.bin", "wb") as f:
        f.write(struct.pack("<ff", float(delta_f_true), float(theta_true)))

    print(f"  {name:<30s}  Δf_set={delta_f_hz:+9.1f} Hz  "
          f"→  Δf_truth={delta_f_true:+9.2f}  θ_truth={theta_true:+.4f}  "
          f"|C_0|²={mag2_c0:7.2f} |C_1|²={mag2_c1:7.2f}"
          + (f"  (SNR={snr_db} dB)" if snr_db is not None else ""))


def main():
    print(f"[ref] generating golden cases under {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Spec §7.1 test vectors, mapped to our 5 PASS cases.
    # n_id_2 is held at 0 for all cases (kernel doesn't depend on which PSS;
    # we just need a consistent ref).
    build_case("case_0_no_cfo",         n_id_2=0, delta_f_hz=0.0)
    build_case("case_1_small_positive", n_id_2=0, delta_f_hz=500.0)
    build_case("case_2_mid_positive",   n_id_2=0, delta_f_hz=5000.0)
    build_case("case_3_negative",       n_id_2=0, delta_f_hz=-10000.0)
    build_case("case_4_cfo_plus_noise", n_id_2=0, delta_f_hz=5000.0,
                                         snr_db=10.0, rng_seed=42)

    print("[ref] done.")


if __name__ == "__main__":
    main()
