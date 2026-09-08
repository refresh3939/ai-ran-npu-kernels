#!/usr/bin/env python3
# ============================================================================
# pss_acquisition_ref.py — Layer 1 PSS-based acquisition reference
#
# 项目位置: kernels/rx/pss_acquisition/scripts/pss_acquisition_ref.py
#
# 算法 (3GPP TS 38.211 §7.4.2.2):
#   For each freq_hyp in {-50..+50 kHz, step 5 kHz} (21 hyp):
#     rx_derot = rx · exp(-j·2π·freq_hyp·n·Ts)
#     For each pss_id in {0,1,2}:
#       For each time_offset in search window:
#         corr_metric = |Σ conj(pss_tmpl[pss_id]) · rx_derot[t:t+N_FFT]|²
#   argmax → (best_freq, best_pss_id, best_time)
#
# Outputs (data/golden/rx/pss_acquisition/case_X_*/):
#   input.bin    int16 (61440,)    IQ interleaved, Q8.8
#   truth.bin    3 fp32            [injected_cfo_hz, injected_pss_id, injected_time_shift]
#   output.bin   4 fp32            [est_cfo_hz, est_pss_id, est_time_shift, peak_metric]
#   meta.json    metadata
# ============================================================================

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ─── 5G NR μ=1 + SSB constants ──────────────────────────────────────────────
N_FFT             = 2048
SCS_HZ            = 30_000
SAMPLE_RATE_HZ    = N_FFT * SCS_HZ                # 61_440_000
CP_LEN_FIRST      = 176
CP_LEN_OTHER      = 144
SYM_LEN           = N_FFT + CP_LEN_OTHER          # 2192
Q_SCALE           = 256
INT16_MAX         = 32767

N_SC_SSB          = 240
N_SC_PSS          = 127
SYMBOLS_PER_SSB   = 4
SSB_START_SYM     = 0
N_SYMBOL_PER_SLOT = 14
N_SAMPLE_PER_SLOT = CP_LEN_FIRST + N_FFT + (N_SYMBOL_PER_SLOT - 1) * SYM_LEN
assert N_SAMPLE_PER_SLOT == 30720

# ─── Hypothesis search params ───────────────────────────────────────────────
FREQ_HYP_HZ       = np.arange(-50_000, 50_001, 5_000, dtype=np.float64)
N_FREQ_HYP        = len(FREQ_HYP_HZ)              # 21
PSS_IDS           = [0, 1, 2]
N_PSS             = 3
TIME_SEARCH_HALF  = 200
N_TIME_OFFSETS    = 2 * TIME_SEARCH_HALF + 1      # 401

CFO_TOL_HZ        = 2_500.0
TIME_TOL_SAMPLES  = 2

TEST_CASES = [
    (    0.0, 0,    0),
    ( +500.0, 0,    0),
    (+25000.0, 1,  +50),
    (-45000.0, 2,  -80),
    (+12500.0, 0, +130),
]


def gen_pss_sequence(N_ID_2: int) -> np.ndarray:
    """5G NR PSS BPSK sequence, length 127."""
    x = np.zeros(127 + 7, dtype=np.int8)
    x[0:7] = [0, 1, 1, 0, 1, 1, 1]
    for i in range(120):
        x[i + 7] = (x[i + 4] + x[i]) % 2
    d_pss = np.zeros(127, dtype=np.float64)
    for n in range(127):
        d_pss[n] = 1.0 - 2.0 * x[(n + 43 * N_ID_2) % 127]
    return d_pss


def pss_time_domain_template(N_ID_2: int) -> np.ndarray:
    """N_FFT-long complex time-domain PSS template (for correlation)."""
    pss_freq = gen_pss_sequence(N_ID_2)
    sc_grid = np.zeros(N_SC_SSB, dtype=np.complex128)
    sc_grid[56:56 + N_SC_PSS] = pss_freq

    half_ssb = N_SC_SSB // 2
    center = N_FFT // 2
    x_shift = np.zeros(N_FFT, dtype=np.complex128)
    x_shift[center - half_ssb : center + half_ssb] = sc_grid

    x_freq = np.fft.ifftshift(x_shift)
    return np.fft.ifft(x_freq) * N_FFT


def synth_slot_with_ssb(pss_id, time_shift_samples, rng):
    """1-slot complex time domain, SSB at symbol 0..3, rest random QPSK."""
    qpsk = np.array([+1+1j, -1+1j, +1-1j, -1-1j], dtype=np.complex128) / np.sqrt(2)
    half_ssb = N_SC_SSB // 2
    center = N_FFT // 2

    def random_data_symbol():
        N_SC_USED = 1596
        x_shift = np.zeros(N_FFT, dtype=np.complex128)
        hu = N_SC_USED // 2
        x_shift[center - hu : center]              = qpsk[rng.integers(0, 4, size=hu)]
        x_shift[center + 1 : center + 1 + hu]      = qpsk[rng.integers(0, 4, size=hu)]
        x_freq = np.fft.ifftshift(x_shift)
        return np.fft.ifft(x_freq) * (N_FFT / np.sqrt(N_SC_USED))

    def ssb_sym(idx_in_ssb):
        sc = np.zeros(N_SC_SSB, dtype=np.complex128)
        if idx_in_ssb == 0:
            sc[56:56 + N_SC_PSS] = gen_pss_sequence(pss_id)
        elif idx_in_ssb == 2:
            sc[:] = (rng.integers(0, 2, size=N_SC_SSB) * 2 - 1).astype(np.complex128)
        else:
            sc[:] = qpsk[rng.integers(0, 4, size=N_SC_SSB)]

        x_shift = np.zeros(N_FFT, dtype=np.complex128)
        x_shift[center - half_ssb : center + half_ssb] = sc
        x_freq = np.fft.ifftshift(x_shift)
        return np.fft.ifft(x_freq) * (N_FFT / np.sqrt(N_SC_SSB))

    out = np.zeros(N_SAMPLE_PER_SLOT, dtype=np.complex128)
    cursor = 0
    for s in range(N_SYMBOL_PER_SLOT):
        cp_len = CP_LEN_FIRST if s == 0 else CP_LEN_OTHER
        if SSB_START_SYM <= s < SSB_START_SYM + SYMBOLS_PER_SSB:
            x_time = ssb_sym(s - SSB_START_SYM)
        else:
            x_time = random_data_symbol()
        cp = x_time[-cp_len:]
        sym_full = np.concatenate([cp, x_time])
        out[cursor : cursor + cp_len + N_FFT] = sym_full
        cursor += cp_len + N_FFT
    assert cursor == N_SAMPLE_PER_SLOT

    if time_shift_samples != 0:
        out = np.roll(out, time_shift_samples)
    return out


def inject_cfo(x, delta_f_hz):
    n = np.arange(x.shape[0], dtype=np.float64)
    return x * np.exp(1j * 2.0 * np.pi * delta_f_hz * n / SAMPLE_RATE_HZ)


def quantize_to_int16(x_complex):
    iq = np.empty(x_complex.shape[0] * 2, dtype=np.float64)
    iq[0::2] = x_complex.real
    iq[1::2] = x_complex.imag
    return np.clip(np.round(iq * Q_SCALE), -INT16_MAX, INT16_MAX).astype(np.int16)


def dequantize_from_int16(iq_int16):
    iq_f = iq_int16.astype(np.float64) / Q_SCALE
    return iq_f[0::2] + 1j * iq_f[1::2]


def pss_acquisition(rx, expected_ssb_start):
    """Hypothesis search over (freq, pss_id, time)."""
    pss_templates_conj = [np.conj(pss_time_domain_template(p)) for p in PSS_IDS]
    n_arr = np.arange(rx.shape[0], dtype=np.float64)

    best = {"metric": -np.inf, "freq_hz": 0.0, "pss_id": 0, "time_shift": 0}
    time_offsets = np.arange(-TIME_SEARCH_HALF, TIME_SEARCH_HALF + 1)

    for freq_hz in FREQ_HYP_HZ:
        derot = rx * np.exp(-1j * 2.0 * np.pi * freq_hz * n_arr / SAMPLE_RATE_HZ)

        for pss_id_idx, pss_id in enumerate(PSS_IDS):
            tmpl_conj = pss_templates_conj[pss_id_idx]

            for time_shift in time_offsets:
                start = expected_ssb_start + time_shift
                if start < 0 or start + N_FFT > rx.shape[0]:
                    continue
                window = derot[start : start + N_FFT]
                metric = float(np.abs(np.sum(tmpl_conj * window)) ** 2)
                if metric > best["metric"]:
                    best["metric"]     = metric
                    best["freq_hz"]    = float(freq_hz)
                    best["pss_id"]     = int(pss_id)
                    best["time_shift"] = int(time_shift)
    return best


def resolve_out_root(cli_out):
    if cli_out:
        return Path(cli_out)
    env = os.environ.get("AIRAN_DATA_DIR")
    if env:
        return Path(env) / "golden" / "rx" / "pss_acquisition"
    here = Path(__file__).resolve()
    return here.parents[4] / "data" / "golden" / "rx" / "pss_acquisition"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", nargs="?", default=None)
    p.add_argument("--seed", type=lambda s: int(s, 0), default=0xC0FE0002)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    out_root = resolve_out_root(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # SSB starts at sample CP_LEN_FIRST (skipping symbol 0's CP)
    expected_ssb_start = CP_LEN_FIRST   # = 176

    print(f"[ref] output:         {out_root}")
    print(f"[ref] seed:           0x{args.seed:08X}")
    print(f"[ref] freq hyps:      {N_FREQ_HYP} from {FREQ_HYP_HZ[0]:.0f} to {FREQ_HYP_HZ[-1]:.0f} Hz")
    print(f"[ref] pss cands:      {N_PSS}")
    print(f"[ref] time offsets:   {N_TIME_OFFSETS} around start={expected_ssb_start}")
    total_hyp = N_FREQ_HYP * N_PSS * N_TIME_OFFSETS
    print(f"[ref] total hyps/case: {total_hyp}, ops≈{total_hyp * N_FFT // 1_000_000}M")
    print()

    rng = np.random.default_rng(args.seed)

    results = []
    for case_id, (cfo_hz, pss_id, time_shift) in enumerate(TEST_CASES):
        case_dir = out_root / f"case_{case_id}_pss{pss_id}_cfo{int(cfo_hz):+d}_t{time_shift:+d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        clean = synth_slot_with_ssb(pss_id, time_shift, rng)
        rx_cplx = inject_cfo(clean, cfo_hz)
        iq_i16 = quantize_to_int16(rx_cplx)
        rx_dequant = dequantize_from_int16(iq_i16)

        t0 = time.time()
        result = pss_acquisition(rx_dequant, expected_ssb_start)
        elapsed = time.time() - t0

        iq_i16.tofile(case_dir / "input.bin")
        np.array([cfo_hz, float(pss_id), float(time_shift)],
                 dtype=np.float32).tofile(case_dir / "truth.bin")
        np.array([result["freq_hz"], float(result["pss_id"]),
                  float(result["time_shift"]), float(result["metric"])],
                 dtype=np.float32).tofile(case_dir / "output.bin")

        meta = {
            "case_id":              case_id,
            "injected_cfo_hz":      float(cfo_hz),
            "injected_pss_id":      int(pss_id),
            "injected_time_shift":  int(time_shift),
            "est_cfo_hz":           float(result["freq_hz"]),
            "est_pss_id":           int(result["pss_id"]),
            "est_time_shift":       int(result["time_shift"]),
            "peak_metric":          float(result["metric"]),
            "expected_ssb_start":   int(expected_ssb_start),
            "freq_hyp_step_hz":     5000,
            "n_freq_hyp":           N_FREQ_HYP,
            "n_pss":                N_PSS,
            "time_search_half":     TIME_SEARCH_HALF,
            "elapsed_seconds":      float(elapsed),
            "seed":                 f"0x{args.seed:08X}",
        }
        with open(case_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        results.append(meta)
        print(f"[case {case_id}] inject (cfo={cfo_hz:+.0f}Hz, pss={pss_id}, t={time_shift:+d})  "
              f"→ est (cfo={result['freq_hz']:+.0f}Hz, pss={result['pss_id']}, t={result['time_shift']:+d})  "
              f"metric={result['metric']:.2e}  [{elapsed:.1f}s]")

    if not args.no_verify:
        print()
        print("=" * 100)
        print(" Verification: injected vs estimated")
        print("=" * 100)
        print(f"{'case':<5} {'inj cfo':>10} {'est cfo':>10} {'inj pss':>8} {'est pss':>8} "
              f"{'inj t':>7} {'est t':>7} {'verdict':>9}")
        print("-" * 100)
        all_pass = True
        for r in results:
            cfo_ok = abs(r["est_cfo_hz"] - r["injected_cfo_hz"]) <= CFO_TOL_HZ
            pss_ok = r["est_pss_id"] == r["injected_pss_id"]
            time_ok = abs(r["est_time_shift"] - r["injected_time_shift"]) <= TIME_TOL_SAMPLES
            ok = cfo_ok and pss_ok and time_ok
            mark = "PASS" if ok else "FAIL"
            print(f"{r['case_id']:<5} {r['injected_cfo_hz']:>+10.0f} {r['est_cfo_hz']:>+10.0f} "
                  f"{r['injected_pss_id']:>8} {r['est_pss_id']:>8} "
                  f"{r['injected_time_shift']:>+7d} {r['est_time_shift']:>+7d}  {mark}")
            if not ok:
                all_pass = False
                if not cfo_ok: print(f"       ↑ CFO err {abs(r['est_cfo_hz']-r['injected_cfo_hz']):.0f} Hz > {CFO_TOL_HZ:.0f}")
                if not pss_ok: print(f"       ↑ PSS mismatch")
                if not time_ok: print(f"       ↑ time err {abs(r['est_time_shift']-r['injected_time_shift'])} samples > {TIME_TOL_SAMPLES}")
        print("=" * 100)
        if all_pass:
            print("结果: 全部 PASS")
        else:
            print("结果: 有 FAIL")
            sys.exit(1)


if __name__ == "__main__":
    main()
