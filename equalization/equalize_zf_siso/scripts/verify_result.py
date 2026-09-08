#!/usr/bin/env python3
"""
verify_result.py — equalizer verifier (full-grid fp16, 3 output streams).

Streams (kernel out vs truth, fp16 [14,1596] natural order):
  out_xhat_re / out_xhat_im / out_no_eff

ALL 1596 used SC/sym verified (padded grid drops nothing; padding stripped by
host). N0 is an INPUT, not checked.

PASS (allclose-style, robust across magnitudes):
  |out - truth| <= ATOL + RTOL*|truth| ; allow <=0.1% fp16 edge mismatches.
"""
import sys, os
from pathlib import Path
import numpy as np

N_SYM, N_SC = 14, 1596
STREAMS = ["xhat_re", "xhat_im", "no_eff"]
ATOL, RTOL, MAX_FAIL_FRAC = 1.0e-2, 3.0e-2, 1.0e-3
CASES = ["case_0_awgn_only", "case_1_flat_fading", "case_2_freq_selective",
         "case_3_freq_time_var", "case_4_low_snr"]


def load(p):
    return np.fromfile(p, dtype=np.float16).reshape(N_SYM, N_SC).astype(np.float32)


def main():
    root = Path(os.environ.get("AIRAN_DATA_DIR",
                               str(Path(__file__).resolve().parent.parent)))
    golden = root / "data" / "golden"
    output = root / "data" / "ascend_output"
    print(f"Golden : {golden}\nOutput : {output}")
    print(f"Criterion: |out-truth| <= {ATOL} + {RTOL}*|truth|, fail-frac <= {MAX_FAIL_FRAC:.1e}, "
          f"all {N_SYM*N_SC} SC/stream\n")

    n_pass = 0
    for name in CASES:
        gd, od = golden / name, output / name
        if not (gd / "truth_xhat_re.bin").exists():
            print(f"  [FAIL] {name}: truth missing");  continue
        if not (od / "out_xhat_re.bin").exists():
            print(f"  [FAIL] {name}: kernel output missing");  continue
        ok = True; rows = []
        for s in STREAMS:
            t = load(gd / f"truth_{s}.bin"); o = load(od / f"out_{s}.bin")
            d = o - t
            bad = np.abs(d) > (ATOL + RTOL * np.abs(t))
            nbad = int(bad.sum()); frac = nbad / bad.size
            s_ok = frac <= MAX_FAIL_FRAC; ok = ok and s_ok
            rows.append((s, float(np.mean(d**2)), float(np.max(np.abs(d))), nbad, frac, s_ok))
        n_pass += int(ok)
        print(f"  {name:<25} {'PASS' if ok else 'FAIL'}")
        for s, mse, mx, nbad, frac, s_ok in rows:
            print(f"      {s:<10} MSE={mse:.4e} maxErr={mx:.4f} fail={nbad} ({frac:.2e})"
                  f"{'' if s_ok else '  <-- FAIL'}")
    print(f"\n{n_pass}/{len(CASES)} PASS")
    sys.exit(0 if n_pass == len(CASES) else 1)


if __name__ == "__main__":
    main()