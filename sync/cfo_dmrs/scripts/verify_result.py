#!/usr/bin/env python3
"""
verify_result.py — Compare cfo_dmrs kernel output vs Python reference.

Invoked by run.sh after the kernel binary has produced
${AIRAN_DATA_DIR}/data/ascend_output/cfo_dmrs_case_N_xxx.bin (8 fp32 per case).

For each case, compares:
  - Kernel's δf (slot 0) vs Python truth (truth.bin, 1 fp32)
  - Sentinel slot 7 == 7.0 (asserts kernel actually wrote)
  - Reports c.re / c.im (slots 1, 2) for diagnostic

Per-case Hz thresholds match cfo_dmrs_ref.py.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Must match cfo_dmrs_ref.py CASE_MANIFEST + DELTA_F_HZ_THRESH
CASES = [
    ('case_0_no_cfo',          1.0),
    ('case_1_small_positive',  5.0),
    ('case_2_large_negative',  5.0),
    ('case_3_doppler_only',    200.0),
    ('case_4_cfo_plus_noise',  50.0),
]

SENTINEL_EXPECTED = 7.0
OUT_F_LEN = 8   # kernel writes 8 fp32


def main():
    data_dir = Path(os.environ.get('AIRAN_DATA_DIR', '.'))
    golden_dir = data_dir / 'data' / 'golden'
    ascend_dir = data_dir / 'data' / 'ascend_output'

    print(f"=== verify_result.py — cfo_dmrs v2 ===")
    print(f"  golden = {golden_dir}")
    print(f"  ascend = {ascend_dir}")
    print()

    if not ascend_dir.exists():
        print(f"  [error] ascend_output dir not found — did the kernel run?")
        return 2

    print(f"  {'case':<24} {'δf_kernel':>10} {'δf_truth':>10} {'err (Hz)':>10}  "
          f"{'thresh':>7}  {'c.re':>10} {'c.im':>10}  result")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10}  {'-'*7}  {'-'*10} {'-'*10}  ------")

    n_pass = 0
    for name, thresh_hz in CASES:
        out_path = ascend_dir / f'cfo_dmrs_{name}.bin'
        truth_path = golden_dir / name / 'truth.bin'

        if not out_path.exists():
            print(f"  {name:<24} [MISSING ASCEND OUTPUT: {out_path.name}]")
            continue
        if not truth_path.exists():
            print(f"  {name:<24} [MISSING TRUTH: {truth_path.name}]")
            continue

        # Kernel output: 8 fp32
        kernel_buf = np.fromfile(out_path, dtype=np.float32)
        if kernel_buf.size < OUT_F_LEN:
            print(f"  {name:<24} [SHORT OUTPUT: {kernel_buf.size} < {OUT_F_LEN}]")
            continue

        df_kernel = float(kernel_buf[0])
        c_re      = float(kernel_buf[1])
        c_im      = float(kernel_buf[2])
        sentinel  = float(kernel_buf[7])

        # Truth: 1 fp32
        truth_buf = np.fromfile(truth_path, dtype=np.float32)
        df_truth = float(truth_buf[0])
        df_err   = df_kernel - df_truth

        sentinel_ok = (sentinel == SENTINEL_EXPECTED)
        err_ok      = abs(df_err) < thresh_hz
        ok = sentinel_ok and err_ok

        if not sentinel_ok:
            tag = f"[NO-SENTINEL got {sentinel}]"
        elif ok:
            tag = "[PASS]"
        else:
            tag = "[FAIL]"

        print(f"  {name:<24} {df_kernel:>+10.3f} {df_truth:>+10.3f} {df_err:>+10.4f}  "
              f"{thresh_hz:>7.1f}  {c_re:>+10.2f} {c_im:>+10.2f}  {tag}")

        if ok:
            n_pass += 1

    print()
    print(f"  Total: {n_pass} / {len(CASES)} PASS")
    return 0 if n_pass == len(CASES) else 1


if __name__ == '__main__':
    sys.exit(main())
