#!/usr/bin/env python3
"""
verify_result.py — Compare dmrs_gen kernel output vs Python golden.

Reads ${AIRAN_DATA_DIR}/data/ascend_output/dmrs_gen_case_N_xxx_{re,im}.bin
([2,896] fp16 each) and compares to data/golden/case_N_xxx/x_{re,im}.bin.
Expect bit-exact (max |err| == 0).
"""

import os
import sys
import numpy as np
from pathlib import Path

CASES = [
    "case_0_baseline",
    "case_1_nid0",
    "case_2_slot7",
    "case_3_nscid1",
    "case_4_nid_large",
]
ERR_THRESH = 1e-3   # fp16 ULP; expect exact 0


def main():
    data_dir = Path(os.environ.get("AIRAN_DATA_DIR", "."))
    golden_dir = data_dir / "data" / "golden"
    ascend_dir = data_dir / "data" / "ascend_output"

    print("=== verify_result.py — dmrs_gen ===")
    print(f"  golden = {golden_dir}")
    print(f"  ascend = {ascend_dir}")
    print()

    if not ascend_dir.exists():
        print("  [error] ascend_output dir not found — did the kernel run?")
        return 2

    print(f"  {'case':<20} {'re max|err|':>12} {'im max|err|':>12}  result")
    print(f"  {'-'*20} {'-'*12} {'-'*12}  ------")

    n_pass = 0
    for name in CASES:
        re_g = golden_dir / name / "x_re.bin"
        im_g = golden_dir / name / "x_im.bin"
        re_a = ascend_dir / f"dmrs_gen_{name}_re.bin"
        im_a = ascend_dir / f"dmrs_gen_{name}_im.bin"

        if not (re_g.exists() and im_g.exists() and re_a.exists() and im_a.exists()):
            print(f"  {name:<20} [MISSING FILE]")
            continue

        g_re = np.fromfile(re_g, dtype=np.float16).astype(np.float32)
        g_im = np.fromfile(im_g, dtype=np.float16).astype(np.float32)
        a_re = np.fromfile(re_a, dtype=np.float16).astype(np.float32)
        a_im = np.fromfile(im_a, dtype=np.float16).astype(np.float32)

        if a_re.size != g_re.size or a_im.size != g_im.size:
            print(f"  {name:<20} [SIZE MISMATCH]")
            continue

        re_err = float(np.max(np.abs(a_re - g_re)))
        im_err = float(np.max(np.abs(a_im - g_im)))
        ok = (re_err < ERR_THRESH) and (im_err < ERR_THRESH)

        print(f"  {name:<20} {re_err:>12.5f} {im_err:>12.5f}  {'[PASS]' if ok else '[FAIL]'}")
        if ok:
            n_pass += 1

    print()
    print(f"  Total: {n_pass} / {len(CASES)} PASS")
    return 0 if n_pass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
