#!/usr/bin/env python3
"""
verify_result.py — Compare NPU kernel output against truth and Python NPU-path golden.

Reads (paths are kernel-local now):
  - <kernel_dir>/data/ascend_output/case_X_*.bin      (NPU kernel: [R_re, R_im, Δf, pad...] fp32, 32B)
  - <kernel_dir>/data/golden/case_X_*/truth.bin       (injected truth, fp32)
  - <kernel_dir>/data/golden/case_X_*/output.bin      (Python NPU-path est, fp32)
  - <kernel_dir>/data/golden/case_X_*/R_complex.bin   (Python R_re, R_im, fp32)

Reports per case:
  - truth vs NPU kernel est (Hz error)
  - Python NPU-path est vs NPU kernel est (ULP-ish drift; should be ~0)
  - R complex agreement
"""

import os
import sys
from pathlib import Path
import numpy as np


CASES = [
    ("case_0_cfo_+0hz",      0.0),
    ("case_1_cfo_+500hz",   +500.0),
    ("case_2_cfo_-1200hz", -1200.0),
    ("case_3_cfo_+5000hz", +5000.0),
    ("case_4_cfo_-10000hz",-10000.0),
]


def kernel_root() -> Path:
    """
    Resolve kernel directory.
      1. AIRAN_DATA_DIR env (set by run.sh to kernel dir)
      2. fallback: this script's parents[1] (.../cfo_estimate/scripts/.. = cfo_estimate/)
    """
    env = os.environ.get("AIRAN_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1]


def tol_hz(truth: float) -> float:
    return max(50.0, abs(truth) * 0.05)


def main():
    root = kernel_root()
    golden_root = root / "data" / "golden"
    ascend_root = root / "data" / "ascend_output"

    if not ascend_root.exists():
        print(f"[verify] FAIL: {ascend_root} 不存在,kernel 没跑")
        sys.exit(1)

    print(f"[verify] golden:  {golden_root}")
    print(f"[verify] ascend:  {ascend_root}")
    print()
    print("=" * 100)
    print(f"{'case':<22} {'truth':>10} {'NPU est':>11} {'kernel-truth':>13} "
          f"{'ref est':>11} {'kernel-ref':>11} {'verdict':>8}")
    print("-" * 100)

    n_pass = 0
    for dir_name, truth in CASES:
        kpath = ascend_root / f"{dir_name}.bin"
        tpath = golden_root / dir_name / "truth.bin"
        opath = golden_root / dir_name / "output.bin"
        rpath = golden_root / dir_name / "R_complex.bin"

        if not kpath.exists():
            print(f"  {dir_name:<22} MISSING kernel output: {kpath}")
            continue

        kernel_out = np.fromfile(kpath, dtype=np.float32)
        truth_f    = float(np.fromfile(tpath, dtype=np.float32)[0])
        ref_npu    = float(np.fromfile(opath, dtype=np.float32)[0])
        ref_R      = np.fromfile(rpath, dtype=np.float32)

        kernel_R_re = float(kernel_out[0])
        kernel_R_im = float(kernel_out[1])
        kernel_est  = float(kernel_out[2])

        err_truth  = abs(kernel_est - truth_f)
        err_ref    = abs(kernel_est - ref_npu)
        tol        = tol_hz(truth_f)
        ok         = err_truth < tol
        mark       = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1

        print(f"  {dir_name:<22} {truth_f:>+10.2f} {kernel_est:>+11.4f} "
              f"{err_truth:>13.4f} {ref_npu:>+11.4f} {err_ref:>11.6f}     {mark}")

        # R 复数对比 (sanity, 不参与 pass/fail 但帮助调试)
        if not ok or err_ref > 1.0:
            dR_re = abs(kernel_R_re - ref_R[0])
            dR_im = abs(kernel_R_im - ref_R[1])
            print(f"    ↪ R debug: kernel=({kernel_R_re:+.4f},{kernel_R_im:+.4f}) "
                  f"ref=({float(ref_R[0]):+.4f},{float(ref_R[1]):+.4f})  "
                  f"|ΔR|=({dR_re:.4e},{dR_im:.4e})")

    print("=" * 100)
    print(f"[verify] {n_pass}/{len(CASES)} PASS")

    if n_pass < len(CASES):
        sys.exit(1)


if __name__ == "__main__":
    main()