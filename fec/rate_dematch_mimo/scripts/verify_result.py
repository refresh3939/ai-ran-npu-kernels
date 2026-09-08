#!/usr/bin/env python3
"""Verify NPU dumps when a caller elects to preserve the temporary data root."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rate_dematch_ref import C_NUM, LDPC_N, MAX_LAYERS, N_2Z


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    failures = 0
    for layers in range(1, MAX_LAYERS + 1):
        name = f"rank{layers}"
        expected = np.fromfile(
            args.root / "golden" / name / "ldpc_llr.bin", dtype=np.int16)
        actual_path = args.root / "ascend_output" / name / "ldpc_llr.bin"
        if not actual_path.exists():
            print(f"{name}: missing {actual_path}")
            failures += 1
            continue
        actual = np.fromfile(actual_path, dtype=np.int16)
        size_ok = actual.size == C_NUM * LDPC_N
        exact = size_ok and np.array_equal(actual, expected)
        prefix_zero = size_ok and not np.any(actual.reshape(C_NUM, LDPC_N)[:, :N_2Z])
        ok = size_ok and exact and prefix_zero
        print(f"{name}: size={size_ok} exact={exact} prefix_zero={prefix_zero} "
              f"{'PASS' if ok else 'FAIL'}")
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
