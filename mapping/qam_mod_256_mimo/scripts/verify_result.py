#!/usr/bin/env python3
"""Elementwise Rank-1..4 NPU verification, including compact tails."""
from pathlib import Path

import numpy as np

from gen_data import MAX_LAYERS, NDATA_PAD, NDATA_RE


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = 0
    for layers in range(1, MAX_LAYERS + 1):
        name = f"rank{layers}"
        stride = layers * NDATA_PAD
        valid = layers * NDATA_RE
        golden = root / "data" / "golden" / name
        output = root / "data" / "ascend_output" / name
        expected_re = np.fromfile(golden / "d_re.bin", dtype=np.uint16)
        expected_im = np.fromfile(golden / "d_im.bin", dtype=np.uint16)
        actual_re = np.fromfile(output / "d_re.bin", dtype=np.uint16)
        actual_im = np.fromfile(output / "d_im.bin", dtype=np.uint16)
        size_ok = all(x.size == stride for x in (
            expected_re, expected_im, actual_re, actual_im
        ))
        exact = size_ok and np.array_equal(actual_re, expected_re) \
            and np.array_equal(actual_im, expected_im)
        valid_exact = size_ok and np.array_equal(
            actual_re[:valid], expected_re[:valid]
        ) and np.array_equal(actual_im[:valid], expected_im[:valid])
        tail_zero = size_ok and not np.any(actual_re[valid:]) \
            and not np.any(actual_im[valid:])
        ok = size_ok and exact and valid_exact and tail_zero
        print(
            f"{name:5s} elements={stride:6d} size={size_ok} "
            f"valid_exact={valid_exact} tail_zero={tail_zero} "
            f"all_exact={exact} {'PASS' if ok else 'FAIL'}"
        )
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
