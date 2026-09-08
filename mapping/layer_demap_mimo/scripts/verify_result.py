#!/usr/bin/env python3
"""Verify bit-exact layer_demap output, order, and zero padding."""
from pathlib import Path

import numpy as np

from gen_data import NDATA_PAD, NDATA_RE, QM


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = 0
    for layers in range(1, 5):
        name = f"rank{layers}"
        shape = (QM, layers * NDATA_PAD)
        actual_path = root / "data" / "ascend_output" / name / "cw_llr.bin"
        expected_path = root / "data" / "golden" / name / "cw_llr.bin"
        actual = np.fromfile(actual_path, dtype=np.int16)
        expected = np.fromfile(expected_path, dtype=np.int16)
        size_ok = actual.size == np.prod(shape)
        if size_ok:
            actual = actual.reshape(shape)
            expected = expected.reshape(shape)
            exact = np.array_equal(actual, expected)
            tail_zero = not np.any(actual[:, layers * NDATA_RE :])
        else:
            exact = False
            tail_zero = False
        ok = size_ok and exact and tail_zero
        print(
            f"{name:5s} size={size_ok} exact={exact} "
            f"tail_zero={tail_zero} {'PASS' if ok else 'FAIL'}"
        )
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
