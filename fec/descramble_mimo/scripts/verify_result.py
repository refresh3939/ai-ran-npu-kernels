#!/usr/bin/env python3
"""Verify stream permutation, Gold sign multiplication, and zero tails."""
from pathlib import Path

import numpy as np

from gen_data import MAX_LAYERS, NDATA_PAD, NDATA_RE, QM, SLOTS_BY_RANK


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = 0
    for layers in range(1, MAX_LAYERS + 1):
        name = f"rank{layers}"
        slots = SLOTS_BY_RANK[layers - 1]
        stride = layers * NDATA_PAD
        valid = layers * NDATA_RE
        shape = (slots, QM, stride)
        actual = np.fromfile(
            root / "data" / "ascend_output" / name / "llr_nr.bin", dtype=np.int16)
        expected = np.fromfile(
            root / "data" / "golden" / name / "llr_nr.bin", dtype=np.int16)
        size_ok = actual.size == int(np.prod(shape))
        exact = size_ok and np.array_equal(actual, expected)
        tail_zero = False
        if size_ok:
            actual = actual.reshape(shape)
            tail_zero = not np.any(actual[:, :, valid:])
        ok = size_ok and exact and tail_zero
        print(
            f"{name:5s} slots={slots:2d} size={size_ok} exact={exact} "
            f"tail_zero={tail_zero} {'PASS' if ok else 'FAIL'}"
        )
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
