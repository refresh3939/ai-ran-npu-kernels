#!/usr/bin/env python3
"""Verify NPU output, canonical stream order, and non-consuming zero tails."""
import os
from pathlib import Path

import numpy as np

from gen_data import MAX_LAYERS, MAX_SLOTS, N_DATA_PAD, N_DATA_RE, QM


def main() -> int:
    data_root = Path(os.environ.get("AIRAN_DATA_DIR", Path(__file__).parents[1] / "data"))
    failures = 0
    for layers in range(1, MAX_LAYERS + 1):
        name = f"rank{layers}"
        valid = layers * N_DATA_RE
        stride = layers * N_DATA_PAD
        shape = (MAX_SLOTS, QM, stride)
        expected = np.fromfile(
            data_root / "golden" / name / "bits_nr.bin", dtype=np.int16
        )
        actual = np.fromfile(
            data_root / "ascend_output" / name / "bits_nr.bin", dtype=np.int16
        )
        expected_size = int(np.prod(shape))
        size_ok = expected.size == expected_size and actual.size == expected_size
        exact = size_ok and np.array_equal(actual, expected)
        tail_zero = domain = False
        if size_ok:
            actual = actual.reshape(shape)
            tail_zero = not np.any(actual[:, :, valid:])
            domain = set(np.unique(actual[:, :, :valid]).tolist()) <= {0, 1}
        ok = size_ok and exact and tail_zero and domain
        print(
            f"{name:5s} slots={MAX_SLOTS:2d} size={size_ok} exact={exact} "
            f"tail_zero={tail_zero} domain={domain} "
            f"{'PASS' if ok else 'FAIL'}"
        )
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
