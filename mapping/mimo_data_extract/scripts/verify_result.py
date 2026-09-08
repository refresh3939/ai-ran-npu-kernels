#!/usr/bin/env python3
"""Verify exact values, ordering, and output padding for mimo_data_extract."""
from pathlib import Path

import numpy as np

from gen_data import CASES, NDATA_PAD, NDATA_RE


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = 0
    for name, layers, _ in CASES:
        exact = True
        tail_zero = True
        for plane in ("re", "im", "no_eff"):
            actual = np.fromfile(
                root / "data" / "ascend_output" / name / f"data_{plane}.bin",
                dtype=np.float16,
            )
            expected = np.fromfile(
                root / "data" / "golden" / name / f"data_{plane}.bin",
                dtype=np.float16,
            )
            if actual.size != layers * NDATA_PAD:
                exact = False
                tail_zero = False
                continue
            actual = actual.reshape(layers, NDATA_PAD)
            expected = expected.reshape(layers, NDATA_PAD)
            exact &= np.array_equal(actual.view(np.uint16), expected.view(np.uint16))
            tail_zero &= not np.any(actual[:, NDATA_RE:].view(np.uint16))
        ok = exact and tail_zero
        print(
            f"{name:16s} exact={exact} tail_zero={tail_zero} "
            f"{'PASS' if ok else 'FAIL'}"
        )
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
