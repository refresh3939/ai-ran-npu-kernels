#!/usr/bin/env python3
"""Verify physical [L,8,12,1600] QAM batch output bit-exactly."""
from pathlib import Path

import numpy as np

from gen_data import LLR_PAD, LAYERS_MAX, NSC_USED, QM


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = 0
    for layers in range(1, LAYERS_MAX + 1):
        name = f"rank{layers}"
        shape = (layers, QM, 12, LLR_PAD)
        actual = np.fromfile(root / "data" / "ascend_output" / name / "layer_llr.bin",
                             dtype=np.int16)
        expected = np.fromfile(root / "data" / "golden" / name / "layer_llr.bin",
                               dtype=np.int16)
        size_ok = actual.size == int(np.prod(shape))
        exact = size_ok and np.array_equal(actual, expected)
        pad_exercised = False
        if size_ok:
            actual = actual.reshape(shape)
            pad_exercised = bool(np.any(actual[:, :, :, NSC_USED:]))
        ok = size_ok and exact and pad_exercised
        print(f"{name:5s} size={size_ok} exact={exact} physical_pad_exercised="
              f"{pad_exercised} {'PASS' if ok else 'FAIL'}")
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
