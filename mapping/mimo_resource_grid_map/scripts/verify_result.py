#!/usr/bin/env python3
"""Semantic verification of NPU outputs written by the C++ runner."""
from pathlib import Path

import numpy as np

from gen_data import CASES, NSYM, NSC_USED, NSC_PAD


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = 0
    for name, ports, dmrs_mask in CASES:
        layers = len(ports)
        shape = (layers, NSYM, NSC_PAD)
        actual_re = np.fromfile(root / "data" / "ascend_output" / f"{name}_re.bin", dtype=np.float16).reshape(shape)
        actual_im = np.fromfile(root / "data" / "ascend_output" / f"{name}_im.bin", dtype=np.float16).reshape(shape)
        expected_re = np.fromfile(root / "data" / "golden" / name / "grid_re.bin", dtype=np.float16).reshape(shape)
        expected_im = np.fromfile(root / "data" / "golden" / name / "grid_im.bin", dtype=np.float16).reshape(shape)
        exact = np.array_equal(actual_re.view(np.uint16), expected_re.view(np.uint16)) and np.array_equal(
            actual_im.view(np.uint16), expected_im.view(np.uint16)
        )
        tail_zero = not np.any(actual_re[:, :, NSC_USED:]) and not np.any(actual_im[:, :, NSC_USED:])
        dmrs_symbols = [symbol for symbol in range(NSYM) if (dmrs_mask >> symbol) & 1]
        dmrs_clean = True
        for layer, port in enumerate(ports):
            delta = (port - 1000) // 2
            unused = np.arange(1 - delta, NSC_USED, 2)
            for symbol in dmrs_symbols:
                dmrs_clean &= not np.any(actual_re[layer, symbol, unused])
                dmrs_clean &= not np.any(actual_im[layer, symbol, unused])
        ok = exact and tail_zero and dmrs_clean
        print(f"{name:26s} exact={exact} tail_zero={tail_zero} dmrs_clean={dmrs_clean} {'PASS' if ok else 'FAIL'}")
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
