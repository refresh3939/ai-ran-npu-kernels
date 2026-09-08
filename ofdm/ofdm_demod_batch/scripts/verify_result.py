#!/usr/bin/env python3
"""Validate separated fp16 batch output against the kernel-aligned golden."""
import os
import sys
from pathlib import Path

import numpy as np


N_SYMBOL, P, Q = 14, 32, 64
ABS_TOL = 0.05
BATCH_SIZE = int(os.environ.get("OFDM_BATCH_SIZE", "8"))
ROOT = Path(os.environ.get("AIRAN_DATA_DIR") or Path(__file__).resolve().parent.parent)


def main():
    paths = {
        "golden": ROOT / "data/golden/stage4_dft64.bin",
        "re": ROOT / "data/ascend_output/output_re.bin",
        "im": ROOT / "data/ascend_output/output_im.bin",
    }
    for path in paths.values():
        if not path.exists():
            print(f"[verify] missing: {path}")
            return 1

    count = BATCH_SIZE * N_SYMBOL * P * Q
    golden = np.fromfile(paths["golden"], np.complex64)
    re = np.fromfile(paths["re"], np.float16)
    im = np.fromfile(paths["im"], np.float16)
    if golden.size != count or re.size != count or im.size != count:
        print(f"[verify] size mismatch: golden={golden.size}, re={re.size}, im={im.size}, expected={count}")
        return 1

    golden = golden.reshape(BATCH_SIZE, N_SYMBOL, P, Q)
    actual = (re.astype(np.float32) + 1j * im.astype(np.float32)).reshape(
        BATCH_SIZE, N_SYMBOL, P, Q)
    passed = 0
    overall = 0.0
    print(f"[verify] batch={BATCH_SIZE}, shape=({N_SYMBOL},{P},{Q}), tolerance={ABS_TOL}")
    for batch in range(BATCH_SIZE):
        batch_max = 0.0
        for symbol in range(N_SYMBOL):
            error = float(np.max(np.maximum(
                np.abs(actual[batch, symbol].real - golden[batch, symbol].real),
                np.abs(actual[batch, symbol].imag - golden[batch, symbol].imag))))
            batch_max = max(batch_max, error)
            overall = max(overall, error)
            passed += int(error <= ABS_TOL)
        print(f"  batch {batch:2d}: max |err|={batch_max:7.4f}  "
              f"[{'OK' if batch_max <= ABS_TOL else 'FAIL'}]")

    expected = BATCH_SIZE * N_SYMBOL
    print(f"[verify] overall max |err|={overall:.4f}")
    if passed == expected:
        print(f"[verify] ========== PASS ({passed}/{expected}) ==========")
        return 0
    print(f"[verify] ========== FAIL ({passed}/{expected}) ==========")
    return 1


if __name__ == "__main__":
    sys.exit(main())
