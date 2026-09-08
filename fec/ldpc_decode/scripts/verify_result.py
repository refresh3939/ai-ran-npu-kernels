#!/usr/bin/env python3
"""Independent LDPC decoder result verification.

Reads the selected dataset from AIRAN_DATA_DIR and the NPU dump from
AIRAN_OUTPUT_DIR. This script only verifies; it does not generate data, build,
or launch the kernel.
"""

import os
import sys
from pathlib import Path

import numpy as np

N_CB = 143
K = 8448

KERNEL_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("AIRAN_DATA_DIR", KERNEL_DIR / "data/snr_5db"))
OUTPUT_DIR = Path(os.environ.get("AIRAN_OUTPUT_DIR", KERNEL_DIR / "data/ascend_output"))
TRUTH = DATA_DIR / "info_bits.bin"
ACTUAL = OUTPUT_DIR / "decoded_bits.bin"


def load_exact(path: Path, dtype, count: int, label: str) -> np.ndarray:
    if not path.exists():
        print(f"[FAIL] {label} does not exist: {path}")
        sys.exit(1)
    values = np.fromfile(path, dtype=dtype)
    if values.size != count:
        print(f"[FAIL] {label} has {values.size} elements; expected {count}: {path}")
        sys.exit(1)
    return values


def main() -> None:
    print("=" * 70)
    print("[verify_result] ldpc_decode  decoded_bits vs info_bits")
    print(f"  truth  = {TRUTH}")
    print(f"  actual = {ACTUAL}")
    print("=" * 70)

    truth = load_exact(TRUTH, np.uint8, N_CB * K, "truth").reshape(N_CB, K)
    actual = load_exact(ACTUAL, np.int8, N_CB * K, "actual").reshape(N_CB, K)

    diff = actual.astype(np.uint8) != truth
    bit_errors = int(diff.sum())
    block_errors = int(diff.any(axis=1).sum())
    ber = bit_errors / diff.size
    bler = block_errors / N_CB

    print(f"  BER  = {ber:.8%} ({bit_errors}/{diff.size})")
    print(f"  BLER = {bler:.8%} ({block_errors}/{N_CB})")

    # Match the host launcher's decoder tolerance.
    if ber <= 5e-4:
        print("[PASS] BER is within 0.05%")
        return

    failing = np.flatnonzero(diff.any(axis=1))[:8]
    if failing.size:
        print("  first failing CBs:")
        for cb in failing:
            print(f"    CB {cb}: {int(diff[cb].sum())}/{K} bit errors")
    print("[FAIL] BER exceeds 0.05%")
    sys.exit(1)


if __name__ == "__main__":
    main()
