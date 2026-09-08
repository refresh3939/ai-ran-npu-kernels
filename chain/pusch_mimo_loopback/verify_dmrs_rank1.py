#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def gold(cinit: int, length: int) -> np.ndarray:
    total = length + 1600
    x1 = np.zeros(total + 31, np.uint8)
    x2 = np.zeros(total + 31, np.uint8)
    x1[0] = 1
    for bit in range(31):
        x2[bit] = (cinit >> bit) & 1
    for n in range(total):
        x1[n + 31] = x1[n + 3] ^ x1[n]
        x2[n + 31] = x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n]
    return x1[1600:1600 + length] ^ x2[1600:1600 + length]


def expected(slot: int, symbol: int) -> tuple[np.ndarray, np.ndarray]:
    nid = 321
    cinit = ((1 << 17) * (14 * slot + symbol + 1) * (2 * nid + 1) +
             2 * nid) % (1 << 31)
    seq = gold(cinit, 2 * 798).reshape(798, 2)
    scale = np.float16(1 / np.sqrt(2.0))
    re = np.zeros(896, np.float16)
    im = np.zeros(896, np.float16)
    re[:798] = ((1 - 2 * seq[:, 0].astype(np.int16)).astype(np.float16) * scale).astype(np.float16)
    im[:798] = ((1 - 2 * seq[:, 1].astype(np.int16)).astype(np.float16) * scale).astype(np.float16)
    return re, im


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--result", required=True, type=Path)
    a = p.parse_args()
    digest = hashlib.sha256()
    for slot in range(23):
        got_re = np.fromfile(a.root / f"slot{slot:02d}_re.bin", np.float16)
        got_im = np.fromfile(a.root / f"slot{slot:02d}_im.bin", np.float16)
        if got_re.size != 1792 or got_im.size != 1792:
            raise RuntimeError(f"slot {slot}: DMRS shape mismatch")
        for ds, symbol in enumerate((2, 11)):
            exp_re, exp_im = expected(slot, symbol)
            if not np.array_equal(got_re.reshape(2, 896)[ds].view(np.uint16), exp_re.view(np.uint16)):
                raise RuntimeError(f"slot {slot} symbol {symbol}: DMRS real mismatch")
            if not np.array_equal(got_im.reshape(2, 896)[ds].view(np.uint16), exp_im.view(np.uint16)):
                raise RuntimeError(f"slot {slot} symbol {symbol}: DMRS imag mismatch")
        digest.update(got_re.tobytes()); digest.update(got_im.tobytes())
    result = {"schema_version": 1, "status": "PASS", "rank": 1, "slots": 23,
              "operator": "mimo_dmrs_gen", "cell_id": 321,
              "ports": [1000], "logical_shape_per_slot": [1, 2, 896],
              "bit_exact_gold_qpsk": True, "sha256": digest.hexdigest()}
    a.result.parent.mkdir(parents=True, exist_ok=True)
    a.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
