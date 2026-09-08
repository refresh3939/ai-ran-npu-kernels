#!/usr/bin/env python3
"""Verify batch output, fused stream permutation, Gold, and zero tails."""
from pathlib import Path

import numpy as np

from gen_data import (
    MAX_LAYERS,
    NDATA_PAD,
    NDATA_RE,
    QAM_STREAM_TO_NR_BIT,
    QM,
    SLOTS_BY_RANK,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = 0
    for layers in range(1, MAX_LAYERS + 1):
        name = f"rank{layers}"
        slots = SLOTS_BY_RANK[layers - 1]
        stride = layers * NDATA_PAD
        valid = layers * NDATA_RE
        shape = (slots, QM, stride)
        golden = root / "data" / "golden" / name
        actual = np.fromfile(
            root / "data" / "ascend_output" / name / "bits_qam.bin",
            dtype=np.int16,
        )
        expected = np.fromfile(golden / "bits_qam.bin", dtype=np.int16)
        bits_nr = np.fromfile(golden / "bits_nr.bin", dtype=np.int16)
        gold = np.fromfile(golden / "gold.bin", dtype=np.int16)
        size_ok = all(x.size == int(np.prod(shape)) for x in (actual, expected, bits_nr, gold))
        exact = size_ok and np.array_equal(actual, expected)
        tail_zero = domain = inverse = False
        if size_ok:
            actual = actual.reshape(shape)
            bits_nr = bits_nr.reshape(shape)
            gold = gold.reshape(shape)
            tail_zero = not np.any(actual[:, :, valid:])
            domain = set(np.unique(actual[:, :, :valid]).tolist()) <= {0, 1}
            recovered = np.empty((slots, QM, valid), dtype=np.int16)
            for qam_stream, nr_bit in enumerate(QAM_STREAM_TO_NR_BIT):
                recovered[:, nr_bit] = np.bitwise_xor(
                    actual[:, qam_stream, :valid], gold[:, nr_bit, :valid]
                )
            inverse = np.array_equal(recovered, bits_nr[:, :, :valid])
        ok = size_ok and exact and tail_zero and domain and inverse
        print(
            f"{name:5s} slots={slots:2d} size={size_ok} exact={exact} "
            f"tail_zero={tail_zero} domain={domain} inverse={inverse} "
            f"{'PASS' if ok else 'FAIL'}"
        )
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
