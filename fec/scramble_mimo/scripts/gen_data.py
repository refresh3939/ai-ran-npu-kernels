#!/usr/bin/env python3
"""Independent Rank-1..4 slot-batch vectors for scramble_mimo."""
from pathlib import Path

import numpy as np

QM = 8
NDATA_RE = 19152
NDATA_PAD = 19200
MAX_LAYERS = 4
SLOTS_BY_RANK = (23, 7, 3, 2)
N_ID = 321
RNTI = 12345
CODEWORD_INDEX = 0
GOLD_NC = 1600
QAM_STREAM_TO_NR_BIT = (0, 2, 4, 6, 1, 3, 5, 7)


def gold_sequence(c_init: int, length: int) -> np.ndarray:
    total = length + GOLD_NC
    x1 = np.zeros(total + 31, dtype=np.uint8)
    x2 = np.zeros(total + 31, dtype=np.uint8)
    x1[0] = 1
    for bit in range(31):
        x2[bit] = (c_init >> bit) & 1
    for n in range(total):
        x1[n + 31] = x1[n + 3] ^ x1[n]
        x2[n + 31] = x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n]
    return x1[GOLD_NC : GOLD_NC + length] ^ x2[GOLD_NC : GOLD_NC + length]


def make_case(layers: int, num_slots: int, root: Path) -> None:
    valid = layers * NDATA_RE
    stride = layers * NDATA_PAD
    rng = np.random.default_rng(0x5C000 + layers)
    bits_nr = np.full((num_slots, QM, stride), np.int16(0x5A5A), dtype=np.int16)
    bits_nr[:, :, :valid] = rng.integers(
        0, 2, size=(num_slots, QM, valid), dtype=np.int16
    )

    c_init = (RNTI << 15) | (CODEWORD_INDEX << 14) | N_ID
    serial = gold_sequence(c_init, num_slots * valid * QM)
    serial = serial.reshape(num_slots, valid, QM)
    gold = np.zeros_like(bits_nr)
    gold[:, :, :valid] = serial.transpose(0, 2, 1).astype(np.int16)

    bits_qam = np.zeros_like(bits_nr)
    for qam_stream, nr_bit in enumerate(QAM_STREAM_TO_NR_BIT):
        bits_qam[:, qam_stream, :valid] = np.bitwise_xor(
            bits_nr[:, nr_bit, :valid], gold[:, nr_bit, :valid]
        )

    case = root / f"rank{layers}"
    case.mkdir(parents=True, exist_ok=True)
    bits_nr.tofile(case / "bits_nr.bin")
    gold.tofile(case / "gold.bin")
    bits_qam.tofile(case / "bits_qam.bin")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "golden"
    for layers in range(1, MAX_LAYERS + 1):
        make_case(layers, SLOTS_BY_RANK[layers - 1], root)
    print(f"generated Rank 1..4 scramble_mimo cases in {root}")


if __name__ == "__main__":
    main()
