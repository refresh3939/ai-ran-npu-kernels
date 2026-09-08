#!/usr/bin/env python3
"""Independent Rank-1..4 compact-codeword descramble_mimo vectors."""
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
NR_BIT_TO_QAM_STREAM = (0, 4, 1, 5, 2, 6, 3, 7)


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
    return x1[GOLD_NC:GOLD_NC + length] ^ x2[GOLD_NC:GOLD_NC + length]


def make_case(layers: int, num_slots: int, root: Path) -> None:
    valid = layers * NDATA_RE
    stride = layers * NDATA_PAD
    rng = np.random.default_rng(0xD35C000 + layers)
    llr_qam = rng.integers(
        -2560, 2561, size=(num_slots, QM, stride), dtype=np.int16)
    # Poison the storage tail: it must not be consumed and output must be zero.
    llr_qam[:, :, valid:] = np.int16(0x5A5A)

    c_init = (RNTI << 15) | (CODEWORD_INDEX << 14) | N_ID
    c = gold_sequence(c_init, num_slots * valid * QM)
    c = c.reshape(num_slots, valid, QM)
    sign = np.ones((num_slots, QM, stride), dtype=np.int16)
    sign[:, :, :valid] = (1 - 2 * c.astype(np.int16)).transpose(0, 2, 1)

    llr_nr = np.zeros_like(llr_qam)
    compact = llr_qam[:, NR_BIT_TO_QAM_STREAM, :valid].astype(np.int32)
    llr_nr[:, :, :valid] = np.clip(
        compact * sign[:, :, :valid].astype(np.int32), -32768, 32767
    ).astype(np.int16)

    case = root / f"rank{layers}"
    case.mkdir(parents=True, exist_ok=True)
    llr_qam.tofile(case / "llr_qam.bin")
    sign.tofile(case / "sign.bin")
    llr_nr.tofile(case / "llr_nr.bin")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "golden"
    for layers in range(1, MAX_LAYERS + 1):
        make_case(layers, SLOTS_BY_RANK[layers - 1], root)
    print(f"generated Rank 1..4 descramble_mimo cases in {root}")


if __name__ == "__main__":
    main()
