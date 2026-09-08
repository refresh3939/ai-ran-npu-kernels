#!/usr/bin/env python3
"""Independent TS 38.212 rate-matching vectors for Rank 1..4."""
import os
from pathlib import Path

import numpy as np

QM = 8
MAX_LAYERS = 4
MAX_SLOTS = 23
N_DATA_RE = 19152
N_DATA_PAD = 19200
C_NUM = 143
N_CB_BUF = 25344


def descriptors(layers: int, num_slots: int) -> np.ndarray:
    # TS 38.212 5.4.2.1, one codeword: G'=G/(N_L*Qm).
    g_prime = num_slots * N_DATA_RE
    floor_symbols, gamma = divmod(g_prime, C_NUM)
    symbols = np.full(C_NUM, floor_symbols, dtype=np.uint32)
    if gamma:
        symbols[C_NUM - gamma :] += 1
    desc = np.zeros((C_NUM, 4), dtype=np.uint32)
    desc[:, 0] = layers * QM * symbols
    desc[:, 1] = 0
    desc[:, 2] = N_CB_BUF
    desc[:, 3] = np.arange(C_NUM, dtype=np.uint32) * N_CB_BUF
    assert int(desc[:, 0].sum()) == num_slots * QM * layers * N_DATA_RE
    return desc


def rate_match(code_blocks: np.ndarray, layers: int, num_slots: int,
               desc: np.ndarray) -> np.ndarray:
    valid = layers * N_DATA_RE
    stride = layers * N_DATA_PAD
    output = np.zeros((num_slots, QM, stride), dtype=np.int16)
    plane_length = num_slots * valid
    for bit in range(QM):
        serial = np.empty(plane_length, dtype=np.int16)
        cursor = 0
        for cb in range(C_NUM):
            e, k0, ncb, _ = (int(value) for value in desc[cb])
            eq = e // QM
            selected = (k0 + bit * eq + np.arange(eq, dtype=np.uint32)) % ncb
            serial[cursor : cursor + eq] = code_blocks[cb, selected]
            cursor += eq
        assert cursor == plane_length
        output[:, bit, :valid] = serial.reshape(num_slots, valid)
    return output


def make_case(layers: int, num_slots: int, root: Path) -> None:
    rng = np.random.default_rng(0x524D000 + layers)
    code_blocks = rng.integers(
        0, 2, size=(C_NUM, N_CB_BUF), dtype=np.int8
    )
    desc = descriptors(layers, num_slots)
    bits_nr = rate_match(code_blocks, layers, num_slots, desc)
    case = root / f"rank{layers}"
    case.mkdir(parents=True, exist_ok=True)
    code_blocks.tofile(case / "code_blocks.bin")
    desc.tofile(case / "rm_desc.bin")
    bits_nr.tofile(case / "bits_nr.bin")
    repeated = bool(np.any(desc[:, 0] > N_CB_BUF))
    print(
        f"generated rank{layers} slots={num_slots} "
        f"shape={bits_nr.shape} E=[{desc[0, 0]},{desc[-1, 0]}] "
        f"repetition={repeated}"
    )


def main() -> None:
    data_root = Path(os.environ.get("AIRAN_DATA_DIR", Path(__file__).parents[1] / "data"))
    root = data_root / "golden"
    for layers in range(1, MAX_LAYERS + 1):
        make_case(layers, MAX_SLOTS, root)


if __name__ == "__main__":
    main()
