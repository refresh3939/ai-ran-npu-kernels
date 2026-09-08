#!/usr/bin/env python3
"""Independent compact-codeword Rank-1..4 vectors for qam_mod_256_mimo."""
from pathlib import Path

import numpy as np

QM = 8
NDATA_RE = 19152
NDATA_PAD = 19200
MAX_LAYERS = 4
QAM_STREAM_TO_NR_BIT = (0, 2, 4, 6, 1, 3, 5, 7)
D16 = np.float16(1.0 / np.sqrt(170.0))


def mod_level(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray,
              c3: np.ndarray) -> np.ndarray:
    """The qam_mod_256_siso 16-PAM equation, evaluated as exact integers."""
    d0 = 2 * c0 - 1
    d1 = 2 * c1 - 1
    d2 = 2 * c2 - 1
    e3 = 3 - 2 * c3
    return d0 * (8 - d1 * (4 - d2 * e3))


def make_case(layers: int, root: Path) -> None:
    valid = layers * NDATA_RE
    stride = layers * NDATA_PAD
    rng = np.random.default_rng(0x256000 + layers)
    bits = np.full((QM, stride), np.int16(0x5A5A), dtype=np.int16)
    bits[:, :valid] = rng.integers(0, 2, size=(QM, valid), dtype=np.int16)

    # Cover all 256 constellation points in every Rank case. q=0..3 are the
    # I nibble c0..c3 and q=4..7 are the Q nibble c0..c3.
    symbols = np.arange(256, dtype=np.uint16)
    for stream in range(QM):
        bits[stream, :256] = ((symbols >> stream) & 1).astype(np.int16)

    i_level = mod_level(*(bits[q, :valid].astype(np.int32) for q in range(4)))
    q_level = mod_level(*(bits[q, :valid].astype(np.int32) for q in range(4, 8)))
    d_re = np.zeros(stride, dtype=np.float16)
    d_im = np.zeros(stride, dtype=np.float16)
    d_re[:valid] = (i_level.astype(np.float16) * D16).astype(np.float16)
    d_im[:valid] = (q_level.astype(np.float16) * D16).astype(np.float16)

    case = root / f"rank{layers}"
    case.mkdir(parents=True, exist_ok=True)
    bits.tofile(case / "bits_qam.bin")
    d_re.tofile(case / "d_re.bin")
    d_im.tofile(case / "d_im.bin")

    unique = np.unique(np.stack((d_re[:256], d_im[:256]), axis=1), axis=0)
    assert unique.shape[0] == 256
    assert not np.any(d_re[valid:]) and not np.any(d_im[valid:])
    print(
        f"rank{layers}: input=[8,{stride}] valid/stream={valid} "
        f"output=[{stride}] constellation_points={unique.shape[0]}"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "golden"
    for layers in range(1, MAX_LAYERS + 1):
        make_case(layers, root)
    print(f"generated Rank 1..4 qam_mod_256_mimo cases in {root}")


if __name__ == "__main__":
    main()
