#!/usr/bin/env python3
"""Independent bit-exact golden generator for PUSCH layer_map."""
from pathlib import Path

import numpy as np

NDATA_RE = 19152
NDATA_PAD = 19200


def make_plane(layers: int, seed: int, poison: int) -> np.ndarray:
    index = np.arange(layers * NDATA_PAD, dtype=np.uint32)
    # Treat fp16 as opaque words so copies also exercise NaNs and signed zero.
    words = ((index * np.uint32(4051) + np.uint32(seed * 7919)) & 0xFFFF).astype(
        np.uint16
    )
    words[layers * NDATA_RE :] = np.uint16(poison)
    return words


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "golden"
    for layers in range(1, 5):
        d_re = make_plane(layers, 2 * layers, 0x7E01)
        d_im = make_plane(layers, 2 * layers + 1, 0xFE01)
        layer_re = np.zeros((layers, NDATA_PAD), dtype=np.uint16)
        layer_im = np.zeros_like(layer_re)
        layer_re[:, :NDATA_RE] = d_re[: layers * NDATA_RE].reshape(
            NDATA_RE, layers
        ).T
        layer_im[:, :NDATA_RE] = d_im[: layers * NDATA_RE].reshape(
            NDATA_RE, layers
        ).T

        case = root / f"rank{layers}"
        case.mkdir(parents=True, exist_ok=True)
        d_re.tofile(case / "d_re.bin")
        d_im.tofile(case / "d_im.bin")
        layer_re.tofile(case / "layer_re.bin")
        layer_im.tofile(case / "layer_im.bin")
    print(f"generated rank1..rank4 layer_map cases in {root}")


if __name__ == "__main__":
    main()
