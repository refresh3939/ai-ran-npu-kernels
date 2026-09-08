#!/usr/bin/env python3
"""Independent golden generator for PUSCH layer_demap."""
from pathlib import Path

import numpy as np

QM = 8
NDATA_RE = 19152
NDATA_PAD = 19200
NDATA_SYM = 12
NSC_USED = 1596
NSC_PAD = 1600


def make_input(layers: int) -> np.ndarray:
    layer = np.arange(layers, dtype=np.int32)[:, None, None]
    stream = np.arange(QM, dtype=np.int32)[None, :, None]
    symbol = np.arange(NDATA_PAD, dtype=np.int32)[None, None, :]
    # Exercise signs and int16 edges while retaining a unique deterministic mix.
    values = ((symbol * 251 + stream * 4051 + layer * 7919 + 16384) % 65536) - 32768
    result = values.astype(np.int16)
    result = result.reshape(layers, QM, NDATA_SYM, NSC_PAD)
    result[:, :, :, NSC_USED:] = np.int16(0x5A5A)  # poison every QAM row padding
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "golden"
    for layers in range(1, 5):
        layer_llr = make_input(layers)
        cw_llr = np.zeros((QM, layers * NDATA_PAD), dtype=np.int16)
        # [L,Qm,N] -> [Qm,N,L] -> flatten, preserving the physical q stream.
        valid = layer_llr[:, :, :, :NSC_USED].transpose(1, 2, 3, 0)
        cw_llr[:, : layers * NDATA_RE] = valid.reshape(QM, -1)
        case = root / f"rank{layers}"
        case.mkdir(parents=True, exist_ok=True)
        layer_llr.tofile(case / "layer_llr.bin")
        cw_llr.tofile(case / "cw_llr.bin")
    print(f"generated rank1..rank4 layer_demap cases in {root}")


if __name__ == "__main__":
    main()
