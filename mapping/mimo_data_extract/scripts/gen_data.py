#!/usr/bin/env python3
"""Independent golden generator for mimo_data_extract."""
from pathlib import Path

import numpy as np

NSYM = 14
NSC_USED = 1596
NSC_PAD = 1664
NGRID = NSYM * NSC_PAD
NDATA_RE = 12 * NSC_USED
NDATA_PAD = 19200

CASES = [
    ("rank1_default", 1, (1 << 2) | (1 << 11)),
    ("rank2_default", 2, (1 << 2) | (1 << 11)),
    ("rank3_edge_dmrs", 3, (1 << 0) | (1 << 13)),
    ("rank4_alt_dmrs", 4, (1 << 1) | (1 << 12)),
]


def values(count: int, seed: int) -> np.ndarray:
    index = np.arange(count, dtype=np.int64)
    magnitude = ((index * 37 + seed * 101) % 1900 + 1).astype(np.float32) / 128.0
    sign = np.where(((index + seed) & 1) == 0, 1.0, -1.0).astype(np.float32)
    return (magnitude * sign).astype(np.float16)


def generate_case(root: Path, name: str, layers: int, dmrs_mask: int, seed: int) -> None:
    data_symbols = [symbol for symbol in range(NSYM) if not ((dmrs_mask >> symbol) & 1)]
    dmrs_symbols = [symbol for symbol in range(NSYM) if (dmrs_mask >> symbol) & 1]
    assert len(data_symbols) == 12 and len(dmrs_symbols) == 2

    xhat_re = values(layers * NGRID, seed).reshape(layers, NSYM, NSC_PAD)
    xhat_im = values(layers * NGRID, seed + 17).reshape(layers, NSYM, NSC_PAD)
    no_eff = (np.abs(values(layers * NGRID, seed + 31)) + np.float16(0.03125)).astype(
        np.float16
    ).reshape(layers, NSYM, NSC_PAD)

    # Poison every source location that the operator must ignore.
    xhat_re[:, :, NSC_USED:] = np.float16(101.0)
    xhat_im[:, :, NSC_USED:] = np.float16(-102.0)
    no_eff[:, :, NSC_USED:] = np.float16(103.0)
    xhat_re[:, dmrs_symbols, :NSC_USED] = np.float16(111.0)
    xhat_im[:, dmrs_symbols, :NSC_USED] = np.float16(-112.0)
    no_eff[:, dmrs_symbols, :NSC_USED] = np.float16(113.0)

    data_re = np.zeros((layers, NDATA_PAD), dtype=np.float16)
    data_im = np.zeros_like(data_re)
    data_no_eff = np.zeros_like(data_re)
    for ordinal, symbol in enumerate(data_symbols):
        start = ordinal * NSC_USED
        stop = start + NSC_USED
        data_re[:, start:stop] = xhat_re[:, symbol, :NSC_USED]
        data_im[:, start:stop] = xhat_im[:, symbol, :NSC_USED]
        data_no_eff[:, start:stop] = no_eff[:, symbol, :NSC_USED]

    case = root / name
    case.mkdir(parents=True, exist_ok=True)
    arrays = {
        "xhat_re.bin": xhat_re,
        "xhat_im.bin": xhat_im,
        "no_eff.bin": no_eff,
        "data_re.bin": data_re,
        "data_im.bin": data_im,
        "data_no_eff.bin": data_no_eff,
    }
    for filename, array in arrays.items():
        array.tofile(case / filename)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "golden"
    for seed, case in enumerate(CASES, start=1):
        generate_case(root, *case, seed=seed)
    print(f"generated {len(CASES)} mimo_data_extract cases in {root}")


if __name__ == "__main__":
    main()
