#!/usr/bin/env python3
"""Independent golden generator for mimo_resource_grid_map."""
from pathlib import Path

import numpy as np

NSYM = 14
NSC_USED = 1596
NSC_PAD = 1664
NGRID = NSYM * NSC_PAD
DMRS_RE = 798
DMRS_PAD = 896
DATA_PAD = 19200
INVALID = np.uint32(0xFFFFFFFF)

CASES = [
    ("case_0_rank1_port1000", [1000], (1 << 2) | (1 << 11)),
    ("case_1_rank2_shared", [1000, 1001], (1 << 2) | (1 << 11)),
    ("case_2_rank2_disjoint", [1000, 1002], (1 << 2) | (1 << 11)),
    ("case_3_rank3_reordered", [1003, 1000, 1002], (1 << 2) | (1 << 11)),
    ("case_4_rank4_alt_mask", [1003, 1002, 1001, 1000], (1 << 1) | (1 << 12)),
]


def values(count: int, seed: int) -> np.ndarray:
    index = np.arange(count, dtype=np.int64)
    magnitude = ((index * 37 + seed * 101) % 1000 + 1).astype(np.float32) / 128.0
    sign = np.where(((index + seed) & 1) == 0, 1.0, -1.0).astype(np.float32)
    return (magnitude * sign).astype(np.float16)


def generate_case(root: Path, name: str, ports: list[int], dmrs_mask: int, seed: int) -> None:
    layers = len(ports)
    dmrs_symbols = [symbol for symbol in range(NSYM) if (dmrs_mask >> symbol) & 1]
    data_symbols = [symbol for symbol in range(NSYM) if not ((dmrs_mask >> symbol) & 1)]
    assert len(dmrs_symbols) == 2 and len(data_symbols) == 12

    layer_re = values(layers * DATA_PAD, seed).reshape(layers, DATA_PAD)
    layer_im = values(layers * DATA_PAD, seed + 19).reshape(layers, DATA_PAD)
    dmrs_re = values(layers * 2 * DMRS_PAD, seed + 41).reshape(layers, 2, DMRS_PAD)
    dmrs_im = values(layers * 2 * DMRS_PAD, seed + 67).reshape(layers, 2, DMRS_PAD)
    # Poison producer padding. A correct mapper must never expose these values.
    layer_re[:, 12 * NSC_USED :] = np.float16(123.0)
    layer_im[:, 12 * NSC_USED :] = np.float16(-123.0)
    dmrs_re[:, :, DMRS_RE:] = np.float16(111.0)
    dmrs_im[:, :, DMRS_RE:] = np.float16(-111.0)

    grid_re = np.zeros((layers, NSYM, NSC_PAD), dtype=np.float16)
    grid_im = np.zeros_like(grid_re)
    data_offset = np.full(DATA_PAD, INVALID, dtype=np.uint32)
    dmrs_offset = np.full((4, 2, DMRS_PAD), INVALID, dtype=np.uint32)

    compact = 0
    for symbol in data_symbols:
        grid_re[:, symbol, :NSC_USED] = layer_re[:, compact : compact + NSC_USED]
        grid_im[:, symbol, :NSC_USED] = layer_im[:, compact : compact + NSC_USED]
        for subcarrier in range(NSC_USED):
            data_offset[compact + subcarrier] = np.uint32(2 * (symbol * NSC_PAD + subcarrier))
        compact += NSC_USED

    for layer, port in enumerate(ports):
        delta = (port - 1000) // 2
        subcarriers = 2 * np.arange(DMRS_RE) + delta
        for dmrs, symbol in enumerate(dmrs_symbols):
            grid_re[layer, symbol, subcarriers] = dmrs_re[layer, dmrs, :DMRS_RE]
            grid_im[layer, symbol, subcarriers] = dmrs_im[layer, dmrs, :DMRS_RE]
            dmrs_offset[layer, dmrs, :DMRS_RE] = (
                2 * (symbol * NSC_PAD + subcarriers)
            ).astype(np.uint32)

    case_dir = root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "layer_re.bin": layer_re,
        "layer_im.bin": layer_im,
        "dmrs_re.bin": dmrs_re,
        "dmrs_im.bin": dmrs_im,
        "grid_re.bin": grid_re,
        "grid_im.bin": grid_im,
        "data_dst_offset.bin": data_offset,
        "dmrs_dst_offset.bin": dmrs_offset,
    }
    for filename, array in arrays.items():
        array.tofile(case_dir / filename)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "golden"
    for seed, case in enumerate(CASES, start=1):
        generate_case(root, *case, seed=seed)
    print(f"generated {len(CASES)} mimo_resource_grid_map cases in {root}")


if __name__ == "__main__":
    main()
