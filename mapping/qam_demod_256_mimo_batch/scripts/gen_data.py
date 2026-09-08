#!/usr/bin/env python3
"""Generate independent Rank-1..4 physical-layout QAM-demapper vectors."""
from pathlib import Path

import numpy as np

LAYERS_MAX = 4
QM = 8
NSYM = 14
DATA_PHYS = (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13)
NSC_USED = 1596
GRID_PAD = 1664
LLR_PAD = 1600
D = 1.0 / np.sqrt(170.0)
Q_SCALE = 32
LLR_CLIP = 2560
Y_CLIP = 1000


def axis_llr(y: float, noise: float) -> np.ndarray:
    yh = np.float16(y)
    nh = np.float16(noise)
    scale = np.float16(D * Q_SCALE) / nh
    threshold = np.float16(D * D * Q_SCALE) / nh
    u = int(np.rint(np.float16(yh * scale)))
    t = int(np.rint(threshold))
    u = max(-Y_CLIP, min(Y_CLIP, u))
    a = abs(u)
    clipped = {k: min(a, k * t) for k in (2, 4, 6, 8, 10, 12, 14)}
    sign_sum = sum(max(-k * t, min(k * t, u)) for k in clipped)
    values = (
        32 * u - 4 * sign_sum,
        80 * t - 16 * a - 4 * (clipped[2] + clipped[4] + clipped[6])
        + 4 * (clipped[10] + clipped[12] + clipped[14]),
        -24 * t - 8 * a + 4 * clipped[2] - 4 * clipped[6]
        + 16 * clipped[8] - 4 * clipped[10] + 4 * clipped[14],
        -8 * t - 4 * a + 8 * clipped[4] - 8 * clipped[8] + 8 * clipped[12],
    )
    # Existing kernel's standard polarity is positive => bit 0.
    return -np.clip(np.asarray(values, dtype=np.int32), -LLR_CLIP, LLR_CLIP).astype(np.int16)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "golden"
    rng = np.random.default_rng(0x256BA7C)
    x_re = rng.normal(0.0, 0.7, (LAYERS_MAX, NSYM, GRID_PAD)).astype(np.float16)
    x_im = rng.normal(0.0, 0.7, (LAYERS_MAX, NSYM, GRID_PAD)).astype(np.float16)
    noises = np.asarray((0.0125, 0.02, 0.05, 0.2), dtype=np.float16)
    no_eff = np.empty_like(x_re)
    for layer in range(LAYERS_MAX):
        no_eff[layer] = noises[layer]

    expected = np.empty((LAYERS_MAX, QM, len(DATA_PHYS), LLR_PAD), dtype=np.int16)
    for layer in range(LAYERS_MAX):
        for ds, physical_symbol in enumerate(DATA_PHYS):
            for sc in range(LLR_PAD):
                expected[layer, :4, ds, sc] = axis_llr(
                    float(x_re[layer, physical_symbol, sc]),
                    float(no_eff[layer, physical_symbol, sc]),
                )
                expected[layer, 4:, ds, sc] = axis_llr(
                    float(x_im[layer, physical_symbol, sc]),
                    float(no_eff[layer, physical_symbol, sc]),
                )

    for layers in range(1, LAYERS_MAX + 1):
        case = root / f"rank{layers}"
        case.mkdir(parents=True, exist_ok=True)
        x_re[:layers].tofile(case / "x_re.bin")
        x_im[:layers].tofile(case / "x_im.bin")
        no_eff[:layers].tofile(case / "no_eff.bin")
        expected[:layers].tofile(case / "layer_llr.bin")
    print(f"generated rank1..rank4 qam_demod_256_batch cases in {root}")


if __name__ == "__main__":
    main()
