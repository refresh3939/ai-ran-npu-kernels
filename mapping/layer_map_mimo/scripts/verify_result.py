#!/usr/bin/env python3
"""Verify layer_map order, zero padding, and layer_demap inverse geometry."""
from pathlib import Path

import numpy as np

from gen_data import NDATA_PAD, NDATA_RE

NDATA_SYMBOLS = 12
NSC_USED = 1596
QAM_LLR_STRIDE = 1600
PHYSICAL_PAD_POISON = np.uint16(0x6B6B)


def verify_plane(root: Path, name: str, layers: int, plane: str) -> tuple[bool, ...]:
    output_path = root / "data" / "ascend_output" / name / f"layer_{plane}.bin"
    golden_path = root / "data" / "golden" / name / f"layer_{plane}.bin"
    input_path = root / "data" / "golden" / name / f"d_{plane}.bin"
    actual = np.fromfile(output_path, dtype=np.uint16)
    expected = np.fromfile(golden_path, dtype=np.uint16)
    source = np.fromfile(input_path, dtype=np.uint16)
    size_ok = actual.size == layers * NDATA_PAD
    if not size_ok:
        return False, False, False, False

    actual = actual.reshape(layers, NDATA_PAD)
    exact = np.array_equal(actual.reshape(-1), expected)
    tail_zero = not np.any(actual[:, NDATA_RE:])

    # layer_map is compact [L,19152 valid + 48 tail], while layer_demap reads
    # [L,Qm,12,1600]. Equal allocation sizes do not make those layouts equal:
    # reinsert four poisoned lanes at the end of each physical OFDM row first.
    physical = np.full(
        (layers, NDATA_SYMBOLS, QAM_LLR_STRIDE),
        PHYSICAL_PAD_POISON,
        dtype=np.uint16,
    )
    physical[:, :, :NSC_USED] = actual[:, :NDATA_RE].reshape(
        layers, NDATA_SYMBOLS, NSC_USED
    )
    physical_pad_preserved = np.all(
        physical[:, :, NSC_USED:] == PHYSICAL_PAD_POISON
    )

    # This is layer_demap's per-q mapping after it skips row padding:
    # cw[L*i+l] = layer[l,ds,sc], i=ds*1596+sc.
    compact = physical[:, :, :NSC_USED].reshape(layers, NDATA_RE)
    recovered = np.zeros(layers * NDATA_PAD, dtype=np.uint16)
    recovered[: layers * NDATA_RE] = compact.T.reshape(-1)
    inverse = np.array_equal(
        recovered[: layers * NDATA_RE], source[: layers * NDATA_RE]
    ) and not np.any(recovered[layers * NDATA_RE :])
    return size_ok, exact, tail_zero, physical_pad_preserved, inverse


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = 0
    for layers in range(1, 5):
        name = f"rank{layers}"
        results = [verify_plane(root, name, layers, plane) for plane in ("re", "im")]
        size_ok = all(result[0] for result in results)
        exact = all(result[1] for result in results)
        tail_zero = all(result[2] for result in results)
        physical_pad = all(result[3] for result in results)
        inverse = all(result[4] for result in results)
        ok = size_ok and exact and tail_zero and physical_pad and inverse
        print(
            f"{name:5s} size={size_ok} exact={exact} tail_zero={tail_zero} "
            f"physical_pad={physical_pad} demap_inverse={inverse} "
            f"{'PASS' if ok else 'FAIL'}"
        )
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
