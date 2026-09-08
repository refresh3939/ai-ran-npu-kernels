#!/usr/bin/env python3
"""Independent NR rate-dematching oracle and contract checks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

QM = 8
MAX_LAYERS = 4
MAX_SLOTS = 23
N_DATA_RE = 19152
N_DATA_PAD = 19200
C_NUM = 143
LDPC_N = 26112
N_2Z = 768
N_CB = LDPC_N - N_2Z
SCALE = 8
CLIP = 5120


def descriptors(layers: int, slots: int) -> np.ndarray:
    if not 1 <= layers <= MAX_LAYERS or not 1 <= slots <= MAX_SLOTS:
        raise ValueError("layers must be 1..4 and slots must be 1..23")
    quantum = layers * QM
    low_units, num_high = divmod(slots * N_DATA_RE, C_NUM)
    first_high = C_NUM - num_high
    desc = np.zeros((C_NUM, 4), dtype=np.uint32)
    symbol_offset = 0
    for cb in range(C_NUM):
        units = low_units + (cb >= first_high)
        e = quantum * units
        desc[cb] = (e, 0, N_CB, symbol_offset)
        symbol_offset += e // QM
    assert int(desc[:, 0].sum(dtype=np.uint64)) == slots * QM * layers * N_DATA_RE
    assert symbol_offset == slots * layers * N_DATA_RE
    return desc


def reference(cw_llr: np.ndarray, layers: int, slots: int) -> np.ndarray:
    stride = layers * N_DATA_PAD
    valid = layers * N_DATA_RE
    expected_shape = (slots, QM, stride)
    if cw_llr.shape != expected_shape or cw_llr.dtype != np.int16:
        raise ValueError(f"cw_llr must be int16{expected_shape}")
    logical = cw_llr[:, :, :valid].transpose(1, 0, 2).reshape(QM, -1)
    desc = descriptors(layers, slots)
    output = np.zeros((C_NUM, LDPC_N), dtype=np.int16)
    for cb, (e_u32, k0_u32, ncb_u32, off_u32) in enumerate(desc):
        e, k0, ncb, off = map(int, (e_u32, k0_u32, ncb_u32, off_u32))
        eq = e // QM
        acc = np.zeros(ncb, dtype=np.int32)
        for bit in range(QM):
            values = np.clip(
                logical[bit, off:off + eq].astype(np.int32) * SCALE,
                -CLIP,
                CLIP,
            )
            source = 0
            circular = (k0 + bit * eq) % ncb
            while source < eq:
                count = min(eq - source, ncb - circular)
                acc[circular:circular + count] += values[source:source + count]
                source += count
                circular = 0
        output[cb, N_2Z:] = np.clip(acc, -CLIP, CLIP).astype(np.int16)
    return output


def contract_checks() -> None:
    # Exact SISO legacy distribution and total-G invariant.
    siso = descriptors(1, 23)
    assert np.all(siso[:87, 0] == 24640)
    assert np.all(siso[87:, 0] == 24648)
    assert int(siso[-1, 3] + siso[-1, 0] // QM) == 23 * N_DATA_RE

    # Every supported rank/slot pair covers each valid codeword symbol once
    # per canonical bit plane; padding is excluded by construction.
    for layers in range(1, MAX_LAYERS + 1):
        for slots in range(1, MAX_SLOTS + 1):
            desc = descriptors(layers, slots)
            assert np.all(desc[:, 0] % (QM * layers) == 0)
            assert np.all(desc[:, 1] == 0)
            assert np.all(desc[:, 2] == N_CB)
            ends = desc[:, 3].astype(np.uint64) + desc[:, 0] // QM
            assert np.array_equal(desc[1:, 3], ends[:-1].astype(np.uint32))
            assert int(ends[-1]) == slots * layers * N_DATA_RE

    # Tail poison cannot affect output, including Rank-4 repetition combining.
    rng = np.random.default_rng(0x52444D31)
    layers, slots = 4, 23
    valid, stride = layers * N_DATA_RE, layers * N_DATA_PAD
    base = rng.integers(-900, 901, size=(slots, QM, stride), dtype=np.int16)
    changed = base.copy()
    base[:, :, valid:] = np.int16(0x1111)
    changed[:, :, valid:] = np.int16(0x6A6A)
    assert np.array_equal(reference(base, layers, slots),
                          reference(changed, layers, slots))


def generate(root: Path) -> None:
    rng = np.random.default_rng(0x52444D31)
    for layers in range(1, MAX_LAYERS + 1):
        slots = MAX_SLOTS
        valid, stride = layers * N_DATA_RE, layers * N_DATA_PAD
        cw_llr = rng.integers(
            -900, 901, size=(slots, QM, stride), dtype=np.int16)
        # Poison differs by rank and must never enter G or the oracle output.
        cw_llr[:, :, valid:] = np.int16(0x5A00 + layers)
        ldpc_llr = reference(cw_llr, layers, slots)
        case = root / "golden" / f"rank{layers}"
        case.mkdir(parents=True, exist_ok=True)
        cw_llr.tofile(case / "cw_llr.bin")
        ldpc_llr.tofile(case / "ldpc_llr.bin")
    print(f"generated Rank 1..4, 23-slot cases under {root / 'golden'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    contract_checks()
    print("host contract PASS: Rank 1..4 x slot 1..23, SISO parity, tail exclusion")
    if args.generate:
        if args.root is None:
            parser.error("--generate requires --root")
        generate(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
