#!/usr/bin/env python3
"""Generate Rate Match golden data and simulate the optimized v5c schedule.

The kernel assigns two complete output streams to every AIV, assembles full
19152-RE slots, and writes each stream pair without output RMW traffic.
"""
import os
from pathlib import Path

import numpy as np

from rate_match_ref import (
    C_NUM,
    G,
    N_CB_BUF,
    N_LAYERS,
    N_SLOT,
    N_STREAMS,
    N_SYM,
    N_SYM_PAD,
    Q_M,
    compute_Er,
    rate_match_forward,
)

BLOCK_DIM = 4
STREAMS_PER_AIV = N_STREAMS // BLOCK_DIM


def simulate_and_verify(seed=4):
    """Mirror stream ownership, sequential CB assembly, and slot flushes."""
    e_all = compute_Er(G, C_NUM, Q_M, N_LAYERS)
    eq_all = e_all // Q_M
    rng = np.random.default_rng(seed)
    cw = rng.integers(0, 2, size=(C_NUM, N_CB_BUF), dtype=np.int8)
    layout_ref = rate_match_forward(cw)
    layout = np.zeros((N_SLOT, N_STREAMS, N_SYM_PAD), dtype=np.int16)

    crossing = 0
    for aiv in range(BLOCK_DIM):
        first_stream = aiv * STREAMS_PER_AIV
        slot_pair = np.empty((STREAMS_PER_AIV, N_SYM), dtype=np.int16)
        slot = 0
        c0 = 0
        for cb, eq_value in enumerate(eq_all):
            eq = int(eq_value)
            re0 = c0 - slot * N_SYM
            count0 = min(eq, N_SYM - re0)
            count1 = eq - count0
            pair = cw[
                cb,
                first_stream * eq:(first_stream + STREAMS_PER_AIV) * eq,
            ].astype(np.int16).reshape(STREAMS_PER_AIV, eq)
            slot_pair[:, re0:re0 + count0] = pair[:, :count0]
            if re0 + count0 == N_SYM:
                layout[slot, first_stream:first_stream + STREAMS_PER_AIV, :N_SYM] = slot_pair
                slot += 1
                if count1:
                    slot_pair[:, :count1] = pair[:, count0:]
                    if aiv == 0:
                        crossing += 1
            c0 += eq
        assert slot == N_SLOT
        assert c0 == N_SLOT * N_SYM

    bad = int(np.count_nonzero(layout != layout_ref))
    pad_ok = bool(np.all(layout[:, :, N_SYM:] == 0))
    print("--- optimized stream-owned slot-batch simulation ---")
    print(f"  mismatching int16 elements : {bad}  (0 = bit-exact)")
    print(f"  CBs crossing a slot boundary: {crossing}/{C_NUM}")
    print(f"  padding [{N_SYM}:{N_SYM_PAD}] all zero: {pad_ok}")
    print(f"  result : {'PASS' if bad == 0 and pad_ok else 'FAIL'}")
    return bad == 0 and pad_ok


def dump_and_verify(out_dir=None, seed=4):
    if out_dir is None:
        root = Path(os.environ.get("AIRAN_DATA_DIR", "../../.."))
        out_dir = root / "data/golden/tx/rate_match"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    cw = rng.integers(0, 2, size=(C_NUM, N_CB_BUF), dtype=np.int8)
    layout_ref = rate_match_forward(cw)
    assert cw.dtype == np.int8 and cw.shape == (C_NUM, N_CB_BUF)
    assert layout_ref.dtype == np.int16
    assert layout_ref.shape == (N_SLOT, N_STREAMS, N_SYM_PAD)

    cw_path = out_dir / "codeword_in.bin"
    layout_path = out_dir / "layout_ref.bin"
    cw.tofile(cw_path)
    layout_ref.tofile(layout_path)
    assert cw_path.stat().st_size == C_NUM * N_CB_BUF
    assert layout_path.stat().st_size == N_SLOT * N_STREAMS * N_SYM_PAD * 2

    print(f"--- dumped to {out_dir} ---")
    print(f"  codeword_in.bin : int8  [{C_NUM},{N_CB_BUF}] ({cw_path.stat().st_size} bytes)")
    print(
        f"  layout_ref.bin  : int16 [{N_SLOT},{N_STREAMS},{N_SYM_PAD}] "
        f"({layout_path.stat().st_size} bytes)"
    )
    return True


if __name__ == "__main__":
    if simulate_and_verify():
        dump_and_verify()
