#!/usr/bin/env python3
"""Build and validate artifacts for the v5 rate-dematch kernel.

The simulator mirrors the production data path:

1. aligned GM->UB block loads into a fixed [8, LPM] source layout;
2. eight sequential vector copies from [8,LPM] into packed e[8,EQ];
3. minimal high-class padding clear, int16 scale/clip, and aligned output write.

The two source slots only change scheduling, so the NumPy model validates both
slots by alternating them while keeping the mathematical result unchanged.
"""

from pathlib import Path
import os

import numpy as np

try:
    from .rate_dematch_ref import (
        Q_M, C_NUM, LDPC_N, N_2Z, N_SYM, N_SYM_PAD, N_STREAMS, N_SLOT,
        N_LAYERS, G, QAM_Q_SCALE, LDPC_Q_SCALE, LDPC_CLIP, compute_Er,
        rate_dematch_inverse,
    )
except ImportError:  # Direct execution: python3 scripts/build_kernel_artifacts.py
    from rate_dematch_ref import (
        Q_M, C_NUM, LDPC_N, N_2Z, N_SYM, N_SYM_PAD, N_STREAMS, N_SLOT,
        N_LAYERS, G, QAM_Q_SCALE, LDPC_Q_SCALE, LDPC_CLIP, compute_Er,
        rate_dematch_inverse,
    )

SLOT_STRIDE = N_STREAMS * N_SYM_PAD
STREAM_STRIDE = N_SYM_PAD
SCALE_INT = LDPC_Q_SCALE // QAM_Q_SCALE
ALIGN = 16
LPM = 3168
INPUT_RING_DEPTH = 2


def _cb_geometry(cb, e_all):
    """Return E/EQ/c0/delta and exact aligned DataCopy block descriptors."""
    e_cb = int(e_all[cb])
    eq = e_cb // Q_M
    c0 = int(e_all[:cb].sum() // Q_M)
    s0, re0 = divmod(c0, N_SYM)
    end_slot = (c0 + eq - 1) // N_SYM

    if end_slot == s0:
        re_a = (re0 // ALIGN) * ALIGN
        length = ((re0 + eq - re_a + ALIGN - 1) // ALIGN) * ALIGN
        blocks = [(s0, re_a, length, 0)]
    else:
        re_a0 = (re0 // ALIGN) * ALIGN
        length0 = N_SYM - re_a0
        re_hi = (c0 + eq) - (s0 + 1) * N_SYM
        length1 = ((re_hi + ALIGN - 1) // ALIGN) * ALIGN
        blocks = [(s0, re_a0, length0, 0), (s0 + 1, 0, length1, length0)]

    return e_cb, eq, c0, re0 % ALIGN, blocks


def _load_cb(flat_input, blocks):
    """Mirror MTE2 into one fixed-stride [stream,LPM] UB source slot."""
    ub_src = np.zeros(N_STREAMS * LPM, dtype=np.int16)
    for slot, re_a, length, dst_in_stream in blocks:
        for stream in range(N_STREAMS):
            gm = slot * SLOT_STRIDE + stream * STREAM_STRIDE + re_a
            ub = stream * LPM + dst_in_stream
            ub_src[ub : ub + length] = flat_input[gm : gm + length]
    return ub_src


def _simulate_kernel(descram, e_all):
    flat_input = descram.reshape(-1)
    lam = np.zeros((C_NUM, LDPC_N), dtype=np.int16)
    ring = [None] * INPUT_RING_DEPTH

    for cb in range(C_NUM):
        e_cb, eq, _c0, delta, blocks = _cb_geometry(cb, e_all)
        slot = cb % INPUT_RING_DEPTH
        ring[slot] = _load_cb(flat_input, blocks)

        e_pad = ((e_cb + ALIGN - 1) // ALIGN) * ALIGN
        # Nonzero sentinel makes a missing padding clear visible immediately.
        ub_e = np.full(e_pad, 12345, dtype=np.int32)
        if e_pad != e_cb:
            clear_start = (e_cb // ALIGN) * ALIGN
            ub_e[clear_start : clear_start + ALIGN] = 0
        for stream in range(N_STREAMS):
            src = stream * LPM + delta
            dst = stream * eq
            ub_e[dst : dst + eq] = ring[slot][src : src + eq]
        ub_e = np.clip(ub_e * SCALE_INT, -LDPC_CLIP, LDPC_CLIP).astype(np.int16)
        assert not ub_e[e_cb:e_pad].any()
        lam[cb, N_2Z : N_2Z + e_pad] = ub_e

    return lam


def simulate_and_verify(seed=2):
    e_all = compute_Er(G, C_NUM, Q_M, N_LAYERS)
    rng = np.random.default_rng(seed)
    descram = np.zeros((N_SLOT, N_STREAMS, N_SYM_PAD), dtype=np.int16)
    descram[:, :, :N_SYM] = rng.integers(
        -2560, 2561, (N_SLOT, N_STREAMS, N_SYM), dtype=np.int16
    )
    _, lam_ref, _ = rate_dematch_inverse(descram, scale=1.0)
    lam_sim = _simulate_kernel(descram, e_all)
    bad = int(np.count_nonzero(lam_sim != lam_ref))

    e_elems = ((int(e_all.max()) + 15) // 16) * 16 + 64
    ub_bytes = (INPUT_RING_DEPTH * 25344 * 2) + e_elems * 2
    print("--- v5 kernel simulation vs reference ---")
    print(f"  mismatching int16 elements : {bad}")
    print("  gather/template storage    : 0 bytes")
    print(f"  input ring depth           : {INPUT_RING_DEPTH}")
    print(f"  UB static budget           : {ub_bytes / 1024:.1f} KiB / 256 KiB")
    print(f"  result                     : {'PASS' if bad == 0 else 'FAIL'}")
    return bad == 0


def dump_and_verify(out_dir=None, seed=2):
    if out_dir is None:
        root = Path(os.environ.get("AIRAN_DATA_DIR", "../../.."))
        out_dir = root / "data/golden/rx/rate_dematch"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    e_all = compute_Er(G, C_NUM, Q_M, N_LAYERS)
    rng = np.random.default_rng(seed)
    descram = np.zeros((N_SLOT, N_STREAMS, N_SYM_PAD), dtype=np.int16)
    descram[:, :, :N_SYM] = rng.integers(
        -2560, 2561, (N_SLOT, N_STREAMS, N_SYM), dtype=np.int16
    )
    _, lam_ref, _ = rate_dematch_inverse(descram, scale=1.0)
    lam_sim = _simulate_kernel(descram, e_all)
    bad = int(np.count_nonzero(lam_sim != lam_ref))

    descram.tofile(out_dir / "descram_in.bin")
    lam_ref.tofile(out_dir / "lam_ref.bin")
    print(f"artifacts: {out_dir}")
    print(f"  descram_in.bin int16  [{N_SLOT},{N_STREAMS},{N_SYM_PAD}]")
    print(f"  lam_ref.bin    int16  [{C_NUM},{LDPC_N}]")
    print(f"  reload/kernel-sim mismatches: {bad} ({'PASS' if bad == 0 else 'FAIL'})")
    return bad == 0


if __name__ == "__main__":
    if simulate_and_verify():
        raise SystemExit(0 if dump_and_verify() else 1)
    raise SystemExit(1)
