#!/usr/bin/env python3
"""
mimo_dmrs_gen_ref.py — Golden generator for K-layer orthogonal DMRS kernel.

Writes per-case golden bins matching the physical compatibility layout:
    x_re.bin / x_im.bin : [4, 2, 896] fp16
The logical prefix is [L,2,896]; all inactive physical layers are zero.
    cinit.bin           : [16] int32 (2 active values followed by ABI padding)
    metadata.bin        : [32] uint32 private descriptor checked independently

Kernel output layout (layer-major): out[(l*N_SYM + s)*N_PAD + k]
  = base_gold(c_init[s])[k] × wf_l[k&1] × wt_l[s]   for k in [0,798), else 0.

Reuses dmrs_ref.py (dmrs_cinit, dmrs_qpsk) — base sequence identical to SISO.
"""
import os
import numpy as np
from pathlib import Path
import dmrs_ref as D

N_RE   = 798
N_PAD  = 896
N_SYM  = 2
MAX_NL = 4
DMRS_SYMS = D.DMRS_SYMS   # (2, 11)

# TS 38.211 Type-1 single-symbol port semantics shared with grid mapping/LS:
# comb delta={0,0,1,1}, Wf={++,+-,++,+-}, Wt=+1 for every occasion.
def port_semantics(port):
    if port not in (1000, 1001, 1002, 1003):
        raise ValueError(f"unsupported port {port}")
    index = port - 1000
    return index, index // 2, index & 1, (0, 0)  # negative-sign bits


def wf_pattern(port, n=N_RE):
    w = np.ones(n, dtype=np.float32)
    _, _, wf_odd_negative, _ = port_semantics(port)
    if wf_odd_negative:
        w[1::2] = -1.0          # odd k → -1
    return w


def wt_pattern(port):
    _, _, _, wt_negative = port_semantics(port)
    return np.where(np.asarray(wt_negative), -1.0, 1.0).astype(np.float32)


def build_metadata(ports):
    words = np.zeros(32, dtype=np.uint32)
    words[0] = 0x4D444731
    words[1] = len(ports)
    words[2] = N_SYM
    words[3:3 + N_SYM] = DMRS_SYMS
    for layer, port in enumerate(ports):
        index, comb_delta, wf_odd_negative, wt_negative = port_semantics(port)
        words[7 + layer] = index
        words[11 + layer] = comb_delta
        words[15 + layer] = wf_odd_negative
        words[19 + layer * N_SYM:19 + (layer + 1) * N_SYM] = wt_negative
    words[27] = N_SYM * N_PAD
    words[28] = N_PAD
    return words

# Cases cover rank, disjoint/shared combs, n_scid, and non-canonical port order.
CASE_MANIFEST = [
    dict(idx=0, name="baseline",  slot=0, n_id=1,   n_scid=0, ports=(1000,)),
    dict(idx=1, name="nid0",      slot=0, n_id=0,   n_scid=0, ports=(1000,1002)),
    dict(idx=2, name="slot7",     slot=7, n_id=1,   n_scid=0, ports=(1001,1000)),
    dict(idx=3, name="nscid1",    slot=0, n_id=1,   n_scid=1, ports=(1002,1000,1003)),
    dict(idx=4, name="nid_large", slot=0, n_id=300, n_scid=0, ports=(1003,1002,1001,1000)),
]

def gen_case_golden(slot, n_id, n_scid, ports):
    """Return cinit and zero-padded physical [4,2,896] output."""
    cinit = np.zeros(16, dtype=np.int32)
    cinit[:N_SYM] = [D.dmrs_cinit(slot, n_id, n_scid, int(l)) for l in DMRS_SYMS]

    x_re = np.zeros((MAX_NL, N_SYM, N_PAD), dtype=np.float16)
    x_im = np.zeros((MAX_NL, N_SYM, N_PAD), dtype=np.float16)

    for s, l in enumerate(DMRS_SYMS):
        base = D.dmrs_qpsk(int(cinit[s]), N_RE)   # complex64[798], shared base
        for layer, port in enumerate(ports):
            occ = wf_pattern(port) * wt_pattern(port)[s]
            r = base * occ
            x_re[layer, s, :N_RE] = r.real.astype(np.float16)
            x_im[layer, s, :N_RE] = r.imag.astype(np.float16)
    return cinit, x_re, x_im


def main():
    out_dir = Path(os.environ.get("AIRAN_DATA_DIR", ".")) / "data" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== mimo_dmrs_gen reference (standard logical output + physical capacity) ===")
    print(f"  physical layout: x_re/x_im [{MAX_NL},2,{N_PAD}] fp16, layer-major")
    print(f"  output: {out_dir}\n")

    for cfg in CASE_MANIFEST:
        cinit, x_re, x_im = gen_case_golden(cfg["slot"], cfg["n_id"], cfg["n_scid"], cfg["ports"])
        metadata = build_metadata(cfg["ports"])
        cdir = out_dir / f"case_{cfg['idx']}_{cfg['name']}"
        cdir.mkdir(parents=True, exist_ok=True)
        cinit.tofile(cdir / "cinit.bin")
        metadata.tofile(cdir / "metadata.bin")
        x_re.tofile(cdir / "x_re.bin")
        x_im.tofile(cdir / "x_im.bin")
        # sanity: layer 0 follows its configured port's OCC sequence.
        base0 = D.dmrs_qpsk(int(cinit[0]), N_RE)
        l0 = x_re[0, 0, :N_RE].astype(np.float32) + 1j*x_im[0, 0, :N_RE].astype(np.float32)
        layer0_expected = base0 * wf_pattern(cfg["ports"][0])
        match_port = np.allclose(l0, layer0_expected.astype(np.complex64), atol=2e-3)
        inactive_zero = not np.any(x_re[len(cfg['ports']):]) and not np.any(x_im[len(cfg['ports']):])
        print(f"  case {cfg['idx']} {cfg['name']:10s} L={len(cfg['ports'])} ports={cfg['ports']} "
              f"c_init=[0x{int(cinit[0]):08X},0x{int(cinit[1]):08X}]  "
              f"layer0_port_ok: {match_port} inactive_zero: {inactive_zero}")

    print("\n=== golden written ===")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
