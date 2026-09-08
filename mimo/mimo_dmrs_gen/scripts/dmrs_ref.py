#!/usr/bin/env python3
# ============================================================================
# dmrs_ref.py — SINGLE SOURCE OF TRUTH for 5G NR PUSCH DMRS pilots.
# TS 38.211 §6.4.1.1.1 (Gold sequence) + §7.4.1.1.1 c_init.
#
# Both sides MUST derive the DMRS pilot from the SAME (n_id, n_scid, slot, l):
#   TX : dmrs_gen kernel  (matricized Gold: g1=x1, gmat=x2 unit responses)
#   RX : channel_est X_ref  (= re-run dmrs_gen with identical c_init)
# This file proves the scalar Gold reference == the TX matricized form
# (tx_chain.cpp::BuildDmrsMatrix) bit-for-bit, so RX re-generation is exact.
#
# Contract (locked, matches tx_chain + channel_est):
#   - DMRS symbols  l ∈ {2, 11}        (front-loaded + additional position)
#   - comb-2, port 0, CDM group 0      (even SC 0,2,…,1594)
#   - N_RE = 798 real pilots / sym     (= 1596/2);  TX pads→896, CE pads→832
#   - QPSK: r[k] = (1-2c[2k])/√2 + j(1-2c[2k+1])/√2
#   - Nc = 1600 Gold offset, M = 1596 gold bits
# ============================================================================
import numpy as np

# ── locked constants (MUST match tx_chain BuildDmrsMatrix + channel_est) ────
NC          = 1600
N_RE        = 798            # real pilots per DMRS symbol
M           = 2 * N_RE       # 1596 gold bits
N_PAD_TX    = 896            # dmrs_gen buffer width [2,896]
N_PAD_CE    = 832            # channel_est X_ref width [2,832]
DMRS_SYMS   = (2, 11)        # DMRS_SYM_0, DMRS_SYM_1
INV_SQRT2   = np.float32(1.0 / np.sqrt(2.0))


# ── c_init (TS 38.211 6.4.1.1.1.1) — matches tx_chain::fill_cinit ──────────
def dmrs_cinit(slot, n_id, n_scid, l):
    """c_init for OFDM symbol l in slot. Identical to tx_chain fill_cinit[l]."""
    v = ((1 << 17) * (14 * slot + l + 1) * (2 * n_id + 1)
         + (2 * n_id + n_scid))
    return int(v & 0x7FFFFFFF)


# ── scalar Gold sequence c[0:M] (TS 38.211 5.2.1) ──────────────────────────
def _gold_bits(c_init, length=M, nc=NC):
    n = length + nc
    x1 = np.zeros(n + 31, dtype=np.int8); x1[0] = 1
    for i in range(n):
        x1[i + 31] = (x1[i + 3] ^ x1[i]) & 1
    x2 = np.zeros(n + 31, dtype=np.int8)
    for j in range(31):
        x2[j] = (c_init >> j) & 1
    for i in range(n):
        x2[i + 31] = (x2[i + 3] ^ x2[i + 2] ^ x2[i + 1] ^ x2[i]) & 1
    return (x1[nc:nc + length] ^ x2[nc:nc + length]).astype(np.int8)


def dmrs_qpsk(c_init, n_re=N_RE):
    """Scalar TS 38.211 DMRS pilots: complex64[n_re]."""
    c = _gold_bits(c_init, 2 * n_re)
    re = (1.0 - 2.0 * c[0::2]).astype(np.float32) * INV_SQRT2
    im = (1.0 - 2.0 * c[1::2]).astype(np.float32) * INV_SQRT2
    return (re + 1j * im).astype(np.complex64)


# ── TX matricized form (replicates tx_chain::BuildDmrsMatrix) ──────────────
def _x1_split():
    """g1: x1 bits split re/im (g1[k]=x1[NC+2k], g1[N_PAD+k]=x1[NC+2k+1])."""
    n = M + NC
    x1 = np.zeros(n + 31, dtype=np.int8); x1[0] = 1
    for i in range(n):
        x1[i + 31] = (x1[i + 3] ^ x1[i]) & 1
    seq = x1[NC:NC + M]
    return seq[0::2].copy(), seq[1::2].copy()      # re bits, im bits  (len 798)


def _x2_unit_planes():
    """gmat: for each c_init bit i, x2 unit-response, split re/im. [31][2][798]."""
    n = M + NC
    planes = np.zeros((31, 2, N_RE), dtype=np.int8)
    for i in range(31):
        x2 = np.zeros(n + 31, dtype=np.int8)
        x2[i] = 1
        for k in range(n):
            x2[k + 31] = (x2[k + 3] ^ x2[k + 2] ^ x2[k + 1] ^ x2[k]) & 1
        seq = x2[NC:NC + M]
        planes[i, 0] = seq[0::2]
        planes[i, 1] = seq[1::2]
    return planes


def dmrs_qpsk_matricized(c_init, _g1=None, _gmat=None):
    """Reconstruct pilots the way dmrs_gen kernel does: c = x1 ⊕ (⊕_i bit_i·gmat_i)."""
    g1_re, g1_im = _g1 if _g1 is not None else _x1_split()
    gmat = _gmat if _gmat is not None else _x2_unit_planes()
    x2_re = np.zeros(N_RE, dtype=np.int8)
    x2_im = np.zeros(N_RE, dtype=np.int8)
    for i in range(31):
        if (c_init >> i) & 1:
            x2_re ^= gmat[i, 0]
            x2_im ^= gmat[i, 1]
    c_re = g1_re ^ x2_re
    c_im = g1_im ^ x2_im
    re = (1.0 - 2.0 * c_re).astype(np.float32) * INV_SQRT2
    im = (1.0 - 2.0 * c_im).astype(np.float32) * INV_SQRT2
    return (re + 1j * im).astype(np.complex64)


# ── X_ref builder for channel_est (fp16 separate planes [2, N_PAD_CE]) ──────
def build_xref(slot, n_id, n_scid, pad=N_PAD_CE):
    """Return (x_re, x_im) fp16 [2, pad]: 798 real Gold pilots + zero pad.
       This is EXACTLY what channel_est needs as X_ref for the given slot."""
    x_re = np.zeros((2, pad), dtype=np.float16)
    x_im = np.zeros((2, pad), dtype=np.float16)
    for s, l in enumerate(DMRS_SYMS):
        r = dmrs_qpsk(dmrs_cinit(slot, n_id, n_scid, l))
        x_re[s, :N_RE] = r.real.astype(np.float16)
        x_im[s, :N_RE] = r.imag.astype(np.float16)
    return x_re, x_im


# ── self-test: scalar == matricized (proves RX re-gen == TX dmrs_gen) ───────
def selftest():
    print("=== dmrs_ref self-test (scalar Gold == TX matricized form) ===")
    g1 = _x1_split()
    gmat = _x2_unit_planes()
    ok = True
    # sweep representative params incl. the two DMRS symbols
    for (slot, n_id, n_scid) in [(0, 0, 0), (0, 1, 0), (5, 300, 1), (22, 500, 0)]:
        for l in DMRS_SYMS:
            ci = dmrs_cinit(slot, n_id, n_scid, l)
            a = dmrs_qpsk(ci)
            b = dmrs_qpsk_matricized(ci, g1, gmat)
            match = np.array_equal(a, b)
            ok &= match
            print(f"  slot={slot:2d} n_id={n_id:3d} n_scid={n_scid} l={l:2d} "
                  f"c_init=0x{ci:08X}  scalar==matricized: {match}")
    # also check raw c_init bit-extreme cases
    for ci in [0, 1, 0x7FFFFFFF, 0x55555555]:
        ok &= np.array_equal(dmrs_qpsk(ci), dmrs_qpsk_matricized(ci, g1, gmat))
    print(f"  ALL: {'PASS' if ok else 'FAIL'}")
    # X_ref shape/values sanity
    xr, xi = build_xref(0, 0, 0)
    unit = np.allclose(np.abs(xr[:, :N_RE].astype(np.float32))**2
                       + np.abs(xi[:, :N_RE].astype(np.float32))**2, 1.0, atol=2e-3)
    print(f"  X_ref[2,{N_PAD_CE}] fp16, |pilot|²≈1 (798 real): {unit}, "
          f"pad[{N_RE}:] zero: {bool((xr[:, N_RE:]==0).all() and (xi[:, N_RE:]==0).all())}")
    return ok


if __name__ == "__main__":
    selftest()
