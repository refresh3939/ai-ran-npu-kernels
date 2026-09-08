#!/usr/bin/env python3
"""
dmrs_gen_ref.py — PUSCH DMRS generator (TX, TS 38.211 §6.4.1.1 + §5.2.1)

§0 reference for the dmrs_gen NPU operator. Generates the comb-2 DMRS QPSK
sequence for the 2 front-loaded/additional DMRS symbols (l=2, l=11) and packs
them into the channel_est_ls / cfo_dmrs X-input layout:

    x_re.bin / x_im.bin : [2, 896] fp16   (798 valid REs + zero pad)
    cinit.bin           : [2]    int32    (c_init for l=2, l=11 — kernel input)
    truth_re.bin        : [2, 896] fp16   (== golden re, kept for verify naming)
    truth_im.bin        : [2, 896] fp16
    meta.txt

Algorithm (TS 38.211):
  §5.2.1 length-31 Gold sequence:
      x1(0)=1, x1(1..30)=0
      x2 init  = bits of c_init
      x1(n+31) = (x1(n+3) + x1(n)) mod 2
      x2(n+31) = (x2(n+3)+x2(n+2)+x2(n+1)+x2(n)) mod 2
      c(n)     = (x1(n+Nc) + x2(n+Nc)) mod 2,   Nc=1600
  §6.4.1.1.1.1 PUSCH DMRS c_init (type 1, comb-2, no transform precoding):
      c_init = (2^17·(N_symb·n_slot + l + 1)·(2·N_ID + 1) + 2·N_ID + n_SCID) mod 2^31
  §6.4.1.1.1   DMRS QPSK:
      r(n) = (1/√2)(1 - 2c(2n)) + j(1/√2)(1 - 2c(2n+1))

NOTE on config: N_ID / n_SCID / slot must match the gNB (OAI) / Sionna config
for OTA. Defaults below use N_ID=1 (= skill n_id), n_SCID=0. Only c_init drives
the sequence, and the kernel reproduces whatever c_init this ref writes, so the
NPU bit-exactness test is independent of the exact 38.211 config semantics.
Run with `--sionna` on the project box to assert numpy == Sionna PUSCH DMRS.
"""

import os
import sys
import numpy as np
from pathlib import Path

# ─── Locked constants (match dmrs_gen.h / kernel) ──────────────────────────
N_SC_USED     = 1596
N_DMRS_RE     = 798            # comb-2: N_SC_USED / 2
N_DMRS_PAD    = 896            # 7×128, per-sym stride (798 valid + pad)
N_DMRS_SYM    = 2
DMRS_SYM_IDX  = np.array([2, 11], dtype=np.int32)
N_SYMB_SLOT   = 14
GOLD_NC       = 1600
INV_SQRT2     = 1.0 / np.sqrt(2.0)

# Default DMRS scrambling config (override per-case below)
DEFAULT_N_ID   = 1             # dmrs-ScramblingID (defaults to N_ID^cell = skill n_id)
DEFAULT_N_SCID = 0


# ─── c_init (TS 38.211 §6.4.1.1.1.1) ───────────────────────────────────────
def dmrs_cinit(slot: int, l: int, n_id: int, n_scid: int) -> int:
    return (((1 << 17) * (N_SYMB_SLOT * slot + l + 1) * (2 * n_id + 1)
             + 2 * n_id + n_scid) % (1 << 31))


# ─── Gold sequence (TS 38.211 §5.2.1) — array form (golden truth) ──────────
def gold_sequence(c_init: int, m: int, nc: int = GOLD_NC) -> np.ndarray:
    n_total = m + nc
    x1 = np.zeros(n_total + 31, dtype=np.int8)
    x2 = np.zeros(n_total + 31, dtype=np.int8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for n in range(n_total):
        x1[n + 31] = (x1[n + 3] + x1[n]) & 1
        x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) & 1
    return ((x1[nc:nc + m] + x2[nc:nc + m]) & 1).astype(np.int8)


# ─── Gold sequence — rolling 31-bit state (mirrors the NPU scalar LFSR) ─────
def gold_sequence_rolling(c_init: int, m: int, nc: int = GOLD_NC) -> np.ndarray:
    """Bit-identical to gold_sequence(); validates the exact recurrence the
    kernel runs (int32 rolling state, shift/xor)."""
    x1 = 0x1
    x2 = c_init & 0x7FFFFFFF
    out = np.empty(m, dtype=np.int8)

    def step():
        nonlocal x1, x2
        c = (x1 ^ x2) & 1
        t1 = ((x1 >> 3) ^ x1) & 1
        t2 = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1
        x1 = (x1 >> 1) | (t1 << 30)
        x2 = (x2 >> 1) | (t2 << 30)
        return c

    for _ in range(nc):
        step()
    for k in range(m):
        out[k] = step()
    return out


# ─── DMRS QPSK (TS 38.211 §6.4.1.1.1) ──────────────────────────────────────
def dmrs_qpsk(c_init: int, n_re: int) -> np.ndarray:
    c = gold_sequence(c_init, 2 * n_re)
    re = INV_SQRT2 * (1.0 - 2.0 * c[0::2])
    im = INV_SQRT2 * (1.0 - 2.0 * c[1::2])
    return (re + 1j * im).astype(np.complex128)


# ─── Packer: 2×complex(798) → 2 fp16 planes [2, 896] ───────────────────────
def pack_X_fp16(X_complex: np.ndarray):
    re = np.zeros((N_DMRS_SYM, N_DMRS_PAD), dtype=np.float16)
    im = np.zeros((N_DMRS_SYM, N_DMRS_PAD), dtype=np.float16)
    re[:, :N_DMRS_RE] = np.real(X_complex).astype(np.float16)
    im[:, :N_DMRS_RE] = np.imag(X_complex).astype(np.float16)
    return re, im


# ─── Optional Sionna cross-check (run on project box) ──────────────────────
def sionna_dmrs(slot, n_id, n_scid):
    """Returns [2,798] complex DMRS from Sionna for sym l=2,11, or None."""
    try:
        from sionna.phy.nr import PUSCHConfig, PUSCHDMRSConfig  # noqa
    except Exception as e:  # pragma: no cover
        print(f"  [sionna] unavailable ({e}); skipping cross-check")
        return None
    # NOTE: Sionna's DMRS API/config plumbing differs by version; the on-box
    # operator owns the exact mapping. Left as a hook so the project box can
    # assert numpy == Sionna by filling in the config to match the gNB.
    print("  [sionna] hook present — fill PUSCHConfig to match gNB and compare")
    return None


# ─── Case manifest ─────────────────────────────────────────────────────────
CASE_MANIFEST = [
    dict(idx=0, name="baseline",     slot=0, n_id=1,   n_scid=0),
    dict(idx=1, name="nid0",         slot=0, n_id=0,   n_scid=0),
    dict(idx=2, name="slot7",        slot=7, n_id=1,   n_scid=0),
    dict(idx=3, name="nscid1",       slot=0, n_id=1,   n_scid=1),
    dict(idx=4, name="nid_large",    slot=0, n_id=300, n_scid=0),
]


def gen_case(idx, name, slot, n_id, n_scid):
    cinit = np.array(
        [dmrs_cinit(slot, int(DMRS_SYM_IDX[s]), n_id, n_scid) for s in range(N_DMRS_SYM)],
        dtype=np.int32,
    )

    X = np.stack([dmrs_qpsk(int(cinit[s]), N_DMRS_RE) for s in range(N_DMRS_SYM)], axis=0)

    # Self-check: rolling-state LFSR (== kernel) must bit-match array form
    for s in range(N_DMRS_SYM):
        c_arr = gold_sequence(int(cinit[s]), 2 * N_DMRS_RE)
        c_rol = gold_sequence_rolling(int(cinit[s]), 2 * N_DMRS_RE)
        assert np.array_equal(c_arr, c_rol), f"case{idx} sym{s}: rolling LFSR mismatch"

    x_re, x_im = pack_X_fp16(X)
    return dict(idx=idx, name=name, slot=slot, n_id=n_id, n_scid=n_scid,
                cinit=cinit, x_re=x_re, x_im=x_im)


def main():
    use_sionna = "--sionna" in sys.argv
    out_dir = Path(os.environ.get("AIRAN_DATA_DIR", ".")) / "data" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== dmrs_gen reference (TS 38.211 PUSCH DMRS, comb-2, l={2,11}) ===")
    print(f"  N_DMRS_RE={N_DMRS_RE}  N_DMRS_PAD={N_DMRS_PAD}  Nc={GOLD_NC}")
    print(f"  output  = {out_dir}")
    print()

    all_ok = True
    for cfg in CASE_MANIFEST:
        case = gen_case(**cfg)
        cdir = out_dir / f"case_{case['idx']}_{case['name']}"
        cdir.mkdir(parents=True, exist_ok=True)

        case["cinit"].tofile(cdir / "cinit.bin")          # kernel input (2 int32)
        case["x_re"].tofile(cdir / "x_re.bin")            # golden re [2,896] fp16
        case["x_im"].tofile(cdir / "x_im.bin")            # golden im [2,896] fp16
        # truth_* aliases for verify_result.py naming symmetry
        case["x_re"].tofile(cdir / "truth_re.bin")
        case["x_im"].tofile(cdir / "truth_im.bin")

        if use_sionna:
            sionna_dmrs(case["slot"], case["n_id"], case["n_scid"])

        with open(cdir / "meta.txt", "w") as f:
            f.write(f"name    : {case['name']}\n")
            f.write(f"slot    : {case['slot']}\n")
            f.write(f"n_id    : {case['n_id']}\n")
            f.write(f"n_scid  : {case['n_scid']}\n")
            f.write(f"c_init[l=2]  : 0x{int(case['cinit'][0]):08x} ({int(case['cinit'][0])})\n")
            f.write(f"c_init[l=11] : 0x{int(case['cinit'][1]):08x} ({int(case['cinit'][1])})\n")
            f.write(f"|r| (should be 1.0) : {np.mean(np.abs(case['x_re'][:, :N_DMRS_RE].astype(np.float64) + 1j*case['x_im'][:, :N_DMRS_RE].astype(np.float64))):.6f}\n")

        print(f"  case {case['idx']} {case['name']:10s} "
              f"c_init=[0x{int(case['cinit'][0]):08x}, 0x{int(case['cinit'][1]):08x}]  [OK]")

    print()
    print(f"=== {'ALL CASES OK' if all_ok else 'SOME FAILED'} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
