#!/usr/bin/env python3
# ============================================================================
# dmrs_ref.py — 5G NR PUSCH DMRS reference sequence (TS 38.211 §6.4.1.1)
#
# SINGLE SOURCE OF TRUTH for the DMRS pilot sequence r(n), shared by:
#   - TX dmrs_gen   (kernel; cross-check: TX-view dmrs[2,896] fp16)
#   - RX channel_est X_ref / pilot.bin   (RX-view pilot[2,798] cint16 Q14)
#
# Reproduces tx_chain BuildDmrsMatrix + fill_cinit BIT-FOR-BIT so RX regenerates
# the *exact* transmitted pilots (real OTA receiver behaviour: no file passing,
# RX knows the DMRS config and locally regenerates the reference).
#
# Gold sequence (TS 38.211 §5.2.1):
#   x1(n+31) = (x1(n+3) + x1(n)) mod 2,            x1: [1,0,0,...]
#   x2(n+31) = (x2(n+3)+x2(n+2)+x2(n+1)+x2(n))mod2, x2(i)=(c_init>>i)&1, i=0..30
#   c(n) = (x1(n+Nc) + x2(n+Nc)) mod 2,            Nc = 1600
# DMRS QPSK (§6.4.1.1.1):
#   r(n) = (1/sqrt2)(1-2c(2n)) + j(1/sqrt2)(1-2c(2n+1))
# c_init (PUSCH DMRS §6.4.1.1.1.1; == tx_chain fill_cinit):
#   c_init = ( 2^17 (14*slot + l + 1)(2*N_ID + 1) + (2*N_ID + n_scid) ) mod 2^31
# ============================================================================
import os
import numpy as np

# --- config (must match TX dmrs_gen launch params) --------------------------
N_ID     = 0           # scrambling ID
N_SCID   = 0           # DMRS scrambling ID flag
SLOT     = 0           # slot number within frame
DMRS_L   = (2, 11)     # DMRS OFDM symbol indices

N_RE     = 798         # DMRS REs per symbol (comb-2 over 1596 raw SC)
N_PAD_TX = 896         # dmrs_gen output buffer width (798 real + 98 zero-pad)
NC       = 1600        # Gold sequence offset


def dmrs_cinit(slot, l, n_id=N_ID, n_scid=N_SCID):
    v = ((1 << 17) * (14 * slot + l + 1) * (2 * n_id + 1)
         + (2 * n_id + n_scid))
    return int(v & 0x7FFFFFFF)


def gold_sequence(c_init, length):
    """TS 38.211 5.2.1 length-31 Gold sequence c(0..length-1)."""
    n_total = length + NC
    x1 = np.zeros(n_total + 31, dtype=np.int8)
    x2 = np.zeros(n_total + 31, dtype=np.int8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for n in range(n_total):
        x1[n + 31] = (x1[n + 3] ^ x1[n]) & 1
        x2[n + 31] = (x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n]) & 1
    return (x1[NC:NC + length] ^ x2[NC:NC + length]) & 1


def dmrs_seq(slot, l, n_id=N_ID, n_scid=N_SCID, n_re=N_RE):
    """r(n), n=0..n_re-1, complex64 unit-modulus QPSK (|r|=1)."""
    c = gold_sequence(dmrs_cinit(slot, l, n_id, n_scid), 2 * n_re)
    r = (1.0 / np.sqrt(2.0)) * ((1 - 2 * c[0::2]).astype(np.float32)
                                + 1j * (1 - 2 * c[1::2]).astype(np.float32))
    return r.astype(np.complex64)


def dmrs_all(slot=SLOT, n_id=N_ID, n_scid=N_SCID, n_re=N_RE):
    """[2, n_re] complex64 for the two DMRS symbols {2,11}."""
    return np.stack([dmrs_seq(slot, l, n_id, n_scid, n_re) for l in DMRS_L])


# --- format emitters --------------------------------------------------------
def to_tx_fp16(r2, n_pad=N_PAD_TX):
    """TX-view dmrs[2, n_pad] fp16 separate re/im (matches dmrs_gen output)."""
    re = np.zeros((2, n_pad), dtype=np.float16)
    im = np.zeros((2, n_pad), dtype=np.float16)
    re[:, :r2.shape[1]] = r2.real.astype(np.float16)
    im[:, :r2.shape[1]] = r2.imag.astype(np.float16)
    return re, im


def to_cint16_pilot(r2, q_bits=14):
    """RX-view pilot[2, N_RE] cint16 Q14 interleaved (== channel_est pilot.bin)."""
    q = 1 << q_bits
    qmax, qmin = (1 << 15) - 1, -(1 << 15)
    out = np.empty((r2.shape[0], r2.shape[1] * 2), dtype=np.int16)
    out[:, 0::2] = np.clip(np.round(r2.real * q), qmin, qmax).astype(np.int16)
    out[:, 1::2] = np.clip(np.round(r2.imag * q), qmin, qmax).astype(np.int16)
    return out


def cross_check_dmrs_gen(golden_re_path, golden_im_path, tol=0):
    """Compare TX-view fp16 vs dmrs_gen standalone golden (bit-exact expected)."""
    r2 = dmrs_all()
    re, im = to_tx_fp16(r2)
    g_re = np.fromfile(golden_re_path, dtype=np.float16).reshape(2, -1)
    g_im = np.fromfile(golden_im_path, dtype=np.float16).reshape(2, -1)
    w = min(re.shape[1], g_re.shape[1])
    bad = int((re[:, :w].view(np.uint16) != g_re[:, :w].view(np.uint16)).sum()
              + (im[:, :w].view(np.uint16) != g_im[:, :w].view(np.uint16)).sum())
    print(f"  dmrs_ref vs dmrs_gen golden : {bad} fp16-bit mismatches "
          f"({'PASS' if bad <= tol else 'FAIL'})  [width {w}]")
    return bad <= tol


if __name__ == "__main__":
    r2 = dmrs_all()
    print("=== dmrs_ref (TS 38.211 6.4.1.1, PUSCH DMRS type-1 comb-2) ===")
    print(f"  config: N_ID={N_ID} n_scid={N_SCID} slot={SLOT} syms={DMRS_L} N_RE={N_RE}")
    for s, l in enumerate(DMRS_L):
        ci = dmrs_cinit(SLOT, l)
        print(f"  sym {l:2d}: c_init=0x{ci:08X}  r[0:3]={np.round(r2[s,:3],4)}  "
              f"|r|={np.abs(r2[s,0]):.4f}")
    re, im = to_tx_fp16(r2)
    pilot = to_cint16_pilot(r2)
    print(f"  TX-view  dmrs[2,{N_PAD_TX}] fp16  (real pilots [:, :{N_RE}])")
    print(f"  RX-view  pilot[2,{N_RE}] cint16 Q14 interleaved (== pilot.bin)")
    out = os.environ.get("DMRS_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        re.tofile(f"{out}/dmrs_re.bin"); im.tofile(f"{out}/dmrs_im.bin")
        pilot.tofile(f"{out}/pilot.bin")
        print(f"  dumped TX dmrs_re/im.bin + RX pilot.bin -> {out}")
