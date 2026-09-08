#!/usr/bin/env python3
# ============================================================================
# rate_match_ref.py  —  5G NR TX rate matching reference (TS 38.212 §5.4.2)
# STANDARDIZED CONTRACT (M1b): rate_match consumes ldpc_encode NATIVE output.
#
#   ldpc_encode --> [ rate_match ] --> scramble --> qam256_mod
#
# In  (= ldpc_encode native output): int8 [C_NUM, N_CB_BUF=25344] per-CB
#     PUNCTURED codeword cw (2Z already removed), bits {0,1}, contiguous.
#     cw[cb, 0:E] = transmitted (rv0).  NO 2Z prefix, NO int16, NO host repack.
# Out (= scramble input): int16 [N_SLOT, N_STREAMS, N_SYM_PAD] stream-major
#     bits; pad REs = 0.
#
# (旧契约 int16 [C_NUM,26112] 母码读 d[cb,2Z:2Z+E] 与 ldpc_encode 实际 25344
#  步长不符 -> host repack 插 2Z gap+cast -> CB>0 错位 768*cb. 已废弃.)
#
# Forward = bit-level inverse of rate_dematch_inverse:
#   e = cw[cb, 0:E]   (was d[cb, 2Z:2Z+E])
#   out[slot, b, c0+i] = cw[cb, b*EQ + i]
#
# RX rate_dematch UNCHANGED: still outputs 26112-mother lam with 2Z erasure at
# lam[0:2Z]. Only the TX read offset moved 2Z->0 (encoder already punctured).
# ============================================================================
import numpy as np

Q_M        = 8
N_SYM      = 19152
N_SYM_PAD  = 19200
N_STREAMS  = 8
N_SLOT     = 23
N_LAYERS   = 1

C_NUM      = 143
LDPC_Z     = 384
LDPC_K     = 8448
LDPC_N     = 26112          # mother-code length (RX lam width; inverse only)
N_2Z       = 2 * LDPC_Z     # 768  (RX-side erasure prefix only)
N_CB_BUF   = 25344          # LDPC_N - 2Z : TX punctured-codeword length (int8 in)
RV_IDX     = 0
K_PRIME    = LDPC_K

G          = N_SLOT * N_SYM * Q_M          # 3,523,968

QAM_Q_SCALE  = 32
LDPC_Q_SCALE = 256
LDPC_CLIP    = 5120

SLOT_STRIDE   = N_STREAMS * N_SYM_PAD      # 153600
STREAM_STRIDE = N_SYM_PAD                  # 19200


def compute_Er(G, C, Qm, NL):
    assert G % (NL * Qm) == 0
    base = G // (NL * Qm)
    q    = base // C
    nceil = base % C
    E = np.empty(C, dtype=np.int64)
    for r in range(C):
        E[r] = NL * Qm * (q if r <= C - nceil - 1 else q + 1)
    assert E.sum() == G and np.all(E % Qm == 0) and np.all(E <= N_CB_BUF)
    return E


def build_gather_lut(E):
    src_flat = np.empty(int(E.sum()), dtype=np.int64)
    cb_off   = np.zeros(C_NUM, dtype=np.int64)
    acc = 0; f_base = 0
    for cb in range(C_NUM):
        Ecb = int(E[cb]); EQ = Ecb // Q_M
        cb_off[cb] = acc
        n = np.arange(Ecb)
        m = (n % EQ) * Q_M + (n // EQ)
        k = f_base + m
        ge = k // Q_M; b = k % Q_M
        slot = ge // N_SYM; re = ge % N_SYM
        src_flat[acc:acc + Ecb] = (slot * (N_STREAMS * N_SYM_PAD)
                                   + b * N_SYM_PAD + re)
        acc += Ecb; f_base += Ecb
    return src_flat, cb_off, E


def rate_dematch_inverse(descram_out, scale=None):
    """RX side (UNCHANGED): gather stream layout -> 26112-mother lam[2Z:2Z+E]."""
    E = compute_Er(G, C_NUM, Q_M, N_LAYERS)
    src_flat, cb_off, _ = build_gather_lut(E)
    flat_in = descram_out.reshape(-1)
    gathered = flat_in[src_flat].astype(np.float64)
    lam = np.zeros((C_NUM, LDPC_N), dtype=np.float64)
    for cb in range(C_NUM):
        Ecb = int(E[cb]); off = int(cb_off[cb])
        lam[cb, N_2Z:N_2Z + Ecb] = gathered[off:off + Ecb]
    lam_real = lam / QAM_Q_SCALE
    lam_q8_8 = None
    if scale is not None:
        w = np.round(lam_real * scale * LDPC_Q_SCALE)
        lam_q8_8 = np.clip(w, -LDPC_CLIP, LDPC_CLIP).astype(np.int16)
    return lam_real, lam_q8_8, E


def _forward_to_descram_layout(cw, E):
    """cw[cb, 0:Ecb] = punctured transmitted bits (already [0:E])."""
    f_stream = np.zeros(G, dtype=cw.dtype)
    f_base = 0
    for cb in range(C_NUM):
        Ecb = int(E[cb]); EQ = Ecb // Q_M
        e = cw[cb, 0:Ecb]
        f = np.empty(Ecb, dtype=cw.dtype)
        idx_i = np.arange(EQ)
        for j in range(Q_M):
            f[idx_i * Q_M + j] = e[idx_i + j * EQ]
        f_stream[f_base:f_base + Ecb] = f
        f_base += Ecb
    # Kernel ABI writes int16 layout even though the LDPC input bits are int8.
    descram = np.zeros((N_SLOT, N_STREAMS, N_SYM_PAD), dtype=np.int16)
    k = np.arange(G); ge = k // Q_M; b = k % Q_M
    slot = ge // N_SYM; re = ge % N_SYM
    descram[slot, b, re] = f_stream
    return descram


def _cb_start(E):
    EQ = (E // Q_M).astype(np.int64)
    c0 = np.concatenate(([0], np.cumsum(EQ)[:-1]))
    return EQ, c0


def rate_match_forward(cw):
    """out[slot,b,c0+i] = cw[cb, b*EQ + i]  (PUNCTURED cw read at [0:E])."""
    E = compute_Er(G, C_NUM, Q_M, N_LAYERS)
    EQ_all, c0_all = _cb_start(E)
    # Keep the reference file dtype identical to layout_gm/main.cpp.
    out = np.zeros((N_SLOT, N_STREAMS, N_SYM_PAD), dtype=np.int16)
    flat = out.reshape(-1)
    for cb in range(C_NUM):
        EQ = int(EQ_all[cb]); c0 = int(c0_all[cb])
        e = cw[cb, 0:N_STREAMS * EQ]                # <-- [0:E], no 2Z offset
        for b in range(N_STREAMS):
            seg = e[b * EQ:(b + 1) * EQ]
            for i in range(EQ):
                ge = c0 + i
                slot = ge // N_SYM; re = ge % N_SYM
                flat[slot * SLOT_STRIDE + b * STREAM_STRIDE + re] = seg[i]
    return out


def selftest(seed=3):
    print("=== rate_match forward reference self-test (38.212 5.4.2, rv0) ===")
    print(f"    contract: cw int8 [C_NUM={C_NUM}, N_CB_BUF={N_CB_BUF}] punctured, read [0:E]")
    E = compute_Er(G, C_NUM, Q_M, N_LAYERS)
    rng = np.random.default_rng(seed)
    cw = rng.integers(0, 2, size=(C_NUM, N_CB_BUF)).astype(np.int8)  # int8 punctured
    layout = rate_match_forward(cw)

    okA = np.array_equal(layout, _forward_to_descram_layout(cw, E))
    print(f"  forward == _forward_to_descram_layout : {okA}")

    _, lam, _ = rate_dematch_inverse(layout.astype(np.int16), scale=1.0)
    okB = True
    for cb in range(C_NUM):
        Ecb = int(E[cb])
        exp = cw[cb, 0:Ecb].astype(np.int32) * 8
        got = lam[cb, N_2Z:N_2Z + Ecb].astype(np.int32)
        if not np.array_equal(got, exp) or lam[cb, :N_2Z].any() \
           or lam[cb, N_2Z + Ecb:].any():
            okB = False; print(f"  CB {cb}: round-trip MISMATCH"); break
    print(f"  rate_dematch_inverse(forward(cw)) recovers e : {okB}")

    okC = bool((layout[:, :, N_SYM:] == 0).all())
    print(f"  pad REs [{N_SYM}:{N_SYM_PAD}] all zero       : {okC}")

    ok = okA and okB and okC
    print(f"  rate_match forward bit-exact inverse of rate_dematch: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    selftest()
