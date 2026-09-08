#!/usr/bin/env python3
# ============================================================================
# rate_dematch_ref.py  —  5G NR RX rate de-matching reference (TS 38.212 §5.4.2)
#
# Position in chain:
#   qam256_demod ──► descramble ──► [ rate_dematch ] ──► ldpc_decode
#
# Input  (= descramble output):  int16 Q11.5  [N_SLOT, N_STREAMS, N_SYM_PAD]
#                                stream-major, descrambled, NOT collapsed.
# Output (= ldpc_decode lam_in): int16 Q8.8   [C_NUM, LDPC_N]
#                                per-CB: [0:2Z]=0, [2Z:2Z+Ncb]=circular buffer.
#
# This file owns the FULL inverse of NR rate matching:
#   1. collapse        : skip pad REs, [slot][b][re] -> f_stream[ge*Qm + b]
#   2. CB de-concat    : split f_stream into C CBs of length E_r (§5.4.2.1)
#   3. bit de-interleave: §5.4.2.2 inverse,  e[n] = f[(n//Qm)+(n%Qm)*(E//Qm)]
#   4. circular buffer : rv0/k0=0 -> buf[0:E]=e, buf[E:Ncb]=0 (sub-buffer punct)
#   5. 2Z prefix       : lam[0:2Z]=0 (always-punctured systematic);
#                        filler (if K'<K) -> +LLR_CLIP   [GUARDED, see below]
#   6. rescale         : Q11.5 -> Q8.8, factor pinned empirically vs locked LDPC
#
# WHAT IS STANDARD vs WHAT IS OUR CHOICE
#   - Steps 1-5 (bit positions / puncturing) are 38.212-defined  -> MUST match.
#   - Step 6 LLR scale is NOT 3GPP -> fixed-point impl param, tuned for LDPC.
#
# VALIDATION
#   - selftest_roundtrip(): forward(38.212) then inverse -> identity on bit
#     positions. Runs here, proves invertibility + LUT correctness. NO Sionna.
#   - sionna_golden(): proves SPEC compliance (our forward == Sionna). Stub;
#     verify exact Sionna 1.2.2 API on the dev machine before trusting.
# ============================================================================
import numpy as np

# ---- Config (matches existing qam256_demod / descramble / ldpc_decode dims) --
Q_M        = 8           # 256-QAM
N_SYM      = 19152       # valid REs per slot
N_SYM_PAD  = 19200       # padded stride per stream (descramble GM layout)
N_STREAMS  = 8
N_SLOT     = 23
N_LAYERS   = 1

C_NUM      = 143         # code blocks
LDPC_Z     = 384
LDPC_K     = 8448        # KB*Z (systematic, incl filler region)
LDPC_N     = 26112       # NFULL*Z full codeword (incl 2Z punctured systematic)
N_2Z       = 2 * LDPC_Z  # 768  always-punctured systematic
N_CB_BUF   = 25344       # circular buffer = N - 2Z (no LBRM) = LDPC_N_RAW

RV_IDX     = 0           # only rv0 supported here (k0=0, no HARQ combining)
K_PRIME    = LDPC_K      # info+CRC per CB == K  =>  NO filler bits

# derived total coded bits per TB
G          = N_SLOT * N_SYM * Q_M          # 3,523,968

# Q-format
QAM_Q_SCALE  = 32        # Q11.5
QAM_CLIP     = 2560      # +-80.0 real
LDPC_Q_SCALE = 256       # Q8.8
LDPC_CLIP    = 5120      # +-20.0 real
FILLER_LLR   = LDPC_CLIP # known-0 filler -> large +LLR (used only if K'<K)


# ----------------------------------------------------------------------------
# 38.212 §5.4.2.1 : per-CB rate-matched length E_r
# ----------------------------------------------------------------------------
def compute_Er(G, C, Qm, NL):
    assert G % (NL * Qm) == 0, "G must be divisible by N_L*Q_m"
    base = G // (NL * Qm)              # 440496
    q    = base // C                   # floor(G/(NL*Qm*C)) = 3080
    nceil = base % C                   # # of CBs taking ceil = 56
    E = np.empty(C, dtype=np.int64)
    for r in range(C):
        if r <= C - nceil - 1:        # first (C-nceil) CBs: floor
            E[r] = NL * Qm * q
        else:                         # last nceil CBs: ceil
            E[r] = NL * Qm * (q + 1)
    assert E.sum() == G, (E.sum(), G)
    assert np.all(E % Qm == 0)
    assert np.all(E <= N_CB_BUF), "E_r > N_cb would mean repetition (rv/LBRM)"
    return E


# ----------------------------------------------------------------------------
# Host gather LUT.  For each CB, src_idx[n] = linear index into the
# descramble-output buffer [N_SLOT, N_STREAMS, N_SYM_PAD] for lam[cb, 2Z+n],
# n in [0, E_r).  Encodes: de-interleave (§5.4.2.2) + CB de-concat + collapse.
# Returns: src_flat (int32, len G), cb_off (len C), E (len C).
# ----------------------------------------------------------------------------
def build_gather_lut(E):
    src_flat = np.empty(int(E.sum()), dtype=np.int64)
    cb_off   = np.zeros(C_NUM, dtype=np.int64)
    acc = 0
    f_base = 0  # running offset of CB cb inside f_stream (= CB concat order)
    for cb in range(C_NUM):
        Ecb = int(E[cb]); EQ = Ecb // Q_M
        cb_off[cb] = acc
        n = np.arange(Ecb)
        # §5.4.2.2 inverse (row-column, Qm rows): e[n] = f_cb[(n%EQ)*Qm + (n//EQ)]
        m = (n % EQ) * Q_M + (n // EQ)
        # f_cb[m] = f_stream[f_base + m]
        k  = f_base + m
        ge = k // Q_M          # global RE index (valid REs only, across slots)
        b  = k %  Q_M          # bit-position within symbol == stream index
        slot = ge // N_SYM
        re   = ge %  N_SYM     # re < N_SYM (valid), but stride uses N_SYM_PAD
        src_flat[acc:acc + Ecb] = (slot * (N_STREAMS * N_SYM_PAD)
                                   + b * N_SYM_PAD + re)
        acc    += Ecb
        f_base += Ecb
    return src_flat, cb_off, E


# ----------------------------------------------------------------------------
# RX reference. descram_out: [N_SLOT, N_STREAMS, N_SYM_PAD] Q11.5 (int).
# Returns lam_in [C_NUM, LDPC_N] as float (real LLR) BEFORE Q-format rescale,
# plus the int16 Q8.8 quantized version for the chosen `scale`.
# ----------------------------------------------------------------------------
def rate_dematch_inverse(descram_out, scale=None):
    if K_PRIME != LDPC_K:
        raise NotImplementedError(
            "Filler bits (K'<K) not implemented; current config K'=K (no filler). "
            "If enabled: lam[cb, K':K] must be set to +FILLER_LLR and skipped "
            "during the circular-buffer mapping (38.212 §5.4.2.1).")
    if RV_IDX != 0:
        raise NotImplementedError("Only rv0 (k0=0, no HARQ combining) supported.")

    E = compute_Er(G, C_NUM, Q_M, N_LAYERS)
    src_flat, cb_off, _ = build_gather_lut(E)

    flat_in = descram_out.reshape(-1)                 # linear view of GM buffer
    gathered = flat_in[src_flat].astype(np.float64)   # one big gather (= kernel)

    lam = np.zeros((C_NUM, LDPC_N), dtype=np.float64)  # [0:2Z]=0 prefix built-in
    for cb in range(C_NUM):
        Ecb = int(E[cb]); off = int(cb_off[cb])
        # buf[0:Ecb] = received e ; buf[Ecb:Ncb] = 0 (sub-buffer puncture tail)
        lam[cb, N_2Z:N_2Z + Ecb] = gathered[off:off + Ecb]
        # lam[cb, N_2Z+Ecb : N_2Z+N_CB_BUF] stays 0
    # real LLR carried in Q11.5 integer units so far; convert to real:
    lam_real = lam / QAM_Q_SCALE

    lam_q8_8 = None
    if scale is not None:
        # scale is applied to the real LLR, then encode Q8.8 + clip
        w = np.round(lam_real * scale * LDPC_Q_SCALE)
        w = np.clip(w, -LDPC_CLIP, LDPC_CLIP).astype(np.int16)
        lam_q8_8 = w
    return lam_real, lam_q8_8, E


# ----------------------------------------------------------------------------
# Forward NR rate matching (38.212), rv0/no-filler — ONLY for the self-test.
# Produces a descramble-output-layout buffer from per-CB circular buffers `d`.
# ----------------------------------------------------------------------------
def _forward_to_descram_layout(d_cb, E):
    # d_cb: list/array [C_NUM, N_CB_BUF] circular buffers (values to track)
    f_stream = np.zeros(G, dtype=d_cb.dtype)
    f_base = 0
    for cb in range(C_NUM):
        Ecb = int(E[cb]); EQ = Ecb // Q_M
        e = d_cb[cb, 0:Ecb]                       # bit selection rv0: first E of d
        f = np.empty(Ecb, dtype=d_cb.dtype)
        # §5.4.2.2 forward (row-column, Qm rows): f[i*Qm + j] = e[i + j*EQ]
        idx_i = np.arange(EQ)
        for j in range(Q_M):
            f[idx_i * Q_M + j] = e[idx_i + j * EQ]
        f_stream[f_base:f_base + Ecb] = f
        f_base += Ecb
    # scatter f_stream into [N_SLOT, N_STREAMS, N_SYM_PAD] (valid REs only)
    descram = np.zeros((N_SLOT, N_STREAMS, N_SYM_PAD), dtype=d_cb.dtype)
    k = np.arange(G)
    ge = k // Q_M; b = k % Q_M
    slot = ge // N_SYM; re = ge % N_SYM
    descram[slot, b, re] = f_stream
    return descram


def selftest_roundtrip():
    E = compute_Er(G, C_NUM, Q_M, N_LAYERS)
    # distinct sentinels: d_cb[cb, k] = cb*100000 + k  (float, exact in f64)
    d_cb = np.zeros((C_NUM, N_CB_BUF), dtype=np.float64)
    for cb in range(C_NUM):
        d_cb[cb] = cb * 100000 + np.arange(N_CB_BUF)
    descram = _forward_to_descram_layout(d_cb, E)
    lam_real, _, E2 = rate_dematch_inverse(descram, scale=None)
    assert np.array_equal(E, E2)
    # lam_real is in Q11.5/32 units; sentinels were placed as raw ints -> /32
    lam = lam_real * QAM_Q_SCALE
    ok = True
    for cb in range(C_NUM):
        Ecb = int(E[cb])
        exp = d_cb[cb, 0:Ecb]                       # rv0: received == d[0:E]
        got = lam[cb, N_2Z:N_2Z + Ecb]
        if not np.allclose(got, exp):
            ok = False; print(f"  CB {cb}: MISMATCH"); break
        # tails / prefix must be exactly zero
        if lam[cb, 0:N_2Z].any() or lam[cb, N_2Z + Ecb:].any():
            ok = False; print(f"  CB {cb}: nonzero in punctured region"); break
    print(f"E_r distribution: {np.unique(E, return_counts=True)}")
    print(f"sum(E)=G : {E.sum()} == {G} : {E.sum()==G}")
    print(f"round-trip identity (forward 38.212 -> inverse): {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------
# Sionna golden (STUB).  Run on the dev machine in the `sionna` conda env.
# Purpose: (a) prove our forward == Sionna's rate matcher (spec compliance),
#          (b) generate realistic channel LLRs to PIN the rescale `scale`.
# NOTE: exact Sionna 1.2.2 API NOT asserted here — verify with
#       `inspect.signature(...)` before use. The interleaver/bit-selection are
#       inside sionna.phy.fec.ldpc utilities; pull the index map and compare to
#       build_gather_lut() at the bit level (must be diff==0).
# ----------------------------------------------------------------------------
def sionna_golden():
    raise NotImplementedError(
        "Fill in on dev machine:\n"
        "  1. Build a TB with the config above (256QAM, 1 layer, 143 CB, BG1 Z=384,\n"
        "     rv0, TBS giving K'=8448 / no filler).\n"
        "  2. Run encode + rate_match + scramble + 256QAM map + AWGN + demod to get\n"
        "     the descramble-output-layout LLR array AND Sionna's de-rate-matched\n"
        "     reference lam (length-N per CB, 0 at punctured).\n"
        "  3. assert our build_gather_lut() reproduces Sionna's interleave/selection\n"
        "     indices bit-exactly (diff==0).\n"
        "  4. Sweep `scale` in rate_dematch_inverse(); feed int16 Q8.8 to the locked\n"
        "     ldpc_decode; pick scale giving BER 0.0000% (matches verified regime).")


if __name__ == "__main__":
    print("=== rate_dematch reference self-test (38.212 §5.4.2, rv0, no filler) ===")
    selftest_roundtrip()
