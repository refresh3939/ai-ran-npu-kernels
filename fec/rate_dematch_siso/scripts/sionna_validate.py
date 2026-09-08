#!/usr/bin/env python3
# ============================================================================
# sionna_validate.py — validate rate_dematch_ref against Sionna 1.2.x (TS 38.212)
# Run in the `sionna` conda env on the dev machine (same dir as rate_dematch_ref.py).
#
# CHECK 1  check_interleaver()  [pure Python, no NPU]
#   Black-box validation of §5.4.2.2 bit-interleave (and its inverse) against
#   Sionna's own rate matcher — WITHOUT extracting any Sionna internals.
#   Trick: LDPC5GEncoder(..., num_bits_per_symbol=Qm) applies the §5.4.2.2
#   interleaver with that Q_m. Encoding the SAME u with Q_m=1 (identity
#   interleave) gives the post-bit-selection sequence e; with Q_m=8 gives the
#   interleaved f. Bit-selection (§5.4.2.1) does NOT depend on Q_m, so the only
#   difference between the two outputs is the interleaver. Hence:
#       our_interleave(e, Qm=8)  must equal  f          (forward)
#       our_deinterleave(f, Qm=8) must equal e          (inverse)
#   If Sionna's convention differs from ours, this prints False and tells us
#   exactly that — self-correcting, zero guessing.
#
# CHECK 2  make_golden(scale, snr_db=None)  [Python now, NPU to finish]
#   Build a full-TB descramble-layout LLR array from per-CB Sionna encodings,
#   run rate_dematch_inverse, dump int16 Q8.8 lam_in for the LOCKED ldpc_decode.
#   Sweep `scale`; the locked decoder's BER picks it.
#
# VERIFIED API (Sionna 1.2.x docs, nvlabs.github.io/sionna):
#   from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
#   LDPC5GEncoder(k, n, num_bits_per_symbol=None, bg=None, ...) ; c = encoder(u)
#   u:[...,k] float {0,1}  ->  c:[...,n] float {0,1}
#   >>> CONFIRM exact kwargs with inspect.signature(LDPC5GEncoder.__init__)
#       on your machine before trusting; if num_bits_per_symbol=1 is rejected,
#       use the smallest Q_m Sionna accepts and adjust the check accordingly.
# ============================================================================
import numpy as np
from rate_dematch_ref import (
    Q_M, C_NUM, LDPC_K, LDPC_N, N_2Z, N_CB_BUF, N_SYM, N_SYM_PAD,
    N_STREAMS, N_SLOT, N_LAYERS, G, QAM_Q_SCALE, QAM_CLIP,
    LDPC_Q_SCALE, LDPC_CLIP, compute_Er, build_gather_lut, rate_dematch_inverse,
)

BG = "bg1"  # Sionna expects the string "bg1"/"bg2", not an int


# ---- §5.4.2.2 helpers, convention B (row-column, Qm rows) -------------------
def interleave_5422(e, Qm):
    E = e.shape[-1]; EQ = E // Qm
    f = np.empty_like(e); i = np.arange(EQ)
    for j in range(Qm):
        f[..., i * Qm + j] = e[..., i + j * EQ]
    return f

def deinterleave_5422(f, Qm):
    E = f.shape[-1]; EQ = E // Qm
    n = np.arange(E); src = (n % EQ) * Qm + (n // EQ)
    return f[..., src]


def _encoder(k, n, qm):
    from sionna.phy.fec.ldpc import LDPC5GEncoder
    return LDPC5GEncoder(k, n, num_bits_per_symbol=qm, bg=BG)


def check_interleaver():
    print("--- CHECK 1: §5.4.2 interleave/selection vs Sionna (black-box) ---")
    k = LDPC_K
    n = int(compute_Er(G, C_NUM, Q_M, N_LAYERS)[0])  # current E_low=24640
    EQ = n // Q_M
    rng = np.random.default_rng(0)
    B = 64                            # batch: 64-bit column signatures => unique
    u = rng.integers(0, 2, size=(B, k)).astype(np.float32)
    e = np.asarray(_encoder(k, n, 1)(u)).astype(np.int64)   # Qm=1 -> e
    f = np.asarray(_encoder(k, n, 8)(u)).astype(np.int64)   # Qm=8 -> f

    # (0) sanity: is f a permutation of e? (same per-row popcount)
    same_mult = np.array_equal(e.sum(1), f.sum(1))
    print(f"  popcount(e)==popcount(f) per row (f is a perm of e): {same_mult}")
    if not same_mult:
        print("  -> Qm changes MORE than interleaving (bit-selection differs). "
              "num_bits_per_symbol semantics not as assumed; stop and inspect.")
        return False

    # (1) test the two row-column conventions directly
    A = e.copy(); A_f = np.empty_like(e); i = np.arange(EQ)
    for j in range(Q_M): A_f[:, i + j*EQ] = e[:, i*Q_M + j]     # convention A
    B_f = interleave_5422(e, Q_M)                               # convention B (ours)
    okA = np.array_equal(A_f, f)
    okB = np.array_equal(B_f, f)
    print(f"  convention A (f[i+j*EQ]=e[i*Qm+j]) matches Sionna : {okA}")
    print(f"  convention B (f[i*Qm+j]=e[i+j*EQ]) matches Sionna : {okB}  <- ours")
    if okB:
        # confirm our inverse too
        okinv = np.array_equal(deinterleave_5422(f, Q_M), e)
        print(f"  our deinterleave(f)==e                            : {okinv}")
        return okinv

    # (2) neither simple convention matched -> recover the exact permutation P
    #     where f[:,a] = e[:,P[a]] via 64-bit column signatures, then print form.
    w = (1 << np.arange(B, dtype=np.uint64))
    sig_e = (e.astype(np.uint64) * w).sum(0)
    sig_f = (f.astype(np.uint64) * w).sum(0)
    order = np.argsort(sig_e, kind="stable")
    pos = np.searchsorted(sig_e[order], sig_f)
    if np.any(sig_e[order][pos] != sig_f):
        print("  -> column signatures not 1:1 (collisions/not a clean perm); "
              "increase B or inspect e/f directly."); return False
    P = order[pos]                      # f[:,a] = e[:,P[a]]
    a = np.arange(n)
    matchB = np.array_equal(P, (a // Q_M) + (a % Q_M) * EQ)  # B as gather f[a]=e[P[a]]
    print(f"  recovered permutation P (f[a]=e[P[a]]); matches B-form: {matchB}")
    print(f"  P[:16]={P[:16]}")
    print("  -> paste P[:16] back; I'll express it as a closed form and fix the LUT.")
    return False


# ---- CHECK 2: golden generator ---------------------------------------------
def make_golden(scale, snr_db=None, out_dir="data/golden/rx/rate_dematch", seed=1):
    """Encode 143 CBs with Sionna, lay f into descramble layout, run our inverse,
    dump int16 Q8.8 lam_in for the locked ldpc_decode. snr_db=None => clean LLRs
    (structural check: locked LDPC should give BER 0). Set snr_db for realistic
    scale pinning (adds the 256QAM+AWGN+Max-Log path — TODO mark below)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    E = compute_Er(G, C_NUM, Q_M, N_LAYERS)
    rng = np.random.default_rng(seed)

    # f_stream = concat of per-CB interleaved codewords (CB concat order, §5.5)
    f_stream = np.empty(G, dtype=np.int8)
    info_bits = np.zeros((C_NUM, LDPC_K), dtype=np.int8)   # systematic golden
    off = 0
    for cb in range(C_NUM):
        Ecb = int(E[cb])
        enc = _encoder(LDPC_K, Ecb, Q_M)
        u = rng.integers(0, 2, size=(1, LDPC_K)).astype(np.float32)
        c = np.asarray(enc(u))[0].astype(np.int8)          # [Ecb] = f for this CB
        info_bits[cb] = u[0].astype(np.int8)
        f_stream[off:off + Ecb] = c
        off += Ecb

    # bit -> clean LLR in Q11.5 units (descramble output convention: +conf=bit0)
    #   LLR sign: L = log P(0)/P(1); bit 0 -> +, bit 1 -> -
    if snr_db is None:
        llr_q115 = np.where(f_stream == 0, QAM_CLIP, -QAM_CLIP).astype(np.int16)
    else:
        raise NotImplementedError(
            "Noisy path: map f -> 256QAM symbols (38.211), add AWGN at snr_db, "
            "run the qam256_demod Max-Log + Q11.5 quantization to get realistic "
            "LLRs, then continue. Mirror qam256_demod.h scaling exactly.")

    # scatter into descramble layout [N_SLOT, N_STREAMS, N_SYM_PAD]
    descram = np.zeros((N_SLOT, N_STREAMS, N_SYM_PAD), dtype=np.int16)
    k = np.arange(G); ge = k // Q_M; b = k % Q_M
    slot = ge // N_SYM; re = ge % N_SYM
    descram[slot, b, re] = llr_q115

    _, lam_q8_8, _ = rate_dematch_inverse(descram, scale=scale)  # int16 [143,26112]

    descram.tofile(f"{out_dir}/descram_in.bin")
    lam_q8_8.tofile(f"{out_dir}/lam_in.bin")
    info_bits.tofile(f"{out_dir}/info_bits.bin")
    nz = int((lam_q8_8 != 0).sum())
    print(f"--- CHECK 2: golden written (scale={scale}, snr_db={snr_db}) ---")
    print(f"  lam_in.bin : int16 [{C_NUM},{LDPC_N}]  nonzero={nz} (expect ~{G})")
    print(f"  -> run locked ldpc_decode on lam_in.bin; compare decoded_bits to "
          f"info_bits.bin. clean LLR must give BER 0. Then sweep scale w/ snr_db.")
    return lam_q8_8, info_bits


def check_lam_bitexact(seed=1):
    """End-to-end bit-exact validation — NO NPU / NO BER / NO CRC.
    Sionna enc8 -> f (interleaved, transmitted); enc1 -> e (circular-buffer order,
    ground truth). Push f through descramble layout + rate_dematch_inverse, then
    assert: at lam[cb, 2Z:2Z+E] the recovered bit (sign) == e_cb, and lam is exactly
    0 in the 2Z prefix and the sub-buffer-puncture tail. Scale-agnostic."""
    print("--- CHECK 2: lam_in bit-exact vs Sionna (no NPU/BER/CRC) ---")
    E = compute_Er(G, C_NUM, Q_M, N_LAYERS)
    rng = np.random.default_rng(seed)
    f_stream = np.empty(G, dtype=np.int8)
    e_all = []
    off = 0
    for cb in range(C_NUM):
        Ecb = int(E[cb])
        u = rng.integers(0, 2, size=(1, LDPC_K)).astype(np.float32)
        f = np.asarray(_encoder(LDPC_K, Ecb, Q_M)(u))[0].astype(np.int8)  # interleaved
        e = np.asarray(_encoder(LDPC_K, Ecb, 1)(u))[0].astype(np.int8)    # circ. buffer
        f_stream[off:off + Ecb] = f
        e_all.append(e)
        off += Ecb
    # descramble layout, clean LLR convention: bit0 -> +, bit1 -> -
    llr = np.where(f_stream == 0, QAM_CLIP, -QAM_CLIP).astype(np.int16)
    descram = np.zeros((N_SLOT, N_STREAMS, N_SYM_PAD), dtype=np.int16)
    k = np.arange(G); ge = k // Q_M; b = k % Q_M
    slot = ge // N_SYM; re = ge % N_SYM
    descram[slot, b, re] = llr
    _, lam, _ = rate_dematch_inverse(descram, scale=1.0)   # int16 [143, 26112]
    bad_sign = bad_zero = 0
    for cb in range(C_NUM):
        Ecb = int(E[cb]); e = e_all[cb]
        recv = lam[cb, N_2Z:N_2Z + Ecb]
        bad_sign += int(((recv > 0) != (e == 0)).sum())           # sign must == bit
        if lam[cb, :N_2Z].any() or lam[cb, N_2Z + Ecb:].any():
            bad_zero += 1
    ok = (bad_sign == 0 and bad_zero == 0)
    print(f"  received-position sign mismatches : {bad_sign}")
    print(f"  CBs with nonzero punctured region : {bad_zero}")
    print(f"  lam_in bit-exact vs Sionna        : {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok1 = check_interleaver()
    print(f"interleaver spec-compliance: {'PASS' if ok1 else 'FAIL'}\n")
    if ok1:
        ok2 = check_lam_bitexact()
        print(f"\nrate_dematch end-to-end: {'PASS' if ok2 else 'FAIL'}")
