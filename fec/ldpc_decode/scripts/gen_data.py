#!/usr/bin/env python3
# ============================================================================
# gen_data.py — Generate LDPC test datasets via sionna
#
# Single-purpose script: sionna LDPC encode + AWGN + ref minsum decode,
# emits one dataset per SNR into kernels/rx/ldpc_decode/data/snr_Xdb/.
#
# Requires the sionna conda env (tensorflow + sionna).
#
# Usage:
#   python gen_data.py                         # default SNRs: 3, 5, 10 dB
#   python gen_data.py --snr-list 3 5 10 15    # custom SNRs
#
# Output per SNR (kernels/rx/ldpc_decode/data/snr_<X>db/):
#   lam_in.bin          int16 (143, 26112)  kernel input (negated quantized LLR)
#   info_bits.bin       uint8 (143, 8448)   truth (8424 random info + 24-bit CRC-24B tail)
#   ref_decoded.bin     uint8 (143, 8448)   sionna minsum @ MAX_ITER=20 (ceiling)
#   codeword.bin        uint8 (143, 25344)  sionna's c (pre-channel)
#   llr_in.bin          float32 (143, 25344) channel LLR (pre-quantize)
#   sigma2.bin          float32 (1,)        channel noise variance
#   degrees.bin         int16 (46,)         shift-table-derived
#   edge_offsets.bin    int32 (47,)         shift-table-derived
#   decoded_bits.bin    int8  (143, 8448)   = ref_decoded (for main.cpp [verify/bits])
#   crc_ok.bin, iter_count.bin, lam_out.bin, prev_msg_out.bin (zeros, kernel fills)
#
# CRC-24B tail (5G NR TS 38.212 §5.1):
#   info_bits[cb, 0:8424]   = random info
#   info_bits[cb, 8424:8448] = CRC-24B of the random info, MSB-first
#   This lets [verify/crc] in main.cpp end-to-end test the kernel's CRC pipeline:
#   bit-perfect decode → kernel-computed CRC matches tail → crc_ok=1 per CB.
#
# Data layout (lam_in.bin):
#   [   0 :  768]  = 0           (2Z punctured zeros)
#   [ 768 :26112]  = -quantize(channel_LLR)
#   sign: lam > 0 ↔ bit = 0,  lam < 0 ↔ bit = 1
# ============================================================================

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (must match airan:: in ldpc_decode.h)
# ---------------------------------------------------------------------------
LDPC_Z       = 384
LDPC_KB      = 22
LDPC_MB      = 46
LDPC_NFULL   = LDPC_KB + LDPC_MB           # 68
LDPC_K       = LDPC_KB * LDPC_Z            # 8448
LDPC_N_RAW   = 66 * LDPC_Z                 # 25344
LDPC_C_NUM   = 143
LAM_ELEMS_PER_CB = LDPC_NFULL * LDPC_Z     # 26112

Q_SCALE      = 256
LLR_CLIP_FX  = 20 * Q_SCALE                # ±5120

# 5G NR TS 38.212 §5.1: CRC-24B for TB / CB attachment
# Poly D^24 + D^23 + D^6 + D^5 + D + 1 = 0x800063
# Used as CB-CRC tail (last 24 bits of each K=8448 info block).
CRC24B_POLY      = 0x800063
CRC_LEN          = 24
LDPC_K_NO_CRC    = LDPC_K - CRC_LEN        # 8424 actual info bits per CB

SNR_LIST_DEFAULT = [3, 5, 10]              # dB
SEED             = 42
REF_MAX_ITER     = 20                      # sionna ref decode iter count


# ---------------------------------------------------------------------------
# CRC-24B (bit-exact equivalent to kernel's slice-by-8 in ldpc_decode_kernel.cpp)
# ---------------------------------------------------------------------------
def crc24b_per_cb(info_bits):
    """
    Append CRC-24B (5G NR TS 38.212 §5.1) to each CB.

    Input:  info_bits  uint8 (N_CB, LDPC_K_NO_CRC=8424)   pure info bits
    Output: u_full     uint8 (N_CB, LDPC_K=8448)          info + 24-bit CRC tail
    Output: crc_int    int32 (N_CB,)                      per-CB CRC remainder

    The CRC is bit-serial over info bits in array order (bit 0 first),
    flushed with 24 zero bits, producing a 24-bit remainder. The remainder
    is then packed MSB-first into bits[8424..8447] so that the kernel's
    tail_int = bits[8424]<<23 | ... | bits[8447]<<0 equals the remainder.

    Bit-exact equivalence with kernel's slice-by-8 CRC is verified by
    scripts/verify_slice_by_8.py.
    """
    n_cb, k_in = info_bits.shape
    assert k_in == LDPC_K_NO_CRC, f"info_bits shape {info_bits.shape} != (*, {LDPC_K_NO_CRC})"

    crc_out = np.zeros(n_cb, dtype=np.uint32)
    # Vectorize across CBs but bit-serial within each CB (8424 + 24 iterations).
    crc_reg = np.zeros(n_cb, dtype=np.uint32)
    for i in range(k_in):
        top = ((crc_reg >> 23) ^ info_bits[:, i].astype(np.uint32)) & 1
        crc_reg = (crc_reg << 1) & 0xFFFFFF
        crc_reg ^= top * CRC24B_POLY
    # Flush 24 zero bits
    for _ in range(CRC_LEN):
        top = (crc_reg >> 23) & 1
        crc_reg = (crc_reg << 1) & 0xFFFFFF
        crc_reg ^= top * CRC24B_POLY
    crc_out[:] = crc_reg

    # Pack 24-bit CRC into uint8 bits MSB-first: bit 23 → first slot, bit 0 → last
    crc_bits = np.zeros((n_cb, CRC_LEN), dtype=np.uint8)
    for i in range(CRC_LEN):
        crc_bits[:, i] = ((crc_out >> (CRC_LEN - 1 - i)) & 1).astype(np.uint8)

    u_full = np.concatenate([info_bits, crc_bits], axis=1)
    assert u_full.shape == (n_cb, LDPC_K)
    return u_full, crc_out


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def data_root():
    """AI-RAN-NPU project root (for shift table, s56_ref/, etc)."""
    env = os.environ.get("AIRAN_DATA_DIR")
    if env:
        return Path(env)
    # scripts/ → ldpc_decode/ → rx/ → kernels/ → AI-RAN-NPU/
    try:
        return Path(__file__).resolve().parents[4]
    except IndexError:
        return Path(__file__).resolve().parent


def kernel_dir():
    """The kernel directory (ldpc_decode/) — where local data/ lives."""
    # scripts/gen_data.py → scripts/ → ldpc_decode/
    return Path(__file__).resolve().parents[1]


def shift_table_path():
    """Use the decoder-local weight layout, matching tx/ldpc_encode."""
    return kernel_dir() / "data/weights/ldpc_bg1_z384_shifts/shift_table.bin"


def snr_dir(snr_db):
    """Per-SNR dataset output directory (kernel-local)."""
    return kernel_dir() / f"data/snr_{snr_db}db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quantize_llr_vec(llr_f32):
    """Sionna LLR (float32) → int16 Q8.8 lam (negated, clipped)."""
    q = np.rint(llr_f32 * Q_SCALE).astype(np.int32)
    q = np.clip(q, -LLR_CLIP_FX, LLR_CLIP_FX)
    return (-q).astype(np.int16)


def build_lam_in(llr_channel):
    """sionna llr (N_CB, N_RAW=25344) → kernel lam_in (N_CB, 26112) int16."""
    n_cb = llr_channel.shape[0]
    lam = np.zeros((n_cb, LAM_ELEMS_PER_CB), dtype=np.int16)
    lam[:, 2 * LDPC_Z : 2 * LDPC_Z + LDPC_N_RAW] = quantize_llr_vec(llr_channel)
    return lam


def build_padded_tables():
    sh = np.fromfile(shift_table_path(), dtype=np.int16).reshape(LDPC_MB, LDPC_NFULL)
    deg = np.array([int(np.sum(sh[br] != -1)) for br in range(LDPC_MB)], dtype=np.int16)
    eoff = np.zeros(LDPC_MB + 1, dtype=np.int32)
    eoff[1:] = np.cumsum(deg)
    return sh, deg, eoff


# ---------------------------------------------------------------------------
# Sionna pipeline
# ---------------------------------------------------------------------------
def run_sionna_pipeline(ebn0_db, LDPC5GEncoder, LDPC5GDecoder, tf, seed=SEED):
    """
    One SNR pass: random info → encode → BPSK+AWGN → LLR → ref minsum decode.

    Returns dict with: info_bits, codeword, llr, ref_minsum, sigma2.
    """
    rng = np.random.default_rng(seed)

    # 1. Random info bits (8424 per CB) + CRC-24B tail (24 per CB) → K=8448 total.
    #    Without the CRC tail, kernel-computed CRC has nothing to match against
    #    and [verify/crc] always reports 0 pass (false failure). With it, a
    #    bit-perfect decode yields crc_ok=1, end-to-end testing the CRC pipeline.
    u_info_only = rng.integers(0, 2, size=(LDPC_C_NUM, LDPC_K_NO_CRC), dtype=np.uint8)
    u, crc_per_cb = crc24b_per_cb(u_info_only)
    print(f"[crc24b] appended CB-CRC: K_info={LDPC_K_NO_CRC} + CRC={CRC_LEN} = K={LDPC_K}; "
          f"CRC[cb=0]=0x{int(crc_per_cb[0]):06x}")

    # 2. LDPC encode
    enc = LDPC5GEncoder(k=LDPC_K, n=LDPC_N_RAW)
    c = enc(tf.convert_to_tensor(u.astype(np.float32))).numpy().astype(np.uint8)
    assert c.shape == (LDPC_C_NUM, LDPC_N_RAW), f"encode shape {c.shape}"

    # 3. BPSK + AWGN → channel LLR
    ebn0 = 10 ** (ebn0_db / 10.0)
    rate = LDPC_K / LDPC_N_RAW
    sigma2 = 1.0 / (2.0 * rate * ebn0)
    sigma = np.sqrt(sigma2)
    x = 1.0 - 2.0 * c.astype(np.float32)         # BPSK: bit 0 → +1, bit 1 → −1
    noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    y = x + noise
    llr = (-2.0 / sigma2) * y                    # Sionna LLR = log P(b=1)/P(b=0)
    corr_sign = (np.sign(llr) == np.where(c == 1, +1.0, -1.0)).mean()
    print(f"[ch] Eb/N0={ebn0_db} dB  sigma2={sigma2:.4f}  "
          f"|llr|mean={np.abs(llr).mean():.3f}  correct-sign frac={corr_sign:.3f}")

    # 4. Sionna minsum decode (ceiling reference, MAX_ITER=20)
    #
    # CRITICAL: prune_pcm=False — at (k=8448, n=25344, rate=1/3), the default
    # prune_pcm=True has a reshape bug where pruning interacts with the 2Z
    # puncture and produces a zero-dim tf.reshape, crashing decode.
    # (Confirmed in ldpc_decode_ref.py — same workaround.)
    #
    # Sionna versions disagree on kwarg names — try several pairs.
    dec = None
    last_err = None
    for cn_kw, info_kw in [
        ("cn_update",      "return_infobits"),    # sionna 1.2.x
        ("cn_type",        "return_infobits"),
        ("cn_update_type", "return_infobits"),
        ("cn_update",      "return_info_bits"),
        ("cn_type",        "return_info_bits"),
    ]:
        try:
            dec = LDPC5GDecoder(
                enc,
                num_iter=REF_MAX_ITER,
                hard_out=True,
                prune_pcm=False,
                cn_schedule="flooding",
                **{cn_kw: "minsum", info_kw: True},
            )
            print(f"[dec] LDPC5GDecoder built: {cn_kw}=minsum, {info_kw}=True, "
                  f"prune_pcm=False, cn_schedule=flooding, num_iter={REF_MAX_ITER}")
            break
        except TypeError as e:
            last_err = e
            continue
    if dec is None:
        # Last resort: try without cn_schedule (very old sionna)
        for cn_kw, info_kw in [
            ("cn_update", "return_infobits"),
            ("cn_type",   "return_infobits"),
        ]:
            try:
                dec = LDPC5GDecoder(enc, num_iter=REF_MAX_ITER, hard_out=True,
                                    prune_pcm=False, **{cn_kw: "minsum", info_kw: True})
                print(f"[dec] LDPC5GDecoder built (no cn_schedule): {cn_kw}=minsum")
                break
            except TypeError as e:
                last_err = e
                continue
    if dec is None:
        raise RuntimeError(f"Could not construct LDPC5GDecoder: {last_err}")

    # sionna 1.2.2 quirk: also pass num_iter at call site
    try:
        u_hat = dec(tf.convert_to_tensor(llr), num_iter=REF_MAX_ITER).numpy().astype(np.uint8)
    except TypeError:
        u_hat = dec(tf.convert_to_tensor(llr)).numpy().astype(np.uint8)
    ber  = (u_hat != u).mean()
    bler = (u_hat != u).any(axis=1).mean()
    print(f"[dec] sionna minsum @ MAX_ITER={REF_MAX_ITER}: "
          f"BER={ber:.2e}  BLER={bler:.3f}  ({int(bler*LDPC_C_NUM)}/{LDPC_C_NUM} CB fail)")

    return {
        "info_bits":  u,
        "codeword":   c,
        "llr":        llr,
        "ref_minsum": u_hat,
        "sigma2":     sigma2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate LDPC test datasets via sionna")
    ap.add_argument("--snr-list", type=int, nargs="+", default=None,
                    help=f"SNRs in dB (default: {SNR_LIST_DEFAULT})")
    args = ap.parse_args()

    snr_list = args.snr_list or SNR_LIST_DEFAULT
    print(f"[gen-data] SNRs to generate: {snr_list} dB")
    print(f"[gen-data] N_CB={LDPC_C_NUM}  K={LDPC_K}  N_RAW={LDPC_N_RAW}  seed={SEED}")
    print(f"[gen-data] ref decoder: sionna minsum @ MAX_ITER={REF_MAX_ITER}")
    print()

    # Pre-flight: sionna available?
    try:
        import tensorflow as tf
        import sionna as _sn  # noqa: F401
        try:
            from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
        except ImportError:
            from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
    except ImportError as e:
        print(f"[err] sionna/tensorflow not available: {e}", file=sys.stderr)
        print(f"      Activate a conda env with sionna installed.", file=sys.stderr)
        sys.exit(1)

    # Pre-flight: shift table?
    if not shift_table_path().exists():
        print(f"[err] shift table not found: {shift_table_path()}", file=sys.stderr)
        sys.exit(1)

    # Shared metadata (same for all SNRs)
    sh, deg, eoff = build_padded_tables()
    total_edges = int(deg.sum())
    print(f"[shift_table] total_edges={total_edges}  max_deg={int(deg.max())}")
    print()

    for snr_db in snr_list:
        print(f"================================================================")
        print(f"  SNR = {snr_db} dB")
        print(f"================================================================")
        out = snr_dir(snr_db)
        out.mkdir(parents=True, exist_ok=True)

        data = run_sionna_pipeline(snr_db, LDPC5GEncoder, LDPC5GDecoder, tf)

        # Build kernel-format lam_in
        lam_in = build_lam_in(data["llr"])
        print(f"[layout] lam_in shape={lam_in.shape}  dtype={lam_in.dtype}  "
              f"sample lam[0,768:776]={lam_in[0, 768:776].tolist()}")

        # Write per-SNR files
        lam_in.tofile             (out / "lam_in.bin")
        data["info_bits"].tofile  (out / "info_bits.bin")
        data["codeword"].tofile   (out / "codeword.bin")
        data["llr"].tofile        (out / "llr_in.bin")
        data["ref_minsum"].tofile (out / "ref_decoded.bin")
        deg.tofile                (out / "degrees.bin")
        eoff.tofile               (out / "edge_offsets.bin")
        np.array([data["sigma2"]], dtype=np.float32).tofile(out / "sigma2.bin")

        # main.cpp also requires these (sized buffers); populate with neutral content.
        # crc_ok.bin: now that info_bits embeds a real CRC-24B tail per CB, every
        # bit-perfect decode should yield kernel crc_ok=1. We set gold = all 1
        # so [verify/crc] match_gold == kernel pass when LDPC is bit-perfect.
        np.ones (LDPC_C_NUM,                              dtype=np.uint8).tofile(out / "crc_ok.bin")
        np.zeros(LDPC_C_NUM,                              dtype=np.int32).tofile(out / "iter_count.bin")
        np.zeros((LDPC_C_NUM, LAM_ELEMS_PER_CB),          dtype=np.int16).tofile(out / "lam_out.bin")
        np.zeros((LDPC_C_NUM, total_edges, LDPC_Z),       dtype=np.int16).tofile(out / "prev_msg_out.bin")
        # decoded_bits.bin is what main.cpp compares against in [verify/bits].
        # Populate with ref_minsum so that comparison reads as kernel-vs-ref.
        data["ref_minsum"].astype(np.int8).tofile(out / "decoded_bits.bin")

        print(f"[save] wrote {out}/  ({len(list(out.iterdir()))} files)")
        print()

    print("[gen-data] DONE")
    print(f"  Outputs: {kernel_dir()}/data/snr_{{...}}db/")
    print(f"  Next step:")
    print(f"    bash run.sh -r npu -v Ascend310P1 -s 5")


if __name__ == "__main__":
    main()
