#!/usr/bin/env python3
# ============================================================================
# verify.py — Verify LDPC kernel decoded output against truth
#
# Single-purpose: read kernel_decoded_bits.bin + info_bits.bin, compute BER.
# Does NOT build kernel, does NOT run kernel, does NOT generate data.
# (Use bash run.sh to build/run; use gen_data.py to generate datasets.)
#
# Usage:
#   python verify.py --snr 5         # verify against kernel-local data/snr_5db/info_bits.bin
#   python verify.py --staged        # verify against whatever's currently in s56_ref/
#   python verify.py --stage 5       # copy data/snr_5db/* -> AIRAN/data/s56_ref/
#                                    # (use before 'bash run.sh' to switch SNR)
#
# Layout:
#   kernels/rx/ldpc_decode/data/snr_<X>db/     ← gen_data.py output (kernel-local)
#   AIRAN-NPU/data/s56_ref/                    ← main.cpp reads from here (project root)
#
# Typical 3-step workflow:
#   conda activate sionna
#   python scripts/gen_data.py                      # once: generate datasets
#
#   conda activate base
#   python scripts/verify.py --stage 5              # stage data for kernel
#   bash run.sh -r npu -v Ascend310P1               # kernel dumps kernel_decoded_bits.bin
#   python scripts/verify.py --snr 5                # see BER vs truth
# ============================================================================

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (must match airan:: in ldpc_decode.h)
# ---------------------------------------------------------------------------
LDPC_Z       = 384
LDPC_KB      = 22
LDPC_K       = LDPC_KB * LDPC_Z            # 8448
LDPC_C_NUM   = 143


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def data_root():
    """AI-RAN-NPU project root — where data/s56_ref/ lives (main.cpp reads from here)."""
    env = os.environ.get("AIRAN_DATA_DIR")
    if env:
        return Path(env)
    # scripts/ -> ldpc_decode/ -> rx/ -> kernels/ -> AI-RAN-NPU/
    try:
        return Path(__file__).resolve().parents[4]
    except IndexError:
        return Path(__file__).resolve().parent


def kernel_dir():
    """The kernel directory (ldpc_decode/) — where local data/ lives."""
    # scripts/verify.py -> scripts/ -> ldpc_decode/
    return Path(__file__).resolve().parents[1]


def s56_dir():
    """Where main.cpp reads from / writes kernel_decoded_bits.bin to.
       Lives under AIRAN_DATA_DIR (project root)."""
    return data_root() / "data/s56_ref"


def snr_dir(snr_db):
    """Per-SNR dataset cache (output of gen_data.py, kernel-local)."""
    return kernel_dir() / f"data/snr_{snr_db}db"


# ---------------------------------------------------------------------------
# Stage: copy snr_Xdb/* -> s56_ref/  (so main.cpp picks up the right SNR)
# ---------------------------------------------------------------------------
def cmd_stage(snr_db):
    src = snr_dir(snr_db)
    if not src.is_dir():
        print(f"[err] {src} does not exist.", file=sys.stderr)
        print(f"      Run 'python scripts/gen_data.py' first.", file=sys.stderr)
        sys.exit(1)

    dst = s56_dir()
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            n += 1
    print(f"[stage] {src} -> {dst}  ({n} files)")
    print(f"        Next: bash run.sh -r npu -v Ascend310P1")


# ---------------------------------------------------------------------------
# Verify: read info_bits + kernel_decoded_bits, compute BER
# ---------------------------------------------------------------------------
def do_verify(info_bits_path, ref_path, kbits_path, label):
    """Compute kernel-vs-truth BER, and ref-vs-truth ceiling if ref available."""
    print(f"\n{'='*70}")
    print(f"[verify] {label}")
    print(f"{'='*70}")
    print(f"  info_bits          = {info_bits_path}")
    print(f"  kernel_decoded     = {kbits_path}")
    print(f"  ref_decoded        = {ref_path if ref_path else '(skipped)'}")

    if not info_bits_path.exists():
        print(f"\n[err] info_bits.bin missing - gen_data.py hasn't produced this SNR yet")
        sys.exit(1)
    if not kbits_path.exists():
        print(f"\n[err] kernel_decoded_bits.bin missing - kernel hasn't run yet")
        print(f"      Run 'bash run.sh -r npu -v Ascend310P1' first.")
        sys.exit(1)

    info_bits = np.fromfile(info_bits_path, dtype=np.uint8).reshape(LDPC_C_NUM, LDPC_K)
    kbits     = np.fromfile(kbits_path,     dtype=np.int8 ).reshape(LDPC_C_NUM, LDPC_K).astype(np.uint8)

    # Kernel vs truth
    diff = (kbits != info_bits)
    ber  = diff.mean()
    bler = diff.any(axis=1).mean()
    n_fail = int(bler * LDPC_C_NUM)
    print(f"\n[kernel-vs-truth] kernel.decoded vs info_bits:")
    print(f"  BER  = {ber*100:.4f} %")
    print(f"  BLER = {bler*100:.2f} %    ({n_fail}/{LDPC_C_NUM} CB fail)")

    # Optional ref ceiling
    if ref_path and ref_path.exists():
        ref = np.fromfile(ref_path, dtype=np.uint8).reshape(LDPC_C_NUM, LDPC_K)
        ref_diff = (ref != info_bits)
        ref_ber  = ref_diff.mean()
        ref_bler = ref_diff.any(axis=1).mean()
        print(f"\n[ref-vs-truth]    sionna_minsum vs info_bits (ceiling):")
        print(f"  BER  = {ref_ber*100:.4f} %")
        print(f"  BLER = {ref_bler*100:.2f} %    ({int(ref_bler*LDPC_C_NUM)}/{LDPC_C_NUM} CB fail)")

        gap = ber - ref_ber
        if   gap >  1e-4: tag = f"+{gap*100:.4f} pp WORSE than sionna ref -> likely fp16 chaos (see SKILL)"
        elif gap < -1e-4: tag = f"{-gap*100:.4f} pp BETTER than sionna ref (surprising, recheck)"
        else:             tag = "matches sionna ref within +/- 0.01 pp"
        print(f"\n[gap] kernel - ref = {tag}")
    else:
        print(f"\n[ref-vs-truth]    skipped (no ref_decoded.bin available)")


def cmd_verify_snr(snr_db):
    d = snr_dir(snr_db)
    do_verify(
        info_bits_path = d / "info_bits.bin",
        ref_path       = d / "ref_decoded.bin",
        kbits_path     = s56_dir() / "kernel_decoded_bits.bin",
        label          = f"SNR = {snr_db} dB",
    )


def cmd_verify_staged():
    """Verify against whatever's currently in data/s56_ref/."""
    do_verify(
        info_bits_path = s56_dir() / "info_bits.bin",
        ref_path       = s56_dir() / "ref_decoded.bin",
        kbits_path     = s56_dir() / "kernel_decoded_bits.bin",
        label          = "staged (data/s56_ref/)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Verify LDPC kernel output against truth (gen_data.py output)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--snr",    type=int,
                   help="Verify against kernel-local data/snr_<X>db/info_bits.bin")
    g.add_argument("--staged", action="store_true",
                   help="Verify against whatever's currently in data/s56_ref/")
    g.add_argument("--stage",  type=int, metavar="SNR",
                   help="Copy snr_<X>db/* -> s56_ref/ (before running kernel)")
    args = ap.parse_args()

    if args.stage is not None:
        cmd_stage(args.stage)
    elif args.staged:
        cmd_verify_staged()
    else:
        cmd_verify_snr(args.snr)


if __name__ == "__main__":
    main()