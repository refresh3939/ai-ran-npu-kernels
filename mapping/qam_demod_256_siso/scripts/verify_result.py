#!/usr/bin/env python3
# ============================================================================
# verify_result.py — 256-QAM demod kernel output checker
#
# Compares kernel-produced LLR against:
#   1. output_llr.bin         (int16 pipeline mirror) — MUST be bit-exact
#   2. output_llr_sionna.bin  (Sionna float ref)      — informational
#
# Kernel writes 6-stream layout (... wait, 256-QAM has 8 bits per symbol so
# it's 8-stream): out[b * N_SYM_PAD + i] = LLR for bit_b of sym_i.
# We transpose to NR interleave order [i*Q_M + b] before comparing against
# the golden files.
#
# Usage:
#   python3 verify_result.py [--data-root /path/to/data] [--kernel-out PATH]
#
# Defaults:
#   --data-root          : AIRAN_DATA_DIR env, else ../../../
#   --kernel-out         : <data-root>/data/ascend_output/rx/qam256_demod/output_llr.bin
#                          (8-stream layout from kernel; tool transposes)
# ============================================================================
import os, sys, argparse
import numpy as np

N_SYM         = 10960
Q_M           = 8
BLOCK_DIM     = 4
SYM_PER_AIV   = 2816
N_SYM_PAD     = BLOCK_DIM * SYM_PER_AIV  # 11264
Q_SCALE       = 32
LLR_CLIP_FX   = 2560

def transpose_8stream_to_nr(kernel_8s):
    """Kernel 8-stream layout [b, i] (b in NR bit order across streams 0..7)
       -> NR bit interleave [i, b]."""
    # kernel_8s shape: (Q_M, N_SYM_PAD)
    return kernel_8s[:, :N_SYM].T.copy()   # (N_SYM, Q_M)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("AIRAN_DATA_DIR", "../../.."))
    ap.add_argument("--kernel-out", default=None,
                    help="Path to kernel-produced output_llr.bin (8-stream layout)")
    ap.add_argument("--golden-dir", default=None)
    ap.add_argument("--tol-bit-exact", type=int, default=0,
                    help="Max allowed LSB diff vs int16 pipeline ref (default 0)")
    ap.add_argument("--tol-sionna-warn", type=int, default=30,
                    help="LSB threshold above which Sionna mismatch is reported as warning")
    args = ap.parse_args()

    golden_dir = args.golden_dir or os.path.join(args.data_root, "data/golden/rx/qam256_demod")
    kernel_out = args.kernel_out or os.path.join(
        args.data_root, "data/ascend_output/rx/qam256_demod/output_llr.bin")

    p_pipeline = os.path.join(golden_dir, "output_llr.bin")
    p_sionna   = os.path.join(golden_dir, "output_llr_sionna.bin")

    print("="*70)
    print("256-QAM Demap verify_result")
    print("="*70)
    print(f"  kernel output: {kernel_out}")
    print(f"  golden dir:    {golden_dir}")
    print(f"  N_SYM={N_SYM}  Q_M={Q_M}  Q_SCALE={Q_SCALE}  LLR_CLIP={LLR_CLIP_FX}")
    print()

    # ---- Load golden files ----
    for f in (p_pipeline, p_sionna):
        if not os.path.isfile(f):
            print(f"[ERROR] missing golden file: {f}")
            print(f"        run qam256_demod_ref.py first")
            return 1
    gold_pipeline = np.fromfile(p_pipeline, dtype=np.int16)
    gold_sionna   = np.fromfile(p_sionna,   dtype=np.int16)
    expect_size = N_SYM * Q_M
    if gold_pipeline.size != expect_size or gold_sionna.size != expect_size:
        print(f"[ERROR] golden size mismatch: pipeline={gold_pipeline.size} "
              f"sionna={gold_sionna.size} expected={expect_size}")
        return 1
    gold_pipeline = gold_pipeline.reshape(N_SYM, Q_M)
    gold_sionna   = gold_sionna  .reshape(N_SYM, Q_M)

    # ---- Load kernel output ----
    if not os.path.isfile(kernel_out):
        print(f"[ERROR] kernel output not found: {kernel_out}")
        print(f"        run the kernel first (bash run.sh -r npu -v Ascend310P1)")
        return 1
    kernel_raw = np.fromfile(kernel_out, dtype=np.int16)
    expect_kernel_size = Q_M * N_SYM_PAD
    if kernel_raw.size != expect_kernel_size:
        print(f"[ERROR] kernel output size {kernel_raw.size} != expected {expect_kernel_size} "
              f"(8 streams × {N_SYM_PAD} padded syms)")
        return 1
    kernel_8s = kernel_raw.reshape(Q_M, N_SYM_PAD)

    # Sentinel check (0xAA fill survivors mean kernel didn't write somewhere)
    sentinels = int((kernel_8s[:, :N_SYM].view(np.uint16) == 0xAAAA).sum())
    if sentinels:
        print(f"[WARN] {sentinels} sentinel 0xAAAA bytes survived in kernel output "
              f"(kernel left {sentinels} positions un-written)")

    kernel_nr = transpose_8stream_to_nr(kernel_8s)

    # ============================================================================
    # Check 1: bit-exact vs int16 pipeline
    # ============================================================================
    diff_p = np.abs(kernel_nr.astype(np.int32) - gold_pipeline.astype(np.int32))
    diff_count_p = int((diff_p > args.tol_bit_exact).sum())
    max_err_p    = int(diff_p.max())
    print("[Check 1/2] vs int16 pipeline reference (bit-exact target)")
    print(f"  diff:     {diff_count_p} / {diff_p.size}  (tolerance: {args.tol_bit_exact} LSB)")
    print(f"  max_err:  {max_err_p} LSB")
    if diff_count_p > 0:
        worst_idx = int(np.argmax(diff_p))
        sym, bit = divmod(worst_idx, Q_M)
        bit_name = ["I_b3","Q_b3","I_b2","Q_b2","I_b1","Q_b1","I_b0","Q_b0"][bit]
        print(f"  worst:    sym={sym}  bit={bit} ({bit_name})  "
              f"kernel={int(kernel_nr[sym,bit])}  gold={int(gold_pipeline[sym,bit])}")
        # Show full LLR vector for the worst symbol
        print(f"  sym {sym}  kernel: {kernel_nr[sym].tolist()}")
        print(f"  sym {sym}  gold:   {gold_pipeline[sym].tolist()}")
    if diff_count_p > 0:
        print("[Check 1/2] FAIL")
        rc = 1
    else:
        print("[Check 1/2] PASS (bit-exact)")
        rc = 0

    # ============================================================================
    # Check 2: informational vs Sionna float reference
    # ============================================================================
    diff_s = np.abs(kernel_nr.astype(np.int32) - gold_sionna.astype(np.int32))
    print()
    print("[Check 2/2] vs Sionna float reference (informational)")
    print(f"  max_err:  {int(diff_s.max())} LSB  ({int(diff_s.max())/Q_SCALE:.3f} real)")
    print(f"  mean:     {diff_s.mean():.2f} LSB")
    pct_gt = 100.0 * (diff_s > args.tol_sionna_warn).sum() / diff_s.size
    print(f"  >{args.tol_sionna_warn} LSB:  {pct_gt:.2f}%  (Q11.5 quantization noise, normal)")

    # ---- Per-bit breakdown (helps diagnose if one bit is way off) ----
    print()
    print("  per-bit max_err (pipeline / sionna):")
    bit_names = ["I_b3","Q_b3","I_b2","Q_b2","I_b1","Q_b1","I_b0","Q_b0"]
    for b in range(Q_M):
        mp = int(diff_p[:, b].max())
        ms = int(diff_s[:, b].max())
        marker = "  FAIL" if mp > args.tol_bit_exact else ""
        print(f"    {bit_names[b]:6s}: pipeline={mp:4d}  sionna={ms:4d}{marker}")

    print()
    print("="*70)
    print("VERIFY OVERALL:", "PASS" if rc == 0 else "FAIL")
    print("="*70)
    return rc

if __name__ == "__main__":
    sys.exit(main())
