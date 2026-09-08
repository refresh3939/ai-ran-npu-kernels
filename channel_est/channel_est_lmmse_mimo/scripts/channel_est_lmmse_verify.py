#!/usr/bin/env python3
"""Chunked verification for configured channel-estimation output."""

from __future__ import annotations

import os
import argparse
import sys
import time
from pathlib import Path

import numpy as np

NR = int(os.environ.get("NR", "64"))
NL = int(os.environ.get("NL", "2"))
RANK = int(os.environ.get("RANK", "96"))
N_SYMBOL, N_SC_PAD = 14, 1664
N_ELEMS = NR * 16 * N_SYMBOL * N_SC_PAD
CASE = f"case_0_m{NR}_k{NL}_r{RANK}"


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def checked_memmap(path: Path, shape: tuple[int, ...] = (N_ELEMS,)) -> np.memmap:
    if not path.is_file():
        fail(f"missing {path}")
    expected = N_ELEMS * np.dtype(np.float16).itemsize
    if path.stat().st_size != expected:
        fail(f"wrong size for {path}: {path.stat().st_size}, expected {expected}")
    return np.memmap(path, mode="r", dtype=np.float16, shape=shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube-time-fused", action="store_true")
    parser.add_argument("--cube-time-post", action="store_true")
    args = parser.parse_args()
    if args.cube_time_fused and args.cube_time_post:
        fail("Cube time variants are mutually exclusive")
    root = Path(os.environ.get("AIRAN_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    gold_dir = root / "golden" / CASE
    out_dir = root / "ascend_output" / CASE
    stale_minutes = float(os.environ.get("STALE_MIN", "10"))
    if args.cube_time_post:
        outputs = [out_dir / "h_cube_time_post_re.bin", out_dir / "h_cube_time_post_im.bin"]
    elif args.cube_time_fused:
        outputs = [out_dir / "h_cube_time_fused_re.bin", out_dir / "h_cube_time_fused_im.bin"]
    else:
        outputs = [out_dir / "h_re.bin", out_dir / "h_im.bin"]
    for path in outputs:
        if not path.exists():
            fail(f"missing output {path}; run bash run.sh first")
    age = (time.time() - max(path.stat().st_mtime for path in outputs)) / 60.0
    if age > stale_minutes:
        fail(f"output is {age:.1f} minutes old (limit {stale_minutes:.1f}); refuse stale validation")

    expected_counts = {
        1: [798],
        2: [798, 798],
        3: [399, 399, 798],
        4: [399, 399, 399, 399],
    }[NL]
    count_path = gold_dir / "pilot_count.bin"
    if count_path.stat().st_size != 32:
        fail(f"wrong pilot_count backing size: {count_path.stat().st_size}, expected 32")
    counts = np.fromfile(count_path, dtype=np.uint16, count=2 * NL).reshape(NL, 2)
    count_ok = counts[:, 0].tolist() == expected_counts and bool(np.all(counts[:, 0] == counts[:, 1]))
    b_tail_zero = True
    expected_b_elems = NL * RANK * 832
    for component in ("factor_b_re.bin", "factor_b_im.bin"):
        path = gold_dir / component
        if path.stat().st_size != expected_b_elems * np.dtype(np.float16).itemsize:
            fail(f"wrong size for {path}")
        packed = np.memmap(path, mode="r", dtype=np.float16,
                           shape=(NL, RANK // 16, 832 // 16, 16, 16))
        rows = packed.transpose(0, 1, 3, 2, 4).reshape(NL, RANK, 832)
        for layer, count in enumerate(expected_counts):
            b_tail_zero &= bool(np.all(rows[layer, :, count:] == 0))

    gr = checked_memmap(gold_dir / "gold_h_re.bin")
    gi = checked_memmap(gold_dir / "gold_h_im.bin")
    ar = checked_memmap(outputs[0])
    ai = checked_memmap(outputs[1])
    maximum = 0.0
    squared_error = 0.0
    squared_gold = 0.0
    finite = True
    chunk = 1 << 20
    for start in range(0, N_ELEMS, chunk):
        stop = min(start + chunk, N_ELEMS)
        actual = ar[start:stop].astype(np.float32) + 1j * ai[start:stop].astype(np.float32)
        golden = gr[start:stop].astype(np.float32) + 1j * gi[start:stop].astype(np.float32)
        finite &= bool(np.isfinite(actual).all())
        error = np.abs(actual - golden)
        maximum = max(maximum, float(error.max(initial=0.0)))
        squared_error += float(np.vdot(error, error).real)
        squared_gold += float(np.vdot(golden, golden).real)
    nrmse = np.sqrt(squared_error / max(squared_gold, 1e-30))
    passed = finite and maximum <= 0.1 and nrmse <= 0.03 and count_ok and b_tail_zero
    if args.cube_time_post:
        suffix = " (cube_time_post_gemm)"
    elif args.cube_time_fused:
        suffix = " (cube_time_fused)"
    else:
        suffix = ""
    inactive = ar.reshape(NR, 16, N_SYMBOL, N_SC_PAD)[:, NL:]
    inactive_zero = bool(np.all(inactive == 0))
    passed &= inactive_zero
    print(f"[shape] [{NR},16,{N_SYMBOL},{N_SC_PAD}] fp16 split-complex{suffix}; active-L={NL}")
    print(f"[padding] inactive layers zero={inactive_zero}")
    print(f"[observation] count={counts[:, 0].tolist()} expected={expected_counts}; "
          f"B tail zero={b_tail_zero}")
    print(f"[check] finite={finite} max_abs={maximum:.4e} (<=0.1) nrmse={nrmse:.4e} (<=0.03)")
    print(f"=== {'PASS' if passed else 'FAIL'} ===")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
