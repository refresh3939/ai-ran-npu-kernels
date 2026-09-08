#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

NR, D, PAD = int(os.environ.get("DMRS_LS_NR", "64")), 2, 832
if NR not in (16, 32, 64):
    raise ValueError("DMRS_LS_NR must be 16, 32, or 64")
CASES = ("case_rank1", "case_rank2_disjoint", "case_rank3_mixed",
         "case_rank4_occ")


def compare(actual_path: Path, expected_path: Path, dtype, atol: float = 0.0) -> bool:
    actual = np.fromfile(actual_path, dtype=dtype)
    expected = np.fromfile(expected_path, dtype=dtype)
    if actual.size != expected.size:
        print(f"  FAIL {actual_path.name}: size {actual.size} != {expected.size}")
        return False
    if np.issubdtype(np.dtype(dtype), np.floating):
        finite = bool(np.isfinite(actual).all())
        err = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
        ok = finite and err <= atol
        print(f"  {'PASS' if ok else 'FAIL'} {actual_path.name}: max_abs={err:.6g}")
        return ok
    ok = bool(np.array_equal(actual, expected))
    print(f"  {'PASS' if ok else 'FAIL'} {actual_path.name}: exact={ok}")
    return ok


def main() -> int:
    root = Path(os.environ.get("AIRAN_DATA_DIR",
                               Path(__file__).resolve().parents[1] / "data"))
    all_ok = True
    for name in CASES:
        gold, actual = root / "golden" / name, root / "ascend_output" / name
        cfg = json.loads((gold / "case.json").read_text())
        print(f"[verify] {name}: count={cfg['pilot_count']}")
        all_ok &= compare(actual / "h_ls_re.bin", gold / "h_ls_re.bin", np.float16, 0.004)
        all_ok &= compare(actual / "h_ls_im.bin", gold / "h_ls_im.bin", np.float16, 0.004)
        all_ok &= compare(actual / "pilot_sc.bin", gold / "pilot_sc.bin", np.uint16)
        all_ok &= compare(actual / "pilot_count.bin", gold / "pilot_count.bin", np.uint16)
        # Half reductions differ slightly in order; enforce useful relative agreement.
        got_noise = np.fromfile(actual / "noise_var_rx.bin", dtype=np.float16).astype(np.float32)
        exp_noise = np.fromfile(gold / "noise_var_rx.bin", dtype=np.float16).astype(np.float32)
        rel = float(np.max(np.abs(got_noise - exp_noise) / np.maximum(exp_noise, 1e-5)))
        noise_ok = bool(np.isfinite(got_noise).all() and rel <= 0.2)
        print(f"  {'PASS' if noise_ok else 'FAIL'} noise_var_rx.bin: max_rel={rel:.4f}")
        all_ok &= noise_ok
        canonical = (actual / "ce_natural_contract.txt").exists()
        print(f"  {'PASS' if canonical else 'FAIL'} canonical natural CE contract")
        all_ok &= canonical
        if cfg.get("legacy_ce_pack_compatible",
                   cfg["legacy_ce64x16_pack_compatible"]):
            for stem in ("ce_hls_re", "ce_hls_im", "ce_hls_neg_im"):
                all_ok &= compare(actual / f"{stem}.bin", gold / f"{stem}.bin", np.float16, 0.004)
        else:
            skipped = (actual / "legacy_ce_pack_skipped.txt").exists()
            print(f"  {'PASS' if skipped else 'FAIL'} legacy 798-only pack skipped")
            all_ok &= skipped
    print("[verify] PASS" if all_ok else "[verify] FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
