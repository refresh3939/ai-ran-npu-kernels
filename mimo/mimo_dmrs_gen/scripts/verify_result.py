#!/usr/bin/env python3
"""verify_result.py — mimo_dmrs_gen kernel output vs golden ([NL,2,896] fp16)."""
import os, sys
import numpy as np
from pathlib import Path

CASES = ["case_0_baseline","case_1_nid0","case_2_slot7","case_3_nscid1","case_4_nid_large"]
ERR_THRESH = 1e-3
NL, N_SYM, N_PAD, N_RE = 4, 2, 896, 798

def main():
    dd = Path(os.environ.get("AIRAN_DATA_DIR","."))
    gdir, adir = dd/"data"/"golden", dd/"data"/"ascend_output"
    print("=== verify mimo_dmrs_gen ===")
    if not adir.exists():
        print("  [error] no ascend_output"); return 2
    n_pass = 0
    for name in CASES:
        try:
            g_re = np.fromfile(gdir/name/"x_re.bin", dtype=np.float16).astype(np.float32)
            g_im = np.fromfile(gdir/name/"x_im.bin", dtype=np.float16).astype(np.float32)
            a_re = np.fromfile(adir/f"mimo_dmrs_{name}_re.bin", dtype=np.float16).astype(np.float32)
            a_im = np.fromfile(adir/f"mimo_dmrs_{name}_im.bin", dtype=np.float16).astype(np.float32)
        except FileNotFoundError:
            print(f"  {name:<20} [MISSING]"); continue
        if a_re.size != g_re.size:
            print(f"  {name:<20} [SIZE {a_re.size} vs {g_re.size}]"); continue
        re_err = float(np.max(np.abs(a_re-g_re))); im_err = float(np.max(np.abs(a_im-g_im)))
        ok = re_err < ERR_THRESH and im_err < ERR_THRESH
        print(f"  {name:<20} re={re_err:.5f} im={im_err:.5f}  {'[PASS]' if ok else '[FAIL]'}")
        if not ok:
            gr = g_re.reshape(NL,N_SYM,N_PAD); ar = a_re.reshape(NL,N_SYM,N_PAD)
            for l in range(NL):
                le = float(np.max(np.abs(ar[l]-gr[l])))
                print(f"       layer{l} re_err={le:.5f}")
        if ok: n_pass += 1
    print(f"\n  Total: {n_pass}/{len(CASES)} PASS")
    return 0 if n_pass==len(CASES) else 1

if __name__ == "__main__":
    sys.exit(main())
