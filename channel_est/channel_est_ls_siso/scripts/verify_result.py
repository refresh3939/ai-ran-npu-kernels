#!/usr/bin/env python3
"""
verify_result.py — v6.2 verifier (natural-order H on 1664-padded grid).
"""
import sys, os
from pathlib import Path
import numpy as np

N_SYM = 14
N_SC_USED = 1664

CASES = [
    ("case_0_awgn_only",      1e-3, 0.05, 0.01),
    ("case_1_flat_fading",    1e-3, 0.05, 0.01),
    ("case_2_freq_selective", 5e-3, 0.10, 0.01),
    ("case_3_freq_time_var",  1e-2, 0.15, 0.01),
    ("case_4_low_snr",        1.0,  3.0,  0.01),
]


def load_grid(path):
    return np.fromfile(path, dtype=np.float16).reshape(N_SYM, N_SC_USED).astype(np.float32)


def main():
    # Default: kernel-local data/ (2 levels up from this script: scripts/ → channel_est_ls/)
    root = Path(os.environ.get("AIRAN_DATA_DIR",
                               str(Path(__file__).resolve().parent.parent)))
    golden = root / "data" / "golden"
    out_d  = root / "data" / "ascend_output"
    print(f"Golden : {golden}")
    print(f"Output : {out_d}")
    print(f"Grid   : [14, {N_SC_USED}] padded NATURAL ORDER (raw 1596 + zero-pad 68)")

    n_pass = 0
    for name, h_mse_tol, h_err_tol, ev_rel_tol in CASES:
        gd = golden / name
        od = out_d / name
        files = ["truth_h_re.bin", "truth_h_im.bin", "truth_err_var.bin"]
        outs  = ["out_h_re.bin",   "out_h_im.bin",   "out_err_var.bin"]
        missing = [f for f, p in zip(files+outs, [gd/f for f in files] + [od/o for o in outs]) if not p.exists()]
        if missing:
            print(f"  [FAIL] {name}: missing {missing}"); continue

        tr_re = load_grid(gd / "truth_h_re.bin")
        tr_im = load_grid(gd / "truth_h_im.bin")
        o_re  = load_grid(od / "out_h_re.bin")
        o_im  = load_grid(od / "out_h_im.bin")
        tr_ev = load_grid(gd / "truth_err_var.bin")
        o_ev  = load_grid(od / "out_err_var.bin")

        d_re = (o_re - tr_re).flatten()
        d_im = (o_im - tr_im).flatten()
        mse_re = float(np.mean(d_re ** 2)); mse_im = float(np.mean(d_im ** 2))
        mxe_re = float(np.max(np.abs(d_re))); mxe_im = float(np.max(np.abs(d_im)))
        agg_mse = (mse_re + mse_im) / 2.0
        max_err = max(mxe_re, mxe_im)

        ev_exp = float(tr_ev[0, 0]); ev_got = float(o_ev[0, 0])
        ev_rel = abs(ev_got - ev_exp) / max(abs(ev_exp), 1e-30)
        ev_max_abs = float(np.max(np.abs(o_ev - tr_ev)))

        ok_h = (agg_mse < h_mse_tol) and (max_err < h_err_tol)
        ok_ev = (ev_rel < ev_rel_tol) and (ev_max_abs < 1e-3)
        ok = ok_h and ok_ev
        if ok: n_pass += 1
        s = "PASS" if ok else "FAIL"
        print(f"  {name:<25}  {s}")
        print(f"      H_re   MSE={mse_re:.4e}  maxErr={mxe_re:.4f}    "
              f"H_im   MSE={mse_im:.4e}  maxErr={mxe_im:.4f}")
        print(f"      agg_MSE={agg_mse:.4e}  max_err={max_err:.4f}  "
              f"tol(MSE<{h_mse_tol:.2e}, err<{h_err_tol})  {'✓' if ok_h else '✗'}")
        print(f"      err_var expect={ev_exp:.4e}  got={ev_got:.4e}  "
              f"rel_err={ev_rel:.2e}  {'✓' if ok_ev else '✗'}")

    print(f"\n{n_pass}/{len(CASES)} PASS")
    sys.exit(0 if n_pass == len(CASES) else 1)


if __name__ == "__main__":
    main()
