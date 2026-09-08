#!/usr/bin/env python3
"""
verify_result.py — re_demap kernel 验证

纯置换算子 → 逐 element bit-exact, TOL=0.
比较 kernel 输出 yused_re/im (fp16 [14,1600]) vs golden.
"""
import os
import sys
import numpy as np

N_SYMBOL = 14
N_SC_PAD = 1664

AIRAN_DATA_DIR = os.environ.get("AIRAN_DATA_DIR") or \
                 os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

G_RE = f"{AIRAN_DATA_DIR}/data/golden/yused_re.bin"
G_IM = f"{AIRAN_DATA_DIR}/data/golden/yused_im.bin"
K_RE = f"{AIRAN_DATA_DIR}/data/ascend_output/yused_re.bin"
K_IM = f"{AIRAN_DATA_DIR}/data/ascend_output/yused_im.bin"


def main():
    for p in (G_RE, G_IM, K_RE, K_IM):
        if not os.path.exists(p):
            print(f"[verify] missing: {p}")
            sys.exit(1)

    g_re = np.fromfile(G_RE, dtype=np.float16).reshape(N_SYMBOL, N_SC_PAD)
    g_im = np.fromfile(G_IM, dtype=np.float16).reshape(N_SYMBOL, N_SC_PAD)
    k_re = np.fromfile(K_RE, dtype=np.float16).reshape(N_SYMBOL, N_SC_PAD)
    k_im = np.fromfile(K_IM, dtype=np.float16).reshape(N_SYMBOL, N_SC_PAD)

    print(f"[verify] shape=({N_SYMBOL},{N_SC_PAD})  TOL=0 (纯置换)\n")
    print("[verify] per-symbol mismatch (re / im):")
    n_pass = 0
    for s in range(N_SYMBOL):
        bad_re = int((k_re[s] != g_re[s]).sum())
        bad_im = int((k_im[s] != g_im[s]).sum())
        ok = (bad_re == 0 and bad_im == 0)
        n_pass += ok
        # 首个错位 element 便于定位 gather offset bug
        extra = ""
        if not ok:
            idx = np.where((k_re[s] != g_re[s]) | (k_im[s] != g_im[s]))[0]
            extra = f"  first_bad_idx={idx[0]} g=({g_re[s,idx[0]]:.3f},{g_im[s,idx[0]]:.3f}) " \
                    f"k=({k_re[s,idx[0]]:.3f},{k_im[s,idx[0]]:.3f})"
        print(f"  sym {s:2d}:  re={bad_re:5d}  im={bad_im:5d}   [{'OK' if ok else 'FAIL'}]{extra}")

    print()
    if n_pass == N_SYMBOL:
        print(f"[verify] ========== PASS ({n_pass}/{N_SYMBOL}) ==========")
        return 0
    print(f"[verify] ========== FAIL ({n_pass}/{N_SYMBOL}) ==========")
    return 1


if __name__ == "__main__":
    sys.exit(main())