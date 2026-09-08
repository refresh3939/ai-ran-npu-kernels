#!/usr/bin/env python3
"""
verify_result.py — ssb_fft kernel 端到端验证

比 kernel R_re/R_im (fp16 [144]) vs golden R_re/R_im (ref 产, fp16 [144]).
Tolerance: max|err| < 0.05  (fp16 量化 + cube fp32 累加→fp16 cast).
"""
import os, sys
import numpy as np

N_DMRS_RE = 144
ABS_TOL   = 0.05

AIRAN_DATA_DIR = os.environ.get("AIRAN_DATA_DIR") or \
                 os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

G_RE = f"{AIRAN_DATA_DIR}/data/golden/R_re.bin"
G_IM = f"{AIRAN_DATA_DIR}/data/golden/R_im.bin"
K_RE = f"{AIRAN_DATA_DIR}/data/ascend_output/R_re.bin"
K_IM = f"{AIRAN_DATA_DIR}/data/ascend_output/R_im.bin"


def main():
    for p in (G_RE, G_IM, K_RE, K_IM):
        if not os.path.exists(p):
            print(f"[verify] missing: {p}"); sys.exit(1)

    g_re = np.fromfile(G_RE, dtype=np.float16).astype(np.float32)
    g_im = np.fromfile(G_IM, dtype=np.float16).astype(np.float32)
    k_re = np.fromfile(K_RE, dtype=np.float16).astype(np.float32)
    k_im = np.fromfile(K_IM, dtype=np.float16).astype(np.float32)

    assert g_re.size == N_DMRS_RE, f"golden R_re size {g_re.size} != {N_DMRS_RE}"
    assert k_re.size == N_DMRS_RE, f"kernel R_re size {k_re.size} != {N_DMRS_RE}"

    print(f"[verify] golden = {G_RE}")
    print(f"[verify] kernel = {K_RE}")
    print(f"[verify] N_DMRS_RE = {N_DMRS_RE}  tolerance = {ABS_TOL}\n")

    er = np.abs(k_re - g_re)
    ei = np.abs(k_im - g_im)
    m  = float(max(er.max(), ei.max()))
    n_bad = int(((er > ABS_TOL) | (ei > ABS_TOL)).sum())

    # 前 8 个 RE 抽样
    print("[verify] sample (first 8 RE):  golden(re,im) | kernel(re,im) | |err|")
    for i in range(8):
        print(f"  RE {i:3d}: ({g_re[i]:+.4f},{g_im[i]:+.4f}) | "
              f"({k_re[i]:+.4f},{k_im[i]:+.4f}) | {max(er[i],ei[i]):.4f}")

    print()
    print(f"[verify] OVERALL max|err| = {m:.4f}   bad RE = {n_bad}/{N_DMRS_RE}")
    print(f"[verify] golden |R| range = {np.abs(g_re+1j*g_im).min():.3f} .. "
          f"{np.abs(g_re+1j*g_im).max():.3f}")
    print(f"[verify] kernel |R| range = {np.abs(k_re+1j*k_im).min():.3f} .. "
          f"{np.abs(k_re+1j*k_im).max():.3f}\n")

    if n_bad == 0:
        print(f"[verify] ========== PASS (144/144) ==========")
        return 0
    print(f"[verify] ========== FAIL ({N_DMRS_RE - n_bad}/{N_DMRS_RE}) ==========")
    return 1


if __name__ == "__main__":
    sys.exit(main())
