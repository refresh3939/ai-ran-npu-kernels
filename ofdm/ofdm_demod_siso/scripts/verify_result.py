#!/usr/bin/env python3
"""
verify_result.py — OFDM demod kernel 端到端验证

比较 kernel 输出 (fp16 re/im 拆分) vs sionna 黄金参考 stage4_dft64 (complex64).
Tolerance: max |err| <= 0.05（unitary fp16 权重与 Cube 累加误差）
"""
import os
import sys
import numpy as np

# ─── 参数 ─────────────────────────────────────────────────────
N_FFT    = 2048
N_SYMBOL = 14
P        = 32
Q        = 64
ABS_TOL  = 0.05   # unitary fp16 权重 + Cube 累加误差

# ─── 路径 (scripts/ 上一层是 kernel 目录) ─────────────────────
AIRAN_DATA_DIR = os.environ.get("AIRAN_DATA_DIR") or \
                 os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GOLDEN    = f"{AIRAN_DATA_DIR}/data/golden/stage4_dft64.bin"
KERN_RE   = f"{AIRAN_DATA_DIR}/data/ascend_output/output_re.bin"
KERN_IM   = f"{AIRAN_DATA_DIR}/data/ascend_output/output_im.bin"


def main():
    # 检查文件存在
    for p in (GOLDEN, KERN_RE, KERN_IM):
        if not os.path.exists(p):
            print(f"[verify] missing: {p}")
            sys.exit(1)

    # ─── 读 golden (complex64, shape: 14 × 32 × 64) ──────────
    golden = np.fromfile(GOLDEN, dtype=np.complex64).reshape(N_SYMBOL, P, Q)

    # ─── 读 kernel 输出 (fp16, shape: 14 × 32 × 64) ──────────
    kern_re = np.fromfile(KERN_RE, dtype=np.float16).reshape(N_SYMBOL, P, Q)
    kern_im = np.fromfile(KERN_IM, dtype=np.float16).reshape(N_SYMBOL, P, Q)

    print(f"[verify] golden    = {GOLDEN}")
    print(f"[verify] kernel re = {KERN_RE}")
    print(f"[verify] kernel im = {KERN_IM}")
    print(f"[verify] shape     = ({N_SYMBOL}, {P}, {Q})  tolerance = {ABS_TOL}\n")

    # ─── 逐 sym 比较 ─────────────────────────────────────────
    print("[verify] per-symbol max |err|:")
    overall_max = 0.0
    n_pass      = 0
    for s in range(N_SYMBOL):
        er = float(np.abs(kern_re[s].astype(np.float32) - golden[s].real).max())
        ei = float(np.abs(kern_im[s].astype(np.float32) - golden[s].imag).max())
        m  = max(er, ei)
        overall_max = max(overall_max, m)
        ok = m <= ABS_TOL
        if ok:
            n_pass += 1
        flag = "OK" if ok else "FAIL"
        print(f"  sym {s:2d}:  re={er:7.4f}  im={ei:7.4f}   [{flag}]")

    # ─── 范围对比 ────────────────────────────────────────────
    g_abs = max(np.abs(golden.real).max(), np.abs(golden.imag).max())
    k_abs = max(np.abs(kern_re).max(), np.abs(kern_im).max())
    print()
    print(f"[verify] OVERALL max |err| = {overall_max:.4f}")
    print(f"[verify] tolerance         = {ABS_TOL}")
    print(f"[verify] golden max |val|  = {g_abs:.4f}")
    print(f"[verify] kernel max |val|  = {k_abs:.4f}")
    print()

    # ─── 总结 ────────────────────────────────────────────────
    if n_pass == N_SYMBOL:
        print(f"[verify] ========== PASS ({n_pass}/{N_SYMBOL}) ==========")
        return 0
    print(f"[verify] ========== FAIL ({n_pass}/{N_SYMBOL}) ==========")
    return 1


if __name__ == "__main__":
    sys.exit(main())
