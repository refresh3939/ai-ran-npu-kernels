#!/usr/bin/env python3
"""
verify_result.py — OFDM mod kernel 端到端验证

比较 kernel 输出 (int16 Q8.8 separated re/im, 30720 时域样点, 含 CP)
vs golden output_re/im.bin (Python ofdm_mod 参考).
Tolerance: max |err| < 0.1 (de-quantized, fp16 cube 累加 + 量化噪声).
"""
import os
import sys
import numpy as np

# ─── 参数 (与 ofdm_mod.h / ref 一致) ──────────────────────────
N_FFT     = 2048
N_SYMBOL  = 14
N_SAMPLE  = 30720
# de-quant 用空口增益 AIR_SCALE (= ofdm_mod.h OUT_SCALE), 与满量程输出自洽.
# (旧 /256 是 v2 小输出时代的绝对门槛; v3 填满 int16 后 /256 会人为放大误差.)
AIR_SCALE = 1600.0
CP_LENS   = np.array([176] + [144] * 13, dtype=np.int64)
# de-quant 后是 unitary 时域单位 (RMS~0.88). tol=0.05 ≈ 3.6% 信号RMS (256QAM EVM 上限),
# 对实测 fp16 floor (~0.002 单位功率 / ~0.011 高PAPR) 留 4.5~27x 余量.
ABS_TOL   = 0.05

AIRAN_DATA_DIR = os.environ.get("AIRAN_DATA_DIR") or \
                 os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

G_RE = f"{AIRAN_DATA_DIR}/data/golden/output_re.bin"
G_IM = f"{AIRAN_DATA_DIR}/data/golden/output_im.bin"
K_RE = f"{AIRAN_DATA_DIR}/data/ascend_output/output_re.bin"
K_IM = f"{AIRAN_DATA_DIR}/data/ascend_output/output_im.bin"


def main():
    for p in (G_RE, G_IM, K_RE, K_IM):
        if not os.path.exists(p):
            print(f"[verify] missing: {p}")
            sys.exit(1)

    g_re = np.fromfile(G_RE, dtype=np.int16).astype(np.float32) / AIR_SCALE
    g_im = np.fromfile(G_IM, dtype=np.int16).astype(np.float32) / AIR_SCALE
    k_re = np.fromfile(K_RE, dtype=np.int16).astype(np.float32) / AIR_SCALE
    k_im = np.fromfile(K_IM, dtype=np.int16).astype(np.float32) / AIR_SCALE

    print(f"[verify] golden re = {G_RE}")
    print(f"[verify] kernel re = {K_RE}")
    print(f"[verify] samples   = {N_SAMPLE}  tolerance = {ABS_TOL}\n")

    for name, k, g in (("re", k_re, g_re), ("im", k_im, g_im)):
        if k.size != N_SAMPLE or g.size != N_SAMPLE:
            print(f"[verify] {name} size mismatch: kernel={k.size} golden={g.size} (want {N_SAMPLE})")
            sys.exit(1)

    # 逐 symbol body 区间比较 (CP 区间与 body 尾部相同, 一起覆盖)
    offs = np.concatenate([[0], np.cumsum(CP_LENS + N_FFT)])
    print("[verify] per-symbol body max |err|:")
    overall_max = 0.0
    n_pass = 0
    for s in range(N_SYMBOL):
        start = int(offs[s] + CP_LENS[s])
        sl = slice(start, start + N_FFT)
        er = float(np.abs(k_re[sl] - g_re[sl]).max())
        ei = float(np.abs(k_im[sl] - g_im[sl]).max())
        m = max(er, ei)
        overall_max = max(overall_max, m)
        ok = m <= ABS_TOL
        n_pass += int(ok)
        print(f"  sym {s:2d}:  re={er:7.4f}  im={ei:7.4f}   [{'OK' if ok else 'FAIL'}]")

    # 整段 (含 CP) 比较, 抓 CP 插入 bug
    full_max = max(float(np.abs(k_re - g_re).max()),
                   float(np.abs(k_im - g_im).max()))

    g_abs = max(float(np.abs(g_re).max()), float(np.abs(g_im).max()))
    k_abs = max(float(np.abs(k_re).max()), float(np.abs(k_im).max()))
    rms   = float(np.sqrt(np.mean(g_re**2 + g_im**2)))
    evm   = full_max / rms * 100.0 if rms > 0 else 0.0
    print()
    print(f"[verify] body OVERALL max |err| = {overall_max:.4f}  (de-quant /AIR_SCALE)")
    print(f"[verify] full-slot   max |err| = {full_max:.4f}  (含 CP)")
    print(f"[verify] tolerance              = {ABS_TOL}  (margin {ABS_TOL/full_max:.1f}x)" if full_max > 0
          else f"[verify] tolerance              = {ABS_TOL}")
    print(f"[verify] peak |err| / signal RMS = {evm:.2f}%   (256QAM EVM 上限 3.5%)")
    print(f"[verify] golden max |val|       = {g_abs:.4f}")
    print(f"[verify] kernel max |val|       = {k_abs:.4f}")
    print()

    cp_ok = full_max <= ABS_TOL

    # ── 交织 IQ 检查: out_iq[0::2]==plane_re, out_iq[1::2]==plane_im (位精确) ──
    K_IQ = f"{AIRAN_DATA_DIR}/data/ascend_output/output_iq.bin"
    iq_ok = True
    if os.path.exists(K_IQ):
        kr_i = np.fromfile(K_RE, dtype=np.int16)
        ki_i = np.fromfile(K_IM, dtype=np.int16)
        iq_i = np.fromfile(K_IQ, dtype=np.int16)
        if iq_i.size != 2 * N_SAMPLE:
            print(f"[verify] IQ size mismatch: {iq_i.size} (want {2*N_SAMPLE})")
            iq_ok = False
        else:
            n_re = int(np.count_nonzero(iq_i[0::2] != kr_i))
            n_im = int(np.count_nonzero(iq_i[1::2] != ki_i))
            iq_ok = (n_re == 0 and n_im == 0)
            print(f"[verify] interleave  iq[0::2]vs re mismatches = {n_re}")
            print(f"[verify] interleave  iq[1::2]vs im mismatches = {n_im}   [{'OK' if iq_ok else 'FAIL'}]")
    else:
        print(f"[verify] (no output_iq.bin — 交织检查跳过)")
    print()

    if n_pass == N_SYMBOL and cp_ok and iq_ok:
        print(f"[verify] ========== PASS ({n_pass}/{N_SYMBOL}, CP OK, IQ OK) ==========")
        return 0
    print(f"[verify] ========== FAIL ({n_pass}/{N_SYMBOL}, CP {'OK' if cp_ok else 'FAIL'}, IQ {'OK' if iq_ok else 'FAIL'}) ==========")
    return 1


if __name__ == "__main__":
    sys.exit(main())
