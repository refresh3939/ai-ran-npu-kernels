#!/usr/bin/env python3
"""
verify_result.py — cfo_compensate kernel 端到端验证 (交织输出)

kernel 输出 out_iq.bin (交织 int16 Q8.8) vs golden output.bin (同格式).
阈值: max|Δ| ≤ 2 LSB (fp16 复乘 + round 量化).
"""
import os
import sys
import numpy as np

N_SLOT  = 30720
LSB_TOL = 2

AIRAN_DATA_DIR = os.environ.get("AIRAN_DATA_DIR") or \
                 os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GOLDEN  = f"{AIRAN_DATA_DIR}/data/golden/output.bin"
KERN_IQ = f"{AIRAN_DATA_DIR}/data/ascend_output/out_iq.bin"


def main():
    for p in (GOLDEN, KERN_IQ):
        if not os.path.exists(p):
            print(f"[verify] missing: {p}")
            sys.exit(1)

    g = np.fromfile(GOLDEN,  dtype=np.int16)
    k = np.fromfile(KERN_IQ, dtype=np.int16)
    if k.size != g.size:
        print(f"[verify] size 不符: kernel {k.size} vs golden {g.size}")
        return 1

    print(f"[verify] golden = {GOLDEN}")
    print(f"[verify] kernel = {KERN_IQ}")
    print(f"[verify] N={N_SLOT} (交织 {2*N_SLOT} int16)  tolerance = {LSB_TOL} LSB\n")

    if (k == -21846).all():     # 0xAA sentinel
        print("[verify] ✗ 全 0xAA — kernel 没写 (Quirk #4 输出位置? / Gather 没跑?)")
        return 1

    d = np.abs(k.astype(np.int32) - g.astype(np.int32))
    de_re, de_im = d[0::2], d[1::2]
    overall = int(d.max())
    n_bad = int((d > LSB_TOL).sum())

    print(f"[verify] max|Δre| = {int(de_re.max())}  max|Δim| = {int(de_im.max())}")
    print(f"[verify] OVERALL max|Δ| = {overall} LSB   超阈样点 = {n_bad}/{2*N_SLOT}")
    print(f"[verify] golden |val| max = {int(np.abs(g).max())}\n")

    if overall <= LSB_TOL:
        print(f"[verify] ========== PASS (max|Δ|={overall} ≤ {LSB_TOL}) ==========")
        return 0
    print(f"[verify] ========== FAIL (max|Δ|={overall} > {LSB_TOL}) ==========")
    bad = np.where(d > LSB_TOL)[0]
    if bad.size:
        # 诊断: 错位是偶(re)/奇(im)? 落哪个子块? (定位 AIV / Gather 单位错)
        sample = bad // 2
        st = sample // 2048
        par = "re" if (bad[0] % 2 == 0) else "im"
        print(f"[verify] 首错 @int16#{bad[0]} ({par}), 子块 {sorted(set((st).tolist()))[:8]}")
        if np.array_equal(k[0::2], g[1::2]) or (de_re > de_im).all():
            print("[verify] 提示: re/im 像是错位 → Gather idx 单位 (element vs byte) 可能反了")
    return 1


if __name__ == "__main__":
    sys.exit(main())