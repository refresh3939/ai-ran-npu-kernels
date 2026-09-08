"""
interpolate ×8 verify — kernel 交织单 buffer 输出 vs golden (都交织, 直接比)。

kernel 输出 (对外交织, drop-in 喂 USRP DAC):
    data/ascend_output/output.bin     int16 IQ 交织 [2*N_OUT]
golden (interpolate_ref.py):
    data/golden/output.bin            int16 IQ 交织 [2*N_OUT]

阈值: fp16-sim 自检 1 LSB, kernel fp16 累加同模型 → 容差 2 LSB。
"""
import os
import sys
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_KERNEL_DIR = _SCRIPT_DIR.parent

N_OUT   = 1_228_800
TOL_LSB = 2            # 最大允许 |err| (LSB)

_DATA = Path(os.environ.get("AIRAN_DATA_DIR", str(_KERNEL_DIR)))
GOLD   = _DATA / "data" / "golden" / "output.bin"
OUT    = _DATA / "data" / "ascend_output" / "output.bin"


def main():
    for p in (GOLD, OUT):
        if not p.exists():
            print(f"[verify] ✗ 缺文件: {p}")
            sys.exit(1)

    g = np.fromfile(GOLD, dtype=np.int16).astype(np.int64)
    y = np.fromfile(OUT,  dtype=np.int16).astype(np.int64)
    if g.size != 2 * N_OUT:
        print(f"[verify] ✗ golden 大小 {g.size} != {2*N_OUT}")
        sys.exit(1)
    if y.size != 2 * N_OUT:
        print(f"[verify] ✗ kernel 输出大小 {y.size} != {2*N_OUT} (应为交织 [I,Q,...])")
        sys.exit(1)

    gi, gq = g[0::2], g[1::2]          # 去交织对比 (诊断用)
    yi, yq = y[0::2], y[1::2]
    ei = np.abs(yi - gi)
    eq = np.abs(yq - gq)

    thirds = [("head", 0, N_OUT // 3),
              ("mid",  N_OUT // 3, 2 * N_OUT // 3),
              ("tail", 2 * N_OUT // 3, N_OUT)]
    core_bounds = [307200, 614400, 921600]      # OUT_PER_CORE 边界 (4 核)

    print("=" * 60)
    print(" interpolate ×8 verify  (kernel 交织  vs  golden 交织)")
    print("=" * 60)
    print(f"  {'region':<8} {'max|eI|':>8} {'max|eQ|':>8} {'#bad':>8}")
    for name, a, b in thirds:
        mi = ei[a:b].max(); mq = eq[a:b].max()
        nb = int((ei[a:b] > TOL_LSB).sum() + (eq[a:b] > TOL_LSB).sum())
        print(f"  {name:<8} {mi:>8d} {mq:>8d} {nb:>8d}")

    print(f"  -- 核边界 ±8 (max|e|) --")               # ±8 = 一个 8m 输出块
    for cb in core_bounds:
        lo, hi = max(0, cb - 8), min(N_OUT, cb + 8)
        print(f"     @{cb:<7} eI={ei[lo:hi].max():>4d} eQ={eq[lo:hi].max():>4d}")

    # 交织结构自检: 若 8 相 / I/Q 整体错位 (transpose 方向错), eI/eQ 会同时爆大
    mxI, mxQ = int(ei.max()), int(eq.max())
    argI, argQ = int(ei.argmax()), int(eq.argmax())
    print("-" * 60)
    print(f"  全局 max|eI|={mxI} @{argI}   max|eQ|={mxQ} @{argQ}   阈值={TOL_LSB}")

    ok = (mxI <= TOL_LSB) and (mxQ <= TOL_LSB)
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
