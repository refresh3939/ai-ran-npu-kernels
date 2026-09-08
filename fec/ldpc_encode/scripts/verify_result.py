#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# kernels/tx/ldpc_encode/scripts/verify_result.py
#
# 对齐 decode 的 scripts/verify.py 模式：单一职责，只比对，不 build / 不跑 kernel。
#
# 读：
#   data/golden/output.bin          int8 (n_g, N_RAW)   ref.py 产出（默认 4 CB）
#   data/ascend_output/output.bin   int8 (n_a, N_RAW)   main.cpp dump
# 比：
#   golden 不足 n_a 时按 4-CB tile 上去（与 main.cpp 内部 tile 一致），int8 bit-exact。
#
# 用法：
#   python scripts/verify_result.py
#   AIRAN_DATA_DIR=/path/to/data python scripts/verify_result.py   # 覆盖数据根
# ============================================================================

import os
import sys
from pathlib import Path
import numpy as np

# 必须与 ldpc_encode.h 一致
N_RAW = 25344

HERE   = Path(__file__).resolve().parent          # .../ldpc_encode/scripts
KERNEL = HERE.parent                               # .../ldpc_encode
DATA   = Path(os.environ["AIRAN_DATA_DIR"]) if os.environ.get("AIRAN_DATA_DIR") \
         else (KERNEL / "data")

GOLDEN = DATA / "golden" / "output.bin"
ACTUAL = DATA / "ascend_output" / "output.bin"


def _load(path, label):
    if not path.exists():
        print(f"[FAIL] {label} 不存在: {path}")
        if label == "golden":
            print("       先跑: python scripts/ldpc_ref.py")
        else:
            print("       先跑: bash run.sh -r npu -v Ascend310P1")
        sys.exit(1)
    a = np.fromfile(path, dtype=np.int8)
    if a.size % N_RAW != 0:
        print(f"[FAIL] {label} size {a.size} 非 N_RAW({N_RAW}) 整数倍")
        sys.exit(1)
    return a.reshape(-1, N_RAW)


def main():
    print("=" * 70)
    print("[verify_result] ldpc_encode  golden vs ascend_output")
    print(f"  golden = {GOLDEN}")
    print(f"  actual = {ACTUAL}")
    print("=" * 70)

    golden = _load(GOLDEN, "golden")          # (n_g, N_RAW)
    actual = _load(ACTUAL, "actual")          # (n_a, N_RAW)
    n_g, n_a = golden.shape[0], actual.shape[0]
    print(f"  golden CB = {n_g}   actual CB = {n_a}")

    # golden 不足时按前 4 CB 循环 tile（与 main.cpp: src_cb = cb % 4 一致）
    if n_g < n_a:
        base = min(n_g, 4)
        idx = np.arange(n_a) % base
        golden = golden[idx]
        print(f"  golden tile {base}-CB -> {n_a}-CB（mod {base}）")
    elif n_g > n_a:
        golden = golden[:n_a]
        print(f"  golden 截断 -> {n_a}-CB")

    diff = (golden != actual)
    mism = int(diff.sum())
    total = golden.size
    per_cb_fail = int(diff.any(axis=1).sum())

    if mism == 0:
        print(f"\n[PASS] {total} 个 int8 全 bit-exact  ({n_a}/{n_a} CB)")
        sys.exit(0)

    print(f"\n[FAIL] {mism}/{total} bit mismatch ({mism/total*100:.4f}%)  "
          f"{per_cb_fail}/{n_a} CB fail")
    print("  per-CB diff (前 8 CB):")
    for cb in range(min(n_a, 8)):
        d = int((golden[cb] != actual[cb]).sum())
        print(f"    CB {cb}: diff={d} / {N_RAW}")
    flat_g, flat_a = golden.flatten(), actual.flatten()
    for idx in np.where(flat_g != flat_a)[0][:5]:
        print(f"    idx={idx}: golden={flat_g[idx]} actual={flat_a[idx]}")
    sys.exit(1)


if __name__ == "__main__":
    main()