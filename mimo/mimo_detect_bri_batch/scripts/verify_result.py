#!/usr/bin/env python3
"""Validate production x_hat/no_eff outputs against float64 exact LMMSE."""
import glob
import os
import sys
import time

import numpy as np


def die(message):
    sys.exit(f"[FAIL] {message}")


if len(sys.argv) != 1:
    die("clean 最终版只支持端到端验收，不再提供中间阶段选择")

nr = int(os.environ.get("NR", 64))
nl = int(os.environ.get("NL", 16))
nl_real = int(os.environ.get("NL_REAL", nl))
pack = int(os.environ.get("PACK", 0))
p = (nl // nl_real) if pack else 1
n_sc_pad = int(os.environ.get("AIRAN_SCPAD", 1664))
n_re = 14 * n_sc_pad
stale_min = float(os.environ.get("STALE_MIN", 10))
case = (f"case_0_m{nr}_k{nl}_sc{n_sc_pad}"
        + (f"_r{nl_real}" if nl_real != nl else "")
        + ("_pk" if p > 1 else ""))

here = os.path.dirname(os.path.abspath(__file__))
root = os.environ.get("AIRAN_DATA_DIR", os.path.join(here, ".."))
gold = os.path.join(root, "data", "golden", case)
asc = os.path.join(root, "data", "ascend_output", case)


def load(directory, name):
    path = os.path.join(directory, name + ".bin")
    if not os.path.exists(path):
        die(f"缺文件 {path}")
    return np.fromfile(path, np.float16).astype(np.float64)


print("=== verify [out] 端到端 x̂ / no_eff (唯一验收判据) ===")
print(f"[case] {case}   (NR={nr} NL={nl} NL_REAL={nl_real} PACK={pack} P={p})")
print(f"[asc ] {asc}")
outputs = sorted(glob.glob(os.path.join(asc, "*.bin")))
if not outputs:
    die(f"{asc} 里没有 .bin，请先运行 bash run.sh")
age = (time.time() - max(os.path.getmtime(path) for path in outputs)) / 60
print(f"[age ] 最新输出 {age:.1f} 分钟前")
if age > stale_min:
    die(f"输出比 {stale_min:.0f} 分钟更旧，拒绝验证陈旧结果")

mask = load(gold, "mask").reshape(n_re)
idx = np.where(mask > 0.5)[0]
region = np.ix_(np.arange(nl_real), idx)
gx = (load(gold, "xhat_re").reshape(nl, n_re)
      + 1j * load(gold, "xhat_im").reshape(nl, n_re))[region]
ax = (load(asc, "xhat_re").reshape(nl, n_re)
      + 1j * load(asc, "xhat_im").reshape(nl, n_re))[region]
evm = np.sqrt(np.mean(np.abs(ax - gx) ** 2) / np.mean(np.abs(gx) ** 2))

gne = load(gold, "no_eff").reshape(nl, n_re)[region]
ane = load(asc, "no_eff").reshape(nl, n_re)[region]
noise_error = np.median(np.abs(ane - gne) / (np.abs(gne) + 1e-3))

evm_ok = evm < 0.05
noise_ok = noise_error < 0.10
print(f"\n[x_hat ] EVM={evm*100:.2f}%  (门限<5%)   {'PASS' if evm_ok else 'FAIL'}")
print(f"[no_eff] rel median={noise_error:.2e}  (门限<1e-01)   {'PASS' if noise_ok else 'FAIL'}")
print(f"\n=== RESULT: {'PASS' if evm_ok and noise_ok else 'FAIL'} ===")
sys.exit(0 if evm_ok and noise_ok else 1)
