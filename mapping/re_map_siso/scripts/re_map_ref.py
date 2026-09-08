"""
re_map reference — RE 映射 (resource mapper), TX 链.

职责: 把 1596 个 used SC 散布回 2048 全 FFT bin 网格, guard 填 0.
本算子 **无算术**, 纯置换 (scatter, 用 inverse-gather 实现) → kernel vs golden 必须 TOL=0 bit-exact.

re_map 是 re_demap 的逆置换:
    demap(map(y)) = y      (used SC 上恒等回环)
    map 输出网格的内存排布 == demap 输入网格 (raw stage4 k1k2), 供 TX ofdm_mod (IFFT) 直接吃.

跑法:
    cd ~/AI-RAN-NPU/kernels/tx/re_map
    python3 scripts/re_map_ref.py

输出:
    weights/scatter_idx.bin             (uint32, N_FFT, BYTE offset, kernel Gather 用)
    data/golden/input_re.bin/im.bin     (fp16, [14,1664], used SC, kernel 入; [1596:1664]=pad=0)
    data/golden/grid_re.bin/im.bin      (fp16, [14,2048], 全 FFT 网格, golden; guard=0)

═══════════════════════════════════════════════════════════════════════════
约定 A / B 与 re_demap **完全一致** (镜像), 改 demap 就同步改这里:
═══════════════════════════════════════════════════════════════════════════

[约定 A] OFDM 网格内存排布 (raw stage4 k1k2, 与 ofdm_demod 输出一致)
    grid[sym*2048 + m] = X2[k1, k2],  k1 = m // 64,  k2 = m % 64
    natural FFT bin  n = k1 + 32*k2  =>  k1 = n % 32, k2 = n // 32
    => mem offset(n) = (n % 32)*64 + (n // 32)
    若以后 ofdm_mod 改吃 natural-order, 把 OFDM_OUTPUT_RAW_K1K2=False.

[约定 B] 1596 used SC 如何居中布进 2048 (与 demap used_fft_bins 同序)
    2048 = 226 + 1596 + 226, guard 整数对称 → 无 DC null.
    输出顺序 (grid SC 0 = 最负频):
        负频 798: natural bin 1250..2047
        DC    1 : natural bin 0
        正频 797: natural bin 1..797
    => used_fft_bins = [1250..2047, 0, 1..797], 共 1596.

═══════════════════════════════════════════════════════════════════════════
inverse-gather 思路 (scatter 的 NPU 实现):
    map 是 scatter (out[elem[u]] = in[u]), 但 dav_m200 只有可靠的 Gather.
    => 翻成 "按输出位 m 反查输入下标":
         inv[m] = u        若 m == elem[u] (used 位)
         inv[m] = ZERO_SLOT 否则           (guard 位 → 输入 UB 的 0 槽)
       out[m] = src[inv[m]],  其中 src = [in(1664) , 0(zero slot)]
    guard 位全部指向同一个 0, 一次 Gather 把 2048 网格写满 (含 0), 无需预清网格.
═══════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_KERNEL_DIR = _SCRIPT_DIR.parent
WEIGHTS_DIR = _KERNEL_DIR / "weights"
GOLDEN_DIR  = _KERNEL_DIR / "data" / "golden"

# ── 锁定常量 (镜像 re_demap) ──────────────────────────────────────
N_FFT      = 2048
N_SYMBOL   = 14
N_SC_USED  = 1596
N_SC_PAD   = 1664          # 输入 used SC 宽度, 13*128, 128-align; 1596 有效 + 68 pad
P          = 32            # Cooley-Tukey 内 DFT
Q          = 64            # Cooley-Tukey 外 DFT
GUARD      = (N_FFT - N_SC_USED) // 2      # 226
ZERO_SLOT  = N_SC_PAD      # 输入 UB 第 1664 个 half = 保证 0 (guard 的 gather 源)

OFDM_OUTPUT_RAW_K1K2 = True   # [约定 A]


# ════════════════════════════════════════════════════════════════
# [约定 B] used SC → natural FFT bin  (与 demap 同序)
# ════════════════════════════════════════════════════════════════
def used_fft_bins():
    """返回 (N_SC_USED,) int64: 输出顺序 u → natural FFT bin n."""
    neg = np.arange(N_FFT - (N_SC_USED // 2), N_FFT)     # 1250..2047  (798)
    dc  = np.array([0], dtype=np.int64)                  # 0           (1)
    pos = np.arange(1, N_SC_USED - N_SC_USED // 2)       # 1..797      (797)
    bins = np.concatenate([neg, dc, pos]).astype(np.int64)
    assert bins.shape[0] == N_SC_USED, bins.shape
    return bins


# ════════════════════════════════════════════════════════════════
# [约定 A] natural FFT bin → ofdm 网格内存 offset (element)
# ════════════════════════════════════════════════════════════════
def bin_to_mem_offset(bins):
    if OFDM_OUTPUT_RAW_K1K2:
        k1 = bins % P
        k2 = bins // P
        return (k1 * Q + k2).astype(np.int64)
    return bins.astype(np.int64)


# ════════════════════════════════════════════════════════════════
# inverse-gather: 输出内存位 m → 输入下标 (used→u; guard→ZERO_SLOT)
# ════════════════════════════════════════════════════════════════
def build_scatter_index():
    bins = used_fft_bins()
    elem = bin_to_mem_offset(bins)                       # (1596,) 每个 used SC 的内存位

    # 自检: 合法置换子集
    assert elem.min() >= 0 and elem.max() < N_FFT, (elem.min(), elem.max())
    assert len(np.unique(elem)) == N_SC_USED, "scatter 目标位有重复"

    inv = np.full(N_FFT, ZERO_SLOT, dtype=np.int64)      # 默认 guard → zero slot
    inv[elem] = np.arange(N_SC_USED, dtype=np.int64)     # used 位 ← input u
    return inv, elem                                     # inv:(2048,)  elem:(1596,)


def scatter_apply(yused, inv):
    """yused: (N_SYMBOL, N_SC_PAD); 末尾补 1 列 0 作 ZERO_SLOT; gather → (N_SYMBOL, N_FFT)."""
    zero_col = np.zeros((N_SYMBOL, 1), dtype=yused.dtype)
    src = np.concatenate([yused, zero_col], axis=1)      # (14, 1665); col 1664 = 0
    return src[:, inv]                                   # (14, 2048)


# ════════════════════════════════════════════════════════════════
# Dump
# ════════════════════════════════════════════════════════════════
def dump(seed=20260601):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    inv, elem = build_scatter_index()

    # kernel Gather 吃 BYTE offset (fp16 = 2 B/elem)
    idx_byte = (inv * 2).astype(np.uint32)
    idx_byte.tofile(f"{WEIGHTS_DIR}/scatter_idx.bin")

    # 模拟 TX 输入 used SC (纯置换, 值无所谓; pad [1596:1664]=0, 反正不被 gather)
    rng = np.random.default_rng(seed)
    in_re = np.zeros((N_SYMBOL, N_SC_PAD), dtype=np.float16)
    in_im = np.zeros((N_SYMBOL, N_SC_PAD), dtype=np.float16)
    in_re[:, :N_SC_USED] = (rng.standard_normal((N_SYMBOL, N_SC_USED)) * 8.0).astype(np.float16)
    in_im[:, :N_SC_USED] = (rng.standard_normal((N_SYMBOL, N_SC_USED)) * 8.0).astype(np.float16)
    in_re.tofile(f"{GOLDEN_DIR}/input_re.bin")
    in_im.tofile(f"{GOLDEN_DIR}/input_im.bin")

    grid_re = scatter_apply(in_re, inv)                  # (14,2048) fp16
    grid_im = scatter_apply(in_im, inv)
    grid_re.astype(np.float16).tofile(f"{GOLDEN_DIR}/grid_re.bin")
    grid_im.astype(np.float16).tofile(f"{GOLDEN_DIR}/grid_im.bin")

    # ── 自检 1: guard 位全 0 ─────────────────────────────────────
    guard_mask = np.ones(N_FFT, dtype=bool)
    guard_mask[elem] = False
    assert np.all(grid_re[:, guard_mask] == 0), "guard 非 0"
    assert np.all(grid_im[:, guard_mask] == 0), "guard 非 0"

    # ── 自检 2: demap(grid) 回环 == 原 used SC (恒等) ────────────
    recovered_re = grid_re[:, elem]                      # (14,1596) demap gather
    recovered_im = grid_im[:, elem]
    assert np.array_equal(recovered_re, in_re[:, :N_SC_USED]), "回环 re 失败"
    assert np.array_equal(recovered_im, in_im[:, :N_SC_USED]), "回环 im 失败"

    print("=" * 64)
    print(" re_map reference — 数据生成 + 自检 (inverse-gather scatter)")
    print("=" * 64)
    print(f"  N_FFT={N_FFT}  N_SC_USED={N_SC_USED}  N_SC_PAD={N_SC_PAD}  guard={GUARD}/{GUARD}")
    print(f"  约定A OFDM_OUTPUT_RAW_K1K2 = {OFDM_OUTPUT_RAW_K1K2}   ZERO_SLOT={ZERO_SLOT}")
    b = used_fft_bins()
    print(f"  约定B used bins[0:3]={b[:3].tolist()} ... [797:800]={b[797:800].tolist()} "
          f"... [-3:]={b[-3:].tolist()}")
    g0 = int(np.where(inv == ZERO_SLOT)[0][0])           # 第一个 guard 内存位
    print(f"  inv[elem[0]]={inv[elem[0]]} (=0)  guard 位 inv[{g0}]={inv[g0]} (=ZERO_SLOT {ZERO_SLOT})")
    print(f"  scatter 目标位 elem[0:3]={elem[:3].tolist()}")
    print(f"\n  weights → {WEIGHTS_DIR}/scatter_idx.bin   ({idx_byte.nbytes} B)")
    print(f"  golden  → {GOLDEN_DIR}/input_re.bin|im.bin, grid_re.bin|im.bin")
    print("\n  ✅ 自检通过 (guard=0 + demap 回环恒等) — 跑 bash run.sh")
    print("=" * 64)


if __name__ == "__main__":
    dump()
