"""
re_demap reference — RE 解映射 (resource demapper), RX 链入口.

职责: 从 ofdm_demod 输出的全 FFT bin 网格里抽出 1596 个 used SC.
本算子 **无算术**, 纯置换 (gather) → kernel vs golden 必须 TOL=0 bit-exact.

跑法:
    cd ~/AI-RAN-NPU/kernels/rx/re_demap
    python3 scripts/re_demap_ref.py

输出:
    weights/gather_idx.bin              (uint32, N_SC_PAD, BYTE offset, kernel Gather 用)
    data/golden/input_re.bin/im.bin     (fp16, [14,2048], 模拟 ofdm 输出, kernel 入)
    data/golden/yused_re.bin/im.bin     (fp16, [14,1600], golden)

═══════════════════════════════════════════════════════════════════════════
两条约定 (gather 表正确性的全部依赖) — 改这里, kernel 不动:
═══════════════════════════════════════════════════════════════════════════

[约定 A] OFDM 输出内存排布
    你上传的 ofdm_demod_kernel.cpp 写的是 raw stage4 (没做 Stage5 reshape, 没 fftshift):
        out[sym*2048 + m] = X2[k1, k2],  k1 = m // 64,  k2 = m % 64
    Cooley-Tukey Stage5 约定: natural FFT bin  n = k1 + 32*k2
        => k1 = n % 32,  k2 = n // 32
        => 内存 offset(n) = k1*64 + k2 = (n % 32)*64 + (n // 32)
    若以后 ofdm 改成输出 natural-order(已 fftshift), 把 OFDM_OUTPUT_RAW_K1K2=False.

[约定 B] 1596 used SC 怎么从 2048 里选 (★ Task 0: 待 Sionna PDSCH config 确认)
    N_FFT=2048, N_SC_USED=1596 = 133 PRB × 12.
    2048 = 226 + 1596 + 226  → 保护带 226/226 整数 → 推定 **无 DC null**(PDSCH DC 不打孔).
    居中布局, 输出顺序 = fftshift 序 (grid SC 0 = 最负频):
        负频 798 个: natural bin 1250..2047   (SC -798..-1)
        DC      1 个: natural bin 0
        正频 797 个: natural bin 1..797        (SC +1..+797)
    => used_fft_bins = [1250..2047, 0, 1..797], 共 1596.
    若 Sionna 配置不同(DC null / 非对称 guard / 半 SC 偏移), 只改 used_fft_bins().
"""

import os
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_KERNEL_DIR = _SCRIPT_DIR.parent
WEIGHTS_DIR = _KERNEL_DIR / "weights"
GOLDEN_DIR  = _KERNEL_DIR / "data" / "golden"

# ── 锁定常量 ─────────────────────────────────────────────────────
N_FFT      = 2048
N_SYMBOL   = 14
N_SC_USED  = 1596
N_SC_PAD   = 1664          # 13*128, 128-align (下游 equalizer); 1596 有效 + 68 pad
P          = 32            # Cooley-Tukey 内 DFT
Q          = 64            # Cooley-Tukey 外 DFT
GUARD      = (N_FFT - N_SC_USED) // 2      # 226

OFDM_OUTPUT_RAW_K1K2 = True   # [约定 A] 见上


# ════════════════════════════════════════════════════════════════
# [约定 B] used SC → natural FFT bin  (★ Task 0 待确认)
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
# [约定 A] natural FFT bin → ofdm 输出内存 offset (element)
# ════════════════════════════════════════════════════════════════
def bin_to_mem_offset(bins):
    if OFDM_OUTPUT_RAW_K1K2:
        k1 = bins % P
        k2 = bins // P
        return (k1 * Q + k2).astype(np.int64)
    return bins.astype(np.int64)   # 已 natural-order 时直接用 bin


def build_gather_index():
    """idx_elem[u] = ofdm 输出里第 u 个 used SC 所在 element offset.
    pad 到 N_SC_PAD: [N_SC_USED:] = 0 (指向 element 0), golden 同样复制 → 仍 bit-exact."""
    bins = used_fft_bins()
    elem = bin_to_mem_offset(bins)                       # (1596,)

    # 自检: 合法置换子集
    assert elem.min() >= 0 and elem.max() < N_FFT, (elem.min(), elem.max())
    assert len(np.unique(elem)) == N_SC_USED, "gather 索引有重复"

    idx_elem = np.zeros(N_SC_PAD, dtype=np.int64)
    idx_elem[:N_SC_USED] = elem                          # [1596:] = 0 (pad → elem 0)
    return idx_elem


def gather_apply(grid, idx_elem):
    """grid: (N_SYMBOL, N_FFT); 返回 (N_SYMBOL, N_SC_PAD). 纯置换."""
    return grid[:, idx_elem]


# ════════════════════════════════════════════════════════════════
# Dump
# ════════════════════════════════════════════════════════════════
def dump(seed=20260601):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    idx_elem = build_gather_index()

    # kernel Gather 吃 BYTE offset (fp16 = 2 bytes/elem); 逐 sym 表 (v1 kernel)
    idx_byte = (idx_elem * 2).astype(np.uint32)
    idx_byte.tofile(f"{WEIGHTS_DIR}/gather_idx.bin")

    # 模拟 ofdm 输出 (纯置换, 用什么值无所谓; fp16 保证搬运 bit-exact)
    rng = np.random.default_rng(seed)
    grid_re = (rng.standard_normal((N_SYMBOL, N_FFT)) * 8.0).astype(np.float16)
    grid_im = (rng.standard_normal((N_SYMBOL, N_FFT)) * 8.0).astype(np.float16)
    grid_re.tofile(f"{GOLDEN_DIR}/input_re.bin")
    grid_im.tofile(f"{GOLDEN_DIR}/input_im.bin")

    yused_re = gather_apply(grid_re, idx_elem)           # (14,1600) fp16
    yused_im = gather_apply(grid_im, idx_elem)
    yused_re.astype(np.float16).tofile(f"{GOLDEN_DIR}/yused_re.bin")
    yused_im.astype(np.float16).tofile(f"{GOLDEN_DIR}/yused_im.bin")

    # 二次自检: golden == 手工 gather (恒等, 但确认 dtype/shape)
    assert np.array_equal(yused_re, grid_re[:, idx_elem])
    assert np.array_equal(yused_im, grid_im[:, idx_elem])

    print("=" * 64)
    print(" re_demap reference — 数据生成 + 自检")
    print("=" * 64)
    print(f"  N_FFT={N_FFT}  N_SC_USED={N_SC_USED}  N_SC_PAD={N_SC_PAD}  guard={GUARD}/{GUARD}")
    print(f"  约定A OFDM_OUTPUT_RAW_K1K2 = {OFDM_OUTPUT_RAW_K1K2}")
    b = used_fft_bins()
    print(f"  约定B used bins[0:3]={b[:3].tolist()} ... [797:800]={b[797:800].tolist()} "
          f"... [-3:]={b[-3:].tolist()}")
    print(f"  idx_elem[0:3]={idx_elem[:3].tolist()}  pad[1596:1600]={idx_elem[1596:1600].tolist()}")
    print(f"\n  weights → {WEIGHTS_DIR}/gather_idx.bin   ({idx_byte.nbytes} B)")
    print(f"  golden  → {GOLDEN_DIR}/input_re.bin|im.bin, yused_re.bin|im.bin")
    print("\n  ✅ 自检通过 — 跑 bash run.sh")
    print("=" * 64)


if __name__ == "__main__":
    dump()