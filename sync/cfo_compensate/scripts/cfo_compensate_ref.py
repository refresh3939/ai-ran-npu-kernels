"""
cfo_compensate reference — 时域 CFO 去旋转 (de-rotate), 5G NR PUSCH RX.

Per-slot 热路径算子. Drop-in BEFORE ofdm_demod:
    输入/输出 = ofdm_demod 输入格式 (int16 IQ 交织 [30720] @61.44M, Q8.8).

跑法:
    cd ~/AI-RAN-NPU/kernels/rx/cfo_compensate
    python3 scripts/cfo_compensate_ref.py

输出 (跟 kernel main.cpp 的 AIRAN_DATA_DIR 期望一致):
    kernels/rx/cfo_compensate/data/golden/
        input.bin       int16 Q8.8 交织 [30720]  (一个 slot, window 后)
        output.bin      int16 Q8.8 交织 [30720]  de-rotated  ← verify 比这个
        params.bin      fp32 [2] = (cfo_hz, phi0)  ← tiling 读
        theta.bin       fp16 [30720]  相位 ramp (debug 板斧 2)
        cos.bin/sin.bin fp16 [30720]  (debug 板斧 2)

────────────────────────────────────────────────────────────────────
契约 (gate 2 — 已钉. tech lead 若不同意, 改这里再重跑):

  符号:   de-rotate = exp(j·θ_n),  θ_n = phi0 + SIGN·2π·cfo·n/Fs,  SIGN = -1.
          即接收基带被认为多了 +cfo 旋转 (exp(+j2π·cfo·n/Fs)), 本算子去掉它.
          ★ SIGN 必须跟 cfo_estimate(L2)/cfo_dmrs(L3) δf 累加符号一致.
            拿 L2 δf 做基准; 实测 USRP 流后若发现反向, 只翻 SIGN 这一处.

  相位连续性: CFO 是跨 slot 连续旋转, 每 slot 不能从 phi0=0 重置.
          host 传 phi0 = 本 slot 第 0 样点的累积起始相位 (rad).
          slot 间连续由 host 维护:
              phi0_{k} = wrap(phi0_{k-1} + SIGN·2π·cfo·N_SLOT/Fs)
          kernel 内 n 用 slot 内相对索引 [0, N_SLOT), phi0 兜住绝对相位.
          → 见底部 test_cross_slot_continuity(): 证明这套规则 == 全流连续 de-rotate.

  range reduction: kernel 内 θ 跨度 = |SIGN·2π·cfo·N_SLOT/Fs|.
          cfo=±45kHz(L1粗残差) → ~141 rad(~22 turns); cfo=±15kHz(L2) → ~47 rad.
          有界且小, lib/math Cos/Sin range-reduce 扛得住. 但 kernel 仍先 wrap 到
          [-π,π] 再喂 Cos/Sin (host 传的 phi0 已 wrap, ramp 自身有界) — 别赌大角.

  Q 格式: int16 Q8.8 (scale=256), 跟 ofdm_demod INPUT_SCALE=1/256 对齐.
          ★ gate 1: 实测 USRP int16 标度前不写死; 此处先用 Q8.8 占位.
────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
from pathlib import Path

# ============================================================
# 路径 — scripts/ 在 kernel 目录下
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_KERNEL_DIR = _SCRIPT_DIR.parent
DEFAULT_GOLDEN_DIR = _KERNEL_DIR / "data" / "golden"

# ============================================================
# 锁定常量 (改前先确认 kernel 同步)
# ============================================================
FS_DATA           = 61_440_000.0   # 数据全速率
N_SAMPLE_PER_SLOT = 30720          # 一个 slot 时域样点 (含 CP)
Q_SCALE           = 256            # Q8.8, 跟 ofdm_demod 对齐
SIGN              = -1.0           # de-rotate 方向; 实测后可翻 (见契约)

TWO_PI = 2.0 * np.pi


# ============================================================
# fp64 精确参考 (golden 源)
# ============================================================
def cfo_compensate_f64(x, cfo_hz, phi0=0.0, n_start=0):
    """x: (..., N) complex. 返回 de-rotated, fp64 精确.
       n_start: slot 内相对样点起点 (契约里 host 永远传 0, phi0 兜绝对相位).
    """
    N = x.shape[-1]
    n = np.arange(n_start, n_start + N, dtype=np.float64)
    theta = phi0 + SIGN * TWO_PI * cfo_hz * n / FS_DATA
    return (x.astype(np.complex128) * np.exp(1j * theta)).astype(np.complex64)


# ============================================================
# fp16-sim 参考 (模拟 kernel 数值路径, 用来预估误差)
#   kernel: 建 fp32 index ramp → wrap mod 2π → Cast half → Cos/Sin(half) → 复乘(half)
# ============================================================
def cfo_compensate_f16sim(x, cfo_hz, phi0=0.0, n_start=0):
    N = x.shape[-1]
    n = np.arange(n_start, n_start + N, dtype=np.float32)
    dtheta = np.float32(SIGN * TWO_PI * cfo_hz / FS_DATA)
    theta = (np.float32(phi0) + n * dtheta).astype(np.float32)     # fp32 ramp
    # range reduction → [-π, π]
    theta_w = (np.mod(theta + np.float32(np.pi), np.float32(TWO_PI))
               - np.float32(np.pi)).astype(np.float32)
    cos_t = np.cos(theta_w).astype(np.float16)
    sin_t = np.sin(theta_w).astype(np.float16)
    # 输入 dequant (int16 Q8.8 → half)
    xr = (np.round(x.real * Q_SCALE).astype(np.int16).astype(np.float16) / Q_SCALE)
    xi = (np.round(x.imag * Q_SCALE).astype(np.int16).astype(np.float16) / Q_SCALE)
    out_r = (xr * cos_t - xi * sin_t).astype(np.float16)
    out_i = (xr * sin_t + xi * cos_t).astype(np.float16)
    return (out_r.astype(np.float32) + 1j * out_i.astype(np.float32)).astype(np.complex64)


# ============================================================
# Q8.8 打包 (跟 ofdm_demod_ref.complex_to_q8_8_int16 一致)
# ============================================================
def complex_to_q8_8_int16(z):
    re = np.round(z.real * Q_SCALE).astype(np.int16)
    im = np.round(z.imag * Q_SCALE).astype(np.int16)
    return np.stack([re, im], axis=-1).reshape(*z.shape[:-1], 2 * z.shape[-1])


# ============================================================
# 自检
# ============================================================
def test_f64_vs_direct(atol=1e-5):
    """phi0=0, n_start=0 时 f64 路径 == 教科书 x·exp(-j2π·cfo·n/Fs)."""
    rng = np.random.default_rng(0)
    N = N_SAMPLE_PER_SLOT
    x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
    cfo = 12345.0
    y = cfo_compensate_f64(x, cfo, phi0=0.0)
    n = np.arange(N)
    y_ref = x * np.exp(SIGN * 1j * TWO_PI * cfo * n / FS_DATA)
    err = np.abs(y - y_ref).max()
    print(f"[f64 vs direct          ] max|err| = {err:.3e}  (tol={atol})")
    assert err < atol, err


def test_cross_slot_continuity(atol=1e-5):
    """★ 契约核心: 两个连续 slot 各自传 (phi0, n_start=0), 拼起来必须 == 全流
       一次性连续 de-rotate. 证明 host 的 phi0 更新规则正确, slot 边界无相位跳变.
    """
    rng = np.random.default_rng(1)
    N = N_SAMPLE_PER_SLOT
    x_full = (rng.standard_normal(2 * N) + 1j * rng.standard_normal(2 * N)).astype(np.complex64)
    cfo = -8700.0

    # 全流连续 (绝对 n = 0..2N)
    y_full = cfo_compensate_f64(x_full, cfo, phi0=0.0, n_start=0)

    # 分 slot: slot0 phi0=0; slot1 phi0 = host 更新规则
    phi0_s0 = 0.0
    y_s0 = cfo_compensate_f64(x_full[:N], cfo, phi0=phi0_s0, n_start=0)

    phi0_s1 = phi0_s0 + SIGN * TWO_PI * cfo * N / FS_DATA       # host 维护
    # wrap 不影响 exp() 结果, 但模拟 kernel 输入
    phi0_s1_wrapped = (phi0_s1 + np.pi) % TWO_PI - np.pi
    y_s1 = cfo_compensate_f64(x_full[N:], cfo, phi0=phi0_s1_wrapped, n_start=0)

    y_split = np.concatenate([y_s0, y_s1])
    err = np.abs(y_full - y_split).max()
    print(f"[cross-slot continuity  ] max|err| = {err:.3e}  (tol={atol})")
    assert err < atol, f"slot 边界相位跳变! err={err}"


def test_int16_quant(slot, cfo, phi0):
    """golden(int16) round-trip + fp16-sim 误差预报 (verify 阈值 ~1-2 LSB)."""
    y_f64 = cfo_compensate_f64(slot, cfo, phi0=phi0)
    y_f16 = cfo_compensate_f16sim(slot, cfo, phi0=phi0)
    g_i16 = complex_to_q8_8_int16(y_f64)
    f_i16 = complex_to_q8_8_int16(y_f16)
    lsb = np.abs(g_i16.astype(np.int32) - f_i16.astype(np.int32)).max()
    print(f"[fp16-sim vs f64 golden ] max|Δ| = {lsb} LSB  (Q8.8; 预期 kernel ~此量级)")
    return y_f64


# ============================================================
# Golden dump
# ============================================================
def dump_goldens(slot, cfo_hz, phi0, out_root=None):
    if out_root is None:
        out_root = str(DEFAULT_GOLDEN_DIR)
    os.makedirs(out_root, exist_ok=True)

    y = cfo_compensate_f64(slot, cfo_hz, phi0=phi0)

    complex_to_q8_8_int16(slot).tofile(f"{out_root}/input.bin")
    complex_to_q8_8_int16(y).tofile(f"{out_root}/output.bin")
    np.array([cfo_hz, phi0], dtype=np.float32).tofile(f"{out_root}/params.bin")

    # swap-trick aux (交错布局, 长度 2N):
    #   input_swap = [i0,r0,i1,r1,...]  (IQ 预对调, int16 Q8.8)
    #   a2 = [c0,c0,c1,c1,...]          (cos 每个复制, fp16)
    #   b2 = [-s0,s0,-s1,s1,...]        (sin 交替符号, fp16; SIGN 已在 cos/sin 里)
    # [SWAP-DEVICE] swap_idx: IQ 对调 byte-offset 表 (kernel 内部 Gather 用, 代替 input_swap).
    #   xswap[2k]=xfull[2k+1](i), xswap[2k+1]=xfull[2k](r). byte offset (half=2B).
    #   每 SUB_TILE=2048 相同 pattern, 长 2*SUB_TILE=4096 uint32.
    SUB_TILE = 2048
    swap_idx = np.zeros(2 * SUB_TILE, dtype=np.uint32)
    for k in range(SUB_TILE):
        swap_idx[2 * k]     = (2 * k + 1) * 2   # i byte offset
        swap_idx[2 * k + 1] = (2 * k) * 2       # r byte offset
    swap_idx.tofile(f"{out_root}/swap_idx.bin")

    n = np.arange(N_SAMPLE_PER_SLOT, dtype=np.float32)
    dth = np.float32(SIGN * TWO_PI * cfo_hz / FS_DATA)
    th = (np.float32(phi0) + n * dth).astype(np.float32)
    thw = (np.mod(th + np.float32(np.pi), np.float32(TWO_PI)) - np.float32(np.pi)).astype(np.float32)
    cos = np.cos(thw).astype(np.float16); sin = np.sin(thw).astype(np.float16)
    a2 = np.empty(2 * N_SAMPLE_PER_SLOT, np.float16); a2[0::2] = cos; a2[1::2] = cos
    b2 = np.empty(2 * N_SAMPLE_PER_SLOT, np.float16); b2[0::2] = -sin; b2[1::2] = sin
    a2.tofile(f"{out_root}/a2.bin")
    b2.tofile(f"{out_root}/b2.bin")

    # debug 板斧 2: 相位 ramp / cos / sin (fp16)
    n = np.arange(N_SAMPLE_PER_SLOT, dtype=np.float32)
    dtheta = np.float32(SIGN * TWO_PI * cfo_hz / FS_DATA)
    theta = (np.float32(phi0) + n * dtheta).astype(np.float32)
    theta_w = (np.mod(theta + np.float32(np.pi), np.float32(TWO_PI))
               - np.float32(np.pi)).astype(np.float32)
    theta_w.astype(np.float16).tofile(f"{out_root}/theta.bin")
    np.cos(theta_w).astype(np.float16).tofile(f"{out_root}/cos.bin")
    np.sin(theta_w).astype(np.float16).tofile(f"{out_root}/sin.bin")

    print(f"\n[dump] goldens → {out_root}/")
    for name in ["input", "swap_idx", "a2", "b2", "output", "params", "theta", "cos", "sin"]:
        f = f"{out_root}/{name}.bin"
        print(f"  {name + '.bin':<14} {os.path.getsize(f) / 1024.0:>7.2f} KiB")
    print(f"\n[dump] tiling 标量: cfo_hz={cfo_hz:.3f}  phi0={phi0:.6f} rad  SIGN={SIGN:+.0f}")


# ============================================================
# Driver
# ============================================================
if __name__ == "__main__":
    print("=" * 64)
    print(" cfo_compensate — Python reference + 自检")
    print("=" * 64)
    print(f"  KERNEL_DIR = {_KERNEL_DIR}")
    print(f"  GOLDEN_DIR = {DEFAULT_GOLDEN_DIR}")
    print(f"  Fs={FS_DATA/1e6:.2f}MHz  N_SLOT={N_SAMPLE_PER_SLOT}  Q8.8  SIGN={SIGN:+.0f}\n")

    test_f64_vs_direct()
    test_cross_slot_continuity()

    # 确定性测试 slot (含一个有代表性的 phi0≠0, 验 kernel 不重置)
    rng = np.random.default_rng(seed=12345)
    slot = ((rng.standard_normal(N_SAMPLE_PER_SLOT)
             + 1j * rng.standard_normal(N_SAMPLE_PER_SLOT))
            * (1.0 / np.sqrt(2))).astype(np.complex64)
    CFO_HZ = 9800.0       # 占位; 真上用 L2 估计
    PHI0   = 0.7853981634 # ≈ π/4, 模拟非首 slot 的累积起始相位

    test_int16_quant(slot, CFO_HZ, PHI0)
    dump_goldens(slot, CFO_HZ, PHI0)

    print("\n" + "=" * 64)
    print(" ✅ Python 自检通过 — 锁契约后翻 Ascend C kernel")
    print("=" * 64)
