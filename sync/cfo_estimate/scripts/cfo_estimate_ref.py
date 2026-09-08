#!/usr/bin/env python3
# ============================================================================
# cfo_estimate_ref.py  —  CP-based CFO estimator (Moose 1994) reference
#
# 位置(项目内): kernels/rx/cfo_estimate/scripts/cfo_estimate_ref.py
#
# 作用:
#   1. 合成 5G NR μ=1 时域 IQ 信号 (1 slot, 14 OFDM symbols, 含 CP)
#   2. 注入已知 CFO Δf (Hz)
#   3. 量化到 int16 Q8.8 (匹配 ofdm_demod 输入格式 / 未来 SDR ADC 输出)
#   4. 用 CP autocorrelation (Moose) 算法估出 Δf
#   5. 验证 估出值 vs 真值
#   6. 写 golden:input.bin / truth.bin / output.bin / meta.json
#
# 算法(srsRAN sync.c 实现,产品级):
#   per symbol s:  r[s] = Σ_{n=0..CP-1} conj(x[cp_start+n]) · x[cp_start+CP+n]
#   R = Σ_s r[s]                              (跨 symbol 累加, 提升 SNR)
#   Δf_norm = -angle(R) / (2π)                (归一化到 [-0.5, 0.5] 子载波)
#   Δf_hz   = Δf_norm · SCS_HZ                (SCS=30 kHz @ μ=1)
#
#   注:  N_useful=2048, 一个 sample 时长 T_s = 1/SAMPLE_RATE.
#         x[n+N_useful] / x[n] = exp(j·2π·Δf·N_useful·T_s) = exp(j·2π·Δf/SCS).
#         所以 angle(R) = 2π · Δf / SCS,反解即上式。
#
# 输出文件(写到 <kernel_dir>/data/golden/case_<id>/):
#   input.bin       int16  (N_SAMPLE_PER_SLOT * 2,)   IQ 交错,Q8.8 量化,含 CP
#   truth.bin       float32 (1,)                       注入的 Δf (Hz)
#   output.bin      float32 (1,)                       Python ref 估出的 Δf (Hz, int16 路径)
#   meta.json       dict                               SCS / N_FFT / CP_LENS 等
#
# 用法:
#   python3 cfo_estimate_ref.py [output_dir]
#   默认 output_dir = <kernel_dir>/data/golden/
#   通过 AIRAN_DATA_DIR=<kernel_dir> 环境变量可覆盖 kernel 根
#
# 命令行 flag:
#   --no-verify   只生成 golden,不打印验证报告
#   --seed N      随机数种子(默认 0xC0FE0001)
# ============================================================================

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# 5G NR μ=1 参数(锁定值,与 SKILL.md §2 / ofdm_demod 一致)
# ----------------------------------------------------------------------------
N_FFT             = 2048
SCS_HZ            = 30_000                       # μ=1
SAMPLE_RATE_HZ    = N_FFT * SCS_HZ               # 61_440_000
N_SYMBOL_PER_SLOT = 14
CP_LEN_FIRST      = 176
CP_LEN_OTHER      = 144
N_SAMPLE_PER_SLOT = 30_720                       # 176 + 2048 + 13*(144+2048)
N_RB              = 133
N_SC_USED         = 1596                         # 133 RB × 12 SC
Q_SCALE           = 256                          # Q8.8
INT16_MAX         = 32767

CP_LENS = np.array(
    [CP_LEN_FIRST] + [CP_LEN_OTHER] * (N_SYMBOL_PER_SLOT - 1),
    dtype=np.int32,
)
assert int(CP_LENS.sum()) + N_SYMBOL_PER_SLOT * N_FFT == N_SAMPLE_PER_SLOT, \
    "CP 长度与 slot sample 数不匹配"


# ----------------------------------------------------------------------------
# 测试 case:覆盖 0 / 小 / 中 / 大 / 极限 / 负 / 正
# 单位:Hz。Unambiguous range ±SCS/2 = ±15000 Hz。
# ----------------------------------------------------------------------------
TEST_CASES_HZ = [
    0.0,        # 零频偏 sanity
    +500.0,     # 典型 LO drift
    -1200.0,    # 中等 + 反向
    +5000.0,    # 大频偏(约 SCS 的 1/6)
    -10000.0,   # 接近 unambiguous 边界的 2/3
]


# ----------------------------------------------------------------------------
# Step 1 — 合成时域 OFDM 信号(1 slot)
# ----------------------------------------------------------------------------
def synth_ofdm_slot_time_domain(rng: np.random.Generator) -> np.ndarray:
    """
    生成 1 slot 的复数时域 IQ 信号,**含 CP**。

    每个 OFDM symbol:
      - 在 freq domain 填 N_SC_USED 个随机 QPSK 子载波(其余 N_FFT-N_SC_USED 个为 0)
      - DC 子载波(index 0)留空(5G NR 实际有 guard,这里简化)
      - 子载波映射到 fftshift 位置(DC 在 N_FFT/2)
      - 反 fftshift,IFFT(归一化到单位平均功率)
      - 前置 CP(symbol 0 的 CP=176,其他=144)

    返回: shape (N_SAMPLE_PER_SLOT,) complex128
    """
    # QPSK constellation: ±1/√2 ± j/√2,单位功率
    qpsk = (np.array([+1, -1, +1, -1], dtype=np.float64) +
            1j * np.array([+1, +1, -1, -1], dtype=np.float64)) / np.sqrt(2.0)

    out = np.zeros(N_SAMPLE_PER_SLOT, dtype=np.complex128)
    cursor = 0
    for s in range(N_SYMBOL_PER_SLOT):
        # ---- freq-domain symbol(fftshift 视图,DC 在中间)----
        x_freq_shift = np.zeros(N_FFT, dtype=np.complex128)
        center = N_FFT // 2
        # 把 N_SC_USED 个子载波放在中间,跳过 DC
        half = N_SC_USED // 2
        idx_neg = slice(center - half,     center)              # 798 个
        idx_pos = slice(center + 1,        center + 1 + half)   # 798 个
        sym_neg = qpsk[rng.integers(0, 4, size=half)]
        sym_pos = qpsk[rng.integers(0, 4, size=half)]
        x_freq_shift[idx_neg] = sym_neg
        x_freq_shift[idx_pos] = sym_pos

        # ---- 反 fftshift, IFFT ----
        x_freq = np.fft.ifftshift(x_freq_shift)
        # numpy ifft 公式: x[n] = (1/N) Σ X[k] exp(j 2π kn/N)
        # 若 N_SC_USED 个子载波各单位功率, |x[n]|² 期望 = N_SC_USED / N²
        # 要让时域平均功率 ≈ 1, 乘以 N/sqrt(N_SC_USED):
        #   缩放后 |x|² 期望 = (N/sqrt(Nused))² · (Nused/N²) = 1
        x_time = np.fft.ifft(x_freq) * (N_FFT / np.sqrt(N_SC_USED))

        # ---- 加 CP(取 useful 末尾 CP_LEN 个 sample 前置)----
        cp_len = CP_LENS[s]
        cp = x_time[-cp_len:]
        sym_with_cp = np.concatenate([cp, x_time])
        assert sym_with_cp.shape[0] == cp_len + N_FFT

        out[cursor:cursor + cp_len + N_FFT] = sym_with_cp
        cursor += cp_len + N_FFT

    assert cursor == N_SAMPLE_PER_SLOT, f"cursor={cursor} vs {N_SAMPLE_PER_SLOT}"

    # Sanity:平均功率应约 1.0
    avg_power = float(np.mean(np.abs(out) ** 2))
    assert 0.5 < avg_power < 2.0, f"平均功率异常: {avg_power:.3f}"

    return out


# ----------------------------------------------------------------------------
# Step 2 — 注入 CFO
# ----------------------------------------------------------------------------
def inject_cfo(x: np.ndarray, delta_f_hz: float) -> np.ndarray:
    """
    时域注入 CFO: x'[n] = x[n] · exp(j·2π·Δf·n·T_s)
    其中 T_s = 1 / SAMPLE_RATE_HZ
    """
    n = np.arange(x.shape[0], dtype=np.float64)
    phase = 2.0 * np.pi * delta_f_hz * n / SAMPLE_RATE_HZ
    rot = np.exp(1j * phase)
    return x * rot


# ----------------------------------------------------------------------------
# Step 3 — int16 Q8.8 量化(I 和 Q 交错,匹配 ofdm_demod 输入格式)
# ----------------------------------------------------------------------------
def quantize_to_int16(x_complex: np.ndarray) -> np.ndarray:
    """
    输入 complex (N,) -> 输出 int16 (2N,) [I0,Q0,I1,Q1,...]
    Q8.8 量化:value_int16 = round(value_float * 256), clip ±32767
    """
    iq = np.empty(x_complex.shape[0] * 2, dtype=np.float64)
    iq[0::2] = x_complex.real
    iq[1::2] = x_complex.imag
    scaled = np.round(iq * Q_SCALE)
    clipped = np.clip(scaled, -INT16_MAX, INT16_MAX).astype(np.int16)
    return clipped


def dequantize_from_int16(iq_int16: np.ndarray) -> np.ndarray:
    """int16 交错 IQ -> complex128"""
    iq_f = iq_int16.astype(np.float64) / Q_SCALE
    return iq_f[0::2] + 1j * iq_f[1::2]


# ----------------------------------------------------------------------------
# Step 4 — CP-based CFO 估计(Moose 算法)
# ----------------------------------------------------------------------------
def estimate_cfo_moose(x: np.ndarray) -> float:
    """
    输入: 1 slot 时域 IQ (complex, 含 CP), shape (N_SAMPLE_PER_SLOT,)
    输出: 估计的 Δf (Hz)

    算法:
      per symbol s, 在 CP 窗口内对 x 和 x+N_FFT 做共轭相关
        r[s] = Σ_{n=0..CP_LEN[s]-1} conj(x[cp_start+n]) * x[cp_start+CP+n]
      跨 symbol 累加: R = Σ_s r[s]
      Δf_norm = angle(R) / (2π)           注意符号:见下
      Δf_hz   = Δf_norm * SCS_HZ

    符号约定:
      注入是 x'[n] = x[n] · exp(j·2π·Δf·n·T_s)
      所以 x'[n+N] / x'[n] = exp(j·2π·Δf·N·T_s) = exp(j·2π·Δf/SCS)
      故 conj(x'[n]) · x'[n+N] 的 angle = 2π·Δf/SCS  → Δf = angle(R)/(2π) · SCS
    """
    cumsum_offsets = np.concatenate([[0], np.cumsum(CP_LENS + N_FFT)])  # (15,)

    R = 0.0 + 0.0j
    for s in range(N_SYMBOL_PER_SLOT):
        cp_start = int(cumsum_offsets[s])
        cp_len   = int(CP_LENS[s])
        # 注意:srsRAN/Moose 用 conj(x[cp]) · x[cp + N_useful],N_useful=N_FFT
        a = x[cp_start          : cp_start + cp_len]
        b = x[cp_start + N_FFT  : cp_start + N_FFT + cp_len]
        r_s = np.sum(np.conj(a) * b)
        R += r_s

    delta_f_normalized = np.angle(R) / (2.0 * np.pi)   # in subcarrier units
    delta_f_hz = delta_f_normalized * SCS_HZ
    return float(delta_f_hz)


# ----------------------------------------------------------------------------
# Step 4b — NPU 模拟路径(精确镜像 Ascend C kernel 的 op chain)
# ----------------------------------------------------------------------------
def cp_autocorr_npu(x_int16_interleaved: np.ndarray):
    """
    模拟 NPU kernel Phase 1:CP autocorrelation,跨 symbol 累加。

    输入: int16 IQ 交错 (shape (N_SAMPLE_PER_SLOT*2,)) - 来自 input.bin
    输出: (R_re, R_im) 两个 fp32 标量

    NPU kernel 实际会做:
      - 读 int16 IQ, Cast 到 fp32 (Q8.8 反量化: /256)
      - per symbol s, 对 CP 窗口做共轭乘累加(complex MAC = 4 个实数 MAC)
        re += a_re*b_re + a_im*b_im   (= conj(a).re * b)
        im += a_re*b_im - a_im*b_re   (= conj(a).im * b)
      - 跨 symbol 累加得到 R

    Python 这里用 fp32 数学,精度等价 NPU fp32 路径(差异 ≤ 1-2 ULP)
    """
    # int16 -> fp32, Q8.8 反量化
    iq_f32 = (x_int16_interleaved.astype(np.float32) / np.float32(Q_SCALE))
    x_re = iq_f32[0::2]    # I
    x_im = iq_f32[1::2]    # Q

    cumsum_offsets = np.concatenate([[0], np.cumsum(CP_LENS + N_FFT)])

    R_re = np.float32(0.0)
    R_im = np.float32(0.0)
    for s in range(N_SYMBOL_PER_SLOT):
        cp_start = int(cumsum_offsets[s])
        cp_len   = int(CP_LENS[s])
        a_re = x_re[cp_start          : cp_start + cp_len]
        a_im = x_im[cp_start          : cp_start + cp_len]
        b_re = x_re[cp_start + N_FFT  : cp_start + N_FFT + cp_len]
        b_im = x_im[cp_start + N_FFT  : cp_start + N_FFT + cp_len]
        # conj(a) * b = (a_re - j a_im)(b_re + j b_im)
        #             = (a_re*b_re + a_im*b_im) + j(a_re*b_im - a_im*b_re)
        R_re += np.float32(np.sum(a_re * b_re + a_im * b_im))
        R_im += np.float32(np.sum(a_re * b_im - a_im * b_re))

    return float(R_re), float(R_im)


def atan2_npu(b: float, a: float) -> float:
    """
    模拟 NPU kernel Phase 2:用单参数 Atan(t) 拼出 atan2(b, a) ∈ [-π, π]。

    NPU op 链(每行对应 1 个 Ascend C basic API 调用):
        A = Abs(a)                   # Abs (fp32)
        B = Abs(b)
        num = Min(A, B)              # Min
        den = Max(A, B)              # Max
        den = Maxs(den, 1e-30)       # 防 div-by-zero (Maxs scalar)
        t = num / den                # Div, t ∈ [0, 1]
        phi = Atan(t)                # 高阶 API, phi ∈ [0, π/4]

        # 象限还原(单元素标量逻辑,不是热路径)
        theta_q1 = phi   if A >= B   else  (π/2 - phi)
        theta    = theta_q1                 if a >= 0
                   else (π - theta_q1)

        # 处理 b 的符号
        if b < 0:  theta = -theta

    返回: θ ∈ [-π, π]
    """
    # 用 fp32 算,镜像 NPU
    a_f = np.float32(a)
    b_f = np.float32(b)

    A = np.float32(abs(a_f))
    B = np.float32(abs(b_f))
    num = min(A, B)
    den = max(A, B)
    den = max(den, np.float32(1e-30))   # Maxs 防 div-by-zero
    t = np.float32(num) / np.float32(den)

    # Atan(t):用 np.arctan 模拟(NPU 内部是分段泰勒,误差极小,1e-5 量级)
    phi = np.float32(np.arctan(t))

    # 象限还原
    if A >= B:
        theta_q1 = phi
    else:
        theta_q1 = np.float32(np.pi / 2.0) - phi

    if a_f >= 0:
        theta = theta_q1
    else:
        theta = np.float32(np.pi) - theta_q1

    if b_f < 0:
        theta = -theta

    return float(theta)


def estimate_cfo_npu(x_int16_interleaved: np.ndarray) -> tuple:
    """
    完整 NPU 模拟路径:int16 IQ → CP autocorr → atan2 拼装 → Δf

    返回: (delta_f_hz, R_re, R_im, theta_rad)
           ↑ 主输出       ↑ 给 kernel 调试用的中间量
    """
    R_re, R_im = cp_autocorr_npu(x_int16_interleaved)
    theta = atan2_npu(R_im, R_re)
    # Δf_hz = θ / (2π) · SCS
    delta_f_hz = float(np.float32(theta) / np.float32(2.0 * np.pi) * np.float32(SCS_HZ))
    return delta_f_hz, R_re, R_im, theta


# ----------------------------------------------------------------------------
# 路径解析(项目布局对齐)
# ----------------------------------------------------------------------------
def resolve_paths(cli_out: str | None):
    """
    解析输出根目录。优先级:
      1. CLI 参数
      2. AIRAN_DATA_DIR 环境变量(指向 kernel 目录, 自动拼 data/golden)
      3. 默认: <kernel_dir>/data/golden/
         (从 __file__ 推:scripts/ 上去一层是 kernels/rx/cfo_estimate/)
    """
    if cli_out:
        return Path(cli_out)
    env = os.environ.get("AIRAN_DATA_DIR")
    if env:
        return Path(env) / "data" / "golden"
    here = Path(__file__).resolve()
    # here = .../kernels/rx/cfo_estimate/scripts/cfo_estimate_ref.py
    # parents[0]=scripts/, [1]=cfo_estimate/
    kernel_dir = here.parents[1]
    return kernel_dir / "data" / "golden"


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="CP-based CFO estimator reference + golden generator"
    )
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="输出根目录(默认: data/golden/rx/cfo_estimate/)")
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0xC0FE0001,
                        help="随机种子(默认 0xC0FE0001)")
    parser.add_argument("--no-verify", action="store_true",
                        help="只生成 golden,不打印验证报告")
    args = parser.parse_args()

    out_root = resolve_paths(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[ref] 输出根: {out_root}")
    print(f"[ref] 种子:   0x{args.seed:08X}")
    print(f"[ref] SCS:    {SCS_HZ} Hz, sample rate: {SAMPLE_RATE_HZ} Hz")
    print(f"[ref] Unambiguous range: ±{SCS_HZ/2:.0f} Hz")
    print()

    rng = np.random.default_rng(args.seed)

    # 一次合成,多 case 复用同一 OFDM 信号(仅 CFO 不同)— 便于对比
    clean_signal = synth_ofdm_slot_time_domain(rng)
    print(f"[synth] 合成 1 slot 时域 IQ: shape={clean_signal.shape}, "
          f"avg_power={np.mean(np.abs(clean_signal)**2):.3f}")

    results = []
    for case_id, delta_f_hz in enumerate(TEST_CASES_HZ):
        case_dir = out_root / f"case_{case_id}_cfo_{int(delta_f_hz):+d}hz"
        case_dir.mkdir(parents=True, exist_ok=True)

        # --- 注入 CFO ---
        rx_complex = inject_cfo(clean_signal, delta_f_hz)

        # --- 浮点路径估计(sanity check 算法本身)---
        est_f64 = estimate_cfo_moose(rx_complex)

        # --- int16 量化(NPU 输入路径)---
        iq_int16 = quantize_to_int16(rx_complex)
        rx_dequant = dequantize_from_int16(iq_int16)
        est_int16  = estimate_cfo_moose(rx_dequant)

        # --- NPU op chain 模拟(精确镜像 Ascend C kernel)---
        # 这一路输入是 int16(和 NPU 一样),内部全程 fp32(和 NPU 一样),
        # atan2 用单参数 Atan + 象限还原(和 NPU 一样)
        est_npu, R_re, R_im, theta_rad = estimate_cfo_npu(iq_int16)

        # --- 写文件 ---
        # output.bin 现在是 NPU 路径估值(kernel 输出与此对齐, ≤2 ULP)
        iq_int16.tofile(case_dir / "input.bin")
        np.array([delta_f_hz], dtype=np.float32).tofile(case_dir / "truth.bin")
        np.array([est_npu],    dtype=np.float32).tofile(case_dir / "output.bin")
        # 额外把中间量也存,kernel 调试时方便分阶段对比
        np.array([R_re, R_im], dtype=np.float32).tofile(case_dir / "R_complex.bin")
        np.array([theta_rad],  dtype=np.float32).tofile(case_dir / "theta.bin")

        meta = {
            "case_id":                  case_id,
            "injected_cfo_hz":          float(delta_f_hz),
            "estimated_cfo_hz_float":   float(est_f64),
            "estimated_cfo_hz_int16":   float(est_int16),
            "estimated_cfo_hz_npu":     float(est_npu),
            "R_re":                     float(R_re),
            "R_im":                     float(R_im),
            "theta_rad":                float(theta_rad),
            "abs_error_float_hz":       float(abs(est_f64  - delta_f_hz)),
            "abs_error_int16_hz":       float(abs(est_int16 - delta_f_hz)),
            "abs_error_npu_hz":         float(abs(est_npu  - delta_f_hz)),
            "fft_size":                 N_FFT,
            "scs_hz":                   SCS_HZ,
            "sample_rate_hz":           SAMPLE_RATE_HZ,
            "n_symbol_per_slot":        N_SYMBOL_PER_SLOT,
            "n_sample_per_slot":        N_SAMPLE_PER_SLOT,
            "cp_lens":                  CP_LENS.tolist(),
            "q_scale":                  Q_SCALE,
            "input_dtype":              "int16",
            "input_layout":             "interleaved_iq",
            "input_shape":              [N_SAMPLE_PER_SLOT * 2],
            "unambiguous_range_hz":     [-SCS_HZ // 2, SCS_HZ // 2],
            "seed":                     f"0x{args.seed:08X}",
        }
        with open(case_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        results.append(meta)

    # ------------------------------------------------------------
    # 验证报告
    # ------------------------------------------------------------
    if not args.no_verify:
        print()
        print("=" * 92)
        print(" 验证报告 — 注入 vs 估计 (3 条路径)")
        print(" path A: numpy complex128 全浮点(算法 sanity)")
        print(" path B: numpy + int16 量化(量化噪声)")
        print(" path C: NPU 模拟 - int16 + fp32 op chain + Atan 拼 atan2(kernel golden)")
        print("=" * 92)
        print(f"{'case':<5} {'inject(Hz)':>11} "
              f"{'A:est':>11} {'A:err':>9} "
              f"{'B:est':>13} {'B:err':>9} "
              f"{'C:est_NPU':>13} {'C:err':>9}")
        print("-" * 92)
        all_pass = True
        for r in results:
            inj   = r["injected_cfo_hz"]
            ef64  = r["estimated_cfo_hz_float"]
            errf  = r["abs_error_float_hz"]
            ei16  = r["estimated_cfo_hz_int16"]
            erri  = r["abs_error_int16_hz"]
            enpu  = r["estimated_cfo_hz_npu"]
            errn  = r["abs_error_npu_hz"]

            # 验证标准 —— 三层
            # A 浮点:    < 1 Hz       (纯算法 sanity)
            # B int16:   max(50 Hz, 5% × |truth|)  (量化噪声)
            # C NPU路径: max(50 Hz, 5% × |truth|)  (kernel 应能达到)
            tol_int16 = max(50.0, 0.05 * abs(inj))
            tol_npu   = max(50.0, 0.05 * abs(inj))
            f64_ok = errf < 1.0
            i16_ok = erri < tol_int16
            npu_ok = errn < tol_npu
            ok = f64_ok and i16_ok and npu_ok
            mark = "✓" if ok else "✗"
            print(f"{r['case_id']:<5} {inj:>11.2f} "
                  f"{ef64:>11.4f} {errf:>9.4f} "
                  f"{ei16:>13.4f} {erri:>9.4f} "
                  f"{enpu:>13.4f} {errn:>9.4f}  {mark}")
            if not ok:
                all_pass = False
                if not f64_ok:
                    print(f"       ↑ A: 浮点误差超 1 Hz, 算法可能有问题")
                if not i16_ok:
                    print(f"       ↑ B: int16 误差超阈值 {tol_int16:.1f} Hz")
                if not npu_ok:
                    print(f"       ↑ C: NPU 路径误差超阈值 {tol_npu:.1f} Hz")
        print("=" * 92)
        if all_pass:
            print("结果: 全部 PASS — 三条路径互相印证,NPU kernel 可以照 path C 翻译")
        else:
            print("结果: 有 FAIL,见上↑")
            sys.exit(1)

    print()
    print(f"[ref] 已写 {len(results)} 个 case → {out_root}/")
    for r in results:
        print(f"       case_{r['case_id']}: Δf={r['injected_cfo_hz']:+.0f} Hz")


if __name__ == "__main__":
    main()