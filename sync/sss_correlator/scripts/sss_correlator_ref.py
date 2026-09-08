#!/usr/bin/env python3
"""
sss_correlator_ref.py — SSS Correlator Python Reference

5G NR Secondary Synchronization Signal Correlator, working at 7.68 MSPS
in the same coordinate system as PSS Correlator.

References:
    3GPP TS 38.211 v15.x §7.4.2.3 (SSS sequence)
    Wang & Berggren, IEEE ICC 2018 (DOI 10.1109/ICC.2018.8422145)
    Tuninato et al., EURASIP JWCN 2023 (DOI 10.1186/s13638-023-02317-5)

Clean Room: 不参考 MATLAB / OAI / srsRAN / py3gpp 源码.

输入 (运行时):
    rx           : cint16 量化, 长度 N_SEARCH=153600 @ 7.68 MSPS
    mu_t         : int, PSS 输出, SSB sym0 useful 部分起点
    n_id_2       : int ∈ {0,1,2}, PSS 输出
    g_hat        : int ∈ {-1,0,+1}, PSS 输出
    sss_ref_table: [3, 336, 127] cint16, 离线常量

输出 (per case):
    n_id_1       : int ∈ [0, 336)
    pcid         : int = 3*n_id_1 + n_id_2
    peak         : float, 最大相关值平方
    second       : float, 第二大相关值平方 (Wang-Berggren 双阈值)
    refined_mu_sss: int, refined SSB sym 2 起点 (= sym_2_start_nominal + tau*)
    tau_star     : int ∈ [-T_HALF, +T_HALF], 最佳 tau 偏移

测试: 8 个 case, 各自 synth 完整 SSB 信号. 落 golden 到
      AIRAN_DATA_DIR/golden/rx/sss_correlator/

作者: Refresh, AI-RAN-NPU
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np


# =============================================================================
# 锁定常量 (跟 PSS Correlator 一致, 7.68 MSPS 坐标系)
# =============================================================================

# 采样率 / 帧结构
FS_HZ           = 7_680_000           # 7.68 MSPS
SCS_HZ          = 30_000              # subcarrier spacing
N_FFT           = 256                 # SSB 频段 FFT 长度
N_SC_SSB        = 240                 # SSB 占 20 RB × 12
N_SC_PSS        = 127                 # PSS / SSS 都占 SSB 中心 127 SC

# CP 长度 @ 7.68 MSPS (由 61.44 域缩放: 176/8=22→44 是因为 SCS=30kHz μ=1
# 双倍 sym/slot, 实际 CP_FIRST_768=44, CP_OTHER_768=36, 跟 PSS Correlator 一致)
CP_FIRST_768    = 44
CP_OTHER_768    = 36

# SSB 4 个 OFDM 符号布局 @ 7.68 MSPS
#   sym 0: CP_FIRST + N_FFT  = 44 + 256 = 300  (PSS)
#   sym 1: CP_OTHER + N_FFT  = 36 + 256 = 292  (PBCH)
#   sym 2: CP_OTHER + N_FFT  = 36 + 256 = 292  (SSS)
#   sym 3: CP_OTHER + N_FFT  = 36 + 256 = 292  (PBCH)
SYM0_LEN        = CP_FIRST_768 + N_FFT   # 300
SYM_OTHER_LEN   = CP_OTHER_768 + N_FFT   # 292
N_SSB_SAMPLES   = SYM0_LEN + 3 * SYM_OTHER_LEN  # 300 + 876 = 1176

# 搜索窗口
N_SEARCH        = 153_600             # 20 ms @ 7.68 MSPS

# SSS 候选数 / sym 2 内 useful 起点偏移
N_ID_1_COUNT    = 336
N_ID_2_COUNT    = 3

# Joint search 半窗
T_HALF          = 4                   # tau ∈ [-4, +4], 9 个位置
N_TAU           = 2 * T_HALF + 1      # 9

# 量化 (cint16 Q2.13 - 跟 PSS 一致)
Q_FRAC_BITS     = 13
Q_SCALE         = 1 << Q_FRAC_BITS    # 8192
Q_INV           = 1.0 / Q_SCALE
INT16_MAX       = 32767
INT16_MIN       = -32768

# 数据目录
PROJECT_ROOT    = Path(os.environ.get(
    "AIRAN_DATA_DIR",
    str(Path.home() / "AI-RAN-NPU" / "data")
))
GOLDEN_DIR      = PROJECT_ROOT / "golden" / "rx" / "sss_correlator"


# =============================================================================
# 3GPP TS 38.211 §7.4.2.2  PSS 频域序列 (合成 SSB 用)
# =============================================================================

def gen_pss_sequence(n_id_2: int) -> np.ndarray:
    """生成长度 127 的 PSS BPSK 序列 (TS 38.211 §7.4.2.2.1)."""
    x = np.zeros(127 + 7, dtype=np.int8)
    x[0:7] = [0, 1, 1, 0, 1, 1, 1]
    for i in range(120):
        x[i + 7] = (x[i + 4] + x[i]) % 2
    d = np.zeros(127, dtype=np.float64)
    for n in range(127):
        d[n] = 1.0 - 2.0 * x[(n + 43 * n_id_2) % 127]
    return d


# =============================================================================
# 3GPP TS 38.211 §7.4.2.3  SSS 频域序列
# =============================================================================

def _gen_x0_sequence() -> np.ndarray:
    """LFSR x_0: x_0[i+7] = (x_0[i+4] + x_0[i]) mod 2, 初值 [1,0,0,0,0,0,0]."""
    x0 = np.zeros(127 + 7, dtype=np.int8)
    x0[0] = 1  # x_0[0:7] = [1, 0, 0, 0, 0, 0, 0]
    for i in range(120):
        x0[i + 7] = (x0[i + 4] + x0[i]) % 2
    return x0[:127]


def _gen_x1_sequence() -> np.ndarray:
    """LFSR x_1: x_1[i+7] = (x_1[i+1] + x_1[i]) mod 2, 初值 [1,0,0,0,0,0,0]."""
    x1 = np.zeros(127 + 7, dtype=np.int8)
    x1[0] = 1  # x_1[0:7] = [1, 0, 0, 0, 0, 0, 0]
    for i in range(120):
        x1[i + 7] = (x1[i + 1] + x1[i]) % 2
    return x1[:127]


# 模块级缓存 — 两条 m-序列只生成一次
_X0_SEQ = _gen_x0_sequence()
_X1_SEQ = _gen_x1_sequence()


def gen_sss_sequence(n_id_1: int, n_id_2: int) -> np.ndarray:
    """
    生成长度 127 的 SSS BPSK 序列 (TS 38.211 §7.4.2.3.1).

    公式:
        m_0 = 15 * floor(n_id_1 / 112) + 5 * n_id_2
        m_1 = n_id_1 mod 112
        d_SSS(n) = (1 - 2*x_0((n+m_0) mod 127)) * (1 - 2*x_1((n+m_1) mod 127))
    """
    m0 = 15 * (n_id_1 // 112) + 5 * n_id_2
    m1 = n_id_1 % 112

    n_arr = np.arange(127, dtype=np.int32)
    idx0 = (n_arr + m0) % 127
    idx1 = (n_arr + m1) % 127

    # BPSK: 0 -> +1, 1 -> -1
    seq0 = 1.0 - 2.0 * _X0_SEQ[idx0]
    seq1 = 1.0 - 2.0 * _X1_SEQ[idx1]

    return seq0 * seq1  # ±1, length 127


def build_sss_ref_table() -> np.ndarray:
    """
    构造完整 SSS 参考表 [3, 336, 127] (float).
    返回未量化的 ±1 BPSK 序列, 量化在使用前做.
    """
    table = np.zeros((N_ID_2_COUNT, N_ID_1_COUNT, N_SC_PSS), dtype=np.float64)
    for n2 in range(N_ID_2_COUNT):
        for n1 in range(N_ID_1_COUNT):
            table[n2, n1] = gen_sss_sequence(n1, n2)
    return table


# =============================================================================
# 频域 → 时域转换 (SSB sym 通用)
# =============================================================================

def sc_to_time_ssb_sym(sc_grid_127: np.ndarray) -> np.ndarray:
    """
    把 127 个中心 SC (BPSK 或复数) 映射到 N_FFT=256 时域信号.

    映射规则 (与 PSS 一致):
      - SSB 占中心 240 SC, PSS/SSS 占中心 127 SC
      - 240 SC 居中放置于 256-FFT bin: bin [center-120, center+120)
      - 127 SC 居中放置于 240 SC 内: SC[56, 56+127)
      - 然后 ifftshift + IFFT * N_FFT (抵消 1/N)
    """
    assert sc_grid_127.shape == (N_SC_PSS,), f"expected (127,), got {sc_grid_127.shape}"

    half_ssb = N_SC_SSB // 2    # 120
    center = N_FFT // 2          # 128

    # 240 SC 容器, PSS/SSS 居中
    sc_240 = np.zeros(N_SC_SSB, dtype=np.complex128)
    sc_240[56 : 56 + N_SC_PSS] = sc_grid_127

    # 240 SC 放进 256 grid 中央
    x_shift = np.zeros(N_FFT, dtype=np.complex128)
    x_shift[center - half_ssb : center + half_ssb] = sc_240

    x_freq = np.fft.ifftshift(x_shift)
    x_time = np.fft.ifft(x_freq) * N_FFT
    return x_time


# =============================================================================
# 时域 SSS 参考模板生成 (NPU 主路径)
# =============================================================================

def gen_sss_time_template(n_id_1: int, n_id_2: int) -> np.ndarray:
    """
    生成长度 N_FFT=256 的复数时域 SSS 模板 (NPU 时域相关用).

    步骤:
      1. SSS 频域 BPSK ±1 序列 (长 127)
      2. 居中放入 N_FFT=256 频域 grid (跟 PSS 一样的 SC 映射)
      3. ifftshift + IFFT * N_FFT

    数学等价:
      时域相关 Σ rx_seg(n) · conj(T(n)) = 频域相关 Σ R_SSS(k) · conj(D(k))
      (Parseval 定理). NPU 走时域路径, 离线 T 存为 complex cint16.
    """
    sss_freq = gen_sss_sequence(n_id_1, n_id_2).astype(np.complex128)
    return sc_to_time_ssb_sym(sss_freq)  # [256] complex


def build_sss_time_table() -> np.ndarray:
    """构造完整时域 SSS 参考表 [3, 336, 256] complex (浮点)."""
    table = np.zeros((N_ID_2_COUNT, N_ID_1_COUNT, N_FFT), dtype=np.complex128)
    for n2 in range(N_ID_2_COUNT):
        for n1 in range(N_ID_1_COUNT):
            table[n2, n1] = gen_sss_time_template(n1, n2)
    return table


# =============================================================================
# 合成 SSB 时域信号 (4 个 sym)
# =============================================================================

def synth_ssb_time(n_id_1: int, n_id_2: int, rng: np.random.Generator) -> np.ndarray:
    """
    合成完整 SSB 时域信号, 长度 N_SSB_SAMPLES=1176.

    布局:
      sym 0: PSS                              -> [0, 300)
      sym 1: PBCH (random QPSK 填充)          -> [300, 592)
      sym 2: SSS                              -> [592, 884)
      sym 3: PBCH (random QPSK 填充)          -> [884, 1176)
    """
    # ---- sym 0: PSS ----
    pss_freq = gen_pss_sequence(n_id_2).astype(np.complex128)
    pss_time = sc_to_time_ssb_sym(pss_freq)
    cp0 = pss_time[N_FFT - CP_FIRST_768 :]
    sym0 = np.concatenate([cp0, pss_time])
    assert sym0.shape == (SYM0_LEN,)

    # ---- sym 2: SSS ----
    sss_freq = gen_sss_sequence(n_id_1, n_id_2).astype(np.complex128)
    sss_time = sc_to_time_ssb_sym(sss_freq)
    cp2 = sss_time[N_FFT - CP_OTHER_768 :]
    sym2 = np.concatenate([cp2, sss_time])
    assert sym2.shape == (SYM_OTHER_LEN,)

    # ---- sym 1, sym 3: PBCH (random QPSK 仅作占位) ----
    qpsk = np.array([+1 + 1j, -1 + 1j, +1 - 1j, -1 - 1j],
                    dtype=np.complex128) / np.sqrt(2)

    def random_pbch_sym() -> np.ndarray:
        # SSB 240 SC 全部填随机 QPSK
        sc_240 = qpsk[rng.integers(0, 4, size=N_SC_SSB)]
        half_ssb = N_SC_SSB // 2
        center = N_FFT // 2
        x_shift = np.zeros(N_FFT, dtype=np.complex128)
        x_shift[center - half_ssb : center + half_ssb] = sc_240
        x_freq = np.fft.ifftshift(x_shift)
        x_time = np.fft.ifft(x_freq) * N_FFT
        cp = x_time[N_FFT - CP_OTHER_768 :]
        return np.concatenate([cp, x_time])

    sym1 = random_pbch_sym()
    sym3 = random_pbch_sym()

    ssb = np.concatenate([sym0, sym1, sym2, sym3])
    assert ssb.shape == (N_SSB_SAMPLES,)
    return ssb


# =============================================================================
# 合成完整 rx (SSB + 噪声 + CFO + 量化)
# =============================================================================

def synth_rx(n_id_1: int, n_id_2: int,
             mu_true: int,
             cfo_hz: float,
             snr_db: float,
             rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """
    合成 1 个 case 的完整 rx[153600] cint16 信号.

    参数:
      mu_true : SSB 在 rx 中的起点 (sym 0 CP 起点的位置)
      cfo_hz  : 注入 CFO

    返回:
      rx_cint16 : [153600, 2] int16, IQ 交错
      meta      : dict, 含 ground truth 字段
    """
    # 1. 合成 SSB
    ssb = synth_ssb_time(n_id_1, n_id_2, rng)  # [1176] complex

    # 2. 嵌入 rx 时域
    rx = np.zeros(N_SEARCH, dtype=np.complex128)
    assert 0 <= mu_true <= N_SEARCH - N_SSB_SAMPLES, \
        f"mu_true={mu_true} out of range [0, {N_SEARCH - N_SSB_SAMPLES}]"
    rx[mu_true : mu_true + N_SSB_SAMPLES] = ssb

    # 其余样本是随机 QPSK 数据 (低功率, 不干扰 SSB 检测)
    n_other = N_SEARCH - N_SSB_SAMPLES
    other_mask = np.ones(N_SEARCH, dtype=bool)
    other_mask[mu_true : mu_true + N_SSB_SAMPLES] = False
    qpsk = np.array([+1 + 1j, -1 + 1j, +1 - 1j, -1 - 1j],
                    dtype=np.complex128) / np.sqrt(2)
    rx[other_mask] = qpsk[rng.integers(0, 4, size=int(other_mask.sum()))] * 0.3

    # 3. 注入 CFO (整段)
    n_arr = np.arange(N_SEARCH, dtype=np.float64)
    rx = rx * np.exp(1j * 2.0 * np.pi * cfo_hz * n_arr / FS_HZ)

    # 4. 加 AWGN
    sig_power = np.mean(np.abs(rx[mu_true : mu_true + N_SSB_SAMPLES]) ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = (rng.standard_normal(N_SEARCH) + 1j * rng.standard_normal(N_SEARCH))
    noise *= np.sqrt(noise_power / 2.0)
    rx = rx + noise

    # 5. 量化到 cint16 (Q2.13)
    # 先归一化使峰值落在 ~ ±2 (Q2.13 范围 [-4, +4))
    peak = np.max(np.abs(rx))
    scale_pre = 2.0 / max(peak, 1e-12)
    rx_scaled = rx * scale_pre

    rx_int_re = np.clip(np.round(rx_scaled.real * Q_SCALE), INT16_MIN, INT16_MAX).astype(np.int16)
    rx_int_im = np.clip(np.round(rx_scaled.imag * Q_SCALE), INT16_MIN, INT16_MAX).astype(np.int16)
    rx_cint16 = np.stack([rx_int_re, rx_int_im], axis=-1)  # [N_SEARCH, 2]

    # 计算 ground truth: SSB sym 2 useful 部分起点 (从 mu_true 算起)
    #   = mu_true + sym0_len + CP2 + sym1_len + CP2 起点
    # 等价于:
    #   sym_2_useful_start = mu_true + SYM0_LEN + SYM_OTHER_LEN + CP_OTHER_768
    #                      = mu_true + 300 + 292 + 36 = mu_true + 628
    sym_2_useful_start = mu_true + SYM0_LEN + SYM_OTHER_LEN + CP_OTHER_768

    # PSS useful 起点 (PSS Correlator 的 mu_t 输出语义)
    # = mu_true + CP_FIRST_768
    mu_t_pss = mu_true + CP_FIRST_768

    # G_hat 计算: integer CFO ∈ {-1, 0, +1}
    if cfo_hz >= 15000.0:
        g_hat = +1
    elif cfo_hz < -15000.0:
        g_hat = -1
    else:
        g_hat = 0

    meta = {
        "n_id_1": int(n_id_1),
        "n_id_2": int(n_id_2),
        "pcid": int(3 * n_id_1 + n_id_2),
        "mu_true": int(mu_true),
        "mu_t_pss": int(mu_t_pss),
        "sym_2_useful_start": int(sym_2_useful_start),
        "cfo_hz": float(cfo_hz),
        "snr_db": float(snr_db),
        "g_hat": int(g_hat),
        "scale_pre": float(scale_pre),
    }
    return rx_cint16, meta


# =============================================================================
# SSS Correlator 算子 (T=4 joint search, fp64 reference)
# =============================================================================

def sss_correlator_ref(rx_cint16: np.ndarray,
                       mu_t_pss: int,
                       n_id_2: int,
                       g_hat: int,
                       sss_ref_table_127: np.ndarray) -> dict:
    """
    SSS Correlator 参考实现 (fp64 numpy).

    输入:
      rx_cint16        : [N_SEARCH, 2] int16, IQ 交错
      mu_t_pss         : int, PSS useful 起点 (PSS Correlator 输出)
      n_id_2           : int ∈ {0,1,2}
      g_hat            : int ∈ {-1,0,+1}
      sss_ref_table_127: [3, 336, 127] float ±1, 离线常量

    输出 dict:
      n_id_1, pcid, peak, second, refined_sym2_start, tau_star,
      metric_grid (debug, [N_TAU, 336])
    """
    # 反量化
    rx = (rx_cint16[:, 0].astype(np.float64) + 1j * rx_cint16[:, 1].astype(np.float64)) * Q_INV

    # 1. CFO 反旋 (整段 153600 反旋, 或仅 sym 2 邻域反旋 - 这里反旋邻域)
    cfo_compensate_hz = -g_hat * SCS_HZ
    # 2. SSB 起点 = mu_t_pss - CP_FIRST_768
    ssb_start_nominal = mu_t_pss - CP_FIRST_768
    # 3. sym 2 useful 起点 nominal
    sym_2_useful_start_nominal = ssb_start_nominal + SYM0_LEN + SYM_OTHER_LEN + CP_OTHER_768

    # 4. 选择 n_id_2 对应的 336 候选子表
    ref_subset = sss_ref_table_127[n_id_2]  # [336, 127] ±1

    # 5. 9 个 tau 位置, 各跑一次 FFT-256 + GEMM
    metric_grid = np.zeros((N_TAU, N_ID_1_COUNT), dtype=np.float64)

    for tau_idx, tau in enumerate(range(-T_HALF, T_HALF + 1)):
        sym_2_start = sym_2_useful_start_nominal + tau

        # 边界保护
        if sym_2_start < 0 or sym_2_start + N_FFT > N_SEARCH:
            metric_grid[tau_idx] = -np.inf
            continue

        # 取 256 个样本
        seg = rx[sym_2_start : sym_2_start + N_FFT]

        # CFO 反旋 (相对位置 0..255)
        # 注意: 全局相位偏移 (mu_t_pss 处累积的相位) 在相干检测中无所谓,
        #       只要 sym2 内部 256 sample 间相位连续即可
        n_arr = np.arange(N_FFT, dtype=np.float64)
        seg_derot = seg * np.exp(1j * 2.0 * np.pi * cfo_compensate_hz * n_arr / FS_HZ)

        # FFT-256
        X_freq_shift = np.fft.fftshift(np.fft.fft(seg_derot))  # bin 排列居中, DC 在 N_FFT/2

        # 提取中心 127 SC
        # SSB 240 SC 居中于 N_FFT=256: bin [128-120, 128+120) = [8, 248)
        # PSS/SSS 127 SC 居中于 240 SC: SSB-relative [56, 56+127), 即 FFT-bin [8+56, 8+56+127) = [64, 191)
        sss_rx_freq = X_freq_shift[64 : 64 + N_SC_PSS]  # [127] complex

        # 复数相关: corr[j] = Σ_k sss_rx_freq[k] * conj(D[j, k])
        # D 是实数 BPSK ±1, conj 不变 → corr[j] = Σ_k sss_rx_freq[k] * D[j, k]
        # → ref_subset @ sss_rx_freq, but ref is [336,127], rx is [127] complex
        corr = ref_subset @ sss_rx_freq  # [336] complex

        # |·|²
        metric = np.abs(corr) ** 2
        metric_grid[tau_idx] = metric

    # 全局 argmax
    flat_idx = int(np.argmax(metric_grid))
    tau_idx_star = flat_idx // N_ID_1_COUNT
    n_id_1_star = flat_idx % N_ID_1_COUNT
    tau_star = tau_idx_star - T_HALF
    peak = float(metric_grid[tau_idx_star, n_id_1_star])

    # 第二大 (Wang-Berggren 双阈值):
    # 关键: second 应该是"其它 n_id_1 的最大值", 而不是"所有 (tau, n_id_1)
    # 里的第二大". 因为同一个正确 n_id_1 在相邻 tau 会有相干旁瓣 (sinc-like),
    # 这些不是"竞争候选", 排除掉.
    # 先对每个 n_id_1 在 tau 维度取 max → [336]
    metric_per_n1 = np.max(metric_grid, axis=0)  # [336]
    # 再 mask 掉 n_id_1_star 求第二大
    metric_per_n1_masked = metric_per_n1.copy()
    metric_per_n1_masked[n_id_1_star] = -np.inf
    second = float(np.max(metric_per_n1_masked))

    refined_sym2_start = sym_2_useful_start_nominal + tau_star
    pcid = 3 * n_id_1_star + n_id_2

    return {
        "n_id_1": int(n_id_1_star),
        "pcid": int(pcid),
        "peak": peak,
        "second": second,
        "refined_sym2_start": int(refined_sym2_start),
        "tau_star": int(tau_star),
        "metric_grid": metric_grid,
    }


# =============================================================================
# SSS Correlator 时域路径 (NPU 主路径 fp64 reference)
# =============================================================================

def sss_correlator_time_ref(rx_cint16: np.ndarray,
                             mu_t_pss: int,
                             n_id_2: int,
                             g_hat: int,
                             sss_time_table: np.ndarray) -> dict:
    """
    时域 SSS Correlator 参考实现 (NPU 主路径 fp64).

    跟频域版本数学等价 (Parseval), 但实现走时域内积:
      Φ^(j,τ) = Σ_n rx_seg^(τ)(n) · conj(T^(j)(n))

    其中 T^(j)(n) 是时域 SSS 模板 (N_FFT=256 复数), 离线生成.

    输入:
      sss_time_table : [3, 336, 256] complex
      其余参数跟频域版一致.

    输出 dict (跟频域版一致 + 一个 "metric_grid" debug field).
    """
    # 反量化
    rx = (rx_cint16[:, 0].astype(np.float64) + 1j * rx_cint16[:, 1].astype(np.float64)) * Q_INV

    # 1. CFO 反旋频率 (单边)
    cfo_compensate_hz = -g_hat * SCS_HZ

    # 2. SSB 起点 = mu_t_pss - CP_FIRST_768
    ssb_start_nominal = mu_t_pss - CP_FIRST_768

    # 3. sym 2 useful 起点 nominal
    sym_2_useful_start_nominal = ssb_start_nominal + SYM0_LEN + SYM_OTHER_LEN + CP_OTHER_768

    # 4. 选 n_id_2 对应子表 [336, 256] complex
    T_subset = sss_time_table[n_id_2]

    # 5. 9 个 tau 位置, 复用一段 [-4, +4+256) 反旋后的信号
    metric_grid = np.zeros((N_TAU, N_ID_1_COUNT), dtype=np.float64)

    # 取整段 [sym_2_start - T_HALF, sym_2_start + T_HALF + N_FFT)
    seg_start = sym_2_useful_start_nominal - T_HALF
    seg_end = sym_2_useful_start_nominal + T_HALF + N_FFT

    if seg_start < 0 or seg_end > N_SEARCH:
        # 越界保护 (不应触发)
        metric_grid[:] = -np.inf
        return {
            "n_id_1": 0, "pcid": 3 * 0 + n_id_2,
            "peak": 0.0, "second": 0.0,
            "refined_sym2_start": sym_2_useful_start_nominal,
            "tau_star": 0,
            "metric_grid": metric_grid,
        }

    seg_full = rx[seg_start : seg_end]  # 长度 = 2*T_HALF + N_FFT = 264

    # CFO 反旋整段 (相对 seg_start = 0 起算的相位)
    n_arr_full = np.arange(len(seg_full), dtype=np.float64)
    seg_derot = seg_full * np.exp(1j * 2.0 * np.pi * cfo_compensate_hz * n_arr_full / FS_HZ)

    # 9 个 tau 各切 [tau+T_HALF, tau+T_HALF+N_FFT)
    for tau_idx, tau in enumerate(range(-T_HALF, T_HALF + 1)):
        offset = tau + T_HALF
        rx_seg = seg_derot[offset : offset + N_FFT]  # [256] complex

        # 时域内积: corr[j] = Σ_n rx_seg(n) · conj(T_subset[j, n])
        # = T_subset.conj() @ rx_seg  ... 不对, 这是 conj(T)·rx, 我们要的是 rx·conj(T)
        # 等价: corr = (T_subset.conj() @ rx_seg) 元素是 Σ_n conj(T[j,n]) * rx_seg(n)
        # = Σ_n rx_seg(n) * conj(T[j,n])  ✓ 一样
        corr = T_subset.conj() @ rx_seg  # [336] complex
        metric = np.abs(corr) ** 2
        metric_grid[tau_idx] = metric

    # 全局 argmax
    flat_idx = int(np.argmax(metric_grid))
    tau_idx_star = flat_idx // N_ID_1_COUNT
    n_id_1_star = flat_idx % N_ID_1_COUNT
    tau_star = tau_idx_star - T_HALF
    peak = float(metric_grid[tau_idx_star, n_id_1_star])

    # second: 排除当前最优 n_id_1, 在其他 n_id_1 的 tau-max 上取
    metric_per_n1 = np.max(metric_grid, axis=0)
    metric_per_n1_masked = metric_per_n1.copy()
    metric_per_n1_masked[n_id_1_star] = -np.inf
    second = float(np.max(metric_per_n1_masked))

    refined_sym2_start = sym_2_useful_start_nominal + tau_star
    pcid = 3 * n_id_1_star + n_id_2

    return {
        "n_id_1": int(n_id_1_star),
        "pcid": int(pcid),
        "peak": peak,
        "second": second,
        "refined_sym2_start": int(refined_sym2_start),
        "tau_star": int(tau_star),
        "metric_grid": metric_grid,
    }


# =============================================================================
# 测试 case 定义 (8 个 case, 覆盖 N_ID^(1) 范围 + SNR + CFO)
# =============================================================================

# (n_id_1, n_id_2, mu_true, cfo_hz, snr_db, label)
TEST_CASES = [
    (  0, 0,  10000,      0.0, 30.0, "clean_min_n_id_1"),
    (167, 1,  50000,   -800.0, 30.0, "mid_pos_pss1"),
    (335, 2,  90000,    +800.0, 30.0, "max_n_id_1_pss2"),
    ( 42, 0,  20000,  +10000.0, 30.0, "small_pos_cfo_pcid_126"),
    (168, 1,  60000,  -10000.0, 30.0, "small_neg_cfo_pcid_505"),
    (333, 0,  30000,  +25000.0, 30.0, "int_cfo_p1_pcid_999"),
    ( 50, 1,  80000,   +5000.0,  0.0, "low_snr_0db"),
    (100, 2, 100000,   -3000.0, -3.0, "low_snr_n3db"),
]


def _verdict(meta: dict, result: dict) -> Tuple[bool, str]:
    """判定单 case 是否 PASS."""
    # n_id_1 必须 exact match
    ok_n1 = (result["n_id_1"] == meta["n_id_1"])

    # tau_star 必须在 ±2 sample 容差内 (理想是 0, 但允许 fp64 噪声引入 ±1~2)
    ok_tau = abs(result["tau_star"]) <= 2

    # peak / second ratio 必须 > 阈值
    if result["second"] > 0:
        ratio = result["peak"] / result["second"]
    else:
        ratio = float("inf")

    # 低 SNR 时 ratio 会变小, 但仍应 >2 (双阈值要求通常 T1=3)
    ok_ratio = ratio > 2.0

    ok = ok_n1 and ok_tau and ok_ratio
    if ok:
        msg = f"PASS"
    else:
        reasons = []
        if not ok_n1:
            reasons.append(f"n_id_1 mismatch (est={result['n_id_1']} vs truth={meta['n_id_1']})")
        if not ok_tau:
            reasons.append(f"tau_star={result['tau_star']} out of ±2")
        if not ok_ratio:
            reasons.append(f"ratio={ratio:.2f} <= 2.0")
        msg = "FAIL: " + "; ".join(reasons)
    return ok, msg


# =============================================================================
# Main: 跑所有 cases + 落 golden
# =============================================================================

def write_case_golden(case_dir: Path,
                      rx_cint16: np.ndarray,
                      meta: dict,
                      result: dict) -> None:
    """落 case_N/{input.bin, truth.json, expected.json}."""
    case_dir.mkdir(parents=True, exist_ok=True)

    # input.bin: cint16 IQ 交错, 跟 PSS Correlator 一致
    rx_cint16.astype(np.int16).tofile(case_dir / "input.bin")

    # truth.json: ground truth (合成时的真值)
    with open(case_dir / "truth.json", "w") as f:
        json.dump(meta, f, indent=2)

    # expected.json: SSS Correlator 期望输出
    expected = {k: v for k, v in result.items() if k != "metric_grid"}
    with open(case_dir / "expected.json", "w") as f:
        json.dump(expected, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=GOLDEN_DIR,
                        help=f"Golden output directory (default: {GOLDEN_DIR})")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed")
    parser.add_argument("--no-golden", action="store_true",
                        help="Don't write golden files, only print verdict")
    args = parser.parse_args()

    print(f"[sss_correlator_ref] Output dir: {args.output_dir}")
    print(f"[sss_correlator_ref] Building SSS reference tables...")
    t0 = time.time()
    sss_freq_table = build_sss_ref_table()        # [3, 336, 127] real ±1
    sss_time_table = build_sss_time_table()       # [3, 336, 256] complex
    print(f"[sss_correlator_ref]   freq table  done in {time.time()-t0:.2f}s")
    print(f"[sss_correlator_ref]   time table  shape={sss_time_table.shape} "
          f"max|val|={np.max(np.abs(sss_time_table)):.3f}")

    # 离线落 sss_time_table.bin (NPU 用)
    if not args.no_golden:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # 量化策略: 时域 SSS 复数, |val| 量级 ~ 1-2 (实测稍后确认)
        # 用 Q2.13 跟 PSS 一致: val * 8192, clip 到 int16
        max_abs = float(np.max(np.abs(sss_time_table)))
        # 留 1 dB 头, scale 使得 max_abs → ~28000 (< 32767)
        scale_table = (28000.0 / Q_SCALE) / max_abs
        time_table_scaled = sss_time_table * scale_table

        # 落 [3, 336, 256, 2] int16, IQ 交错
        re = np.clip(np.round(time_table_scaled.real * Q_SCALE),
                      INT16_MIN, INT16_MAX).astype(np.int16)
        im = np.clip(np.round(time_table_scaled.imag * Q_SCALE),
                      INT16_MIN, INT16_MAX).astype(np.int16)
        out = np.stack([re, im], axis=-1)  # [3, 336, 256, 2]
        out.tofile(args.output_dir / "sss_time_table.bin")
        print(f"[sss_correlator_ref] Wrote sss_time_table.bin "
              f"({out.nbytes/1024:.1f} KB, scale_table={scale_table:.6f})")

        # 落 twiddle_g.bin (NPU 用)
        # 3 个 g 的 CFO 反旋 twiddle, 每个 SEG_LEN=264 cint16 复
        # twiddle[g, n] = exp(-j 2π · g·SCS · n / FS)  (g ∈ {-1, 0, +1})
        SEG_LEN_PY = 2 * T_HALF + N_FFT  # 264
        twid_table = np.zeros((3, SEG_LEN_PY), dtype=np.complex128)
        for g_idx, g in enumerate([-1, 0, +1]):
            n_arr = np.arange(SEG_LEN_PY, dtype=np.float64)
            twid_table[g_idx] = np.exp(-1j * 2.0 * np.pi * g * SCS_HZ * n_arr / FS_HZ)

        # twiddle 量级 ~1, 用 Q2.13 (= Q_SCALE) 量化
        twid_re = np.clip(np.round(twid_table.real * Q_SCALE), INT16_MIN, INT16_MAX).astype(np.int16)
        twid_im = np.clip(np.round(twid_table.imag * Q_SCALE), INT16_MIN, INT16_MAX).astype(np.int16)
        twid_out = np.stack([twid_re, twid_im], axis=-1)  # [3, 264, 2]
        twid_out.tofile(args.output_dir / "twiddle_g.bin")
        print(f"[sss_correlator_ref] Wrote twiddle_g.bin "
              f"({twid_out.nbytes/1024:.1f} KB)")

    print(f"\n[sss_correlator_ref] Running {len(TEST_CASES)} cases (TIME-DOMAIN path)...")
    print(f"[sss_correlator_ref]   (also cross-checking against FREQ-DOMAIN ref)")
    print("="*120)

    pass_count = 0
    fail_cases = []
    cross_check_match = 0

    for case_idx, (n1, n2, mu_true, cfo, snr, label) in enumerate(TEST_CASES):
        case_name = f"case_{case_idx}_{label}"
        rng = np.random.default_rng(args.seed + case_idx)

        # 合成
        rx_cint16, meta = synth_rx(n1, n2, mu_true, cfo, snr, rng)

        # 跑 SSS Correlator 时域路径 (主)
        t0 = time.time()
        result = sss_correlator_time_ref(rx_cint16,
                                          mu_t_pss=meta["mu_t_pss"],
                                          n_id_2=meta["n_id_2"],
                                          g_hat=meta["g_hat"],
                                          sss_time_table=sss_time_table)
        elapsed_time = time.time() - t0

        # Cross-check: 频域路径 (互验证)
        result_freq = sss_correlator_ref(rx_cint16,
                                          mu_t_pss=meta["mu_t_pss"],
                                          n_id_2=meta["n_id_2"],
                                          g_hat=meta["g_hat"],
                                          sss_ref_table_127=sss_freq_table)
        if result["n_id_1"] == result_freq["n_id_1"]:
            cross_check_match += 1

        ok, msg = _verdict(meta, result)
        if ok:
            pass_count += 1
        else:
            fail_cases.append(case_name)

        # 打印
        ratio = result["peak"] / max(result["second"], 1e-12)
        det_snr_db = 10.0 * np.log10(ratio)
        cross = "✓" if result["n_id_1"] == result_freq["n_id_1"] else "✗"
        print(f"║ {case_name:35s}  truth: n1={meta['n_id_1']:3d} pcid={meta['pcid']:4d}  "
              f"|  est: n1={result['n_id_1']:3d} pcid={result['pcid']:4d} "
              f"tau={result['tau_star']:+d} ratio={ratio:6.1f} ({det_snr_db:5.1f}dB)  "
              f"freq={cross}  [{elapsed_time*1000:.1f}ms]  {msg}")

        # 落 golden
        if not args.no_golden:
            write_case_golden(args.output_dir / case_name,
                              rx_cint16, meta, result)

    print("="*120)
    print(f"[sss_correlator_ref] PASS:  {pass_count}/{len(TEST_CASES)}")
    print(f"[sss_correlator_ref] Cross-check (time vs freq): {cross_check_match}/{len(TEST_CASES)}")
    if fail_cases:
        print(f"[sss_correlator_ref] FAIL cases: {fail_cases}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())