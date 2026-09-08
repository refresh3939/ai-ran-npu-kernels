"""
pss_correlator_ref.py
=====================
PSS 时域相关 + 整数 CFO 联合搜索 — numpy clean room 参考实现

算法来源:
    Tuninato et al., EURASIP JWCN 2023 (DOI 10.1186/s13638-023-02317-5) Eq. 13
    Wang & Berggren, IEEE ICC 2018 (DOI 10.1109/ICC.2018.8422145)
    3GPP TS 38.211 v15.x §7.4.2.2 (PSS m-sequence)
                       §7.4.3.1 (SSB 频域结构)

Clean Room: 不参考 MATLAB / OAI / srsRAN / py3gpp 源码。

输入 (运行时):
    y_in        : 接收 IQ, cint16 量化 (Q?.? 见下)，长度 N_SEARCH = 153600 @ 7.68 MSPS
    pss_ref     : 3 个 PSS 时域参考, cint16, shape [3, 256]
    twiddle_cfo : 3 个 G ∈ {-1, 0, +1} 的反旋因子, cint16, shape [3, N_SEARCH]

输出:
    mu_t        : int,        PSS 起始样本位置
    n_id_2      : int ∈ {0,1,2}
    g_hat       : int ∈ {-1, 0, +1}
    peak_metric : float, 相关峰幅度平方
    noise_floor : float, median of |C|² (单峰鲁棒)

测试:
    本脚本自合成 SSB + AWGN + CFO + 量化, 跑 8 个 case, 落 golden 到
    AIRAN_DATA_DIR/golden/rx/pss_correlator/

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
# 锁定常量 (跟 Ascend C kernel 一致)
# =============================================================================

# 采样率 / 帧结构 (FR1, SCS=30kHz, 重采样到 SSB 频段)
FS_HZ           = 7_680_000          # 7.68 MSPS
SCS_HZ          = 30_000             # subcarrier spacing
N_FFT           = 256                # SSB 频段 FFT 长度
N_SC_SSB        = 240                # SSB 占 20 RB × 12 = 240 子载波
N_SC_PSS        = 127                # PSS 占 SSB 中心 127 个子载波

# 搜索窗口
N_SEARCH        = 153_600            # 20 ms @ 7.68 MSPS
N_MF            = 256                # 匹配滤波器长度 (= N_FFT, 不含 CP)
M               = N_SEARCH - N_MF    # 滑窗位置数 = 153344
N_PSS           = 3                  # PSS 候选数 (N_ID^(2))

# 整数 CFO 假设
G_LIST          = np.array([-1, 0, +1], dtype=np.int32)
N_G             = len(G_LIST)

# 量化 (cint16 Q2.13: 范围 [-4, +4),精度 ~ 1.22e-4)
#   选 Q2.13 因为合成后归一化信号在 [-2, +2] 量级内,留 1 bit 给 CFO+noise 余量
Q_FRAC_BITS     = 13
Q_SCALE         = 1 << Q_FRAC_BITS   # 8192
Q_INV           = 1.0 / Q_SCALE

# CP 长度 (用于合成 SSB; 重采样到 7.68 MSPS 下的近似 CP)
#   30 kHz SCS @ 30.72 MSPS: CP_LEN_FIRST=176, CP_LEN_OTHER=144
#   按 7.68/30.72=1/4 缩放: CP_FIRST≈44, CP_OTHER≈36
CP_FIRST_768    = 44
CP_OTHER_768    = 36

# SSB 总长度 (4 个 OFDM 符号 @ 7.68 MSPS)
#   sym 0 (first): CP_FIRST + N_FFT
#   sym 1-3 (other): CP_OTHER + N_FFT  each
N_SSB_SAMPLES   = (CP_FIRST_768 + N_FFT) + 3 * (CP_OTHER_768 + N_FFT)
#               = (44 + 256) + 3*(36 + 256) = 300 + 876 = 1176

# 数据目录 (kernel-local 自包含, 跟 OFDM 一致)
#   优先级: AIRAN_DATA_DIR > 基于脚本自动推断 (scripts/ 上一级)
#   推断: scripts/pss_correlator_ref.py → scripts 上一级 = kernel 目录 → 加 data/
_THIS_FILE      = Path(__file__).resolve()
_KERNEL_DIR     = _THIS_FILE.parent.parent          # scripts/ 的上一级
PROJECT_ROOT    = Path(os.environ.get(
    "AIRAN_DATA_DIR",
    str(_KERNEL_DIR)
)) / "data"
GOLDEN_DIR      = PROJECT_ROOT / "golden" / "rx" / "pss_correlator"


# =============================================================================
# 3GPP TS 38.211 §7.4.2.2  PSS 频域序列生成
# =============================================================================

def gen_pss_sequence(N_ID_2: int) -> np.ndarray:
    """
    生成长度 127 的 PSS BPSK 序列 (频域).

    3GPP TS 38.211 §7.4.2.2.1:
        x(i+7) = (x(i+4) + x(i)) mod 2
        initial: [x(6),...,x(0)] = [1,1,1,0,1,1,0]    (i.e. x(0..6) = 0,1,1,0,1,1,1)
        d_PSS(n) = 1 - 2*x((n + 43*N_ID_2) mod 127)

    Returns:
        np.ndarray shape (127,) of ±1.0, dtype float64
    """
    assert N_ID_2 in (0, 1, 2)
    x = np.zeros(127 + 7, dtype=np.int8)
    x[0:7] = [0, 1, 1, 0, 1, 1, 1]
    for i in range(120):
        x[i + 7] = (x[i + 4] + x[i]) % 2

    d = np.empty(127, dtype=np.float64)
    for n in range(127):
        d[n] = 1.0 - 2.0 * x[(n + 43 * N_ID_2) % 127]
    return d


def pss_time_template(N_ID_2: int) -> np.ndarray:
    """
    生成 N_ID_2 对应的 PSS 时域模板 (长度 N_FFT=256, complex).

    流程:
        1. gen_pss_sequence → 127 个 BPSK 频域符号
        2. 映射到 SSB 的中心 127 个子载波
        3. SSB 占 240 子载波居中放进 N_FFT=256 的栅格
        4. ifftshift + IFFT → 时域 256 复数样本

    Returns:
        shape (256,) complex128, 已归一化 (IFFT × N_FFT 抵消)
    """
    pss_freq = gen_pss_sequence(N_ID_2)              # (127,) ±1.0

    # SSB 240 子载波居中, PSS 在中心 127:
    #   ssb_grid[k] for k in [0, 240),  pss 在 k ∈ [56, 56+127)
    ssb_grid = np.zeros(N_SC_SSB, dtype=np.complex128)
    pss_lo   = (N_SC_SSB - N_SC_PSS) // 2            # = 56
    ssb_grid[pss_lo:pss_lo + N_SC_PSS] = pss_freq

    # SSB 240 子载波放进 N_FFT=256 居中
    fft_grid = np.zeros(N_FFT, dtype=np.complex128)
    ssb_lo   = (N_FFT - N_SC_SSB) // 2               # = 8
    fft_grid[ssb_lo:ssb_lo + N_SC_SSB] = ssb_grid

    # ifftshift + IFFT
    #   等价于把 DC 移到中心然后 IFFT
    x_shift = np.fft.ifftshift(fft_grid)
    x_time  = np.fft.ifft(x_shift) * N_FFT           # × N_FFT 抵消 IFFT 的 1/N 归一化

    return x_time.astype(np.complex128)


# =============================================================================
# CFO 反旋因子表 (twiddle)
# =============================================================================

def gen_cfo_twiddle(N_search: int = N_SEARCH) -> np.ndarray:
    """
    生成 3 个整数 CFO 假设的反旋因子.

    对每个 G ∈ {-1, 0, +1}:
        twiddle_G[n] = exp(-j · 2π · G · SCS · n / fs)

    用于把接收信号按 G 反旋: y_G[n] = y[n] · twiddle_G[n]

    Returns:
        shape (3, N_search) complex128
    """
    n = np.arange(N_search, dtype=np.float64)
    twiddle = np.zeros((N_G, N_search), dtype=np.complex128)
    for gi, G in enumerate(G_LIST):
        phase = -2.0 * np.pi * float(G) * SCS_HZ * n / FS_HZ
        twiddle[gi] = np.exp(1j * phase)
    return twiddle


# =============================================================================
# 量化 / 反量化
# =============================================================================

def quantize_to_cint16(x: np.ndarray, scale: int = Q_SCALE) -> np.ndarray:
    """
    复数 → 交错 int16 IQ.

    输入:  shape (N,) complex
    输出:  shape (2N,) int16, [I0,Q0,I1,Q1,...]
    """
    re = np.round(x.real * scale).astype(np.int64)
    im = np.round(x.imag * scale).astype(np.int64)
    re = np.clip(re, -32768, 32767).astype(np.int16)
    im = np.clip(im, -32768, 32767).astype(np.int16)
    out = np.empty(2 * x.size, dtype=np.int16)
    out[0::2] = re
    out[1::2] = im
    return out


def dequantize_from_cint16(iq: np.ndarray, scale_inv: float = Q_INV) -> np.ndarray:
    """交错 int16 → complex128 (反量化)."""
    re = iq[0::2].astype(np.float64) * scale_inv
    im = iq[1::2].astype(np.float64) * scale_inv
    return re + 1j * im


# =============================================================================
# 测试信号合成
# =============================================================================

def synth_ssb_window(pss_id: int,
                     mu_true: int,
                     rng: np.random.Generator,
                     n_search: int = N_SEARCH) -> np.ndarray:
    """
    合成一段 N_search 长度的时域 IQ, 在 mu_true 位置插入一个 SSB.

    SSB 结构 (4 个 OFDM 符号):
        sym 0: PSS  (CP_FIRST + N_FFT)
        sym 1: PBCH (CP_OTHER + N_FFT)    用随机 QPSK 占位
        sym 2: SSS  (CP_OTHER + N_FFT)    用随机 QPSK 占位
        sym 3: PBCH (CP_OTHER + N_FFT)    用随机 QPSK 占位

    SSB 前后填充随机 QPSK 数据符号 (粗略模拟其它信道内容).

    Args:
        pss_id   : ∈ {0,1,2}, 决定哪个 PSS time template 用
        mu_true  : SSB 起点样本索引 (= sym 0 的 CP 起点)
        rng      : np.random.Generator

    Returns:
        shape (n_search,) complex128
    """
    assert 0 <= mu_true < n_search - N_SSB_SAMPLES, \
        f"mu_true={mu_true} 越界, 需在 [0, {n_search - N_SSB_SAMPLES})"

    # 1. 生成 SSB
    pss_t = pss_time_template(pss_id)                # (256,) complex

    # PSS symbol: CP + useful
    pss_sym = np.concatenate([pss_t[-CP_FIRST_768:], pss_t])    # (CP_FIRST+256,)

    def random_data_sym(cp_len: int) -> np.ndarray:
        """合成一个 OFDM 符号: 240 个 SSB 子载波 QPSK + IFFT + CP."""
        qpsk = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        ssb_data = qpsk[rng.integers(0, 4, size=N_SC_SSB)]
        # 放进 256 栅格
        fft_grid = np.zeros(N_FFT, dtype=np.complex128)
        ssb_lo   = (N_FFT - N_SC_SSB) // 2
        fft_grid[ssb_lo:ssb_lo + N_SC_SSB] = ssb_data
        x_shift = np.fft.ifftshift(fft_grid)
        x_time  = np.fft.ifft(x_shift) * N_FFT
        return np.concatenate([x_time[-cp_len:], x_time])

    sym1 = random_data_sym(CP_OTHER_768)
    sym2 = random_data_sym(CP_OTHER_768)
    sym3 = random_data_sym(CP_OTHER_768)

    ssb = np.concatenate([pss_sym, sym1, sym2, sym3])
    assert ssb.size == N_SSB_SAMPLES, f"SSB size {ssb.size} != {N_SSB_SAMPLES}"

    # 2. 合成 N_search 长背景 (随机 QPSK 时域, 模拟旁边的 PDSCH/PDCCH)
    qpsk = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
    bg   = qpsk[rng.integers(0, 4, size=n_search)] * 0.3  # 背景能量低一些, 避免压过 SSB

    # 3. 插入 SSB
    bg[mu_true:mu_true + N_SSB_SAMPLES] = ssb

    return bg


def inject_cfo(x: np.ndarray, cfo_hz: float, fs_hz: float = FS_HZ) -> np.ndarray:
    """对时域信号注入 CFO: y[n] = x[n] · exp(j·2π·f·n/fs)."""
    n = np.arange(x.size, dtype=np.float64)
    phase = 2.0 * np.pi * cfo_hz * n / fs_hz
    return x * np.exp(1j * phase)


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """加复 AWGN, SNR 定义为 信号功率 / 噪声功率."""
    sig_power   = np.mean(np.abs(x) ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    # 复噪声: re, im 各 N(0, σ²/2)
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    return x + noise


# =============================================================================
# 核心算子 — PSS 相关 + 整数 CFO 联合搜索
# =============================================================================

def _pss_template_bank() -> np.ndarray:
    """
    返回 3 个 PSS 时域模板, shape (3, 256) complex.
    """
    bank = np.zeros((N_PSS, N_FFT), dtype=np.complex128)
    for l in range(N_PSS):
        bank[l] = pss_time_template(l)
    return bank


def pss_correlator(y_in: np.ndarray,
                   pss_ref: np.ndarray = None,
                   twiddle_cfo: np.ndarray = None
                   ) -> Tuple[int, int, int, float, float, np.ndarray]:
    """
    PSS 相关核心算子 (numpy clean room 参考).

    Args:
        y_in        : complex128 shape (N_SEARCH,) — 反量化后的接收 IQ
        pss_ref     : complex128 shape (3, 256) or None (None → 内部生成)
        twiddle_cfo : complex128 shape (3, N_SEARCH) or None (None → 内部生成)

    Returns:
        mu_t       : int — 检测到的 PSS sym0 起点 (sym0 CP 起点; 不是 useful 部分起点)
        n_id_2     : int ∈ {0,1,2}
        g_hat      : int ∈ {-1, 0, +1}
        peak       : float — 峰值 |C|²
        noise_floor: float — median |C|²
        metric_cube: complex 立方体 |C|² shape (N_G, N_PSS, M+1=153345)
                     供 debug 用; 真正 OTA kernel 不输出这个
    """
    assert y_in.size == N_SEARCH, f"y_in size {y_in.size} != {N_SEARCH}"

    if pss_ref is None:
        pss_ref = _pss_template_bank()
    if twiddle_cfo is None:
        twiddle_cfo = gen_cfo_twiddle()

    # ─── Stage 1: CFO 反旋, 3 路 G 一起做 ───
    #   y_G[g, n] = y_in[n] · twiddle_cfo[g, n]
    y_G = y_in[None, :] * twiddle_cfo                # (3, N_search)

    # ─── Stage 2+3: FFT-based 相关 ───
    #   对每个 (g, l): C[g, l, m] = Σ_{n=0..255} y_G[g, m+n] · conj(pss_ref[l, n])
    #
    # 这是带 conj 的滑动相关 = 把 pss_ref reverse 并 conj 后做卷积:
    #   等价于 scipy.signal.fftconvolve(y_G, conj(pss_ref)[::-1], mode='valid')
    #
    # 直接用 numpy:
    #   - 把 pss_ref reverse + conj, 然后跟 y_G 做 'valid' 卷积
    #   - 卷积长度 = N_search + N_MF - 1
    #   - valid 输出长度 = N_search - N_MF + 1 = M + 1 = 153345
    #
    # 这里我们用 FFT-based: 比直接 np.correlate 快 ~100x

    L_fft = 1 << int(np.ceil(np.log2(N_SEARCH + N_MF - 1)))     # 262144

    # FFT y_G (3, L_fft)
    Y_fft = np.fft.fft(y_G, n=L_fft, axis=-1)

    # 对每个 l: H_l = FFT(reverse_conj(pss_ref[l])) = conj(FFT(pss_ref[l]))?
    #   不! 滑动相关 C[m] = Σ y[m+n] · conj(d[n])
    #                    = (y ⊛ conj(d[::-1]))[m + N_MF - 1]   (linear conv)
    #   FFT 上: FFT(conj(d[::-1])) = conj(FFT(d))    (DFT 性质: reverse 在频域是共轭)
    # 等一下,精确写:
    #   令 h[n] = conj(d[N_MF-1 - n]), 则 (y * h)[m+N_MF-1] = Σ_n y[m+N_MF-1-(N_MF-1-n)] conj(d[n])
    #                                                       = Σ_n y[m+n] conj(d[n])  ✓
    #   FFT(h) where h[n] = conj(d[N_MF-1-n]):
    #     = Σ_n conj(d[N_MF-1-n]) e^{-j2πkn/L}
    #     令 m = N_MF-1-n, n=N_MF-1-m:
    #     = Σ_m conj(d[m]) e^{-j2πk(N_MF-1-m)/L}
    #     = e^{-j2πk(N_MF-1)/L} · Σ_m conj(d[m]) e^{j2πkm/L}
    #     = e^{-j2πk(N_MF-1)/L} · conj(Σ_m d[m] e^{-j2πkm/L})
    #     = e^{-j2πk(N_MF-1)/L} · conj(D[k])
    # 所以: FFT(h) = e^{-j2πk(N_MF-1)/L} · conj(FFT(d, n=L))
    #
    # 实现上更简单的等价: 直接 zero-pad d 到 L_fft 然后做相关:
    #   C[m] = IFFT(Y * conj(D))[m]    其中 D = FFT(d, n=L_fft)
    # 因为: Σ_n y[m+n] conj(d[n])  ←→  在频域是 Y[k] · conj(D[k]) (循环相关)
    # 用循环相关; 因为我们 zero-pad 到 L_fft >= N_search+N_MF-1, 循环 = 线性
    # 但输出索引: C[m] = Σ_n y[(m+n) mod L_fft] conj(d[n]) for m ∈ [0, L_fft)
    # valid 区间: m ∈ [0, N_search - N_MF + 1) = [0, M+1)

    pss_padded = np.zeros((N_PSS, L_fft), dtype=np.complex128)
    pss_padded[:, :N_MF] = pss_ref
    D_fft = np.fft.fft(pss_padded, n=L_fft, axis=-1)     # (3, L_fft)

    # 循环相关 = IFFT(Y * conj(D))
    # 广播: Y_fft (3,1,L_fft) × conj(D_fft) (1,3,L_fft) → (3,3,L_fft)
    Cfreq = Y_fft[:, None, :] * np.conj(D_fft)[None, :, :]   # (N_G, N_PSS, L_fft)
    C_full = np.fft.ifft(Cfreq, axis=-1)                      # (N_G, N_PSS, L_fft)

    # 取 valid 区: m ∈ [0, M+1)
    M_OUT = M + 1                                              # 153345
    C = C_full[:, :, :M_OUT]                                   # (3, 3, 153345)

    # ─── Stage 4: |·|² + argmax ───
    metric = (C.real ** 2 + C.imag ** 2).astype(np.float64)    # (3, 3, M+1)

    # argmax over (g, l, m)
    flat_idx = int(np.argmax(metric))
    g_idx, l_idx, mu_t = np.unravel_index(flat_idx, metric.shape)

    g_hat       = int(G_LIST[g_idx])
    n_id_2      = int(l_idx)
    peak        = float(metric[g_idx, l_idx, mu_t])

    # ─── Stage 5: noise_floor = median ───
    #   单峰对 median 几乎无影响
    noise_floor = float(np.median(metric))

    return int(mu_t), n_id_2, g_hat, peak, noise_floor, metric


# =============================================================================
# 测试 case 定义 (Task A: 8 个 case)
# =============================================================================

# (pss_id, cfo_hz, mu_true, snr_db, label)
TEST_CASES = [
    (0,      0.0,  12345, 30.0,   "clean_pss0"),
    (1, +10000.0,  50000, 30.0,   "small_pos_cfo_pss1"),
    (2, -10000.0,  88888, 30.0,   "small_neg_cfo_pss2"),
    (0, +25000.0,   1000, 30.0,   "int_cfo_p1_pss0"),
    (1, -25000.0, 100000, 30.0,   "int_cfo_n1_pss1"),
    (2, +45000.0,  70000, 30.0,   "boundary_pos_pss2"),
    (1,  +5000.0,  60000,  0.0,   "low_snr_0db_pss1"),
    (0,  -8000.0,  30000, -3.0,   "low_snr_n3db_pss0"),
]


def _expected_g(cfo_hz: float) -> int:
    """
    根据 CFO 给出期望 G_hat.

    分界: G=0 区间 [-15kHz, +15kHz), G=+1 区间 [+15kHz, +45kHz], G=-1 区间 [-45kHz, -15kHz)
    边界 ±45kHz 时 |residual| = SCS/2 = 15kHz, 临界但仍单射.
    """
    if cfo_hz >= 15000.0:
        return +1
    elif cfo_hz < -15000.0:
        return -1
    else:
        return 0


def _verdict(case: dict, est_mu: int, est_n2: int, est_g: int,
             peak: float, noise: float) -> Tuple[bool, str]:
    """
    判定单 case 是否 PASS.

    PASS 条件:
      - n_id_2 必须 exact match
      - g_hat  必须 exact match
      - mu_t   允许 ±2 sample 容差 (合成时 SSB 起点 = sym0 CP 起点; PSS useful 在 CP 之后,
               峰位置 ≈ mu_true + CP_FIRST_768 = mu_true + 44)
               但由于 PSS template 是 useful 部分 (256 sample), 不含 CP, 相关峰应在
               mu_true + CP_FIRST_768 = mu_true + 44.
      - peak/noise 比 ≥ 6 dB (峰显著)
    """
    expected_mu  = case["mu_true"] + CP_FIRST_768
    expected_n2  = case["pss_id"]
    expected_g   = _expected_g(case["cfo_hz"])

    mu_ok    = abs(est_mu - expected_mu) <= 2
    n2_ok    = est_n2 == expected_n2
    g_ok     = est_g  == expected_g
    snr_lin  = peak / max(noise, 1e-12)
    snr_ok   = snr_lin >= 4.0  # 6 dB

    passed = mu_ok and n2_ok and g_ok and snr_ok
    reason = []
    if not mu_ok: reason.append(f"mu Δ={est_mu - expected_mu}")
    if not n2_ok: reason.append(f"n_id_2 exp={expected_n2}")
    if not g_ok:  reason.append(f"g exp={expected_g}")
    if not snr_ok: reason.append(f"peak/noise={snr_lin:.2f}<4")

    return passed, ("PASS" if passed else "FAIL:" + ",".join(reason))


# =============================================================================
# 主程序
# =============================================================================

def run_one_case(case_idx: int, pss_id: int, cfo_hz: float,
                 mu_true: int, snr_db: float, label: str,
                 rng_seed: int, out_root: Path,
                 pss_bank_q: np.ndarray, twiddle_q: np.ndarray,
                 save_golden: bool = True) -> dict:
    """
    跑一个 case: 合成 → 注入 CFO → 加噪 → 量化 → 反量化 → 跑算子 → 验证.
    """
    rng = np.random.default_rng(rng_seed)

    # 1. 合成 + CFO + AWGN
    clean    = synth_ssb_window(pss_id, mu_true, rng)
    with_cfo = inject_cfo(clean, cfo_hz)
    noisy    = add_awgn(with_cfo, snr_db, rng)

    # 2. 量化到 cint16 (NPU 输入格式)
    iq_i16 = quantize_to_cint16(noisy)

    # 3. 反量化 (模拟 NPU 内部反量化路径)
    rx = dequantize_from_cint16(iq_i16)

    # 4. 跑算子: 用反量化后的 PSS template + twiddle (跟 NPU 一致)
    pss_ref_dq = dequantize_from_cint16(pss_bank_q.reshape(-1))
    pss_ref_dq = pss_ref_dq.reshape(N_PSS, N_MF)

    twiddle_dq = np.zeros((N_G, N_SEARCH), dtype=np.complex128)
    for gi in range(N_G):
        twiddle_dq[gi] = dequantize_from_cint16(twiddle_q[gi])

    t0 = time.time()
    mu_t, n_id_2, g_hat, peak, noise_floor, _metric = pss_correlator(
        rx, pss_ref=pss_ref_dq, twiddle_cfo=twiddle_dq
    )
    t_elapsed = time.time() - t0

    # 5. 验证
    case = dict(pss_id=pss_id, cfo_hz=cfo_hz, mu_true=mu_true, snr_db=snr_db, label=label)
    passed, msg = _verdict(case, mu_t, n_id_2, g_hat, peak, noise_floor)

    # 6. 落 golden
    if save_golden:
        cfo_tag = f"{int(cfo_hz):+d}".replace("+", "p").replace("-", "n")
        case_dir = out_root / f"case_{case_idx}_pss{pss_id}_cfo{cfo_tag}_t{mu_true}_snr{int(snr_db)}"
        case_dir.mkdir(parents=True, exist_ok=True)

        iq_i16.tofile(case_dir / "input.bin")
        # truth: mu_t, n_id_2, g_hat, peak, noise_floor (float32 简化)
        truth = np.array([
            float(mu_true + CP_FIRST_768),   # 期望 mu_t
            float(pss_id),
            float(_expected_g(cfo_hz)),
            float(peak),                      # 实测 peak (Python ref 给的)
            float(noise_floor),
            float(cfo_hz),                    # 注入的 CFO
            float(snr_db),
        ], dtype=np.float32)
        truth.tofile(case_dir / "truth.bin")

        # case meta
        meta = dict(
            case_idx=case_idx, label=label, pss_id=pss_id,
            cfo_hz=cfo_hz, mu_true=mu_true, snr_db=snr_db,
            expected_mu_t=mu_true + CP_FIRST_768,
            expected_n_id_2=pss_id,
            expected_g_hat=_expected_g(cfo_hz),
            est_mu_t=mu_t, est_n_id_2=n_id_2, est_g_hat=g_hat,
            peak=peak, noise_floor=noise_floor,
            peak_over_noise_db=10 * np.log10(max(peak / max(noise_floor, 1e-12), 1e-12)),
            passed=passed, verdict_msg=msg,
            elapsed_sec=t_elapsed,
        )
        with open(case_dir / "case_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return dict(case_idx=case_idx, label=label,
                inj_cfo=cfo_hz, inj_pss=pss_id, inj_mu=mu_true, inj_snr=snr_db,
                est_mu=mu_t, est_n2=n_id_2, est_g=g_hat,
                peak=peak, noise=noise_floor,
                passed=passed, verdict_msg=msg, elapsed=t_elapsed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default=None,
                   help="golden 输出目录 (覆盖 AIRAN_DATA_DIR)")
    p.add_argument("--seed", type=lambda s: int(s, 0), default=0xC0FE0003)
    p.add_argument("--no-save", action="store_true", help="跑测试但不落 golden")
    p.add_argument("--cases", type=str, default="all", help="逗号分隔索引或 'all'")
    args = p.parse_args()

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = GOLDEN_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    # case 选择
    if args.cases == "all":
        sel = list(range(len(TEST_CASES)))
    else:
        sel = [int(s) for s in args.cases.split(",")]

    print(f"[ref] PSS Correlator numpy clean-room reference")
    print(f"[ref] N_SEARCH={N_SEARCH}, N_MF={N_MF}, M+1={M+1}, N_G={N_G}, N_PSS={N_PSS}")
    print(f"[ref] output:  {out_root}")
    print(f"[ref] seed:    0x{args.seed:08X}")
    print(f"[ref] cases:   {sel}")
    print(f"[ref] 总假设数/case: {N_G * N_PSS * (M+1)} = {N_G*N_PSS*(M+1)/1e6:.1f}M")
    print()

    # 共享常量: PSS bank + twiddle (落一份到 golden 根)
    print("[ref] generating shared constants (pss bank + twiddle)...")
    pss_bank = _pss_template_bank()                          # (3, 256) complex
    pss_bank_q = np.zeros((N_PSS, 2 * N_MF), dtype=np.int16)
    for l in range(N_PSS):
        pss_bank_q[l] = quantize_to_cint16(pss_bank[l])
    pss_bank_q.tofile(out_root / "pss_ref.bin")

    twiddle = gen_cfo_twiddle()                              # (3, N_search) complex
    twiddle_q = np.zeros((N_G, 2 * N_SEARCH), dtype=np.int16)
    for gi in range(N_G):
        twiddle_q[gi] = quantize_to_cint16(twiddle[gi])
    twiddle_q.tofile(out_root / "twiddle.bin")
    print(f"[ref]   pss_ref.bin   : {pss_bank_q.nbytes} bytes")
    print(f"[ref]   twiddle.bin   : {twiddle_q.nbytes} bytes")
    print()

    # 跑 cases
    results = []
    for case_idx in sel:
        pss_id, cfo_hz, mu_true, snr_db, label = TEST_CASES[case_idx]
        # 每个 case 独立 seed
        case_seed = (args.seed + case_idx) & 0xFFFFFFFF
        r = run_one_case(case_idx, pss_id, cfo_hz, mu_true, snr_db, label,
                         case_seed, out_root, pss_bank_q, twiddle_q,
                         save_golden=not args.no_save)
        results.append(r)
        print(f"[case {case_idx}] {label:28s} "
              f"inj(cfo={int(cfo_hz):+6d},pss={pss_id},mu={mu_true:6d},snr={int(snr_db):+3d}) "
              f"→ est(mu={r['est_mu']:6d},n2={r['est_n2']},g={r['est_g']:+d}) "
              f"peak={r['peak']:.2e} noise={r['noise']:.2e} "
              f"[{r['elapsed']:.1f}s] {r['verdict_msg']}")

    # 总结
    n_pass = sum(1 for r in results if r["passed"])
    n_tot  = len(results)
    print()
    print("=" * 110)
    print(f" Verification: {n_pass}/{n_tot} PASS")
    print("=" * 110)
    print(f"{'case':6s} {'label':28s} {'inj_cfo':>8s} {'est_g':>5s} {'inj_pss':>7s} "
          f"{'est_n2':>6s} {'mu_exp':>8s} {'mu_est':>8s} {'p/n_dB':>7s} verdict")
    print("-" * 110)
    for r in results:
        exp_mu = r["inj_mu"] + CP_FIRST_768
        pn_db  = 10 * np.log10(max(r["peak"] / max(r["noise"], 1e-12), 1e-12))
        print(f"{r['case_idx']:6d} {r['label']:28s} {int(r['inj_cfo']):+8d} "
              f"{r['est_g']:+5d} {r['inj_pss']:7d} {r['est_n2']:6d} "
              f"{exp_mu:8d} {r['est_mu']:8d} {pn_db:7.1f} {r['verdict_msg']}")
    print("=" * 110)

    if not args.no_save:
        # 全局 meta
        meta = dict(
            n_search=N_SEARCH, n_mf=N_MF, m_out=M+1,
            n_g=N_G, n_pss=N_PSS,
            fs_hz=FS_HZ, scs_hz=SCS_HZ,
            q_frac_bits=Q_FRAC_BITS, q_scale=Q_SCALE,
            cp_first_768=CP_FIRST_768, cp_other_768=CP_OTHER_768,
            n_ssb_samples=N_SSB_SAMPLES,
            seed=f"0x{args.seed:08X}",
            n_cases=n_tot, n_pass=n_pass,
            cases=[dict(case_idx=r["case_idx"], label=r["label"],
                        passed=r["passed"], msg=r["verdict_msg"]) for r in results],
        )
        with open(out_root / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n[ref] golden saved to: {out_root}")

    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())