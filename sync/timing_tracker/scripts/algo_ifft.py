"""
timing_tracker 算法 4: IFFT 法 (delay profile peak)

算法:
    h[n] = IFFT(H[k])
    n* = argmax |h[n]|²
    亚样本插值: 抛物线拟合 h[n*-1], h[n*], h[n*+1]

特点:
- 大 timing offset 直接出整数样本 (Layer 1 适用)
- 多径下 peak 位置 = 最强径的 delay, 不是 timing offset
  (这是 spec 模糊点: timing offset 在多径下究竟是哪个?)
- 亚样本需插值, 不天然
- 复杂度高: 220 → 256 (补零) → IFFT
- NPU 实现成本: 一个 256-IFFT, 跟 ofdm_demod matmul-DFT 同款,
  可以复用 v25.6 的 twiddle/matmul 模板, 但是 220 维 K 不对齐麻烦

适用场景判断:
- Layer 1 精时序对齐 (PSS 之后, 已知大致位置 ±N samples) ← yes
- Layer 2 per-slot 跟踪 (ΔT < 1 sample) ← 大材小用, 且亚样本插值精度看运气
"""
import numpy as np


def ifft_ref(H_ls, N_fft=4096, ifft_size=4096, dtype=np.complex128):
    """IFFT delay profile + 抛物线插值.
    
    Args:
        H_ls: [n_sym, K]
        ifft_size: IFFT 长度 (K=220 补零到 ifft_size)
                   分辨率 = N_fft / ifft_size samples.
                   ifft_size = N_fft → 1 sample 分辨率, 亚样本靠抛物线插值.
    """
    H = H_ls.astype(dtype)
    n_sym, K = H.shape
    
    # 两个符号的 H 平均, 再 IFFT (信噪比改善 3 dB)
    H_avg = H.mean(axis=0)              # [K]
    
    # 补零到 ifft_size
    H_pad = np.zeros(ifft_size, dtype=dtype)
    H_pad[:K] = H_avg
    
    h_time = np.fft.ifft(H_pad)
    power = np.abs(h_time)**2
    
    # 整数峰位置
    n_peak = int(np.argmax(power))
    
    # 抛物线插值 (亚样本)
    # 取峰 ±1 (循环边界处理)
    n_m = (n_peak - 1) % ifft_size
    n_p = (n_peak + 1) % ifft_size
    y_m = power[n_m]
    y_0 = power[n_peak]
    y_p = power[n_p]
    
    denom = (y_m - 2*y_0 + y_p)
    if abs(denom) < 1e-12:
        offset = 0.0
    else:
        offset = 0.5 * (y_m - y_p) / denom
    
    n_fine = n_peak + offset
    
    # IFFT 域索引 → 时序漂移 (samples in N_fft 系统采样率)
    delta_T = n_fine * (N_fft / ifft_size)
    
    # IFFT 输出循环, 大于 ifft_size/2 视为负 timing
    if n_fine > ifft_size / 2:
        delta_T -= N_fft  # 转成 [-N/2, +N/2) 区间
    
    # 符号约定: 频域线性相位 exp(+j·2π·k·ΔT/N) 在 IFFT 后峰出现在 -ΔT
    # 所以最终 timing offset = -delta_T
    return -delta_T


def ifft_fp16_sim(H_ls, N_fft=4096, ifft_size=256):
    """fp16 模拟 (用 fp32 IFFT 当占位, NPU 实际是 fp16 matmul-DFT).
    
    这里偷懒: IFFT 用 fp64 算, 但输入 H_ls 已 fp16 量化.
    严格 fp16 IFFT 需要 matmul-based DFT 跟 ofdm_demod_v25.6 同款,
    评测阶段不必这么深.
    """
    H = H_ls.astype(np.complex128)
    re16 = H.real.astype(np.float16).astype(np.float64)
    im16 = H.imag.astype(np.float16).astype(np.float64)
    H16 = re16 + 1j*im16
    
    return ifft_ref(H16, N_fft=N_fft, ifft_size=ifft_size, dtype=np.complex64)
