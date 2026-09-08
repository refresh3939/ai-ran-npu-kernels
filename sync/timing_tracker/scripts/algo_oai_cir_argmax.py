"""
timing_tracker 算法: OAI CIR-argmax

3GPP / 5G 商用范式. 依据:

1. OAI positioning testbed (arXiv 2508.19736):
   "The CIR h_m,t is computed via IDFT from the estimated Channel Frequency
   Response (CFR) vector w_m,t. The ToA of the signal at antenna m and time
   index t is estimated as the delay corresponding to the strongest multipath
   component, identified by the peak magnitude of the CIR vector."

2. OAI positioning system model (arXiv 2508.19736 同源):
   "The CIR h_{m,t} is computed via IDFT from the estimated Channel Frequency
   Response (CFR) vector w_{m,t}."

3. DRM/DVB-T 学术综述 (IEEE):
   "OFDM symbol synchronization based on channel estimation. The first path
   time delay of channel determines symbol timing offset. We perform IFFT to
   obtain channel impulse response (CIR). Zero-padding and windowing are used
   to increase accuracy before IFFT."

算法:
    1. H_avg[k] = (H_ls[0,k] + H_ls[1,k]) / 2     # 时域平均 2 个 DMRS 符号
    2. H_pad[N_fft] = [H_avg, 0..0]               # 零填充到 N_fft
    3. h[n] = IFFT{H_pad}                         # CIR (channel impulse response)
    4. p[n] = |h[n]|²                             # 功率延迟谱 (PDP)
    5. n_peak = argmax(p[n])  在 [-W, W) 内       # 最强径 (OAI 做法)
    6. n_fine = n_peak + parabolic_interp         # 亚样本精度
    7. delta_T = n_fine 映回 [-N/2, N/2)

为什么 argmax 不是 first-path:
- 3GPP TS 38.533 标准 timing 参考点是 "first detected path"
- OAI 实际开源代码用 argmax (strongest path) 简化
- LoS 主导场景下 argmax == first path
- NLoS 多径下 argmax 偏向"质心 delay", 偏差被闭环 TA 命令吸收
- argmax 简单稳健, LED + threshold 在小子载波数 (K << N_fft) 下不可靠

为什么搜索窗 [-W, W):
- per-slot 跟踪场景, timing 漂移 < 1 sample/slot
- W ~ CP/4 = 72 samples 已经覆盖任何合理漂移
- 限定窗避免远端 echo / aliasing 干扰
"""
import numpy as np


# 默认参数 (5G NR FR1, 30 kHz SCS, N_fft=4096, CP=288)
DEFAULT_N_FFT = 4096
DEFAULT_SEARCH_WINDOW = 72         # ~ CP/4, per-slot 漂移上限


def oai_cir_argmax_ref(H_ls,
                       N_fft=DEFAULT_N_FFT,
                       search_window=DEFAULT_SEARCH_WINDOW,
                       dtype=np.complex128):
    """OAI CIR-argmax timing 估计.
    
    Args:
        H_ls:           shape [n_sym, K], 复数 LS 信道估计
        N_fft:          系统 FFT 长度 (CIR 分辨率 = 1 sample)
        search_window:  ±W samples, 搜索范围
        dtype:          IFFT 内部精度
    
    Returns:
        delta_T_samples: float
    """
    H = H_ls.astype(dtype)
    n_sym, K = H.shape

    # ── Step 1. 时域平均 ────────────────────────────────────────────
    # 假设 sym 2 和 sym 11 timing 几乎相同 (per-slot 漂移 << 1 sample)
    H_avg = H.mean(axis=0)                            # [K]

    # ── Step 2. 零填充 ──────────────────────────────────────────────
    H_pad = np.zeros(N_fft, dtype=dtype)
    H_pad[:K] = H_avg

    # ── Step 3. IFFT 得 CIR ─────────────────────────────────────────
    h_cir = np.fft.ifft(H_pad)                        # [N_fft], complex

    # ── Step 4. 功率延迟谱 ──────────────────────────────────────────
    p = (h_cir.real**2 + h_cir.imag**2).astype(np.float64)

    # ── Step 5. argmax 在搜索窗内 ────────────────────────────────────
    # 循环重排: tau ∈ [-W, +W) 对应 p 的 [N_fft-W..N_fft, 0..W)
    W = search_window
    tau_indices = np.concatenate([
        np.arange(N_fft - W, N_fft),                  # 负 timing
        np.arange(0, W)                               # 正 timing
    ])
    p_window = p[tau_indices]                         # [2W]
    tau_values = np.concatenate([
        np.arange(-W, 0),
        np.arange(0, W)
    ]).astype(np.float64)

    idx = int(np.argmax(p_window))                    # OAI 做法: 直接 argmax

    # ── Step 6. 抛物线插值 (亚样本) ─────────────────────────────────
    if 0 < idx < 2*W - 1:
        y_m = p_window[idx - 1]
        y_0 = p_window[idx]
        y_p = p_window[idx + 1]
        denom = (y_m - 2*y_0 + y_p)
        if abs(denom) > 1e-12 and y_0 > y_m and y_0 > y_p:
            offset = 0.5 * (y_m - y_p) / denom
        else:
            offset = 0.0
    else:
        offset = 0.0

    # ── Step 7. 输出带符号的 delta_T ─────────────────────────────────
    # 符号约定: 频域线性相位 exp(+j·2π·k·ΔT/N) 在 IFFT 后峰出现在 -ΔT,
    # 所以输出取反才是真实 timing offset
    delta_T = -(tau_values[idx] + offset)

    return float(delta_T)


def oai_cir_argmax_fp16_sim(H_ls,
                            N_fft=DEFAULT_N_FFT,
                            search_window=DEFAULT_SEARCH_WINDOW):
    """fp16 量化模拟.
    
    NPU 实现注意:
    - IFFT 用 matmul-based DFT (ofdm_demod_v25.6 模板可复用)
    - 全 N_fft=4096 IFFT 太重, 实际部署可用 N_fft=256 zoom-IFFT
      (代价: 时域分辨率 = N_fft/256 = 16 samples, 亚样本插值精度下降)
    - 这里 ref 用 fp64 IFFT, 仅模拟输入/输出量化, 是精度上限参考
    
    fp16 主要误差来源:
    - H_avg 累加 (2 项, 影响小)
    - IFFT matmul: 输入 fp16, Cube 累加器 fp22, 输出 fp16
    - |h|² 平方累加 (re² + im²)
    - 比较器 argmax (fp16 比较 OK)
    """
    H = H_ls.astype(np.complex128)
    re16 = H.real.astype(np.float16).astype(np.float64)
    im16 = H.imag.astype(np.float16).astype(np.float64)
    H16 = re16 + 1j*im16

    def fp16(x): return np.asarray(x).astype(np.float16).astype(np.float64)

    # Step 1: 平均
    H_avg = H16.mean(axis=0)
    H_avg = fp16(H_avg.real) + 1j*fp16(H_avg.imag)

    # Step 2-3: 零填充 + IFFT
    H_pad = np.zeros(N_fft, dtype=np.complex128)
    H_pad[:H_ls.shape[1]] = H_avg
    h_cir = np.fft.ifft(H_pad)
    h_re = fp16(h_cir.real)
    h_im = fp16(h_cir.imag)

    # Step 4: |h|²
    p = fp16(fp16(h_re*h_re) + fp16(h_im*h_im))

    # Step 5: argmax
    W = search_window
    tau_indices = np.concatenate([
        np.arange(N_fft - W, N_fft),
        np.arange(0, W)
    ])
    p_window = p[tau_indices]
    tau_values = np.concatenate([np.arange(-W, 0), np.arange(0, W)]).astype(np.float64)

    idx = int(np.argmax(p_window))

    # Step 6: 抛物线插值
    if 0 < idx < 2*W - 1:
        y_m = p_window[idx - 1]
        y_0 = p_window[idx]
        y_p = p_window[idx + 1]
        denom = (y_m - 2*y_0 + y_p)
        if abs(denom) > 1e-12 and y_0 > y_m and y_0 > y_p:
            offset = 0.5 * (y_m - y_p) / denom
        else:
            offset = 0.0
    else:
        offset = 0.0

    return float(-(tau_values[idx] + offset))
