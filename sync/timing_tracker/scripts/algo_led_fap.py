"""
timing_tracker 算法: LED-FAP (Leading-Edge Detection First-Arrival-Path)

5G 商用 / srsRAN / OAI 范式. 多径下找"第一径"而非"质心",
符合 OFDM 符号该对齐 CP 起点的物理意义.

算法:
    Step 1. 时域平均两个 DMRS 符号 (3 dB SNR 增益)
            H_avg[k] = (H_ls[0,k] + H_ls[1,k]) / 2

    Step 2. 零填充 + IFFT 到 CIR
            H_pad[N_fft] = [H_avg[0..K-1], 0, ..., 0]   (220 → 4096)
            h_cir[n]     = IFFT{H_pad}                  (复 fp16, 长 4096)
            p[n]         = |h_cir[n]|²                  (功率, fp32)

    Step 3. 噪声底估计
            noise_floor  = median(p) * fudge_factor
            threshold    = noise_floor * K_factor       (K_factor=4 即 6 dB)

    Step 4. Leading-edge detection (限定搜索窗 ±W samples, W ~ CP/4)
            windows: n ∈ [N_fft-W, N_fft) ∪ [0, W)      (循环卷积处理负 timing)
            n_first  = min{ n ∈ windows : p[n] > threshold }

    Step 5. 抛物线插值 (亚样本精度)
            n_fine   = n_first + 0.5·(p[n-1]-p[n+1]) / (p[n-1]-2·p[n]+p[n+1])

    Step 6. 符号转换
            if n_fine > N_fft/2: ΔT = n_fine - N_fft    (负 timing)
            else:                ΔT = n_fine

Refs:
    - srsRAN: pusch_channel_estimator_*, CIR-based first-path
    - Qualcomm US Patent 7907593 (timing detector + tracking loop)
    - DRM/DVB-T2 学术: IFFT-based CIR first-path with parabolic interp
"""
import numpy as np


# 默认参数 (5G NR FR1, 30 kHz SCS, N_fft=4096, CP=288)
DEFAULT_N_FFT = 4096
DEFAULT_SEARCH_WINDOW = 72        # ~ CP/4, per-slot 漂移上限
DEFAULT_REL_THRESHOLD_DB = -12.0  # threshold = max(p) × 10^(rel_db/10), 即主峰下 12 dB
DEFAULT_NOISE_FLOOR_DB = 6.0      # 同时要求 threshold > noise_floor × 10^(K/10)
DEFAULT_NOISE_FUDGE = 1.2


def led_fap_ref(H_ls,
                N_fft=DEFAULT_N_FFT,
                search_window=DEFAULT_SEARCH_WINDOW,
                rel_threshold_db=DEFAULT_REL_THRESHOLD_DB,
                noise_floor_db=DEFAULT_NOISE_FLOOR_DB,
                noise_fudge=DEFAULT_NOISE_FUDGE,
                dtype=np.complex128):
    """LED-FAP timing 估计.
    
    Args:
        H_ls:             shape [n_sym, K], 复数 LS 信道估计
        N_fft:            系统 FFT 长度 (CIR 分辨率 = 1 sample)
        search_window:    ±W samples, 限定 leading-edge 搜索范围
        rel_threshold_db: threshold = max(p) × 10^(rel_db/10), 主峰下多少 dB
                          (srsRAN 默认 -12 dB, 即首径要 ≥ 主峰的 1/16 功率)
        noise_floor_db:   threshold_abs = noise_floor × 10^(K/10), 同时要求
                          threshold = max(rel_thr, abs_thr) 防止纯噪声场景误判
        noise_fudge:      median → noise_floor 修正
        dtype:            IFFT 内部精度
    
    Returns:
        delta_T_samples: float
    """
    H = H_ls.astype(dtype)
    n_sym, K = H.shape

    # ── Step 1. 两个符号时域平均 ────────────────────────────────────
    H_avg = H.mean(axis=0)                            # [K]

    # ── Step 2. 零填充 IFFT 到 CIR ──────────────────────────────────
    H_pad = np.zeros(N_fft, dtype=dtype)
    H_pad[:K] = H_avg
    h_cir = np.fft.ifft(H_pad)                        # [N_fft], complex
    p = (h_cir.real**2 + h_cir.imag**2).astype(np.float64)

    # ── Step 3. 阈值 (相对主峰 + 绝对噪声底, 取最大) ────────────────
    W = search_window
    # 在搜索窗内找主峰 (避开远端 echo)
    tau_indices = np.concatenate([
        np.arange(N_fft - W, N_fft),
        np.arange(0, W)
    ])
    p_window = p[tau_indices]
    tau_values = np.concatenate([np.arange(-W, 0), np.arange(0, W)]).astype(np.float64)

    p_max = float(p_window.max())
    threshold_rel = p_max * 10**(rel_threshold_db / 10.0)
    
    noise_floor = float(np.median(p)) * noise_fudge
    threshold_abs = noise_floor * 10**(noise_floor_db / 10.0)
    
    threshold = max(threshold_rel, threshold_abs)

    # ── Step 4. Leading-edge detection ──────────────────────────────
    above = p_window > threshold
    if not above.any():
        idx = int(np.argmax(p_window))
    else:
        idx = int(np.argmax(above))      # 第一个 True

    # ── Step 5. 抛物线插值 ───────────────────────────────────────────
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

    delta_T = tau_values[idx] + offset

    return float(delta_T)


def led_fap_fp16_sim(H_ls,
                     N_fft=DEFAULT_N_FFT,
                     search_window=DEFAULT_SEARCH_WINDOW,
                     rel_threshold_db=DEFAULT_REL_THRESHOLD_DB,
                     noise_floor_db=DEFAULT_NOISE_FLOOR_DB,
                     noise_fudge=DEFAULT_NOISE_FUDGE):
    """fp16 模拟.
    
    关键: IFFT 在 NPU 上需要 matmul-DFT (ofdm_demod_v25.6 同款), 长度 N_fft=4096
    在 dav_m200 上是大开销. 实际部署会选 IFFT_size = max search range × 2,
    比如 256, 牺牲一些精度换性能. 这里 ref 用 N_fft 保留精度上限.
    """
    H = H_ls.astype(np.complex128)
    re16 = H.real.astype(np.float16).astype(np.float64)
    im16 = H.imag.astype(np.float16).astype(np.float64)
    H16 = re16 + 1j*im16

    def fp16(x): return np.asarray(x).astype(np.float16).astype(np.float64)

    # Step 1: 平均
    H_avg = H16.mean(axis=0)
    H_avg = fp16(H_avg.real) + 1j*fp16(H_avg.imag)

    # Step 2: 零填充 + IFFT
    H_pad = np.zeros(N_fft, dtype=np.complex128)
    H_pad[:H_ls.shape[1]] = H_avg
    h_cir = np.fft.ifft(H_pad)
    h_re = fp16(h_cir.real)
    h_im = fp16(h_cir.imag)

    # Step 3: |h|²
    p = fp16(fp16(h_re*h_re) + fp16(h_im*h_im))

    # Step 4: 阈值
    W = search_window
    tau_indices = np.concatenate([
        np.arange(N_fft - W, N_fft),
        np.arange(0, W)
    ])
    p_window = p[tau_indices]
    tau_values = np.concatenate([np.arange(-W, 0), np.arange(0, W)]).astype(np.float64)

    p_max = float(p_window.max())
    threshold_rel = p_max * 10**(rel_threshold_db / 10.0)
    
    noise_floor = float(np.median(p)) * noise_fudge
    threshold_abs = noise_floor * 10**(noise_floor_db / 10.0)
    
    threshold = max(threshold_rel, threshold_abs)

    above = p_window > threshold
    if not above.any():
        idx = int(np.argmax(p_window))
    else:
        idx = int(np.argmax(above))

    # Step 5: 抛物线插值
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

    return float(tau_values[idx] + offset)
