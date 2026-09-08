"""
timing_tracker 算法 3: 加权最小二乘 (WLS, Yang 2001 IEEE)

模型:
    arg(H[s,k]) = arg(H_ideal[s]) + 2π·ΔT·k/N_FFT + noise
    weight:     w[s,k] = |H[s,k]|²    (Rayleigh 衰落下 ML 权重)

WLS 闭式解 (对每个 s, 然后平均):
    slope = (S_w·S_kp - S_k·S_p) / (S_w·S_kk - S_k²)
    
    其中 S_w  = Σ w
         S_k  = Σ w·k
         S_kk = Σ w·k²
         S_p  = Σ w·φ
         S_kp = Σ w·k·φ
    
    ΔT = slope · N_FFT / (2π)

特点:
- 多径鲁棒 (Rayleigh 衰落下 MSE 比差分法低 1 个量级, Yang 2001 Fig 5)
- 数值友好: 累加的是 w·k·φ, 量级合理 (k ∈ [0, 220), φ ∈ [-π, π])
- 不需要 unwrap (前提: ΔT ∈ ±N/2)
- 代价: K 个 arctan2 (CANN 8.0.0 lib/math/atan.h 提供)
"""
import numpy as np


def wls_ref(H_ls, N_fft=4096, dtype=np.float64):
    """WLS timing 估计 (unwrap + 加权线性回归).
    
    对每个符号独立做 WLS, 再用功率加权平均.
    
    用 np.unwrap 处理跨 ±π 边界, 这样对 ΔT 在 ±N_fft/2 范围内都稳.
    """
    H = H_ls.astype(np.complex128)
    n_sym, K = H.shape
    k = np.arange(K, dtype=dtype)
    
    # 先除掉每个符号的全局相位 (用 k=0 的 H 做参考)
    # 这样 unwrap 起点固定, 减少漂移
    H_ref = H * np.conj(H[:, 0:1]) / (np.abs(H[:, 0:1]) + 1e-12)
    
    phi = np.unwrap(np.angle(H_ref), axis=1).astype(dtype)   # [n_sym, K]
    w = (np.abs(H)**2).astype(dtype)                          # [n_sym, K]
    
    slopes = []
    weights = []
    for s in range(n_sym):
        ws = w[s]
        ps = phi[s]
        S_w  = ws.sum()
        S_k  = (ws * k).sum()
        S_kk = (ws * k * k).sum()
        S_p  = (ws * ps).sum()
        S_kp = (ws * k * ps).sum()
        
        denom = S_w * S_kk - S_k * S_k
        if abs(denom) < 1e-12:
            slope = 0.0
        else:
            slope = (S_w * S_kp - S_k * S_p) / denom
        slopes.append(slope)
        weights.append(S_w)
    
    slopes = np.array(slopes)
    weights = np.array(weights)
    slope_avg = (slopes * weights).sum() / weights.sum()
    
    return slope_avg * N_fft / (2*np.pi)


def wls_fp16_sim(H_ls, N_fft=4096):
    """fp16 模拟 (向量化).
    
    每个中间张量量化到 fp16, 累加后再量化. 这模拟 NPU 一次 vector 操作 fp16
    输入 → fp16 输出的行为. 累加器内部精度由 WholeReduceSum<half> 决定,
    经验上比纯 fp16 stepping 稍好.
    """
    H = H_ls.astype(np.complex128)
    re16 = H.real.astype(np.float16).astype(np.float64)
    im16 = H.imag.astype(np.float16).astype(np.float64)
    
    n_sym, K = H.shape
    k = np.arange(K, dtype=np.float64)
    k16 = k.astype(np.float16).astype(np.float64)
    
    def fp16(x): return np.asarray(x).astype(np.float16).astype(np.float64)
    
    # arctan2 (输入 fp16, 输出 fp16)
    phi = np.arctan2(im16, re16)
    phi16 = fp16(phi)
    
    # |H|²
    w16 = fp16(fp16(re16*re16) + fp16(im16*im16))
    
    slopes = []
    weights = []
    for s in range(n_sym):
        ws = w16[s]
        ps = phi16[s]
        # 向量化乘法 (每步 fp16), 一次性累加 (np.sum 在 fp64 里, 但累加器
        # 的"sum" 行为模拟 WholeReduceSum<half> 的总体效果 — 实际 NPU 上
        # 累加器宽度 ~ fp22, 比纯 fp16 stepping 稍好)
        S_w  = fp16(np.sum(ws))
        S_k  = fp16(np.sum(fp16(ws * k16)))
        S_kk_terms = fp16(ws * fp16(k16*k16))   # 注意: k² 最大 ~48k, 接近 fp16 上限
        S_kk = fp16(np.sum(S_kk_terms))
        S_p  = fp16(np.sum(fp16(ws*ps)))
        S_kp = fp16(np.sum(fp16(fp16(ws*k16)*ps)))
        
        denom = float(S_w*S_kk - S_k*S_k)
        if abs(denom) < 1e-6:
            slopes.append(0.0)
        else:
            slope = float(S_w*S_kp - S_k*S_p) / denom
            slopes.append(slope)
        weights.append(float(S_w))
    
    slopes = np.array(slopes)
    weights = np.array(weights)
    slope_avg = (slopes*weights).sum() / weights.sum()
    return slope_avg * N_fft / (2*np.pi)
