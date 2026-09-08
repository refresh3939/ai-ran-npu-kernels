"""
timing_tracker 算法 2: 归一化差分法 (Tretter-style)

算法:
    z[s,k]   = conj(H[s,k]) · H[s,k+1]
    z_norm[s,k] = z[s,k] / |z[s,k]|         <-- 关键改动: 等幅
    c        = Σ_{s,k} z_norm[s,k]
    ΔT       = atan2(c.im, c.re) · N_FFT / (2π)

跟纯差分法对比:
- 强 RE 不再主导累加 (深 fading 的 RE 跟无 fading 的 RE 等权)
- 多径下偏到"等权相位质心", 不是"功率质心"
- 多一次除法 (438 个 sqrt + 除法), 开销可控
- fp16 精度: 归一化后 |z|=1, 累加器值域 [-K, K], fp16 范围内
- 一次 atan2

架构跟 cfo_dmrs 几乎一样, 增加一步 normalization.
"""
import numpy as np


def diff_normalized_ref(H_ls, N_fft=4096, dtype=np.complex128, eps=1e-12):
    """Tretter-style 归一化差分.
    
    eps 防 |z|=0 时除零.
    """
    H = H_ls.astype(dtype)
    z = np.conj(H[:, :-1]) * H[:, 1:]      # [n_sym, K-1]
    mag = np.abs(z) + eps
    z_norm = z / mag
    c = z_norm.sum()
    delta_theta = np.angle(c)
    return delta_theta * N_fft / (2*np.pi)


def diff_normalized_fp16_sim(H_ls, N_fft=4096):
    """fp16 模拟. 归一化后 |z|=1, 累加器值域 [-K, K], fp16 友好."""
    H = H_ls.astype(np.complex128)
    H_re16 = H.real.astype(np.float16).astype(np.float64)
    H_im16 = H.imag.astype(np.float16).astype(np.float64)
    
    def fp16(x): return np.asarray(x).astype(np.float16).astype(np.float64)
    
    re_k   = H_re16[:, :-1]
    im_k   = H_im16[:, :-1]
    re_k1  = H_re16[:, 1:]
    im_k1  = H_im16[:, 1:]
    
    z_re = fp16(fp16(re_k*re_k1) + fp16(im_k*im_k1))
    z_im = fp16(fp16(re_k*im_k1) - fp16(im_k*re_k1))
    
    # 归一化 (fp16 rsqrt 近似)
    mag2 = fp16(fp16(z_re*z_re) + fp16(z_im*z_im))
    mag = fp16(np.sqrt(mag2 + 1e-12))
    z_re_n = fp16(z_re / mag)
    z_im_n = fp16(z_im / mag)
    
    # WholeReduceSum<half> 等效语义
    c_re = fp16(z_re_n.sum())
    c_im = fp16(z_im_n.sum())
    
    delta_theta = np.arctan2(float(c_im), float(c_re))
    return delta_theta * N_fft / (2*np.pi)
