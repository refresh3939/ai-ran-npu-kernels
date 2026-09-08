"""
timing_tracker 算法 1: 相邻子载波差分法 (LAYER12_DEV_GUIDE.md 选定方案)

算法:
    z[s,k]   = conj(H[s,k]) · H[s,k+1]
    c        = Σ_{s,k} z[s,k]
    ΔT       = atan2(c.im, c.re) · N_FFT / (2π)

特点:
- 最简洁, 最廉价
- 一次 atan2
- 跟 cfo_dmrs 架构高度一致
- 已知缺陷: fp16 累加器精度 + 多径下估计偏到质心 delay
"""
import numpy as np


def diff_ref(H_ls, N_fft=4096, dtype=np.complex128):
    """guide 默认算法.
    
    Args:
        H_ls: shape [n_sym, K], 复数 LS 信道估计
        N_fft: 系统 FFT 长度
        dtype: 中间累加用的复数类型 (np.complex128 = fp64, np.complex64 = fp32)
    
    Returns:
        delta_T_samples: float
    """
    H = H_ls.astype(dtype)
    product = np.conj(H[:, :-1]) * H[:, 1:]   # [n_sym, K-1]
    c = product.sum()
    delta_theta = np.angle(c)
    return delta_theta * N_fft / (2*np.pi)


def diff_ref_fp16_sim(H_ls, N_fft=4096):
    """模拟 NPU fp16 实现.
    
    每个 conj·prod 用 fp16, 累加用 WholeReduceSum<half> 等效语义
    (numpy 向量化 sum 之后再 fp16 round, 模拟硬件累加器有效宽度).
    cfo_dmrs MSE=8.7e-6 经验值就是这种语义下达到的.
    """
    H = H_ls.astype(np.complex128)
    H_re16 = H.real.astype(np.float16).astype(np.float64)
    H_im16 = H.imag.astype(np.float16).astype(np.float64)
    
    def fp16(x): return np.asarray(x).astype(np.float16).astype(np.float64)
    
    re_k   = H_re16[:, :-1]
    im_k   = H_im16[:, :-1]
    re_k1  = H_re16[:, 1:]
    im_k1  = H_im16[:, 1:]
    
    # conj(H[k])·H[k+1] 的 4 个乘法 + 2 个加法, 每步 fp16
    z_re = fp16(fp16(re_k*re_k1) + fp16(im_k*im_k1))
    z_im = fp16(fp16(re_k*im_k1) - fp16(im_k*re_k1))
    
    # WholeReduceSum<half> 等效: numpy sum 之后 round 到 fp16
    c_re = fp16(z_re.sum())
    c_im = fp16(z_im.sum())
    
    delta_theta = np.arctan2(float(c_im), float(c_re))
    return delta_theta * N_fft / (2*np.pi)
