"""
timing_tracker 评测公共工具

提供:
- 单径/多径信道模型
- DMRS H_ls 仿真生成 (复用 cfo_dmrs 的 220 子载波结构)
- fp16/fp64 精度转换
- 评测指标
"""
import numpy as np


# === 系统参数 (跟 LAYER12_DEV_GUIDE.md 对齐) ===
N_FFT = 4096            # 系统 FFT 长度
FS = 30.72e6            # 系统采样率
K_DMRS = 220            # DMRS 子载波数 (cfo_dmrs 既有)
N_SYM_PAIR = 2          # sym 2 和 sym 11


def gen_ideal_Hls(K=K_DMRS, n_sym=N_SYM_PAIR, seed=None):
    """生成理想 H_ls (单径平坦信道, 单位幅度).
    
    单径平坦信道: H(k) ≡ H0 (常数 across k), 每个符号一个全局相位.
    timing offset 通过 apply_timing_offset 在频域加.
    """
    rng = np.random.default_rng(seed)
    # 每个符号一个全局相位 (不同符号 H0 可以不同, 模拟 CFO 跨符号旋转)
    phase_per_sym = rng.uniform(-np.pi, np.pi, size=(n_sym, 1))
    H = np.ones((n_sym, K), dtype=np.complex128) * np.exp(1j * phase_per_sym)
    return H


def gen_multipath_Hls(K=K_DMRS, n_sym=N_SYM_PAIR, n_taps=3, max_delay=10,
                     seed=None):
    """多径信道下的 H_ls.
    
    多径在频域上产生非线性相位 (frequency-selective fading).
    用于检验 timing 估计器在多径下的鲁棒性.
    """
    rng = np.random.default_rng(seed)
    # 多径功率延迟谱 (指数衰减)
    delays = rng.uniform(0, max_delay, n_taps)
    amps = rng.rayleigh(scale=1.0/np.sqrt(n_taps), size=n_taps) * \
           np.exp(-delays/max_delay)
    phases = rng.uniform(-np.pi, np.pi, n_taps)
    
    k = np.arange(K)
    H = np.zeros((n_sym, K), dtype=np.complex128)
    for s in range(n_sym):
        # 多径 H(k) = Σ a_l · exp(jφ_l) · exp(-j2π·τ_l·k/N_FFT)
        for l in range(n_taps):
            H[s] += amps[l] * np.exp(1j * phases[l]) * \
                    np.exp(-1j * 2*np.pi * delays[l] * k / N_FFT)
    return H


def apply_timing_offset(H, delta_T_samples, N=N_FFT):
    """在频域信道上施加 timing offset.
    
    时序漂移 ΔT samples → 频域线性相位 exp(j·2π·k·ΔT/N).
    """
    K = H.shape[-1]
    k = np.arange(K)
    return H * np.exp(1j * 2*np.pi * k * delta_T_samples / N)


def add_awgn(H, snr_db):
    """加 AWGN, SNR 按 H 平均功率定义."""
    sig_pow = np.mean(np.abs(H)**2)
    noise_pow = sig_pow / (10**(snr_db/10))
    noise = np.sqrt(noise_pow/2) * (
        np.random.randn(*H.shape) + 1j*np.random.randn(*H.shape)
    )
    return H + noise


def to_fp16_complex(H):
    """模拟 NPU fp16 复数存储: 实部虚部各 fp16."""
    re = H.real.astype(np.float16).astype(np.float64)
    im = H.imag.astype(np.float16).astype(np.float64)
    return re + 1j*im


def make_test_case(delta_T_true, snr_db, channel='single',
                   K=K_DMRS, n_sym=N_SYM_PAIR, seed=None):
    """生成一个完整测试 case.
    
    Returns:
        H_ls_clean: fp64 ground truth (含 timing offset, 无噪声)
        H_ls_noisy: fp64 含噪声
        H_ls_fp16:  fp16 量化后 (模拟 NPU 输入)
    """
    if channel == 'single':
        H = gen_ideal_Hls(K, n_sym, seed=seed)
    elif channel == 'multipath':
        H = gen_multipath_Hls(K, n_sym, seed=seed)
    else:
        raise ValueError(f"unknown channel: {channel}")
    
    H_offset = apply_timing_offset(H, delta_T_true)
    # AWGN seed 跟信道分开, 这样同一信道可以跑不同 SNR realization
    H_noisy = add_awgn(H_offset, snr_db)
    H_fp16 = to_fp16_complex(H_noisy)
    
    return H_offset, H_noisy, H_fp16


# === 评测指标 ===
def compute_metrics(estimates, truths):
    """estimates, truths: array, 同长度."""
    estimates = np.asarray(estimates)
    truths = np.asarray(truths)
    err = estimates - truths
    return {
        'rmse': float(np.sqrt(np.mean(err**2))),
        'bias': float(np.mean(err)),
        'std':  float(np.std(err)),
        'max_abs_err': float(np.max(np.abs(err))),
    }
