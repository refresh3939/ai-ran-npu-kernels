"""
timing_tracker 4 算法精度评测.

场景:
    A. 单径平坦 + 不同 SNR
    B. 单径平坦 + 不同 ΔT
    C. 多径
    D. fp16 量化下的精度损失

输出: 完整对比表
"""
import numpy as np
import time
from common import make_test_case, compute_metrics, N_FFT, K_DMRS

from algo_diff import diff_ref, diff_ref_fp16_sim
from algo_diff_norm import diff_normalized_ref, diff_normalized_fp16_sim
from algo_wls import wls_ref, wls_fp16_sim
from algo_ifft import ifft_ref, ifft_fp16_sim


# === 评测配置 ===
N_TRIALS = 100      # 每个 (算法, 场景) 跑多少次 Monte Carlo

ALGOS_FP64 = [
    ('diff',       diff_ref),
    ('diff_norm',  diff_normalized_ref),
    ('wls',        wls_ref),
    ('ifft',       ifft_ref),
]

# fp16 仿真版 (慢, 用少量 trials)
N_TRIALS_FP16 = 30
ALGOS_FP16 = [
    ('diff',       diff_ref_fp16_sim),
    ('diff_norm',  diff_normalized_fp16_sim),
    ('wls',        wls_fp16_sim),
    ('ifft',       ifft_fp16_sim),
]


def run_scenario(name, delta_T_list, snr_db, channel, algos, n_trials,
                 input_dtype='clean'):
    """跑一个场景.
    
    Args:
        delta_T_list: 列表, 每个 trial 用一个 ΔT (或全相同)
        input_dtype: 'clean' / 'noisy' / 'fp16' (传给 algo 的 H_ls 用哪一份)
    """
    results = {n: {'est': [], 'truth': []} for n, _ in algos}
    
    for trial in range(n_trials):
        # 选 ΔT
        dt_true = delta_T_list[trial % len(delta_T_list)]
        H_clean, H_noisy, H_fp16 = make_test_case(
            dt_true, snr_db, channel=channel,
            K=K_DMRS, n_sym=2, seed=trial*1000 + hash(name)%999,
        )
        if input_dtype == 'clean':
            H_in = H_clean
        elif input_dtype == 'noisy':
            H_in = H_noisy
        elif input_dtype == 'fp16':
            H_in = H_fp16
        
        for algo_name, algo_fn in algos:
            try:
                est = algo_fn(H_in)
            except Exception as e:
                est = np.nan
            results[algo_name]['est'].append(est)
            results[algo_name]['truth'].append(dt_true)
    
    print(f"\n=== {name} ===")
    print(f"  channel={channel}, snr={snr_db} dB, input={input_dtype}, "
          f"n_trials={n_trials}")
    print(f"  {'algo':<12} {'rmse':>10} {'bias':>10} {'std':>10} {'max_err':>10}")
    metrics_table = {}
    for algo_name, _ in algos:
        m = compute_metrics(results[algo_name]['est'],
                            results[algo_name]['truth'])
        metrics_table[algo_name] = m
        print(f"  {algo_name:<12} {m['rmse']:>10.4e} {m['bias']:>+10.3e} "
              f"{m['std']:>10.4e} {m['max_abs_err']:>10.4e}")
    return metrics_table


def main():
    print("="*70)
    print("timing_tracker 算法精度评测 (numpy reference)")
    print(f"K_DMRS={K_DMRS}, N_FFT={N_FFT}, n_trials={N_TRIALS}")
    print("="*70)
    
    rng = np.random.default_rng(42)
    
    # ────────────────────────────────────────────────────────────
    # 场景 A: 单径 + 不同 SNR + 中等 ΔT (0.5 samples)
    # ────────────────────────────────────────────────────────────
    print("\n\n##### Scenario A: 单径 vs SNR (ΔT_true = 0.5 samples) #####")
    for snr in [30, 20, 10, 0, -3]:
        run_scenario(f"A_snr{snr}", [0.5], snr, 'single',
                     ALGOS_FP64, N_TRIALS, input_dtype='noisy')
    
    # ────────────────────────────────────────────────────────────
    # 场景 B: 单径 + 不同 ΔT + 高 SNR (无噪声极限)
    # ────────────────────────────────────────────────────────────
    print("\n\n##### Scenario B: 单径 vs ΔT (clean, fp64, 看算法本身偏差) #####")
    for dt in [0.001, 0.01, 0.1, 1.0, 10.0]:
        run_scenario(f"B_dt{dt}", [dt], 999, 'single',
                     ALGOS_FP64, N_TRIALS, input_dtype='clean')
    
    # ────────────────────────────────────────────────────────────
    # 场景 C: 多径
    # ────────────────────────────────────────────────────────────
    print("\n\n##### Scenario C: 多径 (3 taps, max_delay=5 samples) #####")
    for snr in [30, 10, 0]:
        # 多径下 ΔT 定义稍模糊, 用 0 (信道内禀延迟模拟 timing offset)
        run_scenario(f"C_mp_snr{snr}", [0.0], snr, 'multipath',
                     ALGOS_FP64, N_TRIALS, input_dtype='noisy')
    
    # ────────────────────────────────────────────────────────────
    # 场景 D: fp16 量化精度评估
    # ────────────────────────────────────────────────────────────
    print("\n\n##### Scenario D: fp16 量化 vs ΔT (clean input, "
          "看 fp16 累加器精度) #####")
    print("  (慢, 每场景 n_trials=%d)" % N_TRIALS_FP16)
    for dt in [0.01, 0.1, 1.0]:
        t0 = time.time()
        run_scenario(f"D_fp16_dt{dt}", [dt], 999, 'single',
                     ALGOS_FP16, N_TRIALS_FP16, input_dtype='clean')
        print(f"  ({time.time()-t0:.1f}s)")
    
    # ────────────────────────────────────────────────────────────
    # 场景 E: 工作点 — SFO 1ppm 等价 ΔT
    # ────────────────────────────────────────────────────────────
    # SFO 1ppm @ fs=30.72MHz, slot=0.5ms → ΔT/slot = 1e-6 × 30.72e6 × 0.5e-3
    #                                              = 0.01536 samples
    print("\n\n##### Scenario E: 工作点 SFO=1ppm (ΔT≈0.015 samples) #####")
    dt_work = 0.015
    for snr in [30, 10, 0]:
        run_scenario(f"E_work_snr{snr}", [dt_work], snr, 'single',
                     ALGOS_FP64, N_TRIALS, input_dtype='noisy')
    
    print("\n" + "="*70)
    print("评测结束")
    print("="*70)


if __name__ == '__main__':
    main()
