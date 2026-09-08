"""
verify_result.py — PSS Correlator 端到端 verify

支持指定 case: python3 verify_result.py [case_dir_name]
默认: case_0_pss0_cfop0_t12345_snr30

NPU 输出: reduce[g=0..2][aiv*300+tile=0..1199][16 fp32 = 12 valid + 4 pad]
   valid 12 = 3 l × (peak, idx, sum, count)
Host 做 final reduce → 5 scalar (g_hat, n_id_2, mu_t, peak, noise_floor)
"""
import numpy as np
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from pss_correlator_ref import N_SEARCH, N_G, N_MF, N_PSS, dequantize_from_cint16

KERNEL_DIR  = _THIS.parent.parent
METRIC_BIN  = KERNEL_DIR / "data" / "ascend_output" / "stage23_metric.bin"
CASE_ROOT   = KERNEL_DIR / "data" / "golden" / "rx" / "pss_correlator"

M_PER_AIV     = 38336
TOTAL_TILES   = 300
M_TILE        = 128
PER_TILE_FP32 = 16
N_VALID_FIELDS = 12
CP_FIRST_768  = 44


def do_final_reduce(metric_bin_path):
    raw = np.fromfile(metric_bin_path, dtype=np.float32)
    expected = N_G * 4 * TOTAL_TILES * PER_TILE_FP32
    if raw.size != expected:
        raise RuntimeError(f"size {raw.size} != expected {expected}")

    arr_with_pad = raw.reshape(N_G, 4, TOTAL_TILES, PER_TILE_FP32)
    arr = arr_with_pad[:, :, :, :N_VALID_FIELDS].reshape(N_G, 4, TOTAL_TILES, N_PSS, 4)

    peaks    = arr[:, :, :, :, 0]
    idxs_loc = arr[:, :, :, :, 1].astype(np.int32)
    sums     = arr[:, :, :, :, 2]
    counts   = arr[:, :, :, :, 3]

    TAIL_MU = 64
    is_valid = np.ones_like(peaks, dtype=bool)
    tail_mask = (idxs_loc[:, :, 299, :] >= TAIL_MU)
    is_valid[:, :, 299, :] = ~tail_mask
    peaks_filtered = np.where(is_valid, peaks, -np.inf)

    flat_idx = np.argmax(peaks_filtered)
    g_hat, aiv_hat, tile_hat, l_hat = np.unravel_index(flat_idx, peaks.shape)
    peak_value = float(peaks[g_hat, aiv_hat, tile_hat, l_hat])
    idx_local = int(idxs_loc[g_hat, aiv_hat, tile_hat, l_hat])

    m_global = int(aiv_hat) * M_PER_AIV + int(tile_hat) * M_TILE + idx_local
    mu_t = m_global
    ssb_start = mu_t - CP_FIRST_768

    total_sum   = float(sums.sum())
    total_count = float(counts.sum())
    noise_floor = total_sum / total_count if total_count > 0 else 0.0
    G_hat = int(g_hat) - 1

    return {
        "g_hat": G_hat,
        "n_id_2": int(l_hat),
        "mu_t": mu_t,
        "ssb_start": ssb_start,
        "peak": peak_value,
        "noise_floor": noise_floor,
    }


def parse_case_truth(case_dir: Path):
    """case 名格式: case_N_pssX_cfo{p|n}YYYY_tZZZ_snr{D|-D}
       例如: case_3_pss0_cfop25000_t1000_snr30
            → pss=0, CFO=+25000 Hz, t=1000, SNR=30 dB
       CFO 是 Hz 单位 (不是 G_IDX!)
       G_IDX = round(CFO_Hz / 30000), 在 {-1, 0, +1} 内
    """
    name = case_dir.name
    parts = name.split("_")

    # n_id_2
    n_id_2_truth = int(parts[2][3:])

    # CFO Hz: "cfop10000" → +10000, "cfon10000" → -10000
    cfo_str = parts[3]
    if cfo_str.startswith("cfop"):
        cfo_hz = int(cfo_str[4:])
    elif cfo_str.startswith("cfon"):
        cfo_hz = -int(cfo_str[4:])
    else:
        cfo_hz = int(cfo_str[3:])  # fallback

    # G_IDX = round(CFO / SUBCARRIER_30kHz), clamp 到 {-1, 0, +1}
    g_truth_raw = round(cfo_hz / 30000.0)
    # NPU 搜索范围只 ±1, 超出的 case 实际 detect 边界 G
    g_truth = max(-1, min(1, g_truth_raw))

    # ssb_start
    ssb_start_truth = int(parts[4][1:])

    # SNR (parts[5] = "snr30" 或 "snr-3" or "snr0")
    snr_str = parts[5][3:]
    snr_db = int(snr_str)

    return {
        "case_name": name,
        "n_id_2_truth": n_id_2_truth,
        "cfo_hz_truth": cfo_hz,
        "g_truth": g_truth,
        "ssb_start_truth": ssb_start_truth,
        "mu_t_truth": ssb_start_truth + CP_FIRST_768,
        "snr_db": snr_db,
    }


def main():
    case_name = sys.argv[1] if len(sys.argv) > 1 else "case_0_pss0_cfop0_t12345_snr30"
    case_dir = CASE_ROOT / case_name

    if not METRIC_BIN.exists():
        print(f"[verify_result] FAIL: {METRIC_BIN} not found")
        return 1

    result = do_final_reduce(METRIC_BIN)
    truth = parse_case_truth(case_dir) if case_dir.exists() else None

    name_str = truth["case_name"] if truth else "(unknown case)"
    print(f"╔══ {name_str} " + "═" * (66 - len(name_str)))
    print(f"║ NPU: g={result['g_hat']:+d}  l={result['n_id_2']}  "
          f"mu_t={result['mu_t']}  peak={result['peak']:.3e}  noise={result['noise_floor']:.3e}  "
          f"SNR={10*np.log10(result['peak']/result['noise_floor']):.1f} dB")
    if truth:
        print(f"║ Tru: g={truth['g_truth']:+d}  l={truth['n_id_2_truth']}  "
              f"mu_t={truth['mu_t_truth']}  (CFO truth={truth['cfo_hz_truth']:+d} Hz, "
              f"SNR golden={truth['snr_db']} dB)")
        ok = (result['g_hat'] == truth['g_truth']
              and result['n_id_2'] == truth['n_id_2_truth']
              and result['mu_t'] == truth['mu_t_truth'])
        print(f"╚══ {'PASS ✓' if ok else 'FAIL ✗'} " + "═" * 60)
        return 0 if ok else 1
    else:
        print(f"╚══ (no truth, can't auto-verify) " + "═" * 36)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())