#!/usr/bin/env python3
# ============================================================================
# gen_pss_templates.py — 预生成 3 个 PSS 时域模板,kernel 启动时一次读入
#
# 输出: data/golden/rx/pss_acquisition/pss_templates.bin
#   layout: [3, 2048, 2] int16 Q8.8 (3 PSS × 2048 samples × {I, Q} interleaved)
#   实际复数虚部恒为 0 (PSS 在频域是实数 BPSK),但为了与 input.bin 同款 IQ
#   交错格式,虚部写 0 保留 GatherMask 拆分路径
#
# 注:模板算法严格与 pss_acquisition_ref.py 中的 pss_time_domain_template() 一致
# ============================================================================
import os
import sys
from pathlib import Path
import numpy as np

# 与 pss_acquisition.h 锁定一致
N_FFT       = 2048
N_SC_SSB    = 240
N_SC_PSS    = 127
Q_SCALE     = 256
INT16_MAX   = 32767


def gen_pss_sequence(N_ID_2: int) -> np.ndarray:
    """5G NR PSS BPSK sequence, length 127.  3GPP TS 38.211 §7.4.2.2."""
    x = np.zeros(127 + 7, dtype=np.int8)
    x[0:7] = [0, 1, 1, 0, 1, 1, 1]
    for i in range(120):
        x[i + 7] = (x[i + 4] + x[i]) % 2
    d_pss = np.zeros(127, dtype=np.float64)
    for n in range(127):
        d_pss[n] = 1.0 - 2.0 * x[(n + 43 * N_ID_2) % 127]
    return d_pss


def pss_time_domain_template(N_ID_2: int) -> np.ndarray:
    """N_FFT-long complex time-domain PSS template."""
    pss_freq = gen_pss_sequence(N_ID_2)
    sc_grid = np.zeros(N_SC_SSB, dtype=np.complex128)
    sc_grid[56:56 + N_SC_PSS] = pss_freq

    half_ssb = N_SC_SSB // 2
    center = N_FFT // 2
    x_shift = np.zeros(N_FFT, dtype=np.complex128)
    x_shift[center - half_ssb : center + half_ssb] = sc_grid

    x_freq = np.fft.ifftshift(x_shift)
    return np.fft.ifft(x_freq) * N_FFT


def quantize_to_int16(x_complex: np.ndarray) -> np.ndarray:
    iq = np.empty(x_complex.shape[0] * 2, dtype=np.float64)
    iq[0::2] = x_complex.real
    iq[1::2] = x_complex.imag
    return np.clip(np.round(iq * Q_SCALE), -INT16_MAX, INT16_MAX).astype(np.int16)


def main():
    # 输出位置: 与 input.bin 同目录上一级
    env = os.environ.get("AIRAN_DATA_DIR")
    if env:
        out_root = Path(env) / "golden" / "rx" / "pss_acquisition"
    else:
        here = Path(__file__).resolve()
        out_root = here.parents[4] / "data" / "golden" / "rx" / "pss_acquisition"
    out_root.mkdir(parents=True, exist_ok=True)

    out_path = out_root / "pss_templates.bin"

    # 3 个 PSS 模板叠成 [3, N_FFT, 2] int16
    templates = np.zeros((3, N_FFT * 2), dtype=np.int16)
    for pss_id in range(3):
        x_cplx = pss_time_domain_template(pss_id)
        templates[pss_id] = quantize_to_int16(x_cplx)

        # 打印峰值给人眼检查
        peak_abs = np.max(np.abs(templates[pss_id]))
        rms = np.sqrt(np.mean(templates[pss_id].astype(np.float64) ** 2))
        print(f"  pss_id={pss_id}: peak_abs={peak_abs}, rms={rms:.1f}")

    # 序列化: pss_id 0 全部 -> pss_id 1 全部 -> pss_id 2 全部
    # 每个 PSS 模板长度 = 2 * N_FFT * 2 B = 8192 B
    # 总 = 3 * 8192 = 24576 B = 24 KB
    templates.tofile(out_path)

    print(f"\n  output: {out_path}")
    print(f"  size:   {out_path.stat().st_size} B = {out_path.stat().st_size // 1024} KB")
    print(f"  layout: [3, {N_FFT*2}] int16 = [3 PSS × {N_FFT} samples × (I,Q) interleaved]")


if __name__ == "__main__":
    main()
