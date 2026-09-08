"""
decimate ×8 reference for 5G NR RX SDR front-end (61.44 MHz → 7.68 MHz).

跑这个脚本会生成 FIR 系数 (weights) + golden bins,并自检算法正确性。

跑法:
    cd ~/AI-RAN-NPU/kernels/rx/decimate
    python3 scripts/decimate_ref.py

输出位置 (跟 kernel main.cpp 的 AIRAN_DATA_DIR 期望一致):
    <kernel_dir>/weights/        FIR 系数 (fp16, 两种 layout)
    <kernel_dir>/data/golden/    input / output / per-stage

Algorithm: 抗混叠 lowpass FIR + ↓8 抽取 (polyphase decimator)。
  y[m] = Σ_{k=0}^{L-1} h[k] · x[8m − k]          (causal, x[neg]=0)
       = Σ_{p=0}^{7} Σ_{n=0}^{NPT-1} h_p[n] · xd_p[m−n]   (8-phase 等价式)
  其中 h_p[n] = h[8n+p],  xd_p[j] = x[8j−p]  (x 的第 p 相, ↓8 offset −p)。

  - direct 式 (lfilter+↓8) 作 fp64 golden。
  - polyphase 8-phase 式断言 bit-exact 等价 (这是 kernel 要实现的分解)。
  - fp16-sim 路估计 NPU 上 fp16 系数 + Cube fp32 累加的量化误差。

对齐契约 (硬约束,见三算子文档 §1):
  输出 = pss_correlator 输入 = sss_correlator 输入 (int16 IQ 交织 @7.68M)。
  group delay = (L-1)/2 个 @61.44M 样点 = 常数时延 → pss_correlator 峰值搜索自动吸收。

GATE 1 (未钉): USRP int16 标度 / Q 格式。INPUT_Q 是占位值,实测 USRP 流后改。
  这里取 INPUT_Q 使样点幅度 ≤ 2048,保证 Cast(int16→half) 无损 (fp16 尾数 11 bit)。
  真实 USRP 大幅度样点要么 host 预缩放,要么接受 Cast 精度损失 —— 实测后定。
"""

import os
import numpy as np
from pathlib import Path
from scipy.signal import firwin, lfilter, freqz


# ============================================================
# 默认路径 — scripts/ 上一层是 kernel 目录 (同 ofdm_demod_ref.py)
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent          # kernels/rx/decimate/scripts
_KERNEL_DIR = _SCRIPT_DIR.parent                        # kernels/rx/decimate

DEFAULT_WEIGHTS_DIR = _KERNEL_DIR / "weights"
DEFAULT_GOLDEN_DIR  = _KERNEL_DIR / "data" / "golden"


# ============================================================
# 锁定参数 (LOCKED — 改这里必须同步改 kernel decimate.h)
# ============================================================
FS_DATA   = 61_440_000          # 输入全速率
FS_SSB    =  7_680_000          # 输出 SSB 率
DECIM     = 8                   # 抽取因子 = FS_DATA / FS_SSB

N_OUT     = 153_600             # 输出复样点数 (= pss N_SEARCH)
N_IN      = N_OUT * DECIM       # 1,228,800 输入复样点

# FIR 抗混叠设计
N_PHASE     = DECIM             # = 8 相
N_PHASE_TAP = 12                # 每相 tap 数 (典型 8~12)
L_TAP       = N_PHASE * N_PHASE_TAP   # = 96 总 tap 数
FC_HZ       = FS_DATA / 16      # ≈ 3.84 MHz = 输出 Nyquist (= FS_SSB/2) = 混叠折叠点
KAISER_BETA = 8.0               # Kaiser 窗 β (~ stopband)

GROUP_DELAY = (L_TAP - 1) / 2.0 # @61.44M 样点的常数群时延

# GATE 1 占位:USRP int16 标度。取 512 使 |样点| ≤ ~2048 → Cast(int16→half) 无损。
INPUT_Q   = 512


# ============================================================
# FIR 设计 (kernel 从 .bin 加载这些系数)
# ============================================================
def design_fir():
    """firwin lowpass, DC gain = 1 (Σh=1) → 输出与输入同量程, int16 不溢出。"""
    h = firwin(L_TAP, cutoff=FC_HZ, fs=FS_DATA,
               window=("kaiser", KAISER_BETA), pass_zero=True)
    return h.astype(np.float64)


FIR_TAPS = design_fir()
assert FIR_TAPS.shape == (L_TAP,)


# ============================================================
# 抽取 reference — direct 式 (fp64 golden 来源)
# ============================================================
def decimate_direct_fp64(x_complex, h):
    """y[m] = Σ_k h[k] x[8m−k], x[neg]=0。
       用 lfilter(零初态 = 前补零) 再 ↓8,等价上式。"""
    yf = lfilter(h, [1.0], x_complex.astype(np.complex128))   # 全率滤波, 长度 N_IN
    yd = yf[::DECIM]                                           # ↓8 取每第 8 点
    assert yd.shape[0] == N_OUT
    return yd


# ============================================================
# 抽取 reference — polyphase 8-phase 式 (kernel 要实现的分解)
# ============================================================
def decimate_polyphase_fp64(x_complex, h):
    """y[m] = Σ_p (h_p ⊛ xd_p)[m],  h_p[n]=h[8n+p],  xd_p[j]=x[8j−p]。
       向量化, 无 Python 内层循环。"""
    x = x_complex.astype(np.complex128)
    # 前补 (L_TAP) 个零,处理负索引历史
    xpad = np.concatenate([np.zeros(L_TAP, dtype=np.complex128), x])
    base = np.arange(N_OUT) * DECIM + L_TAP    # xpad 中对应 x[8m] 的位置

    y = np.zeros(N_OUT, dtype=np.complex128)
    for p in range(N_PHASE):
        hp = h[p::N_PHASE]                     # h_p[n] = h[8n+p], 长 NPT
        for n in range(N_PHASE_TAP):
            # x[8m − (8n+p)] = xpad[base − 8n − p]
            y += hp[n] * xpad[base - DECIM * n - p]
    return y


# ============================================================
# 抽取 reference — fp16-sim 路 (估计 NPU 量化误差)
#   mirror kernel: Cast int16→half(无损) → taps fp16 → Cube fp32 累加 → round int16
# ============================================================
def decimate_polyphase_fp16sim(x_int16_complex, h):
    """输入是 int16 量程的复值 (整数浮点)。taps fp16, 累加 fp32 (Cube 行为)。"""
    h16 = h.astype(np.float16).astype(np.float64)              # 系数走 fp16 量化
    xr = x_int16_complex.real.astype(np.float16).astype(np.float64)
    xi = x_int16_complex.imag.astype(np.float16).astype(np.float64)
    xc = (xr + 1j * xi)                                        # Cast int16→half (≤2048 无损)
    xpad = np.concatenate([np.zeros(L_TAP), xc])
    base = np.arange(N_OUT) * DECIM + L_TAP
    y = np.zeros(N_OUT, dtype=np.complex128)                   # Cube L0C = fp32 累加
    for p in range(N_PHASE):
        hp = h16[p::N_PHASE]
        for n in range(N_PHASE_TAP):
            y += hp[n] * xpad[base - DECIM * n - p]
    return y


# ============================================================
# 测试信号 (确定性, 含带内 + 带外音以验抗混叠)
# ============================================================
def make_test_signal():
    t = np.arange(N_IN) / FS_DATA
    sig = np.zeros(N_IN, dtype=np.complex128)
    # 带内音 (应保留): ±0.5, 1.5, −3.0 MHz
    for f, a in [(0.5e6, 1.0), (1.5e6, 0.7), (-3.0e6, 0.5)]:
        sig += a * np.exp(2j * np.pi * f * t)
    # 带外音 (应被抑制,否则混叠进 7.68M 带): 8 MHz, −20 MHz
    OOB = [(8.0e6, 0.6), (-20.0e6, 0.6)]
    for f, a in OOB:
        sig += a * np.exp(2j * np.pi * f * t)
    # 小确定性噪声
    rng = np.random.default_rng(seed=20260607)
    sig += (rng.standard_normal(N_IN) + 1j * rng.standard_normal(N_IN)) * 0.02
    # 归一到峰值 ~0.9 再 ×INPUT_Q → int16
    sig /= (np.max(np.abs(sig)) / 0.9)
    return sig, OOB


def to_int16_complex(sig):
    re = np.round(sig.real * INPUT_Q).astype(np.int16)
    im = np.round(sig.imag * INPUT_Q).astype(np.int16)
    return re.astype(np.float64) + 1j * im.astype(np.float64), re, im


def complex_to_interleaved_int16(re_i16, im_i16):
    return np.stack([re_i16, im_i16], axis=-1).reshape(-1).astype(np.int16)


# ============================================================
# 自检
# ============================================================
def verify_polyphase_vs_direct(x_i16c, atol=1e-6):
    yd = decimate_direct_fp64(x_i16c, FIR_TAPS)
    yp = decimate_polyphase_fp64(x_i16c, FIR_TAPS)
    err = np.abs(yd - yp).max()
    print(f"[polyphase vs direct(lfilter+↓8)] max|err| = {err:.3e}  (tol={atol})")
    assert err < atol, f"polyphase 分解和 direct 不等价: {err}"
    return yd


def verify_antialiasing(yd, x_i16c, oob, min_rej_db=40.0):
    """带外音在输出谱里应被抑制。报告每个带外音折叠位置的抑制比。"""
    Y = np.fft.fftshift(np.fft.fft(yd) / N_OUT)
    fout = np.fft.fftshift(np.fft.fftfreq(N_OUT, d=1.0 / FS_SSB))
    # 输入谱中带内最强音作参考电平
    inband_ref = np.abs(np.fft.fft(x_i16c)).max() / N_IN
    print("[antialiasing] 带外音抑制:")
    worst = 1e9
    for f, a in oob:
        falias = ((f + FS_SSB / 2) % FS_SSB) - FS_SSB / 2   # 折叠到 [-Fs/2,Fs/2)
        idx = np.argmin(np.abs(fout - falias))
        lvl = np.abs(Y[idx])
        rej = 20 * np.log10(inband_ref / (lvl + 1e-12))
        worst = min(worst, rej)
        print(f"  f_in={f/1e6:+5.1f}MHz → alias {falias/1e6:+5.2f}MHz  rej≈{rej:5.1f} dB")
    print(f"[antialiasing] 最差抑制 ≈ {worst:.1f} dB  (要求 ≥ {min_rej_db})")
    return worst


def verify_fp16sim(yd_fp64, x_i16c):
    """fp16-sim (kernel 模型) vs fp64 golden, int16 量程下的 LSB 误差。"""
    y16 = decimate_polyphase_fp16sim(x_i16c, FIR_TAPS)
    e_re = np.abs(np.round(y16.real) - np.round(yd_fp64.real)).max()
    e_im = np.abs(np.round(y16.imag) - np.round(yd_fp64.imag)).max()
    print(f"[fp16-sim vs fp64] round int16 误差: re={e_re:.0f} im={e_im:.0f} LSB")
    return max(e_re, e_im)


# ============================================================
# Golden dump
# ============================================================
def dump_goldens(out_root=None, weights_root=None):
    if out_root is None:
        out_root = str(DEFAULT_GOLDEN_DIR)
    if weights_root is None:
        weights_root = str(DEFAULT_WEIGHTS_DIR)
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(weights_root, exist_ok=True)

    sig, oob = make_test_signal()
    x_i16c, re_i16, im_i16 = to_int16_complex(sig)

    # 自检 (上 NPU 前必须全绿)
    yd   = verify_polyphase_vs_direct(x_i16c)
    rej  = verify_antialiasing(yd, x_i16c, oob)
    lsb  = verify_fp16sim(yd, x_i16c)

    # golden 输出: fp64 滤波 → round int16
    out_re_i16 = np.round(yd.real).astype(np.int16)
    out_im_i16 = np.round(yd.imag).astype(np.int16)

    # --- weights (fp16) ---
    # 1) flat taps, 直序 (route A 向量 MAC / 通用)
    FIR_TAPS.astype(np.float16).tofile(f"{weights_root}/fir_taps.bin")
    # 2) polyphase [8, NPT] : poly[p, n] = h[8n+p]  (route B Cube / 分相 FIR)
    poly = np.stack([FIR_TAPS[p::N_PHASE] for p in range(N_PHASE)], axis=0)  # (8, NPT)
    assert poly.shape == (N_PHASE, N_PHASE_TAP)
    poly.astype(np.float16).tofile(f"{weights_root}/fir_taps_poly.bin")

    # --- goldens ---
    complex_to_interleaved_int16(re_i16, im_i16).tofile(f"{out_root}/input.bin")
    complex_to_interleaved_int16(out_re_i16, out_im_i16).tofile(f"{out_root}/output.bin")
    # fp64 输出 (未 round) 供 phase-dump 调试对比
    np.stack([yd.real, yd.imag], -1).reshape(-1).astype(np.float32).tofile(
        f"{out_root}/output_fp64.bin")

    print(f"\n[dump] weights → {weights_root}/")
    for n in ["fir_taps", "fir_taps_poly"]:
        f = f"{weights_root}/{n}.bin"
        print(f"  {n+'.bin':<20} {os.path.getsize(f)/1024:>8.2f} KiB")
    print(f"[dump] goldens → {out_root}/")
    for n in ["input", "output", "output_fp64"]:
        f = f"{out_root}/{n}.bin"
        print(f"  {n+'.bin':<20} {os.path.getsize(f)/1024:>8.2f} KiB")

    return rej, lsb


# ============================================================
# Driver
# ============================================================
if __name__ == "__main__":
    print("=" * 64)
    print(" decimate ×8 (61.44M→7.68M) — 数据生成 + 自检")
    print("=" * 64)
    print(f"  KERNEL_DIR  = {_KERNEL_DIR}")
    print(f"  N_IN={N_IN}  N_OUT={N_OUT}  DECIM={DECIM}")
    print(f"  FIR: L={L_TAP} (8相×{N_PHASE_TAP})  fc={FC_HZ/1e6:.2f}MHz  β={KAISER_BETA}")
    print(f"  group delay = {GROUP_DELAY} @61.44M 样点  (常数时延)")
    print(f"  INPUT_Q={INPUT_Q} (GATE 1 占位)")
    # FIR 频响快速核查
    w, Hf = freqz(FIR_TAPS, worN=8192, fs=FS_DATA)
    stop_mask = w >= FC_HZ * 1.2
    if stop_mask.any():
        sb = 20 * np.log10(np.abs(Hf[stop_mask]).max() + 1e-12)
        print(f"  FIR stopband (>{FC_HZ*1.2/1e6:.2f}MHz) peak ≈ {sb:.1f} dB")
    print()

    rej, lsb = dump_goldens()

    print("\n" + "=" * 64)
    ok = (rej >= 40.0) and (lsb <= 2.0)
    print(f"  {'✅' if ok else '❌'}  抗混叠 {rej:.1f}dB  fp16误差 {lsb:.0f}LSB  →  "
          f"{'跑 bash run.sh 测 kernel' if ok else '调 FIR / INPUT_Q'}")
    print("=" * 64)
