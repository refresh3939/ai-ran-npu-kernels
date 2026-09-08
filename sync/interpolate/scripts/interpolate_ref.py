"""
interpolate ×8 reference for 5G NR TX SDR back-end (7.68 MHz → 61.44 MHz).

decimate 的镜像：decimate 是 RX 下变频 (↓8 polyphase 抗混叠)，
interpolate 是 TX 上变频 (↑8 polyphase 抗镜像)，I/O 契约方向反转。

跑这个脚本会生成 FIR 系数 (weights) + golden bins，并自检算法正确性。

跑法:
    cd ~/AI-RAN-NPU/kernels/tx/interpolate
    python3 scripts/interpolate_ref.py

输出位置 (跟 kernel main.cpp 的 AIRAN_DATA_DIR 期望一致):
    <kernel_dir>/weights/        FIR 系数 (fp16, 两种 layout)
    <kernel_dir>/data/golden/    input / output / per-stage

Algorithm: 上采样 (↑8 零填) + 抗镜像 lowpass FIR (polyphase interpolator)。
  v[n]  = x[n/8]  (n%8==0)  else 0                 (零填上采样)
  y[n]  = Σ_{k=0}^{L-1} h[k] · v[n−k]               (全速率抗镜像 FIR, causal)
        = Σ_{n=0}^{NPT-1} h_p[n] · x[m−n]   (8-phase 等价式, n_out=8m+p)
  其中 h_p[n] = h[8n+p]  (h 的第 p 相子滤波器),  每相只卷原始 x (无零填浪费)。

  - direct 式 (零填 + lfilter) 作 fp64 golden。
  - polyphase 8-phase 式断言 bit-exact 等价 (这是 kernel 要实现的分解)。
  - fp16-sim 路估计 NPU 上 fp16 系数 + 累加的量化误差。

抗镜像 vs 抗混叠:
  decimate 滤 FS_DATA/16 以下、抑 ↓8 折叠的混叠 (DC gain=1, 输出同量程)。
  interpolate 滤同一截止 FS_SSB/2=3.84M、抑零填产生的 7.68M/15.36M/... 镜像。
  零填把能量摊到 8 倍带宽 → 通带需 ×8 增益补回 (Σh=8) → 输出与输入同量程。

对齐契约 (硬约束):
  输入  = TX 基带 IQ 交织 @7.68M (int16)，由上游 ifft/cp_add 产出 (此处占位测试信号)。
  输出  = USRP DAC 喂入 IQ 交织 @61.44M (int16)。
  group delay = (L−1)/2 个 @61.44M 样点 = 常数时延 (loopback RX 侧 pss 峰值搜索吸收)。

GATE 1 (未钉): USRP int16 标度 / Q 格式。INPUT_Q 是占位值，实测 USRP 流后改。
  取 INPUT_Q 使输入/输出样点幅度 ≤ ~2048，保证 Cast(int16↔half) 无损 (fp16 尾数 11 bit)。
"""

import os
import numpy as np
from pathlib import Path
from scipy.signal import firwin, lfilter, freqz


# ============================================================
# 默认路径 — scripts/ 上一层是 kernel 目录 (同 decimate_ref.py)
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent          # kernels/tx/interpolate/scripts
_KERNEL_DIR = _SCRIPT_DIR.parent                        # kernels/tx/interpolate

DEFAULT_WEIGHTS_DIR = _KERNEL_DIR / "weights"
DEFAULT_GOLDEN_DIR  = _KERNEL_DIR / "data" / "golden"


# ============================================================
# 锁定参数 (LOCKED — 改这里必须同步改 kernel interpolate.h)
# ============================================================
FS_SSB    =  7_680_000          # 输入基带率 (= decimate 输出率)
FS_DATA   = 61_440_000          # 输出全速率 (= USRP DAC 率)
INTERP    = 8                   # 上采样因子 = FS_DATA / FS_SSB

N_IN      = 153_600             # 输入复样点数 (= decimate N_OUT, 镜像)
N_OUT     = N_IN * INTERP       # 1,228,800 输出复样点 (= decimate N_IN)

# FIR 抗镜像设计 (和 decimate 同一 fc/β，仅 DC 增益 ×8)
N_PHASE     = INTERP            # = 8 相
N_PHASE_TAP = 12                # 每相 tap 数 NPT (典型 8~12)
L_TAP       = N_PHASE * N_PHASE_TAP   # = 96 总 tap 数
FC_HZ       = FS_DATA / 16      # ≈ 3.84 MHz = FS_SSB/2 = 输入 Nyquist = 镜像折叠点
KAISER_BETA = 8.0               # Kaiser 窗 β (~ stopband)

GROUP_DELAY = (L_TAP - 1) / 2.0 # @61.44M 样点的常数群时延

# GATE 1 占位:USRP int16 标度. 取 512 使 |样点| ≤ ~2048 → Cast(int16↔half) 无损.
INPUT_Q   = 512

# ── kernel 并相 (Gather) 参数 (与 interpolate.h 锁定一致) ─────────────────────
#   每 tile IN_TILE 输入 → OUT_TILE 输出; cmat[8, IN_TILE] int32 phase-major.
#   gather idx 是 per-tile byte-offset (cmat 内, 跨 tile 复用): out32[k]=cmat[idx[k]/4]
KIN_TILE  = 1280
KOUT_TILE = KIN_TILE * INTERP   # 10240


def make_gather_index():
    """out32[k] = cmat[(k%8)*IN_TILE + k//8]  →  byte offset = ((k%8)*IN_TILE+k//8)*4
       (cmat int32 [8, IN_TILE] phase-major; reinterpret int16 即 [I,Q,...] 全速率交织)"""
    k = np.arange(KOUT_TILE, dtype=np.int64)
    p = k % INTERP
    m = k // INTERP
    return ((p * KIN_TILE + m) * 4).astype(np.uint32)   # byte offset, int32 元素 ×4


# ============================================================
# FIR 设计 (kernel 从 .bin 加载这些系数)
#   firwin DC gain = 1 (Σh=1)，再 ×INTERP → Σh=8 → 补零填能量损失，输出同量程。
# ============================================================
def design_fir():
    h = firwin(L_TAP, cutoff=FC_HZ, fs=FS_DATA,
               window=("kaiser", KAISER_BETA), pass_zero=True)
    return (h * INTERP).astype(np.float64)        # 抗镜像增益 ×8


FIR_TAPS = design_fir()
assert FIR_TAPS.shape == (L_TAP,)


# ============================================================
# 插值 reference — direct 式 (fp64 golden 来源)
#   零填 ↑8 → 全速率 FIR(lfilter, 零初态 = 前补零) = causal y[n]=Σ h[k] v[n−k]
# ============================================================
def interpolate_direct_fp64(x_complex, h):
    v = np.zeros(N_OUT, dtype=np.complex128)
    v[::INTERP] = x_complex.astype(np.complex128)              # 零填上采样
    y = lfilter(h, [1.0], v)                                   # 全率抗镜像 FIR
    assert y.shape[0] == N_OUT
    return y


# ============================================================
# 插值 reference — polyphase 8-phase 式 (kernel 要实现的分解)
#   y[8m+p] = Σ_n h_p[n]·x[m−n],  h_p[n]=h[8n+p]。每相 NPT 抽头卷原始 x。
#   向量化，无 Python 内层循环。
# ============================================================
def interpolate_polyphase_fp64(x_complex, h):
    x = x_complex.astype(np.complex128)
    # 前补 (NPT) 个零，处理负索引历史 x[m−n], n<NPT
    xpad = np.concatenate([np.zeros(N_PHASE_TAP, dtype=np.complex128), x])
    base = np.arange(N_IN) + N_PHASE_TAP                       # xpad 中 x[m] 的位置

    y = np.zeros(N_OUT, dtype=np.complex128)
    for p in range(N_PHASE):
        hp = h[p::N_PHASE]                                     # h_p[n] = h[8n+p], 长 NPT
        yp = np.zeros(N_IN, dtype=np.complex128)
        for n in range(N_PHASE_TAP):
            yp += hp[n] * xpad[base - n]                       # x[m−n]
        y[p::N_PHASE] = yp                                     # 散布到 8m+p
    return y


# ============================================================
# 插值 reference — fp16-sim 路 (估计 NPU 量化误差)
#   mirror kernel: Cast int16→half(无损) → taps fp16 → 累加 → round int16
# ============================================================
def interpolate_polyphase_fp16sim(x_int16_complex, h):
    h16 = h.astype(np.float16).astype(np.float64)             # 系数走 fp16 量化
    xr = x_int16_complex.real.astype(np.float16).astype(np.float64)
    xi = x_int16_complex.imag.astype(np.float16).astype(np.float64)
    xc = (xr + 1j * xi)                                       # Cast int16→half (≤2048 无损)
    xpad = np.concatenate([np.zeros(N_PHASE_TAP), xc])
    base = np.arange(N_IN) + N_PHASE_TAP
    y = np.zeros(N_OUT, dtype=np.complex128)
    for p in range(N_PHASE):
        hp = h16[p::N_PHASE]
        yp = np.zeros(N_IN, dtype=np.complex128)
        for n in range(N_PHASE_TAP):
            yp += hp[n] * xpad[base - n]
        y[p::N_PHASE] = yp
    return y


# ============================================================
# 测试信号 (确定性, 全带内 — 验抗镜像看输出谱里镜像被压)
# ============================================================
def make_test_signal():
    t = np.arange(N_IN) / FS_SSB
    sig = np.zeros(N_IN, dtype=np.complex128)
    # 带内音 — 留 guard ≤2.5MHz, 一阶镜像 (7.68−|f|≥5.18M) 落 96-tap β=8 深阻带。
    # (满占 SSB ±3.6M 的镜像落过渡带 → 要更多 tap, 是 FIR-design GATE, Refresh owns)
    tones = [(0.5e6, 1.0), (1.5e6, 0.7), (-2.5e6, 0.5)]
    for f, a in tones:
        sig += a * np.exp(2j * np.pi * f * t)
    # 小确定性噪声
    rng = np.random.default_rng(seed=20260611)
    sig += (rng.standard_normal(N_IN) + 1j * rng.standard_normal(N_IN)) * 0.02
    # 归一到峰值 ~0.9 再 ×INPUT_Q → int16
    sig /= (np.max(np.abs(sig)) / 0.9)
    return sig, tones


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
    yd = interpolate_direct_fp64(x_i16c, FIR_TAPS)
    yp = interpolate_polyphase_fp64(x_i16c, FIR_TAPS)
    err = np.abs(yd - yp).max()
    print(f"[polyphase vs direct(↑8+lfilter)] max|err| = {err:.3e}  (tol={atol})")
    assert err < atol, f"polyphase 分解和 direct 不等价: {err}"
    return yd


def verify_antiimaging(yd, sig_norm, tones, min_rej_db=40.0):
    """镜像 = 输入音在 f ± k·FS_SSB 的副本。输出谱里这些位置应被抑制。"""
    Y = np.fft.fftshift(np.fft.fft(yd) / N_OUT)
    fout = np.fft.fftshift(np.fft.fftfreq(N_OUT, d=1.0 / FS_DATA))
    print("[antiimaging] 镜像抑制 (相对该音的通带电平):")
    worst = 1e9; worst_f = 0.0
    for f, a in tones:
        i0 = np.argmin(np.abs(fout - f))
        base_lvl = np.abs(Y[i0])
        local = 1e9; local_img = 0.0
        for k in range(1, 4):                     # 一阶~三阶镜像
            for fk in (f + k * FS_SSB, f - k * FS_SSB):
                if abs(fk) >= FS_DATA / 2:
                    continue
                ik = np.argmin(np.abs(fout - fk))
                rej = 20 * np.log10(base_lvl / (np.abs(Y[ik]) + 1e-12))
                if rej < local:
                    local, local_img = rej, fk
        print(f"  f={f/1e6:+5.1f}MHz  通带={base_lvl:7.2f}  最差镜像@{local_img/1e6:+6.2f}MHz  抑制≈{local:5.1f} dB")
        if local < worst:
            worst, worst_f = local, f
    print(f"[antiimaging] 最差镜像抑制 ≈ {worst:.1f} dB (音 {worst_f/1e6:+.1f}MHz)  (要求 ≥ {min_rej_db})")
    return worst


def verify_fp16sim(yd_fp64, x_i16c):
    y16 = interpolate_polyphase_fp16sim(x_i16c, FIR_TAPS)
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

    sig, tones = make_test_signal()
    x_i16c, re_i16, im_i16 = to_int16_complex(sig)

    # 自检 (上 NPU 前必须全绿)
    yd   = verify_polyphase_vs_direct(x_i16c)
    rej  = verify_antiimaging(yd, sig, tones)
    lsb  = verify_fp16sim(yd, x_i16c)

    # golden 输出: fp64 滤波 → round int16
    out_re_i16 = np.round(yd.real).astype(np.int16)
    out_im_i16 = np.round(yd.imag).astype(np.int16)

    # --- weights (fp16) ---
    # 1) flat taps, 直序 (通用 / route A 全速率 MAC)
    FIR_TAPS.astype(np.float16).tofile(f"{weights_root}/fir_taps.bin")
    # 2) polyphase [8, NPT] : poly[p, n] = h[8n+p]  (route B 分相 FIR / Cube)
    poly = np.stack([FIR_TAPS[p::N_PHASE] for p in range(N_PHASE)], axis=0)  # (8, NPT)
    assert poly.shape == (N_PHASE, N_PHASE_TAP)
    poly.astype(np.float16).tofile(f"{weights_root}/fir_taps_poly.bin")
    # 注: 并相 Gather 索引不再 dump — 由 interpolate_tiling.cpp 的 GenerateTiling host 端算好
    #     (搭 tiling_gm 喂 kernel, 保接口与 decimate 对称). make_gather_index() 仅留作公式参考。

    # --- goldens ---
    complex_to_interleaved_int16(re_i16, im_i16).tofile(f"{out_root}/input.bin")
    complex_to_interleaved_int16(out_re_i16, out_im_i16).tofile(f"{out_root}/output.bin")
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
    print(" interpolate ×8 (7.68M→61.44M) — 数据生成 + 自检")
    print("=" * 64)
    print(f"  KERNEL_DIR  = {_KERNEL_DIR}")
    print(f"  N_IN={N_IN}  N_OUT={N_OUT}  INTERP={INTERP}")
    print(f"  FIR: L={L_TAP} (8相×{N_PHASE_TAP})  fc={FC_HZ/1e6:.2f}MHz  β={KAISER_BETA}  Σh={FIR_TAPS.sum():.3f}")
    print(f"  group delay = {GROUP_DELAY} @61.44M 样点  (常数时延)")
    print(f"  INPUT_Q={INPUT_Q} (GATE 1 占位)")
    # FIR 频响快速核查
    w, Hf = freqz(FIR_TAPS, worN=8192, fs=FS_DATA)
    stop_mask = w >= FC_HZ * 1.2
    if stop_mask.any():
        sb = 20 * np.log10(np.abs(Hf[stop_mask]).max() / INTERP + 1e-12)  # 归一通带增益
        print(f"  FIR stopband (>{FC_HZ*1.2/1e6:.2f}MHz) peak ≈ {sb:.1f} dB (相对通带)")
    print()

    rej, lsb = dump_goldens()

    print("\n" + "=" * 64)
    ok = (rej >= 40.0) and (lsb <= 2.0)
    print(f"  {'✅' if ok else '❌'}  抗镜像 {rej:.1f}dB  fp16误差 {lsb:.0f}LSB  →  "
          f"{'跑 bash run.sh 测 kernel' if ok else '调 FIR / INPUT_Q'}")
    print("=" * 64)