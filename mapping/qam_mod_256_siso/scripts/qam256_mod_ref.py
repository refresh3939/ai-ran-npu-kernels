#!/usr/bin/env python3
# ============================================================================
# qam256_mod_ref.py — 256-QAM Modulation Mapper golden 生成(对称 qam256_demod)
#
# qam256_demod 的逆算子:8 个 bit 流 → 复星座点。星座/比特映射与 demod 的
# BIT_TABLE 严格互逆(已 round-trip 验证:无噪 mod→demod 0 误比特)。
#
# 输入 ABI 对称 demod 输出：[Q_M,N_DATA_SYM,1600] int16 {0,1}；
#   每个数据符号前 1596 项有效、末 4 项为零 padding。
#   stream 0..3 = I 轴 (c0,c1,c2,c3);stream 4..7 = Q 轴 (c0,c1,c2,c3)
#   (c0=符号位;与 demod 输出 0=I_b3 1=I_b2 2=I_b1 3=I_b0 ... 同位)。
#
# 输出布局(对称 demod 输入):全网格 [N_SYMBOL=14, N_SC_PAD=1664] fp16 × 2
#   x_re / x_im。data 符号写 phys 行 [phys, 0:N_SC_USED]=level*D;
#   尾部 [N_SC_USED:N_SC_PAD]=0;DMRS 符号 {2,11} 整行=0。
# ============================================================================
import os
import numpy as np

N_SYMBOL    = 14
DMRS_SYMS   = {2, 11}
N_DATA_SYM  = N_SYMBOL - len(DMRS_SYMS)     # 12
N_SC_USED   = 1596
N_SC_PAD    = 1664
N_RE_DATA   = N_DATA_SYM * N_SC_USED        # 19152
N_SYM_PAD   = ((N_RE_DATA + 127) // 128) * 128   # 19200
Q_M         = 8

DATA_SYM_TO_PHYS = [s for s in range(N_SYMBOL) if s not in DMRS_SYMS]
assert len(DATA_SYM_TO_PHYS) == N_DATA_SYM

D_256QAM = 1.0 / np.sqrt(170.0)
D16      = np.float16(D_256QAM)

PAM_LEVELS = np.array([-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15], dtype=np.int32)


# ---- 单轴 4-bit → PAM 电平闭式(精确反演 demod BIT_TABLE,已验证)----------
#   level = d0*(8 - d1*(4 - d2*(3 - 2*c3))),  d_k = 2*c_k - 1
# kernel 在 half 下逐位算同一闭式(电平 |.|<=15,fp16 精确)。
def mod_level_vec(c0, c1, c2, c3):
    d0 = 2*c0 - 1; d1 = 2*c1 - 1; d2 = 2*c2 - 1; e3 = 3 - 2*c3
    return (d0 * (8 - d1 * (4 - d2 * e3))).astype(np.int32)


def kernel_modulate(bits_stream):
    """bits_stream: [Q_M, N_SC_USED] int (一个数据符号的 8 流) -> (x_re_h, x_im_h)[N_SC_USED] fp16
       与 kernel half 运算 bit-exact:level(精确整数,fp16 精确) * float16(D)。"""
    cI = bits_stream[0:4]; cQ = bits_stream[4:8]
    I = mod_level_vec(cI[0], cI[1], cI[2], cI[3])     # int 电平
    Q = mod_level_vec(cQ[0], cQ[1], cQ[2], cQ[3])
    x_re = (np.float16(I.astype(np.float16)) * D16).astype(np.float16)
    x_im = (np.float16(Q.astype(np.float16)) * D16).astype(np.float16)
    return x_re, x_im


# ---- bit 图样生成器:返回 [Q_M, N_RE_DATA] int16 {0,1} -----------------------
def bits_random(seed):
    def f():
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=(Q_M, N_RE_DATA), dtype=np.int16)
    return f

def bits_const(val):
    def f():
        return np.full((Q_M, N_RE_DATA), val, dtype=np.int16)
    return f

def bits_exhaustive(seed):
    """前 256 RE 遍历全部 256 个星座点(每 RE 一个 8-bit 组合),其余随机。"""
    def f():
        rng = np.random.default_rng(seed)
        b = rng.integers(0, 2, size=(Q_M, N_RE_DATA), dtype=np.int16)
        for n in range(256):
            for s in range(Q_M):
                b[s, n] = (n >> s) & 1
        return b
    return f

# name, bits_gen, seed
CASES = [
    ("case_0_random_a",   bits_random(0xB01),      None),
    ("case_1_all_zero",   bits_const(0),           None),  # (0,0,0,0)->-15 角点
    ("case_2_all_one",    bits_const(1),            None),  # (1,1,1,1)->+5
    ("case_3_exhaustive", bits_exhaustive(0xB04),  None),  # 覆盖全部 256 点
    ("case_4_random_b",   bits_random(0xB05),      None),
]


def gen_case(data_dir, name, bits_gen, _seed):
    gdir = os.path.join(data_dir, "golden", name)
    odir = os.path.join(data_dir, "ascend_output", name)
    os.makedirs(gdir, exist_ok=True); os.makedirs(odir, exist_ok=True)

    bits_valid = bits_gen()                       # [Q_M, N_RE_DATA] {0,1}
    assert bits_valid.shape == (Q_M, N_RE_DATA)

    # 输出全网格 [N_SYMBOL, N_SC_PAD] fp16,DMRS/padding = 0
    x_re = np.zeros((N_SYMBOL, N_SC_PAD), dtype=np.float16)
    x_im = np.zeros((N_SYMBOL, N_SC_PAD), dtype=np.float16)
    for di in range(N_DATA_SYM):
        sp  = DATA_SYM_TO_PHYS[di]
        seg = bits_valid[:, di*N_SC_USED:(di+1)*N_SC_USED]   # [Q_M, N_SC_USED]
        xr, xi = kernel_modulate(seg.astype(np.int32))
        x_re[sp, :N_SC_USED] = xr
        x_im[sp, :N_SC_USED] = xi

    # 与当前默认 qam_demod 输出对称的 symbol-padded ABI。总 allocation 仍是
    # [Q_M,19200]，但物理解释为 [Q_M,N_DATA_SYM,1600]；每符号末 4 项为 0。
    in_bits_padded = np.zeros((Q_M, N_DATA_SYM, 1600), dtype=np.int16)
    for di in range(N_DATA_SYM):
        in_bits_padded[:, di, :N_SC_USED] = \
            bits_valid[:, di*N_SC_USED:(di+1)*N_SC_USED]
    assert not np.any(in_bits_padded[:, :, N_SC_USED:])

    in_bits_padded.tofile(os.path.join(gdir, "input_bits_padded.bin"))
    x_re.tofile   (os.path.join(gdir, "x_re.bin"))
    x_im.tofile   (os.path.join(gdir, "x_im.bin"))

    # 唯一星座点数(诊断)
    pts = set()
    for di in range(min(N_DATA_SYM, 4)):
        sp = DATA_SYM_TO_PHYS[di]
        for sc in range(0, N_SC_USED, 7):
            pts.add((float(x_re[sp, sc]), float(x_im[sp, sc])))
    print(f"  [{name:18s}]  REs={N_RE_DATA}  sampled-constel-pts={len(pts)}  "
          f"x range=[{x_re.min():+.4f},{x_re.max():+.4f}]")


def main():
    kdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.environ.get("AIRAN_DATA_DIR", os.path.join(kdir, "data"))
    print("=== qam256_mod golden (对称 qam256_demod, multi-case) ===")
    print(f"  data_dir = {data_dir}")
    print(f"  Input  : [{Q_M},{N_DATA_SYM},1600] symbol-padded int16 bits")
    print(f"  Output : [{N_SYMBOL},{N_SC_PAD}] fp16 x_re/x_im (DMRS{sorted(DMRS_SYMS)}/pad=0)\n")
    for name, gen, seed in CASES:
        gen_case(data_dir, name, gen, seed)
    print(f"\n  {len(CASES)} cases written. Kernel 必须 bit-exact 匹配 x_re/x_im.bin。")


if __name__ == "__main__":
    main()
