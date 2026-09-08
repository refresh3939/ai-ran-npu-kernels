#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# kernels/tx/ldpc_encode/scripts/ldpc_ref.py
#
# T4 ldpc_encode 参考 + 权重生成（对齐 decode 的 scripts/gen_data.py 模式）
#
# 产出（全部本地，self-contained）：
#   data/golden/input.bin                       int8 (C_NUM, K=8448)
#   data/golden/output.bin                      int8 (C_NUM, N_RAW=25344)   puncture 2Z 后
#   data/weights/ldpc_bg1_z384_shifts/shift_A.bin   int16 (G*KB      = 88)
#   data/weights/ldpc_bg1_z384_shifts/shift_Bi.bin  int16 (G*G*MAXW  = 64)
#   data/weights/ldpc_bg1_z384_shifts/shift_C.bin   int16 (MBMG*KB   = 924)
#   data/weights/ldpc_bg1_z384_shifts/shift_D.bin   int16 (MBMG*G    = 168)
#   （-1 = 无边；shift 表是 kernel 直接消费的权重，dense A/B_inv/C/D 不落盘）
#
# 三重自校验（任一不过即 raise）：
#   (a) dense matmul 分解  H·c = 0
#   (b) shift-form numpy 复算（与 kernel 逐 op 同构） == dense 分解输出
#   (c) 与 Sionna 官方 LDPC5GEncoder bit-exact
#
# 用法：
#   conda activate sionna
#   python scripts/ldpc_ref.py
# ============================================================================

import os
import sys
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# 路径（本地 self-contained；可被 AIRAN_DATA_DIR 覆盖，与 main.cpp 对齐）
# ---------------------------------------------------------------------------
HERE   = Path(__file__).resolve().parent          # .../ldpc_encode/scripts
KERNEL = HERE.parent                               # .../ldpc_encode
DATA   = Path(os.environ["AIRAN_DATA_DIR"]) if os.environ.get("AIRAN_DATA_DIR") \
         else (KERNEL / "data")
GOLDEN_DIR = DATA / "golden"
WEIGHT_DIR = DATA / "weights" / "ldpc_bg1_z384_shifts"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 参数（LDPC BG1 Z=384 锁定；必须与 ldpc_encode.h 一致）
# ---------------------------------------------------------------------------
Z        = 384
KB       = 22
MB       = 46
G        = 4
MBMG     = MB - G                  # 42
MAXW     = 4                       # SHIFT_BI_MAXW
K        = KB * Z                  # 8448
FOUR_Z   = G * Z                   # 1536
KB_Z     = K                       # 8448
KB4_Z    = (KB + G) * Z            # 9984
MB4_Z    = MBMG * Z                # 16128
N_FULL   = (KB + MB) * Z           # 26112
N_RAW    = N_FULL - 2 * Z          # 25344
INFO_OUT = K - 2 * Z               # 7680
C_NUM    = 4                         # golden 样本数（main.cpp 内部 tile 到 LDPC_C_NUM）

assert K == 8448 and N_RAW == 25344 and INFO_OUT == 7680


# ---------------------------------------------------------------------------
# GF(2) 求逆（Gauss-Jordan）
# ---------------------------------------------------------------------------
def gf2_inv(M_in: np.ndarray) -> np.ndarray:
    n = M_in.shape[0]
    assert M_in.shape == (n, n), f"not square: {M_in.shape}"
    A = np.hstack([(M_in.astype(np.uint8) & 1), np.eye(n, dtype=np.uint8)])
    for i in range(n):
        if A[i, i] == 0:
            pivot = -1
            for r in range(i + 1, n):
                if A[r, i] == 1:
                    pivot = r
                    break
            if pivot < 0:
                raise ValueError(f"Singular at col {i}")
            A[[i, pivot]] = A[[pivot, i]]
        for r in range(n):
            if r != i and A[r, i] == 1:
                A[r] = A[r] ^ A[i]
    return A[:, n:]


# ---------------------------------------------------------------------------
# Sionna PCM
# ---------------------------------------------------------------------------
def build_sionna_encoder():
    from sionna.phy.fec.ldpc import LDPC5GEncoder
    enc = LDPC5GEncoder(k=K, n=N_RAW)
    z_sel  = getattr(enc, "_z", None)  or getattr(enc, "z", None)
    bg_sel = getattr(enc, "_bg", None) or getattr(enc, "bg", None)
    print(f"[sionna] k={enc.k}  n={enc.n}  z={z_sel}  bg={bg_sel}")
    if z_sel != Z:
        raise RuntimeError(f"Sionna 选了 Z={z_sel}，不是 {Z}")
    if bg_sel not in ("bg1", "BG1"):
        raise RuntimeError(f"Sionna 选了 BG={bg_sel}，不是 BG1")
    pcm = getattr(enc, "_pcm", None)
    if pcm is None:
        pcm = getattr(enc, "pcm", None)
    if pcm is None:
        raise RuntimeError("no pcm attr")
    H = pcm.toarray().astype(np.uint8) & 1
    expected = (MB * Z, (KB + MB) * Z)
    if H.shape != expected:
        raise RuntimeError(f"PCM shape {H.shape} != {expected}")
    return enc, H


def split_blocks(H):
    A  = H[:FOUR_Z, :KB_Z]          # 1536 × 8448
    B  = H[:FOUR_Z, KB_Z:KB4_Z]     # 1536 × 1536
    C_ = H[FOUR_Z:, :KB_Z]          # 16128 × 8448
    D  = H[FOUR_Z:, KB_Z:KB4_Z]     # 16128 × 1536
    return A, B, C_, D


# ---------------------------------------------------------------------------
# dense matmul 分解编码（与 ldpc_encode_ref 旧逻辑一致）
# ---------------------------------------------------------------------------
def encode_matmul(s, A, B_inv, C_, D):
    s32 = s.astype(np.int32)
    As  = (A.astype(np.int32)     @ s32.T)                  & 1
    p_a = (B_inv.astype(np.int32) @ As.astype(np.int32))    & 1
    Cs  =  C_.astype(np.int32)    @ s32.T
    Dp  =  D.astype(np.int32)     @ p_a.astype(np.int32)
    p_b = (Cs + Dp) & 1
    c_full = np.concatenate([s32, p_a.T, p_b.T], axis=1).astype(np.int8)
    return c_full                                            # (C_NUM, N_FULL)


# ---------------------------------------------------------------------------
# shift 表抽取（kernel 约定：out[r] = in[(r+k) mod Z]，即 H[r][(r+k)%Z]=1）
# 每块抽 row-0 的 1 位置作为候选 shift，再整块重建断言，convention-agnostic。
# ---------------------------------------------------------------------------
def _circ(k):
    B = np.zeros((Z, Z), dtype=np.uint8)
    r = np.arange(Z)
    B[r, (r + k) % Z] = 1
    return B

def extract_single(Hsub, n_br, n_bc, name):
    out = np.full(n_br * n_bc, -1, dtype=np.int16)
    for br in range(n_br):
        for bc in range(n_bc):
            blk = Hsub[br*Z:(br+1)*Z, bc*Z:(bc+1)*Z] & 1
            if blk.sum() == 0:
                continue
            ones0 = np.where(blk[0] == 1)[0]
            if ones0.size != 1:
                raise ValueError(f"{name}({br},{bc}): row0 有 {ones0.size} 个 1，非单 circulant")
            k = int(ones0[0])
            if not np.array_equal(blk, _circ(k)):
                raise ValueError(f"{name}({br},{bc}): 块 != circ(k={k})（kernel 约定）")
            out[br*n_bc + bc] = k
    return out

def extract_multi(M, n_br, n_bc, maxw, name):
    out = np.full(n_br * n_bc * maxw, -1, dtype=np.int16)
    for br in range(n_br):
        for bc in range(n_bc):
            blk = (M[br*Z:(br+1)*Z, bc*Z:(bc+1)*Z] & 1).astype(np.uint8)
            if blk.sum() == 0:
                continue
            shifts = np.where(blk[0] == 1)[0]
            if shifts.size > maxw:
                raise ValueError(f"{name}({br},{bc}): {shifts.size} 项 circulant > MAXW={maxw}")
            recon = np.zeros((Z, Z), dtype=np.uint8)
            for k in shifts:
                recon ^= _circ(int(k))
            if not np.array_equal(recon, blk):
                raise ValueError(f"{name}({br},{bc}): 非 circulant 之和（row0 推导失败）")
            for w, k in enumerate(sorted(int(x) for x in shifts)):
                out[(br*n_bc + bc)*maxw + w] = k
    return out


# ---------------------------------------------------------------------------
# shift-form 复算（与 kernel 逐 op 同构：XOR == 整数累加后 &1）
# ---------------------------------------------------------------------------
def _gidx(k):
    return (np.arange(Z) + k) % Z

def encode_via_shifts(s, sA, sBi, sC, sD):
    C = s.shape[0]
    sblk = s.reshape(C, KB, Z).astype(np.uint8)

    Ax = np.zeros((C, G, Z), dtype=np.uint8)
    for br in range(G):
        acc = np.zeros((C, Z), dtype=np.uint8)
        for bc in range(KB):
            k = int(sA[br*KB + bc])
            if k < 0:
                continue
            acc ^= sblk[:, bc, :][:, _gidx(k)]
        Ax[:, br, :] = acc

    Pa = np.zeros((C, G, Z), dtype=np.uint8)
    for br in range(G):
        acc = np.zeros((C, Z), dtype=np.uint8)
        for bc in range(G):
            for w in range(MAXW):
                k = int(sBi[(br*G + bc)*MAXW + w])
                if k < 0:
                    continue
                acc ^= Ax[:, bc, :][:, _gidx(k)]
        Pa[:, br, :] = acc

    Pb = np.zeros((C, MBMG, Z), dtype=np.uint8)
    for br in range(MBMG):
        acc = np.zeros((C, Z), dtype=np.uint8)
        for bc in range(KB):
            k = int(sC[br*KB + bc])
            if k < 0:
                continue
            acc ^= sblk[:, bc, :][:, _gidx(k)]
        for bc in range(G):
            k = int(sD[br*G + bc])
            if k < 0:
                continue
            acc ^= Pa[:, bc, :][:, _gidx(k)]
        Pb[:, br, :] = acc

    info_out = s[:, 2*Z:]
    out = np.concatenate([info_out, Pa.reshape(C, G*Z), Pb.reshape(C, MBMG*Z)],
                         axis=1).astype(np.int8)
    return out                                              # (C, N_RAW)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("T4 ldpc_encode | Sionna 参考 + shift 表生成（self-contained）")
    print(f"  DATA = {DATA}")
    print("=" * 64)

    # 1. PCM
    enc, H = build_sionna_encoder()
    A, B, C_, D = split_blocks(H)
    print(f"[split] A={A.shape}  B={B.shape}  C={C_.shape}  D={D.shape}")

    # 2. B_inv
    print("[gf2] 求 B_inv (1536×1536) ...")
    B_inv = gf2_inv(B)
    if not np.array_equal((B.astype(np.int32) @ B_inv.astype(np.int32)) & 1,
                          np.eye(FOUR_Z, dtype=np.uint8)):
        raise RuntimeError("B · B_inv != I")
    print("[gf2] B_inv ✓")

    # 3. 抽 shift 表
    shift_A  = extract_single(A,  G,    KB, "A")
    shift_C  = extract_single(C_, MBMG, KB, "C")
    shift_D  = extract_single(D,  MBMG, G,  "D")
    shift_Bi = extract_multi (B_inv, G, G, MAXW, "Bi")
    print(f"[shift] A={shift_A.size} Bi={shift_Bi.size} C={shift_C.size} D={shift_D.size}  抽取+重建 ✓")

    # 4. 测试输入
    rng = np.random.default_rng(seed=0x181c8001)
    s = rng.integers(0, 2, size=(C_NUM, K), dtype=np.int8)

    # 5a. dense 分解 + syndrome
    c_full = encode_matmul(s, A, B_inv, C_, D)
    if np.any((H.astype(np.int32) @ c_full.astype(np.int32).T) & 1):
        raise RuntimeError("syndrome 非零，H·c≠0")
    c_punct = c_full[:, 2*Z:]
    assert c_punct.shape == (C_NUM, N_RAW)
    print("[check-a] H·c = 0  ✓")

    # 5b. shift-form 复算 == dense 分解
    c_shift = encode_via_shifts(s, shift_A, shift_Bi, shift_C, shift_D)
    d = int(np.sum(c_shift != c_punct))
    if d != 0:
        raise RuntimeError(f"[check-b] shift-form 复算 vs dense 有 {d} bit 差异（shift 表错）")
    print("[check-b] shift-form 复算 == dense 分解  bit-exact ✓")

    # 5c. Sionna 官方 encoder bit-exact
    import tensorflow as tf
    c_sionna = enc(tf.constant(s.astype(np.float32))).numpy().astype(np.int8)
    if c_sionna.shape == c_punct.shape:
        d = int(np.sum(c_punct != c_sionna))
        if d == 0:
            print("[check-c] 与 Sionna LDPC5GEncoder bit-exact ✓")
        else:
            raise RuntimeError(f"[check-c] 与 Sionna 有 {d} bit 差异")
    else:
        raise RuntimeError(f"[check-c] shape: sionna={c_sionna.shape} vs ours={c_punct.shape}")

    # 6. 落盘（shift 表 + golden，全本地）
    shift_A .astype(np.int16).tofile(WEIGHT_DIR / "shift_A.bin")
    shift_Bi.astype(np.int16).tofile(WEIGHT_DIR / "shift_Bi.bin")
    shift_C .astype(np.int16).tofile(WEIGHT_DIR / "shift_C.bin")
    shift_D .astype(np.int16).tofile(WEIGHT_DIR / "shift_D.bin")
    s      .astype(np.int8).tofile(GOLDEN_DIR / "input.bin")
    c_punct.astype(np.int8).tofile(GOLDEN_DIR / "output.bin")

    print(f"[weights] -> {WEIGHT_DIR}")
    print(f"  shift_A {shift_A.nbytes}B  shift_Bi {shift_Bi.nbytes}B  "
          f"shift_C {shift_C.nbytes}B  shift_D {shift_D.nbytes}B")
    print(f"[golden]  -> {GOLDEN_DIR}")
    print(f"  input.bin  {s.shape}  {s.nbytes}B")
    print(f"  output.bin {c_punct.shape}  {c_punct.nbytes}B")
    print("\n[done] 三重自校验通过，权重/golden 就绪")


if __name__ == "__main__":
    main()
