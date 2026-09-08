#!/usr/bin/env python3
# ============================================================================
# mimo_dmrs_ref.py — K 层正交 DMRS (3GPP 38.211 Type 1) + 分离验证
#
# 复用 dmrs_ref.py 的 Gold 序列 (dmrs_cinit / dmrs_qpsk 不变).
# 层间正交靠: CDM组(comb偏移 Δ) + 频域OCC(wf) + 时域OCC(wt).
#
# 3GPP 38.211 Table 6.4.1.1.3-1 (DMRS Type 1, 最多 4 端口):
#   port 1000: CDM0, Δ=0(偶), wf=[+1,+1], wt=[+1,+1]
#   port 1001: CDM0, Δ=0(偶), wf=[+1,-1], wt=[+1,+1]
#   port 1002: CDM1, Δ=1(奇), wf=[+1,+1], wt=[+1,+1]
#   port 1003: CDM1, Δ=1(奇), wf=[+1,-1], wt=[+1,+1]
#
# 关键: K 层共享同一个 Gold 基序列 (c_init 不依赖层!),
#       层间区分靠 (Δ, wf, wt),不是不同 Gold.
# ============================================================================
import numpy as np
import dmrs_ref as D   # 复用你现有的 (dmrs_cinit, dmrs_qpsk, _gold_bits)

# ── 3GPP 38.211 Type 1 端口配置 (port 1000..1003) ──────────────────────────
# 每个端口: (cdm_group, delta, wf, wt)
#   cdm_group: 0 或 1  (决定占偶/奇子载波)
#   delta:     comb 偏移 (0=偶SC, 1=奇SC) == cdm_group (Type 1)
#   wf: 频域 OCC, 长度2, 作用在 comb 内相邻的一对导频 (2k, 2k+1 位置)
#   wt: 时域 OCC, 长度2, 作用在两个 DMRS 符号 {l=2, l=11}
PORT_CFG = {
    1000: dict(cdm=0, delta=0, wf=np.array([+1, +1]), wt=np.array([+1, +1])),
    1001: dict(cdm=0, delta=0, wf=np.array([+1, -1]), wt=np.array([+1, +1])),
    1002: dict(cdm=1, delta=1, wf=np.array([+1, +1]), wt=np.array([+1, +1])),
    1003: dict(cdm=1, delta=1, wf=np.array([+1, -1]), wt=np.array([+1, +1])),
}
PORTS_4 = [1000, 1001, 1002, 1003]   # NL=4
PORTS_2 = [1000, 1001]               # NL=2 (同 CDM 组, 靠 OCC 区分)

N_SC = 1596          # 使用子载波数
N_RE = D.N_RE        # 798 = comb-2 每符号导频数
DMRS_SYMS = D.DMRS_SYMS   # (2, 11)


def gen_layer_dmrs(slot, n_id, n_scid, port):
    """生成单个端口(层)的 DMRS, 映射到完整 1596 子载波网格.
    返回 grid[2, N_SC] complex: 2 个 DMRS 符号 × 1596 子载波 (非导频位置=0).
    """
    cfg = PORT_CFG[port]
    delta = cfg['delta']       # comb 偏移: 该层占 SC = delta, delta+2, delta+4, ...
    wf, wt = cfg['wf'], cfg['wt']

    grid = np.zeros((2, N_SC), dtype=np.complex64)

    for si, l in enumerate(DMRS_SYMS):
        # 基础 Gold 序列 (复用你的, c_init 不依赖端口/层!)
        c_init = D.dmrs_cinit(slot, n_id, n_scid, l)
        r = D.dmrs_qpsk(c_init, N_RE)      # complex64[798], 基导频

        # 应用 OCC: wf 作用在 comb 内相邻导频对, wt 作用在符号间
        # wf[k%2]: 第 k 个导频乘 wf[k%2] (相邻一对 [wf0, wf1])
        wf_seq = np.tile(wf, N_RE // 2 + 1)[:N_RE]   # [wf0,wf1,wf0,wf1,...]
        r_occ = r * wf_seq * wt[si]                   # 乘频域+时域 OCC

        # comb 映射: 第 k 个导频放到子载波 (delta + 2k)
        sc_idx = delta + 2 * np.arange(N_RE)
        valid = sc_idx < N_SC
        grid[si, sc_idx[valid]] = r_occ[valid]

    return grid   # [2, 1596]


def gen_all_layers(slot, n_id, n_scid, ports):
    """K 层 DMRS. 返回 [K, 2, N_SC] complex."""
    return np.stack([gen_layer_dmrs(slot, n_id, n_scid, p) for p in ports], axis=0)


# ============================================================================
# 验证 1: 层间正交性 (相关矩阵应接近对角)
# ============================================================================
def verify_orthogonality(ports):
    print(f"\n=== 验证1: {len(ports)}层 DMRS 正交性 ===")
    layers = gen_all_layers(0, 0, 0, ports)   # [K,2,N_SC]
    K = len(ports)
    # 把 2 符号 × N_SC 展平成一个向量, 算层间相关
    vecs = layers.reshape(K, -1)   # [K, 2*N_SC]
    # 归一化相关矩阵
    G = vecs @ vecs.conj().T       # [K,K] Gram
    d = np.sqrt(np.real(np.diag(G)))
    Gn = np.abs(G) / np.outer(d, d)
    print("  归一化相关矩阵 |<层i, 层j>|:")
    for i in range(K):
        print("   ", " ".join(f"{Gn[i,j]:.3f}" for j in range(K)))
    off_diag = Gn - np.eye(K)
    max_off = np.abs(off_diag).max()
    print(f"  最大非对角(层间串扰) = {max_off:.4f}")
    print(f"  正交性: {'✓ PASS (对角≈1, 非对角≈0)' if max_off < 0.05 else '✗ 有串扰'}")
    return max_off < 0.05


# ============================================================================
# 验证 2: 端到端信道分离 (M×K 信道 → 分离出各层 H)
# ============================================================================
def verify_channel_separation(ports, NR=4, seed=1):
    K = len(ports)
    print(f"\n=== 验证2: NR={NR}, NL={K} 信道分离 ===")
    rng = np.random.default_rng(seed)

    # K 层 DMRS
    layers = gen_all_layers(0, 0, 0, ports)   # [K,2,N_SC]

    # 真实 M×K 信道 (每个 (Rx天线, 层) 一个复增益, 频域简化为常数/RE)
    # 为看清分离, 用频域平坦但层/天线不同的信道
    H_true = (rng.standard_normal((NR, K)) + 1j*rng.standard_normal((NR, K))) / np.sqrt(2)
    # [NR, K]

    # 接收: 每根天线收到 K 层叠加 (每符号每子载波)
    # y[NR, 2, N_SC] = Σ_k H[NR,k] * layers[k, 2, N_SC]
    y = np.einsum('rk,ksc->rsc', H_true, layers)   # [NR, 2, N_SC]
    # 加噪声
    noise_std = 0.01
    y += noise_std * (rng.standard_normal(y.shape) + 1j*rng.standard_normal(y.shape))/np.sqrt(2)

    # 接收端分离: 对每层, 用该层的 DMRS 相关 (正交性抵消其他层)
    # LS: H_hat[r,k] = <y[r], layer[k]> / <layer[k],layer[k]>
    H_hat = np.zeros((NR, K), dtype=np.complex64)
    for k in range(K):
        lk = layers[k].reshape(-1)          # [2*N_SC]
        norm_k = np.real(lk @ lk.conj())
        for r in range(NR):
            yr = y[r].reshape(-1)
            H_hat[r, k] = (yr @ lk.conj()) / norm_k

    err = np.abs(H_hat - H_true)
    print(f"  H 估计误差: 最大={err.max():.4f}, 平均={err.mean():.4f}")
    print(f"  分离效果: {'✓ PASS (H_hat ≈ H_true, 各层干净分离)' if err.max()<0.05 else '✗ 层间串扰'}")

    # 打印对比 (前2天线)
    print("  H_true vs H_hat (天线0):")
    for k in range(K):
        print(f"    层{k}: 真={H_true[0,k]:.3f}  估={H_hat[0,k]:.3f}")
    return err.max() < 0.05


if __name__ == "__main__":
    print("="*70)
    print(" K 层正交 DMRS (3GPP 38.211 Type 1) — numpy 验证")
    print("="*70)

    # NL=2 (同 CDM 组, OCC 区分)
    ok1 = verify_orthogonality(PORTS_2)
    ok2 = verify_channel_separation(PORTS_2, NR=4)

    # NL=4 (2 CDM组 × 2 OCC)
    ok3 = verify_orthogonality(PORTS_4)
    ok4 = verify_channel_separation(PORTS_4, NR=8)

    print("\n" + "="*70)
    print(f" 总结: NL=2 正交={ok1} 分离={ok2} | NL=4 正交={ok3} 分离={ok4}")
    print(f" {'✓ 全部 PASS — K 层正交 DMRS 可分离各层信道' if all([ok1,ok2,ok3,ok4]) else '✗ 有问题'}")
    print("="*70)
