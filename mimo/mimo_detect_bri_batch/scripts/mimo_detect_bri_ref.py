#!/usr/bin/env python3
"""
§0 参考: mimo_detect_bri — massive MIMO LMMSE, Block-Richardson 迭代求逆 (维度由 NR/NL 环境变量决定)
  按《基于 AI 核的高效物理层算法和算子设计二阶段报告》2.4 节 BRI:
    A=HᴴH+σ²I 分块对角预条件 M (B×B块, Cholesky) + Richardson 迭代 (全 batched GEMM)
  子载波打包: batch 个 RE 一起, 所有矩阵乘是 [batch,K,K] batched GEMM -> Cube
  默认: NR=64, NL=16, B=8, L=5 (报告 Table 8 U=16 推荐)
  环境变量: NR / NL / NL_REAL / BRI_B / BRI_L / AIRAN_SCPAD / SNR_DB / PACK / SHARED
  PACK=1: 打包模式 —— 一个 [M,16] 瓦片装 P=16/NL_REAL 个 RE, 零补零.
          golden 的 xhat/no_eff/mask/no 与 PACK=0 【逐字节相同】(A 块对角, BRI 精确解耦),
          只有 hrm_re/im 和 yvpad_re/im 的布局不同 -> 所以用 _pk 后缀分目录.
  收敛法则 (违反必挂): B = NL_REAL/2  且  NR >= 8*B   <=>  NR >= 4*NL_REAL
"""
import os, numpy as np

NR   = int(os.environ.get("NR", 64))      # Rx 天线
NL   = int(os.environ.get("NL", 16))      # 层 (kernel 维度, 恒 16)
NLR  = int(os.environ.get("NL_REAL", NL)) # 真实层数; < NL 时 H 的高列补零, kernel 一行不改
BLK  = int(os.environ.get("BRI_B", 8))    # 块大小
PACK = int(os.environ.get("PACK", 0))     # 1 = 打包布局 (P=16/NL_REAL 个 RE 共用一个瓦片)
NLAY = int(os.environ.get("BRI_L", 5))    # 迭代层数
BATCH= int(os.environ.get("BATCH", 64))   # 子载波打包
GROUP_BATCH = int(os.environ.get("GROUP_BATCH", 8))
SNR_DB = float(os.environ.get("SNR_DB", 20.0))
SEED = 20260531

# 5G grid (与链路一致)
N_SYMBOL = 14
N_SC_PAD = int(os.environ.get('AIRAN_SCPAD', 1664))
N_SC_USED = min(1596, N_SC_PAD - 4)
N_RE = N_SYMBOL * N_SC_PAD
QN = 1.0/np.sqrt(170.0)

DATA_DIR = os.environ.get("AIRAN_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
assert 1 <= NLR <= NL, f"NL_REAL={NLR} 必须在 [1, NL={NL}]"
P = (NL // NLR) if PACK else 1            # 一个瓦片里的 RE 数
if PACK:
    assert NL % NLR == 0, f"打包要求 NL_REAL 整除 {NL}; 可用 1/2/4/8/16"
    assert NLR % BLK == 0, f"BRI_B={BLK} 必须整除 NL_REAL={NLR}, 否则预条件块会跨 RE"
    assert N_RE % (NL * P) == 0, f"N_RE={N_RE} 必须被 16*P={NL*P} 整除 (staging block)"
# NLR < NL 时加 _r 后缀; P>1 时再加 _pk (打包与补零的 hrm 布局不同, 不能共用目录)
CASE = (f"case_0_m{NR}_k{NL}_sc{N_SC_PAD}"
        + (f"_r{NLR}" if NLR != NL else "") + ("_pk" if P > 1 else ""))
GOLD_DIR = os.path.join(DATA_DIR, "golden", CASE)

def w16(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.astype(np.float16).tofile(path)

def gen_case(rng):
    """频域相关 massive MIMO 信道 (抽头时延, 保证子载波打包内 H 平滑相关)"""
    used = np.zeros(N_RE, bool)
    grid = used.reshape(N_SYMBOL, N_SC_PAD); grid[:, :N_SC_USED] = True
    H = np.zeros((NR, NL, N_SYMBOL, N_SC_PAD), complex)
    ntap = 8
    sc = np.arange(N_SC_PAD)
    for sym in range(N_SYMBOL):
        for mu in range(NR):
            for k in range(NLR):          # 第 NLR..NL-1 层保持全零 (补零层)
                g = (rng.standard_normal(ntap)+1j*rng.standard_normal(ntap))/np.sqrt(2*ntap)
                tau = rng.uniform(0, 16.0, ntap)          # 小时延 -> 频域平坦 (相干带宽 >> batch)
                H[mu,k,sym,:] = np.sum(g[None,:]*np.exp(-2j*np.pi*sc[:,None]*tau[None,:]/N_SC_PAD), axis=1)
    H = H.reshape(NR, NL, N_RE)
    x = np.zeros((NL, N_RE), complex)
    x[:NLR] = ((rng.integers(0,16,(NLR,N_RE))*2-15)+1j*(rng.integers(0,16,(NLR,N_RE))*2-15))*QN
    x[:, ~used] = 0
    no = 10**(-SNR_DB/10)
    y = np.einsum('rln,ln->rn', H, x) + (rng.standard_normal((NR,N_RE))+1j*rng.standard_normal((NR,N_RE)))*np.sqrt(no/2)
    no_grid = np.full(N_RE, no)
    return H, y, no_grid, x, used

def exact_lmmse(H, y, no, used):
    """精确逐 RE LMMSE (golden 基准, 对齐 Sionna)"""
    idx = np.where(used)[0]
    Hu = np.transpose(H[:,:,idx],(2,0,1))                # [n,M,K]
    yu = np.transpose(y[:,idx],(1,0)); nfl = no[idx]
    A = np.conj(np.transpose(Hu,(0,2,1)))@Hu + nfl[:,None,None]*np.eye(NL)
    m = np.einsum('bkm,bm->bk', np.conj(np.transpose(Hu,(0,2,1))), yu)
    xm = np.linalg.solve(A, m[...,None])[...,0]
    Ainv = np.linalg.inv(A)
    d = 1 - nfl[:,None]*np.real(np.diagonal(Ainv,axis1=-2,axis2=-1))
    d = np.maximum(d, 1e-3)
    xhat = xm/d; no_eff = (1-d)/d
    xg = np.zeros((NL,N_RE),complex); ng = np.zeros((NL,N_RE))
    xg[:,idx] = xhat.T; ng[:,idx] = no_eff.T
    return xg, ng

def bri_lmmse(H, y, no, used, B=BLK, L=NLAY):
    """BRI 求逆版 (kernel 将实现的): 块预条件 + Richardson 迭代, 全 batched GEMM"""
    idx = np.where(used)[0]
    Hu = np.transpose(H[:,:,idx],(2,0,1)); yu = np.transpose(y[:,idx],(1,0)); nfl = no[idx]
    A = np.conj(np.transpose(Hu,(0,2,1)))@Hu + nfl[:,None,None]*np.eye(NL)   # ① Cube: HᴴH+σ²I
    m = np.einsum('bkm,bm->bk', np.conj(np.transpose(Hu,(0,2,1))), yu)        # ① Cube: Hᴴy
    # ② 块对角预条件 M⁻¹
    Minv = np.zeros_like(A)
    for s in range(0, NL, B):
        e = min(s+B, NL); Minv[:,s:e,s:e] = np.linalg.inv(A[:,s:e,s:e])       # 8×8 Cholesky/direct
    I = np.eye(NL)[None]
    X = Minv.copy()                                                          # X_0 = M⁻¹
    for _ in range(L):
        X = X + Minv @ (I - A @ X)                                          # ③ BRI: 全 batched GEMM
    xm = np.einsum('bkj,bj->bk', X, m)                                       # ④ x = X·m
    dinv = np.real(np.diagonal(X, axis1=-2, axis2=-1))
    d = np.maximum(1 - nfl[:,None]*dinv, 1e-3)
    xhat = xm/d; no_eff = (1-d)/d
    xg = np.zeros((NL,N_RE),complex); ng = np.zeros((NL,N_RE))
    xg[:,idx] = xhat.T; ng[:,idx] = no_eff.T
    return xg, ng


def bri_lmmse_shared(H, y, no, used, B=BLK, L=NLAY, group=None):
    """RB 共享版: 每 group 个连续 RE 共享同一个 A⁻¹ (用组内首个 RE 的 H 算).
       m=Hᴴy 和 x̂=X·m 仍逐 RE. 验证共享掉多少精度."""
    if group is None: group = BATCH
    idx = np.where(used)[0]
    Hu = np.transpose(H[:,:,idx],(2,0,1)); yu = np.transpose(y[:,idx],(1,0)); nfl = no[idx]
    n = len(idx)
    # A/m 逐 RE 先算 (m 要逐 RE; A 只在组首用)
    A_all = np.conj(np.transpose(Hu,(0,2,1)))@Hu + nfl[:,None,None]*np.eye(NL)
    m = np.einsum('bkm,bm->bk', np.conj(np.transpose(Hu,(0,2,1))), yu)
    xm = np.zeros((n, NL), complex); dinv = np.zeros((n, NL))
    I = np.eye(NL)[None]
    for g0 in range(0, n, group):
        g1 = min(g0+group, n)
        A = A_all[g0:g0+1]                             # ★ 组首的 A, 共享
        Minv = np.zeros_like(A)
        for s in range(0, NL, B):
            e = min(s+B, NL); Minv[:,s:e,s:e] = np.linalg.inv(A[:,s:e,s:e])
        X = Minv.copy()
        for _ in range(L): X = X + Minv @ (I - A @ X)  # 单份 X=A⁻¹
        Xg = X[0]                                       # [NL,NL] 共享给组内所有 RE
        for r in range(g0, g1):
            xm[r] = Xg @ m[r]                           # x̂=X·m_r 逐 RE
            dinv[r] = np.real(np.diag(Xg))              # 对角 (共享, 用于 unbias)
    d = np.maximum(1 - nfl[:,None]*dinv, 1e-3)
    xhat = xm/d; no_eff = (1-d)/d
    xg = np.zeros((NL,N_RE),complex); ng = np.zeros((NL,N_RE))
    xg[:,idx] = xhat.T; ng[:,idx] = no_eff.T
    return xg, ng


def dump_A(H, y, no, used, nre_dump=64):
    """阶段① 对照: 前 nre_dump 个 RE 的 A=HᴴH+σ²I 和 m=Hᴴy (re/im)"""
    idx = np.where(used)[0][:nre_dump]
    Hu = np.transpose(H[:,:,idx],(2,0,1)); yu = np.transpose(y[:,idx],(1,0)); nfl = no[idx]
    A = np.conj(np.transpose(Hu,(0,2,1)))@Hu + nfl[:,None,None]*np.eye(NL)
    m = np.einsum('bkm,bm->bk', np.conj(np.transpose(Hu,(0,2,1))), yu)
    w16(os.path.join(GOLD_DIR,"A_re.bin"), A.real); w16(os.path.join(GOLD_DIR,"A_im.bin"), A.imag)
    w16(os.path.join(GOLD_DIR,"m_re.bin"), m.real); w16(os.path.join(GOLD_DIR,"m_im.bin"), m.imag)
    # 阶段② 对照: M^-1 = blockdiag(8×8 块逆)
    Minv = np.zeros_like(A)
    for sblk in range(0, NL, BLK):
        e = min(sblk+BLK, NL); Minv[:, sblk:e, sblk:e] = np.linalg.inv(A[:, sblk:e, sblk:e])
    w16(os.path.join(GOLD_DIR,"minv_re.bin"), Minv.real); w16(os.path.join(GOLD_DIR,"minv_im.bin"), Minv.imag)
    Ainv = np.linalg.inv(A)   # 完整逆, 对照 BRI 的 X
    w16(os.path.join(GOLD_DIR,"xinv_re.bin"), Ainv.real); w16(os.path.join(GOLD_DIR,"xinv_im.bin"), Ainv.imag)
    print(f"[io] 阶段①② 对照 A/m/Minv[{nre_dump},{NL},{NL}] -> {GOLD_DIR}/")

def dump_h_remajor(H, y):
    """RE-major: h_rm[re] = H[:,:,re] [M,K] 连续; y_rm[re]=y[:,re] [M]. kernel Gram 连续载入."""
    Hrm = np.transpose(H,(2,0,1))          # [N_RE, M, K]
    yrm = np.transpose(y,(1,0))            # [N_RE, M]
    w16(os.path.join(GOLD_DIR,"hrm_re.bin"), Hrm.real); w16(os.path.join(GOLD_DIR,"hrm_im.bin"), Hrm.imag)
    # yvpad [N_RE,NR,NL]: y 复制到 NL 列, kernel 的 MatchFilter 一次算出 m 的每一列
    yvpad = np.repeat(yrm[:,:,None], NL, axis=2)   # [N_RE,NR,NL]
    w16(os.path.join(GOLD_DIR,"yvpad_re.bin"), yvpad.real); w16(os.path.join(GOLD_DIR,"yvpad_im.bin"), yvpad.imag)
    # clean 最终数据流：连续 GB 个独立 RE 的 y 分别放在前 GB 列。
    assert N_RE % GROUP_BATCH == 0 and GROUP_BATCH <= NL
    yg = np.zeros((N_RE // GROUP_BATCH, NR, NL), complex)
    yg[:, :, :GROUP_BATCH] = np.transpose(
        yrm.reshape(N_RE // GROUP_BATCH, GROUP_BATCH, NR), (0, 2, 1))
    w16(os.path.join(GOLD_DIR,"yvgroup_re.bin"), yg.real)
    w16(os.path.join(GOLD_DIR,"yvgroup_im.bin"), yg.imag)
    print(f"[io] 补零布局 H[{N_RE},{NR},{NL}] yv[{N_RE},{NR},{NL}] -> {GOLD_DIR}/")
    print(f"[io] Sparse Match RHS ygroup[{N_RE//GROUP_BATCH},{NR},{NL}]，较 yvpad 缩小 {GROUP_BATCH}x")


def check_rule():
    """收敛法则自检. 2026-07-26: 整晚在追一个算法上无解的配置, 教训写进工具里."""
    beff = min(BLK, NLR)
    rho  = 2.0*np.sqrt(beff/NR)                       # Neumann 谱半径的经验估计
    ok_n = NR >= 8*beff                               # Neumann 收敛
    ok_b = 2*BLK >= NLR                               # BRI 收敛 (预条件子覆盖度)
    print(f"[法则] NR={NR} NL_REAL={NLR} B={BLK} PACK={PACK} P={P}"
          f"  ->  rho~2*sqrt(B_eff/NR)={rho:.3f}")
    print(f"[法则]   Neumann  NR >= 8*min(B,NL_REAL) : {NR} >= {8*beff}  {'OK' if ok_n else '★违反★'}")
    print(f"[法则]   BRI      B  >= NL_REAL/2        : {BLK} >= {NLR/2:g}  {'OK' if ok_b else '★违反★'}")
    if not (ok_n and ok_b):
        print(f"[法则] ⚠ 这个配置算法上不收敛, kernel 再对也会 FAIL. 推荐 B={max(1,NLR//2)}, NR>={4*NLR}")
    return ok_n and ok_b



def re_of(u, p):
    """打包排列: unit u = b*16+f 的第 p 个槽  ->  RE = b*(16P) + 16p + f
       于是攒 16 个 unit 做【一次】16x16 转置后, 第 p*NLR+l 行沿 f 展开
       正好是连续 16 个 RE 的第 l 层 -> 直写标准 [NL][N_RE], 32B 对齐天然满足."""
    return (u // NL) * (NL * P) + NL * p + (u % NL)


def no_perslot(no):
    """no.bin 改成逐槽布局 [N_UNIT][16]: 第 k 个槽的 σ² = no[re_of(u, k//NLR)].
       P=1 时 16 个值相同 -> 与旧的逐 RE 标量语义等价.
       动机: 打包后一个瓦片的 P 个 RE 相隔 16 个子载波、会跨 RB, 共用一个 σ²
       在 20dB 上惩罚仅 +0.004pp, 但 5dB 时 +1.6pp、0dB 时 +5.7pp (超门限)."""
    U = N_RE // P
    u = np.arange(U)
    out = np.zeros((U, NL))
    for p in range(P):
        out[:, p*NLR:(p+1)*NLR] = np.asarray(no)[re_of(u, p)][:, None]
    return out


def dump_h_packed(H, y):
    """打包布局: hrm[u] 是 [M,16] 瓦片, 列 p*NLR+l = H[re_of(u,p)][m][l]. 零补零."""
    Hrm = np.transpose(H, (2, 0, 1))       # [N_RE, NR, NL]
    yrm = np.transpose(y, (1, 0))          # [N_RE, NR]
    U = N_RE // P
    u = np.arange(U)
    Hp = np.zeros((U, NR, NL), complex)
    Yp = np.zeros((U, NR, NL), complex)
    for p in range(P):
        re = re_of(u, p)
        assert re.max() < N_RE and len(set(re.tolist())) == U, "排列越界或有重复"
        Hp[:, :, p*NLR:(p+1)*NLR] = Hrm[re][:, :, :NLR]
        Yp[:, :, p*NLR:(p+1)*NLR] = yrm[re][:, :, None]   # y 复制到该槽的 NLR 列
    if P == 1:                             # 自检: P=1 必须退化成恒等, 与旧布局逐位相同
        assert np.array_equal(Hp, Hrm) and np.allclose(Yp, np.repeat(yrm[:, :, None], NL, axis=2))
        print("[pack] P=1 自检通过: 布局与非打包版逐位相同")
    w16(os.path.join(GOLD_DIR, "hrm_re.bin"), Hp.real); w16(os.path.join(GOLD_DIR, "hrm_im.bin"), Hp.imag)
    w16(os.path.join(GOLD_DIR, "yvpad_re.bin"), Yp.real); w16(os.path.join(GOLD_DIR, "yvpad_im.bin"), Yp.imag)
    print(f"[pack] P={P}  unit={U}  瓦片 [{NR},{NL}] 装 {P} 个 RE, 利用率 {100*P*NLR/NL:.0f}% (零补零)")
    print(f"[pack] 排列: unit u=b*16+f 的槽 p -> RE = b*{NL*P} + {NL}*p + f")
    print(f"[io] 打包 H[{U},{NR},{NL}] yv[{U},{NR},{NL}] -> {GOLD_DIR}/")


def main():
    check_rule()
    rng = np.random.default_rng(SEED)
    H, y, no, xt, used = gen_case(rng)
    xg, ng = exact_lmmse(H, y, no, used)              # 精确基准
    xb, nb = bri_lmmse(H, y, no, used)                # 逐 RE BRI 版
    idx = np.where(used)[0]
    R = np.s_[:NLR, idx]                              # 只看真实层, 补零层丢弃
    evm = np.sqrt(np.mean(np.abs((xb-xg)[R])**2)/np.mean(np.abs(xg[R])**2))
    ner = np.median(np.abs((nb-ng)[R])/(np.abs(ng[R])+1e-3))
    print(f"[BRI/f64] NR={NR} NL={NL} NL_REAL={NLR} B={BLK} L={NLAY} batch={BATCH} SNR={SNR_DB}dB")
    print(f"[BRI/f64] 逐RE BRI vs 精确 LMMSE: x_hat EVM={evm*100:.2f}%  no_eff relmed={ner:.2e}")
    # RB 共享 A⁻¹ 的精度代价 —— 探索性分析, 默认关闭 (SHARED=1 打开)
    if int(os.environ.get("SHARED", 0)):
        for grp in [16, 12, 8, 4]:
            xs, _ = bri_lmmse_shared(H, y, no, used, group=grp)
            evms = np.sqrt(np.mean(np.abs((xs-xg)[R])**2)/np.mean(np.abs(xg[R])**2))
            print(f"[共享/f64] group={grp:2d}: EVM={evms*100:.2f}%  "
                  f"({'PASS' if evms<0.05 else 'FAIL'}, 门限5%)")
    print(f"[BRI/f64] {'PASS' if evm<0.05 else 'CHECK'}  (报告: SE 差 ≤1.5%, EVM ~2% 对应)")
    # golden —— 只写真正被 main.cpp / verify_result.py 读的文件.
    # (曾经还写 h_re/h_im/y_re/y_im/yrm/hh, 没有任何消费者, 64x16 每 case 白占 ~203MB)
    for nm,a in [("no", no_perslot(no) if P > 1 else no), ("mask", used.astype(np.float32)),
                 ("xhat_re",xg.real), ("xhat_im",xg.imag), ("no_eff",ng)]:
        w16(os.path.join(GOLD_DIR, nm+".bin"), np.asarray(a))
    dump_A(H, y, no, used)
    if PACK:
        dump_h_packed(H, y)
    else:
        dump_h_remajor(H, y)
    print(f"[io] no.bin = " + (f"逐槽 [{N_RE//P},{NL}]" if P > 1 else f"逐 RE 标量 [{N_RE}]"))
    print(f"[io] golden -> {GOLD_DIR}/  (精确基准; BRI kernel 输出对它验 EVM, 只比前 {NLR} 层)")

if __name__ == "__main__":
    main()
