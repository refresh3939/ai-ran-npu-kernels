#!/usr/bin/env python3
"""
§0 参考: precode_zf — massive MIMO 下行 ZF 预编码, Block-Richardson 迭代求逆 (Cube)

  下行:  y = H x + n,  x = G s      H=[NL,NT] (层×发射天线), G=[NT,NL], s=[NL]
  ZF:    G = Hᴴ (H Hᴴ)⁻¹,  逐列单位归一 ‖g_c‖=1  (对齐 Sionna rzf_precoding_matrix(alpha=0))

  ★ 复用 mimo_detect_bri 的两个同构点 (本脚本要证实/证伪的核心):
   ① Gram 复用: 存 W=Hᵀ [N_UNIT,NT,NL] (= detector 的 hrm 布局), detector 的 WᴴW = A_det,
      而 ZF 要的 A = HHᴴ = conj(A_det).  取共轭 = Aim 取负 = 反对称还原 Sub(Xᵀ,X) 操作数对调, 零成本.
      kernel 为数值稳定加入 εI；精确 golden 仍是纯 ZF，二者用 EVM 门控.
   ② 归一捷径: ‖g_c‖² = (Xᴴ A X)[c,c] --X=A⁻¹--> = A⁻¹[c,c] = diag(X)[c]
      而 diag(X) 是 detector unbias 已在算的量 -> 不必显式成 G 再求列范数:
         ŝ = s/√diag(X) ;  u = X·ŝ ;  x = conj(W)·u
      ⚠ X 是 BRI 近似解, 恒等式会退化, 本脚本量化.

  维度: NT(发射天线)=64, NL(层)=16, B=8, L=5.  NT/NL 可用环境变量覆盖.
"""
import os, numpy as np

NT   = int(os.environ.get("NT", 64))      # 发射天线 (= detector 的 M 轴)
NL   = int(os.environ.get("NL", 16))      # 打包后的列数 (= fractal 边长 16)
NLR  = int(os.environ.get("NLR", NL))     # 真实层数 (NLR<NL 时启用 PACK)
PACK_MODE = int(os.environ.get("PACK", "1" if NLR < NL else "0"))
PACK = (NL // NLR) if PACK_MODE else 1    # 每瓦片装 PACK 个 RE；0 为普通补零模式
BLK  = int(os.environ.get("BRI_B", 8))    # 预条件块大小
NLAY = int(os.environ.get("BRI_L", 5))    # Richardson 迭代层数
SNR_DB = float(os.environ.get("SNR_DB", 20.0))   # 仅 sanity 用
SEED = 20260531

N_SYMBOL = 14
N_SC_PAD = int(os.environ.get('AIRAN_SCPAD', 1664))
N_SC_USED = min(1596, N_SC_PAD - 4)
N_RE = N_SYMBOL * N_SC_PAD
QN = 1.0/np.sqrt(170.0)
DET_EPS = 1e-3

DATA_DIR = os.environ.get("AIRAN_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
CASE = (f"case_0_m{NT}_k{NL}_sc{N_SC_PAD}"
        + (f"_r{NLR}" if NLR != NL else "") + ("_pk" if PACK > 1 else ""))
GOLD_DIR = os.path.join(DATA_DIR, "golden", CASE)

def w16(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.astype(np.float16).tofile(path)


def gen_case(rng):
    """频域相关 massive MIMO 下行信道 H=[NL,NT,N_RE] (抽头时延, 组内平滑相关)"""
    used = np.zeros(N_RE, bool)
    used.reshape(N_SYMBOL, N_SC_PAD)[:, :N_SC_USED] = True
    H = np.zeros((NL, NT, N_SYMBOL, N_SC_PAD), complex)
    ntap = 8; sc = np.arange(N_SC_PAD)
    for sym in range(N_SYMBOL):
        for k in range(NLR):          # 只前 NLR 层有信道 (NLR..NL-1 补零)
            for m in range(NT):
                g = (rng.standard_normal(ntap)+1j*rng.standard_normal(ntap))/np.sqrt(2*ntap)
                tau = rng.uniform(0, 16.0, ntap)
                H[k,m,sym,:] = np.sum(g[None,:]*np.exp(-2j*np.pi*sc[:,None]*tau[None,:]/N_SC_PAD), axis=1)
    H = H.reshape(NL, NT, N_RE)
    H[:, :, ~used] = 0
    s = np.zeros((NL, N_RE), complex)
    s[:NLR] = ((rng.integers(0,16,(NLR,N_RE))*2-15)+1j*(rng.integers(0,16,(NLR,N_RE))*2-15))*QN
    s[:, ~used] = 0
    return H, s, used


def exact_zf(H, s, used):
    """精确 ZF golden (对齐 Sionna): G=Hᴴ(HHᴴ)⁻¹, 逐列单位归一, x=Gs"""
    idx = np.where(used)[0]
    Hu = np.transpose(H[:,:,idx],(2,0,1))          # [n,NL,NT]
    su = np.transpose(s[:,idx],(1,0))              # [n,NL]
    Hh = np.conj(np.transpose(Hu,(0,2,1)))         # [n,NT,NL]
    A  = Hu @ Hh                                   # [n,NL,NL] = H Hᴴ
    Areg = A + (DET_EPS if NLR<NL else 0.0)*np.eye(NL)[None]   # 补零层奇异才兜底
    G  = Hh @ np.linalg.inv(Areg)                  # [n,NT,NL]
    nrm = np.sqrt(np.sum(np.abs(G)**2, axis=1, keepdims=True))
    nrm = np.maximum(nrm, 1e-12)                   # 补零列 nrm=0, 避免除零
    G   = G / nrm
    x   = (G @ su[...,None])[...,0]                # [n,NT]
    xg = np.zeros((NT,N_RE),complex); xg[:,idx] = x.T
    return xg, G, A, nrm[:,0,:]


def bri_zf(H, s, used, B=BLK, L=NLAY):
    """BRI 版 (kernel 将实现的): A=HHᴴ -> 块对角预条件 + Richardson -> X≈A⁻¹
       归一走捷径 diag(X); x = conj(W)·(X·ŝ), 全 batched GEMM"""
    idx = np.where(used)[0]
    Hu = np.transpose(H[:,:,idx],(2,0,1))          # [n,NL,NT]
    su = np.transpose(s[:,idx],(1,0))              # [n,NL]
    W  = np.transpose(Hu,(0,2,1))                  # [n,NT,NL] = Hᵀ  (kernel 的 wrm)
    Hh = np.conj(W)                                # [n,NT,NL] = Hᴴ
    A  = Hu @ Hh                                   # ① Cube Gram: A=HHᴴ
    # ② 块对角预条件 M⁻¹
    Minv = np.zeros_like(A)
    for st in range(0, NL, B):
        e = min(st+B, NL)
        Minv[:,st:e,st:e] = np.linalg.inv(A[:,st:e,st:e]+DET_EPS*np.eye(e-st)[None])
    A = A + DET_EPS*np.eye(NL)[None]               # 与 kernel 一致的固定数值加载
    I = np.eye(NL)[None]
    X = Minv.copy()
    for _ in range(L):
        X = X + Minv @ (I - A @ X)                 # ③ BRI
    # ④ 归一捷径: ‖g_c‖² = diag(X)  (X=A⁻¹ 时严格成立)
    dg = np.maximum(np.real(np.diagonal(X, axis1=-2, axis2=-1)), DET_EPS)   # [n,NL]
    sh = su / np.sqrt(dg)                          # ŝ = s/‖g_c‖
    u  = np.einsum('bij,bj->bi', X, sh)            # ⑤ u = X·ŝ
    x  = np.einsum('bmi,bi->bm', Hh, u)            # ⑥ x = conj(W)·u = Hᴴ u
    xg = np.zeros((NT,N_RE),complex); xg[:,idx] = x.T
    return xg, X, dg



# ═══════════════════ PACK: NLR<16 时 P 个 RE 共用一个 16 列瓦片 ═══════════════════
#   §0 已证 (EVM 0.000%): ① A=HHᴴ 打包后有跨 RE 项(实测最大~41), 必须块对角掩码抹掉;
#   ② X/u 打包算 (块对角隔离跨 RE); ③ 末段 x=Hᴴu 沿层求和会叠加 P 个 RE, 必须按 RE 解包.
def _pack_mask():
    """块对角掩码 [NL,NL]: 只保留每个 RE 的 NLR×NLR 块, 跨 RE 项抹零"""
    M = np.zeros((NL, NL))
    for p in range(PACK):
        M[p*NLR:(p+1)*NLR, p*NLR:(p+1)*NLR] = 1.0
    return M

def bri_zf_pack(H, s, used, B=BLK, L=NLAY):
    """PACK 版: P 个 RE 的 NLR 层拼进 16 列瓦片, 掩码 Gram, 末段按 RE 解包.
       与 bri_zf 数值等价 (§0 EVM 0.000%), 但 Vector op ~/P (BRI/Gram/BlockInv 打包)。"""
    idx = np.where(used)[0]
    n = len(idx)
    assert n % PACK == 0, f"used RE 数 {n} 必须被 PACK={PACK} 整除 (host 端保证)"
    nt = n // PACK
    Hu = np.transpose(H[:NLR,:,idx],(2,0,1))       # [NLR,NT,n] -> [n,NLR,NT]
    su = np.transpose(s[:NLR][:,idx],(1,0))        # [NLR,n] -> [n,NLR]

    # ── 打包: 瓦片 t 的列 [p*NLR..(p+1)*NLR) = RE(t*PACK+p) 的 NLR 层 ──
    Hpk = np.zeros((nt, NL, NT), complex)          # [nt,16,NT]
    spk = np.zeros((nt, NL), complex)
    for t in range(nt):
        for p in range(PACK):
            re = t*PACK + p
            Hpk[t, p*NLR:(p+1)*NLR, :] = Hu[re]
            spk[t, p*NLR:(p+1)*NLR]    = su[re]
    Hh = np.conj(np.transpose(Hpk,(0,2,1)))        # [nt,NT,16]

    # ① Gram + 掩码 (跨 RE 项抹零)
    Mpack = _pack_mask()[None]
    A0 = (Hpk @ Hh) * Mpack
    # ② 块对角 M⁻¹  ③ BRI
    Minv = np.zeros_like(A0)
    for st in range(0, NL, B):
        e = min(st+B, NL)
        Minv[:,st:e,st:e] = np.linalg.inv(A0[:,st:e,st:e]+DET_EPS*np.eye(e-st)[None])
    A = A0 + DET_EPS*np.eye(NL)[None]
    I = np.eye(NL)[None]; X = Minv.copy()
    for _ in range(L): X = X + Minv @ (I - A @ X)
    # ④ 归一 + u (打包)
    dg = np.maximum(np.real(np.diagonal(X, axis1=-2, axis2=-1)), DET_EPS)   # [nt,16]
    u  = np.einsum('bij,bj->bi', X, spk/np.sqrt(dg))                        # [nt,16]

    # ⑤ 末段【解包】: 每 RE 用自己的 NLR 层 u 和自己的 W, x=conj(H)·u  [NT]
    x = np.zeros((n, NT), complex)
    for t in range(nt):
        for p in range(PACK):
            re = t*PACK + p
            x[re] = np.conj(Hu[re].T) @ u[t, p*NLR:(p+1)*NLR]               # [NT,NLR]·[NLR]
    xg = np.zeros((NT, N_RE), complex); xg[:, idx] = x.T
    return xg, X, dg


def dump_binaries(H, s, used, xg):
    """RE-major 布局 (与 detector 的 hrm 同构, kernel Gram 可一次 DataCopy 搬 GB 个 RE)"""
    W = np.transpose(H,(2,1,0))          # [N_RE, NT, NL] = Hᵀ  ★ kernel 的 wrm
    srm = np.transpose(s,(1,0))          # [N_RE, NL]           ★ RE-major s
    w16(os.path.join(GOLD_DIR,"wrm_re.bin"), W.real); w16(os.path.join(GOLD_DIR,"wrm_im.bin"), W.imag)
    w16(os.path.join(GOLD_DIR,"srm_re.bin"), srm.real); w16(os.path.join(GOLD_DIR,"srm_im.bin"), srm.imag)
    w16(os.path.join(GOLD_DIR,"mask.bin"), used.astype(np.float32))
    w16(os.path.join(GOLD_DIR,"x_re.bin"), xg.real); w16(os.path.join(GOLD_DIR,"x_im.bin"), xg.imag)
    print(f"[io] RE-major W[{N_RE},{NT},{NL}] s[{N_RE},{NL}] x[{NT},{N_RE}] -> {GOLD_DIR}/")



def re_of(u, p):
    """打包排列 (抄 detector): unit u=b*16+f 的第 p 个槽 -> RE = b*(16P) + 16p + f.
       攒 16 unit 转置后第 p*NLR+l 行 = 连续 16 个 RE 的第 l 层 -> 直写标准 [NL][N_RE]。"""
    return (u // NL) * (NL * PACK) + NL * p + (u % NL)

def dump_binaries_pack(H, s, used, xg):
    """打包布局 golden: wrm[u] 是 [NT,16] 瓦片, 列 p*NLR+l = W[re_of(u,p)][m][l]. 零补零。
       x 仍是标准 [NT,N_RE] (末段已解包, 每 RE 独立)。"""
    W = np.transpose(H,(2,1,0))          # [N_RE,NT,NL] = Hᵀ
    srm = np.transpose(s,(1,0))          # [N_RE,NL]
    U = N_RE // PACK
    u = np.arange(U)
    Wp = np.zeros((U, NT, NL), complex)
    sp = np.zeros((U, NL), complex)
    for p in range(PACK):
        re = re_of(u, p)
        assert re.max() < N_RE and len(set(re.tolist())) == U, "排列越界或重复"
        Wp[:, :, p*NLR:(p+1)*NLR] = W[re][:, :, :NLR]     # 该槽装 RE 的 NLR 层
        sp[:, p*NLR:(p+1)*NLR]    = srm[re][:, :NLR]
    if PACK == 1:
        assert np.array_equal(Wp, W) and np.array_equal(sp, srm)
        print("[pack] P=1 自检: 布局与非打包逐位相同")
    w16(os.path.join(GOLD_DIR,"wrm_re.bin"), Wp.real); w16(os.path.join(GOLD_DIR,"wrm_im.bin"), Wp.imag)
    w16(os.path.join(GOLD_DIR,"srm_re.bin"), sp.real);  w16(os.path.join(GOLD_DIR,"srm_im.bin"), sp.imag)
    w16(os.path.join(GOLD_DIR,"mask.bin"), used.astype(np.float32))
    w16(os.path.join(GOLD_DIR,"x_re.bin"), xg.real); w16(os.path.join(GOLD_DIR,"x_im.bin"), xg.imag)
    print(f"[pack] P={PACK} unit={U} 瓦片[{NT},{NL}]装{PACK}RE 利用率{100*PACK*NLR/NL:.0f}%")
    print(f"[io] 打包 W[{U},{NT},{NL}] s[{U},{NL}] x[{NT},{N_RE}] -> {GOLD_DIR}/")


def dump_stage_u(H, s, used, nre_dump=64, B=BLK, L=NLAY):
    """PrecodeG 的中间量对照: diag(X) 与 u = X·ŝ (ŝ = s/√diag(X))"""
    idx = np.where(used)[0][:nre_dump]
    Hu = np.transpose(H[:,:,idx],(2,0,1)); su = np.transpose(s[:,idx],(1,0))
    Hh = np.conj(np.transpose(Hu,(0,2,1))); A = Hu @ Hh
    Minv = np.zeros_like(A)
    for st in range(0, NL, B):
        e = min(st+B, NL); Minv[:,st:e,st:e] = np.linalg.inv(A[:,st:e,st:e]+DET_EPS*np.eye(e-st)[None])
    A = A + DET_EPS*np.eye(NL)[None]
    I = np.eye(NL)[None]; X = Minv.copy()
    for _ in range(L): X = X + Minv @ (I - A @ X)
    dg = np.maximum(np.real(np.diagonal(X, axis1=-2, axis2=-1)), DET_EPS)
    sh = su / np.sqrt(dg)
    u  = np.einsum('bij,bj->bi', X, sh)
    xfull = np.einsum('bmi,bi->bm', Hh, u)          # x = conj(W)·u = Hᴴ u  [nre,NT]
    w16(os.path.join(GOLD_DIR,"dgx.bin"), dg)
    w16(os.path.join(GOLD_DIR,"u_re.bin"), u.real); w16(os.path.join(GOLD_DIR,"u_im.bin"), u.imag)
    w16(os.path.join(GOLD_DIR,"xf_re.bin"), xfull.real); w16(os.path.join(GOLD_DIR,"xf_im.bin"), xfull.imag)
    # raw->used 索引表 (kernel 按 raw RE dump, golden 按 used 顺序)
    w16(os.path.join(GOLD_DIR,"dumpidx.bin"), idx.astype(np.float32))
    print(f"[io] PrecodeG 对照 dgx/u[{len(idx)},{NL}] + dumpidx -> {GOLD_DIR}/")


def dump_stage(H, s, used, A, X, nre_dump=64):
    """阶段对照: 前 nre_dump 个 RE 的 A / M⁻¹ / X (逐段验 kernel)"""
    idx = np.where(used)[0][:nre_dump]
    Hu = np.transpose(H[:,:,idx],(2,0,1)); Hh = np.conj(np.transpose(Hu,(0,2,1)))
    Ad = Hu @ Hh
    w16(os.path.join(GOLD_DIR,"A_re.bin"), Ad.real); w16(os.path.join(GOLD_DIR,"A_im.bin"), Ad.imag)
    Minv = np.zeros_like(Ad)
    for st in range(0, NL, BLK):
        e = min(st+BLK, NL); Minv[:,st:e,st:e] = np.linalg.inv(Ad[:,st:e,st:e]+DET_EPS*np.eye(e-st)[None])
    w16(os.path.join(GOLD_DIR,"minv_re.bin"), Minv.real); w16(os.path.join(GOLD_DIR,"minv_im.bin"), Minv.imag)
    Ai = np.linalg.inv(Ad+DET_EPS*np.eye(NL)[None])
    w16(os.path.join(GOLD_DIR,"xinv_re.bin"), Ai.real); w16(os.path.join(GOLD_DIR,"xinv_im.bin"), Ai.imag)
    print(f"[io] 阶段对照 A/Minv/Ainv[{nre_dump},{NL},{NL}] -> {GOLD_DIR}/")


def _sionna_zf_matrix(Hu_tf):
    """Sionna 1.2.x 无 zero_forcing_precoder, 用 rzf_precoding_matrix(alpha=0) 退化为纯 ZF"""
    import importlib, inspect
    syms = {}
    for mp in ("sionna.phy.mimo.precoding", "sionna.phy.mimo", "sionna.mimo.precoding", "sionna.mimo"):
        try: m = importlib.import_module(mp)
        except Exception: continue
        for nm in ("rzf_precoding_matrix", "zero_forcing_precoder"):
            fn = getattr(m, nm, None)
            if callable(fn) and nm not in syms: syms[nm] = (fn, mp)
    if not syms: raise ModuleNotFoundError("无 ZF/RZF precoder")
    if "rzf_precoding_matrix" in syms:
        fn, mp = syms["rzf_precoding_matrix"]
        p = inspect.signature(fn).parameters
        akw = next((w for w in ("alpha","reg","regularizer") if w in p), None)
        g = fn(Hu_tf, **({akw: 0.0} if akw else {}))
        return (g.numpy() if hasattr(g,"numpy") else np.asarray(g)), f"{mp}.rzf_precoding_matrix(alpha=0)"
    fn, mp = syms["zero_forcing_precoder"]
    import tensorflow as tf
    out = fn(tf.zeros([Hu_tf.shape[0], Hu_tf.shape[1]], Hu_tf.dtype), Hu_tf, True)
    return out[1].numpy(), f"{mp}.zero_forcing_precoder"


def crosscheck_sionna(H, used, G_gold, nsub=4096):
    """真 Sionna 对齐 (只取前 nsub 个 RE, 省内存)"""
    idx = np.where(used)[0][:nsub]
    Hu = np.transpose(H[:,:,idx],(2,0,1)).astype(np.complex64)
    try:
        import tensorflow as tf
        gs, where = _sionna_zf_matrix(tf.constant(Hu))
    except Exception as e:
        print(f"[sionna] 跳过 ({type(e).__name__}: {e}). cross-check: conda activate sionna"); return
    gg = G_gold[:len(idx)]
    scol = np.sqrt(np.sum(np.abs(gs)**2, axis=1))
    print(f"[sionna] 解析到 {where}")
    print(f"[sionna] Sionna g 列范数: mean={np.mean(scol):.4f} std={np.std(scol):.2e}")
    raw = np.max(np.abs(gs - gg))
    print(f"[sionna] G raw max|Δ|={raw:.2e}  ({'PASS 逐元素对齐' if raw<1e-3 else '⚠ 超阈, 查约定'})")
    assert raw < 1e-3, "golden 与真 Sionna 不一致!"


def main():
    assert 1 <= NLR <= NL, f"NLR={NLR} 必须在 [1,{NL}]"
    if PACK_MODE:
        assert NL % NLR == 0, f"PACK 要求 NLR={NLR} 整除 NL={NL}"
    print(f"=== precode_zf ref | NT={NT} NL={NL} NLR={NLR} PACK={PACK_MODE} P={PACK} B={BLK} L={NLAY} sc={N_SC_PAD} ===")
    rng = np.random.default_rng(SEED)
    H, s, used = gen_case(rng)
    idx = np.where(used)[0]

    xg, G_gold, A_ex, nrm_ex = exact_zf(H, s, used)
    crosscheck_sionna(H, used, G_gold)

    # padded layer 会让完整 16×16 A 必然奇异；收敛诊断只看真实 layer 子空间。
    A_active = A_ex[:, :NLR, :NLR]
    cond = np.linalg.cond(A_active)
    dd = np.abs(np.diagonal(A_active,axis1=-2,axis2=-1)).sum(-1) / (np.abs(A_active).sum((-2,-1)) + 1e-30)
    print(f"[cond   ] A=HHᴴ 条件数: median={np.median(cond):.1f} p99={np.percentile(cond,99):.1f} max={cond.max():.1f}")
    print(f"[cond   ] 对角占优度 (Σ|diag|/Σ|A|): median={np.median(dd):.3f}  (越接近1越好收敛)")

    xb, X_bri, dg_bri = bri_zf(H, s, used)
    evm = np.sqrt(np.mean(np.abs((xb-xg)[:,idx])**2)/np.mean(np.abs(xg[:,idx])**2))
    print(f"[BRI/f64] x EVM={evm*100:.2f}%  (门限5%)  {'PASS' if evm<0.05 else 'FAIL'}")

    # ── ★PACK: NLR<NL 时验打包版 (§0 已证 EVM 0.000% vs 补零) ──
    if PACK > 1:
        xpk, _, _ = bri_zf_pack(H, s, used)
        epk = np.sqrt(np.mean(np.abs((xpk-xg)[:,idx])**2)/np.mean(np.abs(xg[:,idx])**2))
        ediff = np.sqrt(np.mean(np.abs((xpk-xb)[:,idx])**2)/np.mean(np.abs(xb[:,idx])**2))
        print(f"[PACK P={PACK}] x EVM={epk*100:.2f}% vs 精确; vs 补零BRI={ediff*100:.4f}% (应~0, 证打包无损)")
        assert ediff < 0.01, f"打包与补零不一致! diff={ediff*100:.2f}%"

    # ── ★核心风险②: 归一捷径 ‖g_c‖² == diag(X) ──
    nrm_short = np.sqrt(dg_bri)                       # 捷径给的 ‖g_c‖
    rel = (np.abs(nrm_short - nrm_ex)/(nrm_ex+1e-30))[:, :NLR]   # 只看真实层 (补零列 nrm=0 无意义)
    print(f"[归一捷径] ‖g_c‖=√diag(X) vs 精确 (前{NLR}层): rel median={np.median(rel):.2e} p99={np.percentile(rel,99):.2e} max={rel.max():.2e}")
    Xeps = np.linalg.inv(A_ex + DET_EPS*np.eye(NL)[None])
    Hu = np.transpose(H[:,:,idx],(2,0,1)); Hh = np.conj(np.transpose(Hu,(0,2,1)))
    Geps = Hh @ Xeps
    nrm_eps = np.sqrt(np.sum(np.abs(Geps)**2,axis=1))
    dg_eps = np.real(np.diagonal(Xeps,axis1=-2,axis2=-1))
    rel_id = (np.abs(np.sqrt(np.maximum(dg_eps,DET_EPS))-nrm_eps)/(nrm_eps+1e-30))[:,:NLR]
    print(f"[ε加载捷径] 精确 X=(A+εI)⁻¹ 时 √diag vs 实际列范数 (前{NLR}层): max rel={rel_id.max():.2e}")

    # ── BRI 迭代层数扫描 ──
    for L in [3,4,5,6]:
        xl,_,_ = bri_zf(H, s, used, L=L)
        e = np.sqrt(np.mean(np.abs((xl-xg)[:,idx])**2)/np.mean(np.abs(xg[:,idx])**2))
        print(f"[扫描 L={L}] EVM={e*100:.2f}%  {'PASS' if e<0.05 else 'FAIL'}")

    # ── fp16 输入精度 ──
    H16 = H.real.astype(np.float16).astype(np.float64)+1j*H.imag.astype(np.float16).astype(np.float64)
    s16 = s.real.astype(np.float16).astype(np.float64)+1j*s.imag.astype(np.float16).astype(np.float64)
    x16,_,_ = bri_zf(H16, s16, used)
    e16 = np.sqrt(np.mean(np.abs((x16-xg)[:,idx])**2)/np.mean(np.abs(xg[:,idx])**2))
    print(f"[fp16-in] x EVM={e16*100:.2f}%")

    # ── sanity: ZF 干扰置零 (HG 应对角) ──
    HG = np.einsum('bij,bjc->bic', np.transpose(H[:,:,idx],(2,0,1)), G_gold)
    off = HG.copy()
    for l in range(NL): off[:,l,l] = 0
    print(f"[sanity ] H·G 离对角泄漏: max={np.max(np.abs(off)):.2e}  (≈0 => ZF 正确)")

    if PACK > 1:
        # 打包输入布局；验收 golden 始终保留精确 ZF，PACK BRI 仅用于上面的等价性检查。
        xpk, X_bri, _ = bri_zf_pack(H, s, used)
        dump_binaries_pack(H, s, used, xg)
        w16(os.path.join(GOLD_DIR,"bri_x_re.bin"), xpk.real)
        w16(os.path.join(GOLD_DIR,"bri_x_im.bin"), xpk.imag)
    else:
        dump_binaries(H, s, used, xg)
    dump_stage(H, s, used, A_ex, X_bri)
    dump_stage_u(H, s, used)
    print(f"[io] golden -> {GOLD_DIR}/  (精确基准; BRI kernel 输出对它验 EVM)")


if __name__ == "__main__":
    main()
