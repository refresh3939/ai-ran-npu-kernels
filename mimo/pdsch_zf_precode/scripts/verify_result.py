#!/usr/bin/env python3
"""precode_zf kernel 输出 vs 精确 ZF golden, EVM 门控 (BRI 近似)"""
import os, numpy as np
NT=int(os.environ.get("NT",64)); NL=int(os.environ.get("NL",16))
NLR=int(os.environ.get("NLR",NL))
PACK_MODE=int(os.environ.get("PACK", "1" if NLR < NL else "0"))
P=(NL//NLR) if PACK_MODE else 1
N_SYMBOL=14; N_SC_PAD=int(os.environ.get('AIRAN_SCPAD',1664)); N_RE=N_SYMBOL*N_SC_PAD
CASE=(f"case_0_m{NT}_k{NL}_sc{N_SC_PAD}"
      + (f"_r{NLR}" if NLR != NL else "") + ("_pk" if P > 1 else ""))
ROOT=os.environ.get("AIRAN_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
GOLD=os.path.join(ROOT,"data","golden",CASE); ASC=os.path.join(ROOT,"data","ascend_output",CASE)
TOL_EVM=0.05
def load(d,n,expected=None):
    p=os.path.join(d,n+".bin")
    if not os.path.exists(p): print(f"[FAIL] 缺文件 {p}"); raise SystemExit(1)
    a=np.fromfile(p,np.float16)
    if expected is not None and a.size != expected:
        print(f"[FAIL] 文件尺寸不符 {p}: {a.size} half, 期望 {expected}"); raise SystemExit(1)
    return a.astype(np.float64)
for n in ("x_re","x_im"):
    gp=os.path.join(GOLD,n+".bin"); ap=os.path.join(ASC,n+".bin")
    if os.path.exists(gp) and os.path.exists(ap) and os.path.getmtime(ap) < os.path.getmtime(gp):
        print(f"[FAIL] 输出早于 golden，疑似陈旧文件: {ap}"); raise SystemExit(1)
print(f"[case] {CASE} NLR={NLR} PACK={PACK_MODE} P={P}")
mask=load(GOLD,"mask",N_RE).reshape(N_RE); idx=np.where(mask>0.5)[0]
gx=(load(GOLD,"x_re",NT*N_RE).reshape(NT,N_RE)+1j*load(GOLD,"x_im",NT*N_RE).reshape(NT,N_RE))[:,idx]
ax=(load(ASC,"x_re",NT*N_RE).reshape(NT,N_RE)+1j*load(ASC,"x_im",NT*N_RE).reshape(NT,N_RE))[:,idx]
evm=np.sqrt(np.mean(np.abs(ax-gx)**2)/np.mean(np.abs(gx)**2))
ok = evm<TOL_EVM
print(f"[x 发射波束] EVM={evm*100:.2f}%  (门限<{TOL_EVM*100:.0f}%)  {'PASS' if ok else 'FAIL'}")
# 逐天线 EVM (诊断: 定位是否某个 16-天线块整体错 -> 多半是转置/散射 index)
pa=np.sqrt(np.mean(np.abs(ax-gx)**2,1)/(np.mean(np.abs(gx)**2,1)+1e-30))
print(f"[逐天线   ] EVM min={pa.min()*100:.2f}% max={pa.max()*100:.2f}% "
      f"最差天线={int(np.argmax(pa))} (若某 16 天线块整体差 -> 查 Transpose/散射 index)")
# 发射功率 sanity: 列归一 => E|x|^2 应 ≈ E|s|^2 量级
print(f"[功率     ] golden E|x|²={np.mean(np.abs(gx)**2):.4f}  ascend E|x|²={np.mean(np.abs(ax)**2):.4f}")
print(f"=== RESULT: {'PASS' if ok else 'FAIL'} ===")
raise SystemExit(0 if ok else 1)
