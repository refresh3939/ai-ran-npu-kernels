#!/usr/bin/env python3
"""precode_zf 分级验证: kernel 中间量 vs ref golden.
   DUMPSEL: 1=A  2=M⁻¹  3=X  4=u  5=xf(末段 x, 未转置)
   kernel 按 raw-RE dump 到 gX_re/gX_im 前若干 half; golden 按 used-RE 顺序,
   用 dumpidx.bin 对齐, 只比 raw<DUMP_NRE 的 RE.
"""
import os, sys, numpy as np
NT=int(os.environ.get("NT",64)); NL=int(os.environ.get("NL",16))
NLR=int(os.environ.get("NLR",NL)); PACK_MODE=int(os.environ.get("PACK", "1" if NLR < NL else "0"))
P=(NL//NLR) if PACK_MODE else 1
SEL=int(os.environ.get("DUMPSEL","1")); NRE_D=int(os.environ.get("DUMP_NRE","64"))
N_SYMBOL=14; N_SC_PAD=int(os.environ.get('AIRAN_SCPAD',1664)); N_RE=N_SYMBOL*N_SC_PAD
CASE=(f"case_0_m{NT}_k{NL}_sc{N_SC_PAD}"
      + (f"_r{NLR}" if NLR != NL else "") + ("_pk" if P > 1 else ""))
ROOT=os.environ.get("AIRAN_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
GOLD=os.path.join(ROOT,"data","golden",CASE); ASC=os.path.join(ROOT,"data","ascend_output",CASE)

SPEC={1:("A_re","A_im","A = H Hᴴ",                        NL*NL, 0.02),
      2:("minv_re","minv_im","M⁻¹ (块对角逆, Neumann×3 近似)", NL*NL, 0.15),
      3:("xinv_re","xinv_im","X ≈ A⁻¹ (BRI)",              NL*NL, 0.10),
      4:("u_re","u_im","u = X·ŝ (PrecodeG 中段)",           NL,    0.10),
      5:("xf_re","xf_im","x = conj(W)·u (末段, 未转置)",     NT,    0.10)}
if SEL not in SPEC: print(f"[FAIL] DUMPSEL={SEL} 无对照"); sys.exit(1)
gr_n,gi_n,tag,ELEM,TOL = SPEC[SEL]

def load(d,n):
    p=os.path.join(d,n+".bin")
    if not os.path.exists(p): print(f"[FAIL] 缺文件 {p}"); sys.exit(1)
    return np.fromfile(p,np.float16).astype(np.float64)

idx = load(GOLD,"dumpidx").astype(int)
sel = idx < NRE_D
raw = idx[sel]
n   = len(raw)

g = (load(GOLD,gr_n)+1j*load(GOLD,gi_n)).reshape(-1,ELEM)[:len(idx)][sel]
a = (load(ASC,"x_re")[:NRE_D*ELEM]+1j*load(ASC,"x_im")[:NRE_D*ELEM]).reshape(NRE_D,ELEM)[raw]

rel=np.linalg.norm(a-g,axis=1)/(np.linalg.norm(g,axis=1)+1e-30)
print(f"=== DUMPSEL={SEL}: {tag} ===")
print(f"[对齐   ] 比对 {n} 个 used-RE (raw idx {raw.min()}..{raw.max()})")
print(f"[量级   ] golden max|·|={np.abs(g).max():.4f}  ascend max|·|={np.abs(a).max():.4f}")
print(f"[逐RE   ] 相对误差 median={np.median(rel):.2e} p99={np.percentile(rel,99):.2e} max={rel.max():.2e}")
print(f"[逐元素 ] abs 误差 median={np.median(np.abs(a-g)):.2e} max={np.abs(a-g).max():.2e}")

num=np.vdot(g.ravel(),a.ravel())
corr=abs(num)/(np.linalg.norm(g)*np.linalg.norm(a)+1e-30)
print(f"[相关   ] |<g,a>|/(|g||a|)={corr:.4f}  幅度比={np.linalg.norm(a)/(np.linalg.norm(g)+1e-30):.4f}  "
      f"相位={np.angle(num)*180/np.pi:.2f}°")

if ELEM==NL*NL:
    A3=a.reshape(n,NL,NL); G3=g.reshape(n,NL,NL); di=np.arange(NL); mo=~np.eye(NL,dtype=bool)
    print(f"[对角   ] err median={np.median(np.abs(A3[:,di,di]-G3[:,di,di])):.2e}   "
          f"[非对角] err median={np.median(np.abs((A3-G3)[:,mo])):.2e}")
    ec=np.linalg.norm((A3-np.conj(G3)).reshape(n,-1),axis=1)/(np.linalg.norm(G3.reshape(n,-1),axis=1)+1e-30)
    et=np.linalg.norm((A3-np.transpose(G3,(0,2,1))).reshape(n,-1),axis=1)/(np.linalg.norm(G3.reshape(n,-1),axis=1)+1e-30)
    if np.median(ec)<np.median(rel)*0.5: print(f"[共轭   ] ★ vs conj(g) median={np.median(ec):.2e} 更小 -> 共轭方向反了")
    if np.median(et)<np.median(rel)*0.5: print(f"[转置   ] ★ vs gᵀ median={np.median(et):.2e} 更小 -> 行列反了")

ok = np.median(rel) < TOL
print(f"=== RESULT: {'PASS' if ok else 'FAIL'} (门限 median<{TOL}) ===")
sys.exit(0 if ok else 1)
