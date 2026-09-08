"""
ssb_fft reference — SSB PBCH 符号 256-FFT + 抽 144 DMRS RE + de-rotate.
标准化更新 (sim M3 验证后): CP=36, NU-dependent gather (4 套), GATE-3 pbch 交叉校验锁。
"""
import os, sys
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_KERNEL_DIR = _SCRIPT_DIR.parent
DEFAULT_WEIGHTS_DIR = _KERNEL_DIR / "weights"
DEFAULT_GOLDEN_DIR  = _KERNEL_DIR / "data" / "golden"

SSB_N_FFT   = 256
P = Q = 16
assert P * Q == SSB_N_FFT
N_PBCH_SYM  = 3
SSB_CP      = 36                  # ★ 标准 normal CP @7.68M (144÷4); 旧值 18 误算
SSB_STRIDE  = SSB_CP + SSB_N_FFT  # 292
N_DMRS_RE   = 144
Q_SCALE     = 256
NID         = 0
NU          = NID % 4
N_SSB_SC    = 240
L2_PBCH_SC  = list(range(0, 48)) + list(range(192, 240))

def ssb_sc_to_bin(k_ssb: int) -> int:
    return (k_ssb - (N_SSB_SC // 2)) % SSB_N_FFT

def build_dmrs_table(nu: int = NU):
    l1 = [k for k in range(N_SSB_SC) if k % 4 == nu]
    l2 = [k for k in L2_PBCH_SC       if k % 4 == nu]
    l3 = l1[:]
    assert len(l1) == 60 and len(l2) == 24 and len(l3) == 60
    sym_of = [0]*60 + [1]*24 + [2]*60
    sc_of  = l1 + l2 + l3
    bin_of = [ssb_sc_to_bin(k) for k in sc_of]
    gather_idx = [sym_of[k]*SSB_N_FFT + 16*(bin_of[k]%16) + (bin_of[k]//16)
                  for k in range(N_DMRS_RE)]
    return (np.array(sym_of,np.int32), np.array(sc_of,np.int32),
            np.array(bin_of,np.int32), np.array(gather_idx,np.int32))

def dft_matrix_unitary(n):
    k=np.arange(n)[:,None]; i=np.arange(n)[None,:]
    return (np.exp(-2j*np.pi*k*i/n)/np.sqrt(n)).astype(np.complex64)
def twiddle_pq(p,q,n):
    k1=np.arange(p)[:,None]; n2=np.arange(q)[None,:]
    return np.exp(-2j*np.pi*k1*n2/n).astype(np.complex64)
W16  = dft_matrix_unitary(P)
T_PQ = twiddle_pq(P,Q,SSB_N_FFT)

def fft256_complex(x):
    sh=x.shape[:-1]; x_rs=x.reshape(*sh,P,Q)
    X1=np.einsum("ab,...bc->...ac",W16,x_rs); X1_tw=X1*T_PQ
    X2=np.einsum("...ab,cb->...ac",X1_tw,W16)
    return np.moveaxis(X2,-2,-1).reshape(*sh,SSB_N_FFT)
def _cmm(Ar,Ai,Br,Bi,ax):
    return (np.einsum(ax,Ar,Br)-np.einsum(ax,Ai,Bi),
            np.einsum(ax,Ar,Bi)+np.einsum(ax,Ai,Br))
def fft256_4real(x):
    sh=x.shape[:-1]
    Xr=x.real.astype(np.float32).reshape(*sh,P,Q); Xi=x.imag.astype(np.float32).reshape(*sh,P,Q)
    Wr,Wi=W16.real.astype(np.float32),W16.imag.astype(np.float32)
    Tr,Ti=T_PQ.real.astype(np.float32),T_PQ.imag.astype(np.float32)
    X1r,X1i=_cmm(Wr,Wi,Xr,Xi,"ab,...bc->...ac")
    Twr=X1r*Tr-X1i*Ti; Twi=X1r*Ti+X1i*Tr
    X2r,X2i=_cmm(Twr,Twi,Wr.T,Wi.T,"...ab,bc->...ac")
    return (X2r+1j*X2i).astype(np.complex64)

def cp_remove(x_time):
    assert x_time.shape[-1]==N_PBCH_SYM*SSB_STRIDE
    out=np.empty(x_time.shape[:-1]+(N_PBCH_SYM,SSB_N_FFT),np.complex64)
    for s in range(N_PBCH_SYM):
        base=s*SSB_STRIDE+SSB_CP; out[...,s,:]=x_time[...,base:base+SSB_N_FFT]
    return out
def ssb_fft(x_time,gather_idx,derot,return_mid=False):
    x_cp=cp_remove(x_time); X2=fft256_4real(x_cp); R=X2.reshape(-1)[gather_idx]*derot
    return (R,{"cp":x_cp,"X2":X2}) if return_mid else R

def synth_input(rng,sym_of,bin_of,dmrs_vals=None):
    qpsk=lambda n:((rng.integers(0,2,n)*2-1)+1j*(rng.integers(0,2,n)*2-1)).astype(np.complex64)/np.sqrt(2)
    grids=np.zeros((N_PBCH_SYM,SSB_N_FFT),np.complex64)
    occ={0:list(range(N_SSB_SC)),1:L2_PBCH_SC,2:list(range(N_SSB_SC))}
    for s in range(N_PBCH_SYM):
        bins=[ssb_sc_to_bin(k) for k in occ[s]]; grids[s,bins]=qpsk(len(bins))
    if dmrs_vals is not None:
        for j in range(N_DMRS_RE): grids[sym_of[j],bin_of[j]]=dmrs_vals[j]
    dmrs_placed=grids.reshape(-1)[[sym_of[k]*SSB_N_FFT+bin_of[k] for k in range(N_DMRS_RE)]]
    x_time=np.empty(N_PBCH_SYM*SSB_STRIDE,np.complex64)
    for s in range(N_PBCH_SYM):
        body=np.fft.ifft(grids[s])*np.sqrt(SSB_N_FFT)
        x_time[s*SSB_STRIDE:(s+1)*SSB_STRIDE]=np.concatenate([body[-SSB_CP:],body]).astype(np.complex64)
    return x_time, dmrs_placed.astype(np.complex64)

def complex_to_q8_8_int16(z):
    re=np.round(z.real*Q_SCALE).astype(np.int16); im=np.round(z.imag*Q_SCALE).astype(np.int16)
    return np.stack([re,im],-1).reshape(*z.shape[:-1],2*z.shape[-1])

def verify_paths(atol=1e-4):
    rng=np.random.default_rng(0)
    x=(rng.standard_normal((3,SSB_N_FFT))+1j*rng.standard_normal((3,SSB_N_FFT))).astype(np.complex64)
    y_c=fft256_complex(x)
    y_np=(np.fft.fft(x.astype(np.complex128),axis=-1)/np.sqrt(SSB_N_FFT)).astype(np.complex64)
    e1=np.abs(y_c-y_np).max(); print(f"[complex vs unitary np.fft] max|err|={e1:.3e}"); assert e1<atol
    X2=fft256_4real(x); y_4=np.moveaxis(X2,-2,-1).reshape(3,SSB_N_FFT)
    e2=np.abs(y_c-y_4).max(); print(f"[4real vs complex] max|err|={e2:.3e}"); assert e2<atol

def gate3_lock():
    try:
        sys.path.insert(0,str(_KERNEL_DIR.parent/"pbch_dmrs_correlator"/"scripts"))
        import pbch_dmrs_correlator_ref as B
    except Exception as e:
        print(f"[gate3] 跳过 (pbch ref 不可用: {e})"); return None
    tbl=B.build_dmrs_ref_table(); rng=np.random.default_rng(999); npass=0; cases=[]
    for nu in range(4):
        for pcid,i_ssb in [(nu,0),(nu+500,5),(nu+1000,7)]:
            if pcid>=1008 or pcid%4!=nu: continue
            sym_of,sc_of,bin_of,gidx=build_dmrs_table(nu)
            dmrs=B.pbch_dmrs_sequence(pcid,i_ssb)
            x_time,_=synth_input(rng,sym_of,bin_of,dmrs_vals=dmrs)
            R=ssb_fft(x_time,gidx,np.ones(N_DMRS_RE,np.complex64))
            i_det,pk,sec=B.pbch_dmrs_correlator_ref(R.astype(np.complex64),tbl,pcid,8)
            ok=(i_det==i_ssb); npass+=int(ok); cases.append((nu,pcid,i_ssb,i_det,pk/max(sec,1e-9),ok))
    print("\n[gate3] DMRS k 序 vs pbch D 表 交叉校验:")
    for nu,pcid,i_ssb,i_det,ratio,ok in cases:
        print(f"  NU={nu} pcid={pcid:4d} i_ssb={i_ssb} → det={i_det} ratio={ratio:5.1f} {'PASS' if ok else 'FAIL'}")
    print(f"[gate3] {npass}/{len(cases)} PASS ({'k 序锁定' if npass==len(cases) else 'k 序不匹配!'})")
    return npass==len(cases)

def dump(out_root=None,weights_root=None):
    out_root=out_root or str(DEFAULT_GOLDEN_DIR); weights_root=weights_root or str(DEFAULT_WEIGHTS_DIR)
    os.makedirs(out_root,exist_ok=True); os.makedirs(weights_root,exist_ok=True)
    derot=np.ones(N_DMRS_RE,np.complex64)
    W16.real.astype(np.float16).tofile(f"{weights_root}/w16_re.bin")
    W16.imag.astype(np.float16).tofile(f"{weights_root}/w16_im.bin")
    W16.real.T.astype(np.float16).tofile(f"{weights_root}/w16_re_T.bin")
    W16.imag.T.astype(np.float16).tofile(f"{weights_root}/w16_im_T.bin")
    T_PQ.real.astype(np.float16).tofile(f"{weights_root}/twiddle_pq_re.bin")
    T_PQ.imag.astype(np.float16).tofile(f"{weights_root}/twiddle_pq_im.bin")
    derot.real.astype(np.float16).tofile(f"{weights_root}/derot_re.bin")
    derot.imag.astype(np.float16).tofile(f"{weights_root}/derot_im.bin")
    for nu in range(4):
        _,_,_,gidx_nu=build_dmrs_table(nu)
        gidx_nu.astype(np.int32).tofile(f"{weights_root}/gather_idx_nu{nu}.bin")
    sym_of,sc_of,bin_of,gather_idx=build_dmrs_table(0)
    gather_idx.astype(np.int32).tofile(f"{weights_root}/gather_idx.bin")
    rng=np.random.default_rng(seed=12345)
    x_time,dmrs_placed=synth_input(rng,sym_of,bin_of)
    R,mid=ssb_fft(x_time,gather_idx,derot,return_mid=True)
    complex_to_q8_8_int16(x_time).tofile(f"{out_root}/input.bin")
    R.real.astype(np.float16).tofile(f"{out_root}/R_re.bin")
    R.imag.astype(np.float16).tofile(f"{out_root}/R_im.bin")
    mid["X2"].astype(np.complex64).tofile(f"{out_root}/X2_dft16.bin")
    e=np.abs(R-dmrs_placed).max(); print(f"\n[round-trip] max|R-placed_DMRS|={e:.3e}"); assert e<1e-3
    print(f"[dump] weights → {weights_root}/")
    for n in ["gather_idx","gather_idx_nu0","gather_idx_nu1","gather_idx_nu2","gather_idx_nu3"]:
        print(f"  {n+'.bin':<20} {os.path.getsize(f'{weights_root}/{n}.bin'):>6} B")
    print(f"[dump] golden → {out_root}/")
    for n in ["input","R_re","R_im"]:
        print(f"  {n+'.bin':<20} {os.path.getsize(f'{out_root}/{n}.bin'):>6} B")
    print(f"  SSB_CP={SSB_CP} STRIDE={SSB_STRIDE} input={N_PBCH_SYM*SSB_STRIDE}复 = {N_PBCH_SYM*SSB_STRIDE*2} int16")

if __name__=="__main__":
    print("="*64); print(" ssb_fft 数据生成 (CP=36 + NU gather + GATE-3 锁)"); print("="*64)
    verify_paths(); dump(); gate3_lock()
    print("="*64+"\n 完成"); 
