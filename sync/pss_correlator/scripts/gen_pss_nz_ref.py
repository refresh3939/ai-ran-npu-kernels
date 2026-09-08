#!/usr/bin/env python3
# 生成 NZ 布局 pss_ref.bin (kernel NZ 改造后用)
# 布局: [re_nz(K*N=4096 half) | im_nz(4096 half)] = 8192 half = 16384 B
# nz[k*16+l] = pss[l][k] (l=0,1,2; l=3..15=0), 反量化 /Q_SCALE(8192)
# 用法: python3 gen_pss_nz_ref.py <旧pss_ref.bin路径(int16 [3,256,2])> <输出路径>
import numpy as np, sys
src = sys.argv[1] if len(sys.argv)>1 else 'pss_ref.bin'
dst = sys.argv[2] if len(sys.argv)>2 else 'pss_ref_nz.bin'
K,N,NPSS,QS = 256,16,3,8192.0
r = np.fromfile(src, dtype=np.int16).reshape(NPSS,K,2)
re = r[:,:,0].astype(np.float32)/QS
im = r[:,:,1].astype(np.float32)/QS
nz_re = np.zeros((K,N), dtype=np.float16)
nz_im = np.zeros((K,N), dtype=np.float16)
for l in range(NPSS):
    nz_re[:,l] = re[l,:].astype(np.float16)
    nz_im[:,l] = im[l,:].astype(np.float16)
out = np.concatenate([nz_re.flatten(), nz_im.flatten()])  # 8192 half
out.tofile(dst)
print(f'NZ pss_ref: {out.nbytes} B (8192 half) -> {dst}')
print(f'  nz_re[k=1] l012 = {nz_re[1,0]:.3f} {nz_re[1,1]:.3f} {nz_re[1,2]:.3f}')
