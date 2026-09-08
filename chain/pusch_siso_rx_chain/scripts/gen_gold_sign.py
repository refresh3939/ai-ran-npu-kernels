# 生成 descramble 的 compact canonical-NR gold_sign.bin = (1-2c)。
# Gold 只跨有效 RE 连续推进，19200 allocation 的尾部 padding 不消耗序列。
import numpy as np, os, sys

N_SYM=19152; N_SYM_PAD=19200; N_STREAMS=8; N_SLOT=23; GOLD_LEN=1600
N_ID_cell=1; n_RNTI=12345; q=0   # 必须与 TX scramble_ref __main__ 一致

def gold_seq_init(c_init, length):
    Nc=GOLD_LEN; total=length+Nc
    x1=np.zeros(total+31,dtype=np.uint8); x1[0]=1
    x2=np.zeros(total+31,dtype=np.uint8)
    for i in range(31): x2[i]=(c_init>>i)&1
    for n in range(total):
        x1[n+31]=(x1[n+3]+x1[n])&1
        x2[n+31]=(x2[n+3]+x2[n+2]+x2[n+1]+x2[n])&1
    return (x1[Nc:Nc+length]^x2[Nc:Nc+length]).astype(np.uint8)

c_init=((n_RNTI&0xFFFF)<<15)|((q&1)<<14)|(N_ID_cell&0x3FF)
c=gold_seq_init(c_init, N_SLOT*N_SYM*N_STREAMS)
c_re=c.reshape(N_SLOT,N_SYM,N_STREAMS)               # [slot,compact_re,NR bit b]
sign=np.ones((N_SLOT,N_STREAMS,N_SYM_PAD),dtype=np.int16)
sign[:,:,:N_SYM]=(1-2*c_re.astype(np.int16)).transpose(0,2,1)
print(f"sign shape={sign.shape} uniq={np.unique(sign)} bytes={sign.nbytes}")
print(f"expect [23,8,19200]={23*8*19200} elems, {23*8*19200*2} B")
outp=sys.argv[1] if len(sys.argv)>1 else "gold_sign.bin"
sign.tofile(outp); print(f"wrote {outp}")
