#!/usr/bin/env python3
# win_slice 参考实现 —— device 端从常驻 capture 切一个 slot,并推进 rd_ptr。
#
# 语义(全部 device 侧,host 不介入):
#   1) rd_ptr += N_SAMP_SLOT + round(delta_T_prev)     # 上一 slot timing 写的漂移
#   2) slot_iq[0 : 2*N_SAMP_SLOT] = capture[2*rd_ptr : 2*rd_ptr + 2*N_SAMP_SLOT]
#   3) 写回 rd_ptr(供下个 slot)
#
# 数据:capture/slot_iq 都是 int16 IQ interleaved。rd_ptr 是 complex-sample 偏移(int32)。
# delta_T 是 fp32(timing_tracker 输出),round 到整数样本。
#
# 边界:rd_ptr 初值 = mu_t_data - N_SAMP_SLOT(首个 win_slice 推进后正好落在 mu_t_data)。
#       caller 须保证 rd_ptr + N_SAMP_SLOT <= CAPTURE_SAMPLES(不越界)。

import numpy as np, os, struct

N_SAMP_SLOT     = 30720                    # complex / slot
CAPTURE_SAMPLES = 1_228_800                # complex in capture (~40 slot)
DATA_DIR = os.environ.get("AIRAN_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

def win_slice_ref(capture_i16, rd_ptr_in, delta_T_prev):
    """capture_i16: int16[2*CAPTURE_SAMPLES] interleaved. returns (slot_iq_i16[2*N_SAMP_SLOT], rd_ptr_out)."""
    rd_ptr_out = int(rd_ptr_in) + N_SAMP_SLOT + int(np.round(delta_T_prev))
    b = 2 * rd_ptr_out
    slot_iq = capture_i16[b : b + 2 * N_SAMP_SLOT].copy()
    return slot_iq, rd_ptr_out

def _w_i16(path, a): a.astype('<i2').tofile(path)
def _w_i32(path, v): open(path,'wb').write(struct.pack('<i', int(v)))
def _w_f32(path, v): open(path,'wb').write(struct.pack('<f', float(v)))

def gen_case(cid, seed, rd_ptr_in, delta_T_prev):
    rng = np.random.default_rng(seed)
    cap = rng.integers(-2000, 2000, size=2*CAPTURE_SAMPLES, dtype=np.int16)
    slot_iq, rd_out = win_slice_ref(cap, rd_ptr_in, delta_T_prev)
    d = os.path.join(DATA_DIR, "golden", f"case_{cid}")
    os.makedirs(d, exist_ok=True)
    _w_i16(os.path.join(d, "capture.bin"),   cap)
    _w_i32(os.path.join(d, "rd_ptr_in.bin"), rd_ptr_in)
    _w_f32(os.path.join(d, "delta_T.bin"),   delta_T_prev)
    _w_i16(os.path.join(d, "slot_iq_gold.bin"), slot_iq)
    _w_i32(os.path.join(d, "rd_ptr_out_gold.bin"), rd_out)
    print(f"[case {cid}] rd_in={rd_ptr_in} dt={delta_T_prev:+.2f} -> rd_out={rd_out}  slot_iq[0:4]={slot_iq[:4]}")

if __name__ == "__main__":
    # 5 个确定性用例:正/负/零漂移、不同起点
    gen_case(0, 1, rd_ptr_in=0,                       delta_T_prev=0.0)
    gen_case(1, 2, rd_ptr_in=N_SAMP_SLOT,             delta_T_prev=+2.0)
    gen_case(2, 3, rd_ptr_in=5*N_SAMP_SLOT,           delta_T_prev=-3.0)
    gen_case(3, 4, rd_ptr_in=10*N_SAMP_SLOT,          delta_T_prev=+0.49)   # round->0
    gen_case(4, 5, rd_ptr_in=10*N_SAMP_SLOT,          delta_T_prev=-0.51)   # round->-1
    print("done; golden written under", os.path.join(DATA_DIR, "golden"))
