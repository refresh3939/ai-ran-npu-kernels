#!/usr/bin/env python3
# ============================================================================
# gen_data.py — llr_assemble golden 生成 + 参考实现
#
# 纯 int16 置换(per-slot → codeword rope-copy),无 fp64/fp16 之分,参考即精确。
# 生成两个文件到 <算子目录>/data/golden/rx/llr_assemble/:
#   llr_in.bin    int16 [N_SLOT_IN, N_STREAMS, SLOT_PAD]   源:qam per-slot 堆叠
#   codeword.bin  int16 [N_TILE,    N_STREAMS, TILE_PAD]   期望输出:descramble 输入
#
# 二者由同一条扁平 codeword 流按"顺序拼接"映射构造:
#   源按 SLOT_VALID 切、目标按 TILE_VALID 切 → kernel 必须逐位重现 codeword.bin。
#
# ⚠ 链级验证(替换本脚本即可):
#   把 flat 换成真实 qam LLR(从 Sionna chain 各 slot dump)、
#   把 codeword.bin 换成 Sionna 真实 codeword 排布。
#   若 Sionna 的 RE→codeword 映射 == 顺序拼接 → 同构,kernel 直接 PASS;
#   否则 kernel FAIL,暴露映射差异 → 改 kernel 里 c→(s,p)/c→(t,q) 两行公式。
# ============================================================================
import os
import numpy as np
from pathlib import Path

# ---- 几何(与 llr_assemble.h 一致)----
N_STREAMS  = 8
SLOT_VALID = 19152      # qam N_RE_DATA
SLOT_PAD   = 19200      # qam N_SYM_PAD
N_SLOT_IN  = 24         # ceil(CW_LEN/SLOT_VALID)
TILE_VALID = 10960      # descramble N_SYM
TILE_PAD   = 11264      # descramble N_SYM_PAD
N_TILE     = 41         # descramble N_SLOT_MAX
CW_LEN     = N_TILE * TILE_VALID     # 449360 codeword LLR/stream
LLR_CLIP   = 2560       # qam LLR_CLIP_FX(取值范围参考)


def data_root() -> Path:
    # 数据放算子目录本地。脚本在 <算子目录>/scripts/ 下,上溯一层即算子目录。
    env = os.environ.get("AIRAN_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1]


def main():
    assert N_SLOT_IN * SLOT_VALID >= CW_LEN, "input slots must cover codeword"
    assert CW_LEN % 16 == 0 and SLOT_VALID % 16 == 0 and TILE_VALID % 16 == 0

    rng = np.random.default_rng(0xA1A1)

    # 每 stream 一条扁平 codeword LLR 流(随机 ±LLR_CLIP,足以暴露任何错位)
    flat = rng.integers(-LLR_CLIP, LLR_CLIP + 1,
                        size=(N_STREAMS, CW_LEN), dtype=np.int16)

    # ---- 源布局 [N_SLOT_IN, N_STREAMS, SLOT_PAD]:按 SLOT_VALID 切,尾部/越界补 0 ----
    src = np.zeros((N_SLOT_IN, N_STREAMS, SLOT_PAD), dtype=np.int16)
    for b in range(N_STREAMS):
        for s in range(N_SLOT_IN):
            c0 = s * SLOT_VALID
            if c0 >= CW_LEN:
                break
            n = min(SLOT_VALID, CW_LEN - c0)
            src[s, b, :n] = flat[b, c0:c0 + n]

    # ---- 目标布局 [N_TILE, N_STREAMS, TILE_PAD]:按 TILE_VALID 切,pad 区保持 0 ----
    cw = np.zeros((N_TILE, N_STREAMS, TILE_PAD), dtype=np.int16)
    for b in range(N_STREAMS):
        for t in range(N_TILE):
            c0 = t * TILE_VALID
            cw[t, b, :TILE_VALID] = flat[b, c0:c0 + TILE_VALID]

    out = data_root() / "data" / "golden" / "rx" / "llr_assemble"
    out.mkdir(parents=True, exist_ok=True)
    src.tofile(out / "llr_in.bin")
    cw.tofile(out / "codeword.bin")
    print(f"[gen] {out}/llr_in.bin    {src.shape} int16  {src.nbytes} B")
    print(f"[gen] {out}/codeword.bin  {cw.shape} int16  {cw.nbytes} B")

    # ---- 自检:顺序拼接同构(per-stream 有效区拼平应相等)----
    for b in range(N_STREAMS):
        a = src[:, b, :SLOT_VALID].reshape(-1)[:CW_LEN]
        d = cw[:, b, :TILE_VALID].reshape(-1)
        assert np.array_equal(a, d), f"stream {b}: sequential-concat self-check failed"
    print("[gen] self-check OK: per-stream sequential concat consistent "
          f"({N_STREAMS} streams x {CW_LEN} = {N_STREAMS*CW_LEN} LLR)")


if __name__ == "__main__":
    main()