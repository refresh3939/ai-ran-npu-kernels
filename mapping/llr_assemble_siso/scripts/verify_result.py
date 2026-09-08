#!/usr/bin/env python3
# ============================================================================
# verify_result.py — 对比 kernel 输出 vs golden codeword
#
# 读 <算子目录>/data/golden/rx/llr_assemble/codeword.bin       (期望)
#   <算子目录>/data/ascend_output/rx/llr_assemble/codeword.bin (kernel 输出)
# 校验:valid 区 [:, :, 0..TILE_VALID) 逐位相等;pad 区 [:, :, TILE_VALID..TILE_PAD) 全 0。
# ============================================================================
import os
import sys
import numpy as np
from pathlib import Path

N_STREAMS  = 8
TILE_VALID = 10960
TILE_PAD   = 11264
N_TILE     = 41


def data_root() -> Path:
    env = os.environ.get("AIRAN_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1]   # scripts/ → 算子目录(本地)


def main():
    root   = data_root()
    gold_p = root / "data" / "golden"        / "rx" / "llr_assemble" / "codeword.bin"
    out_p  = root / "data" / "ascend_output" / "rx" / "llr_assemble" / "codeword.bin"

    print("[run.sh] verifying ...")
    print(f"Golden : {gold_p}")
    print(f"Output : {out_p}")
    if not gold_p.exists(): print(f"[err] golden 不存在: {gold_p}"); sys.exit(1)
    if not out_p.exists():  print(f"[err] 输出不存在: {out_p}");   sys.exit(1)

    shape = (N_TILE, N_STREAMS, TILE_PAD)
    gold = np.fromfile(gold_p, np.int16)
    out  = np.fromfile(out_p,  np.int16)
    if gold.size != np.prod(shape) or out.size != np.prod(shape):
        print(f"[err] size 不符: gold={gold.size} out={out.size} expect={int(np.prod(shape))}")
        sys.exit(1)
    gold = gold.reshape(shape)
    out  = out.reshape(shape)

    gv = gold[:, :, :TILE_VALID].astype(np.int32)
    ov = out [:, :, :TILE_VALID].astype(np.int32)
    d  = np.abs(gv - ov)
    diff        = int(np.count_nonzero(d))
    max_err     = int(d.max()) if d.size else 0
    pad_nonzero = int(np.count_nonzero(out[:, :, TILE_VALID:]))

    print(f"Grid   : [{N_TILE}, {N_STREAMS}, {TILE_PAD}]  valid/tile={TILE_VALID} pad/tile={TILE_PAD - TILE_VALID}")
    print(f"  valid diff vs golden: {diff}  max_err={max_err}")
    if diff:
        t, b, q = np.unravel_index(int(d.argmax()), d.shape)
        print(f"  worst: tile={t} stream={b} q={q}  kernel={out[t, b, q]} gold={gold[t, b, q]}")
    print(f"  pad [{TILE_VALID},{TILE_PAD}) nonzero: {pad_nonzero} (should be 0)")

    ok = (diff == 0 and pad_nonzero == 0)
    print("PASS (bit-exact codeword, zero pad)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
