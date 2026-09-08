#!/usr/bin/env python3
# ============================================================================
# verify_result.py — 校验 ascend kernel 输出的 metric 立方
#
# 读取 data/ascend_output/{case_dir}.bin (kernel 写的 GM dump),
# 还原成 [N_FREQ_HYP=21, N_PSS=3, N_TIME_OFFSETS=401] 逻辑形状,
# 做 argmax,与 truth 比对。
#
# GM 布局 (与 kernel 一致):
#   shape [N_HYP_BATCHES=3, N_TIME_OFFSETS=401, OUT_FP32_PER_BLOCK=32]
#   idx:  gm[b, t, h_in_b * 4 + p]  → logical[h_in_b + b*8, p, t]
# ============================================================================
import os
import sys
from pathlib import Path
import numpy as np

# 与 .h 锁定一致
N_FFT             = 2048
N_FREQ_HYP        = 21
FREQ_HYP_MIN_HZ   = -50000.0
FREQ_HYP_STEP_HZ  =   5000.0
N_PSS             = 3
TIME_SEARCH_HALF  = 200
N_TIME_OFFSETS    = 401
N_HYP_BATCHES     = 4
HYP_BATCH         = 6
OUT_HYP_PAD       = 8
OUT_PSS_PAD       = 4
OUT_FP32_PER_BLOCK = OUT_HYP_PAD * OUT_PSS_PAD          # = 32
OUT_FP32_PER_BATCH = N_TIME_OFFSETS * OUT_FP32_PER_BLOCK  # = 12832
OUT_FP32_TOTAL    = N_HYP_BATCHES * OUT_FP32_PER_BATCH    # = 51328

# Pass criteria
CFO_TOL_HZ        = 2500.0
TIME_TOL_SAMPLES  = 2

TEST_CASES = [
    ("case_0_pss0_cfo+0_t+0",         0.0,    0,    0),
    ("case_1_pss0_cfo+500_t+0",      +500.0,  0,    0),
    ("case_2_pss1_cfo+25000_t+50",  +25000.0, 1,  +50),
    ("case_3_pss2_cfo-45000_t-80",  -45000.0, 2,  -80),
    ("case_4_pss0_cfo+12500_t+130", +12500.0, 0, +130),
]


def gm_to_logical(gm_flat):
    """[N_HYP_BATCHES, N_TIME_OFFSETS, OUT_FP32_PER_BLOCK] → [N_FREQ_HYP, N_PSS, N_TIME_OFFSETS]"""
    gm = gm_flat.reshape(N_HYP_BATCHES, N_TIME_OFFSETS, OUT_FP32_PER_BLOCK)
    logical = np.zeros((N_FREQ_HYP, N_PSS, N_TIME_OFFSETS), dtype=np.float32)
    for h in range(N_FREQ_HYP):
        b = h // HYP_BATCH
        h_in_b = h % HYP_BATCH
        for p in range(N_PSS):
            logical[h, p, :] = gm[b, :, h_in_b * OUT_PSS_PAD + p]
    return logical


def main():
    env = os.environ.get("AIRAN_DATA_DIR")
    if env:
        out_root = Path(env) / "data" / "ascend_output"
    else:
        here = Path(__file__).resolve()
        out_root = here.parents[4] / "data" / "ascend_output"

    print(f"[verify] reading from {out_root}")
    print(f"[verify] expecting {OUT_FP32_TOTAL} fp32 = {OUT_FP32_TOTAL * 4} B per case")
    print()
    print(f"{'case':<32} {'inj(cfo,pss,t)':>20} {'est(cfo,pss,t)':>20} {'peak':>10}  {'verdict':>9}")
    print("─" * 110)

    n_pass = 0
    for case_dir, inj_cfo, inj_pss, inj_t in TEST_CASES:
        fpath = out_root / f"{case_dir}.bin"
        if not fpath.exists():
            print(f"{case_dir:<32} MISSING ascend output ({fpath})")
            continue

        raw = np.fromfile(fpath, dtype=np.float32)
        if raw.size != OUT_FP32_TOTAL:
            print(f"{case_dir:<32} BAD SIZE: got {raw.size}, expected {OUT_FP32_TOTAL}")
            continue

        logical = gm_to_logical(raw)

        # argmax over [hyp, pss, t]
        idx = np.unravel_index(logical.argmax(), logical.shape)
        h_best, p_best, t_best = int(idx[0]), int(idx[1]), int(idx[2])
        peak = float(logical[h_best, p_best, t_best])
        cfo_best  = FREQ_HYP_MIN_HZ + h_best * FREQ_HYP_STEP_HZ
        t_shift   = t_best - TIME_SEARCH_HALF

        cfo_ok  = abs(cfo_best - inj_cfo) <= CFO_TOL_HZ
        pss_ok  = p_best == inj_pss
        time_ok = abs(t_shift - inj_t) <= TIME_TOL_SAMPLES
        ok = cfo_ok and pss_ok and time_ok

        inj_str = f"({inj_cfo:+7.0f},{inj_pss},{inj_t:+4d})"
        est_str = f"({cfo_best:+7.0f},{p_best},{t_shift:+4d})"
        verdict = "PASS"
        if not ok:
            tags = []
            if not cfo_ok:  tags.append("cfo")
            if not pss_ok:  tags.append("pss")
            if not time_ok: tags.append("t")
            verdict = "FAIL[" + ",".join(tags) + "]"

        print(f"{case_dir:<32} {inj_str:>20} {est_str:>20} {peak:>10.2e}  {verdict}")
        if ok:
            n_pass += 1

    print("─" * 110)
    print(f"[verify] result: {n_pass} / {len(TEST_CASES)} PASS")
    sys.exit(0 if n_pass == len(TEST_CASES) else 1)


if __name__ == "__main__":
    main()
