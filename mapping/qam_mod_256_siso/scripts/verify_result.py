#!/usr/bin/env python3
# ============================================================================
# verify_result.py — 256-QAM mod kernel 输出独立校验(可选;run.sh/main 已内置校验)
#
# 对比 kernel 产出的 x_re/x_im(全网格 [N_SYMBOL, N_SC_PAD] fp16)与 golden:
#   1. data 行 [phys, 0:N_SC_USED] —— 位精确(bit-exact)
#   2. 尾部 [N_SC_USED:N_SC_PAD] —— 全 0
#   3. DMRS 行 {2,11} —— 全 0
#   4. 无 0xAA sentinel 残留
#
# Usage:
#   python3 verify_result.py [--case case_0_random_a] [--data-root /path/to/data]
# ============================================================================
import os, sys, argparse
import numpy as np

N_SYMBOL  = 14
DMRS_SYMS = {2, 11}
N_SC_USED = 1596
N_SC_PAD  = 1664
N_DATA_SYM= N_SYMBOL - len(DMRS_SYMS)
DATA_SYM_TO_PHYS = [s for s in range(N_SYMBOL) if s not in DMRS_SYMS]


def check_one(kernel_path, golden_path, label):
    g = np.fromfile(golden_path, dtype=np.uint16)
    k = np.fromfile(kernel_path, dtype=np.uint16)
    exp = N_SYMBOL * N_SC_PAD
    if g.size != exp or k.size != exp:
        print(f"  [{label}] size mismatch kernel={k.size} gold={g.size} exp={exp}")
        return False
    g = g.reshape(N_SYMBOL, N_SC_PAD); k = k.reshape(N_SYMBOL, N_SC_PAD)

    sentinel = int((k == 0xAAAA).sum())
    diff = 0; tail_nz = 0; dmrs_nz = 0
    for di, phys in enumerate(DATA_SYM_TO_PHYS):
        diff    += int((k[phys, :N_SC_USED] != g[phys, :N_SC_USED]).sum())
        tail_nz += int((k[phys, N_SC_USED:] != 0).sum())
    for s in DMRS_SYMS:
        dmrs_nz += int((k[s] != 0).sum())

    ok = (diff == 0 and tail_nz == 0 and dmrs_nz == 0 and sentinel == 0)
    print(f"  [{label}] diff={diff} tail_nz={tail_nz} dmrs_nz={dmrs_nz} "
          f"sentinel={sentinel} => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("AIRAN_DATA_DIR", "../../data"))
    ap.add_argument("--case", default="case_0_random_a")
    args = ap.parse_args()

    gdir = os.path.join(args.data_root, "golden", args.case)
    odir = os.path.join(args.data_root, "ascend_output", args.case)

    print("="*60)
    print("256-QAM Mod verify_result")
    print(f"  case: {args.case}")
    print(f"  golden: {gdir}")
    print(f"  kernel: {odir}")
    print("="*60)

    ok = True
    for comp in ("x_re", "x_im"):
        gp = os.path.join(gdir, comp + ".bin")
        kp = os.path.join(odir, comp + ".bin")
        if not os.path.isfile(gp) or not os.path.isfile(kp):
            print(f"  [{comp}] missing file (run ref.py + kernel first)"); ok = False; continue
        ok &= check_one(kp, gp, comp)

    print("="*60)
    print("VERIFY OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
