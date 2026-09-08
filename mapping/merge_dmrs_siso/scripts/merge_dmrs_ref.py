#!/usr/bin/env python3
# ============================================================================
# merge_dmrs_ref.py — §0 reference + golden for merge_dmrs (TX DMRS insertion)
#
# merge_dmrs 把 dmrs_gen 输出 [2,896] comb-2 散射进 grid_used[14,1664]:
#   gu[DMRS_SYM[s], 2k] = dmrs[s,k]   k=0..797   (comb-2, delta=0, 端口0)
#   奇数 SC(2k+1)=0, pad[1596:1664]=0; 数据符号行不变。
#
# 产 golden 到 ${AIRAN_DATA_DIR}/data/golden/case_0/:
#   dmrs_in.bin   [2,896]  fp16   要插入的导频(取自 dmrs_gen golden 或合成)
#   gu_in.bin     [14,1664] fp16  输入栅格(数据符号随机, DMRS 符号=0)
#   gu_merged.bin [14,1664] fp16  期望输出(kernel/host 必须 bit-exact 复现)
#
# 关键不变量(OTA 一致性): RX channel_est_ls 从 sym 2/11 偶数 SC 提取 == 原 dmrs。
# ============================================================================
import os, sys
import numpy as np
from pathlib import Path

N_SYMBOL   = 14
N_SC_USED  = 1596
N_SC_PAD   = 1664
N_DMRS_RE  = 798
N_DMRS_PAD = 896
DMRS_SYM   = [2, 11]
COMB, DELTA = 2, 0


def merge_dmrs(gu_re, gu_im, dmrs_re, dmrs_im):
    out_re, out_im = gu_re.copy(), gu_im.copy()
    k = np.arange(N_DMRS_RE)
    for s, l in enumerate(DMRS_SYM):
        out_re[l, :] = 0.0; out_im[l, :] = 0.0
        out_re[l, COMB*k + DELTA] = dmrs_re[s, :N_DMRS_RE]
        out_im[l, COMB*k + DELTA] = dmrs_im[s, :N_DMRS_RE]
    return out_re, out_im


def extract_dmrs(gu_re, gu_im):
    k = np.arange(N_DMRS_RE)
    er = np.stack([gu_re[l, COMB*k + DELTA] for l in DMRS_SYM])
    ei = np.stack([gu_im[l, COMB*k + DELTA] for l in DMRS_SYM])
    return er, ei


def load_dmrs():
    """优先 dmrs_gen 真 golden(DMRS_GEN_GOLDEN=<case dir>); 否则合成 ±1/√2 QPSK。"""
    dg = os.environ.get("DMRS_GEN_GOLDEN")
    if dg and (Path(dg) / "x_re.bin").exists():
        dr = np.fromfile(Path(dg)/"x_re.bin", np.float16).reshape(2, N_DMRS_PAD)
        di = np.fromfile(Path(dg)/"x_im.bin", np.float16).reshape(2, N_DMRS_PAD)
        print(f"  dmrs from dmrs_gen golden: {dg}")
        return dr, di
    inv = np.float16(1/np.sqrt(2)); rng = np.random.default_rng(0xD)
    dr = np.zeros((2, N_DMRS_PAD), np.float16); di = np.zeros((2, N_DMRS_PAD), np.float16)
    dr[:, :N_DMRS_RE] = (1-2*rng.integers(0,2,(2,N_DMRS_RE)))*inv
    di[:, :N_DMRS_RE] = (1-2*rng.integers(0,2,(2,N_DMRS_RE)))*inv
    print("  dmrs synthetic ±1/√2 QPSK (set DMRS_GEN_GOLDEN=<case dir> for real)")
    return dr, di


def selftest(gu_re0, gu_im0, mr, mi, dr, di):
    print("=== merge_dmrs §0 self-test ===")
    ok_even = all(np.array_equal(mr[l,0:2*N_DMRS_RE:2], dr[s,:N_DMRS_RE]) and
                  np.array_equal(mi[l,0:2*N_DMRS_RE:2], di[s,:N_DMRS_RE])
                  for s,l in enumerate(DMRS_SYM))
    ok_zero = all(bool((mr[l,1:2*N_DMRS_RE:2]==0).all() and (mi[l,1:2*N_DMRS_RE:2]==0).all()
                  and (mr[l,N_SC_USED:]==0).all()) for l in DMRS_SYM)
    data = [s for s in range(N_SYMBOL) if s not in DMRS_SYM]
    ok_data = np.array_equal(mr[data], gu_re0[data]) and np.array_equal(mi[data], gu_im0[data])
    er, ei = extract_dmrs(mr, mi)
    ok_rt = np.array_equal(er, dr[:,:N_DMRS_RE]) and np.array_equal(ei, di[:,:N_DMRS_RE])
    print(f"  [1] DMRS @ 偶数 SC == dmrs        : {ok_even}")
    print(f"  [2] 奇数 SC + pad = 0             : {ok_zero}")
    print(f"  [3] 数据符号行不变                : {ok_data}")
    print(f"  [4] extract(insert)==dmrs (TX↔RX) : {ok_rt}")
    ok = ok_even and ok_zero and ok_data and ok_rt
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    root = Path(os.environ.get("AIRAN_DATA_DIR", Path(__file__).resolve().parents[1]))
    out = root / "data" / "golden" / "case_0"
    out.mkdir(parents=True, exist_ok=True)
    print("=== merge_dmrs reference (comb-2 DMRS insert, l={2,11}) ===")
    print(f"  output = {out}")

    dr, di = load_dmrs()
    rng = np.random.default_rng(7)
    gu_re = rng.standard_normal((N_SYMBOL, N_SC_PAD)).astype(np.float16)
    gu_im = rng.standard_normal((N_SYMBOL, N_SC_PAD)).astype(np.float16)
    gu_re[:, N_SC_USED:] = 0; gu_im[:, N_SC_USED:] = 0
    for l in DMRS_SYM: gu_re[l,:] = 0; gu_im[l,:] = 0      # qam256_mod 留 0

    mr, mi = merge_dmrs(gu_re, gu_im, dr, di)

    dmrs_in = np.zeros((2, N_DMRS_PAD), np.float16); dmrs_in[0]=dr[0]; dmrs_in[1]=dr[1]
    np.stack([dr[0], dr[1]]).astype(np.float16).tofile(out/"dmrs_re_unused.bin")  # placeholder
    # 写 golden(re/im 各拼一个文件: [2,896] 与 [14,1664])
    np.ascontiguousarray(dr).tofile(out/"dmrs_in_re.bin")
    np.ascontiguousarray(di).tofile(out/"dmrs_in_im.bin")
    np.ascontiguousarray(gu_re).tofile(out/"gu_in_re.bin")
    np.ascontiguousarray(gu_im).tofile(out/"gu_in_im.bin")
    np.ascontiguousarray(mr).tofile(out/"gu_merged_re.bin")
    np.ascontiguousarray(mi).tofile(out/"gu_merged_im.bin")
    os.remove(out/"dmrs_re_unused.bin")
    print(f"  wrote dmrs_in_{{re,im}} [2,{N_DMRS_PAD}], gu_in_{{re,im}} / gu_merged_{{re,im}} [{N_SYMBOL},{N_SC_PAD}] fp16")

    return 0 if selftest(gu_re, gu_im, mr, mi, dr, di) else 1


if __name__ == "__main__":
    sys.exit(main())
