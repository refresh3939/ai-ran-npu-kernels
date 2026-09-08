"""
PBCH DMRS Correlator — numpy reference + DMRS table + test vector generator.

Clean-room implementation: only 3GPP TS 38.211 §7.4.1.4 referenced.
Two reference paths:
  - pbch_dmrs_correlator_ref()       : fp64 algorithmic (sanity)
  - pbch_dmrs_correlator_fp16_sim()  : fp16-faithful (matches NPU kernel bit-exactly)

Host-side data prep:
  - prepare_kernel_inputs()  : packs R/D into the exact GM layout the kernel expects
                               (R replicated 16 rows, D transposed to [K,N] + pad-8,
                                D_im_neg separately).
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match pbch_dmrs_correlator.h
# ─────────────────────────────────────────────────────────────────────────────

N_DMRS_RE  = 144           # K (3GPP §7.4.1.4)
L_MAX_HW   = 8             # kernel always evaluates 8 lanes
N_PCID     = 1008
M_MMAD     = 16            # Cube min M (1 real row + 15 redundant)
K_SUB      = N_DMRS_RE     # 144, already 16-aligned (= 9 × 16)
N_SUB      = 16            # Cube min N (L_MAX_HW=8 → 16 pad)

OUT_FP32_COUNT = 24
SENTINEL_F32   = 7.0


# ─────────────────────────────────────────────────────────────────────────────
# DMRS reference sequence (3GPP TS 38.211 §7.4.1.4)
#
#   r_PBCH(m) = (1/sqrt(2)) · [ (1 - 2 c(2m))  +  j (1 - 2 c(2m+1)) ]
#
#   c_init = 2^11 · (i_SSB_bar + 1) · (N_ID/4 + 1)
#          + 2^6  · (i_SSB_bar + 1)
#          + (N_ID mod 4)
#
#   For L_max ∈ {4, 8}: i_SSB_bar = i_SSB.
# ─────────────────────────────────────────────────────────────────────────────

def _gold_seq(c_init, length):
    """Length-N Gold sequence, NC=1600 (TS 38.211 §5.2.1)."""
    NC = 1600
    n_total = length + NC + 31
    x1 = np.zeros(n_total, dtype=np.int8)
    x2 = np.zeros(n_total, dtype=np.int8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for n in range(n_total - 31):
        x1[n + 31] = (x1[n + 3] ^ x1[n]) & 1
        x2[n + 31] = (x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n]) & 1
    return (x1[NC : NC + length] ^ x2[NC : NC + length]) & 1


def pbch_dmrs_sequence(pcid, i_ssb, length=N_DMRS_RE):
    """One PBCH DMRS sequence, complex64, unit-amplitude QPSK / sqrt(2)."""
    i_bar = int(i_ssb)
    c_init = ((1 << 11) * (i_bar + 1) * (pcid // 4 + 1)
              + (1 << 6) * (i_bar + 1)
              + (pcid % 4)) & 0x7FFFFFFF
    c = _gold_seq(c_init, 2 * length)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    re = inv_sqrt2 * (1.0 - 2.0 * c[0::2])
    im = inv_sqrt2 * (1.0 - 2.0 * c[1::2])
    return (re + 1j * im).astype(np.complex64)


def build_dmrs_ref_table():
    """Returns [1008, 8, 144] complex64.  ~7s to build."""
    tbl = np.empty((N_PCID, L_MAX_HW, N_DMRS_RE), dtype=np.complex64)
    for pcid in range(N_PCID):
        for i in range(L_MAX_HW):
            tbl[pcid, i, :] = pbch_dmrs_sequence(pcid, i)
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Algorithmic reference (fp64)
# ─────────────────────────────────────────────────────────────────────────────

def pbch_dmrs_correlator_ref(dmrs_rx, dmrs_ref_table, pcid, l_max):
    """Returns (i_ssb, peak, second) in fp64."""
    assert dmrs_rx.shape == (N_DMRS_RE,)
    assert dmrs_ref_table.shape == (N_PCID, L_MAX_HW, N_DMRS_RE)
    assert l_max in (4, 8)

    ref = dmrs_ref_table[pcid, :l_max, :]                        # [L_max, 144]
    corr = ref.conj() @ dmrs_rx                                  # [L_max] complex
    metric = (corr.real.astype(np.float64) ** 2
              + corr.imag.astype(np.float64) ** 2)

    i_ssb = int(np.argmax(metric))
    peak  = float(metric[i_ssb])
    if l_max > 1:
        mc = metric.copy()
        mc[i_ssb] = -np.inf
        second = float(np.max(mc))
    else:
        second = 0.0
    return i_ssb, peak, second


# ─────────────────────────────────────────────────────────────────────────────
# fp16-faithful simulation reference (matches NPU kernel bit-exactly)
#
# Kernel does:
#   - All inputs are fp16
#   - Cube Mmad accumulates internally in fp32 (CO1 = fp32)
#   - L0C → UB fp32 (no cast)
#   - metric = Mul(re,re) + Mul(im,im) all in fp32
#   - argmax scalar fp32
#
# We mirror exactly:
#   - cast R and D to fp16 first
#   - matmul re/im paths in fp32 (sums of fp16·fp16 promoted)
#   - keep results fp32 through metric and argmax
#
# Karatsuba decomposition is the same as the kernel:
#   corr_re = R_re·D_re + R_im·D_im
#   corr_im = R_im·D_re + R_re·D_im_neg    (= R_im·D_re - R_re·D_im)
# ─────────────────────────────────────────────────────────────────────────────

def pbch_dmrs_correlator_fp16_sim(R_re_fp16, R_im_fp16,
                                  D_re_fp16, D_im_fp16,
                                  l_max):
    """
    Args:
        R_re_fp16, R_im_fp16 : [144] fp16
        D_re_fp16, D_im_fp16 : [L_MAX_HW=8, 144] fp16  (un-transposed view)
        l_max                : 4 or 8

    Returns:
        i_ssb, peak, second, corr_re_fp32[L_MAX_HW], corr_im_fp32[L_MAX_HW]
    """
    R_re_f32 = R_re_fp16.astype(np.float32)
    R_im_f32 = R_im_fp16.astype(np.float32)
    D_re_f32 = D_re_fp16.astype(np.float32)
    D_im_f32 = D_im_fp16.astype(np.float32)
    D_im_neg_f32 = (-D_im_fp16).astype(np.float32)

    # corr_re = R_re·D_re + R_im·D_im
    # corr_im = R_im·D_re + R_re·D_im_neg
    # All fp32 accum (Cube does this internally).
    corr_re = D_re_f32 @ R_re_f32 + D_im_f32     @ R_im_f32       # [L_MAX_HW]
    corr_im = D_re_f32 @ R_im_f32 + D_im_neg_f32 @ R_re_f32       # [L_MAX_HW]

    metric = corr_re ** 2 + corr_im ** 2                          # fp32

    i_ssb = int(np.argmax(metric[:l_max]))
    peak  = float(metric[i_ssb])
    mc = metric[:l_max].copy()
    mc[i_ssb] = -np.inf
    second = float(np.max(mc)) if l_max > 1 else 0.0
    if second < 0.0:
        second = 0.0

    # Pad corr to L_MAX_HW for the kernel-comparison API (lanes 8..15 stay 0 in kernel too)
    corr_re_out = np.zeros(L_MAX_HW, dtype=np.float32)
    corr_im_out = np.zeros(L_MAX_HW, dtype=np.float32)
    corr_re_out[:l_max] = corr_re[:l_max]
    corr_im_out[:l_max] = corr_im[:l_max]
    return i_ssb, peak, second, corr_re_out, corr_im_out


# ─────────────────────────────────────────────────────────────────────────────
# Host-side input packing (matches kernel GM layout)
#
# Quirk #17: B matrix in GM must be [K, N] row-major for LoadData2D ifTranspose=true.
# So host:
#   1. Pick the PCID sub-table: tbl[pcid, :, :]   # [8, 144]
#   2. Pad N: [8, 144] → [16, 144]                  (lanes 8..15 = 0)
#   3. Transpose to [K, N] = [144, 16]
#   4. Split into D_re fp16, D_im fp16, D_im_neg = -D_im fp16
#
# R is replicated 16 rows so the kernel can run a normal [M=16, K=144] GEMM:
#   1. R_re[16, 144] = broadcast(R_re_orig[144], M=16)
#   2. R_im[16, 144] = broadcast(R_im_orig[144], M=16)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_kernel_inputs(dmrs_rx_complex, dmrs_ref_table, pcid):
    """
    Returns dict ready to write to GM:
        R_re_fp16     : [16, 144]  fp16   (4608 B; 16 identical rows)
        R_im_fp16     : [16, 144]  fp16
        D_re_fp16     : [144, 16]  fp16   (K outer, N inner; N[8..15]=0)
        D_im_fp16     : [144, 16]  fp16
        D_im_neg_fp16 : [144, 16]  fp16   (= -D_im, bit-exact sign flip)
        D_un_t_re_fp16: [8, 144]   fp16   (for fp16-sim ref, not for kernel)
        D_un_t_im_fp16: [8, 144]   fp16
    """
    assert dmrs_rx_complex.shape == (N_DMRS_RE,)

    # R replication
    R_re_orig = dmrs_rx_complex.real.astype(np.float16)
    R_im_orig = dmrs_rx_complex.imag.astype(np.float16)
    R_re_mat = np.tile(R_re_orig.reshape(1, -1), (M_MMAD, 1))     # [16, 144]
    R_im_mat = np.tile(R_im_orig.reshape(1, -1), (M_MMAD, 1))

    # D sub-table → pad N → transpose
    sub = dmrs_ref_table[pcid, :, :]                              # [8, 144] complex64
    sub_re_fp16 = sub.real.astype(np.float16)
    sub_im_fp16 = sub.imag.astype(np.float16)

    sub_re_padded = np.zeros((N_SUB, N_DMRS_RE), dtype=np.float16)
    sub_im_padded = np.zeros((N_SUB, N_DMRS_RE), dtype=np.float16)
    sub_re_padded[:L_MAX_HW, :] = sub_re_fp16
    sub_im_padded[:L_MAX_HW, :] = sub_im_fp16

    D_re = sub_re_padded.T.astype(np.float16)                     # [144, 16]
    D_im = sub_im_padded.T.astype(np.float16)
    D_im_neg = (-D_im).astype(np.float16)                         # bit-exact sign flip

    return {
        'R_re_fp16':      np.ascontiguousarray(R_re_mat),
        'R_im_fp16':      np.ascontiguousarray(R_im_mat),
        'D_re_fp16':      np.ascontiguousarray(D_re),
        'D_im_fp16':      np.ascontiguousarray(D_im),
        'D_im_neg_fp16':  np.ascontiguousarray(D_im_neg),
        # un-transposed views (for fp16-sim only, not written to GM)
        'D_un_t_re_fp16': sub_re_fp16,
        'D_un_t_im_fp16': sub_im_fp16,
        'R_re_orig_fp16': R_re_orig,
        'R_im_orig_fp16': R_im_orig,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    print("[self-test] building DMRS reference table (1008 × 8 sequences) ...")
    tbl = build_dmrs_ref_table()
    print(f"[self-test] table built, shape={tbl.shape}, dtype={tbl.dtype}")

    cases = [
        # (pcid, true_i_ssb, l_max, noise_std)
        (0,    0, 4, 0.0),
        (42,   3, 4, 0.0),
        (42,   0, 4, 0.10),
        (505,  7, 8, 0.05),
        (1007, 5, 8, 0.20),
    ]
    rng = np.random.default_rng(20260525)

    n_fail = 0
    for pcid, true_i, l_max, noise_std in cases:
        ref = tbl[pcid, true_i, :]
        if noise_std > 0:
            noise = (rng.standard_normal(N_DMRS_RE) + 1j * rng.standard_normal(N_DMRS_RE)
                    ).astype(np.complex64) * (noise_std / np.sqrt(2))
            rx = (ref + noise).astype(np.complex64)
        else:
            rx = ref.astype(np.complex64)

        # fp64 ref
        i_alg, p_alg, s_alg = pbch_dmrs_correlator_ref(rx, tbl, pcid, l_max)

        # fp16-sim ref
        inp = prepare_kernel_inputs(rx, tbl, pcid)
        i_sim, p_sim, s_sim, _, _ = pbch_dmrs_correlator_fp16_sim(
            inp['R_re_orig_fp16'], inp['R_im_orig_fp16'],
            inp['D_un_t_re_fp16'], inp['D_un_t_im_fp16'],
            l_max,
        )

        ok = (i_alg == true_i) and (i_sim == true_i)
        n_fail += (0 if ok else 1)
        print(f"  pcid={pcid:4d} true={true_i} L={l_max} σ={noise_std:.2f}  "
              f"alg→{i_alg}(p={p_alg:.1f},2nd={s_alg:.1f})  "
              f"fp16→{i_sim}(p={p_sim:.1f},2nd={s_sim:.1f})  "
              f"{'PASS' if ok else 'FAIL'}")

    print(f"\n[self-test] {len(cases)-n_fail}/{len(cases)} passed")
    return n_fail == 0


if __name__ == '__main__':
    ok = _self_test()
    raise SystemExit(0 if ok else 1)
