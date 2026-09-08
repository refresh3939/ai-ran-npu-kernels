"""
PBCH DMRS Correlator — test case generator.

Generates per-case directories under:
  <project_root>/data/golden/case_<N>_<name>/
    inputs/
      r_re.bin           [16, 144] fp16   (R replicated 16 rows)
      r_im.bin           [16, 144] fp16
      d_re.bin           [144, 16] fp16   ([K,N], pad N to 16)
      d_im.bin           [144, 16] fp16
      d_im_neg.bin       [144, 16] fp16   (= -d_im)
      tiling.bin         64 B             (PbchDmrsTilingV1: l_max, pcid, block_dim)
    golden/
      output_ref.bin     24 fp32          (i_ssb, peak, second, l_max, corr_re[8],
                                            corr_im[8], dbg_R_re_0, dbg_D_re_00,
                                            metric_max, sentinel)
      truth.json                          (ground truth for verifier)

<project_root> = parent of this scripts/ directory (the pbch_dmrs_correlator/ kernel folder)
                 i.e. the operator is fully self-contained.

Cases:
  case_0_clean_L4_i0   : clean, L_max=4, i_ssb=0, pcid=42
  case_1_clean_L4_i3   : clean, L_max=4, i_ssb=3, pcid=42
  case_2_clean_L8_i7   : clean, L_max=8, i_ssb=7, pcid=505
  case_3_noisy_L4      : σ=0.10, L_max=4
  case_4_low_snr_L8    : σ=0.30, L_max=8
  case_5_edge_pcid     : pcid=1007 (edge), L_max=8
  case_6_pcid_0        : pcid=0, L_max=4
  case_7_pcid_max_L8   : pcid=1007, L_max=8
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# scripts/ → parent = project root
THIS_DIR = Path(__file__).resolve().parent
PROJ_DIR = THIS_DIR.parent      # pbch_dmrs_correlator/
sys.path.insert(0, str(THIS_DIR))
from pbch_dmrs_correlator_ref import (
    N_DMRS_RE, L_MAX_HW, N_PCID, M_MMAD, K_SUB, N_SUB,
    OUT_FP32_COUNT, SENTINEL_F32,
    build_dmrs_ref_table, prepare_kernel_inputs,
    pbch_dmrs_correlator_ref, pbch_dmrs_correlator_fp16_sim,
)


# ─────────────────────────────────────────────────────────────────────────────
# Output path resolution
#
# Operator-local layout:
#   pbch_dmrs_correlator/
#     ├── scripts/  (this file)
#     ├── data/
#     │   ├── golden/case_*/...
#     │   └── ascend_output/
#     └── ...
#
# AIRAN_DATA_DIR is no longer used. Override via PBCH_DMRS_DATA_ROOT env var
# if you need a non-default location.
# ─────────────────────────────────────────────────────────────────────────────

def resolve_output_root():
    """Returns the directory under which 'data/' will live."""
    if 'PBCH_DMRS_DATA_ROOT' in os.environ:
        return Path(os.environ['PBCH_DMRS_DATA_ROOT'])
    return PROJ_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Tiling binary (matches PbchDmrsTilingV1 in pbch_dmrs_tiling.h, 64 B)
# ─────────────────────────────────────────────────────────────────────────────

def pack_tiling(l_max, pcid, block_dim=4):
    """Returns 64 raw bytes."""
    # struct layout: int32 l_max; int32 pcid; uint32 block_dim; int32 _pad0;
    #                uint32 _pad[12]
    buf = struct.pack('<iiIi', int(l_max), int(pcid), int(block_dim), 0)
    buf += b'\x00' * (64 - len(buf))
    assert len(buf) == 64
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Per-case generator
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_rx(ref_seq, noise_std, rng):
    """ref_seq is complex64 [144]. Returns rx complex64 [144]."""
    if noise_std <= 0:
        return ref_seq.astype(np.complex64)
    noise = (rng.standard_normal(N_DMRS_RE) + 1j * rng.standard_normal(N_DMRS_RE)
            ).astype(np.complex64) * (noise_std / np.sqrt(2))
    return (ref_seq + noise).astype(np.complex64)


def build_golden_output(i_ssb_sim, peak_sim, second_sim,
                       corr_re_sim, corr_im_sim,
                       R_re_orig_fp16, D_re_fp16,
                       l_max):
    """
    Build the 24-fp32 output the kernel should produce (fp16-sim baseline).
    Layout matches pbch_dmrs_correlator.h.
    """
    out = np.zeros(OUT_FP32_COUNT, dtype=np.float32)
    out[0]  = float(i_ssb_sim)
    out[1]  = float(peak_sim)
    out[2]  = float(second_sim)
    out[3]  = float(l_max)
    out[4:12]  = corr_re_sim.astype(np.float32)
    out[12:20] = corr_im_sim.astype(np.float32)
    out[20] = float(R_re_orig_fp16[0])                  # dbg_R_re_0
    # D_re[0, 0] = first half element of D_re GM ([K=0, N=0])
    out[21] = float(D_re_fp16[0, 0])                    # dbg_D_re_00
    out[22] = float(peak_sim)                            # metric_max (= peak)
    out[23] = SENTINEL_F32
    return out


CASES = [
    dict(name='case_0_clean_L4_i0',  pcid=42,   true_i=0, l_max=4, noise=0.0),
    dict(name='case_1_clean_L4_i3',  pcid=42,   true_i=3, l_max=4, noise=0.0),
    dict(name='case_2_clean_L8_i7',  pcid=505,  true_i=7, l_max=8, noise=0.0),
    dict(name='case_3_noisy_L4',     pcid=42,   true_i=2, l_max=4, noise=0.10),
    dict(name='case_4_low_snr_L8',   pcid=505,  true_i=4, l_max=8, noise=0.30),
    dict(name='case_5_edge_pcid',    pcid=1007, true_i=5, l_max=8, noise=0.05),
    dict(name='case_6_pcid_0',       pcid=0,    true_i=0, l_max=4, noise=0.0),
    dict(name='case_7_pcid_max_L8',  pcid=1007, true_i=7, l_max=8, noise=0.0),
]


def generate_case(case_dir, case_spec, tbl, rng):
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / 'inputs').mkdir(exist_ok=True)
    (case_dir / 'golden').mkdir(exist_ok=True)

    pcid   = case_spec['pcid']
    true_i = case_spec['true_i']
    l_max  = case_spec['l_max']
    noise  = case_spec['noise']

    # Synthesise rx
    ref_seq = tbl[pcid, true_i, :]
    rx = synthesize_rx(ref_seq, noise, rng)

    # Prepare kernel inputs
    inp = prepare_kernel_inputs(rx, tbl, pcid)

    # Write input binaries
    (case_dir / 'inputs' / 'r_re.bin').write_bytes(inp['R_re_fp16'].tobytes())
    (case_dir / 'inputs' / 'r_im.bin').write_bytes(inp['R_im_fp16'].tobytes())
    (case_dir / 'inputs' / 'd_re.bin').write_bytes(inp['D_re_fp16'].tobytes())
    (case_dir / 'inputs' / 'd_im.bin').write_bytes(inp['D_im_fp16'].tobytes())
    (case_dir / 'inputs' / 'd_im_neg.bin').write_bytes(inp['D_im_neg_fp16'].tobytes())
    (case_dir / 'inputs' / 'tiling.bin').write_bytes(pack_tiling(l_max, pcid))

    # fp16-sim reference (kernel-faithful) — golden output the kernel must match
    i_sim, p_sim, s_sim, corr_re_sim, corr_im_sim = pbch_dmrs_correlator_fp16_sim(
        inp['R_re_orig_fp16'], inp['R_im_orig_fp16'],
        inp['D_un_t_re_fp16'], inp['D_un_t_im_fp16'],
        l_max,
    )

    golden = build_golden_output(
        i_sim, p_sim, s_sim, corr_re_sim, corr_im_sim,
        inp['R_re_orig_fp16'], inp['D_re_fp16'],
        l_max,
    )
    (case_dir / 'golden' / 'output_ref.bin').write_bytes(golden.tobytes())

    # Algorithmic fp64 ref (sanity)
    i_alg, p_alg, s_alg = pbch_dmrs_correlator_ref(rx, tbl, pcid, l_max)

    truth = {
        'case_name'    : case_spec['name'],
        'pcid'         : int(pcid),
        'true_i_ssb'   : int(true_i),
        'l_max'        : int(l_max),
        'noise_std'    : float(noise),
        'alg_i_ssb'    : int(i_alg),
        'alg_peak'     : float(p_alg),
        'alg_second'   : float(s_alg),
        'fp16_i_ssb'   : int(i_sim),
        'fp16_peak'    : float(p_sim),
        'fp16_second'  : float(s_sim),
        'fp16_corr_re' : corr_re_sim.tolist(),
        'fp16_corr_im' : corr_im_sim.tolist(),
        'dbg_R_re_0'   : float(inp['R_re_orig_fp16'][0]),
        'dbg_D_re_00'  : float(inp['D_re_fp16'][0, 0]),
    }
    (case_dir / 'golden' / 'truth.json').write_text(json.dumps(truth, indent=2))

    return truth


def main():
    out_root = resolve_output_root()
    case_base = out_root / 'data' / 'golden'      # 算子本地 data/golden/
    case_base.mkdir(parents=True, exist_ok=True)
    print(f"[gen] case output root: {case_base}")

    print("[gen] building DMRS reference table (~7s) ...")
    tbl = build_dmrs_ref_table()
    print(f"[gen] table: {tbl.shape}, dtype={tbl.dtype}")

    # Write the full reference table once (informational, not for kernel)
    table_dir = case_base / 'common'
    table_dir.mkdir(exist_ok=True)
    np.save(table_dir / 'dmrs_ref_table.npy', tbl)
    print(f"[gen] saved reference table to {table_dir / 'dmrs_ref_table.npy'} "
          f"({tbl.nbytes / 1024:.1f} KB)")

    # Per-case
    rng = np.random.default_rng(20260525)
    print(f"\n[gen] generating {len(CASES)} test cases:")
    all_truth = []
    for spec in CASES:
        case_dir = case_base / spec['name']
        truth = generate_case(case_dir, spec, tbl, rng)
        all_truth.append(truth)
        print(f"  {spec['name']:30s} pcid={spec['pcid']:4d} "
              f"true_i={spec['true_i']} L={spec['l_max']} σ={spec['noise']:.2f}  "
              f"alg→{truth['alg_i_ssb']}(p={truth['alg_peak']:.1f})  "
              f"fp16→{truth['fp16_i_ssb']}(p={truth['fp16_peak']:.1f})  "
              f"{'✓' if truth['fp16_i_ssb']==spec['true_i'] else '✗'}")

    print(f"\n[gen] done. {len(CASES)} cases written under {case_base}")


if __name__ == '__main__':
    main()
