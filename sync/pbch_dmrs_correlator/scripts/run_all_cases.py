"""
PBCH DMRS Correlator — test driver.

Self-contained, runs out of the operator project root:
  pbch_dmrs_correlator/
    ├── scripts/run_all_cases.py    (this file)
    ├── data/golden/case_*/         (test cases)
    ├── data/ascend_output/         (NPU result dropped here per run)
    └── out/bin/ascendc_kernels_bbit (built by run.sh first)

For each test case under data/golden/:
  1. set PBCH_DMRS_CASE_DIR + PBCH_DMRS_DATA_ROOT env vars
  2. run the kernel (cd out/bin; ./ascendc_kernels_bbit)
  3. read data/ascend_output/output.bin
  4. compare against case_X/golden/output_ref.bin + truth.json

Reports:
  - i_ssb correctness (must match true_i_ssb)
  - argmax peak / second sanity
  - corr_re[0..7] / corr_im[0..7] bit-exact vs fp16-sim ref (informational)
  - per-case latency (re-extracted from kernel stdout)
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJ_DIR = THIS_DIR.parent          # pbch_dmrs_correlator/


def resolve_paths():
    """Operator-local paths. PBCH_DMRS_DATA_ROOT overrides location of data/."""
    data_root = Path(os.environ.get('PBCH_DMRS_DATA_ROOT', str(PROJ_DIR)))
    case_root = data_root / 'data' / 'golden'
    out_dir   = data_root / 'data' / 'ascend_output'
    binary    = PROJ_DIR / 'out' / 'bin' / 'ascendc_kernels_bbit'
    libdir    = PROJ_DIR / 'out' / 'lib'
    ascend_home = os.environ.get('ASCEND_HOME_PATH',
                                 '/usr/local/Ascend/ascend-toolkit/latest')
    return data_root, case_root, out_dir, binary, libdir, ascend_home


def run_one_case(case_name, case_root, binary, libdir, out_dir, data_root, ascend_home):
    case_dir = case_root / case_name
    truth = json.loads((case_dir / 'golden' / 'truth.json').read_text())

    env = os.environ.copy()
    env['PBCH_DMRS_CASE_DIR']  = case_name
    env['PBCH_DMRS_DATA_ROOT'] = str(data_root)
    env['LD_LIBRARY_PATH']     = f"{libdir}:{ascend_home}/lib64:" + env.get('LD_LIBRARY_PATH', '')
    env['ASCEND_GLOBAL_LOG_LEVEL'] = '3'

    proc = subprocess.run([str(binary)], cwd=str(binary.parent),
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return dict(name=case_name, status='RUN_FAIL', truth=truth, npu=None,
                    p50_us=None, stdout=proc.stdout, stderr=proc.stderr)

    # parse p50 latency from stdout
    p50 = None
    m = re.search(r'p50\s+([\d.]+)\s+us', proc.stdout)
    if m: p50 = float(m.group(1))

    # read npu output
    npu_out = np.fromfile(out_dir / 'output.bin', dtype=np.float32)
    if len(npu_out) != 24:
        return dict(name=case_name, status='OUT_BAD_SIZE', truth=truth, npu=npu_out.tolist(),
                    p50_us=p50, stdout=proc.stdout, stderr=proc.stderr)

    npu_i_ssb  = int(npu_out[0])
    npu_peak   = float(npu_out[1])
    npu_second = float(npu_out[2])
    npu_lmax   = int(npu_out[3])
    npu_corr_re = npu_out[4:12].copy()
    npu_corr_im = npu_out[12:20].copy()
    npu_sentinel = float(npu_out[23])

    # compare to fp16-sim ref (kernel-faithful)
    ref_corr_re = np.array(truth['fp16_corr_re'], dtype=np.float32)
    ref_corr_im = np.array(truth['fp16_corr_im'], dtype=np.float32)
    true_i_ssb = int(truth['true_i_ssb'])

    # i_ssb correctness check (against ground truth, not just ref)
    i_ssb_ok = (npu_i_ssb == true_i_ssb)
    sentinel_ok = (npu_sentinel == 7.0)

    # bit-exact corr check (vs fp16-sim) — informational only
    if ref_corr_re.any():
        corr_re_rel = np.max(np.abs(npu_corr_re - ref_corr_re)) / max(np.max(np.abs(ref_corr_re)), 1e-6)
    else:
        corr_re_rel = 0.0
    if ref_corr_im.any():
        corr_im_rel = np.max(np.abs(npu_corr_im - ref_corr_im)) / max(np.max(np.abs(ref_corr_im)), 1e-6)
    else:
        corr_im_rel = 0.0

    status = 'PASS' if (i_ssb_ok and sentinel_ok) else 'FAIL'
    return dict(
        name=case_name, status=status,
        truth=truth,
        npu_i_ssb=npu_i_ssb, npu_peak=npu_peak, npu_second=npu_second,
        npu_lmax=npu_lmax, npu_sentinel=npu_sentinel,
        npu_corr_re=npu_corr_re.tolist(), npu_corr_im=npu_corr_im.tolist(),
        corr_re_rel=corr_re_rel, corr_im_rel=corr_im_rel,
        p50_us=p50, i_ssb_ok=i_ssb_ok, sentinel_ok=sentinel_ok,
        stdout=proc.stdout, stderr=proc.stderr,
    )


def main():
    data_root, case_root, out_dir, binary, libdir, ascend_home = resolve_paths()
    print(f"[test] PROJ_DIR       = {PROJ_DIR}")
    print(f"[test] data_root      = {data_root}")
    print(f"[test] case_root      = {case_root}")
    print(f"[test] binary         = {binary}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not binary.exists():
        print(f"[error] binary not found. run `bash run.sh` first to build.")
        sys.exit(1)
    if not case_root.is_dir():
        print(f"[error] case root missing. run `python3 scripts/gen_test_cases.py` first.")
        sys.exit(1)

    cases = sorted([p.name for p in case_root.iterdir() if p.is_dir() and p.name.startswith('case_')])
    print(f"[test] found {len(cases)} cases\n")

    results = []
    for cn in cases:
        r = run_one_case(cn, case_root, binary, libdir, out_dir, data_root, ascend_home)
        results.append(r)

    # ── Summary ──
    print("\n" + "═" * 120)
    print(f"{'case':32s}  {'status':6s}  {'L':>2s}  {'truth_i':>7s}  {'npu_i':>5s}  "
          f"{'peak':>9s}  {'2nd':>8s}  {'p50µs':>7s}  {'corr_re_rel':>12s}  {'corr_im_rel':>12s}")
    print("─" * 120)
    n_pass = 0
    for r in results:
        n = r['name']
        s = r['status']
        if 'npu_i_ssb' not in r:
            print(f"{n:32s}  {s:6s}  --   --      --       --        --        --        --              --")
            continue
        t = r['truth']
        glyph = '✓' if r['status'] == 'PASS' else '✗'
        if r['status'] == 'PASS':
            n_pass += 1
        print(f"{n:32s}  {s:>4s}{glyph}  "
              f"{t['l_max']:>2d}  {t['true_i_ssb']:>7d}  {r['npu_i_ssb']:>5d}  "
              f"{r['npu_peak']:>9.2f}  {r['npu_second']:>8.2f}  "
              f"{r['p50_us'] if r['p50_us'] else 0:>7.1f}  "
              f"{r['corr_re_rel']:>12.2e}  {r['corr_im_rel']:>12.2e}")
    print("═" * 120)
    print(f"[test] {n_pass}/{len(results)} PASS")

    if n_pass < len(results):
        print("\n[test] failing case diagnostics:")
        for r in results:
            if r['status'] != 'PASS':
                print(f"\n── {r['name']} ──")
                print(f"  status={r['status']}")
                if 'npu_i_ssb' in r:
                    print(f"  i_ssb_ok={r.get('i_ssb_ok')}  sentinel_ok={r.get('sentinel_ok')}")
                    print(f"  npu_sentinel={r.get('npu_sentinel')}")
                print(f"  stdout tail (last 40 lines):")
                for line in r['stdout'].strip().splitlines()[-40:]:
                    print(f"    {line}")
                if r['stderr']:
                    print(f"  stderr:")
                    for line in r['stderr'].strip().splitlines()[-10:]:
                        print(f"    {line}")

    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == '__main__':
    main()
