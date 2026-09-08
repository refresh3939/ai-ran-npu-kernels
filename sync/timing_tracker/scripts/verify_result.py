"""
verify_result.py — 多 case 对比 kernel vs Python fp64 golden

验证项:
  1. CIR fp16 vs fp32 golden_cir  (abs tol 0.01)
  2. delta_T fp32 vs golden_delta_T  (abs tol 0.1 sample, 因为有抛物线插值精度限制)

支持单 case (根目录) 和多 case (case_N 子目录) 自动检测.
"""
import numpy as np
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).parent.parent
OUT_DIR     = PROJECT_DIR / 'data' / 'ascend_output'
GOLDEN_DIR  = PROJECT_DIR / 'data' / 'golden'

N_FFT    = 4096
W_SEARCH = 72

# tolerances
ABS_TOL_CIR = 0.01    # MVP v1 实测 max err = 0.00001
ABS_TOL_DT  = 0.1     # 0.1 sample = 抛物线插值精度上限


def verify_one_case(case_dir_out, case_dir_golden, case_label):
    """Returns (result_tuple, err_msg)."""
    ker_re_path = case_dir_out / 'cir_re.bin'
    ker_im_path = case_dir_out / 'cir_im.bin'
    ker_dt_path = case_dir_out / 'delta_T.bin'
    golden_cir_path = case_dir_golden / 'golden_cir.bin'
    golden_dt_path  = case_dir_golden / 'golden_delta_T.bin'
    
    if not ker_re_path.exists():
        return None, "kernel output not found: " + str(ker_re_path)
    if not golden_cir_path.exists():
        return None, "golden_cir not found: " + str(golden_cir_path)
    
    ker_re = np.fromfile(str(ker_re_path), dtype=np.float16).astype(np.float64)
    ker_im = np.fromfile(str(ker_im_path), dtype=np.float16).astype(np.float64)
    
    if len(ker_re) != N_FFT:
        return None, "size mismatch: %d ne %d" % (len(ker_re), N_FFT)
    
    golden = np.fromfile(str(golden_cir_path), dtype=np.float32).reshape(N_FFT, 2)
    g_re = golden[:, 0].astype(np.float64)
    g_im = golden[:, 1].astype(np.float64)
    
    err_re = np.abs(ker_re - g_re)
    err_im = np.abs(ker_im - g_im)
    
    cir_re_pass = err_re.max() < ABS_TOL_CIR
    cir_im_pass = err_im.max() < ABS_TOL_CIR
    
    # delta_T 对比 (如果有)
    dt_pass = True
    ker_dt = None
    g_dt   = None
    dt_err = None
    if ker_dt_path.exists() and golden_dt_path.exists():
        ker_dt = float(np.fromfile(str(ker_dt_path), dtype=np.float32)[0])
        g_dt   = float(np.fromfile(str(golden_dt_path), dtype=np.float32)[0])
        dt_err = abs(ker_dt - g_dt)
        dt_pass = dt_err < ABS_TOL_DT
    
    return {
        'cir_re_pass': cir_re_pass,
        'cir_im_pass': cir_im_pass,
        'dt_pass'    : dt_pass,
        'mer'        : err_re.max(),
        'mei'        : err_im.max(),
        'ker_dt'     : ker_dt,
        'g_dt'       : g_dt,
        'dt_err'     : dt_err,
    }, None


def main():
    case_dirs_golden = sorted(GOLDEN_DIR.glob('case_*'))
    
    if not case_dirs_golden:
        # 单 case 兼容
        print("[verify] single-case mode (root)")
        r, err = verify_one_case(OUT_DIR, GOLDEN_DIR, 'root')
        if err:
            print("[verify] x %s" % err); sys.exit(1)
        all_pass = r['cir_re_pass'] and r['cir_im_pass'] and r['dt_pass']
        print("[verify] CIR max err = (%.5f, %.5f)" % (r['mer'], r['mei']))
        if r['ker_dt'] is not None:
            print("[verify] delta_T: kernel=%+.4f, golden=%+.4f, err=%.4f" % (r['ker_dt'], r['g_dt'], r['dt_err']))
        print("[verify] OVERALL: %s" % ('PASS' if all_pass else 'FAIL'))
        sys.exit(0 if all_pass else 1)
    
    # Multi-case
    print("[verify] multi-case mode: %d cases" % len(case_dirs_golden))
    print("[verify] tolerances: CIR abs=%g, deltaT abs=%g" % (ABS_TOL_CIR, ABS_TOL_DT))
    print("")
    print("  case | CIR re err | CIR im err | k_dT       | g_dT       | dT err  | result")
    print("  -----+------------+------------+------------+------------+---------+-------")
    
    n_pass = 0
    failures = []
    for case_dir_g in case_dirs_golden:
        case_name = case_dir_g.name
        case_id = int(case_name.split('_')[1])
        case_dir_out = OUT_DIR / case_name
        
        r, err = verify_one_case(case_dir_out, case_dir_g, case_name)
        if err:
            print("  %4d | %s" % (case_id, err))
            failures.append(case_name)
            continue
        all_pass = r['cir_re_pass'] and r['cir_im_pass'] and r['dt_pass']
        status = 'PASS' if all_pass else 'FAIL'
        if all_pass:
            n_pass += 1
        else:
            reasons = []
            if not r['cir_re_pass']: reasons.append('CIR_re')
            if not r['cir_im_pass']: reasons.append('CIR_im')
            if not r['dt_pass']:     reasons.append('dT')
            failures.append("%s(%s)" % (case_name, '+'.join(reasons)))
        
        kdt_s = "%+.4f" % r['ker_dt'] if r['ker_dt'] is not None else "N/A"
        gdt_s = "%+.4f" % r['g_dt']   if r['g_dt']   is not None else "N/A"
        det_s = "%.4f"  % r['dt_err'] if r['dt_err'] is not None else "N/A"
        
        print("  %4d | %.6f   | %.6f   | %-10s | %-10s | %-7s |   %s" % (
            case_id, r['mer'], r['mei'], kdt_s, gdt_s, det_s, status))
    
    print("")
    print("[verify] %d/%d PASS" % (n_pass, len(case_dirs_golden)))
    if failures:
        print("[verify] FAILED: %s" % ', '.join(failures))
        print("[verify] OVERALL: FAIL")
        sys.exit(1)
    else:
        print("[verify] OVERALL: ALL PASS")
        sys.exit(0)


if __name__ == '__main__':
    main()
