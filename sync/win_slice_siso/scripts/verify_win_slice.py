#!/usr/bin/env python3
# verify_win_slice.py —— 对照 ascend_output vs golden,5/5 PASS 判定
import numpy as np, os, struct, sys

N_SAMP_SLOT = 30720
IQ_LEN = 2 * N_SAMP_SLOT
DATA_DIR = os.environ.get("AIRAN_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

def r_i16(p): return np.fromfile(p, dtype='<i2')
def r_i32(p): return struct.unpack('<i', open(p,'rb').read(4))[0]

def main():
    ncase = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    npass = 0
    for c in range(ncase):
        g = os.path.join(DATA_DIR, "golden", f"case_{c}")
        o = os.path.join(DATA_DIR, "ascend_output", f"case_{c}")
        try:
            sg = r_i16(os.path.join(g, "slot_iq_gold.bin"))
            so = r_i16(os.path.join(o, "slot_iq.bin"))
            rg = r_i32(os.path.join(g, "rd_ptr_out_gold.bin"))
            ro = r_i32(os.path.join(o, "rd_ptr_out.bin"))
        except FileNotFoundError as e:
            print(f"[case {c}] FAIL  missing {e.filename}"); continue
        iq_ok = (so.shape == sg.shape) and np.array_equal(so, sg)
        rd_ok = (ro == rg)
        if iq_ok and rd_ok:
            print(f"[case {c}] PASS  rd_out={ro}"); npass += 1
        else:
            nbad = int(np.sum(so != sg)) if so.shape == sg.shape else -1
            print(f"[case {c}] FAIL  iq_ok={iq_ok}(bad={nbad})  rd_ok={rd_ok}(got {ro} want {rg})")
    print(f"\n{npass}/{ncase} PASS")
    sys.exit(0 if npass == ncase else 1)

if __name__ == "__main__":
    main()
