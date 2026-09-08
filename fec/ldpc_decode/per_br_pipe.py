#!/usr/bin/env python3
"""Per-br breakdown: how many br are 'pure V' (no MTE) vs mixed."""
import numpy as np
from pathlib import Path

sh = np.fromfile(Path.home() / "AI-RAN-NPU/weights/ldpc_bg1_z384_shifts/shift_table.bin",
                 dtype=np.int16).reshape(46, 68)

pure_v = []  # no sh==0 and no sh%16==0
mixed = []
pure_mte = []  # all sh%16==0
for br in range(46):
    valid = sh[br][sh[br] >= 0]
    sh0 = (valid == 0).sum()
    sh16 = ((valid != 0) & (valid % 16 == 0)).sum()
    sh_un = (valid % 16 != 0).sum()
    has_mte = (sh0 + sh16) > 0
    has_v = sh_un > 0
    if has_v and not has_mte:
        pure_v.append(br)
    elif has_mte and not has_v:
        pure_mte.append(br)
    else:
        mixed.append((br, sh0+sh16, sh_un))

print(f"Pure V br (only Adds, no MTE3 DataCopy): {len(pure_v)} → {pure_v}")
print()
print(f"Pure MTE br (only DataCopy, no Adds): {len(pure_mte)} → {pure_mte}")
print()
print(f"Mixed br (both): {len(mixed)}")
for br, mte, v in mixed[:5]:
    print(f"  br {br}: {mte} MTE + {v} V")
print(f"  ... (total {len(mixed)} mixed)")
