#!/usr/bin/env python3
# ============================================================================
# gen_bg1_tables.py
#
# One-shot generator: reads shift_table.bin (BG1, Z=384) and emits a C++ header
# `ldpc_bg1_tables.h` with all kernel-needed graph tables as constexpr arrays.
#
# Eliminates kernel GM args: degrees_gm, edge_offsets_gm, packed_bc_gm, packed_s_gm.
# Tables become compile-time constants in the kernel binary (.rodata).
#
# Usage:
#   python3 gen_bg1_tables.py \
#       /path/to/weights/ldpc_bg1_z384_shifts/shift_table.bin \
#       /path/to/kernels/rx/ldpc_decode/ldpc_bg1_tables.h
# ============================================================================
import sys
import numpy as np
from pathlib import Path

LDPC_MB        = 46
LDPC_NFULL     = 68   # KB(22) + MB(46)
LDPC_MAX_DEG   = 19

def emit_header(shift_path: Path, out_path: Path):
    sh = np.fromfile(shift_path, dtype=np.int16).reshape(LDPC_MB, LDPC_NFULL)

    # Derive deg, eoff
    deg  = np.array([int(np.sum(sh[br] != -1)) for br in range(LDPC_MB)],
                    dtype=np.int16)
    eoff = np.zeros(LDPC_MB + 1, dtype=np.int32)
    eoff[1:] = np.cumsum(deg)
    total_edges = int(deg.sum())
    max_deg     = int(deg.max())

    assert max_deg <= LDPC_MAX_DEG, f"max_deg {max_deg} > {LDPC_MAX_DEG}"

    # Build packed_bc, packed_s: row-major [br][kk], pad to MAX_DEG with 0
    packed_bc = np.zeros((LDPC_MB, LDPC_MAX_DEG), dtype=np.int16)
    packed_s  = np.zeros((LDPC_MB, LDPC_MAX_DEG), dtype=np.int16)
    for br in range(LDPC_MB):
        k = 0
        for bc_n in range(LDPC_NFULL):
            s = int(sh[br, bc_n])
            if s < 0:
                continue
            packed_bc[br, k] = bc_n
            packed_s [br, k] = s
            k += 1

    # Emit C++ header
    lines = []
    lines.append("// ============================================================================")
    lines.append("// ldpc_bg1_tables.h  — AUTO-GENERATED. Do not edit by hand.")
    lines.append("//")
    lines.append("// 5G NR LDPC BG1 (Z=384, MB=46, MAX_DEG=19) graph tables as constexpr.")
    lines.append("// Eliminates per-launch GM transfer of degrees/edge_offsets/packed_bc/packed_s.")
    lines.append("//")
    lines.append("// Regenerate with: python3 gen_bg1_tables.py shift_table.bin ldpc_bg1_tables.h")
    lines.append(f"// total_edges={total_edges}  max_deg={max_deg}")
    lines.append("// ============================================================================")
    lines.append("#pragma once")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace airan {")
    lines.append("")

    # BG1_DEG
    lines.append(f"// Variable-node degree per check row (br).")
    lines.append(f"constexpr int16_t BG1_DEG[{LDPC_MB}] = {{")
    for i in range(0, LDPC_MB, 16):
        chunk = deg[i:i+16]
        lines.append("    " + ", ".join(f"{int(v):3d}" for v in chunk) + ",")
    lines.append("};")
    lines.append("")

    # BG1_EOFF
    lines.append(f"// Cumulative edge offsets (length MB+1, monotonic).")
    lines.append(f"constexpr int32_t BG1_EOFF[{LDPC_MB + 1}] = {{")
    for i in range(0, LDPC_MB + 1, 16):
        chunk = eoff[i:i+16]
        lines.append("    " + ", ".join(f"{int(v):4d}" for v in chunk) + ",")
    lines.append("};")
    lines.append("")

    # BG1_PACKED_BC
    lines.append(f"// packed_bc[br][kk]: bit-column index (0..NFULL-1) of the kk-th edge of row br.")
    lines.append(f"// Padded with 0 for kk >= deg[br].")
    lines.append(f"constexpr int16_t BG1_PACKED_BC[{LDPC_MB}][{LDPC_MAX_DEG}] = {{")
    for br in range(LDPC_MB):
        row = packed_bc[br]
        lines.append("    { " + ", ".join(f"{int(v):2d}" for v in row) + " },")
    lines.append("};")
    lines.append("")

    # BG1_PACKED_S
    lines.append(f"// packed_s[br][kk]: cyclic shift (0..Z-1) for the kk-th edge of row br.")
    lines.append(f"// Padded with 0 for kk >= deg[br].")
    lines.append(f"constexpr int16_t BG1_PACKED_S[{LDPC_MB}][{LDPC_MAX_DEG}] = {{")
    for br in range(LDPC_MB):
        row = packed_s[br]
        lines.append("    { " + ", ".join(f"{int(v):3d}" for v in row) + " },")
    lines.append("};")
    lines.append("")

    lines.append("}  // namespace airan")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"[gen] wrote {out_path}")
    print(f"      total_edges={total_edges}  max_deg={max_deg}")
    print(f"      BG1_DEG = {LDPC_MB} × int16 = {LDPC_MB * 2} B")
    print(f"      BG1_EOFF = {LDPC_MB+1} × int32 = {(LDPC_MB+1) * 4} B")
    print(f"      BG1_PACKED_BC = {LDPC_MB}×{LDPC_MAX_DEG} × int16 = {LDPC_MB * LDPC_MAX_DEG * 2} B")
    print(f"      BG1_PACKED_S  = {LDPC_MB}×{LDPC_MAX_DEG} × int16 = {LDPC_MB * LDPC_MAX_DEG * 2} B")
    print(f"      Total .rodata: ~{(LDPC_MB*2 + (LDPC_MB+1)*4 + 2*LDPC_MB*LDPC_MAX_DEG*2)} B")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: gen_bg1_tables.py SHIFT_TABLE_BIN OUT_HEADER", file=sys.stderr)
        sys.exit(1)
    emit_header(Path(sys.argv[1]), Path(sys.argv[2]))
