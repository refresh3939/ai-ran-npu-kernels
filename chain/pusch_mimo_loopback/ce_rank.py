#!/usr/bin/env python3
"""Link natural DMRS-LS observations to an explicit Rank1..4 CE case."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

WEIGHTS = ("factor_b_re.bin", "factor_b_im.bin", "cube_time_fused_d0_re.bin",
           "cube_time_fused_d0_im.bin", "cube_time_fused_d1_re.bin",
           "cube_time_fused_d1_im.bin", "ce_pack_gather_index.bin", "weight_model.bin")


def case(rank: int, nr: int, retained_rank: int) -> str:
    return f"case_0_m{nr}_k{rank}_r{retained_rank}"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path, rank: int, nr: int, retained_rank: int) -> None:
    golden = root / "golden" / case(rank, nr, retained_rank)
    manifest = {"schema": "airan.pusch_mimo.ce.weights.v1", "rank": rank,
                "profile": {"nr": nr, "layers": rank,
                            "ce_rank": retained_rank}, "weights": {}}
    for name in WEIGHTS:
        path = golden / name
        if not path.is_file() or path.stat().st_size == 0 or not any(path.read_bytes()):
            raise RuntimeError(f"missing/empty/all-zero CE weight: {path}")
        manifest["weights"][name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    (root / "weight_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def link(root: Path, ls_root: Path, slot: int, rank: int,
         nr: int, retained_rank: int) -> None:
    golden = root / "golden" / case(rank, nr, retained_rank)
    for source_name, target_name in (
            (f"slot{slot:02d}_h_re.bin", "h_ls_re.bin"),
            (f"slot{slot:02d}_h_im.bin", "h_ls_im.bin"),
            ("pilot_count.bin", "pilot_count.bin"), ("pilot_sc.bin", "pilot_sc.bin")):
        source, target = ls_root / source_name, golden / target_name
        if target.name.startswith("pilot_") and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"CE generated observation model disagrees with DMRS-LS: {target.name}")
        shutil.copyfile(source, target)
    expected = nr * rank * 2 * 832
    for name in ("h_ls_re.bin", "h_ls_im.bin"):
        value = np.fromfile(golden / name, np.float16)
        if value.size != expected or not np.any(value):
            raise RuntimeError(f"invalid actual Rank{rank} CE input {name}")


def collect(root: Path, output: Path, slot: int, rank: int,
            nr: int, retained_rank: int, layer_capacity: int) -> None:
    source = root / "ascend_output" / case(rank, nr, retained_rank)
    output.mkdir(parents=True, exist_ok=True)
    for plane in ("re", "im"):
        path = source / f"h_cube_time_fused_{plane}.bin"
        value = np.fromfile(path, np.float16)
        if value.size != nr * layer_capacity * 14 * 1664:
            raise RuntimeError("CE output physical shape mismatch")
        grid = value.reshape(nr, layer_capacity, 14, 1664)
        if not np.any(grid[:, :rank, :, :1596]):
            raise RuntimeError("CE active layers all-zero")
        if np.any(grid[:, rank:]):
            raise RuntimeError("CE inactive physical layers nonzero")
        shutil.copyfile(path, output / f"slot{slot:02d}_h_{plane}.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "link", "collect"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ls", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--rank", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--ce-rank", type=int, default=96)
    parser.add_argument("--layer-capacity", type=int, default=16)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.rank, args.nr, args.ce_rank)
    elif args.action == "link":
        link(args.root, args.ls, args.slot, args.rank, args.nr, args.ce_rank)
    else:
        collect(args.root, args.out, args.slot, args.rank, args.nr,
                args.ce_rank, args.layer_capacity)


if __name__ == "__main__":
    main()
