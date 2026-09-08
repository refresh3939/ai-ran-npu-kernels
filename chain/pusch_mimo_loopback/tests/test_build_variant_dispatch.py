#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


LOOPBACK = Path(__file__).resolve().parents[1]
KERNELS = LOOPBACK.parents[1]
COMMON = KERNELS / "common"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(LOOPBACK))

from mimo_profile_compiler import compile_profile, evaluate_capabilities
from resolve_build_variant import resolve


def main() -> None:
    profile = COMMON / "profiles/fd8x8_rank1_4.json"
    capabilities = COMMON / "mimo_operator_capabilities.json"
    for rank in range(1, 5):
        plan = evaluate_capabilities(compile_profile(profile, rank), capabilities)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(plan.to_json(), encoding="utf-8")
            build = resolve(path, rank)
        expected_rx = 16 if rank == 4 else 64
        expected_name = ("rx16_layer16_rank4" if rank == 4
                         else "rx64_layer16_rank1_4")
        assert build == {
            "name": expected_name,
            "ce_nr": expected_rx,
            "ce_nl": rank,
            "ce_retained_rank": 96,
            "ce_block_dim": 4,
            "ce_cube_time_fused": True,
            "detector_rx_capacity": expected_rx,
            "detector_layer_capacity": 16,
            "detector_bri_block": 2 if rank == 4 else 8,
        }
    print("[PASS] resolved plan selects and validates the installed build variant")


if __name__ == "__main__":
    main()
