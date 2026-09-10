#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))

from mimo_profile_compiler import ResolvedPlan, compile_profile, evaluate_capabilities
from mimo_profile_generator import RX_CHOICES, TX_CHOICES, build_profile


def main() -> None:
    capabilities = COMMON / "mimo_operator_capabilities.json"
    for tx in TX_CHOICES:
        for rx in RX_CHOICES:
            value = build_profile(tx, rx)
            ranks = value["spatial"]["supported_ranks"]
            assert ranks == list(range(1, min(4, tx, rx) + 1))
            for rank in ranks:
                # The generator is deterministic and compiler-equivalent; use
                # the resolved content path exercised by the coverage sweep.
                import json
                import tempfile
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "profile.json"
                    path.write_text(json.dumps(value), encoding="utf-8")
                    plan = compile_profile(path, rank)
                evaluated = evaluate_capabilities(ResolvedPlan(plan.value), capabilities)
                assert evaluated.value["validation"]["execution_eligibility"] == "eligible"
    print("[PASS] deterministic common-profile generator covers 1..16TX x 1..64RX")


if __name__ == "__main__":
    main()
