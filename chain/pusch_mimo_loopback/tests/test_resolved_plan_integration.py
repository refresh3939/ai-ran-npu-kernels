#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


LOOPBACK = Path(__file__).resolve().parents[1]
KERNELS = LOOPBACK.parents[1]
COMMON = KERNELS / "common"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(LOOPBACK))

from mimo_profile_compiler import compile_profile
import radio_profile as radio


def main() -> None:
    source = COMMON / "profiles/fd8x8_rank1_4.json"
    profile = radio.load_profile("fd8x8_rank1_4")
    with tempfile.TemporaryDirectory() as directory:
        plan_path = Path(directory) / "rank4.json"
        plan_path.write_text(compile_profile(source, 4).to_json())
        os.environ[radio.RESOLVED_PLAN_ENV] = str(plan_path)
        tx = np.zeros((1, 4, 16, 2), np.int16)
        tx[..., 0] = 100
        tx_ant, rx, physical, effective = radio.apply_fd8x8_channel(
            tx, profile, 4, profile.channel_gain(4), 0.0, 123)
        assert tx_ant.shape == (1, 8, 16, 2)
        assert rx.shape == (1, 8, 16, 2)
        assert physical.shape == (8, 8)
        assert effective.shape == (8, 4)

        bad = json.loads(plan_path.read_text())
        bad["active"]["rank"] = 3
        plan_path.write_text(json.dumps(bad))
        try:
            radio.apply_fd8x8_channel(tx, profile, 4, profile.channel_gain(4), 0.0, 123)
        except ValueError as error:
            assert "does not match" in str(error)
        else:
            raise AssertionError("mismatched ResolvedPlan must fail closed")
    os.environ.pop(radio.RESOLVED_PLAN_ENV, None)
    print("[PASS] loopback consumes and validates common ResolvedPlan")


if __name__ == "__main__":
    main()
