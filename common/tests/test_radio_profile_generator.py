#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))

from radio_profile_generator import build_radio_profile


def main() -> None:
    assert build_radio_profile(16, 64)["capabilities"]["supported_ranks"] == [1, 2, 3, 4]
    assert build_radio_profile(8, 8)["radio_name"] == "fd8x8"
    assert build_radio_profile(3, 64)["capabilities"]["supported_ranks"] == [1, 2]
    assert build_radio_profile(4, 3)["capabilities"]["supported_ranks"] == [1, 2, 3]
    for tx, rx in ((0, 8), (17, 8), (8, 0), (8, 65)):
        try:
            build_radio_profile(tx, rx)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid topology accepted: {tx}TX x {rx}RX")
    print("[PASS] static radio generator covers arbitrary 1..16TX x 1..64RX")


if __name__ == "__main__":
    main()
