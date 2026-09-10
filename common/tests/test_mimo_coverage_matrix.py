#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))

from mimo_coverage_matrix import build_matrix


def main() -> None:
    rows = build_matrix(COMMON / "profiles/fd8x8_rank1_4.json",
                        COMMON / "mimo_operator_capabilities.json")
    assert len(rows) == 89
    eligible = [row for row in rows if row["eligibility"] == "eligible"]
    assert len(eligible) == len(rows)
    edge = next(row for row in rows
                if (row["tx"], row["rx"], row["rank"]) == (16, 64, 4))
    assert edge["requested_bucket"] == [64, 4]
    assert edge["execution_bucket"] == [64, 16]
    assert edge["variant"] == "rx64_storage16_rank1_4"
    print("[PASS] common 1..16TX x 1..64RX Rank1-4 coverage matrix")


if __name__ == "__main__":
    main()
