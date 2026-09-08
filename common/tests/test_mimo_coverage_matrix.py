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
    assert len(rows) == 23
    eligible = [row for row in rows if row["eligibility"] == "eligible"]
    assert [(row["tx"], row["rx"], row["rank"]) for row in eligible] == [
        (8, 8, 1), (8, 8, 2), (8, 8, 3), (8, 8, 4)]
    assert [row["execution_bucket"] for row in eligible] == [
        [64, 16], [64, 16], [64, 16], [16, 16]]
    rank16 = next(row for row in rows
                  if row["tx"] == 64 and row["rank"] == 16)
    assert rank16["requested_bucket"] == [64, 16]
    assert rank16["variant"] is None
    assert "execution_variant" in rank16["blockers"]
    print("[PASS] MIMO shape coverage matrix is explicit and fail-closed")


if __name__ == "__main__":
    main()
