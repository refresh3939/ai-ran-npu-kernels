#!/usr/bin/env python3
"""Validate the fused MIMO TX execution receipt and final IQ artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PORTS_BY_RANK = {1: 1, 2: 2, 3: 4, 4: 4}
SAMPLES_PER_SLOT = 30720


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--iq", required=True, type=Path)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    rank = int(receipt["rank"])
    slots = int(receipt["slots"])
    ports = int(receipt["tx_ports"])
    expected_bytes = slots * ports * SAMPLES_PER_SLOT * 2 * 2

    assert receipt["schema"] == "airan.pusch_mimo.tx_chain.v1"
    assert receipt["status"] == "PASS"
    assert 1 <= rank <= 4 and ports == PORTS_BY_RANK[rank]
    assert 1 <= slots <= 23
    assert receipt["output_bytes"] == expected_bytes
    assert receipt["single_acl_context"] is True
    assert receipt["shared_device_arena"] is True
    assert receipt["intermediate_host_files"] is False
    assert args.iq.stat().st_size == expected_bytes

    iq = np.fromfile(args.iq, dtype=np.int16)
    nonzero = int(np.count_nonzero(iq))
    assert nonzero == int(receipt["nonzero_i16"])
    assert nonzero > iq.size // 10, "pathological sparse/all-zero waveform"
    saturated = int(np.count_nonzero((iq == -32768) | (iq == 32767)))
    assert saturated < iq.size // 100, "more than 1% of IQ lanes saturated"

    print(
        f"[PASS] fused TX Rank{rank} ports={ports} slots={slots} "
        f"bytes={expected_bytes} nonzero={nonzero} saturated={saturated}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
