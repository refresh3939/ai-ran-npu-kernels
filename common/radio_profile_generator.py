#!/usr/bin/env python3
"""Generate a static full-digital MIMO radio profile.

The generated file describes installed radio capability only.  Per-slot Rank,
RNTI, scrambling, DMRS and resource allocation belong in a PUSCH grant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def supported_ranks(tx: int, rx: int) -> list[int]:
    if not 1 <= tx <= 16:
        raise ValueError("TX antennas must be in [1,16]")
    if not 1 <= rx <= 64:
        raise ValueError("RX antennas must be in [1,64]")
    # The installed standard NR policy uses four logical ports for Rank3.
    max_rank = min(4, rx, tx if tx != 3 else 2)
    return list(range(1, max_rank + 1))


def build_radio_profile(tx: int, rx: int) -> dict:
    return {
        "schema_version": 1,
        "radio_name": f"fd{tx}x{rx}",
        "topology": {
            "architecture": "full_digital",
            "tx_antennas": tx,
            "tx_rf_chains": tx,
            "rx_antennas": rx,
            "rx_rf_chains": rx,
        },
        "capabilities": {
            "supported_ranks": supported_ranks(tx, rx),
            "pusch_port_policy": "nr_4port",
            "antenna_mapping": "semi_unitary_dft",
        },
        "receiver": {
            "channel_estimator": "lmmse",
            "detector": "bri",
            "max_rx_antennas": 64,
            "rx_buckets": [8, 16, 32, 64],
            "storage_layers": 16,
            "rx_capacity_adapter": "zero_pad_signal_repeat_noise",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tx", required=True, type=int)
    parser.add_argument("--rx", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        value = build_radio_profile(args.tx, args.rx)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
