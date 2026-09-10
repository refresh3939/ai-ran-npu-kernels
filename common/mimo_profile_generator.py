#!/usr/bin/env python3
"""Generate standard single-PUSCH full-digital profiles for common scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TX_CHOICES = (1, 2, 4, 8, 16)
RX_CHOICES = (1, 2, 4, 8, 16, 32, 64)


def build_profile(tx: int, rx: int) -> dict:
    if tx not in TX_CHOICES or rx not in RX_CHOICES:
        raise ValueError(f"TX must be {TX_CHOICES}; RX must be {RX_CHOICES}")
    max_rank = min(4, tx, rx)
    ranks = list(range(1, max_rank + 1))
    ports = {str(rank): (4 if rank == 3 else rank) for rank in ranks}
    precoding_by_rank = {}
    if 3 in ranks:
        precoding_by_rank["3"] = {
            "mode": "codebook", "tpmi": 6, "prg_size_rb": 4}
    return {
        "schema_version": 1,
        "profile_name": f"fd{tx}x{rx}_rank1_{max_rank}",
        "topology": {
            "architecture": "full_digital",
            "tx_antennas": tx,
            "tx_rf_chains": tx,
            "rx_antennas": rx,
            "rx_rf_chains": rx,
        },
        "spatial": {
            "supported_ranks": ranks,
            "port_policy": {"mode": "explicit", "ports_by_rank": ports},
            "precoding": {
                "default_mode": "bypass", "by_rank": precoding_by_rank},
            "antenna_mapping": "semi_unitary_dft",
        },
        "waveform": {
            "qm": 8, "num_symbols": 14, "fft_size": 2048,
            "num_rb": 133, "rb_start": 0, "used_subcarriers": 1596,
            "padded_subcarriers": 1664, "dmrs_symbols": [2, 11],
            "data_re_alignment": 64, "qam_row_alignment": 64,
        },
        "dmrs": {
            "ports_by_rank": {
                str(rank): ([1000, 1002] if rank == 2 else
                            list(range(1000, 1000 + rank)))
                for rank in ranks
            },
            "symbol_mask": 2052, "type": 1, "length": 1,
            "num_cdm_groups_without_data": 2,
        },
        "channel": {
            "model": "deterministic_unitary",
            "gain_by_rank": {str(rank): 0.16 for rank in ranks},
            "awgn_std_int16": 80,
        },
        "receiver": {
            "channel_estimator": "lmmse", "detector": "bri",
            "capacity": {"max_rx_antennas": 64, "max_layers": 4},
            "buckets": {
                "rx_antennas": [8, 16, 32, 64],
                "layers": [1, 2, 4],
            },
            "rx_capacity_adapter": "zero_pad_signal_repeat_noise",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tx", required=True, type=int, choices=TX_CHOICES)
    parser.add_argument("--rx", required=True, type=int, choices=RX_CHOICES)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_profile(args.tx, args.rx), indent=2) + "\n",
                           encoding="utf-8")
    print(f"[PASS] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
