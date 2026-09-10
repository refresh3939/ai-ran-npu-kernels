#!/usr/bin/env python3
"""Fail-closed verification for a fused 16TX x 64RX Rank4 receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGES = (
    "ofdm_demod", "re_demap", "dmrs_gen", "dmrs_ls", "lmmse_ce",
    "io_pack_grouped", "bri_detect", "qam_demod", "layer_demap",
    "descramble", "rate_dematch", "ldpc_decode", "slot_total", "tb_tail",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("bits", type=Path)
    args = parser.parse_args()
    value = json.loads(args.receipt.read_text(encoding="utf-8"))

    expected = {
        "schema": "airan.pusch_mimo.fused_rx.v1",
        "status": "PASS",
        "rank": 4,
        "tx_antennas": 16,
        "rx_antennas": 64,
        "slots": 23,
        "single_acl_context": True,
        "single_stream": True,
        "intermediate_d2h": False,
        "grouped_rhs_on_device": True,
        "bit_errors": 0,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeError(f"receipt {key}={value.get(key)!r}, expected {wanted!r}")
    if args.bits.stat().st_size != 143 * 8448:
        raise RuntimeError("decoded bit output has the wrong exact byte size")
    if not isinstance(value.get("wall_ms"), (int, float)) or value["wall_ms"] <= 0:
        raise RuntimeError("wall_ms is missing or non-positive")
    timings = value.get("p50_us", {})
    for stage in STAGES:
        if not isinstance(timings.get(stage), (int, float)) or timings[stage] < 0:
            raise RuntimeError(f"missing/invalid timing for {stage}")
    print("PUSCH_MIMO_FUSED_RX_NPU PASS: 16TX x 64RX Rank4, 23 slots, LDPC BER=0")


if __name__ == "__main__":
    main()
