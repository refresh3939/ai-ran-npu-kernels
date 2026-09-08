#!/usr/bin/env python3
"""Publish a compact, promotion-safe receipt from one completed device run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGES = [
    "ldpc_encode", "rate_match_mimo", "scramble_mimo", "qam_mod_256_mimo",
    "layer_map_mimo", "mimo_dmrs_gen", "mimo_resource_grid_map",
    "pusch_codebook_precode", "re_map_batch", "ofdm_mod_batch",
    "ofdm_demod_batch", "re_demap_batch", "mimo_dmrs_ls",
    "channel_est_lmmse_mimo", "mimo_detect_io_pack", "mimo_detect_bri_batch",
    "qam_demod_256_mimo_batch", "layer_demap_mimo", "descramble_mimo",
    "rate_dematch_mimo", "ldpc_decode",
]


def load_pass(path: Path, rank: int) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[HARD_FAIL] invalid evidence input {path}: {exc}") from exc
    if value.get("status") != "PASS" or value.get("rank") != rank or value.get("slots") != 23:
        raise SystemExit(f"[HARD_FAIL] rejected evidence input {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.root / "artifacts"
    rank = args.rank
    suffix = "" if rank == 1 else f"_rank{rank}"
    triangle = load_pass(artifacts / f"rank{rank}_triangle_result.json", rank)
    waveform = load_pass(artifacts / f"coded_waveform{suffix}_result.json", rank)
    front_path = (artifacts / "demod_23slot/result.json" if rank == 1 else
                  artifacts / f"demod_rank{rank}_23slot/result.json")
    front = load_pass(front_path, rank)
    if triangle.get("matched_info_ber") != 0.0:
        raise SystemExit("[HARD_FAIL] matched information BER is not zero")
    wrong = triangle.get("wrong_cell_info_ber")
    if not isinstance(wrong, (int, float)) or not 0.4 <= wrong <= 0.6:
        raise SystemExit("[HARD_FAIL] wrong-cell information BER is outside [0.4,0.6]")
    if triangle.get("tx_bits_nonzero", 0) <= 0:
        raise SystemExit("[HARD_FAIL] TX information bits are all zero")
    if rank == 1:
        prefix = load_pass(artifacts / "tx_prefix_result.json", rank)
        for required in ("dmrs_result.json", "grid_map_result.log", "precode_result.log",
                         "dmrs_ls_result.log"):
            path = artifacts / required
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"[HARD_FAIL] missing Rank1 stage receipt {path}")
    else:
        prefix = load_pass(artifacts / f"tx_prefix_rank{rank}_result.json", rank)
        load_pass(artifacts / f"tx_grid_rank{rank}_result.json", rank)
        load_pass(artifacts / f"dmrs_ls_rank{rank}/result.json", rank)
    payload = {
        "schema": "airan.pusch_mimo.e2e_evidence.v1",
        "status": "PASS", "rank": rank, "slots": 23,
        "tx_bits_nonzero": triangle["tx_bits_nonzero"],
        "tx_bits_sha256": triangle["tx_bits_sha256"],
        "precode_mode": "codebook_tpmi6_prg4" if rank == 3 else "bypass",
        "waveform_nrmse": waveform.get("worst_grid_nrmse"),
        "raw_qam_ber": front["raw_qam_ber"],
        "matched_info_ber": triangle["matched_info_ber"],
        "wrong_cell_info_ber": wrong,
        "complete_coded_e2e": True,
        "actual_npu_stages": STAGES,
        "radio_profile": waveform.get("radio_profile", "legacy"),
        "radio_topology": {
            "tx_antennas": waveform.get("tx_antennas", waveform.get("tx_ports")),
            "rx_antennas": waveform.get("rx_antennas"),
            "detector_rx_capacity": waveform.get("detector_rx_capacity"),
        },
        "source_receipts": {
            "tx_prefix_schema": prefix.get("schema", prefix.get("schema_version")),
            "waveform_schema": waveform.get("schema", waveform.get("schema_version")),
            "rx_front_schema": front.get("schema", front.get("schema_version")),
            "triangle_schema": triangle.get("schema", triangle.get("schema_version")),
        },
    }
    host_metrics = waveform.get("host_channel_metrics")
    if host_metrics is not None:
        if host_metrics.get("int16_saturation_count") != 0:
            raise SystemExit("[HARD_FAIL] host channel int16 saturation")
        payload["host_channel_metrics"] = host_metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
