#!/usr/bin/env python3
"""Compile a static radio profile plus one dynamic PUSCH grant.

The split authoring model is converted to the existing strictly validated
resolved plan and fixed runtime ABI. JSON remains a host/control-plane format;
AscendC kernels consume only the compiled binary configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from mimo_profile_compiler import (ProfileError, ResolvedPlan, compile_profile,
                                   evaluate_capabilities)


RADIO_KEYS = {"schema_version", "radio_name", "topology", "capabilities", "receiver"}
GRANT_KEYS = {"schema_version", "grant_name", "slot_number", "rank", "transport",
              "waveform", "dmrs", "precoding", "loopback_channel"}
STANDARD_PORTS = {1: 1, 2: 2, 3: 4, 4: 4}
STANDARD_DMRS_PORTS = {
    1: [1000],
    2: [1000, 1002],
    3: [1000, 1001, 1002],
    4: [1000, 1001, 1002, 1003],
}
CONFIG_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _load(path: Path, expected: set[str], kind: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"cannot read {kind} {path}: {error}") from error
    if not isinstance(value, dict) or set(value) != expected:
        raise ProfileError(f"{kind} must contain exactly {sorted(expected)}")
    if value.get("schema_version") != 1:
        raise ProfileError(f"{kind}.schema_version must be 1")
    return value, payload


def _object(value: Any, where: str, keys: set[str], optional: set[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - keys - optional or keys - set(value):
        raise ProfileError(f"{where} must contain exactly required={sorted(keys)} optional={sorted(optional)}")
    return value


def _integer(value: Any, where: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ProfileError(f"{where} must be an integer in [{low},{high}]")
    return value


def _number(value: Any, where: str, low: float, high: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{where} must be numeric")
    result = float(value)
    if result < low or result > high or (positive and result == 0):
        raise ProfileError(f"{where} must be in {'(' if positive else '['}{low},{high}]")
    return result


def _strict_ints(value: Any, where: str, low: int, high: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ProfileError(f"{where} must be a non-empty array")
    result = [_integer(item, f"{where}[{index}]", low, high)
              for index, item in enumerate(value)]
    if result != sorted(set(result)):
        raise ProfileError(f"{where} must be unique and strictly increasing")
    return result


def _validate_and_merge(radio: dict[str, Any], grant: dict[str, Any],
                        rank_override: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if (not isinstance(radio["radio_name"], str) or
            CONFIG_NAME.fullmatch(radio["radio_name"]) is None):
        raise ProfileError("radio.radio_name must match ^[a-z][a-z0-9_]{0,63}$")
    if (not isinstance(grant["grant_name"], str) or
            CONFIG_NAME.fullmatch(grant["grant_name"]) is None):
        raise ProfileError("grant.grant_name must match ^[a-z][a-z0-9_]{0,63}$")
    topology = _object(radio["topology"], "radio.topology", {
        "architecture", "tx_antennas", "tx_rf_chains", "rx_antennas", "rx_rf_chains"})
    capabilities = _object(radio["capabilities"], "radio.capabilities", {
        "supported_ranks", "pusch_port_policy", "antenna_mapping"})
    receiver = _object(radio["receiver"], "radio.receiver", {
        "channel_estimator", "detector", "max_rx_antennas", "rx_buckets",
        "storage_layers", "rx_capacity_adapter"})
    ranks = _strict_ints(capabilities["supported_ranks"],
                         "radio.capabilities.supported_ranks", 1, 4)
    if capabilities["pusch_port_policy"] != "nr_4port":
        raise ProfileError("only the standard nr_4port PUSCH port policy is implemented")
    if capabilities["antenna_mapping"] not in {"identity", "semi_unitary_dft"}:
        raise ProfileError("unsupported radio.capabilities.antenna_mapping")
    if receiver["storage_layers"] != 16:
        raise ProfileError("radio.receiver.storage_layers must be the internal value 16")
    rx_buckets = _strict_ints(receiver["rx_buckets"], "radio.receiver.rx_buckets", 1, 64)

    transport = _object(grant["transport"], "grant.transport", {
        "rnti", "data_scrambling_id", "codeword_index"})
    waveform = _object(grant["waveform"], "grant.waveform", {
        "qm", "num_symbols", "start_symbol", "fft_size", "num_rb", "rb_start",
        "used_subcarriers", "padded_subcarriers", "data_re_alignment",
        "qam_row_alignment"})
    dmrs = _object(grant["dmrs"], "grant.dmrs", {
        "symbols", "ports", "scrambling_id", "type", "length",
        "additional_position", "mapping_type", "num_cdm_groups_without_data", "n_scid"})
    precoding = _object(grant["precoding"], "grant.precoding", {"mode"}, {"tpmi", "prg_size_rb"})
    loopback = _object(grant["loopback_channel"], "grant.loopback_channel", {
        "model", "gain", "awgn_std_int16"})

    grant_rank = _integer(grant["rank"], "grant.rank", 1, 4)
    rank = _integer(rank_override if rank_override is not None else grant_rank,
                    "active rank", 1, 4)
    if rank not in ranks:
        raise ProfileError(f"Rank{rank} is not supported by radio {radio['radio_name']}")
    num_symbols = _integer(waveform["num_symbols"], "grant.waveform.num_symbols", 1, 14)
    start_symbol = _integer(waveform["start_symbol"], "grant.waveform.start_symbol", 0, 13)
    if start_symbol + num_symbols > 14:
        raise ProfileError("grant symbol allocation exceeds one 14-symbol slot")
    dmrs_symbols = _strict_ints(dmrs["symbols"], "grant.dmrs.symbols", 0, 13)
    if any(symbol < start_symbol or symbol >= start_symbol + num_symbols
           for symbol in dmrs_symbols):
        raise ProfileError("DMRS symbol lies outside the PUSCH symbol allocation")

    ports = dmrs["ports"]
    selected_ports = STANDARD_DMRS_PORTS[rank] if ports == "auto" else ports
    if not isinstance(selected_ports, list) or len(selected_ports) != rank or len(set(selected_ports)) != rank:
        raise ProfileError(f"grant.dmrs.ports must contain exactly {rank} unique ports")
    if any(_integer(port, "grant.dmrs.ports[]", 1000, 1011) < 1000
           for port in selected_ports):
        raise AssertionError("unreachable")
    if selected_ports != STANDARD_DMRS_PORTS[rank]:
        raise ProfileError(
            f"installed nr_4port policy requires Rank{rank} DMRS ports "
            f"{STANDARD_DMRS_PORTS[rank]}")

    mode = precoding["mode"]
    if mode not in {"auto", "bypass", "codebook", "non_codebook"}:
        raise ProfileError("unsupported grant.precoding.mode")
    resolved_mode = ("codebook" if rank == 3 else "bypass") if mode == "auto" else mode
    tpmi = _integer(precoding.get("tpmi", 6 if rank == 3 else 0),
                    "grant.precoding.tpmi", 0, 65535)
    prg_size = _integer(precoding.get("prg_size_rb", waveform["num_rb"]),
                        "grant.precoding.prg_size_rb", 1, 275)

    ports_by_rank = {str(item): STANDARD_PORTS[item] for item in ranks}
    dmrs_by_rank = {str(item): STANDARD_DMRS_PORTS[item] for item in ranks}
    dmrs_by_rank[str(rank)] = list(selected_ports)
    by_rank: dict[str, Any] = {}
    if resolved_mode != "bypass":
        by_rank[str(rank)] = {"mode": resolved_mode, "tpmi": tpmi,
                              "prg_size_rb": prg_size}

    max_rx = _integer(receiver["max_rx_antennas"], "radio.receiver.max_rx_antennas", 1, 64)
    if max_rx < topology["rx_antennas"]:
        raise ProfileError("radio receiver capacity is below physical RX antennas")
    legacy = {
        "schema_version": 1,
        "profile_name": radio["radio_name"],
        "topology": topology,
        "spatial": {
            "supported_ranks": ranks,
            "port_policy": {"mode": "explicit", "ports_by_rank": ports_by_rank},
            "precoding": {"default_mode": "bypass", "by_rank": by_rank},
            "antenna_mapping": capabilities["antenna_mapping"],
        },
        "waveform": {
            "qm": waveform["qm"], "num_symbols": num_symbols,
            "fft_size": waveform["fft_size"], "num_rb": waveform["num_rb"],
            "rb_start": waveform["rb_start"],
            "used_subcarriers": waveform["used_subcarriers"],
            "padded_subcarriers": waveform["padded_subcarriers"],
            "dmrs_symbols": dmrs_symbols,
            "data_re_alignment": waveform["data_re_alignment"],
            "qam_row_alignment": waveform["qam_row_alignment"],
        },
        "dmrs": {
            "ports_by_rank": dmrs_by_rank,
            "symbol_mask": sum(1 << symbol for symbol in dmrs_symbols),
            "type": dmrs["type"], "length": dmrs["length"],
            "num_cdm_groups_without_data": dmrs["num_cdm_groups_without_data"],
        },
        "channel": {
            "model": loopback["model"],
            "gain_by_rank": {str(item): _number(loopback["gain"],
                "grant.loopback_channel.gain", 0.0, 1.0, positive=True) for item in ranks},
            "awgn_std_int16": _number(loopback["awgn_std_int16"],
                "grant.loopback_channel.awgn_std_int16", 0.0, 200.0),
        },
        "receiver": {
            "channel_estimator": receiver["channel_estimator"],
            "detector": receiver["detector"],
            "capacity": {"max_rx_antennas": max_rx, "max_layers": 4},
            "buckets": {"rx_antennas": rx_buckets, "layers": [1, 2, 4]},
            "rx_capacity_adapter": receiver["rx_capacity_adapter"],
        },
    }
    scheduler = {
        "slot_number": _integer(grant["slot_number"], "grant.slot_number", 0, 1023),
        "start_symbol": start_symbol,
        "num_allocated_symbols": num_symbols,
        "rnti": _integer(transport["rnti"], "grant.transport.rnti", 1, 65535),
        "data_scrambling_id": _integer(transport["data_scrambling_id"],
            "grant.transport.data_scrambling_id", 0, 1023),
        "dmrs_scrambling_id": _integer(dmrs["scrambling_id"],
            "grant.dmrs.scrambling_id", 0, 65535),
        "codeword_index": _integer(transport["codeword_index"],
            "grant.transport.codeword_index", 0, 1),
        "dmrs_additional_position": _integer(dmrs["additional_position"],
            "grant.dmrs.additional_position", 0, 3),
        "mapping_type": _integer(dmrs["mapping_type"], "grant.dmrs.mapping_type", 0, 1),
        "n_scid": _integer(dmrs["n_scid"], "grant.dmrs.n_scid", 0, 1),
    }
    return legacy, scheduler


def compile_config(radio_path: Path, grant_path: Path, rank_override: int | None = None,
                   capabilities_path: Path | None = None) -> ResolvedPlan:
    radio, radio_bytes = _load(radio_path, RADIO_KEYS, "radio profile")
    grant, grant_bytes = _load(grant_path, GRANT_KEYS, "PUSCH grant")
    legacy, scheduler = _validate_and_merge(radio, grant, rank_override)
    canonical = json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    scheduler_rank = (rank_override if rank_override is not None
                      else _integer(grant["rank"], "grant.rank", 1, 4))
    with tempfile.TemporaryDirectory() as directory:
        merged_path = Path(directory) / "merged_profile.json"
        merged_path.write_bytes(canonical)
        plan = compile_profile(merged_path, scheduler_rank)
    value = plan.value
    combined_hash = hashlib.sha256(
        radio_bytes + b"\0" + grant_bytes + b"\0" + str(scheduler_rank).encode()).hexdigest()
    value["source"] = {
        "profile_name": radio["radio_name"],
        "profile_schema_version": 1,
        "radio_name": radio["radio_name"],
        "radio_path": str(radio_path.resolve()),
        "radio_sha256": hashlib.sha256(radio_bytes).hexdigest(),
        "grant_name": grant["grant_name"],
        "grant_path": str(grant_path.resolve()),
        "grant_sha256": hashlib.sha256(grant_bytes).hexdigest(),
        "active_rank_override": rank_override,
        "sha256": combined_hash,
    }
    value["scheduler"] = scheduler
    result = ResolvedPlan(value)
    return evaluate_capabilities(result, capabilities_path) if capabilities_path else result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio-profile", type=Path, required=True)
    parser.add_argument("--grant", type=Path, required=True)
    parser.add_argument("--rank", type=int, help="scheduler override for Rank1..4")
    parser.add_argument("--capabilities", type=Path)
    parser.add_argument("--require-eligible", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binary-output", type=Path)
    args = parser.parse_args()
    try:
        plan = compile_config(args.radio_profile, args.grant, args.rank, args.capabilities)
        if args.require_eligible:
            if args.capabilities is None:
                raise ProfileError("--require-eligible requires --capabilities")
            if plan.value["validation"]["execution_eligibility"] != "eligible":
                raise ProfileError("compiled configuration is not executable: " +
                    ", ".join(plan.value["validation"]["blocking_operators"]))
    except ProfileError as error:
        parser.error(str(error))
    if args.output is None and args.binary_output is None:
        print(plan.to_json(), end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(plan.to_json(), encoding="utf-8")
        print(f"[PASS] wrote {args.output}")
    if args.binary_output is not None:
        args.binary_output.parent.mkdir(parents=True, exist_ok=True)
        args.binary_output.write_bytes(plan.to_runtime_config())
        print(f"[PASS] wrote {args.binary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
