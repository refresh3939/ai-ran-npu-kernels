#!/usr/bin/env python3
"""Validate a declarative MIMO profile and compile a per-Rank resolved plan.

This module is intentionally host-only. NPU stages consume the fixed-width
runtime ABI and operator-specific tiling data, never JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_TX_ANTENNAS = 16
MAX_RX_ANTENNAS = 64
MAX_PUSCH_LAYERS = 4
MAX_LOGICAL_PORTS = 4
# Backward-compatible public name used by receiver bucket helpers.
MAX_ANTENNAS = MAX_RX_ANTENNAS
# Public/profile capacity is the standard single-PUSCH Rank limit.  The BRI
# implementation currently pads that axis to a 16-column Cube storage tile.
MAX_LAYERS = MAX_PUSCH_LAYERS
MAX_STORAGE_LAYERS = 16
VALID_QM = {2, 4, 6, 8}
VALID_ARCHITECTURES = {"full_digital", "hybrid", "analog"}
VALID_PRECODING = {"bypass", "codebook", "non_codebook"}
PROFILE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
RUNTIME_CONFIG_FORMAT = "<HHI32s" + "H" * 44 + "ff" + "I" * 4 + "H" * 10 + "I" * 5
RUNTIME_CONFIG_SIZE = struct.calcsize(RUNTIME_CONFIG_FORMAT)
assert RUNTIME_CONFIG_SIZE == 192
ARCHITECTURE_ID = {"full_digital": 1, "hybrid": 2, "analog": 3}
PRECODING_ID = {"bypass": 0, "codebook": 1, "non_codebook": 2}


class ProfileError(ValueError):
    """A fail-closed profile validation or planning error."""


@dataclass(frozen=True)
class ResolvedPlan:
    value: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.value, indent=2, sort_keys=True) + "\n"

    def to_runtime_config(self) -> bytes:
        value = self.value
        active = value["active"]
        topology = value["topology"]
        spatial = value["spatial"]
        waveform = value["waveform"]
        dmrs = value["dmrs"]
        channel = value["channel"]
        receiver = value["receiver"]
        scheduler = value.get("scheduler", {
            "slot_number": 0, "dmrs_scrambling_id": 0,
            "data_scrambling_id": 0, "rnti": 1, "start_symbol": 0,
            "num_allocated_symbols": waveform["num_symbols"],
            "dmrs_additional_position": 0, "mapping_type": 0,
            "codeword_index": 0, "n_scid": 0,
        })
        digest = bytes.fromhex(value["source"]["sha256"])
        ports = list(dmrs["ports"])
        if len(ports) > MAX_STORAGE_LAYERS:
            raise ProfileError("resolved DMRS port count exceeds runtime ABI capacity")
        ports.extend([0] * (MAX_STORAGE_LAYERS - len(ports)))
        flags = 1
        if receiver["adapter_required"]:
            flags |= 2
        halfwords = [
            ARCHITECTURE_ID[topology["architecture"]],
            topology["tx_antennas"], topology["tx_rf_chains"],
            topology["rx_antennas"], topology["rx_rf_chains"],
            active["rank"], active["logical_tx_ports"],
            receiver["rx_bucket"], receiver["layer_bucket"],
            receiver["max_rx_antennas"], receiver["max_layers"],
            waveform["qm"], waveform["num_symbols"], waveform["fft_size"],
            waveform["num_rb"], waveform["rb_start"],
            waveform["used_subcarriers"], waveform["padded_subcarriers"],
            waveform["num_dmrs_symbols"], dmrs["symbol_mask"], len(dmrs["ports"]),
            *ports,
            dmrs["type"], dmrs["length"], dmrs["num_cdm_groups_without_data"],
            scheduler["n_scid"],
            PRECODING_ID[spatial["precoding"]["mode"]],
            spatial["precoding"]["tpmi"], spatial["precoding"]["prg_size_rb"],
        ]
        if len(halfwords) != 44:
            raise AssertionError("runtime ABI halfword field count changed")
        return struct.pack(
            RUNTIME_CONFIG_FORMAT,
            2, RUNTIME_CONFIG_SIZE, flags, digest, *halfwords,
            channel["gain"], channel["awgn_std_int16"],
            waveform["data_re_per_layer"], waveform["data_stride_per_layer"],
            waveform["qam_row_stride"], waveform["grid_re_per_port"],
            scheduler["slot_number"], scheduler["dmrs_scrambling_id"],
            scheduler["data_scrambling_id"], scheduler["rnti"],
            scheduler["start_symbol"], scheduler["num_allocated_symbols"],
            scheduler["dmrs_additional_position"], scheduler["mapping_type"],
            scheduler["codeword_index"], 0, *([0] * 5),
        )


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfileError(f"{where} must be an object")
    return value


def _integer(value: Any, where: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ProfileError(f"{where} must be an integer in [{low},{high}]")
    return value


def _number(value: Any, where: str, low: float, high: float,
            *, exclusive_low: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{where} must be numeric")
    result = float(value)
    lower_ok = result > low if exclusive_low else result >= low
    if not lower_ok or result > high:
        bracket = "(" if exclusive_low else "["
        raise ProfileError(f"{where} must be in {bracket}{low},{high}]")
    return result


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileError(f"{where} must be a non-empty string")
    return value


def _keys(value: dict[str, Any], where: str, required: set[str],
          optional: set[str] = frozenset()) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise ProfileError(f"{where} missing fields {sorted(missing)}")
    if unknown:
        raise ProfileError(f"{where} has unknown fields {sorted(unknown)}")


def _rank_map(value: Any, where: str, ranks: tuple[int, ...],
              value_parser: Any) -> dict[int, Any]:
    raw = _object(value, where)
    result: dict[int, Any] = {}
    for key, item in raw.items():
        if not isinstance(key, str) or not key.isdigit():
            raise ProfileError(f"{where} keys must be decimal Rank strings")
        rank = int(key)
        if rank in result:
            raise ProfileError(f"{where} duplicates Rank{rank}")
        result[rank] = value_parser(item, f"{where}.{key}")
    if set(result) != set(ranks):
        raise ProfileError(f"{where} must define exactly ranks {list(ranks)}")
    return result


def _strict_increasing(values: Any, where: str, low: int, high: int) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ProfileError(f"{where} must be a non-empty array")
    parsed = tuple(_integer(value, f"{where}[{index}]", low, high)
                   for index, value in enumerate(values))
    if tuple(sorted(set(parsed))) != parsed:
        raise ProfileError(f"{where} must be unique and strictly increasing")
    return parsed


def _unique_ints(values: Any, where: str, low: int, high: int) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ProfileError(f"{where} must be a non-empty array")
    parsed = tuple(_integer(value, f"{where}[{index}]", low, high)
                   for index, value in enumerate(values))
    if len(set(parsed)) != len(parsed):
        raise ProfileError(f"{where} must contain unique values")
    return parsed


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _smallest_bucket(value: int, buckets: tuple[int, ...], where: str) -> int:
    try:
        return next(bucket for bucket in buckets if bucket >= value)
    except StopIteration as error:
        raise ProfileError(f"{where}={value} exceeds bucket capacity {buckets[-1]}") from error


def _load_document(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except OSError as error:
        raise ProfileError(f"cannot read profile {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ProfileError(f"invalid JSON in {path}: {error}") from error
    return _object(value, "profile"), hashlib.sha256(payload).hexdigest()


def _plan_value(plan: dict[str, Any], dotted_path: str) -> Any:
    value: Any = plan
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ProfileError(
                f"capability constraint references missing plan field {dotted_path!r}")
        value = value[component]
    return value


def _constraint_failures(plan: dict[str, Any], constraints: Any,
                         where: str) -> list[str]:
    if not isinstance(constraints, list):
        raise ProfileError(f"{where} must be an array")
    failures: list[str] = []
    for index, raw_constraint in enumerate(constraints):
        constraint_where = f"{where}[{index}]"
        constraint = _object(raw_constraint, constraint_where)
        _keys(constraint, constraint_where, {"path"},
              {"allowed", "equals", "minimum", "maximum"})
        dotted_path = _string(constraint["path"], f"{constraint_where}.path")
        predicates = set(constraint) - {"path"}
        if len(predicates) != 1:
            raise ProfileError(
                f"{constraint_where} must define exactly one predicate")
        actual = _plan_value(plan, dotted_path)
        if "allowed" in constraint:
            allowed = constraint["allowed"]
            if not isinstance(allowed, list) or not allowed:
                raise ProfileError(f"{constraint_where}.allowed must be non-empty")
            if actual not in allowed:
                failures.append(f"{dotted_path}={actual!r} not in {allowed!r}")
        elif "equals" in constraint:
            if actual != constraint["equals"]:
                failures.append(
                    f"{dotted_path}={actual!r} != {constraint['equals']!r}")
        elif "minimum" in constraint:
            limit = constraint["minimum"]
            if isinstance(limit, bool) or not isinstance(limit, (int, float)):
                raise ProfileError(f"{constraint_where}.minimum must be numeric")
            if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or
                    actual < limit):
                failures.append(f"{dotted_path}={actual!r} below {limit!r}")
        elif "maximum" in constraint:
            limit = constraint["maximum"]
            if isinstance(limit, bool) or not isinstance(limit, (int, float)):
                raise ProfileError(f"{constraint_where}.maximum must be numeric")
            if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or
                    actual > limit):
                failures.append(f"{dotted_path}={actual!r} above {limit!r}")
    return failures


def evaluate_capabilities(plan: ResolvedPlan, path: Path) -> ResolvedPlan:
    """Evaluate a valid plan against one installed execution target registry."""
    try:
        payload = path.read_bytes()
        registry = _object(json.loads(payload), "capability_registry")
    except OSError as error:
        raise ProfileError(f"cannot read capability registry {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ProfileError(f"invalid capability registry JSON in {path}: {error}") from error
    _keys(registry, "capability_registry", {
        "schema_version", "registry_name", "target", "variants", "operators"
    })
    if registry["schema_version"] != 1:
        raise ProfileError("capability_registry.schema_version must be 1")
    registry_name = _string(registry["registry_name"],
                            "capability_registry.registry_name")
    target = _string(registry["target"], "capability_registry.target")
    operators = registry["operators"]
    if not isinstance(operators, list) or not operators:
        raise ProfileError("capability_registry.operators must be a non-empty array")

    value = dict(plan.value)
    value["receiver"] = dict(plan.value["receiver"])
    variants = registry["variants"]
    if not isinstance(variants, list) or not variants:
        raise ProfileError("capability_registry.variants must be a non-empty array")
    variant_checks: list[dict[str, Any]] = []
    selected_variant: dict[str, Any] | None = None
    variant_names: set[str] = set()
    for index, raw_variant in enumerate(variants):
        where = f"capability_registry.variants[{index}]"
        variant = _object(raw_variant, where)
        _keys(variant, where, {"name", "constraints", "runtime", "build"})
        name = _string(variant["name"], f"{where}.name")
        if name in variant_names:
            raise ProfileError(f"duplicate capability variant {name!r}")
        variant_names.add(name)
        failures = _constraint_failures(value, variant["constraints"],
                                        f"{where}.constraints")
        runtime = _object(variant["runtime"], f"{where}.runtime")
        _keys(runtime, f"{where}.runtime", {"rx_bucket", "layer_bucket"})
        rx_bucket = _integer(runtime["rx_bucket"],
                             f"{where}.runtime.rx_bucket", 1, MAX_ANTENNAS)
        layer_bucket = _integer(runtime["layer_bucket"],
                                f"{where}.runtime.layer_bucket", 1,
                                MAX_STORAGE_LAYERS)
        if (rx_bucket < value["topology"]["rx_antennas"] or
                rx_bucket > value["receiver"]["max_rx_antennas"]):
            failures.append("runtime.rx_bucket is outside active/capacity bounds")
        if layer_bucket < value["active"]["rank"]:
            failures.append("runtime.layer_bucket is below the active PUSCH Rank")
        build = _object(variant["build"], f"{where}.build")
        _keys(build, f"{where}.build", {
            "ce_nr", "ce_nl_source", "ce_retained_rank", "ce_block_dim",
            "ce_cube_time_fused", "detector_rx_capacity",
            "detector_layer_capacity", "detector_bri_block",
        })
        ce_nr = _integer(build["ce_nr"], f"{where}.build.ce_nr", 1, MAX_ANTENNAS)
        ce_nl_source = _string(build["ce_nl_source"],
                               f"{where}.build.ce_nl_source")
        if ce_nl_source != "active.rank":
            raise ProfileError(f"{where}.build.ce_nl_source must be active.rank")
        ce_retained_rank = _integer(build["ce_retained_rank"],
                                    f"{where}.build.ce_retained_rank", 16, 128)
        if ce_retained_rank % 16:
            raise ProfileError(f"{where}.build.ce_retained_rank must be a multiple of 16")
        ce_block_dim = _integer(build["ce_block_dim"],
                                f"{where}.build.ce_block_dim", 1, 4)
        if ce_block_dim not in {1, 2, 4}:
            raise ProfileError(f"{where}.build.ce_block_dim must be 1, 2, or 4")
        ce_cube_time_fused = build["ce_cube_time_fused"]
        if not isinstance(ce_cube_time_fused, bool):
            raise ProfileError(f"{where}.build.ce_cube_time_fused must be boolean")
        detector_rx = _integer(build["detector_rx_capacity"],
                               f"{where}.build.detector_rx_capacity", 1, MAX_ANTENNAS)
        detector_layers = _integer(build["detector_layer_capacity"],
                                   f"{where}.build.detector_layer_capacity", 1,
                                   MAX_STORAGE_LAYERS)
        detector_bri_block = _integer(build["detector_bri_block"],
                                      f"{where}.build.detector_bri_block", 1,
                                      MAX_STORAGE_LAYERS)
        if ce_nr != rx_bucket or detector_rx != rx_bucket:
            failures.append("build RX capacities do not match runtime.rx_bucket")
        if detector_layers != layer_bucket:
            failures.append("build detector layer capacity does not match runtime.layer_bucket")
        if detector_layers % detector_bri_block:
            failures.append("detector layer capacity is not divisible by BRI block")
        active_rank = value["active"]["rank"]
        if rx_bucket < 8 * min(detector_bri_block, active_rank):
            failures.append("detector variant violates NR >= 8*min(BRI_B, active Rank)")
        if 2 * detector_bri_block < active_rank:
            failures.append("detector variant violates 2*BRI_B >= active Rank")
        normalized_build = {
            "ce_nr": ce_nr,
            "ce_nl_source": ce_nl_source,
            "ce_retained_rank": ce_retained_rank,
            "ce_block_dim": ce_block_dim,
            "ce_cube_time_fused": ce_cube_time_fused,
            "detector_rx_capacity": detector_rx,
            "detector_layer_capacity": detector_layers,
            "detector_bri_block": detector_bri_block,
        }
        variant_checks.append({
            "variant": name,
            "status": "match" if not failures else "no_match",
            "failures": failures,
        })
        if selected_variant is None and not failures:
            selected_variant = {
                "name": name,
                "runtime": {"rx_bucket": rx_bucket,
                            "layer_bucket": layer_bucket},
                "build": normalized_build,
            }

    checks: list[dict[str, Any]] = []
    blocking: list[str] = [] if selected_variant is not None else ["execution_variant"]
    if selected_variant is not None:
        receiver = value["receiver"]
        receiver["requested_rx_bucket"] = receiver["rx_bucket"]
        receiver["requested_layer_bucket"] = receiver["layer_bucket"]
        receiver["rx_bucket"] = selected_variant["runtime"]["rx_bucket"]
        receiver["layer_bucket"] = selected_variant["runtime"]["layer_bucket"]
        receiver["capacity_selection_policy"] = "capability_registry_variant"
        receiver["adapter_required"] = (
            receiver["rx_bucket"] != value["topology"]["rx_antennas"] or
            receiver["layer_bucket"] != value["active"]["rank"])
        receiver["ce_shape"] = [
            receiver["rx_bucket"], receiver["layer_bucket"],
            value["waveform"]["num_symbols"],
            value["waveform"]["padded_subcarriers"],
        ]
        receiver["detector_output_shape"] = [
            receiver["layer_bucket"], value["waveform"]["num_symbols"],
            value["waveform"]["padded_subcarriers"],
        ]
        value["dispatch"] = selected_variant
    else:
        value["dispatch"] = {"status": "no_matching_execution_variant"}
    seen: set[str] = set()
    kernel_root = path.resolve().parent.parent
    for index, raw_operator in enumerate(operators):
        where = f"capability_registry.operators[{index}]"
        operator = _object(raw_operator, where)
        _keys(operator, where, {
            "name", "scope", "implementation", "required", "constraints"
        })
        name = _string(operator["name"], f"{where}.name")
        if name in seen:
            raise ProfileError(f"duplicate capability operator {name!r}")
        seen.add(name)
        scope = _string(operator["scope"], f"{where}.scope")
        if scope not in {"tx", "rx", "host_boundary"}:
            raise ProfileError(f"{where}.scope must be tx, rx, or host_boundary")
        implementation = _string(operator["implementation"],
                                 f"{where}.implementation")
        required = operator["required"]
        if not isinstance(required, bool):
            raise ProfileError(f"{where}.required must be boolean")
        failures: list[str] = []
        if required and not (kernel_root / implementation).exists():
            failures.append(f"implementation missing: {implementation}")
        failures.extend(_constraint_failures(
            value, operator["constraints"], f"{where}.constraints"))
        status = "pass" if not failures else ("block" if required else "unsupported_optional")
        checks.append({
            "operator": name,
            "scope": scope,
            "implementation": implementation,
            "required": required,
            "status": status,
            "failures": failures,
        })
        if required and failures:
            blocking.append(name)

    validation = dict(value["validation"])
    validation.update({
        "execution_eligibility": "eligible" if not blocking else "ineligible",
        "capability_registry": {
            "registry_name": registry_name,
            "target": target,
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "blocking_operators": blocking,
        "variant_checks": variant_checks,
        "operator_checks": checks,
    })
    value["validation"] = validation
    return ResolvedPlan(value)


def compile_profile(path: Path, rank: int) -> ResolvedPlan:
    profile, digest = _load_document(path)
    _keys(profile, "profile", {
        "schema_version", "profile_name", "topology", "spatial", "waveform",
        "channel", "receiver"
    }, {"dmrs"})
    if profile["schema_version"] != 1:
        raise ProfileError("profile.schema_version must be 1")
    name = _string(profile["profile_name"], "profile.profile_name")
    if PROFILE_NAME.fullmatch(name) is None:
        raise ProfileError("profile.profile_name must match ^[a-z][a-z0-9_]{0,63}$")

    topology = _object(profile["topology"], "profile.topology")
    _keys(topology, "profile.topology", {
        "architecture", "tx_antennas", "tx_rf_chains", "rx_antennas", "rx_rf_chains"
    })
    architecture = _string(topology["architecture"], "profile.topology.architecture")
    if architecture not in VALID_ARCHITECTURES:
        raise ProfileError(f"unsupported topology architecture {architecture!r}")
    tx_antennas = _integer(topology["tx_antennas"], "profile.topology.tx_antennas", 1, MAX_TX_ANTENNAS)
    tx_rf = _integer(topology["tx_rf_chains"], "profile.topology.tx_rf_chains", 1, MAX_TX_ANTENNAS)
    rx_antennas = _integer(topology["rx_antennas"], "profile.topology.rx_antennas", 1, MAX_RX_ANTENNAS)
    rx_rf = _integer(topology["rx_rf_chains"], "profile.topology.rx_rf_chains", 1, MAX_RX_ANTENNAS)
    if tx_rf > tx_antennas or rx_rf > rx_antennas:
        raise ProfileError("RF chain count cannot exceed physical antenna count")
    if architecture == "full_digital" and (tx_rf != tx_antennas or rx_rf != rx_antennas):
        raise ProfileError("full_digital requires one RF chain per physical antenna")

    spatial = _object(profile["spatial"], "profile.spatial")
    _keys(spatial, "profile.spatial", {"supported_ranks", "port_policy", "precoding"},
          {"antenna_mapping"})
    ranks = _strict_increasing(spatial["supported_ranks"],
                               "profile.spatial.supported_ranks", 1, MAX_PUSCH_LAYERS)
    rank = _integer(rank, "requested_rank", 1, MAX_PUSCH_LAYERS)
    if rank not in ranks:
        raise ProfileError(f"Rank{rank} is not supported by profile {name}")
    if rank > min(tx_rf, rx_rf):
        raise ProfileError(
            f"Rank{rank} exceeds spatial RF limit min(tx_rf={tx_rf},rx_rf={rx_rf})")

    port_policy = _object(spatial["port_policy"], "profile.spatial.port_policy")
    _keys(port_policy, "profile.spatial.port_policy", {"mode"}, {"ports_by_rank"})
    port_mode = _string(port_policy["mode"], "profile.spatial.port_policy.mode")
    if port_mode == "identity":
        if "ports_by_rank" in port_policy:
            raise ProfileError("identity port policy must not define ports_by_rank")
        ports_by_rank = {item: item for item in ranks}
    elif port_mode == "explicit":
        ports_by_rank = _rank_map(
            port_policy.get("ports_by_rank"), "profile.spatial.port_policy.ports_by_rank",
            ranks, lambda value, where: _integer(value, where, 1, MAX_LOGICAL_PORTS))
    else:
        raise ProfileError("profile.spatial.port_policy.mode must be identity or explicit")
    for item, ports in ports_by_rank.items():
        if ports < item or ports > tx_rf:
            raise ProfileError(
                f"Rank{item} logical ports {ports} must be in [{item},tx_rf={tx_rf}]")
    tx_ports = ports_by_rank[rank]

    precoding = _object(spatial["precoding"], "profile.spatial.precoding")
    _keys(precoding, "profile.spatial.precoding", {"default_mode"}, {"by_rank"})
    default_precode = _string(precoding["default_mode"],
                              "profile.spatial.precoding.default_mode")
    if default_precode not in VALID_PRECODING:
        raise ProfileError(f"unsupported default precoding mode {default_precode!r}")
    by_rank = _object(precoding.get("by_rank", {}), "profile.spatial.precoding.by_rank")
    unknown_precode_ranks = set(by_rank) - {str(item) for item in ranks}
    if unknown_precode_ranks:
        raise ProfileError(f"precoding.by_rank has unsupported ranks {sorted(unknown_precode_ranks)}")
    selected_precode = _object(by_rank.get(str(rank), {"mode": default_precode}),
                               f"profile.spatial.precoding.by_rank.{rank}")
    _keys(selected_precode, f"profile.spatial.precoding.by_rank.{rank}", {"mode"},
          {"tpmi", "prg_size_rb"})
    precode_mode = _string(selected_precode["mode"], "precoding.mode")
    if precode_mode not in VALID_PRECODING:
        raise ProfileError(f"unsupported Rank{rank} precoding mode {precode_mode!r}")
    if precode_mode == "bypass" and tx_ports != rank:
        raise ProfileError(f"Rank{rank} bypass requires tx_ports==rank, got {tx_ports}")
    if precode_mode == "codebook" and "tpmi" not in selected_precode:
        raise ProfileError(f"Rank{rank} codebook precoding requires tpmi")

    waveform = _object(profile["waveform"], "profile.waveform")
    _keys(waveform, "profile.waveform", {
        "qm", "num_symbols", "fft_size", "num_rb", "used_subcarriers",
        "padded_subcarriers", "dmrs_symbols", "data_re_alignment", "qam_row_alignment"
    }, {"rb_start"})
    qm = _integer(waveform["qm"], "profile.waveform.qm", 2, 8)
    if qm not in VALID_QM:
        raise ProfileError(f"profile.waveform.qm must be one of {sorted(VALID_QM)}")
    num_symbols = _integer(waveform["num_symbols"], "profile.waveform.num_symbols", 1, 14)
    fft_size = _integer(waveform["fft_size"], "profile.waveform.fft_size", 128, 4096)
    num_rb = _integer(waveform["num_rb"], "profile.waveform.num_rb", 1, 275)
    rb_start = _integer(waveform.get("rb_start", 0), "profile.waveform.rb_start", 0, 274)
    used_sc = _integer(waveform["used_subcarriers"],
                       "profile.waveform.used_subcarriers", 1, 3300)
    padded_sc = _integer(waveform["padded_subcarriers"],
                         "profile.waveform.padded_subcarriers", 1, 4096)
    if used_sc != num_rb * 12:
        raise ProfileError("used_subcarriers must equal num_rb*12")
    if rb_start + num_rb > 275:
        raise ProfileError("rb_start+num_rb exceeds 275 RB profile limit")
    if not used_sc <= padded_sc <= fft_size:
        raise ProfileError("used_subcarriers <= padded_subcarriers <= fft_size is required")
    dmrs_symbols = _strict_increasing(waveform["dmrs_symbols"],
                                      "profile.waveform.dmrs_symbols", 0, num_symbols - 1)
    data_alignment = _integer(waveform["data_re_alignment"],
                              "profile.waveform.data_re_alignment", 1, 4096)
    row_alignment = _integer(waveform["qam_row_alignment"],
                             "profile.waveform.qam_row_alignment", 1, 4096)
    data_symbols = tuple(symbol for symbol in range(num_symbols) if symbol not in dmrs_symbols)
    data_re = len(data_symbols) * used_sc
    data_stride = _align_up(data_re, data_alignment)
    qam_row_stride = _align_up(used_sc, row_alignment)

    dmrs = _object(profile.get("dmrs", {}), "profile.dmrs")
    _keys(dmrs, "profile.dmrs", set(), {
        "ports_by_rank", "symbol_mask", "type", "length", "num_cdm_groups_without_data"
    })
    dmrs_ports_by_rank: dict[int, list[int]] = {}
    if "ports_by_rank" in dmrs:
        dmrs_ports_by_rank = _rank_map(
            dmrs["ports_by_rank"], "profile.dmrs.ports_by_rank", ranks,
            lambda value, where: _unique_ints(value, where, 0, 65535))
        for item, ports in dmrs_ports_by_rank.items():
            if len(ports) != item:
                raise ProfileError(f"Rank{item} must define exactly {item} DMRS ports")
    selected_dmrs_ports = list(dmrs_ports_by_rank.get(rank, ()))
    computed_mask = sum(1 << symbol for symbol in dmrs_symbols)
    symbol_mask = _integer(dmrs.get("symbol_mask", computed_mask),
                           "profile.dmrs.symbol_mask", 0, 65535)
    if symbol_mask != computed_mask:
        raise ProfileError("dmrs.symbol_mask does not match waveform.dmrs_symbols")

    channel = _object(profile["channel"], "profile.channel")
    _keys(channel, "profile.channel", {"model", "gain_by_rank", "awgn_std_int16"})
    channel_model = _string(channel["model"], "profile.channel.model")
    gains = _rank_map(channel["gain_by_rank"], "profile.channel.gain_by_rank", ranks,
                      lambda value, where: _number(value, where, 0.0, 1.0,
                                                   exclusive_low=True))
    awgn = _number(channel["awgn_std_int16"], "profile.channel.awgn_std_int16", 0.0, 200.0)

    receiver = _object(profile["receiver"], "profile.receiver")
    _keys(receiver, "profile.receiver", {
        "channel_estimator", "detector", "capacity", "buckets", "rx_capacity_adapter"
    })
    capacity = _object(receiver["capacity"], "profile.receiver.capacity")
    _keys(capacity, "profile.receiver.capacity", {"max_rx_antennas", "max_layers"})
    max_rx = _integer(capacity["max_rx_antennas"],
                      "profile.receiver.capacity.max_rx_antennas", 1, MAX_ANTENNAS)
    max_layers = _integer(capacity["max_layers"],
                          "profile.receiver.capacity.max_layers", 1, MAX_LAYERS)
    if rx_antennas > max_rx or rank > max_layers:
        raise ProfileError("active RX/Rank exceeds declared receiver capacity")
    buckets = _object(receiver["buckets"], "profile.receiver.buckets")
    _keys(buckets, "profile.receiver.buckets", {"rx_antennas", "layers"})
    rx_buckets = _strict_increasing(buckets["rx_antennas"],
                                    "profile.receiver.buckets.rx_antennas", 1, max_rx)
    layer_buckets = _strict_increasing(buckets["layers"],
                                       "profile.receiver.buckets.layers", 1, max_layers)
    rx_bucket = _smallest_bucket(rx_antennas, rx_buckets, "rx_antennas")
    layer_bucket = _smallest_bucket(rank, layer_buckets, "rank")

    rx_capacity_adapter = _string(receiver["rx_capacity_adapter"],
                                  "profile.receiver.rx_capacity_adapter")
    plan = {
        "schema": "airan.mimo.resolved_plan.v1",
        "source": {
            "profile_name": name,
            "profile_schema_version": 1,
            "sha256": digest,
        },
        "active": {
            "rank": rank,
            "logical_tx_ports": tx_ports,
        },
        "topology": {
            "architecture": architecture,
            "tx_antennas": tx_antennas,
            "tx_rf_chains": tx_rf,
            "rx_antennas": rx_antennas,
            "rx_rf_chains": rx_rf,
        },
        "spatial": {
            "supported_ranks": list(ranks),
            "antenna_mapping": spatial.get("antenna_mapping", "identity"),
            "precoding": {
                "mode": precode_mode,
                "tpmi": selected_precode.get("tpmi", 0),
                "prg_size_rb": selected_precode.get("prg_size_rb", num_rb),
            },
        },
        "waveform": {
            "qm": qm,
            "num_symbols": num_symbols,
            "fft_size": fft_size,
            "num_rb": num_rb,
            "rb_start": rb_start,
            "used_subcarriers": used_sc,
            "padded_subcarriers": padded_sc,
            "dmrs_symbols": list(dmrs_symbols),
            "data_symbols": list(data_symbols),
            "num_dmrs_symbols": len(dmrs_symbols),
            "num_data_symbols": len(data_symbols),
            "data_re_per_layer": data_re,
            "data_stride_per_layer": data_stride,
            "codeword_symbols": rank * data_re,
            "codeword_stride": rank * data_stride,
            "qam_row_stride": qam_row_stride,
            "grid_re_per_port": num_symbols * padded_sc,
        },
        "dmrs": {
            "ports": selected_dmrs_ports,
            "symbol_mask": symbol_mask,
            "type": dmrs.get("type", 1),
            "length": dmrs.get("length", 1),
            "num_cdm_groups_without_data": dmrs.get("num_cdm_groups_without_data", 1),
        },
        "channel": {
            "model": channel_model,
            "gain": gains[rank],
            "awgn_std_int16": awgn,
        },
        "receiver": {
            "channel_estimator": _string(receiver["channel_estimator"],
                                          "profile.receiver.channel_estimator"),
            "detector": _string(receiver["detector"], "profile.receiver.detector"),
            "physical_rx_antennas": rx_antennas,
            "active_layers": rank,
            "rx_bucket": rx_bucket,
            "layer_bucket": layer_bucket,
            "max_rx_antennas": max_rx,
            "max_layers": max_layers,
            "rx_capacity_adapter": rx_capacity_adapter,
            "capacity_selection_policy": "smallest_declared_bucket",
            "adapter_required": (rx_capacity_adapter != "none" or
                                 rx_bucket != rx_antennas or layer_bucket != rank),
            "ce_shape": [rx_bucket, layer_bucket, num_symbols, padded_sc],
            "detector_output_shape": [layer_bucket, num_symbols, padded_sc],
        },
        "validation": {
            "configuration_valid": True,
            "execution_eligibility": "not_evaluated_until_capability_registry",
            "limits": {"max_tx_antennas": MAX_TX_ANTENNAS,
                       "max_rx_antennas": MAX_RX_ANTENNAS,
                       "max_pusch_layers": MAX_PUSCH_LAYERS,
                       "max_internal_storage_layers": MAX_STORAGE_LAYERS},
        },
    }
    return ResolvedPlan(plan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binary-output", type=Path)
    parser.add_argument("--capabilities", type=Path,
                        help="evaluate the plan against an installed target registry")
    parser.add_argument("--require-eligible", action="store_true",
                        help="fail if the capability registry blocks this plan")
    args = parser.parse_args()
    try:
        plan = compile_profile(args.profile, args.rank)
        if args.capabilities is not None:
            plan = evaluate_capabilities(plan, args.capabilities)
        if args.require_eligible:
            if args.capabilities is None:
                raise ProfileError("--require-eligible requires --capabilities")
            if plan.value["validation"]["execution_eligibility"] != "eligible":
                blockers = plan.value["validation"]["blocking_operators"]
                raise ProfileError(
                    f"plan is not executable on selected target; blockers={blockers}")
    except ProfileError as error:
        parser.error(str(error))
    payload = plan.to_json()
    if args.output is None and args.binary_output is None:
        print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"[PASS] wrote {args.output}")
    if args.binary_output is not None:
        args.binary_output.parent.mkdir(parents=True, exist_ok=True)
        args.binary_output.write_bytes(plan.to_runtime_config())
        print(f"[PASS] wrote {args.binary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
