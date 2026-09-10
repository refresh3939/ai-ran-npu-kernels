#!/usr/bin/env python3
"""Machine-executable host contract audit for the PUSCH MIMO chains.

This program deliberately does not launch, emulate, or replace any operator.
It validates the integration manifest, resolves operator source directories,
checks source-backed hard gates, and refuses readiness while any required
operator or semantic bridge is unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


COMMON_ROOT = Path(__file__).resolve().parent.parent / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from mimo_config_compiler import compile_config
from mimo_profile_compiler import ProfileError, compile_profile


EDGE_FIELDS = (
    "dtype",
    "shape",
    "layout",
    "padding",
    "bit_order",
    "active_layers",
    "port_semantics",
)

E2E_STAGES = {
    "ldpc_encode", "rate_match_mimo", "scramble_mimo", "qam_mod_256_mimo",
    "layer_map_mimo", "mimo_dmrs_gen", "mimo_resource_grid_map",
    "pusch_codebook_precode", "re_map_batch", "ofdm_mod_batch",
    "ofdm_demod_batch", "re_demap_batch", "mimo_dmrs_ls",
    "channel_est_lmmse_mimo", "mimo_detect_io_pack", "mimo_detect_bri_batch",
    "qam_demod_256_mimo_batch", "layer_demap_mimo", "descramble_mimo",
    "rate_dematch_mimo", "ldpc_decode",
}


@dataclass
class Finding:
    severity: str
    item: str
    message: str


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("contract matrix root must be an object")
    return value


def validate_geometry(matrix: dict[str, Any], findings: list[Finding]) -> None:
    g = matrix.get("geometry", {})
    expected = {
        "data_re_per_layer": 12 * 1596,
        "data_stride_per_layer": 19200,
        "data_padding_per_layer": 19200 - 12 * 1596,
        "qam_row_stride": 1600,
        "qam_row_padding": 1600 - 1596,
        "fft_size": 32 * 64,
        "dmrs_re_per_comb": 1596 // 2,
        "dmrs_stride": 896,
        "occ_observations_per_shared_comb_layer": 1596 // 4,
        "grid_re_per_slot": 14 * 1664,
    }
    for name, want in expected.items():
        got = g.get(name)
        if got != want:
            findings.append(Finding("error", f"geometry.{name}", f"expected {want}, got {got!r}"))
    maps = matrix.get("bit_order_maps", {})
    qam_to_nr = maps.get("qam_stream_to_nr_bit")
    nr_to_qam = maps.get("nr_bit_to_qam_stream")
    if qam_to_nr != [0, 2, 4, 6, 1, 3, 5, 7]:
        findings.append(Finding("error", "bit_order_maps.qam_stream_to_nr_bit",
                                "must be [0,2,4,6,1,3,5,7]"))
    if not isinstance(qam_to_nr, list) or not isinstance(nr_to_qam, list) or len(qam_to_nr) != 8 or len(nr_to_qam) != 8:
        findings.append(Finding("error", "bit_order_maps", "both maps must be eight-element arrays"))
    elif any(nr_to_qam[nr_bit] != qam_stream for qam_stream, nr_bit in enumerate(qam_to_nr)):
        findings.append(Finding("error", "bit_order_maps", "NR/QAM stream maps are not inverses"))


def validate_observation_models(matrix: dict[str, Any], findings: list[Finding]) -> None:
    """Validate the canonical Rank1-4 DMRS observation and CE/io-pack contract."""
    profiles = matrix.get("observation_profiles")
    contract = matrix.get("observation_contract")
    if profiles is None and contract is None:
        return
    if not isinstance(profiles, list) or not isinstance(contract, dict):
        findings.append(Finding("error", "observation_models",
                                "observation_profiles list and observation_contract object are both required"))
        return
    expected = {
        1: ([1000], [798], ["COMB2_798"], ["2*p, p=0..797"]),
        2: ([1000, 1002], [798, 798], ["COMB2_798", "COMB2_798"],
            ["2*p, p=0..797", "1+2*p, p=0..797"]),
        3: ([1000, 1001, 1002], [399, 399, 798],
            ["FD_OCC2_399", "FD_OCC2_399", "COMB2_798"],
            ["1+4*p, p=0..398", "1+4*p, p=0..398", "1+2*p, p=0..797"]),
        4: ([1000, 1001, 1002, 1003], [399, 399, 399, 399],
            ["FD_OCC2_399"] * 4,
            ["1+4*p, p=0..398", "1+4*p, p=0..398",
             "2+4*p, p=0..398", "2+4*p, p=0..398"]),
    }
    by_rank = {profile.get("rank"): profile for profile in profiles
               if isinstance(profile, dict)}
    if set(by_rank) != set(expected):
        findings.append(Finding("error", "observation_profiles",
                                "must declare exactly one profile for each Rank 1,2,3,4"))
    for rank, (ports, counts, models, locations) in expected.items():
        profile = by_rank.get(rank, {})
        for field, want in (("dmrs_ports", ports), ("pilot_count_per_layer", counts),
                            ("model_per_layer", models),
                            ("pilot_sc_per_layer", locations)):
            if profile.get(field) != want:
                findings.append(Finding("error", f"observation_profiles.rank{rank}.{field}",
                                        f"expected {want}, got {profile.get(field)!r}"))

    expected_contract = {
        "h_ls_shape": ["NR", "L", 2, 832],
        "pilot_count_shape": ["L", 2],
        "pilot_sc_shape": ["L", 2, 832],
        "ce_output_shape": ["NR", 16, 14, 1664],
        "io_pack_input_shape": ["NR", 16, 14, 1664],
        "active_layer_rule": "[0,L)=estimate; [L,16)=positive_zero",
        "weight_binding": ["rank", "pilot_count", "pilot_sc_hash"],
    }
    for field, want in expected_contract.items():
        if contract.get(field) != want:
            findings.append(Finding("error", f"observation_contract.{field}",
                                    f"expected {want}, got {contract.get(field)!r}"))
    if contract.get("ce_output_shape") != contract.get("io_pack_input_shape"):
        findings.append(Finding("error", "observation_contract.ce_to_io_pack",
                                "CE output and io_pack input shapes must match exactly"))
    if contract.get("allow_399_to_798_synthesis") is not False:
        findings.append(Finding("error", "observation_contract.allow_399_to_798_synthesis",
                                "must be false"))


def validate_schema(matrix: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for key in ("schema_version", "chain", "mimo_profile_contract", "stage_order",
                "operators", "edges", "hard_gates"):
        if key not in matrix:
            findings.append(Finding("error", "matrix", f"missing top-level key {key}"))
    if matrix.get("schema_version") != 1:
        findings.append(Finding("error", "matrix", "schema_version must be 1"))

    operators = matrix.get("operators", [])
    ids = [op.get("id") for op in operators if isinstance(op, dict)]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        findings.append(Finding("error", "operators", "operator ids must be non-empty and unique"))
    if matrix.get("stage_order") != ids:
        findings.append(Finding("error", "stage_order", "must exactly match ordered operator ids"))

    known = set(ids)
    edge_ids: set[str] = set()
    for edge in matrix.get("edges", []):
        edge_id = edge.get("id", "<unnamed>")
        if edge_id in edge_ids:
            findings.append(Finding("error", edge_id, "duplicate edge id"))
        edge_ids.add(edge_id)
        if edge.get("producer") not in known or edge.get("consumer") not in known:
            findings.append(Finding("error", edge_id, "edge endpoint is not a declared operator"))
        producer = edge.get("producer_contract", {})
        consumer = edge.get("consumer_contract", {})
        for field in EDGE_FIELDS:
            if field not in producer or field not in consumer:
                findings.append(Finding("error", edge_id, f"missing producer/consumer {field}"))
        adapter = edge.get("adapter")
        if not adapter:
            for field in EDGE_FIELDS:
                if producer.get(field) != consumer.get(field):
                    findings.append(Finding("error", edge_id, f"{field} mismatch without an explicit adapter"))
        elif not adapter.get("operation") or "validated" not in adapter:
            findings.append(Finding("error", edge_id, "adapter requires operation and validated fields"))

    validate_geometry(matrix, findings)
    validate_observation_models(matrix, findings)
    return findings


def validate_mimo_profile_contract(matrix: dict[str, Any], kernel_root: Path,
                                   findings: list[Finding]) -> None:
    spec = matrix.get("mimo_profile_contract")
    if not isinstance(spec, dict):
        findings.append(Finding("error", "mimo_profile_contract", "must be an object"))
        return
    split_required = {"radio_name", "radio_path", "grant_path", "ranks"}
    legacy_required = {"profile_name", "path", "ranks"}
    if set(spec) not in (split_required, legacy_required):
        findings.append(Finding(
            "error", "mimo_profile_contract",
            "must use the split radio/grant contract "
            f"{sorted(split_required)} (or legacy {sorted(legacy_required)}), "
            f"got {sorted(spec)}"))
        return
    ranks = spec.get("ranks")
    if (not isinstance(ranks, list) or not ranks or
            any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks) or
            ranks != sorted(set(ranks))):
        findings.append(Finding("error", "mimo_profile_contract.ranks",
                                "must be a non-empty strictly increasing integer list"))
        return
    plans: list[dict[str, Any]] = []
    try:
        if set(spec) == split_required:
            radio_path = kernel_root / str(spec["radio_path"])
            grant_path = kernel_root / str(spec["grant_path"])
            plans = [compile_config(radio_path, grant_path, rank).value
                     for rank in ranks]
            expected_name = spec["radio_name"]
        else:
            path = kernel_root / str(spec["path"])
            plans = [compile_profile(path, rank).value for rank in ranks]
            expected_name = spec["profile_name"]
    except (ProfileError, OSError, ValueError) as error:
        findings.append(Finding("error", "mimo_profile_contract",
                                f"configuration compilation failed: {error}"))
        return
    if any(plan["source"]["profile_name"] != expected_name for plan in plans):
        findings.append(Finding("error", "mimo_profile_contract.radio_name",
                                "compiled radio name mismatch"))
    if any(plan["spatial"]["supported_ranks"] != ranks for plan in plans):
        findings.append(Finding("error", "mimo_profile_contract.ranks",
                                "compiled supported ranks mismatch"))

    waveform = plans[0]["waveform"]
    geometry = matrix.get("geometry", {})
    expected = {
        "num_symbols": waveform["num_symbols"],
        "num_data_symbols": waveform["num_data_symbols"],
        "used_subcarriers": waveform["used_subcarriers"],
        "padded_subcarriers": waveform["padded_subcarriers"],
        "data_re_per_layer": waveform["data_re_per_layer"],
        "data_stride_per_layer": waveform["data_stride_per_layer"],
        "data_padding_per_layer": (
            waveform["data_stride_per_layer"] - waveform["data_re_per_layer"]),
        "qam_row_stride": waveform["qam_row_stride"],
        "qam_row_padding": waveform["qam_row_stride"] - waveform["used_subcarriers"],
        "fft_size": waveform["fft_size"],
        "grid_re_per_slot": waveform["grid_re_per_port"],
    }
    for field, want in expected.items():
        if geometry.get(field) != want:
            findings.append(Finding(
                "error", f"mimo_profile_contract.geometry.{field}",
                f"contract matrix has {geometry.get(field)!r}, compiled profile requires {want}"))


def resolve_operator(kernel_root: Path, op: dict[str, Any]) -> tuple[Path | None, str | None]:
    failures: list[str] = []
    for relative in op.get("path_candidates", []):
        directory = kernel_root / relative
        if not directory.is_dir():
            failures.append(f"{relative}: directory absent")
            continue
        missing = [name for name in op.get("required_files", []) if not (directory / name).is_file()]
        if missing:
            failures.append(f"{relative}: missing {','.join(missing)}")
            continue
        return directory, None
    return None, "; ".join(failures) if failures else "no path candidates declared"


def audit(matrix: dict[str, Any], kernel_root: Path) -> tuple[list[Finding], dict[str, Path]]:
    findings = validate_schema(matrix)
    resolved: dict[str, Path] = {}
    if any(f.severity == "error" for f in findings):
        return findings, resolved

    validate_mimo_profile_contract(matrix, kernel_root, findings)
    if any(f.severity == "error" for f in findings):
        return findings, resolved

    for requirement in matrix.get("source_requirements", []):
        path = kernel_root / requirement["path"]
        if not path.is_file():
            findings.append(Finding(requirement.get("severity", "hard_fail"), requirement["id"],
                                    f"missing shared source requirement: {requirement['path']}"))

    for op in matrix["operators"]:
        directory, why = resolve_operator(kernel_root, op)
        if directory is None:
            severity = "hard_fail" if op.get("required", True) else "warning"
            findings.append(Finding(severity, op["id"], f"operator unavailable: {why}"))
        else:
            resolved[op["id"]] = directory

    for assertion in matrix.get("source_assertions", []):
        op_dir = resolved.get(assertion["operator"])
        source = op_dir / assertion["file"] if op_dir else None
        if source is None or not source.is_file():
            findings.append(Finding("error", assertion["id"],
                                    f"missing source evidence {assertion['operator']}/{assertion['file']}"))
            continue
        source_text = source.read_text(encoding="utf-8", errors="replace")
        if re.search(assertion["regex"], source_text, flags=re.MULTILINE) is None:
            findings.append(Finding("error", assertion["id"], f"source evidence mismatch: {source}"))

    for edge in matrix["edges"]:
        adapter = edge.get("adapter")
        if adapter and not adapter.get("validated"):
            findings.append(Finding("hard_fail", edge["id"],
                                    f"unvalidated adapter: {adapter['operation']}"))

    for gate in matrix["hard_gates"]:
        evidence_ok = True
        evidence_messages: list[str] = []
        for evidence in gate.get("evidence", []):
            op_dir = resolved.get(evidence["operator"])
            file_candidates = evidence.get("files", [evidence.get("file")])
            source = None
            if op_dir:
                source = next((op_dir / name for name in file_candidates
                               if name and (op_dir / name).is_file()), None)
            if source is None:
                evidence_ok = False
                evidence_messages.append(
                    f"missing {evidence['operator']}/({' or '.join(str(x) for x in file_candidates)})")
                continue
            text = source.read_text(encoding="utf-8", errors="replace")
            if re.search(evidence["regex"], text, flags=re.MULTILINE) is None:
                evidence_ok = False
                evidence_messages.append(f"pattern absent in {source}")
        for artifact in gate.get("artifact_evidence", []):
            source = kernel_root / artifact["path"]
            if not source.is_file():
                # During source-tree development the categorized operators may
                # be audited from a read-only clean tree before this chain is
                # promoted. Keep artifact paths promotion-relative while using
                # the harness's own kernel root as the development fallback.
                source = Path(__file__).resolve().parent.parent / artifact["path"]
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                evidence_ok = False
                evidence_messages.append(f"invalid artifact {source}: {exc}")
                continue
            for field, want in artifact.get("required", {}).items():
                if payload.get(field) != want:
                    evidence_ok = False
                    evidence_messages.append(
                        f"{source}:{field} expected {want!r}, got {payload.get(field)!r}")
            for field, limits in artifact.get("numeric", {}).items():
                value = payload.get(field)
                if not isinstance(value, (int, float)):
                    evidence_ok = False
                    evidence_messages.append(f"{source}:{field} is not numeric")
                    continue
                if "min" in limits and value < limits["min"]:
                    evidence_ok = False
                    evidence_messages.append(f"{source}:{field} below {limits['min']}")
                if "min_exclusive" in limits and value <= limits["min_exclusive"]:
                    evidence_ok = False
                    evidence_messages.append(f"{source}:{field} not above {limits['min_exclusive']}")
                if "max" in limits and value > limits["max"]:
                    evidence_ok = False
                    evidence_messages.append(f"{source}:{field} above {limits['max']}")
            if payload.get("complete_coded_e2e") is True:
                stages = payload.get("actual_npu_stages")
                missing_stages = E2E_STAGES - set(stages if isinstance(stages, list) else [])
                if missing_stages:
                    evidence_ok = False
                    evidence_messages.append(
                        f"{source}: missing actual NPU stages {sorted(missing_stages)}")
                raw_ber = payload.get("raw_qam_ber")
                if not isinstance(raw_ber, (int, float)) or not 0.0 <= raw_ber <= 1.0:
                    evidence_ok = False
                    evidence_messages.append(f"{source}: invalid raw_qam_ber")
                digest = payload.get("tx_bits_sha256")
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    evidence_ok = False
                    evidence_messages.append(f"{source}: invalid tx_bits_sha256")
        if not evidence_ok:
            findings.append(Finding("error", gate["id"], "gate evidence invalid: " + "; ".join(evidence_messages)))
        if gate.get("status") != "resolved":
            findings.append(Finding("hard_fail", gate["id"], gate["message"]))
    return findings, resolved


def emit_human(matrix: dict[str, Any], kernel_root: Path, findings: list[Finding], resolved: dict[str, Path]) -> None:
    print(f"CONTRACT_MATRIX {matrix.get('chain')} schema={matrix.get('schema_version')} root={kernel_root}")
    profile = matrix.get("mimo_profile_contract", {})
    if isinstance(profile, dict) and profile.get("radio_path"):
        print(f"  RADIO {profile.get('radio_name')}: {kernel_root / profile['radio_path']}")
        print(f"  GRANT: {kernel_root / profile['grant_path']}")
    elif isinstance(profile, dict) and profile.get("path"):
        print(f"  PROFILE {profile.get('profile_name')}: {kernel_root / profile['path']}")
    for op_id in matrix.get("stage_order", []):
        if op_id in resolved:
            print(f"  OP {op_id}: FOUND {resolved[op_id]}")
        else:
            print(f"  OP {op_id}: UNAVAILABLE")
    for finding in findings:
        print(f"  {finding.severity.upper()} {finding.item}: {finding.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--kernel-root", type=Path)
    parser.add_argument("--mode", choices=("check", "readiness"), default="check")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    matrix_path = args.matrix.resolve()
    kernel_root = (args.kernel_root.resolve() if args.kernel_root else matrix_path.parents[2])
    try:
        matrix = load_matrix(matrix_path)
        findings, resolved = audit(matrix, kernel_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"CONTRACT_MATRIX ERROR: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps({
            "chain": matrix.get("chain"),
            "kernel_root": str(kernel_root),
            "resolved_operators": {key: str(value) for key, value in resolved.items()},
            "findings": [asdict(finding) for finding in findings],
            "ready": not any(f.severity in ("error", "hard_fail") for f in findings),
        }, indent=2, sort_keys=True))
    else:
        emit_human(matrix, kernel_root, findings, resolved)

    errors = [f for f in findings if f.severity == "error"]
    blockers = [f for f in findings if f.severity == "hard_fail"]
    if errors:
        print("CONTRACT_CHECK FAIL", file=sys.stderr)
        return 1
    if args.mode == "readiness" and blockers:
        print("END_TO_END_READINESS HARD_FAIL", file=sys.stderr)
        return 2
    print("CONTRACT_CHECK PASS" if args.mode == "check" else "END_TO_END_READINESS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
