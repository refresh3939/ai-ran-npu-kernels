#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path


COMMON = Path(__file__).resolve().parents[1]
KERNELS = COMMON.parent
sys.path.insert(0, str(COMMON))

from mimo_profile_compiler import (ProfileError, ResolvedPlan, compile_profile,
                                   evaluate_capabilities)


PROFILE = COMMON / "profiles/fd8x8_rank1_4.json"
CAPABILITIES = COMMON / "mimo_operator_capabilities.json"


def expect_error(document: dict, message: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "invalid.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        try:
            compile_profile(path, document["spatial"]["supported_ranks"][0])
        except ProfileError:
            return
    raise AssertionError(message)


def main() -> None:
    schema = json.loads((COMMON / "mimo_profile.schema.json").read_text())
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["topology"]["properties"]["tx_antennas"]["maximum"] == 64
    assert schema["properties"]["spatial"]["properties"]["supported_ranks"]["items"]["maximum"] == 16

    expected_ports = {1: 1, 2: 2, 3: 4, 4: 4}
    expected_gains = {1: 0.32, 2: 0.32, 3: 0.5, 4: 0.32}
    for rank in range(1, 5):
        plan = compile_profile(PROFILE, rank).value
        assert plan["schema"] == "airan.mimo.resolved_plan.v1"
        assert plan["active"] == {"rank": rank, "logical_tx_ports": expected_ports[rank]}
        assert plan["topology"]["tx_antennas"] == 8
        assert plan["topology"]["rx_antennas"] == 8
        assert plan["channel"]["gain"] == expected_gains[rank]
        assert plan["waveform"]["data_re_per_layer"] == 19152
        assert plan["waveform"]["data_stride_per_layer"] == 19200
        assert plan["waveform"]["qam_row_stride"] == 1600
        assert plan["waveform"]["grid_re_per_port"] == 23296
        assert plan["receiver"]["rx_bucket"] == 8
        assert plan["receiver"]["layer_bucket"] in (1, 2, 4)
        assert plan["validation"]["execution_eligibility"] == (
            "not_evaluated_until_capability_registry")
        evaluated = evaluate_capabilities(compile_profile(PROFILE, rank), CAPABILITIES)
        assert evaluated.value["validation"]["execution_eligibility"] == "eligible"
        assert evaluated.value["validation"]["blocking_operators"] == []
        receiver = evaluated.value["receiver"]
        assert receiver["requested_rx_bucket"] == 8
        assert receiver["requested_layer_bucket"] in (1, 2, 4)
        expected_execution_rx = 16 if rank == 4 else 64
        assert receiver["rx_bucket"] == expected_execution_rx
        assert receiver["layer_bucket"] == 16
        assert receiver["ce_shape"] == [expected_execution_rx, 16, 14, 1664]
        dispatch = evaluated.value["dispatch"]
        expected_variant = ("rx16_layer16_rank4" if rank == 4
                            else "rx64_layer16_rank1_4")
        assert dispatch["name"] == expected_variant
        assert dispatch["build"]["ce_nr"] == expected_execution_rx
        assert dispatch["build"]["ce_retained_rank"] == 96
        assert all(check["status"] == "pass"
                   for check in evaluated.value["validation"]["operator_checks"])
        binary = evaluated.to_runtime_config()
        assert len(binary) == 192
        assert binary[:4] == b"\x01\x00\xc0\x00"
        assert int.from_bytes(binary[4:8], "little") == 3
        assert int.from_bytes(binary[54:56], "little") == expected_execution_rx
        assert int.from_bytes(binary[56:58], "little") == 16

    # Stage 2 removes the old loopback-local configuration copy.
    assert not (KERNELS / "chain/pusch_mimo_loopback/radio_profiles.json").exists()

    base = json.loads(PROFILE.read_text())
    invalid = copy.deepcopy(base)
    invalid["topology"]["tx_antennas"] = 65
    expect_error(invalid, "TX>64 must fail")
    invalid = copy.deepcopy(base)
    invalid["topology"]["rx_rf_chains"] = 4
    expect_error(invalid, "full digital RF mismatch must fail")
    invalid = copy.deepcopy(base)
    invalid["spatial"]["port_policy"]["ports_by_rank"]["3"] = 2
    expect_error(invalid, "ports<Rank must fail")
    invalid = copy.deepcopy(base)
    invalid["receiver"]["buckets"]["rx_antennas"] = [1, 2, 4]
    expect_error(invalid, "missing fitting RX bucket must fail")
    invalid = copy.deepcopy(base)
    invalid["waveform"]["used_subcarriers"] = 1595
    expect_error(invalid, "non-RB-aligned subcarrier count must fail")

    # The authoring model and planner must cover the requested upper boundary
    # even though current NPU execution eligibility is deliberately evaluated
    # only in the later capability-registry stage.
    wide = copy.deepcopy(base)
    wide["profile_name"] = "fd64x64_rank16_contract"
    wide["topology"] = {
        "architecture": "full_digital",
        "tx_antennas": 64,
        "tx_rf_chains": 64,
        "rx_antennas": 64,
        "rx_rf_chains": 64,
    }
    wide["spatial"]["supported_ranks"] = [16]
    wide["spatial"]["port_policy"] = {"mode": "identity"}
    wide["spatial"]["precoding"] = {"default_mode": "bypass"}
    wide["dmrs"]["ports_by_rank"] = {"16": list(range(1000, 1016))}
    wide["channel"]["gain_by_rank"] = {"16": 0.16}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "wide.json"
        path.write_text(json.dumps(wide), encoding="utf-8")
        plan = compile_profile(path, 16).value
    assert plan["active"] == {"rank": 16, "logical_tx_ports": 16}
    assert plan["topology"]["tx_antennas"] == 64
    assert plan["topology"]["rx_antennas"] == 64
    assert plan["receiver"]["rx_bucket"] == 64
    assert plan["receiver"]["layer_bucket"] == 16
    assert plan["receiver"]["ce_shape"] == [64, 16, 14, 1664]
    assert plan["validation"]["execution_eligibility"] == (
        "not_evaluated_until_capability_registry")
    evaluated = evaluate_capabilities(ResolvedPlan(plan), CAPABILITIES)
    assert evaluated.value["validation"]["execution_eligibility"] == "ineligible"
    blockers = evaluated.value["validation"]["blocking_operators"]
    assert "execution_variant" in blockers
    assert "host_full_digital_channel_8x8" in blockers
    assert "layer_map_mimo" in blockers
    assert "mimo_detect_bri_batch" in blockers
    print("[PASS] common MIMO profile compiler, capability registry, 64x64/Rank16 report, and fail-closed cases")


if __name__ == "__main__":
    main()
