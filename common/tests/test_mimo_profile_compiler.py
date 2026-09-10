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

from mimo_profile_compiler import ProfileError, compile_profile, evaluate_capabilities


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
    assert schema["properties"]["topology"]["properties"]["tx_antennas"]["maximum"] == 16
    assert schema["properties"]["spatial"]["properties"]["supported_ranks"]["items"]["maximum"] == 4
    assert schema["properties"]["receiver"]["properties"]["capacity"]["properties"]["max_layers"]["maximum"] == 4

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
        assert receiver["max_layers"] == 4
        expected_execution_rx = 16 if rank == 4 else 64
        assert receiver["rx_bucket"] == expected_execution_rx
        assert receiver["layer_bucket"] == 16
        assert receiver["ce_shape"] == [expected_execution_rx, 16, 14, 1664]
        dispatch = evaluated.value["dispatch"]
        expected_variant = ("rx16_storage16_rank4" if rank == 4
                            else "rx64_storage16_rank1_4")
        assert dispatch["name"] == expected_variant
        assert dispatch["build"]["ce_nr"] == expected_execution_rx
        assert dispatch["build"]["ce_retained_rank"] == 96
        assert all(check["status"] == "pass"
                   for check in evaluated.value["validation"]["operator_checks"])
        binary = evaluated.to_runtime_config()
        assert len(binary) == 192
        assert binary[:4] == b"\x02\x00\xc0\x00"
        assert int.from_bytes(binary[4:8], "little") == 3
        assert int.from_bytes(binary[54:56], "little") == expected_execution_rx
        assert int.from_bytes(binary[56:58], "little") == 16

    # Stage 2 removes the old loopback-local configuration copy.
    assert not (KERNELS / "chain/pusch_mimo_loopback/radio_profiles.json").exists()

    base = json.loads(PROFILE.read_text())
    invalid = copy.deepcopy(base)
    invalid["topology"]["tx_antennas"] = 17
    expect_error(invalid, "TX>16 must fail")
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
    invalid["receiver"]["capacity"]["max_layers"] = 16
    invalid["receiver"]["buckets"]["layers"] = [1, 2, 4, 8, 16]
    expect_error(invalid, "profile receiver capacity above standard Rank4 must fail")
    invalid = copy.deepcopy(base)
    invalid["waveform"]["used_subcarriers"] = 1595
    expect_error(invalid, "non-RB-aligned subcarrier count must fail")

    # A standard single NR PUSCH profile cannot be promoted to Rank16.
    wide = copy.deepcopy(base)
    wide["profile_name"] = "fd16x64_rank16_contract"
    wide["topology"] = {
        "architecture": "full_digital",
        "tx_antennas": 16,
        "tx_rf_chains": 16,
        "rx_antennas": 64,
        "rx_rf_chains": 64,
    }
    wide["spatial"]["supported_ranks"] = [16]
    wide["spatial"]["port_policy"] = {"mode": "identity"}
    wide["spatial"]["precoding"] = {"default_mode": "bypass"}
    wide["dmrs"]["ports_by_rank"] = {"16": list(range(1000, 1016))}
    wide["channel"]["gain_by_rank"] = {"16": 0.16}
    expect_error(wide, "single-PUSCH Rank16 must fail")
    scalable = COMMON / "profiles/fd16x64_rank1_4.json"
    for rank in range(1, 5):
        evaluated = evaluate_capabilities(compile_profile(scalable, rank), CAPABILITIES)
        assert evaluated.value["validation"]["execution_eligibility"] == "eligible"
        assert evaluated.value["topology"]["tx_antennas"] == 16
        assert evaluated.value["topology"]["rx_antennas"] == 64
    print("[PASS] common MIMO profile compiler, 16TX/64RX Rank1-4 execution, standard Rank limit, and fail-closed cases")


if __name__ == "__main__":
    main()
