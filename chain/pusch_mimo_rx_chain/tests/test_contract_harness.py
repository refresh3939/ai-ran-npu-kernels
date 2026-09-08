#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHAIN_DIR = HERE.parent
KERNEL_ROOT = Path(os.environ.get("KERNEL_ROOT", HERE.parents[2])).resolve()
sys.path.insert(0, str(CHAIN_DIR.parent))
import pusch_mimo_contract_harness as harness  # noqa: E402

matrix = json.loads((CHAIN_DIR / "contract_matrix.json").read_text(encoding="utf-8"))
schema_findings = harness.validate_schema(matrix)
assert not schema_findings, schema_findings
assert matrix["claim"] == "staged_device_orchestrator_rank1_to_rank4_e2e_verified"
assert len(matrix["edges"]) == 10
assert matrix["stage_order"][0] == "ofdm_demod_batch"
assert matrix["stage_order"][-1] == "ldpc_decode"

assert [profile["pilot_count_per_layer"] for profile in matrix["observation_profiles"]] == [
    [798], [798, 798], [399, 399, 798], [399, 399, 399, 399]
]
assert matrix["observation_contract"]["ce_output_shape"] == ["NR", 16, 14, 1664]
assert matrix["observation_contract"]["io_pack_input_shape"] == ["NR", 16, 14, 1664]
assert matrix["observation_contract"]["allow_399_to_798_synthesis"] is False
assert [g["id"] for g in matrix["hard_gates"]] == [
    "rx_combined_device_chain_and_nonzero_e2e"
]
assert matrix["hard_gates"][0]["status"] == "resolved"

findings, _ = harness.audit(matrix, KERNEL_ROOT)
assert not [f for f in findings if f.severity == "error"], findings
assert not [f for f in findings if f.severity == "hard_fail"], findings
print("PUSCH_MIMO_RX_HOST_CONTRACT_TEST PASS (Rank1-4 observation and staged-device E2E evidence accepted)")
