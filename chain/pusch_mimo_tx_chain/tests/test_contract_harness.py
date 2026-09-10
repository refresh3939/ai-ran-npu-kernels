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
assert matrix["claim"] == "fused_single_context_tx_rank1_to_rank4_and_staged_coded_e2e_verified"
assert len(matrix["edges"]) == 10
assert matrix["stage_order"][0] == "ldpc_encode"
assert matrix["stage_order"][-1] == "ofdm_mod_batch"
assert matrix["cross_chain_dmrs_contract"]["rx_pilot_count_per_layer"] == [399, 399, 399, 399]
assert matrix["cross_chain_dmrs_contract"]["channel_est_lmmse_mimo_status"] == "supported_npu_verified"
assert any(g["id"] == "tx_combined_device_chain_and_nonzero_e2e" and
           g["status"] == "resolved" for g in matrix["hard_gates"])

findings, _ = harness.audit(matrix, KERNEL_ROOT)
assert not [f for f in findings if f.severity == "error"], findings
assert not [f for f in findings if f.severity == "hard_fail"], findings
print("PUSCH_MIMO_TX_HOST_CONTRACT_TEST PASS (fused TX plus staged coded E2E evidence accepted)")
