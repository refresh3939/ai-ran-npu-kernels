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
assert matrix["claim"] == "fused_rank4_and_staged_rank1_to_rank4_device_e2e_verified"
assert len(matrix["edges"]) == 10
assert matrix["stage_order"][0] == "ofdm_demod_batch"
assert matrix["stage_order"][-1] == "ldpc_decode"
assert "mimo_dmrs_gen" in matrix["stage_order"]

assert [profile["pilot_count_per_layer"] for profile in matrix["observation_profiles"]] == [
    [798], [798, 798], [399, 399, 798], [399, 399, 399, 399]
]
assert matrix["observation_contract"]["ce_output_shape"] == ["NR", 16, 14, 1664]
assert matrix["observation_contract"]["io_pack_input_shape"] == ["NR", 16, 14, 1664]
assert matrix["observation_contract"]["allow_399_to_798_synthesis"] is False
assert [g["id"] for g in matrix["hard_gates"]] == [
    "rx_combined_device_chain_and_nonzero_e2e",
    "rx_rank4_single_process_fused_npu",
]
assert all(g["status"] == "resolved" for g in matrix["hard_gates"])

findings, _ = harness.audit(matrix, KERNEL_ROOT)
assert not [f for f in findings if f.severity == "error"], findings
assert not [f for f in findings if f.severity == "hard_fail"], findings

source = (CHAIN_DIR / "rx_chain.cpp").read_text(encoding="utf-8")
build = (CHAIN_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
assert source.count("aclInit(nullptr)") == 1
assert source.count("aclrtCreateStream(&stream)") == 1
assert "ACL_MEMCPY_DEVICE_TO_HOST" in source
assert source.count("ACL_MEMCPY_DEVICE_TO_HOST") == 1
assert "AIRAN_FUSED_RX_GROUPED_Y=1" in build
for kernel in (
    "ofdm_demod_kernel.cpp", "re_demap_batch_kernel.cpp",
    "mimo_dmrs_gen_kernel.cpp", "mimo_dmrs_ls_kernel.cpp",
    "channel_est_lmmse_kernel.cpp", "mimo_detect_io_pack_kernel.cpp",
    "mimo_detect_bri_kernel.cpp", "qam256_demod_batch_kernel.cpp",
    "layer_demap_kernel.cpp", "descramble_mimo_kernel.cpp",
    "rate_dematch_mimo_kernel.cpp", "ldpc_decode_kernel.cpp",
):
    assert kernel in build, kernel
print("PUSCH_MIMO_RX_HOST_CONTRACT_TEST PASS (fused Rank4 plus staged Rank1-4 evidence accepted)")
