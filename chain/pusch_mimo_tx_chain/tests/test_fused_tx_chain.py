#!/usr/bin/env python3
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


CHAIN = Path(__file__).resolve().parents[1]


def test_device_image_uses_explicit_production_sources():
    cmake = (CHAIN / "CMakeLists.txt").read_text(encoding="utf-8")
    required = [
        "ldpc_encode_kernel.cpp",
        "rate_match_mimo_kernel.cpp",
        "scramble_mimo_kernel.cpp",
        "qam_mod_256_mimo_kernel.cpp",
        "layer_map_kernel.cpp",
        "mimo_dmrs_gen_kernel.cpp",
        "mimo_resource_grid_map_kernel.cpp",
        "pusch_codebook_precode_kernel.cpp",
        "re_map_kernel.cpp",
        "ofdm_mod_kernel.cpp",
    ]
    assert all(name in cmake for name in required)
    assert "file(GLOB" not in cmake
    assert "experiments" not in cmake


def test_host_contains_ordered_fused_launches():
    source = (CHAIN / "tx_chain.cpp").read_text(encoding="utf-8")
    launches = [
        "ldpc_encode_kernel",
        "rm::Enqueue",
        "sm::Enqueue",
        "qm::Enqueue",
        "layer_map_kernel",
        "dg::Enqueue",
        "mimo_resource_grid_map_kernel",
        "pc::Launch",
        "re::Enqueue",
        "ofdm_mod_batch_kernel",
    ]
    execution = source.index("const auto started")
    positions = [source.index(token, execution) for token in launches]
    assert positions == sorted(positions)
    assert source.count("aclInit(") == 1
    assert source.count("aclrtCreateStream(") == 1


def test_receipt_verifier_accepts_exact_final_iq_contract():
    with tempfile.TemporaryDirectory(prefix="mimo-tx-verifier-") as directory:
        root = Path(directory)
        iq_path = root / "tx_iq.bin"
        receipt_path = root / "receipt.json"
        iq = np.ones(1 * 1 * 30720 * 2, dtype=np.int16)
        iq.tofile(iq_path)
        receipt_path.write_text(
            json.dumps(
                {
                    "schema": "airan.pusch_mimo.tx_chain.v1",
                    "rank": 1,
                    "tx_ports": 1,
                    "slots": 1,
                    "output_bytes": iq.nbytes,
                    "nonzero_i16": iq.size,
                    "single_acl_context": True,
                    "shared_device_arena": True,
                    "intermediate_host_files": False,
                    "status": "PASS",
                }
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                "python3",
                str(CHAIN / "scripts" / "verify_tx_chain.py"),
                "--receipt",
                str(receipt_path),
                "--iq",
                str(iq_path),
            ],
            check=True,
        )
