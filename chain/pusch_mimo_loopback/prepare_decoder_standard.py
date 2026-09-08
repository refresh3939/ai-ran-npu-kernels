#!/usr/bin/env python3
"""Derive the decoder QC table directly from the same Sionna BG1 PCM as TX."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sionna.phy.fec.ldpc import LDPC5GEncoder

Z, KB, MB, NFULL = 384, 22, 46, 68


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    encoder = LDPC5GEncoder(k=8448, n=25344)
    if getattr(encoder, "_z", None) != Z or str(getattr(encoder, "_bg", "")).lower() != "bg1":
        raise RuntimeError("Sionna did not select BG1/Z384")
    pcm = getattr(encoder, "_pcm", None)
    if pcm is None or pcm.shape != (MB * Z, NFULL * Z):
        raise RuntimeError("unexpected Sionna PCM")
    table = np.full((MB, NFULL), -1, dtype=np.int16)
    for br in range(MB):
        for bc in range(NFULL):
            block = pcm[br*Z:(br+1)*Z, bc*Z:(bc+1)*Z].tocsr()
            if block.nnz == 0:
                continue
            if block.nnz != Z:
                raise RuntimeError(f"PCM block ({br},{bc}) is not a permutation")
            row0 = block.getrow(0).indices
            if row0.size != 1:
                raise RuntimeError(f"PCM block ({br},{bc}) row0 is not singleton")
            shift = int(row0[0])
            for row in range(Z):
                if not np.array_equal(block.getrow(row).indices,
                                      np.asarray([(row + shift) % Z])):
                    raise RuntimeError(f"PCM block ({br},{bc}) is not circ({shift})")
            table[br, bc] = shift

    info = np.fromfile(work / "artifacts/tx_bits.bin", dtype=np.int8).reshape(143, 8448)
    encoded = np.fromfile(work / "ldpc/ascend_output/output.bin", dtype=np.int8).reshape(143, 25344)
    full = np.concatenate((info[:, :2*Z], encoded), axis=1).reshape(143, NFULL, Z)
    row_index = np.arange(Z)
    for br in range(MB):
        syndrome = np.zeros((143, Z), dtype=np.int8)
        for bc in range(NFULL):
            shift = int(table[br, bc])
            if shift >= 0:
                syndrome ^= full[:, bc, :][:, (row_index + shift) % Z]
        if np.any(syndrome):
            raise RuntimeError(f"actual device encoder output fails Sionna PCM row block {br}")

    out = work / "decoder/weights/ldpc_bg1_z384_shifts"
    data = work / "decoder/data"
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    table.tofile(out / "shift_table.bin")
    degrees = np.count_nonzero(table >= 0, axis=1).astype(np.int16)
    edge_offsets = np.zeros(MB + 1, dtype=np.int32)
    edge_offsets[1:] = np.cumsum(degrees, dtype=np.int32)
    degrees.tofile(data / "degrees.bin")
    edge_offsets.tofile(data / "edge_offsets.bin")
    manifest = {
        "schema_version": 1,
        "source": "Sionna LDPC5GEncoder(k=8448,n=25344) sparse PCM",
        "base_graph": 1,
        "lifting_size": Z,
        "shape": [MB, NFULL],
        "edges": int(degrees.sum()),
        "max_degree": int(degrees.max()),
        "shift_table_sha256": sha(out / "shift_table.bin"),
        "actual_device_encoder_syndrome_nonzero": 0,
    }
    (work / "decoder/standard_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
