#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SLOTS, QM, VALID, STRIDE = 23, 8, 19152, 19200


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    source = np.memmap(root / "scramble/ascend_output/rank1/bits_qam.bin",
                       dtype=np.int16, mode="r", shape=(SLOTS, QM, STRIDE))
    result_dir = root / "artifacts/demod_23slot"
    combined_path = result_dir / "cw_llr_23slot.bin"
    total_errors = 0
    total_bits = 0
    per_slot = []
    with combined_path.open("wb") as combined:
        for slot in range(SLOTS):
            path = result_dir / f"slot{slot:02d}_cw_llr.bin"
            value = np.fromfile(path, dtype=np.int16)
            if value.size != QM * STRIDE:
                raise RuntimeError(f"slot {slot}: cw_llr size mismatch")
            value = value.reshape(QM, STRIDE)
            if np.any(value[:, VALID:]):
                raise RuntimeError(f"slot {slot}: layer_demap tail is nonzero")
            if not np.any(value[:, :VALID]):
                raise RuntimeError(f"slot {slot}: all-zero valid LLR")
            hard = value[:, :VALID] < 0
            errors = int(np.count_nonzero(hard != source[slot, :, :VALID]))
            per_slot.append(errors)
            total_errors += errors
            total_bits += QM * VALID
            value.tofile(combined)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "rank": 1,
        "slots": SLOTS,
        "device_stages": ["channel_est_lmmse_mimo", "mimo_detect_io_pack",
                          "mimo_detect_bri_batch", "qam_demod_256_mimo_batch",
                          "layer_demap_mimo"],
        "raw_qam_bit_errors": total_errors,
        "raw_qam_ber": total_errors / total_bits,
        "per_slot_errors": per_slot,
        "cw_llr_sha256": digest(combined_path),
        "complete_coded_e2e": False,
    }
    (result_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
