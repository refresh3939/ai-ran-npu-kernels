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
    parser.add_argument("--rank", type=int, required=True, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    rank = args.rank
    source = np.memmap(args.root / f"scramble/ascend_output/rank{rank}/bits_qam.bin",
                       np.int16, "r", shape=(SLOTS, QM, rank * STRIDE))
    result_dir = args.root / f"artifacts/demod_rank{rank}_23slot"
    combined_path = result_dir / "cw_llr_23slot.bin"
    total_errors = 0
    per_slot = []
    with combined_path.open("wb") as combined:
        for slot in range(SLOTS):
            path = result_dir / f"slot{slot:02d}_cw_llr.bin"
            value = np.fromfile(path, np.int16)
            if value.size != QM * rank * STRIDE:
                raise RuntimeError(f"slot {slot}: Rank{rank} cw_llr exact size mismatch")
            value = value.reshape(QM, rank * STRIDE)
            if np.any(value[:, rank * VALID:]) or not np.any(value[:, :rank * VALID]):
                raise RuntimeError(f"slot {slot}: Rank{rank} padding/all-zero LLR failure")
            errors = int(np.count_nonzero((value[:, :rank * VALID] < 0) !=
                                          source[slot, :, :rank * VALID]))
            per_slot.append(errors); total_errors += errors; value.tofile(combined)
    total_bits = SLOTS * QM * rank * VALID
    summary = {"schema": "airan.pusch_mimo.rx_front.v1", "status": "PASS",
               "rank": rank, "slots": SLOTS,
               "actual_npu_stages": ["channel_est_lmmse_mimo", "mimo_detect_io_pack",
                   "mimo_detect_bri_batch", "qam_demod_256_mimo_batch", "layer_demap_mimo"],
               "raw_qam_bit_errors": total_errors, "raw_qam_ber": total_errors / total_bits,
               "per_slot_errors": per_slot, "cw_llr_sha256": digest(combined_path)}
    (result_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
