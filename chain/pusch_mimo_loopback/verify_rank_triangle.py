#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    root, rank = args.root, args.rank
    info = np.fromfile(root / "artifacts/tx_bits.bin", np.int8)
    if info.size != 143 * 8448 or not np.any(info):
        raise RuntimeError("tx_bits missing/wrong-size/all-zero")
    matched_path = root / f"decoder_rank{rank}/output_matched/decoded_bits.bin"
    wrong_path = root / f"decoder_rank{rank}/output_wrong/decoded_bits.bin"
    matched, wrong = np.fromfile(matched_path, np.int8), np.fromfile(wrong_path, np.int8)
    if matched.size != info.size or wrong.size != info.size:
        raise RuntimeError("decoder exact output shape mismatch")
    if not np.all((matched == 0) | (matched == 1)) or not np.all((wrong == 0) | (wrong == 1)):
        raise RuntimeError("decoder output is not binary")
    matched_errors = int(np.count_nonzero(matched != info))
    wrong_errors = int(np.count_nonzero(wrong != info))
    wrong_ber = wrong_errors / info.size
    passed = matched_errors == 0 and 0.4 <= wrong_ber <= 0.6
    result = {"schema": "airan.pusch_mimo.triangle.v1",
              "status": "PASS" if passed else "HARD_FAIL", "rank": rank, "slots": 23,
              "matched_cell_id": 321, "wrong_cell_id": 322,
              "tx_bits_nonzero": int(np.count_nonzero(info)),
              "tx_bits_sha256": sha(root / "artifacts/tx_bits.bin"),
              "matched_info_errors": matched_errors,
              "matched_info_ber": matched_errors / info.size,
              "wrong_cell_info_errors": wrong_errors, "wrong_cell_info_ber": wrong_ber,
              "ldpc_input_stride": 26112,
              "actual_npu_stages": ["descramble_mimo", "rate_dematch_mimo", "ldpc_decode"],
              "complete_coded_e2e": passed}
    output = root / f"artifacts/rank{rank}_triangle_result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not passed: raise SystemExit(1)


if __name__ == "__main__":
    main()
