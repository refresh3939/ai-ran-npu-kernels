#!/usr/bin/env python3
"""Materialize the immutable inputs needed by the fused Rank4 RX executable."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


CE_FILES = (
    "factor_b_re.bin",
    "factor_b_im.bin",
    "cube_time_fused_d0_re.bin",
    "cube_time_fused_d0_im.bin",
    "cube_time_fused_d1_re.bin",
    "cube_time_fused_d1_im.bin",
    "ce_pack_gather_index.bin",
    "weight_model.bin",
)
OFDM_FILES = (
    "w_dft32_re.bin",
    "w_dft32_im.bin",
    "w_dft64_re_T.bin",
    "w_dft64_im_T.bin",
    "twiddle_pq_re.bin",
    "twiddle_pq_im.bin",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def copy_checked(source: Path, target: Path, expected_bytes: int | None = None) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"missing/empty staged input: {source}")
    if expected_bytes is not None and source.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"wrong size for {source}: {source.stat().st_size}, expected {expected_bytes}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.staged_root.resolve()
    output = args.output.resolve()

    # 23 slots x 64 RX x 30720 complex values x 2 interleaved int16 lanes.
    copy_checked(root / "artifacts/rx_iq_rank4_23slot.bin", output / "rx_iq.bin",
                 23 * 64 * 30720 * 2 * 2)
    # 143 code blocks x 8448 information bits, one int8 per bit.
    copy_checked(root / "artifacts/tx_bits.bin", output / "expected_bits.bin",
                 143 * 8448)

    for name in OFDM_FILES:
        copy_checked(root / "ofdm_demod/weights" / name,
                     output / "weights/ofdm_demod" / name)

    ce = root / "channel_est_rank4/golden/case_0_m64_k4_r96"
    for name in CE_FILES:
        copy_checked(ce / name, output / "weights/channel_est" / name)

    copy_checked(root / "decoder/weights/ldpc_bg1_z384_shifts/shift_table.bin",
                 output / "weights/ldpc_decode/shift_table.bin")
    for name in ("degrees.bin", "edge_offsets.bin"):
        copy_checked(root / "decoder/data" / name,
                     output / "weights/ldpc_decode" / name)

    files = sorted(path for path in output.rglob("*.bin") if path.is_file())
    manifest = {
        "schema": "airan.pusch_mimo.fused_rx.assets.v1",
        "profile": "fd16x64",
        "rank": 4,
        "slots": 23,
        "files": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[PASS] prepared {len(files)} fused RX inputs at {output}")


if __name__ == "__main__":
    main()
