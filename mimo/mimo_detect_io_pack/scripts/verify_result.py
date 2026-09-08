#!/usr/bin/env python3
"""Chunked, bit-exact verification of all five physical BRI outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

NR = int(os.environ.get("IO_PACK_RX_CAPACITY", "64"))
NL = int(os.environ.get("IO_PACK_LAYER_CAPACITY", "16"))
if NR not in (16, 32, 64) or NL != 16:
    raise ValueError("io-pack storage must be 16/32/64 Rx by 16 layers")
NRE = 14 * 1664
CHUNK_RE = 128
ACTIVE_LAYERS = {16: 4, 32: 8, 64: 9}[NR]


def compare(name: str, shape: tuple[int, ...], golden: Path, actual: Path) -> bool:
    expected_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    if not actual.is_file() or actual.stat().st_size != expected_bytes:
        size = actual.stat().st_size if actual.exists() else -1
        print(f"{name:10s} size={size} expected={expected_bytes} FAIL")
        return False
    # Compare storage bits, including +0 versus -0 and NaN payloads.
    expected = np.memmap(golden, dtype=np.uint16, mode="r", shape=shape)
    observed = np.memmap(actual, dtype=np.uint16, mode="r", shape=shape)
    if len(shape) == 1:
        equal = np.array_equal(expected, observed)
        mismatch = 0 if equal else int(np.count_nonzero(expected != observed))
    else:
        equal = True
        mismatch = 0
        for begin in range(0, shape[0], CHUNK_RE):
            end = min(begin + CHUNK_RE, shape[0])
            delta = expected[begin:end] != observed[begin:end]
            count = int(np.count_nonzero(delta))
            mismatch += count
            equal &= count == 0
    print(f"{name:10s} exact={equal} mismatch={mismatch} {'PASS' if equal else 'FAIL'}")
    return equal


def semantic_spot_checks(root: Path) -> bool:
    golden = root / "golden"
    output = root / "ascend_output"
    h = np.memmap(golden / "h_grid_re.bin", dtype=np.uint16, mode="r", shape=(NR, NL, NRE))
    rx = np.memmap(golden / "rx_grid_im.bin", dtype=np.uint16, mode="r", shape=(NR, NRE))
    hrm = np.memmap(output / "hrm_re.bin", dtype=np.uint16, mode="r", shape=(NRE, NR, NL))
    yv = np.memmap(output / "yvpad_im.bin", dtype=np.uint16, mode="r", shape=(NRE, NR, NL))
    indices = (0, 15, 16, 5823, 5824, 11647, 11648, 17471, 17472, NRE - 1)
    rx_indices = tuple(index for index in (0, 15, 16, 31, 32, 47, 48, 63)
                       if index < NR)
    active_axes_ok = all(
        hrm[re, rx_idx, layer] == h[rx_idx, layer, re]
        for re in indices
        for rx_idx in rx_indices
        for layer in (0, ACTIVE_LAYERS - 1)
    )
    yv_full_width_ok = all(
        np.all(yv[re, rx_idx] == rx[rx_idx, re])
        for re in indices
        for rx_idx in rx_indices
    )
    poisoned_input_ok = all(
        (h[rx_idx, layer, re] & np.uint16(0x7FFF)) != 0
        for re in indices
        for rx_idx in (0, NR - 1)
        for layer in (ACTIVE_LAYERS, 15)
    )
    inactive_positive_zero_ok = True
    for begin in range(0, NRE, CHUNK_RE):
        end = min(begin + CHUNK_RE, NRE)
        if np.any(hrm[begin:end, :, ACTIVE_LAYERS:] != np.uint16(0)):
            inactive_positive_zero_ok = False
            break

    noise = np.fromfile(golden / "noise_var_rx.bin", dtype=np.float16).astype(np.float32)
    noise_sum = np.float32(0.0)
    for value in noise:
        noise_sum = np.float32(noise_sum + value)
    expected_no = np.float16(noise_sum * np.float32(1.0 / NR)).view(np.uint16)
    observed_no = np.memmap(output / "no.bin", dtype=np.uint16, mode="r", shape=(NRE,))
    noise_mean_ok = bool(np.all(observed_no == expected_no))

    manifest = json.loads((golden / "case.json").read_text())
    plan_ok = (
        manifest.get("active_layers") == ACTIVE_LAYERS
        and [item["layer_offset"] for item in manifest.get("layer_plan", [])]
        == ([0] if NR == 16 else [0, 4] if NR == 32 else [0, 4, 7])
        and [item["num_layers"] for item in manifest.get("layer_plan", [])]
        == ([4] if NR == 16 else [4, 4] if NR == 32 else [4, 3, 2])
    )
    checks = {
        "RE/RX/active-layer axes": active_axes_ok,
        "full yvpad 16-column ABI": yv_full_width_ok,
        "inactive input poison": poisoned_input_ok,
        "inactive H positive-zero": inactive_positive_zero_ok,
        "fp32 noise mean": noise_mean_ok,
        "common layer plan": plan_ok,
    }
    for name, ok in checks.items():
        print(f"semantic   {name}={ok} {'PASS' if ok else 'FAIL'}")
    return all(checks.values())


def main() -> int:
    root = Path(os.environ.get("AIRAN_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    golden = root / "golden"
    output = root / "ascend_output"
    results = [
        compare("hrm_re", (NRE, NR, NL), golden / "hrm_re.bin", output / "hrm_re.bin"),
        compare("hrm_im", (NRE, NR, NL), golden / "hrm_im.bin", output / "hrm_im.bin"),
        compare("yvpad_re", (NRE, NR, NL), golden / "yvpad_re.bin", output / "yvpad_re.bin"),
        compare("yvpad_im", (NRE, NR, NL), golden / "yvpad_im.bin", output / "yvpad_im.bin"),
        compare("no", (NRE,), golden / "no.bin", output / "no.bin"),
        semantic_spot_checks(root),
    ]
    print(f"[verify] {'ALL PASS' if all(results) else 'FAIL'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
