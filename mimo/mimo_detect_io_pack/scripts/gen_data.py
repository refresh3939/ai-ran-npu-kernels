#!/usr/bin/env python3
"""Generate bit-exact full-slot vectors for mimo_detect_io_pack."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

NR = int(os.environ.get("IO_PACK_RX_CAPACITY", "64"))
NL = int(os.environ.get("IO_PACK_LAYER_CAPACITY", "16"))
if NR not in (16, 32, 64) or NL != 16:
    raise ValueError("io-pack storage must be 16/32/64 Rx by 16 layers")
NSYM = 14
NSC_PAD = 1664
NRE = NSYM * NSC_PAD
CHUNK_RE = 128
ALLOCATION_LAYERS = (4,)
ALLOCATION_PORTS = (4,)
ALLOCATION_OFFSETS = tuple(sum(ALLOCATION_LAYERS[:index])
                           for index in range(len(ALLOCATION_LAYERS)))
ACTIVE_LAYERS = sum(ALLOCATION_LAYERS)


def raw_memmap(path: Path, shape: tuple[int, ...], mode: str = "w+") -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=np.float16, mode=mode, shape=shape)


def copy_raw(source: Path, destination: np.memmap) -> None:
    expected = destination.size * destination.dtype.itemsize
    if not source.is_file() or source.stat().st_size != expected:
        raise ValueError(f"upstream size mismatch: {source} expected={expected} bytes")
    upstream = np.memmap(source, dtype=np.float16, mode="r", shape=destination.shape)
    for rx in range(NR):
        destination[rx] = upstream[rx]
    destination.flush()


def generate_input(golden: Path, upstream_h_dir: Path | None) -> str:
    rng = np.random.default_rng(20260905)
    rx_re = raw_memmap(golden / "rx_grid_re.bin", (NR, NRE))
    rx_im = raw_memmap(golden / "rx_grid_im.bin", (NR, NRE))
    for rx in range(NR):
        rx_re[rx] = rng.standard_normal(NRE).astype(np.float16)
        rx_im[rx] = rng.standard_normal(NRE).astype(np.float16)
    rx_re.flush()
    rx_im.flush()

    h_re = raw_memmap(golden / "h_grid_re.bin", (NR, NL, NRE))
    h_im = raw_memmap(golden / "h_grid_im.bin", (NR, NL, NRE))
    source = "deterministic_random"
    if upstream_h_dir is not None:
        candidates = [
            ("h_re.bin", "h_im.bin"),
            ("h_cube_time_post_re.bin", "h_cube_time_post_im.bin"),
            ("h_cube_time_fused_re.bin", "h_cube_time_fused_im.bin"),
            ("gold_h_re.bin", "gold_h_im.bin"),
        ]
        pair = next(
            ((upstream_h_dir / re_name, upstream_h_dir / im_name)
             for re_name, im_name in candidates
             if (upstream_h_dir / re_name).is_file() and (upstream_h_dir / im_name).is_file()),
            None,
        )
        if pair is None:
            raise FileNotFoundError(f"no supported H plane pair under {upstream_h_dir}")
        copy_raw(pair[0], h_re)
        copy_raw(pair[1], h_im)
        source = str(upstream_h_dir.resolve())
    else:
        for rx in range(NR):
            for layer in range(NL):
                h_re[rx, layer] = (0.25 * rng.standard_normal(NRE)).astype(np.float16)
                h_im[rx, layer] = (0.25 * rng.standard_normal(NRE)).astype(np.float16)
        h_re.flush()
        h_im.flush()

    # Inactive physical columns are intentionally poisoned. The adapter, not
    # the channel estimator or test data, must establish the BRI zero-padding
    # contract for [active_layers, 16).
    for layer in range(ACTIVE_LAYERS, NL):
        h_re[:, layer, :] = np.float16(0.5 + layer / 64.0)
        h_im[:, layer, :] = np.float16(-0.75 - layer / 64.0)
    h_re.flush()
    h_im.flush()

    # Binary fractions make the fp32 mean and final fp16 conversion exact.
    noise = (np.arange(1, NR + 1, dtype=np.float32) / 1024.0).astype(np.float16)
    noise.tofile(golden / "noise_var_rx.bin")
    return source


def generate_expected(golden: Path) -> None:
    rx_re = raw_memmap(golden / "rx_grid_re.bin", (NR, NRE), "r")
    rx_im = raw_memmap(golden / "rx_grid_im.bin", (NR, NRE), "r")
    h_re = raw_memmap(golden / "h_grid_re.bin", (NR, NL, NRE), "r")
    h_im = raw_memmap(golden / "h_grid_im.bin", (NR, NL, NRE), "r")
    hrm_re = raw_memmap(golden / "hrm_re.bin", (NRE, NR, NL))
    hrm_im = raw_memmap(golden / "hrm_im.bin", (NRE, NR, NL))
    yvpad_re = raw_memmap(golden / "yvpad_re.bin", (NRE, NR, NL))
    yvpad_im = raw_memmap(golden / "yvpad_im.bin", (NRE, NR, NL))

    for begin in range(0, NRE, CHUNK_RE):
        end = min(begin + CHUNK_RE, NRE)
        hrm_re[begin:end] = h_re[:, :, begin:end].transpose(2, 0, 1)
        hrm_im[begin:end] = h_im[:, :, begin:end].transpose(2, 0, 1)
        hrm_re[begin:end, :, ACTIVE_LAYERS:] = np.float16(0.0)
        hrm_im[begin:end, :, ACTIVE_LAYERS:] = np.float16(0.0)
        yvpad_re[begin:end] = rx_re[:, begin:end].T[:, :, None]
        yvpad_im[begin:end] = rx_im[:, begin:end].T[:, :, None]
    for value in (hrm_re, hrm_im, yvpad_re, yvpad_im):
        value.flush()

    noise = np.fromfile(golden / "noise_var_rx.bin", dtype=np.float16).astype(np.float32)
    mean = np.float16(noise.sum(dtype=np.float32) / np.float32(NR))
    np.full(NRE, mean, dtype=np.float16).tofile(golden / "no.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-h-dir", type=Path)
    args = parser.parse_args()
    root = Path(os.environ.get("AIRAN_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    golden = root / "golden"
    golden.mkdir(parents=True, exist_ok=True)
    source = generate_input(golden, args.upstream_h_dir)
    generate_expected(golden)
    manifest = {
        "abi": 1,
        "block_dim": 4,
        "rx_grid_shape": [NR, NSYM, NSC_PAD],
        "h_grid_shape": [NR, NL, NSYM, NSC_PAD],
        "packed_shape": [NRE, NR, NL],
        "noise_shape": [NR],
        "h_source": source,
        "inactive_h_input": "deterministic_nonzero_poison",
        "active_layers": ACTIVE_LAYERS,
        "layer_plan": [
            {
                "pusch_index": index,
                "layer_offset": ALLOCATION_OFFSETS[index],
                "num_layers": ALLOCATION_LAYERS[index],
                "num_tx_ports": ALLOCATION_PORTS[index],
            }
            for index in range(len(ALLOCATION_LAYERS))
        ],
    }
    (golden / "case.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[golden] {golden}")
    print(
        f"[contract] H source={source}; [{NR},16,14,1664] -> [23296,{NR},16]; "
        f"activeL={ACTIVE_LAYERS}, offsets={ALLOCATION_OFFSETS}"
    )


if __name__ == "__main__":
    main()
