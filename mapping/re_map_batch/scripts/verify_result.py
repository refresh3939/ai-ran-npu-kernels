#!/usr/bin/env python3
"""Verify P=1/2/4 RE-map output and chain layout bit-exactly."""
from pathlib import Path

import numpy as np

from gen_data import N_FFT, N_SC_PAD, N_SC_USED, N_SYMBOLS, PORTS, scatter_index


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    index = scatter_index()
    guard = index == N_SC_PAD
    failures = 0
    for ports in PORTS:
        name = f"port{ports}"
        shape = (ports, N_SYMBOLS, N_FFT)
        actual_re = np.fromfile(
            root / "data" / "ascend_output" / name / "fft_grid_re.bin",
            dtype=np.uint16,
        )
        actual_im = np.fromfile(
            root / "data" / "ascend_output" / name / "fft_grid_im.bin",
            dtype=np.uint16,
        )
        expected_re = np.fromfile(
            root / "data" / "golden" / name / "fft_grid_re.bin", dtype=np.uint16
        )
        expected_im = np.fromfile(
            root / "data" / "golden" / name / "fft_grid_im.bin", dtype=np.uint16
        )
        size_ok = all(
            item.size == int(np.prod(shape))
            for item in (actual_re, actual_im, expected_re, expected_im)
        )
        exact = size_ok and np.array_equal(actual_re, expected_re) and np.array_equal(
            actual_im, expected_im
        )
        guard_zero = False
        round_trip = False
        if size_ok:
            actual_re = actual_re.reshape(shape)
            actual_im = actual_im.reshape(shape)
            guard_zero = bool(
                np.all(actual_re[:, :, guard] == 0)
                and np.all(actual_im[:, :, guard] == 0)
            )
            input_re = np.fromfile(
                root / "data" / "golden" / name / "port_grid_re.bin", dtype=np.uint16
            ).reshape(ports, N_SYMBOLS, N_SC_PAD)
            input_im = np.fromfile(
                root / "data" / "golden" / name / "port_grid_im.bin", dtype=np.uint16
            ).reshape(ports, N_SYMBOLS, N_SC_PAD)
            used_destinations = np.flatnonzero(~guard)
            sources = index[used_destinations]
            recovered_re = np.empty((ports, N_SYMBOLS, N_SC_USED), dtype=np.uint16)
            recovered_im = np.empty_like(recovered_re)
            recovered_re[:, :, sources] = actual_re[:, :, used_destinations]
            recovered_im[:, :, sources] = actual_im[:, :, used_destinations]
            round_trip = bool(
                np.array_equal(recovered_re, input_re[:, :, :N_SC_USED])
                and np.array_equal(recovered_im, input_im[:, :, :N_SC_USED])
            )
        ok = size_ok and exact and guard_zero and round_trip
        print(
            f"{name:5s} size={size_ok} exact={exact} guard_zero={guard_zero} "
            f"demap_round_trip={round_trip} {'PASS' if ok else 'FAIL'}"
        )
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
