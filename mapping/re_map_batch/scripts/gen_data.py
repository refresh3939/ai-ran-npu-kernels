#!/usr/bin/env python3
"""Generate independent P=1/2/4 RE-map vectors at the chain boundary."""
from pathlib import Path

import numpy as np

N_SYMBOLS = 14
N_FFT = 2048
N_SC_USED = 1596
N_SC_PAD = 1664
RADIX_P = 32
RADIX_Q = 64
PORTS = (1, 2, 4)


def scatter_index() -> np.ndarray:
    negative = np.arange(N_FFT - N_SC_USED // 2, N_FFT, dtype=np.int64)
    nonnegative = np.arange(0, N_SC_USED - N_SC_USED // 2, dtype=np.int64)
    natural = np.concatenate((negative, nonnegative))
    assert natural.size == N_SC_USED
    destinations = (natural % RADIX_P) * RADIX_Q + natural // RADIX_P
    index = np.full(N_FFT, N_SC_PAD, dtype=np.int64)
    index[destinations] = np.arange(N_SC_USED, dtype=np.int64)
    assert np.unique(destinations).size == N_SC_USED
    assert np.count_nonzero(index == N_SC_PAD) == N_FFT - N_SC_USED
    return index


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "golden"
    rng = np.random.default_rng(0x5EBA7C)
    index = scatter_index()
    for ports in PORTS:
        case = root / f"port{ports}"
        case.mkdir(parents=True, exist_ok=True)
        # A port-dependent bias makes cross-port offset mistakes obvious.
        real = np.zeros((ports, N_SYMBOLS, N_SC_PAD), dtype=np.float16)
        imag = np.zeros_like(real)
        for port in range(ports):
            real[port, :, :N_SC_USED] = (
                rng.normal(0.0, 2.0, (N_SYMBOLS, N_SC_USED)) + 16.0 * port
            ).astype(np.float16)
            imag[port, :, :N_SC_USED] = (
                rng.normal(0.0, 2.0, (N_SYMBOLS, N_SC_USED)) - 16.0 * port
            ).astype(np.float16)
        # Poison upstream storage padding. The adapter/kernel must not leak it.
        real[:, :, N_SC_USED:] = np.float16(123.0)
        imag[:, :, N_SC_USED:] = np.float16(-117.0)
        zero = np.zeros((ports, N_SYMBOLS, 1), dtype=np.float16)
        real_fft = np.concatenate((real, zero), axis=2)[:, :, index]
        imag_fft = np.concatenate((imag, zero), axis=2)[:, :, index]

        real.tofile(case / "port_grid_re.bin")
        imag.tofile(case / "port_grid_im.bin")
        real_fft.tofile(case / "fft_grid_re.bin")
        imag_fft.tofile(case / "fft_grid_im.bin")
    print(f"generated re_map_batch P=1/2/4 cases in {root}")


if __name__ == "__main__":
    main()
