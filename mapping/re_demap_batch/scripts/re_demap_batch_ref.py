#!/usr/bin/env python3
"""Generate deterministic upstream-layout inputs and zero-padded batch goldens."""

import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = int(os.environ.get("RE_DEMAP_BATCH_SIZE", "8"))
N_SYMBOLS, N_FFT = 14, 2048
N_SC_USED, N_SC_PAD = 1596, 1664
P, Q = 32, 64


def used_fft_bins():
    negative = np.arange(N_FFT - N_SC_USED // 2, N_FFT, dtype=np.int64)
    nonnegative = np.arange(0, N_SC_USED - N_SC_USED // 2, dtype=np.int64)
    result = np.concatenate((negative, nonnegative))
    assert result.size == N_SC_USED
    return result


def gather_elements():
    bins = used_fft_bins()
    elements = (bins % P) * Q + bins // P
    assert elements.min() >= 0 and elements.max() < N_FFT
    assert np.unique(elements).size == N_SC_USED
    return elements.astype(np.int64)


def main():
    if not 1 <= BATCH_SIZE <= 64:
        raise ValueError("RE_DEMAP_BATCH_SIZE must be in [1,64]")

    golden_dir = ROOT / "data" / "golden"
    weights_dir = ROOT / "weights"
    golden_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    elements = gather_elements()
    gather_index = np.zeros(N_SC_PAD, dtype=np.uint32)
    gather_index[:N_SC_USED] = (elements * np.dtype(np.float16).itemsize).astype(np.uint32)
    gather_index.tofile(weights_dir / "gather_idx.bin")

    rng = np.random.default_rng(20260906)
    # The [32,64] axes document the exact stage-4 layout from ofdm_demod_batch.
    input_re = (rng.standard_normal((BATCH_SIZE, N_SYMBOLS, P, Q)) * 8).astype(np.float16)
    input_im = (rng.standard_normal((BATCH_SIZE, N_SYMBOLS, P, Q)) * 8).astype(np.float16)
    # Make element zero observably nonzero: copying it into padding must fail.
    for rx in range(BATCH_SIZE):
        for symbol in range(N_SYMBOLS):
            input_re[rx, symbol, 0, 0] = np.float16(64 + rx + symbol / 16)
            input_im[rx, symbol, 0, 0] = np.float16(-64 - rx - symbol / 16)

    flat_re = input_re.reshape(BATCH_SIZE, N_SYMBOLS, N_FFT)
    flat_im = input_im.reshape(BATCH_SIZE, N_SYMBOLS, N_FFT)
    output_re = np.zeros((BATCH_SIZE, N_SYMBOLS, N_SC_PAD), dtype=np.float16)
    output_im = np.zeros_like(output_re)
    output_re[..., :N_SC_USED] = flat_re[..., elements]
    output_im[..., :N_SC_USED] = flat_im[..., elements]

    input_re.tofile(golden_dir / "input_re.bin")
    input_im.tofile(golden_dir / "input_im.bin")
    output_re.tofile(golden_dir / "rx_grid_re.bin")
    output_im.tofile(golden_dir / "rx_grid_im.bin")

    assert np.count_nonzero(output_re[..., N_SC_USED:]) == 0
    assert np.count_nonzero(output_im[..., N_SC_USED:]) == 0
    print(f"[ref] input fp16 [{BATCH_SIZE},14,32,64] (ofdm_demod_batch stage-4)")
    print(f"[ref] output fp16 [{BATCH_SIZE},14,1664], valid=1596, padding=68 zeros")
    print(f"[ref] gather index uint32 [{N_SC_PAD}] byte offsets")


if __name__ == "__main__":
    main()
