#!/usr/bin/env python3
"""Bit-exact validation of the runtime-batch RE demapper and its zero tail."""

import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("AIRAN_DATA_DIR") or Path(__file__).resolve().parent.parent)
BATCH_SIZE = int(os.environ.get("RE_DEMAP_BATCH_SIZE", "8"))
N_SYMBOLS, N_SC_USED, N_SC_PAD = 14, 1596, 1664


def load(path, shape):
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.fromfile(path, dtype=np.float16)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path}: got {values.size} elements, expected {expected}")
    return values.reshape(shape)


def main():
    shape = (BATCH_SIZE, N_SYMBOLS, N_SC_PAD)
    try:
        golden_re = load(ROOT / "data/golden/rx_grid_re.bin", shape)
        golden_im = load(ROOT / "data/golden/rx_grid_im.bin", shape)
        actual_re = load(ROOT / "data/ascend_output/rx_grid_re.bin", shape)
        actual_im = load(ROOT / "data/ascend_output/rx_grid_im.bin", shape)
    except (FileNotFoundError, ValueError) as error:
        print(f"[verify] {error}")
        return 1

    mismatch = (actual_re.view(np.uint16) != golden_re.view(np.uint16)) | \
               (actual_im.view(np.uint16) != golden_im.view(np.uint16))
    padding_nonzero = np.count_nonzero(actual_re[..., N_SC_USED:].view(np.uint16)) + \
                      np.count_nonzero(actual_im[..., N_SC_USED:].view(np.uint16))

    passed_symbols = 0
    for rx in range(BATCH_SIZE):
        bad_by_symbol = np.count_nonzero(mismatch[rx], axis=1)
        passed_symbols += int(np.count_nonzero(bad_by_symbol == 0))
        print(f"  rx {rx:2d}: mismatches={int(bad_by_symbol.sum()):5d}, "
              f"symbols_ok={int(np.count_nonzero(bad_by_symbol == 0)):2d}/14")

    expected_symbols = BATCH_SIZE * N_SYMBOLS
    print(f"[verify] padding nonzero fp16 words: {padding_nonzero}")
    if passed_symbols == expected_symbols and padding_nonzero == 0:
        print(f"[verify] ========== PASS ({passed_symbols}/{expected_symbols}) ==========")
        return 0
    print(f"[verify] ========== FAIL ({passed_symbols}/{expected_symbols}) ==========")
    return 1


if __name__ == "__main__":
    sys.exit(main())
