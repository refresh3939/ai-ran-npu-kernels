"""Bit-exact verifier for scramble_siso grouped/padded output."""

import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scramble_ref import (  # noqa: E402
    N_SLOT_TB, N_STREAMS, N_DATA_SYM, N_SC_USED, QAM_SYM_STRIDE,
    N_SYM_PAD, scramble_ref,
)

INPUT_SHAPE = (N_SLOT_TB, N_STREAMS, N_SYM_PAD)
OUTPUT_SHAPE = (N_SLOT_TB, N_STREAMS, N_DATA_SYM, QAM_SYM_STRIDE)
SENTINEL = np.uint16(0xAAAA).view(np.int16)
N_ID_CELL, N_RNTI, Q = 1, 12345, 0


def data_root() -> Path:
    return Path(os.getenv("AIRAN_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))


def load(path: Path, shape) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.fromfile(path, dtype=np.int16)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path}: {data.size} elements, expected {expected}")
    return data.reshape(shape)


def main() -> int:
    root = data_root()
    golden_dir = root / "golden"
    try:
        golden = load(golden_dir / "golden.bin", OUTPUT_SHAPE)
        out = load(root / "ascend_output" / "output.bin", OUTPUT_SHAPE)
        inp = load(golden_dir / "input.bin", INPUT_SHAPE)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    recomputed, _ = scramble_ref(inp, N_ID_CELL, N_RNTI, Q)
    if not np.array_equal(recomputed, golden):
        print(f"[FAIL] dirty golden: {np.count_nonzero(recomputed != golden)} mismatches")
        return 1

    sentinel_count = int(np.count_nonzero(out == SENTINEL))
    padding_nonzero = int(np.count_nonzero(out[:, :, :, N_SC_USED:]))
    diff = out.astype(np.int32) - golden.astype(np.int32)
    errors = diff != 0
    error_count = int(errors.sum())
    max_error = int(np.abs(diff).max()) if error_count else 0
    print(f"input={INPUT_SHAPE} output={OUTPUT_SHAPE}")
    print(f"sentinel={sentinel_count} padding_nonzero={padding_nonzero}")
    print(f"max_err={max_error} err_count={error_count}/{out.size}")

    if error_count:
        slot, stream, symbol, sc = np.argwhere(errors)[0]
        print(f"first error: slot={slot} qam_stream={stream} symbol={symbol} sc={sc} "
              f"got={out[slot, stream, symbol, sc]} golden={golden[slot, stream, symbol, sc]}")
        print(f"per-stream errors: {errors.sum(axis=(0, 2, 3)).tolist()}")
    if sentinel_count or padding_nonzero or error_count:
        print("[FAIL]")
        return 1
    print("[PASS] bit-exact grouped/padded output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
