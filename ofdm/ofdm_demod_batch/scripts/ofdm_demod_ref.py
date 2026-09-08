#!/usr/bin/env python3
"""Generate weights and deterministic batch goldens for ofdm_demod_batch."""
import importlib.util
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
KERNEL_DIR = SCRIPT_DIR.parent
BASE_REF = KERNEL_DIR.parent / "ofdm_demod_siso" / "scripts" / "ofdm_demod_ref.py"

spec = importlib.util.spec_from_file_location("ofdm_demod_single_ref", BASE_REF)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

BATCH_SIZE = int(os.environ.get("OFDM_BATCH_SIZE", "8"))
if BATCH_SIZE <= 0:
    raise ValueError("OFDM_BATCH_SIZE must be positive")


def cmm(ar, ai, br, bi, axes):
    return (np.einsum(axes, ar, br) - np.einsum(axes, ai, bi),
            np.einsum(axes, ar, bi) + np.einsum(axes, ai, br))


def main():
    golden_dir = KERNEL_DIR / "data" / "golden"
    weights_dir = KERNEL_DIR / "weights"
    golden_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Use the exact fp16 values consumed by Cube/vector instructions.
    w32r = ref.U_DFT_P.real.astype(np.float16)
    w32i = ref.U_DFT_P.imag.astype(np.float16)
    w64r = ref.U_DFT_Q.real.astype(np.float16)
    w64i = ref.U_DFT_Q.imag.astype(np.float16)
    twr = ref.T_PQ.real.astype(np.float16)
    twi = ref.T_PQ.imag.astype(np.float16)
    w32r.tofile(weights_dir / "w_dft32_re.bin")
    w32i.tofile(weights_dir / "w_dft32_im.bin")
    w64r.tofile(weights_dir / "w_dft64_re.bin")
    w64i.tofile(weights_dir / "w_dft64_im.bin")
    w64r.T.tofile(weights_dir / "w_dft64_re_T.bin")
    w64i.T.tofile(weights_dir / "w_dft64_im_T.bin")
    twr.tofile(weights_dir / "twiddle_pq_re.bin")
    twi.tofile(weights_dir / "twiddle_pq_im.bin")

    rng = np.random.default_rng(seed=12345)
    x = ((rng.standard_normal((BATCH_SIZE, ref.N_SAMPLE_PER_SLOT)) +
          1j * rng.standard_normal((BATCH_SIZE, ref.N_SAMPLE_PER_SLOT))) /
         np.sqrt(2.0)).astype(np.complex64)
    in_re = np.round(x.real * ref.Q_SCALE).astype(np.int16)
    in_im = np.round(x.imag * ref.Q_SCALE).astype(np.int16)
    input_iq = np.empty((BATCH_SIZE, ref.N_SAMPLE_PER_SLOT, 2), dtype=np.int16)
    input_iq[..., 0] = in_re
    input_iq[..., 1] = in_im
    input_iq.tofile(golden_dir / "input.bin")

    # Golden starts from the quantized/de-quantized signal actually seen by NPU.
    xq = (in_re.astype(np.float32) + 1j * in_im.astype(np.float32)) / ref.Q_SCALE
    xcp = ref.cp_remove(xq)
    xr = xcp.real.reshape(BATCH_SIZE, ref.N_SYMBOL_PER_SLOT, ref.P, ref.Q)
    xi = xcp.imag.reshape(BATCH_SIZE, ref.N_SYMBOL_PER_SLOT, ref.P, ref.Q)
    s2r, s2i = cmm(w32r.astype(np.float32), w32i.astype(np.float32), xr, xi,
                   "ab,...bc->...ac")
    s3r = s2r * twr.astype(np.float32) - s2i * twi.astype(np.float32)
    s3i = s2r * twi.astype(np.float32) + s2i * twr.astype(np.float32)
    s4r, s4i = cmm(s3r, s3i, w64r.T.astype(np.float32), w64i.T.astype(np.float32),
                   "...ab,bc->...ac")

    (s2r + 1j * s2i).astype(np.complex64).tofile(golden_dir / "stage2_dft32.bin")
    (s3r + 1j * s3i).astype(np.complex64).tofile(golden_dir / "stage3_twiddle.bin")
    (s4r + 1j * s4i).astype(np.complex64).tofile(golden_dir / "stage4_dft64.bin")
    print(f"[ref] batch={BATCH_SIZE} input={input_iq.shape} output={(BATCH_SIZE, ref.N_SYMBOL_PER_SLOT, ref.P, ref.Q)}")
    print(f"[ref] weights -> {weights_dir}")
    print(f"[ref] goldens -> {golden_dir}")


if __name__ == "__main__":
    main()
