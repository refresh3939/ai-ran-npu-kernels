#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NSYM, NSPAD, NSC, NRE, QM = 14, 1664, 1596, 19152, 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument("--detector-rx-capacity", type=int, default=64)
    parser.add_argument("--detector-layer-capacity", type=int, default=16)
    args = parser.parse_args()
    root, rank = args.root, args.rank
    detector_case = (f"case_0_m{args.detector_rx_capacity}_"
                     f"k{args.detector_layer_capacity}_sc1664_r{rank}")
    detected = root / f"detect_rank{rank}/data/ascend_output/{detector_case}"
    shape = (args.detector_layer_capacity, NSYM, NSPAD)
    xr = np.fromfile(detected / "xhat_re.bin", np.float16).reshape(shape)[:rank]
    xi = np.fromfile(detected / "xhat_im.bin", np.float16).reshape(shape)[:rank]
    source_re = np.fromfile(root / f"artifacts/layer_grid_mapped_rank{rank}/slot00_re.bin",
                            np.float16).reshape(rank, NSYM, NSPAD)
    source_im = np.fromfile(root / f"artifacts/layer_grid_mapped_rank{rank}/slot00_im.bin",
                            np.float16).reshape(rank, NSYM, NSPAD)
    data_symbols = [s for s in range(NSYM) if s not in (2, 11)]
    observed = (xr[:, data_symbols, :NSC].astype(np.float32) +
                1j * xi[:, data_symbols, :NSC].astype(np.float32)).reshape(rank, -1)
    source = (source_re[:, data_symbols, :NSC].astype(np.float32) +
              1j * source_im[:, data_symbols, :NSC].astype(np.float32)).reshape(rank, -1)
    layers = []
    for layer in range(rank):
        alpha = np.vdot(source[layer], observed[layer]) / np.vdot(source[layer], source[layer])
        nrmse = float(np.linalg.norm(observed[layer] - alpha * source[layer]) /
                      np.linalg.norm(alpha * source[layer]))
        correlation = float(abs(np.vdot(source[layer], observed[layer])) /
                            (np.linalg.norm(source[layer]) * np.linalg.norm(observed[layer])))
        if not np.isfinite(nrmse) or nrmse > 0.10 or correlation < 0.995:
            raise RuntimeError(f"Rank{rank} detector layer {layer} correlation failed")
        layers.append({"layer": layer, "alpha_re": float(alpha.real),
                       "alpha_im": float(alpha.imag), "nrmse": nrmse,
                       "correlation": correlation})
    llr = np.fromfile(root / f"artifacts/demod_rank{rank}_slot0/cw_llr.bin", np.int16)
    if llr.size != QM * rank * 19200 or not np.any(llr):
        raise RuntimeError(f"Rank{rank} slot0 layer_demap output wrong-size/all-zero")
    llr = llr.reshape(QM, rank * 19200)
    if np.any(llr[:, rank * NRE:]):
        raise RuntimeError(f"Rank{rank} slot0 codeword LLR padding nonzero")
    bits = np.memmap(root / f"scramble/ascend_output/rank{rank}/bits_qam.bin", np.int16,
                     "r", shape=(23, QM, rank * 19200))[0]
    errors = int(np.count_nonzero((llr[:, :rank * NRE] < 0) != bits[:, :rank * NRE]))
    total = QM * rank * NRE
    result = {"schema": "airan.pusch_mimo.slot0_rx.v1", "status": "PASS",
              "rank": rank, "slot": 0, "detector_layers": layers,
              "raw_qam_bit_errors": errors, "raw_qam_ber": errors / total,
              "actual_npu_stages": ["channel_est_lmmse_mimo", "mimo_detect_io_pack",
                  "mimo_detect_bri_batch", "qam_demod_256_mimo_batch", "layer_demap_mimo"]}
    (root / f"artifacts/rank{rank}_slot0_rx_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
