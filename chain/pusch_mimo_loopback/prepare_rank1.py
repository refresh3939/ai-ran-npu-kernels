#!/usr/bin/env python3
"""Prepare/verify the Rank-1 single-slot device milestone.

The reference code here is limited to boundary fixtures and checks.  The four
transform stages are executed by their categorized Ascend device binaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NSYM, NFFT, NSC, NSPAD, NSAMP, NR = 14, 2048, 1596, 1664, 30720, 64
P, Q = 32, 64


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)


def dft(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    return np.exp(-2j * np.pi * k * i / n).astype(np.complex64)


def twiddle() -> np.ndarray:
    k = np.arange(P)[:, None]
    i = np.arange(Q)[None, :]
    return np.exp(-2j * np.pi * k * i / NFFT).astype(np.complex64)


def scatter_index() -> np.ndarray:
    natural = np.concatenate((np.arange(NFFT - NSC // 2, NFFT),
                              np.arange(0, NSC - NSC // 2)))
    destinations = (natural % P) * Q + natural // P
    index = np.full(NFFT, NSPAD, dtype=np.int64)
    index[destinations] = np.arange(NSC)
    return index


def prepare(root: Path) -> None:
    rng = np.random.default_rng(0x52414E4B31)
    idx = scatter_index()
    # Populate the auxiliary P=2/4 cases because the categorized re_map runner
    # deliberately exercises every supported batch.  Only port1 feeds this
    # milestone; the others remain independent device self-check fixtures.
    for ports in (1, 2, 4):
        re = np.zeros((ports, NSYM, NSPAD), dtype=np.float16)
        im = np.zeros_like(re)
        re[:, :, :NSC] = rng.choice([-1.0, 1.0], size=(ports, NSYM, NSC)).astype(np.float16) / np.float16(np.sqrt(2.0))
        im[:, :, :NSC] = rng.choice([-1.0, 1.0], size=(ports, NSYM, NSC)).astype(np.float16) / np.float16(np.sqrt(2.0))
        zero = np.zeros((ports, NSYM, 1), dtype=np.float16)
        fft_re = np.concatenate((re, zero), axis=2)[:, :, idx]
        fft_im = np.concatenate((im, zero), axis=2)[:, :, idx]
        case = root / "re_map" / "golden" / f"port{ports}"
        write(case / "port_grid_re.bin", re)
        write(case / "port_grid_im.bin", im)
        write(case / "fft_grid_re.bin", fft_re)
        write(case / "fft_grid_im.bin", fft_im)

    w32, w64, tw = dft(P), dft(Q), twiddle()
    weights: dict[str, np.ndarray] = {
        "ofdm_mod/weights/iw_dft32_re.bin": (w32.conj() / np.sqrt(P)).real.astype(np.float16),
        "ofdm_mod/weights/iw_dft32_im.bin": (w32.conj() / np.sqrt(P)).imag.astype(np.float16),
        "ofdm_mod/weights/iw_dft64_re_T.bin": (w64.conj() / np.sqrt(Q)).real.T.astype(np.float16),
        "ofdm_mod/weights/iw_dft64_im_T.bin": (w64.conj() / np.sqrt(Q)).imag.T.astype(np.float16),
        "ofdm_mod/weights/itwiddle_pq_re.bin": tw.conj().real.astype(np.float16),
        "ofdm_mod/weights/itwiddle_pq_im.bin": tw.conj().imag.astype(np.float16),
        "ofdm_demod/weights/w_dft32_re.bin": (w32 / np.sqrt(P)).real.astype(np.float16),
        "ofdm_demod/weights/w_dft32_im.bin": (w32 / np.sqrt(P)).imag.astype(np.float16),
        "ofdm_demod/weights/w_dft64_re_T.bin": (w64 / np.sqrt(Q)).real.T.astype(np.float16),
        "ofdm_demod/weights/w_dft64_im_T.bin": (w64 / np.sqrt(Q)).imag.T.astype(np.float16),
        "ofdm_demod/weights/twiddle_pq_re.bin": tw.real.astype(np.float16),
        "ofdm_demod/weights/twiddle_pq_im.bin": tw.imag.astype(np.float16),
    }
    for name, value in weights.items():
        write(root / name, value)
    # re_demap consumes byte offsets into one S4 fp16 row.
    gather = np.empty(NSPAD, dtype=np.uint32)
    natural = np.concatenate((np.arange(NFFT - NSC // 2, NFFT),
                              np.arange(0, NSC - NSC // 2)))
    gather[:NSC] = (((natural % P) * Q + natural // P) * 2).astype(np.uint32)
    gather[NSC:] = 0
    write(root / "re_demap/weights/gather_idx.bin", gather)
    manifest = {"schema_version": 1, "milestone": "rank1_single_slot_ofdm_device_subchain",
                "complete_coded_e2e": False, "rank": 1, "slots": 1,
                "tx_ports": 1, "rx_antennas": 64, "channel": "H[r,0]=0.5",
                "weights": {}}
    for name in sorted(weights):
        path = root / name
        if path.stat().st_size == 0 or not any(path.read_bytes()):
            raise RuntimeError(f"zero/missing weight: {path}")
        manifest["weights"][name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    gp = root / "re_demap/weights/gather_idx.bin"
    manifest["weights"]["re_demap/weights/gather_idx.bin"] = {
        "bytes": gp.stat().st_size, "sha256": digest(gp)}
    (root / "weight_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def channel(root: Path) -> None:
    tx = root / "ofdm_mod/data/ascend_output/output_iq.bin"
    if not tx.is_file() or tx.stat().st_size != NSAMP * 2 * 2:
        raise RuntimeError("TX IQ is missing or has the wrong Rank1 single-slot shape")
    x = np.fromfile(tx, dtype=np.int16).reshape(1, NSAMP, 2)
    if not np.any(x):
        raise RuntimeError("TX IQ is all zero")
    y = np.rint(x.astype(np.float32) * 0.5)
    if np.max(np.abs(y)) >= 32767:
        raise RuntimeError("flat-channel output would clip int16")
    rx = np.broadcast_to(y.astype(np.int16), (NR, NSAMP, 2)).copy()
    write(root / "ofdm_demod/data/golden/input.bin", rx)
    write(root / "artifacts/tx_iq_rank1_slot0.bin", x)
    write(root / "artifacts/rx_iq_rank1_slot0.bin", rx)
    receipt = {"shape_tx": [1, NSAMP, 2], "shape_rx": [NR, NSAMP, 2],
               "tx_sha256": digest(tx),
               "rx_sha256": digest(root / "artifacts/rx_iq_rank1_slot0.bin"),
               "nonzero": int(np.count_nonzero(rx))}
    (root / "artifacts/channel_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def verify(root: Path) -> None:
    src = np.fromfile(root / "re_map/golden/port1/port_grid_re.bin", dtype=np.float16).reshape(NSYM, NSPAD)
    src_i = np.fromfile(root / "re_map/golden/port1/port_grid_im.bin", dtype=np.float16).reshape(NSYM, NSPAD)
    out = np.fromfile(root / "re_demap/data/ascend_output/rx_grid_re.bin", dtype=np.float16)
    out_i = np.fromfile(root / "re_demap/data/ascend_output/rx_grid_im.bin", dtype=np.float16)
    expected = NR * NSYM * NSPAD
    if out.size != expected or out_i.size != expected:
        raise RuntimeError("RX grid shape mismatch")
    out = out.reshape(NR, NSYM, NSPAD).astype(np.float32)
    out_i = out_i.reshape(NR, NSYM, NSPAD).astype(np.float32)
    # OUT_SCALE/receiver-Q-scale * H = 1600/256*0.5 = 3.125.
    target = src.astype(np.float32) * 3.125
    target_i = src_i.astype(np.float32) * 3.125
    err = np.hypot(out[0, :, :NSC] - target[:, :NSC],
                   out_i[0, :, :NSC] - target_i[:, :NSC])
    ref = np.hypot(target[:, :NSC], target_i[:, :NSC])
    nrmse = float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(ref ** 2)))
    max_abs = float(np.max(err))
    pad_nonzero = int(np.count_nonzero(out[:, :, NSC:])) + int(np.count_nonzero(out_i[:, :, NSC:]))
    if not np.any(out) or nrmse > 0.08 or pad_nonzero:
        raise RuntimeError(f"device OFDM loopback failed: nrmse={nrmse} max={max_abs} pad_nonzero={pad_nonzero}")
    result = {"status": "PASS", "scope": "rank1_single_slot_ofdm_device_subchain",
              "complete_coded_e2e": False, "nrmse": nrmse,
              "max_complex_abs": max_abs, "padding_nonzero": pad_nonzero}
    (root / "artifacts/milestone_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "channel", "verify"))
    p.add_argument("--root", required=True, type=Path)
    a = p.parse_args()
    {"prepare": prepare, "channel": channel, "verify": verify}[a.action](a.root)


if __name__ == "__main__":
    main()
