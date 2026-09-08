#!/usr/bin/env python3
"""Data adapter/checker for the Rank1 23-slot OFDM device milestone."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

import prepare_rank1 as one
import radio_profile as radio

SLOTS = 23


def prepare(root: Path) -> None:
    one.prepare(root)
    rng = np.random.default_rng(0x3233534C4F54)
    re = np.zeros((SLOTS, one.NSYM, one.NSPAD), dtype=np.float16)
    im = np.zeros_like(re)
    re[:, :, :one.NSC] = rng.choice([-1.0, 1.0], size=(SLOTS, one.NSYM, one.NSC)).astype(np.float16) / np.float16(np.sqrt(2.0))
    im[:, :, :one.NSC] = rng.choice([-1.0, 1.0], size=(SLOTS, one.NSYM, one.NSC)).astype(np.float16) / np.float16(np.sqrt(2.0))
    one.write(root / "artifacts/source_grid_re.bin", re)
    one.write(root / "artifacts/source_grid_im.bin", im)


def prepare_coded(root: Path) -> None:
    """Prepare transform weights while preserving the actual coded TX prefix."""
    one.prepare(root)
    source = root / "artifacts/port_grid"
    re = np.empty((SLOTS, one.NSYM, one.NSPAD), np.float16)
    im = np.empty_like(re)
    for slot in range(SLOTS):
        re[slot] = np.fromfile(source / f"slot{slot:02d}_re.bin", np.float16).reshape(one.NSYM, one.NSPAD)
        im[slot] = np.fromfile(source / f"slot{slot:02d}_im.bin", np.float16).reshape(one.NSYM, one.NSPAD)
    if not np.any(re) or not np.any(im) or np.any(re[:, :, one.NSC:].view(np.uint16)) or np.any(im[:, :, one.NSC:].view(np.uint16)):
        raise RuntimeError("coded port grid is missing, all-zero, or has nonzero row padding")
    one.write(root / "artifacts/source_grid_re.bin", re)
    one.write(root / "artifacts/source_grid_im.bin", im)


def select_tx(root: Path, slot: int) -> None:
    if slot < 0 or slot >= SLOTS:
        raise ValueError("slot out of range")
    re = np.memmap(root / "artifacts/source_grid_re.bin", dtype=np.float16,
                   mode="r", shape=(SLOTS, one.NSYM, one.NSPAD))[slot]
    im = np.memmap(root / "artifacts/source_grid_im.bin", dtype=np.float16,
                   mode="r", shape=(SLOTS, one.NSYM, one.NSPAD))[slot]
    idx = one.scatter_index()
    zero = np.zeros((one.NSYM, 1), dtype=np.float16)
    fft_re = np.concatenate((re, zero), axis=1)[:, idx]
    fft_im = np.concatenate((im, zero), axis=1)[:, idx]
    case = root / "re_map/golden/port1"
    one.write(case / "port_grid_re.bin", np.asarray(re))
    one.write(case / "port_grid_im.bin", np.asarray(im))
    one.write(case / "fft_grid_re.bin", fft_re)
    one.write(case / "fft_grid_im.bin", fft_im)


def collect_tx(root: Path, slot: int) -> None:
    dst = root / "artifacts/tx_fft"
    dst.mkdir(parents=True, exist_ok=True)
    for plane in ("re", "im"):
        source = root / f"re_map/ascend_output/port1/fft_grid_{plane}.bin"
        if source.stat().st_size != one.NSYM * one.NFFT * 2:
            raise RuntimeError("re_map output size mismatch")
        shutil.copyfile(source, dst / f"slot{slot:02d}_{plane}.bin")


def assemble_tx(root: Path) -> None:
    out = root / "ofdm_mod/data/golden"
    out.mkdir(parents=True, exist_ok=True)
    for plane in ("re", "im"):
        with (out / f"in_{plane}.bin").open("wb") as stream:
            for slot in range(SLOTS):
                stream.write((root / f"artifacts/tx_fft/slot{slot:02d}_{plane}.bin").read_bytes())


def channel(root: Path) -> None:
    tx_path = root / "ofdm_mod/data/ascend_output/output_iq.bin"
    expected = SLOTS * one.NSAMP * 2
    tx = np.fromfile(tx_path, dtype=np.int16)
    if tx.size != expected or not np.any(tx):
        raise RuntimeError("23-slot TX IQ missing, wrong-size, or all-zero")
    tx = tx.reshape(SLOTS, one.NSAMP, 2)
    gain = float(os.environ.get("PUSCH_MIMO_CHANNEL_GAIN", "0.16"))
    if gain <= 0 or gain > 1:
        raise RuntimeError("channel gain must be in (0,1]")
    noise_std = float(os.environ.get("PUSCH_MIMO_CHANNEL_NOISE_STD", "0"))
    if noise_std < 0 or noise_std > 200:
        raise RuntimeError("channel noise std must be in [0,200] int16 units")
    profile = radio.selected_profile()
    if profile is not None:
        tx_port = tx[:, None, :, :]
        tx_ant, rx_physical, h_physical, h_effective = radio.apply_fd8x8_channel(
            tx_port, profile, 1, gain, noise_std, 0x4157474E36345258)
        rx = radio.pad_rx_capacity(profile, rx_physical, rx_axis=1)
        one.write(root / "artifacts/tx_port_iq_rank1_23slot.bin", tx_port)
        one.write(root / "artifacts/tx_iq_rank1_23slot.bin", tx_ant)
        one.write(root / "artifacts/rx_iq_rank1_8rx_23slot.bin", rx_physical)
        one.write(root / "artifacts/rx_iq_rank1_23slot.bin", rx)
        receipt = {
            "schema": "airan.pusch_mimo.fd8x8_host_channel.v1",
            "radio_profile": profile.name,
            "profile_source_sha256": profile.source_sha256,
            "architecture": profile.architecture,
            "rank": 1,
            "logical_tx_ports": 1,
            "tx_antennas": profile.num_tx_antennas,
            "tx_rf_chains": profile.num_tx_rf_chains,
            "rx_antennas": profile.num_rx_antennas,
            "rx_rf_chains": profile.num_rx_rf_chains,
            "detector_rx_capacity": profile.detector_rx_capacity,
            "rx_capacity_adapter": profile.rx_capacity_adapter,
            "tx_antenna_mapping": profile.tx_antenna_mapping,
            "physical_channel": profile.physical_channel,
            "physical_channel_rank": int(np.linalg.matrix_rank(h_physical)),
            "effective_channel_rank": int(np.linalg.matrix_rank(h_effective)),
            "gain": gain,
            "noise_std_requested": noise_std,
            "int16_saturation_count": 0,
            "virtual_rx_nonzero": int(np.count_nonzero(rx[:, profile.num_rx_antennas:])),
        }
        (root / "artifacts/channel_rank1_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n")
        return
    signal = np.rint(tx.astype(np.float32) * gain)
    rx = np.broadcast_to(signal[:, None, :, :], (SLOTS, one.NR, one.NSAMP, 2)).copy()
    if noise_std:
        rng = np.random.default_rng(0x4157474E36345258)
        for slot in range(SLOTS):
            rx[slot] += np.rint(rng.normal(0.0, noise_std, size=rx[slot].shape)).astype(np.float32)
    if np.max(np.abs(rx)) >= 32767:
        raise RuntimeError("23-slot channel output clips int16")
    rx = rx.astype(np.int16)
    one.write(root / "artifacts/tx_iq_rank1_23slot.bin", tx)
    one.write(root / "artifacts/rx_iq_rank1_23slot.bin", rx)
    if not np.any(rx):
        raise RuntimeError("23-slot channel output is all zero")


def select_rx(root: Path, slot: int) -> None:
    rx = np.memmap(root / "artifacts/rx_iq_rank1_23slot.bin", dtype=np.int16,
                   mode="r", shape=(SLOTS, one.NR, one.NSAMP, 2))
    one.write(root / "ofdm_demod/data/golden/input.bin", np.asarray(rx[slot]))


def collect_rx(root: Path, slot: int) -> None:
    dst = root / "artifacts/rx_grid"
    dst.mkdir(parents=True, exist_ok=True)
    for plane in ("re", "im"):
        source = root / f"re_demap/data/ascend_output/rx_grid_{plane}.bin"
        expected = one.NR * one.NSYM * one.NSPAD * 2
        if source.stat().st_size != expected:
            raise RuntimeError("re_demap output size mismatch")
        shutil.copyfile(source, dst / f"slot{slot:02d}_{plane}.bin")


def verify(root: Path) -> None:
    source_re = np.memmap(root / "artifacts/source_grid_re.bin", dtype=np.float16,
                          mode="r", shape=(SLOTS, one.NSYM, one.NSPAD))
    source_im = np.memmap(root / "artifacts/source_grid_im.bin", dtype=np.float16,
                          mode="r", shape=(SLOTS, one.NSYM, one.NSPAD))
    worst_nrmse = 0.0
    worst_max = 0.0
    pad_nonzero = 0
    for slot in range(SLOTS):
        out_re = np.fromfile(root / f"artifacts/rx_grid/slot{slot:02d}_re.bin", dtype=np.float16).reshape(one.NR, one.NSYM, one.NSPAD).astype(np.float32)
        out_im = np.fromfile(root / f"artifacts/rx_grid/slot{slot:02d}_im.bin", dtype=np.float16).reshape(one.NR, one.NSYM, one.NSPAD).astype(np.float32)
        gain = float(os.environ.get("PUSCH_MIMO_CHANNEL_GAIN", "0.16"))
        grid_gain = 6.25 * gain
        profile = radio.selected_profile()
        if profile is None:
            target_re = source_re[slot].astype(np.float32) * grid_gain
            target_im = source_im[slot].astype(np.float32) * grid_gain
            err = np.hypot(out_re[0, :, :one.NSC] - target_re[:, :one.NSC],
                           out_im[0, :, :one.NSC] - target_im[:, :one.NSC])
            ref = np.hypot(target_re[:, :one.NSC], target_im[:, :one.NSC])
        else:
            effective = radio.effective_channel_matrix(profile, 1, gain) * 6.25
            effective = radio.pad_rx_capacity(profile, effective[None, ...], rx_axis=1)[0]
            source = (source_re[slot].astype(np.float32) +
                      1j * source_im[slot].astype(np.float32))
            target = effective[:, 0, None, None] * source[None, :, :]
            actual = out_re + 1j * out_im
            err = np.abs(actual[:, :, :one.NSC] - target[:, :, :one.NSC])
            ref = np.abs(target[:, :, :one.NSC])
        worst_nrmse = max(worst_nrmse, float(np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(ref**2))))
        worst_max = max(worst_max, float(np.max(err)))
        pad_nonzero += int(np.count_nonzero(out_re[:, :, one.NSC:]))
        pad_nonzero += int(np.count_nonzero(out_im[:, :, one.NSC:]))
    gain = float(os.environ.get("PUSCH_MIMO_CHANNEL_GAIN", "0.16"))
    noise_std = float(os.environ.get("PUSCH_MIMO_CHANNEL_NOISE_STD", "0"))
    # The waveform check includes the deliberately injected deterministic AWGN.
    # Keep a bounded SNR-aware gate while independently requiring exact padding.
    allowed_nrmse = max(0.08, min(0.65, 0.0012 * noise_std / gain))
    if worst_nrmse > allowed_nrmse or pad_nonzero:
        raise RuntimeError(f"23-slot verify failed nrmse={worst_nrmse} allowed={allowed_nrmse} padding={pad_nonzero}")
    profile = radio.selected_profile()
    result = {"schema_version": 1, "status": "PASS", "rank": 1, "slots": SLOTS,
              "device_stages": ["re_map_batch", "ofdm_mod_batch", "ofdm_demod_batch", "re_demap_batch"],
              "complete_coded_e2e": False, "worst_grid_nrmse": worst_nrmse,
              "allowed_grid_nrmse": allowed_nrmse,
              "worst_max_complex_abs": worst_max, "padding_nonzero": pad_nonzero,
              "radio_profile": profile.name if profile is not None else "legacy",
              "tx_antennas": profile.num_tx_antennas if profile is not None else 1,
              "rx_antennas": profile.num_rx_antennas if profile is not None else one.NR,
              "detector_rx_capacity": one.NR,
              "tx_iq_sha256": one.digest(root / "artifacts/tx_iq_rank1_23slot.bin"),
              "rx_iq_sha256": one.digest(root / "artifacts/rx_iq_rank1_23slot.bin"),
              "channel_gain": float(os.environ.get("PUSCH_MIMO_CHANNEL_GAIN", "0.16")),
              "channel_noise_std_int16": float(os.environ.get("PUSCH_MIMO_CHANNEL_NOISE_STD", "0"))}
    if profile is not None:
        result["host_channel_metrics"] = json.loads(
            (root / "artifacts/channel_rank1_receipt.json").read_text())
    (root / "artifacts/milestone_23slot_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "prepare-coded", "select-tx", "collect-tx", "assemble-tx", "channel", "select-rx", "collect-rx", "verify"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--slot", type=int)
    a = p.parse_args()
    functions = {"prepare": prepare, "prepare-coded": prepare_coded, "select-tx": select_tx, "collect-tx": collect_tx,
                 "assemble-tx": assemble_tx, "channel": channel, "select-rx": select_rx,
                 "collect-rx": collect_rx, "verify": verify}
    if a.action in ("select-tx", "collect-tx", "select-rx", "collect-rx"):
        if a.slot is None:
            p.error("--slot is required")
        functions[a.action](a.root, a.slot)
    else:
        functions[a.action](a.root)


if __name__ == "__main__":
    main()
