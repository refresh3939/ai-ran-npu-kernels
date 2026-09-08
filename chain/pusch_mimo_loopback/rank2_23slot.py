#!/usr/bin/env python3
"""Fail-closed data orchestration for the Rank2 23-slot OFDM subchain."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

import prepare_rank1 as one
import radio_profile as radio

RANK = 2
PORTS = 2
SLOTS = 23


def execution_rx_capacity() -> int:
    plan = radio.selected_plan()
    if plan is None:
        return one.NR
    if plan["active"]["rank"] != RANK:
        raise RuntimeError("resolved-plan Rank mismatch in waveform stage")
    capacity = int(plan["receiver"]["rx_bucket"])
    physical = int(plan["topology"]["rx_antennas"])
    if capacity < physical or capacity > one.NR:
        raise RuntimeError("invalid resolved execution RX bucket")
    return capacity


def prepare_coded(root: Path) -> None:
    # The categorized re_map runner deliberately runs port1/2/4 cases. This
    # installs explicit nonzero fixtures and all FFT weights for those cases.
    one.prepare(root)
    source = root / f"artifacts/port_grid_rank{RANK}"
    re = np.empty((SLOTS, PORTS, one.NSYM, one.NSPAD), np.float16)
    im = np.empty_like(re)
    for slot in range(SLOTS):
        re[slot] = np.fromfile(source / f"slot{slot:02d}_re.bin", np.float16).reshape(
            PORTS, one.NSYM, one.NSPAD)
        im[slot] = np.fromfile(source / f"slot{slot:02d}_im.bin", np.float16).reshape(
            PORTS, one.NSYM, one.NSPAD)
    if (not np.any(re) or not np.any(im)
            or np.any(re[:, :, :, one.NSC:].view(np.uint16))
            or np.any(im[:, :, :, one.NSC:].view(np.uint16))):
        raise RuntimeError("Rank2 coded port grid missing/all-zero/nonzero-padding")
    one.write(root / f"artifacts/source_grid_rank{RANK}_re.bin", re)
    one.write(root / f"artifacts/source_grid_rank{RANK}_im.bin", im)


def select_tx(root: Path, slot: int) -> None:
    re = np.memmap(root / f"artifacts/source_grid_rank{RANK}_re.bin", dtype=np.float16,
                   mode="r", shape=(SLOTS, PORTS, one.NSYM, one.NSPAD))[slot]
    im = np.memmap(root / f"artifacts/source_grid_rank{RANK}_im.bin", dtype=np.float16,
                   mode="r", shape=(SLOTS, PORTS, one.NSYM, one.NSPAD))[slot]
    idx = one.scatter_index()
    zero = np.zeros((PORTS, one.NSYM, 1), np.float16)
    fft_re = np.concatenate((re, zero), axis=2)[:, :, idx]
    fft_im = np.concatenate((im, zero), axis=2)[:, :, idx]
    case = root / f"re_map/golden/port{PORTS}"
    one.write(case / "port_grid_re.bin", np.asarray(re))
    one.write(case / "port_grid_im.bin", np.asarray(im))
    one.write(case / "fft_grid_re.bin", fft_re)
    one.write(case / "fft_grid_im.bin", fft_im)


def collect_tx(root: Path, slot: int) -> None:
    dst = root / f"artifacts/tx_fft_rank{RANK}"
    dst.mkdir(parents=True, exist_ok=True)
    expected = PORTS * one.NSYM * one.NFFT * 2
    for plane in ("re", "im"):
        source = root / f"re_map/ascend_output/port{PORTS}/fft_grid_{plane}.bin"
        if not source.is_file() or source.stat().st_size != expected:
            raise RuntimeError(f"Rank2 re_map {plane} output must be exactly {expected} bytes")
        shutil.copyfile(source, dst / f"slot{slot:02d}_{plane}.bin")


def assemble_tx(root: Path) -> None:
    out = root / "ofdm_mod/data/golden"
    out.mkdir(parents=True, exist_ok=True)
    for plane in ("re", "im"):
        with (out / f"in_{plane}.bin").open("wb") as stream:
            for slot in range(SLOTS):
                stream.write((root / f"artifacts/tx_fft_rank{RANK}/slot{slot:02d}_{plane}.bin").read_bytes())
        expected = SLOTS * PORTS * one.NSYM * one.NFFT * 2
        if (out / f"in_{plane}.bin").stat().st_size != expected:
            raise RuntimeError("Rank2 OFDM input assembly size mismatch")


def channel_matrix(gain: float) -> np.ndarray:
    profile = radio.selected_profile()
    if profile is not None:
        if profile.tx_ports(RANK) != PORTS:
            raise RuntimeError("radio profile logical-port mapping mismatch")
        effective = radio.effective_channel_matrix(profile, RANK, gain)
        return radio.pad_rx_capacity(profile, effective[None, ...], rx_axis=1)[0]
    antenna = np.arange(one.NR, dtype=np.float32)[:, None]
    layer = np.arange(PORTS, dtype=np.float32)[None, :]
    return gain * np.exp(2j * np.pi * antenna * layer / one.NR).astype(np.complex64)


def channel(root: Path) -> None:
    tx_path = root / "ofdm_mod/data/ascend_output/output_iq.bin"
    expected = SLOTS * PORTS * one.NSAMP * 2
    tx = np.fromfile(tx_path, dtype=np.int16)
    if tx.size != expected or not np.any(tx):
        raise RuntimeError("Rank2 23-slot per-port TX IQ missing/wrong-size/all-zero")
    tx = tx.reshape(SLOTS, PORTS, one.NSAMP, 2)
    gain = float(os.environ.get("PUSCH_MIMO_CHANNEL_GAIN", "0.16"))
    noise_std = float(os.environ.get("PUSCH_MIMO_CHANNEL_NOISE_STD", "90"))
    if not 0 < gain <= 1 or not 0 <= noise_std <= 200:
        raise RuntimeError("invalid channel gain/noise")
    profile = radio.selected_profile()
    if profile is not None:
        if profile.tx_ports(RANK) != PORTS:
            raise RuntimeError("radio profile logical-port mapping mismatch")
        tx_ant, rx_physical, h_physical, h_effective = radio.apply_fd8x8_channel(
            tx, profile, RANK, gain, noise_std, 0x523236345258)
        rx = radio.pad_rx_capacity(profile, rx_physical, rx_axis=1)
        if not np.any(rx_physical):
            raise RuntimeError("8RX physical channel output all-zero")
        one.write(root / f"artifacts/tx_port_iq_rank{RANK}_23slot.bin", tx)
        one.write(root / f"artifacts/tx_iq_rank{RANK}_23slot.bin", tx_ant)
        one.write(root / f"artifacts/rx_iq_rank{RANK}_8rx_23slot.bin", rx_physical)
        one.write(root / f"artifacts/rx_iq_rank{RANK}_23slot.bin", rx)
        receipt = {
            "schema": "airan.pusch_mimo.fd8x8_host_channel.v1",
            "radio_profile": profile.name,
            "profile_source_sha256": profile.source_sha256,
            "architecture": profile.architecture,
            "rank": RANK,
            "logical_tx_ports": PORTS,
            "tx_antennas": profile.num_tx_antennas,
            "tx_rf_chains": profile.num_tx_rf_chains,
            "rx_antennas": profile.num_rx_antennas,
            "rx_rf_chains": profile.num_rx_rf_chains,
            "detector_rx_capacity": execution_rx_capacity(),
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
        (root / f"artifacts/channel_rank{RANK}_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n")
        return
    h = channel_matrix(gain)
    x = tx[..., 0].astype(np.float32) + 1j * tx[..., 1].astype(np.float32)
    y = np.einsum("rp,spn->srn", h, x, optimize=True)
    signal_component = np.stack((y.real, y.imag), axis=-1).astype(np.float32)
    noise_component = np.zeros_like(signal_component)
    if noise_std:
        rng = np.random.default_rng(0x523236345258)
        noise_component = rng.normal(0.0, noise_std, size=signal_component.shape).astype(np.float32)
    composite = signal_component + noise_component
    saturation_count = int(np.count_nonzero(np.abs(composite) >= 32767))
    if saturation_count:
        raise RuntimeError("Rank2 channel output clips int16")
    rx = np.rint(composite).astype(np.int16)
    if not np.any(rx):
        raise RuntimeError("Rank2 channel output all-zero")
    one.write(root / f"artifacts/tx_iq_rank{RANK}_23slot.bin", tx)
    one.write(root / f"artifacts/rx_iq_rank{RANK}_23slot.bin", rx)
    receipt = {
        "schema": "airan.pusch_mimo.host_channel.v1", "rank": RANK,
        "tx_ports": PORTS, "rx_antennas": one.NR, "gain": gain,
        "noise_std_requested": noise_std,
        "signal_component_rms": float(np.sqrt(np.mean(signal_component ** 2))),
        "noise_component_rms": float(np.sqrt(np.mean(noise_component ** 2))),
        "signal_component_abs_max": float(np.max(np.abs(signal_component))),
        "composite_abs_max_before_int16": float(np.max(np.abs(composite))),
        "int16_saturation_count": saturation_count,
    }
    (root / f"artifacts/channel_rank{RANK}_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n")


def select_rx(root: Path, slot: int) -> None:
    rx = np.memmap(root / f"artifacts/rx_iq_rank{RANK}_23slot.bin", dtype=np.int16,
                   mode="r", shape=(SLOTS, one.NR, one.NSAMP, 2))
    one.write(root / "ofdm_demod/data/golden/input.bin", np.asarray(rx[slot]))


def collect_rx(root: Path, slot: int) -> None:
    dst = root / f"artifacts/rx_grid_rank{RANK}"
    dst.mkdir(parents=True, exist_ok=True)
    source_elements = one.NR * one.NSYM * one.NSPAD
    capacity = execution_rx_capacity()
    for plane in ("re", "im"):
        source = root / f"re_demap/data/ascend_output/rx_grid_{plane}.bin"
        if not source.is_file() or source.stat().st_size != source_elements * 2:
            raise RuntimeError(
                f"Rank{RANK} re_demap {plane} output must be exactly "
                f"{source_elements * 2} bytes")
        value = np.fromfile(source, np.float16).reshape(
            one.NR, one.NSYM, one.NSPAD)
        one.write(dst / f"slot{slot:02d}_{plane}.bin",
                  np.asarray(value[:capacity]))


def verify(root: Path) -> None:
    source_re = np.memmap(root / f"artifacts/source_grid_rank{RANK}_re.bin", np.float16,
                          "r", shape=(SLOTS, PORTS, one.NSYM, one.NSPAD))
    source_im = np.memmap(root / f"artifacts/source_grid_rank{RANK}_im.bin", np.float16,
                          "r", shape=(SLOTS, PORTS, one.NSYM, one.NSPAD))
    gain = float(os.environ.get("PUSCH_MIMO_CHANNEL_GAIN", "0.16"))
    noise_std = float(os.environ.get("PUSCH_MIMO_CHANNEL_NOISE_STD", "90"))
    capacity = execution_rx_capacity()
    h = channel_matrix(gain)[:capacity] * 6.25
    worst_nrmse = 0.0
    padding_nonzero = 0
    output_nonzero = 0
    for slot in range(SLOTS):
        out_re = np.fromfile(root / f"artifacts/rx_grid_rank{RANK}/slot{slot:02d}_re.bin",
                             np.float16).reshape(capacity, one.NSYM, one.NSPAD).astype(np.float32)
        out_im = np.fromfile(root / f"artifacts/rx_grid_rank{RANK}/slot{slot:02d}_im.bin",
                             np.float16).reshape(capacity, one.NSYM, one.NSPAD).astype(np.float32)
        src = source_re[slot].astype(np.float32) + 1j * source_im[slot].astype(np.float32)
        target = np.einsum("rp,psn->rsn", h, src, optimize=True)
        actual = out_re + 1j * out_im
        err = actual[:, :, :one.NSC] - target[:, :, :one.NSC]
        ref = target[:, :, :one.NSC]
        nrmse = float(np.sqrt(np.mean(np.abs(err) ** 2)) / np.sqrt(np.mean(np.abs(ref) ** 2)))
        worst_nrmse = max(worst_nrmse, nrmse)
        padding_nonzero += int(np.count_nonzero(actual[:, :, one.NSC:]))
        output_nonzero += int(np.count_nonzero(actual[:, :, :one.NSC]))
    # This is an H-composed per-antenna golden comparison, not a comparison to
    # either individual stream.  The fixed 0.45 ceiling leaves measured AWGN
    # headroom (currently ~0.366) but catches wrong port order/phase/mixing.
    allowed = 0.45
    if output_nonzero == 0 or padding_nonzero or worst_nrmse > allowed:
        raise RuntimeError(
            f"Rank2 waveform verify failed nonzero={output_nonzero} padding={padding_nonzero} "
            f"nrmse={worst_nrmse} allowed={allowed}")
    result = {
        "schema": "airan.pusch_mimo.waveform.v1",
        "status": "PASS",
        "rank": RANK,
        "slots": SLOTS,
        "tx_ports": PORTS,
        "tx_antennas": (radio.selected_profile().num_tx_antennas
                         if radio.selected_profile() is not None else PORTS),
        "rx_antennas": (radio.selected_profile().num_rx_antennas
                         if radio.selected_profile() is not None else one.NR),
        "detector_rx_capacity": capacity,
        "radio_profile": (radio.selected_profile().name
                           if radio.selected_profile() is not None else "legacy"),
        "channel": ("deterministic full-digital physical 8x8"
                    if radio.selected_profile() is not None
                    else "H[r,p]=gain*exp(j*2*pi*r*p/64)"),
        "channel_rank": int(np.linalg.matrix_rank(channel_matrix(gain))),
        "channel_gain": gain,
        "channel_noise_std_int16": noise_std,
        "actual_npu_stages": ["re_map_batch", "ofdm_mod_batch", "ofdm_demod_batch", "re_demap_batch"],
        "worst_grid_nrmse": worst_nrmse,
        "allowed_grid_nrmse": allowed,
        "padding_nonzero": padding_nonzero,
        "host_channel_metrics": json.loads(
            (root / f"artifacts/channel_rank{RANK}_receipt.json").read_text()),
        "tx_iq_sha256": one.digest(root / f"artifacts/tx_iq_rank{RANK}_23slot.bin"),
        "rx_iq_sha256": one.digest(root / f"artifacts/rx_iq_rank{RANK}_23slot.bin"),
    }
    (root / f"artifacts/coded_waveform_rank{RANK}_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    global RANK, PORTS
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare-coded", "select-tx", "collect-tx",
                                            "assemble-tx", "channel", "select-rx",
                                            "collect-rx", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--rank", type=int, default=2, choices=(2, 3, 4))
    args = parser.parse_args()
    RANK = args.rank
    PORTS = 4 if RANK == 3 else RANK
    actions = {"prepare-coded": prepare_coded, "select-tx": select_tx,
               "collect-tx": collect_tx, "assemble-tx": assemble_tx,
               "channel": channel, "select-rx": select_rx,
               "collect-rx": collect_rx, "verify": verify}
    if args.action in ("select-tx", "collect-tx", "select-rx", "collect-rx"):
        if args.slot is None or not 0 <= args.slot < SLOTS:
            parser.error("--slot 0..22 required")
        actions[args.action](args.root, args.slot)
    else:
        actions[args.action](args.root)


if __name__ == "__main__":
    main()
