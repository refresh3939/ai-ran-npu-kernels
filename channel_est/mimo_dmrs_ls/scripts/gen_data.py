#!/usr/bin/env python3
"""Generate deterministic natural-layout LS and CE-pack test vectors."""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

KERNELS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(KERNELS_ROOT / "mimo" / "mimo_dmrs_gen" / "scripts"))
from dmrs_ref import dmrs_cinit, dmrs_qpsk  # noqa: E402

NR, MAX_L, D = int(os.environ.get("DMRS_LS_NR", "64")), 4, 2
if NR not in (16, 32, 64):
    raise ValueError("DMRS_LS_NR must be 16, 32, or 64")
NSYM, SC_USED, SC_PAD = 14, 1596, 1664
N_RE, REF_PAD, PILOT_PAD = 798, 896, 832
DMRS_SYMBOLS = (2, 11)
CASES = (
    ("case_rank1", (1000,)),
    ("case_rank2_disjoint", (1000, 1002)),
    ("case_rank3_mixed", (1000, 1001, 1002)),
    ("case_rank4_occ", (1000, 1001, 1002, 1003)),
)


def qhalf(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float16)


def build_refs(ports: tuple[int, ...]) -> np.ndarray:
    refs = np.zeros((len(ports), D, REF_PAD), dtype=np.complex64)
    for ds, symbol in enumerate(DMRS_SYMBOLS):
        base = dmrs_qpsk(dmrs_cinit(0, 17, 0, symbol), N_RE)
        for layer, port in enumerate(ports):
            occ = np.ones(N_RE, dtype=np.float32)
            if port in (1001, 1003):
                occ[1::2] = -1.0
            refs[layer, ds, :N_RE] = base * occ
    return refs


def build_rx(ports: tuple[int, ...], refs: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nl = len(ports)
    rx = np.zeros((NR, NSYM, SC_PAD), dtype=np.complex64)
    sc = np.arange(SC_USED, dtype=np.float32)
    for antenna in range(NR):
        for ds, symbol in enumerate(DMRS_SYMBOLS):
            for layer, port in enumerate(ports):
                # Smooth, layer-distinct channel; adjacent-pair OCC assumption is explicit.
                amp = 0.55 + 0.06 * layer + 0.002 * antenna
                phase = 0.13 * layer + 0.017 * antenna + 0.00022 * sc + 0.03 * ds
                channel = amp * np.exp(1j * phase)
                delta = (port - 1000) // 2
                positions = delta + 2 * np.arange(N_RE)
                rx[antenna, symbol, positions] += channel[positions] * refs[layer, ds, :N_RE]
            noise = 0.02 / np.sqrt(2.0) * (
                rng.standard_normal(SC_USED) + 1j * rng.standard_normal(SC_USED))
            rx[antenna, symbol, :SC_USED] += noise.astype(np.complex64)
    # Device consumes fp16 split planes, so reference starts from quantized inputs.
    return qhalf(rx.real).astype(np.float32) + 1j * qhalf(rx.imag).astype(np.float32)


def ls_reference(rx: np.ndarray, refs: np.ndarray, ports: tuple[int, ...]):
    nl = len(ports)
    out_re = np.zeros((NR, nl, D, PILOT_PAD), dtype=np.float16)
    out_im = np.zeros_like(out_re)
    pilot_sc = np.zeros((nl, D, PILOT_PAD), dtype=np.uint16)
    pilot_count = np.zeros((MAX_L, D), dtype=np.uint16)
    comb = np.array([(p - 1000) // 2 for p in ports], dtype=np.uint32)
    shared = np.array([np.count_nonzero(comb == c) == 2 for c in comb])

    ref_re = qhalf(refs.real)
    ref_im = qhalf(refs.imag)
    for layer in range(nl):
        count = 399 if shared[layer] else 798
        locations = (comb[layer] + (1 + 4 * np.arange(count) if shared[layer]
                                    else 2 * np.arange(count))).astype(np.uint16)
        pilot_sc[layer, :, :count] = locations
        pilot_count[layer, :] = count
        physical = comb[layer] + 2 * np.arange(PILOT_PAD)
        valid = physical < SC_PAD
        for antenna in range(NR):
            for ds, symbol in enumerate(DMRS_SYMBOLS):
                yr = np.zeros(PILOT_PAD, dtype=np.float16)
                yi = np.zeros(PILOT_PAD, dtype=np.float16)
                yr[valid] = qhalf(rx[antenna, symbol, physical[valid]].real)
                yi[valid] = qhalf(rx[antenna, symbol, physical[valid]].imag)
                ar = qhalf(yr * ref_re[layer, ds, :PILOT_PAD])
                br = qhalf(yi * ref_im[layer, ds, :PILOT_PAD])
                ai = qhalf(yi * ref_re[layer, ds, :PILOT_PAD])
                bi = qhalf(yr * ref_im[layer, ds, :PILOT_PAD])
                hre = qhalf(ar + br)
                him = qhalf(ai - bi)
                if shared[layer]:
                    out_re[antenna, layer, ds, :count] = qhalf(
                        qhalf(hre[0:2 * count:2] + hre[1:2 * count:2]) * np.float16(0.5))
                    out_im[antenna, layer, ds, :count] = qhalf(
                        qhalf(him[0:2 * count:2] + him[1:2 * count:2]) * np.float16(0.5))
                else:
                    out_re[antenna, layer, ds] = hre
                    out_im[antenna, layer, ds] = him

    noise = np.zeros(NR, dtype=np.float16)
    count0 = int(pilot_count[0, 0])
    factor = 1.0 if shared[0] else 0.5
    for antenna in range(NR):
        total = np.float32(0.0)
        for ds in range(D):
            hre = out_re[antenna, 0, ds, :count0]
            him = out_im[antenna, 0, ds, :count0]
            dre = qhalf(hre[1:] - hre[:-1])
            dim = qhalf(him[1:] - him[:-1])
            power = qhalf(qhalf(dre * dre) + qhalf(dim * dim))
            total += np.sum(power.astype(np.float32), dtype=np.float32)
        noise[antenna] = np.float16(total * factor / (D * (count0 - 1)))
    return out_re, out_im, pilot_sc, pilot_count, noise, comb, shared


def metadata(ports: tuple[int, ...], comb: np.ndarray, shared: np.ndarray) -> np.ndarray:
    words = np.zeros(32, dtype=np.uint32)
    words[:10] = [0x4D4C5331, NR, len(ports), D, NSYM, SC_USED, SC_PAD,
                  REF_PAD, PILOT_PAD, NR // 16]
    words[10:12] = DMRS_SYMBOLS
    words[14:14 + len(ports)] = ports
    words[18:18 + len(ports)] = comb
    words[22:22 + len(ports)] = shared.astype(np.uint32)
    return words


def write_case(base: Path, name: str, ports: tuple[int, ...], seed: int) -> None:
    case = base / "golden" / name
    case.mkdir(parents=True, exist_ok=True)
    refs = build_refs(ports)
    rx = build_rx(ports, refs, seed)
    hre, him, sc, count, noise, comb, shared = ls_reference(rx, refs, ports)
    meta = metadata(ports, comb, shared)

    qhalf(rx.real).tofile(case / "rx_re.bin")
    qhalf(rx.imag).tofile(case / "rx_im.bin")
    qhalf(refs.real).tofile(case / "dmrs_ref_re.bin")
    qhalf(refs.imag).tofile(case / "dmrs_ref_im.bin")
    hre.tofile(case / "h_ls_re.bin")
    him.tofile(case / "h_ls_im.bin")
    sc.tofile(case / "pilot_sc.bin")
    count_backing = np.zeros(16, dtype=np.uint16)
    count_backing[:count.size] = count.reshape(-1)
    count_backing.tofile(case / "pilot_count.bin")
    noise.tofile(case / "noise_var_rx.bin")
    meta.tofile(case / "metadata.bin")

    compatible = bool(np.all(count[:len(ports)] == N_RE))
    if compatible:
        ce = np.zeros((16, NR // 8, PILOT_PAD, 16), dtype=np.complex64)
        for layer in range(len(ports)):
            for antenna in range(NR):
                group, local = divmod(antenna, 8)
                for ds in range(D):
                    ce[layer, group, :, 2 * local + ds] = (
                        hre[antenna, layer, ds].astype(np.float32) +
                        1j * him[antenna, layer, ds].astype(np.float32))
        qhalf(ce.real).tofile(case / "ce_hls_re.bin")
        qhalf(ce.imag).tofile(case / "ce_hls_im.bin")
        qhalf(-ce.imag).tofile(case / "ce_hls_neg_im.bin")

    observation_models = ["fd_occ2_399" if value == 399 else "comb2_798"
                          for value in count[:len(ports), 0]]
    manifest = {
        "case": name, "ports": list(ports), "num_layers": len(ports),
        "pilot_count": count[:len(ports), 0].tolist(),
        "observation_models": observation_models,
        "canonical_lmmse_compatible": True,
        "legacy_ce64x16_pack_compatible": compatible and NR == 64,
        "legacy_ce_pack_compatible": compatible,
        "natural_layout": [NR, len(ports), D, PILOT_PAD],
    }
    (case / "case.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[gen] {name}: ports={ports}, count={manifest['pilot_count']}, "
          f"models={observation_models}, legacy-pack={compatible}")


def main() -> None:
    base = Path(os.environ.get("AIRAN_DATA_DIR",
                               Path(__file__).resolve().parents[1] / "data"))
    for seed, (name, ports) in enumerate(CASES, start=20260905):
        write_case(base, name, ports, seed)
    index = np.empty((PILOT_PAD, 16), dtype=np.uint32)
    for pilot in range(PILOT_PAD):
        for column in range(16):
            index[pilot, column] = 2 * (column * PILOT_PAD + pilot)
    (base / "weights").mkdir(parents=True, exist_ok=True)
    index.tofile(base / "weights" / "ce_pack_gather_index.bin")


if __name__ == "__main__":
    main()
