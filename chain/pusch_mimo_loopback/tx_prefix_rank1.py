#!/usr/bin/env python3
"""Strict data plumbing for Rank1 LDPC->layer_map device stages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

SLOTS, QM, NRE, NPAD = 23, 8, 19152, 19200
NCB, KCB, NRAW = 143, 8448, 25344
SLOTS_BY_RANK = {1: 23, 2: 7, 3: 3, 4: 2}
CELL_ID, RNTI = 321, 12345


def put(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value.tofile(path)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def descriptors(layers: int, slots: int) -> np.ndarray:
    symbols0, gamma = divmod(slots * NRE, NCB)
    symbols = np.full(NCB, symbols0, dtype=np.uint32)
    symbols[NCB - gamma:] += 1
    desc = np.zeros((NCB, 4), dtype=np.uint32)
    desc[:, 0] = layers * QM * symbols
    desc[:, 2] = NRAW
    desc[:, 3] = np.arange(NCB, dtype=np.uint32) * NRAW
    return desc


def rate_match(code: np.ndarray, layers: int, slots: int) -> np.ndarray:
    valid, stride = layers * NRE, layers * NPAD
    out = np.zeros((slots, QM, stride), dtype=np.int16)
    desc = descriptors(layers, slots)
    for bit in range(QM):
        serial = []
        for cb in range(NCB):
            eq = int(desc[cb, 0]) // QM
            serial.append(code[cb, (bit * eq + np.arange(eq)) % NRAW])
        out[:, bit, :valid] = np.concatenate(serial).reshape(slots, valid)
    return out


def gold(cinit: int, length: int) -> np.ndarray:
    total = length + 1600
    x1 = np.zeros(total + 31, np.uint8)
    x2 = np.zeros(total + 31, np.uint8)
    x1[0] = 1
    for bit in range(31):
        x2[bit] = (cinit >> bit) & 1
    for n in range(total):
        x1[n + 31] = x1[n + 3] ^ x1[n]
        x2[n + 31] = x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n]
    return x1[1600:1600 + length] ^ x2[1600:1600 + length]


def scramble(bits: np.ndarray, layers: int, slots: int) -> tuple[np.ndarray, np.ndarray]:
    valid, stride = layers * NRE, layers * NPAD
    seq = gold((RNTI << 15) | CELL_ID, slots * valid * QM).reshape(slots, valid, QM)
    gold_planes = np.zeros((slots, QM, stride), dtype=np.int16)
    gold_planes[:, :, :valid] = seq.transpose(0, 2, 1)
    out = np.zeros_like(bits)
    order = (0, 2, 4, 6, 1, 3, 5, 7)
    for qam_stream, nr_bit in enumerate(order):
        out[:, qam_stream, :valid] = bits[:, nr_bit, :valid] ^ gold_planes[:, nr_bit, :valid]
    return gold_planes, out


def qam(bits: np.ndarray, layers: int) -> tuple[np.ndarray, np.ndarray]:
    valid, stride = layers * NRE, layers * NPAD
    def level(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        return (2*a-1) * (8 - (2*b-1) * (4 - (2*c-1) * (3-2*d)))
    b = bits.astype(np.int32)
    re = np.zeros(stride, np.float16)
    im = np.zeros(stride, np.float16)
    scale = np.float16(1 / np.sqrt(170.0))
    re[:valid] = (level(b[0,:valid], b[1,:valid], b[2,:valid], b[3,:valid]).astype(np.float16) * scale).astype(np.float16)
    im[:valid] = (level(b[4,:valid], b[5,:valid], b[6,:valid], b[7,:valid]).astype(np.float16) * scale).astype(np.float16)
    return re, im


def layer_map(values: np.ndarray, layers: int) -> np.ndarray:
    out = np.zeros((layers, NPAD), dtype=values.dtype)
    out[:, :NRE] = values[:layers*NRE].reshape(NRE, layers).T
    return out


def prepare(root: Path, legacy: Path) -> None:
    source_input = np.fromfile(legacy / "golden/input.bin", dtype=np.int8)
    source_output = np.fromfile(legacy / "golden/output.bin", dtype=np.int8)
    if source_input.size != 4*KCB or source_output.size != 4*NRAW:
        raise RuntimeError("Sionna LDPC input/golden has an invalid size")
    base = source_input.reshape(4, KCB)
    info = np.stack([base[i % 4] for i in range(NCB)])
    put(root / "ldpc/golden/input.bin", base)
    put(root / "ldpc/golden/output.bin", source_output)
    put(root / "artifacts/tx_bits.bin", info)
    weights = {}
    for name in ("shift_A.bin", "shift_Bi.bin", "shift_C.bin", "shift_D.bin"):
        src = legacy / "weights/ldpc_bg1_z384_shifts" / name
        dst = root / "ldpc/weights/ldpc_bg1_z384_shifts" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if not any(dst.read_bytes()):
            raise RuntimeError(f"LDPC weight is all zero: {name}")
        weights[name] = {"bytes": dst.stat().st_size, "sha256": hash_file(dst)}
    expected_sizes = {"shift_A.bin": 88, "shift_Bi.bin": 64,
                      "shift_C.bin": 924, "shift_D.bin": 168}
    for name, count in expected_sizes.items():
        if np.fromfile(root / "ldpc/weights/ldpc_bg1_z384_shifts" / name,
                       dtype=np.int16).size != count:
            raise RuntimeError(f"LDPC weight shape mismatch: {name}")
    (root / "artifacts/ldpc_weight_manifest.json").write_text(json.dumps(weights, indent=2, sort_keys=True)+"\n")
    rng = np.random.default_rng(0x5458505245464958)
    for rank in range(1, 5):
        # rate_match_mimo standalone exercises its MAX_SLOTS profile for every rank.
        rate_slots = SLOTS
        code = rng.integers(0, 2, size=(NCB, NRAW), dtype=np.int8)
        rdir = root / f"rate/golden/rank{rank}"
        put(rdir / "code_blocks.bin", code)
        put(rdir / "rm_desc.bin", descriptors(rank, rate_slots))
        put(rdir / "bits_nr.bin", rate_match(code, rank, rate_slots))
        slots = SLOTS_BY_RANK[rank]
        valid, stride = rank*NRE, rank*NPAD
        raw = rng.integers(0, 2, size=(slots, QM, stride), dtype=np.int16)
        raw[:, :, valid:] = 0
        signs, scr = scramble(raw, rank, slots)
        sdir = root / f"scramble/golden/rank{rank}"
        put(sdir / "bits_nr.bin", raw); put(sdir / "gold.bin", signs); put(sdir / "bits_qam.bin", scr)
        qbits = scr[0]
        qre, qim = qam(qbits, rank)
        qdir = root / f"qam/golden/rank{rank}"
        put(qdir / "bits_qam.bin", qbits); put(qdir / "d_re.bin", qre); put(qdir / "d_im.bin", qim)
        ldir = root / f"layer/golden/rank{rank}"
        put(ldir / "d_re.bin", qre); put(ldir / "d_im.bin", qim)
        put(ldir / "layer_re.bin", layer_map(qre, rank)); put(ldir / "layer_im.bin", layer_map(qim, rank))


def link_rate(root: Path, rank: int = 1) -> None:
    code = np.fromfile(root / "ldpc/ascend_output/output.bin", dtype=np.int8)
    if code.size != NCB*NRAW or not np.any(code):
        raise RuntimeError("LDPC device output missing/wrong/all-zero")
    code = code.reshape(NCB, NRAW)
    d = root / f"rate/golden/rank{rank}"
    put(d / "code_blocks.bin", code); put(d / "rm_desc.bin", descriptors(rank, SLOTS)); put(d / "bits_nr.bin", rate_match(code, rank, SLOTS))


def link_scramble(root: Path, rank: int = 1) -> None:
    bits = np.fromfile(root / f"rate/ascend_output/rank{rank}/bits_nr.bin", dtype=np.int16).reshape(SLOTS, QM, rank*NPAD)
    signs, out = scramble(bits, rank, SLOTS)
    d = root / f"scramble/golden/rank{rank}"
    put(d / "bits_nr.bin", bits); put(d / "gold.bin", signs); put(d / "bits_qam.bin", out)


def select_qam(root: Path, slot: int, rank: int = 1) -> None:
    bits = np.fromfile(root / f"scramble/ascend_output/rank{rank}/bits_qam.bin", dtype=np.int16).reshape(SLOTS, QM, rank*NPAD)[slot]
    re, im = qam(bits, rank)
    d = root / f"qam/golden/rank{rank}"
    put(d / "bits_qam.bin", bits); put(d / "d_re.bin", re); put(d / "d_im.bin", im)


def link_layer(root: Path, rank: int = 1) -> None:
    re = np.fromfile(root / f"qam/ascend_output/rank{rank}/d_re.bin", dtype=np.float16)
    im = np.fromfile(root / f"qam/ascend_output/rank{rank}/d_im.bin", dtype=np.float16)
    d = root / f"layer/golden/rank{rank}"
    put(d / "d_re.bin", re); put(d / "d_im.bin", im)
    put(d / "layer_re.bin", layer_map(re, rank)); put(d / "layer_im.bin", layer_map(im, rank))


def collect(root: Path, slot: int, rank: int = 1) -> None:
    d = root / ("artifacts/layer_grid" if rank == 1 else f"artifacts/layer_grid_rank{rank}")
    d.mkdir(parents=True, exist_ok=True)
    for plane in ("re", "im"):
        src = root / f"layer/ascend_output/rank{rank}/layer_{plane}.bin"
        if src.stat().st_size != rank*NPAD*2 or not any(src.read_bytes()):
            raise RuntimeError("layer_map output missing/wrong/all-zero")
        shutil.copyfile(src, d / f"slot{slot:02d}_{plane}.bin")


def verify(root: Path) -> None:
    tx = np.fromfile(root / "artifacts/tx_bits.bin", dtype=np.int8)
    if tx.size != NCB*KCB or not np.any(tx):
        raise RuntimeError("tx_bits false-positive guard failed")
    for slot in range(SLOTS):
        for plane in ("re", "im"):
            value = np.fromfile(root / f"artifacts/layer_grid/slot{slot:02d}_{plane}.bin", dtype=np.float16)
            if value.size != NPAD or not np.any(value[:NRE]) or np.any(value[NRE:].view(np.uint16)):
                raise RuntimeError(f"slot {slot} layer {plane} boundary failed")
    result = {"schema_version": 1, "status": "PASS", "rank": 1, "slots": SLOTS,
              "device_stages": ["ldpc_encode", "rate_match_mimo", "scramble_mimo", "qam_mod_256_mimo", "layer_map_mimo"],
              "tx_bits_nonzero": int(np.count_nonzero(tx)), "tx_bits_sha256": hash_file(root / "artifacts/tx_bits.bin"),
              "encoded_stride": NRAW, "decoder_stride": 26112, "complete_coded_e2e": False}
    (root / "artifacts/tx_prefix_result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare","link-rate","link-scramble","select-qam","link-layer","collect","verify"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--legacy", type=Path)
    p.add_argument("--slot", type=int)
    p.add_argument("--rank", type=int, choices=(1,2,3,4), default=1)
    a = p.parse_args()
    if a.action == "prepare":
        if a.legacy is None: p.error("--legacy required")
        prepare(a.root, a.legacy)
    elif a.action in ("select-qam", "collect"):
        if a.slot is None: p.error("--slot required")
        {"select-qam":select_qam,"collect":collect}[a.action](a.root,a.slot,a.rank)
    elif a.action in ("link-rate", "link-scramble", "link-layer"):
        {"link-rate":link_rate,"link-scramble":link_scramble,"link-layer":link_layer}[a.action](a.root,a.rank)
    else:
        {"verify":verify}[a.action](a.root)


if __name__ == "__main__": main()
