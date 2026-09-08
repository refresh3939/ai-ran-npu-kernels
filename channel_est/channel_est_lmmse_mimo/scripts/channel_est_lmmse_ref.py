#!/usr/bin/env python3
"""Generate a compile-time configured low-rank LMMSE test case.

The public input exactly matches mimo_dmrs_ls: natural [NR,L,D,832] h_ls plus
pilot_count/pilot_sc. Rank3/4 use genuine 399-point two-support OCC covariance
models; Rank3 mixes those with one independent 798-point comb in one launch.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

from observation_model_ref import (
    PORTS_BY_RANK,
    ObservationGeometry,
    build_geometries,
    observation_covariances,
)

NR = int(os.environ.get("NR", "64"))
NL = int(os.environ.get("NL", "2"))
N_SYMBOL, N_DMRS = 14, 2
N_SC_USED, N_SC_PAD = 1596, 1664
N_PILOT, N_PILOT_PAD = 798, 832
RANK = int(os.environ.get("RANK", "96"))
RX_GROUP, N_RX_GROUP = 8, NR // 8
SC_TILE, N_SC_TILE = 16, 100
N_SC_COMPUTE = SC_TILE * N_SC_TILE
CASE = f"case_0_m{NR}_k{NL}_r{RANK}"
SCS_HZ = 30_000.0
N_FFT = 2048
T_SYMBOL = 1.0 / SCS_HZ + 144.0 / (N_FFT * SCS_HZ)
DMRS_SYMBOLS = (2, 11)

assert NR in (8, 16, 32, 64), "NR must be one of 8,16,32,64"
assert NL in (1, 2, 3, 4), "NL must be one of 1,2,3,4"
assert 16 <= RANK <= 128 and RANK % 16 == 0, "RANK must be 16-aligned in [16,128]"


def complex_from_half(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    return real.astype(np.float16).astype(np.float32) + 1j * imag.astype(np.float16).astype(np.float32)


def frequency_covariance(tau_seconds: float) -> np.ndarray:
    delta = (np.arange(N_SC_USED)[:, None] - np.arange(N_SC_USED)[None, :]) * SCS_HZ
    return (1.0 / (1.0 + 2j * np.pi * delta * tau_seconds)).astype(np.complex128)


def j0_fallback(x: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x, dtype=np.float64)
    term = np.ones_like(x, dtype=np.float64)
    for k in range(1, 41):
        result += term
        term *= -(x * x) / (4.0 * k * k)
        if np.max(np.abs(term)) < 1e-14:
            break
    return result


def time_covariance(fd_hz: float) -> np.ndarray:
    try:
        from scipy.special import j0
    except ImportError:
        j0 = j0_fallback
    delta = (np.arange(N_SYMBOL)[:, None] - np.arange(N_SYMBOL)[None, :]) * T_SYMBOL
    return j0(2 * np.pi * fd_hz * delta).astype(np.complex128)


def time_weights(rt: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Per-subcarrier two-pilot Wiener weights, shape [1596,14,2]."""
    p0, p1 = DMRS_SYMBOLS
    rap = rt[:, [p0, p1]]
    aa = rt[p0, p0] + error
    dd = rt[p1, p1] + error
    bb = rt[p0, p1]
    det = aa * dd - np.abs(bb) ** 2
    result = np.empty((N_SC_USED, N_SYMBOL, N_DMRS), dtype=np.complex128)
    result[:, :, 0] = rap[:, 0][None, :] * (dd / det)[:, None]
    result[:, :, 0] += rap[:, 1][None, :] * (-np.conj(bb) / det)[:, None]
    result[:, :, 1] = rap[:, 0][None, :] * (-bb / det)[:, None]
    result[:, :, 1] += rap[:, 1][None, :] * (aa / det)[:, None]
    return result


def make_factors(noise_variance: float, tau_ns: float, fd_hz: float,
                 geometry: ObservationGeometry):
    rf = frequency_covariance(tau_ns * 1e-9)
    rt = time_covariance(fd_hz)
    rhp, rpp = observation_covariances(rf, geometry)
    eigenvalue, eigenvector = np.linalg.eigh(rpp)
    order = np.argsort(eigenvalue.real)[::-1][:RANK]
    lam = np.maximum(eigenvalue[order].real, 0.0)
    ur = eigenvector[:, order]

    # h_freq = A * B * h_ls. B folds D=(lambda+noise)^-1.
    # OCC output is a two-sample arithmetic mean, so independent LS noise is
    # reduced by sum([1/2,1/2]^2)=1/2. The 798-point path keeps factor 1.
    observation_noise = noise_variance * float(np.sum(geometry.support_weight[0] ** 2))
    diag = 1.0 / (lam + observation_noise)
    factor_b = diag[:, None] * ur.conj().T
    factor_a = rhp @ ur

    # Sionna-style non-final-dimension re-scaling, folded into A.
    raa = np.real(np.diag(rf))
    hhat_var = np.real(np.sum((factor_a * diag[None, :]) * factor_a.conj(), axis=1))
    error = np.maximum(raa - hhat_var, 0.0)
    denominator = hhat_var + raa - error
    scale = np.divide(2.0 * raa, denominator, out=np.zeros_like(raa), where=denominator != 0)
    factor_a *= scale[:, None]
    error = np.maximum(scale * (scale - 1.0) * hhat_var + (1.0 - scale) * raa + scale * error, 0.0)
    return factor_a, factor_b, time_weights(rt, error), lam


def pack_factor_b(factor: np.ndarray) -> np.ndarray:
    padded = np.zeros((RANK, N_PILOT_PAD), dtype=np.complex64)
    padded[:, :factor.shape[1]] = factor
    return padded.reshape(RANK // 16, 16, N_PILOT_PAD // 16, 16).transpose(0, 2, 1, 3)


def pack_factor_a(factor: np.ndarray) -> np.ndarray:
    padded = np.zeros((N_SC_COMPUTE, RANK), dtype=np.complex64)
    padded[:N_SC_USED] = factor
    return padded.reshape(N_SC_TILE, SC_TILE, RANK).transpose(0, 2, 1)


def pack_wt(weights: np.ndarray) -> np.ndarray:
    padded = np.zeros((N_SC_COMPUTE, N_SYMBOL, N_DMRS), dtype=np.complex64)
    padded[:N_SC_USED] = weights
    return padded.reshape(N_SC_TILE, SC_TILE, N_SYMBOL, N_DMRS).transpose(0, 2, 3, 1)


def pack_cube_time_factors(a_packed: np.ndarray, wt_packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pre-fold real time weights into A for the experimental Cube path.

    The current Jakes covariance produces real-valued Wiener weights.  Keep the
    complex multiplication here so a non-zero imaginary component is rejected
    explicitly instead of silently changing the estimator.
    """
    if np.max(np.abs(wt_packed.imag)) > 1e-7:
        raise ValueError("cube_time_fused currently requires real time weights")
    fused = a_packed[:, None, None, :, :] * wt_packed[:, :, :, None, :]
    return fused[:, :, 0], fused[:, :, 1]


def pack_post_frequency_time_matrix(wt_packed: np.ndarray) -> np.ndarray:
    """Build P[32,224] so [64,2,16] @ P -> [64,14,16].

    The returned layout is [tile,k_block=2,n_block=14,k0=16,n0=16], matching
    the L0B K-block-major order so one LoadData repeat can load all 28 blocks.
    """
    if np.max(np.abs(wt_packed.imag)) > 1e-7:
        raise ValueError("post-frequency Cube time GEMM currently requires real weights")
    matrix = np.zeros((N_SC_TILE, 32, N_SYMBOL * SC_TILE), dtype=np.float32)
    for symbol in range(N_SYMBOL):
        for sc in range(SC_TILE):
            column = symbol * SC_TILE + sc
            matrix[:, sc, column] = wt_packed[:, symbol, 0, sc].real
            matrix[:, SC_TILE + sc, column] = wt_packed[:, symbol, 1, sc].real
    return matrix.reshape(N_SC_TILE, 2, 16, N_SYMBOL, SC_TILE).transpose(0, 1, 3, 2, 4)


def write_half(path: Path, value: np.ndarray) -> None:
    np.asarray(value, dtype=np.float16).tofile(path)


def pilot_hash(values: np.ndarray) -> int:
    result = 1469598103934665603
    for byte in np.asarray(values, dtype="<u2").tobytes():
        result ^= byte
        result = (result * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return result


def simulate_and_write(gold: Path, a_packed: np.ndarray, b_packed: np.ndarray,
                       wt_packed: np.ndarray, hls: np.ndarray) -> None:
    # Reconstruct exactly the fp16 inputs consumed by the kernel.
    bq = complex_from_half(b_packed.real, b_packed.imag)
    bq = bq.transpose(0, 1, 3, 2, 4).reshape(NL, RANK, N_PILOT_PAD)
    aq = complex_from_half(a_packed.real, a_packed.imag)
    aq = aq.transpose(0, 1, 3, 2).reshape(NL, N_SC_COMPUTE, RANK)
    wq = complex_from_half(wt_packed.real, wt_packed.imag)
    wq = wq.transpose(0, 1, 4, 2, 3).reshape(NL, N_SC_COMPUTE, N_SYMBOL, N_DMRS)
    hq = complex_from_half(hls.real, hls.imag)

    out_shape = (NR, 16, N_SYMBOL, N_SC_PAD)
    out_re = np.memmap(gold / "gold_h_re.bin", mode="w+", dtype=np.float16, shape=out_shape)
    out_im = np.memmap(gold / "gold_h_im.bin", mode="w+", dtype=np.float16, shape=out_shape)
    out_re[:] = 0
    out_im[:] = 0
    for layer in range(NL):
        for group in range(N_RX_GROUP):
            # Cube 1 output is rounded to half before the synchronization scratch.
            t = bq[layer] @ hq[layer, group]
            tq = complex_from_half(t.real, t.imag)
            # Cube 2 output is rounded to half before vector time interpolation.
            hf = aq[layer] @ tq
            hfq = complex_from_half(hf.real, hf.imag)
            for local_rx in range(RX_GROUP):
                pair = hfq[:, 2 * local_rx:2 * local_rx + 2]
                value = np.einsum("csd,cd->sc", wq[layer], pair, optimize=True)
                rx = group * RX_GROUP + local_rx
                out_re[rx, layer, :, :N_SC_COMPUTE] = value.real.astype(np.float16)
                out_im[rx, layer, :, :N_SC_COMPUTE] = value.imag.astype(np.float16)
    out_re.flush()
    out_im.flush()


def self_test() -> None:
    rng = np.random.default_rng(7)
    b = (rng.standard_normal((RANK, N_PILOT)) + 1j * rng.standard_normal((RANK, N_PILOT))).astype(np.complex64)
    a = (rng.standard_normal((N_SC_USED, RANK)) + 1j * rng.standard_normal((N_SC_USED, RANK))).astype(np.complex64)
    bp = pack_factor_b(b)
    ap = pack_factor_a(a)
    bu = bp.transpose(0, 2, 1, 3).reshape(RANK, N_PILOT_PAD)[:, :N_PILOT]
    au = ap.transpose(0, 2, 1).reshape(N_SC_COMPUTE, RANK)[:N_SC_USED]
    np.testing.assert_array_equal(bu, b)
    np.testing.assert_array_equal(au, a)
    x = (rng.standard_normal((32, 16)) + 1j * rng.standard_normal((32, 16))).astype(np.complex64)
    y = (rng.standard_normal((16, 32)) + 1j * rng.standard_normal((16, 32))).astype(np.complex64)
    rr = x.real @ y.real + x.imag @ (-y.imag)
    ii = x.real @ y.imag + x.imag @ y.real
    np.testing.assert_allclose(rr + 1j * ii, x @ y, rtol=2e-6, atol=2e-5)
    wt = rng.standard_normal((N_SC_TILE, N_SYMBOL, N_DMRS, SC_TILE)).astype(np.float32)
    post = pack_post_frequency_time_matrix(wt.astype(np.complex64))
    post_matrix = post.transpose(0, 1, 3, 2, 4).reshape(N_SC_TILE, 32, N_SYMBOL * SC_TILE)
    hf = rng.standard_normal((NR, N_DMRS, SC_TILE)).astype(np.float32)
    expected = np.einsum("rdc,sdc->rsc", hf, wt[0], optimize=True)
    actual = (hf.reshape(NR, 32) @ post_matrix[0]).reshape(NR, N_SYMBOL, SC_TILE)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)
    print("[self-test] packing and split-complex GEMM formulas PASS")


def generate(args: argparse.Namespace) -> None:
    root = Path(os.environ.get("AIRAN_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    gold = root / "golden" / CASE
    gold.mkdir(parents=True, exist_ok=True)
    print(f"[case] {CASE}; separated h_ls; seed={args.seed}; SNR={args.snr_db} dB")
    ports = PORTS_BY_RANK[NL]
    geometries = build_geometries(ports)
    combs = np.asarray([value.comb for value in geometries], dtype=np.uint16)
    factor_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    factors = []
    for geometry in geometries:
        key = (geometry.comb, geometry.count)
        if key not in factor_cache:
            factor_cache[key] = make_factors(
                10.0 ** (-args.snr_db / 10.0), args.tau_ns, args.fd_hz, geometry)
        factors.append(factor_cache[key])
    bp = np.stack([pack_factor_b(value[1]) for value in factors])
    ap = np.stack([pack_factor_a(value[0]) for value in factors])
    wp = np.stack([pack_wt(value[2]) for value in factors])
    fused = [pack_cube_time_factors(ap[layer], wp[layer]) for layer in range(NL)]
    fused0 = np.stack([value[0] for value in fused])
    fused1 = np.stack([value[1] for value in fused])
    post_time = np.stack([pack_post_frequency_time_matrix(wp[layer]) for layer in range(NL)])

    rng = np.random.default_rng(args.seed)
    natural = ((rng.standard_normal((NR, NL, N_DMRS, N_PILOT_PAD)) +
                1j * rng.standard_normal((NR, NL, N_DMRS, N_PILOT_PAD))) /
               np.sqrt(2.0)).astype(np.complex64)
    for layer, geometry in enumerate(geometries):
        natural[:, layer, :, geometry.count:] = 0
    hls = np.zeros((NL, N_RX_GROUP, N_PILOT_PAD, RX_GROUP * N_DMRS), dtype=np.complex64)
    for rx in range(NR):
        group, local_rx = divmod(rx, RX_GROUP)
        for layer in range(NL):
            for dmrs in range(N_DMRS):
                hls[layer, group, :, local_rx * N_DMRS + dmrs] = natural[rx, layer, dmrs]

    pilot_count = np.zeros(16, dtype=np.uint16)
    pilot_sc = np.zeros((NL, N_DMRS, N_PILOT_PAD), dtype=np.uint16)
    for layer, geometry in enumerate(geometries):
        pilot_count[layer * N_DMRS:(layer + 1) * N_DMRS] = geometry.count
        pilot_sc[layer, :, :geometry.count] = geometry.reported_sc.astype(np.uint16)
    gather_index = np.empty((N_PILOT_PAD, RX_GROUP * N_DMRS), dtype=np.uint32)
    for pilot in range(N_PILOT_PAD):
        for column in range(RX_GROUP * N_DMRS):
            gather_index[pilot, column] = 2 * (column * N_PILOT_PAD + pilot)

    write_half(gold / "factor_b_re.bin", bp.real)
    write_half(gold / "factor_b_im.bin", bp.imag)
    write_half(gold / "factor_a_re.bin", ap.real)
    write_half(gold / "factor_a_im.bin", ap.imag)
    write_half(gold / "factor_a_neg_im.bin", -ap.imag)
    write_half(gold / "wt_re.bin", wp.real)
    write_half(gold / "wt_im.bin", wp.imag)
    write_half(gold / "cube_time_fused_d0_re.bin", fused0.real)
    write_half(gold / "cube_time_fused_d0_im.bin", fused0.imag)
    write_half(gold / "cube_time_fused_d1_re.bin", fused1.real)
    write_half(gold / "cube_time_fused_d1_im.bin", fused1.imag)
    write_half(gold / "cube_time_post_matrix.bin", post_time)
    write_half(gold / "h_ls_re.bin", natural.real)
    write_half(gold / "h_ls_im.bin", natural.imag)
    pilot_count.tofile(gold / "pilot_count.bin")
    pilot_sc.tofile(gold / "pilot_sc.bin")
    gather_index.tofile(gold / "ce_pack_gather_index.bin")
    model_counts = np.zeros((4, N_DMRS), dtype=np.uint16)
    model_hashes = np.zeros((4, N_DMRS), dtype=np.uint64)
    model_counts[:NL] = pilot_count[:NL * N_DMRS].reshape(NL, N_DMRS)
    for layer in range(NL):
        for dmrs in range(N_DMRS):
            count = int(model_counts[layer, dmrs])
            model_hashes[layer, dmrs] = pilot_hash(pilot_sc[layer, dmrs, :count])
    model_blob = struct.pack(
        "<I6H8H8Q8I", 0x43455731, 1, 128, NL, N_DMRS, RANK, N_PILOT_PAD,
        *model_counts.reshape(-1).tolist(), *model_hashes.reshape(-1).tolist(),
        *([0] * 8))
    (gold / "weight_model.bin").write_bytes(model_blob)
    write_half(gold / "hls_re.bin", hls.real)
    write_half(gold / "hls_im.bin", hls.imag)
    write_half(gold / "hls_neg_im.bin", -hls.imag)
    simulate_and_write(gold, ap, bp, wp, hls)

    metadata = {
        "case": CASE,
        "shape": {"nr": NR, "nl": NL, "symbols": N_SYMBOL, "sc_used": N_SC_USED,
                  "sc_pad": N_SC_PAD,
                  "pilot_count": [geometry.count for geometry in geometries],
                  "pilot_pad": N_PILOT_PAD, "rank": RANK},
        "contract": "h_ls is already DMRS/OCC separated; columns are 8 Rx x 2 DMRS symbols",
        "model": {"snr_db": args.snr_db, "tau_ns": args.tau_ns, "fd_hz": args.fd_hz,
                  "seed": args.seed,
                  "min_retained_eigenvalue": [float(value[3][-1]) for value in factors],
                  "dmrs_ports": list(ports), "pilot_combs": combs.tolist(),
                  "occ_covariance": "two-support arithmetic-mean observation"},
        "correctness": {"max_abs": 0.1, "nrmse": 0.03},
    }
    (gold / "case.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[io] wrote fp16 inputs and reference output to {gold}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--snr-db", type=float, default=15.0)
    parser.add_argument("--tau-ns", type=float, default=100.0)
    parser.add_argument("--fd-hz", type=float, default=50.0)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        generate(args)


if __name__ == "__main__":
    main()
