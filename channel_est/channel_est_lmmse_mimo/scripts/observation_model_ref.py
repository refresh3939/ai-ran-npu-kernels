#!/usr/bin/env python3
"""Independent 798/399 DMRS observation geometry and covariance reference.

The 399-point path models each OCC output as the arithmetic mean of two
adjacent same-comb LS observations. It never repeats a value into a nominal
798-point vector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

SC_USED = 1596
PILOT_798 = 798
PILOT_399 = 399

PORTS_BY_RANK = {
    1: (1000,),
    2: (1000, 1002),
    3: (1000, 1001, 1002),
    4: (1000, 1001, 1002, 1003),
}


@dataclass(frozen=True)
class ObservationGeometry:
    port: int
    comb: int
    reported_sc: np.ndarray
    support_sc: np.ndarray
    support_weight: np.ndarray

    @property
    def count(self) -> int:
        return int(self.reported_sc.size)


def build_geometries(ports: tuple[int, ...]) -> list[ObservationGeometry]:
    combs = np.asarray([(port - 1000) // 2 for port in ports], dtype=np.int64)
    result: list[ObservationGeometry] = []
    for port, comb in zip(ports, combs, strict=True):
        shared = int(np.count_nonzero(combs == comb)) == 2
        if shared:
            first = comb + 4 * np.arange(PILOT_399, dtype=np.int64)
            support = np.stack((first, first + 2), axis=1)
            reported = first + 1
            weights = np.full((PILOT_399, 2), 0.5, dtype=np.float64)
        else:
            reported = comb + 2 * np.arange(PILOT_798, dtype=np.int64)
            support = reported[:, None]
            weights = np.ones((PILOT_798, 1), dtype=np.float64)
        if np.any(support < 0) or np.any(support >= SC_USED):
            raise ValueError(f"port {port} observation support exceeds the active grid")
        result.append(ObservationGeometry(port, int(comb), reported, support, weights))
    return result


def observation_covariances(
    frequency_covariance: np.ndarray, geometry: ObservationGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Return Rhp[1596,count] and Rpp[count,count] for this observation."""
    if frequency_covariance.shape != (SC_USED, SC_USED):
        raise ValueError("frequency covariance must be [1596,1596]")
    count, width = geometry.support_sc.shape
    rhp = np.zeros((SC_USED, count), dtype=np.complex128)
    rpp = np.zeros((count, count), dtype=np.complex128)
    for left in range(width):
        left_sc = geometry.support_sc[:, left]
        left_weight = geometry.support_weight[:, left]
        rhp += frequency_covariance[:, left_sc] * left_weight[None, :]
        for right in range(width):
            right_sc = geometry.support_sc[:, right]
            right_weight = geometry.support_weight[:, right]
            rpp += (frequency_covariance[np.ix_(left_sc, right_sc)] *
                    left_weight[:, None] * right_weight[None, :])
    return rhp, rpp


def self_test() -> None:
    rank3 = build_geometries(PORTS_BY_RANK[3])
    rank4 = build_geometries(PORTS_BY_RANK[4])
    assert [value.count for value in rank3] == [399, 399, 798]
    assert [value.count for value in rank4] == [399, 399, 399, 399]

    # Small Toeplitz-like covariance is enough to independently verify the
    # two-support averaging formula without performing an eigendecomposition.
    index = np.arange(SC_USED)
    covariance = np.exp(-np.abs(index[:, None] - index[None, :]) / 7.0).astype(np.complex128)
    geometry = rank3[0]
    rhp, rpp = observation_covariances(covariance, geometry)
    first = geometry.support_sc[:, 0]
    second = geometry.support_sc[:, 1]
    np.testing.assert_allclose(rhp, 0.5 * (covariance[:, first] + covariance[:, second]))
    expected_rpp = 0.25 * (
        covariance[np.ix_(first, first)] + covariance[np.ix_(first, second)] +
        covariance[np.ix_(second, first)] + covariance[np.ix_(second, second)])
    np.testing.assert_allclose(rpp, expected_rpp)
    assert rpp.shape == (399, 399)
    assert not np.array_equal(geometry.reported_sc, first)
    print("[PASS] independent observation model: Rank3=399/399/798 Rank4=399x4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        for rank, ports in PORTS_BY_RANK.items():
            counts = [geometry.count for geometry in build_geometries(ports)]
            print(f"Rank{rank}: ports={ports} observations={counts}")


if __name__ == "__main__":
    main()
