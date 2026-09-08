#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import radio_profile as radio


def main() -> None:
    profile = radio.load_profile("fd8x8_rank1_4")
    assert profile.num_tx_antennas == profile.num_tx_rf_chains == 8
    assert profile.num_rx_antennas == profile.num_rx_rf_chains == 8
    assert profile.default_channel_gain_by_rank == {1: 0.32, 2: 0.32, 3: 0.5, 4: 0.32}
    assert profile.default_awgn_std_int16 == 80
    assert len(profile.source_sha256) == 64
    assert profile.supported_ranks == (1, 2, 3, 4)
    for rank, ports in ((1, 1), (2, 2), (3, 4), (4, 4)):
        assert profile.channel_gain(rank) == profile.default_channel_gain_by_rank[rank]
        mapping = radio.tx_antenna_matrix(profile, rank)
        assert mapping.shape == (8, ports)
        np.testing.assert_allclose(mapping.conj().T @ mapping,
                                   np.eye(ports), rtol=1e-5, atol=1e-5)
        physical = radio.physical_channel_matrix(profile, 0.16)
        effective = radio.effective_channel_matrix(profile, rank, 0.16)
        assert physical.shape == (8, 8)
        assert np.linalg.matrix_rank(physical) == 8
        assert effective.shape == (8, ports)
        assert np.linalg.matrix_rank(effective) == ports
        tx = np.zeros((2, ports, 32, 2), np.int16)
        tx[..., 0] = 100
        tx[..., 1] = -50
        tx_ant, rx, _, _ = radio.apply_fd8x8_channel(
            tx, profile, rank, 0.16, 0.0, 123)
        assert tx_ant.shape == (2, 8, 32, 2)
        assert rx.shape == (2, 8, 32, 2)
        padded = radio.pad_rx_capacity(profile, rx)
        assert padded.shape == (2, 64, 32, 2)
        assert np.array_equal(padded[:, :8], rx)
        assert not np.any(padded[:, 8:])
    print("[PASS] full-digital 8TXx8RX Rank1-4 radio profile contract")


if __name__ == "__main__":
    main()
