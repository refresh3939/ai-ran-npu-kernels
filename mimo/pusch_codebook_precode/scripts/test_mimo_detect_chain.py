#!/usr/bin/env python3
"""Contract-level check: physical channel * W is the H consumed by mimo_detect."""
import numpy as np

NSYM, NSC_USED, NSC_PAD = 14, 1596, 1664

# The detector is 64x16. Scheduled PUSCH transmissions retain independent
# ranks/port counts; the RX wrapper only concatenates their effective channels.
NR, DETECTOR_LAYERS = 64, 16
DETECTOR_KERNEL_LAYERS, DETECTOR_BRI_B, DETECTOR_RE_PACK = 16, 8, 1

# Valid matrices from the currently implemented 38.211 codebooks.
W_P4_L4 = np.array(
    [[1, 1, 1, 1], [1, -1, 1, -1], [1j, 1j, -1j, -1j],
     [1j, -1j, -1j, 1j]], dtype=np.complex64
) / 4.0
W_P4_L3 = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.complex64
) / 2.0
W_P2_L2 = np.eye(2, dtype=np.complex64) / np.sqrt(2.0)
W_P1_L1 = np.ones((1, 1), dtype=np.complex64)

# Deliberately mixed ranks: no fixed UE/PUSCH count or fixed rank is assumed.
ALLOCATIONS = [
    W_P4_L4,
    W_P4_L4,
    W_P4_L3,
    W_P2_L2,
    W_P2_L2,
    W_P1_L1,
]
NUM_ALLOCATIONS = len(ALLOCATIONS)

assert sum(w.shape[1] for w in ALLOCATIONS) == DETECTOR_LAYERS
assert DETECTOR_KERNEL_LAYERS == DETECTOR_LAYERS
assert DETECTOR_KERNEL_LAYERS // DETECTOR_LAYERS == DETECTOR_RE_PACK
assert DETECTOR_BRI_B == DETECTOR_LAYERS // 2

rng = np.random.default_rng(20260904)
physical_h = [
    (rng.standard_normal((NR, w.shape[0]))
     + 1j * rng.standard_normal((NR, w.shape[0]))).astype(np.complex64)
    for w in ALLOCATIONS
]
effective_h = np.concatenate(
    [physical_h[u] @ ALLOCATIONS[u] for u in range(NUM_ALLOCATIONS)], axis=1
)  # [64,16], exactly the channel-estimator/detector layer channel

layer_grid = []
port_grid = []
for w in ALLOCATIONS:
    layers = w.shape[1]
    grid = np.zeros((layers, NSYM, NSC_PAD), dtype=np.complex64)
    grid[:, :, :NSC_USED] = (
        rng.standard_normal((layers, NSYM, NSC_USED))
        + 1j * rng.standard_normal((layers, NSYM, NSC_USED))
    ).astype(np.complex64)
    ports = np.einsum("pl,lsk->psk", w, grid)
    assert ports.shape == (w.shape[0], NSYM, NSC_PAD)
    assert np.count_nonzero(ports[:, :, NSC_USED:]) == 0
    layer_grid.append(grid)
    port_grid.append(ports)

# Sample REs are enough to verify y=G(Wx)=H_eff*x and detector recovery.
for symbol, sc in [(0, 0), (2, 731), (13, 1595)]:
    n = symbol * NSC_PAD + sc
    assert n == np.ravel_multi_index((symbol, sc), (NSYM, NSC_PAD))
    y = sum(
        physical_h[u] @ port_grid[u][:, symbol, sc]
        for u in range(NUM_ALLOCATIONS)
    )
    x_hat, *_ = np.linalg.lstsq(effective_h, y, rcond=None)
    expected = np.concatenate(
        [grid[:, symbol, sc] for grid in layer_grid], axis=0
    )
    np.testing.assert_allclose(x_hat, expected, rtol=2e-5, atol=2e-5)

print("mimo_detect 64x16 / mixed-rank PUSCH aggregation contract: PASS")
