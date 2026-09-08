"""5G NR TX scrambling reference plus rate_match -> QAM ABI adaptation.

Input : int16 {0,1} [slot, linear-NR-bit=8, 19200], compact RE order.
Output: int16 {0,1} [slot, grouped-QAM-stream=8, 12, 1600]. Each data
symbol contains 1596 valid bits followed by four explicit zeros.

The Gold sequence advances only over valid modulation bits. Storage padding is
not part of the 38.211 sequence. QAM streams are [I0,I1,I2,I3,Q0,Q1,Q2,Q3].
"""

import os
import numpy as np

N_DATA_SYM = 12
N_SC_USED = 1596
QAM_SYM_STRIDE = 1600
N_SYM = N_DATA_SYM * N_SC_USED
N_SYM_PAD = N_DATA_SYM * QAM_SYM_STRIDE
N_STREAMS = 8
N_SLOT_TB = 23
GOLD_LEN = 1600

# Downstream grouped QAM stream -> input linear modulation bit index.
QAM_STREAM_TO_NR_BIT = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.intp)


def gold_seq_init(c_init: int, length: int) -> np.ndarray:
    total = length + GOLD_LEN
    x1 = np.zeros(total + 31, dtype=np.uint8)
    x2 = np.zeros(total + 31, dtype=np.uint8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for n in range(total):
        x1[n + 31] = (x1[n + 3] + x1[n]) & 1
        x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) & 1
    return (x1[GOLD_LEN:GOLD_LEN + length] ^
            x2[GOLD_LEN:GOLD_LEN + length]).astype(np.uint8)


def make_gold_compact(n_id_cell: int, n_rnti: int, q: int, n_slot: int) -> np.ndarray:
    """Return compact linear-bit Gold storage [slot,8,19200], tail 48 zero."""
    c_init = ((n_rnti & 0xFFFF) << 15) | ((q & 1) << 14) | (n_id_cell & 0x3FF)
    c = gold_seq_init(c_init, n_slot * N_SYM * N_STREAMS)
    gold = np.zeros((n_slot, N_STREAMS, N_SYM_PAD), dtype=np.int16)
    gold[:, :, :N_SYM] = c.reshape(n_slot, N_SYM, N_STREAMS).transpose(0, 2, 1)
    return gold


# Backward-compatible name used by older local callers.
make_gold_stream = make_gold_compact


def scramble_ref(bits_in: np.ndarray, n_id_cell: int, n_rnti: int, q: int):
    """Scramble valid compact bits, group I/Q streams, and insert symbol padding."""
    assert bits_in.dtype == np.int16
    n_slot = bits_in.shape[0]
    assert bits_in.shape == (n_slot, N_STREAMS, N_SYM_PAD)
    assert set(np.unique(bits_in).tolist()) <= {0, 1}

    gold = make_gold_compact(n_id_cell, n_rnti, q, n_slot)
    scrambled_linear = np.bitwise_xor(bits_in[:, :, :N_SYM], gold[:, :, :N_SYM])
    grouped = scrambled_linear[:, QAM_STREAM_TO_NR_BIT, :]
    out = np.zeros((n_slot, N_STREAMS, N_DATA_SYM, QAM_SYM_STRIDE), dtype=np.int16)
    out[:, :, :, :N_SC_USED] = grouped.reshape(n_slot, N_STREAMS, N_DATA_SYM, N_SC_USED)
    return out, gold


def descramble_hard_ref(bits_qam: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Inverse ABI adapter and hard-bit XOR, matching descramble_siso semantics."""
    n_slot = bits_qam.shape[0]
    assert bits_qam.shape == (n_slot, N_STREAMS, N_DATA_SYM, QAM_SYM_STRIDE)
    assert gold.shape == (n_slot, N_STREAMS, N_SYM_PAD)
    compact_grouped = bits_qam[:, :, :, :N_SC_USED].reshape(n_slot, N_STREAMS, N_SYM)
    restored = np.zeros((n_slot, N_STREAMS, N_SYM_PAD), dtype=np.int16)
    for qam_stream, nr_bit in enumerate(QAM_STREAM_TO_NR_BIT):
        restored[:, nr_bit, :N_SYM] = np.bitwise_xor(
            compact_grouped[:, qam_stream, :], gold[:, nr_bit, :N_SYM])
    return restored


if __name__ == "__main__":
    n_id_cell, n_rnti, q, n_slot = 1, 12345, 0, N_SLOT_TB
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=(n_slot, N_STREAMS, N_SYM_PAD), dtype=np.int16)
    bits[:, :, N_SYM:] = 0
    out, gold = scramble_ref(bits, n_id_cell, n_rnti, q)
    back = descramble_hard_ref(out, gold)

    assert set(np.unique(out).tolist()) <= {0, 1}
    assert np.count_nonzero(out[:, :, :, N_SC_USED:]) == 0
    assert np.count_nonzero(gold[:, :, N_SYM:]) == 0
    assert np.array_equal(back, bits)
    print(f"Input shape: {bits.shape}; output shape: {out.shape}")
    print(f"QAM grouped stream -> linear bit: {QAM_STREAM_TO_NR_BIT.tolist()}")
    print("padding, Gold extent, and hard round-trip: PASS")

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
    os.makedirs(outdir, exist_ok=True)
    bits.tofile(os.path.join(outdir, "scramble_input.bin"))
    out.tofile(os.path.join(outdir, "scramble_output.bin"))
    gold.tofile(os.path.join(outdir, "scramble_gold_flat.bin"))
    print(f"Saved {outdir}/")
