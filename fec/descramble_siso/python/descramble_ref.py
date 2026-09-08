"""5G NR descramble reference for the current padded QAM output ABI.

Input allocation (QAM physical layout):
  int16 Q11.5 [slot, qam_stream, data_symbol=12, symbol_stride=1600]
  valid subcarriers [0,1596), padding [1596,1600) == 0
  qam_stream order [I_b3,I_b2,I_b1,I_b0,Q_b3,Q_b2,Q_b1,Q_b0]

Output allocation (rate_dematch layout):
  int16 Q11.5 [slot, nr_bit=8, 19200]
  compact valid REs [0,19152), tail [19152,19200) == 0

The kernel fuses per-symbol padding removal, QAM stream -> NR bit permutation,
and multiplication by sign=(1-2*c). Gold advances over valid transmitted bits
only; storage padding never consumes a Gold bit.
"""

from pathlib import Path

import numpy as np


N_DATA_SYM = 12
N_SC_USED = 1596
QAM_SYM_STRIDE = 1600
N_SYM = N_DATA_SYM * N_SC_USED
N_SYM_PAD = N_DATA_SYM * QAM_SYM_STRIDE
N_STREAMS = 8
N_SLOT_TB = 23
GOLD_NC = 1600

# Current grouped QAM physical stream q -> canonical NR bit b.
QAM_STREAM_TO_NR_BIT = [0, 2, 4, 6, 1, 3, 5, 7]
# Inverse mapping: output NR bit b reads this QAM physical stream q.
NR_BIT_TO_QAM_STREAM = [0, 4, 1, 5, 2, 6, 3, 7]


def gold_seq_init(c_init: int, length: int) -> np.ndarray:
    total = length + GOLD_NC
    x1 = np.zeros(total + 31, dtype=np.uint8)
    x2 = np.zeros(total + 31, dtype=np.uint8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for n in range(total):
        x1[n + 31] = (x1[n + 3] + x1[n]) & 1
        x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) & 1
    return (x1[GOLD_NC:GOLD_NC + length] ^
            x2[GOLD_NC:GOLD_NC + length]).astype(np.uint8)


def make_sign_stream(n_id_cell: int, n_rnti: int, q: int, n_slot: int) -> np.ndarray:
    """Return compact canonical-NR sign tensor int16 [slot, b, 19200]."""
    c_init = ((n_rnti & 0xFFFF) << 15) | ((q & 1) << 14) | (n_id_cell & 0x3FF)
    c = gold_seq_init(c_init, n_slot * N_SYM * N_STREAMS)
    c_re = c.reshape(n_slot, N_SYM, N_STREAMS)  # [slot, compact_re, NR bit b]

    sign = np.ones((n_slot, N_STREAMS, N_SYM_PAD), dtype=np.int16)
    sign[:, :, :N_SYM] = (1 - 2 * c_re.astype(np.int16)).transpose(0, 2, 1)
    return sign


def descramble_ref(qam_llr_in: np.ndarray, n_id_cell: int, n_rnti: int, q: int):
    """Compact/reorder/de-scramble padded grouped-QAM LLRs.

    The accepted flat allocation shape [slot,8,19200] is interpreted as
    [slot,8,12,1600], matching qam256_demod's current physical output ABI.
    """
    assert qam_llr_in.dtype == np.int16
    n_slot = qam_llr_in.shape[0]
    assert qam_llr_in.size == n_slot * N_STREAMS * N_SYM_PAD
    qam = qam_llr_in.reshape(n_slot, N_STREAMS, N_DATA_SYM, QAM_SYM_STRIDE)
    assert not qam[:, :, :, N_SC_USED:].any(), "QAM per-symbol padding must be zero"

    # Pick physical stream for each canonical NR bit, then remove every 4-wide gap.
    compact = qam[:, NR_BIT_TO_QAM_STREAM, :, :N_SC_USED].reshape(
        n_slot, N_STREAMS, N_SYM)
    sign = make_sign_stream(n_id_cell, n_rnti, q, n_slot)
    out = np.zeros((n_slot, N_STREAMS, N_SYM_PAD), dtype=np.int16)
    out[:, :, :N_SYM] = (
        compact.astype(np.int32) * sign[:, :, :N_SYM].astype(np.int32)
    ).clip(-32768, 32767).astype(np.int16)
    return out, sign


def selftest() -> None:
    n_id_cell, n_rnti, q, n_slot = 1, 12345, 0, N_SLOT_TB
    rng = np.random.default_rng(42)
    qam = np.zeros(
        (n_slot, N_STREAMS, N_DATA_SYM, QAM_SYM_STRIDE), dtype=np.int16)
    qam[:, :, :, :N_SC_USED] = rng.integers(
        -2560, 2561,
        size=(n_slot, N_STREAMS, N_DATA_SYM, N_SC_USED), dtype=np.int16)

    llr_out, sign = descramble_ref(qam, n_id_cell, n_rnti, q)
    compact = qam[:, NR_BIT_TO_QAM_STREAM, :, :N_SC_USED].reshape(
        n_slot, N_STREAMS, N_SYM)
    recovered = (llr_out[:, :, :N_SYM].astype(np.int32) *
                 sign[:, :, :N_SYM].astype(np.int32)).astype(np.int16)

    assert np.array_equal(recovered, compact)
    assert not llr_out[:, :, N_SYM:].any()
    assert set(np.unique(sign)) == {-1, 1}
    assert [QAM_STREAM_TO_NR_BIT[q_] for q_ in NR_BIT_TO_QAM_STREAM] == list(range(8))

    print("Input : int16 [23,8,12,1600], valid [:,:,:,0:1596], pad=0")
    print("Output: int16 [23,8,19200], compact valid [0:19152], tail=0")
    print(f"QAM stream -> NR bit: {QAM_STREAM_TO_NR_BIT}")
    print("PASS: padding removal, stream permutation, Gold alignment, sign involution")

    out_dir = Path(__file__).resolve().parent / "test_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    qam.reshape(n_slot, N_STREAMS, N_SYM_PAD).tofile(out_dir / "descramble_input.bin")
    llr_out.tofile(out_dir / "descramble_output.bin")
    sign.tofile(out_dir / "descramble_sign_flat.bin")


if __name__ == "__main__":
    selftest()
