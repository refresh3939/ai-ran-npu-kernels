#!/usr/bin/env python3
"""Physical-radio profiles kept separate from the logical PUSCH ABI.

The existing PUSCH operators model layers and logical antenna ports.  This
module adds the physical antenna/RF-chain topology used by loopback channel
tests without changing Rank or the four-port DMRS contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

import numpy as np


COMMON_ROOT = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from mimo_profile_compiler import ProfileError, compile_profile


PROFILE_ENV = "PUSCH_MIMO_RADIO_PROFILE"
DEFAULT_PROFILE = "legacy"
PROFILE_ROOT = COMMON_ROOT / "profiles"
RESOLVED_PLAN_ENV = "PUSCH_MIMO_RESOLVED_PLAN"


@dataclass(frozen=True)
class RadioProfile:
    name: str
    architecture: str
    num_tx_antennas: int
    num_tx_rf_chains: int
    num_rx_antennas: int
    num_rx_rf_chains: int
    supported_ranks: tuple[int, ...]
    logical_tx_ports_by_rank: dict[int, int]
    tx_antenna_mapping: str
    physical_channel: str
    default_channel_gain_by_rank: dict[int, float]
    default_awgn_std_int16: float
    detector_rx_capacity: int
    rx_capacity_adapter: str
    source_sha256: str

    def tx_ports(self, rank: int) -> int:
        if rank not in self.supported_ranks:
            raise ValueError(f"profile {self.name} does not support Rank{rank}")
        return self.logical_tx_ports_by_rank[rank]

    def channel_gain(self, rank: int) -> float:
        if rank not in self.supported_ranks:
            raise ValueError(f"profile {self.name} does not support Rank{rank}")
        return self.default_channel_gain_by_rank[rank]


@lru_cache(maxsize=None)
def load_profile(name: str) -> RadioProfile:
    path = PROFILE_ROOT / f"{name}.json"
    try:
        raw = json.loads(path.read_text())
        ranks = tuple(int(rank) for rank in raw["spatial"]["supported_ranks"])
        plans = {rank: compile_profile(path, rank).value for rank in ranks}
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ProfileError) as error:
        raise ValueError(f"invalid common MIMO profile {name!r}: {error}") from error
    if not ranks:
        raise ValueError(f"common MIMO profile {name!r} has no supported ranks")
    first = plans[ranks[0]]
    result = RadioProfile(
        name=name,
        architecture=first["topology"]["architecture"],
        num_tx_antennas=first["topology"]["tx_antennas"],
        num_tx_rf_chains=first["topology"]["tx_rf_chains"],
        num_rx_antennas=first["topology"]["rx_antennas"],
        num_rx_rf_chains=first["topology"]["rx_rf_chains"],
        supported_ranks=ranks,
        logical_tx_ports_by_rank={
            rank: plans[rank]["active"]["logical_tx_ports"] for rank in ranks
        },
        tx_antenna_mapping=first["spatial"]["antenna_mapping"],
        physical_channel=first["channel"]["model"],
        default_channel_gain_by_rank={
            rank: plans[rank]["channel"]["gain"] for rank in ranks
        },
        default_awgn_std_int16=first["channel"]["awgn_std_int16"],
        detector_rx_capacity=first["receiver"]["max_rx_antennas"],
        rx_capacity_adapter=first["receiver"]["rx_capacity_adapter"],
        source_sha256=first["source"]["sha256"],
    )
    validate_profile(result)
    return result


def selected_profile() -> RadioProfile | None:
    name = os.environ.get(PROFILE_ENV, DEFAULT_PROFILE)
    return None if name == DEFAULT_PROFILE else load_profile(name)


def validate_profile(profile: RadioProfile) -> None:
    if profile.architecture != "full_digital":
        raise ValueError("only full_digital is implemented")
    if profile.num_tx_rf_chains != profile.num_tx_antennas:
        raise ValueError("full-digital TX requires one RF chain per antenna")
    if profile.num_rx_rf_chains != profile.num_rx_antennas:
        raise ValueError("full-digital RX requires one RF chain per antenna")
    if profile.supported_ranks != (1, 2, 3, 4):
        raise ValueError("the fd8x8 profile must preserve Rank1-4")
    if set(profile.logical_tx_ports_by_rank) != set(profile.supported_ranks):
        raise ValueError("every supported rank needs a logical TX-port mapping")
    for rank, ports in profile.logical_tx_ports_by_rank.items():
        if ports < rank or ports > profile.num_tx_antennas:
            raise ValueError(f"invalid Rank{rank} logical port count {ports}")
    if profile.detector_rx_capacity < profile.num_rx_antennas:
        raise ValueError("detector capacity is smaller than physical RX count")
    if profile.tx_antenna_mapping != "semi_unitary_dft":
        raise ValueError("unsupported TX antenna mapping")
    if profile.physical_channel != "deterministic_unitary":
        raise ValueError("unsupported physical channel")
    if set(profile.default_channel_gain_by_rank) != set(profile.supported_ranks):
        raise ValueError("every supported rank needs a default channel gain")
    if any(not 0.0 < gain <= 1.0
           for gain in profile.default_channel_gain_by_rank.values()):
        raise ValueError("default channel gains must be in (0,1]")
    if not 0.0 < profile.default_awgn_std_int16 <= 200.0:
        raise ValueError("default AWGN must be nonzero and <= 200")
    if (profile.num_tx_antennas, profile.num_rx_antennas) != (8, 8):
        raise ValueError("fd8x8 profile must be physically 8TX x 8RX")
    if profile.rx_capacity_adapter != "zero_pad_signal_repeat_noise":
        raise ValueError("unsupported RX capacity adapter")


def selected_plan() -> dict[str, object] | None:
    path = os.environ.get(RESOLVED_PLAN_ENV)
    if path is None:
        return None
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid resolved MIMO plan {path!r}: {error}") from error
    if value.get("schema") != "airan.mimo.resolved_plan.v1":
        raise ValueError(f"unsupported resolved MIMO plan schema in {path!r}")
    return value


def validate_resolved_plan(profile: RadioProfile, rank: int) -> None:
    plan = selected_plan()
    if plan is None:
        return
    if (plan["source"]["profile_name"] != profile.name or
            plan["source"]["sha256"] != profile.source_sha256 or
            plan["active"]["rank"] != rank or
            plan["active"]["logical_tx_ports"] != profile.tx_ports(rank)):
        raise ValueError("resolved MIMO plan does not match active radio profile/Rank")
    topology = plan["topology"]
    if (topology["tx_antennas"] != profile.num_tx_antennas or
            topology["rx_antennas"] != profile.num_rx_antennas):
        raise ValueError("resolved MIMO plan topology mismatch")


def _dft(size: int) -> np.ndarray:
    row = np.arange(size, dtype=np.float32)[:, None]
    column = np.arange(size, dtype=np.float32)[None, :]
    return (np.exp(-2j * np.pi * row * column / size) /
            np.sqrt(float(size))).astype(np.complex64)


def tx_antenna_matrix(profile: RadioProfile, rank: int) -> np.ndarray:
    """Return semi-unitary F_ant[physical_tx, logical_port]."""
    ports = profile.tx_ports(rank)
    return _dft(profile.num_tx_antennas)[:, :ports]


def physical_channel_matrix(profile: RadioProfile, gain: float) -> np.ndarray:
    """Return a deterministic dense full-rank H[physical_rx,physical_tx]."""
    if not 0.0 < gain <= 1.0:
        raise ValueError("channel gain must be in (0,1]")
    unitary = _dft(profile.num_tx_antennas)
    phase = np.exp(2j * np.pi * (np.arange(profile.num_tx_antennas) + 1) ** 2 / 17)
    channel = unitary @ np.diag(phase.astype(np.complex64)) @ unitary.conj().T
    # Preserve `gain` as the per-coefficient RMS convention used by the
    # legacy 64xP channel instead of as the matrix spectral norm.
    return (gain * np.sqrt(float(profile.num_tx_antennas)) * channel).astype(np.complex64)


def effective_channel_matrix(profile: RadioProfile, rank: int,
                             gain: float) -> np.ndarray:
    return physical_channel_matrix(profile, gain) @ tx_antenna_matrix(profile, rank)


def pad_rx_capacity(profile: RadioProfile, value: np.ndarray,
                    rx_axis: int = 1) -> np.ndarray:
    """Zero-pad an 8Rx tensor to the waveform batch's 64Rx staging ABI."""
    if value.shape[rx_axis] != profile.num_rx_antennas:
        raise ValueError("physical RX axis does not match radio profile")
    shape = list(value.shape)
    shape[rx_axis] = profile.detector_rx_capacity
    result = np.zeros(shape, dtype=value.dtype)
    target = [slice(None)] * value.ndim
    target[rx_axis] = slice(0, profile.num_rx_antennas)
    result[tuple(target)] = value
    return result


def apply_fd8x8_channel(tx_port_iq: np.ndarray, profile: RadioProfile,
                        rank: int, gain: float, noise_std: float,
                        seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map port IQ to 8 TX antennas and propagate through a physical 8x8 H.

    Input is int16 `[slot,logical_port,sample,IQ]`. Returned TX/RX tensors are
    int16 physical antenna tensors. H_phys and H_effective are complex64.
    """
    validate_resolved_plan(profile, rank)
    ports = profile.tx_ports(rank)
    if (tx_port_iq.ndim != 4 or tx_port_iq.shape[1] != ports or
            tx_port_iq.shape[-1] != 2):
        raise ValueError("TX port IQ shape does not match radio profile")
    if tx_port_iq.dtype != np.int16:
        raise ValueError("TX port IQ must be int16")
    if not 0.0 <= noise_std <= 200.0:
        raise ValueError("channel noise std must be in [0,200]")
    port = (tx_port_iq[..., 0].astype(np.float32) +
            1j * tx_port_iq[..., 1].astype(np.float32))
    antenna = np.einsum("tp,spn->stn", tx_antenna_matrix(profile, rank),
                        port, optimize=True)
    antenna_iq_f32 = np.stack((antenna.real, antenna.imag), axis=-1)
    if np.any(np.abs(antenna_iq_f32) >= 32767):
        raise ValueError("8TX antenna mapping clips int16")
    tx_antenna_iq = np.rint(antenna_iq_f32).astype(np.int16)
    antenna_quantized = (tx_antenna_iq[..., 0].astype(np.float32) +
                         1j * tx_antenna_iq[..., 1].astype(np.float32))
    h_physical = physical_channel_matrix(profile, gain)
    signal = np.einsum("rt,stn->srn", h_physical, antenna_quantized,
                       optimize=True)
    composite = np.stack((signal.real, signal.imag), axis=-1).astype(np.float32)
    if noise_std:
        rng = np.random.default_rng(seed)
        composite += rng.normal(0.0, noise_std, size=composite.shape).astype(np.float32)
    if np.any(np.abs(composite) >= 32767):
        raise ValueError("8RX channel output clips int16")
    rx_physical_iq = np.rint(composite).astype(np.int16)
    return (tx_antenna_iq, rx_physical_iq, h_physical,
            effective_channel_matrix(profile, rank, gain))
