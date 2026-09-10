#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))

from mimo_config_compiler import compile_config
from mimo_profile_compiler import ProfileError
from radio_profile_generator import build_radio_profile


def main() -> None:
    radio = COMMON / "radio_profiles/fd16x64.json"
    grant = COMMON / "pusch_grants/full_band.json"
    capabilities = COMMON / "mimo_operator_capabilities.json"
    for rank, ports in ((1, 1), (2, 2), (3, 4), (4, 4)):
        plan = compile_config(radio, grant, rank, capabilities)
        value = plan.value
        assert value["active"] == {"rank": rank, "logical_tx_ports": ports}
        assert value["topology"]["tx_antennas"] == 16
        assert value["topology"]["rx_antennas"] == 64
        assert value["scheduler"]["rnti"] == 12345
        assert value["scheduler"]["data_scrambling_id"] == 321
        assert value["validation"]["execution_eligibility"] == "eligible"
        binary = plan.to_runtime_config()
        assert len(binary) == 192
        assert struct.unpack_from("<2H", binary) == (2, 192)
        assert struct.unpack_from("<10H", binary, 152) == (
            0, 321, 321, 12345, 0, 14, 0, 0, 0, 0)

    raw = json.loads(grant.read_text())
    raw["rank"] = 5
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad_grant.json"
        path.write_text(json.dumps(raw))
        try:
            compile_config(radio, path)
        except ProfileError:
            pass
        else:
            raise AssertionError("Rank5 grant was accepted")
        try:
            compile_config(radio, path, 4)
        except ProfileError:
            pass
        else:
            raise AssertionError("Rank override masked an invalid grant Rank")
    with tempfile.TemporaryDirectory() as directory:
        generated = Path(directory) / "fd12x48.json"
        generated.write_text(json.dumps(build_radio_profile(12, 48)))
        plan = compile_config(generated, grant, 4, capabilities).value
        assert plan["topology"]["tx_antennas"] == 12
        assert plan["topology"]["rx_antennas"] == 48
        assert plan["validation"]["execution_eligibility"] == "eligible"
        bad_ports = json.loads(grant.read_text())
        bad_ports["dmrs"]["ports"] = [1000, 1001, 1002, 1004]
        bad_grant = Path(directory) / "bad_ports.json"
        bad_grant.write_text(json.dumps(bad_ports))
        try:
            compile_config(generated, bad_grant, 4)
        except ProfileError:
            pass
        else:
            raise AssertionError("unsupported explicit DMRS port set was accepted")
    print("[PASS] split radio profile + dynamic PUSCH grant compiler Rank1-4 and ABI v2")


if __name__ == "__main__":
    main()
