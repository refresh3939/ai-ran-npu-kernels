#!/usr/bin/env python3
"""Read and fail-closed validate the build variant in a resolved MIMO plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = (
    "name", "ce_nr", "ce_nl", "ce_retained_rank", "ce_block_dim",
    "ce_cube_time_fused", "detector_rx_capacity", "detector_layer_capacity",
    "detector_bri_block",
)


def resolve(path: Path, rank: int) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "airan.mimo.resolved_plan.v1":
        raise ValueError("unsupported or missing resolved-plan schema")
    if value.get("validation", {}).get("execution_eligibility") != "eligible":
        raise ValueError("resolved plan is not eligible for execution")
    if value.get("active", {}).get("rank") != rank:
        raise ValueError("resolved-plan Rank does not match requested Rank")
    receiver = value.get("receiver", {})
    dispatch = value.get("dispatch", {})
    runtime = dispatch.get("runtime", {})
    build = dispatch.get("build", {})
    result: dict[str, object] = {
        "name": dispatch.get("name"),
        "ce_nr": build.get("ce_nr"),
        "ce_nl": rank,
        "ce_retained_rank": build.get("ce_retained_rank"),
        "ce_block_dim": build.get("ce_block_dim"),
        "ce_cube_time_fused": build.get("ce_cube_time_fused"),
        "detector_rx_capacity": build.get("detector_rx_capacity"),
        "detector_layer_capacity": build.get("detector_layer_capacity"),
        "detector_bri_block": build.get("detector_bri_block"),
    }
    if not isinstance(result["name"], str) or not result["name"]:
        raise ValueError("resolved plan has no selected execution variant")
    for name in FIELDS[1:5] + FIELDS[6:]:
        if isinstance(result[name], bool) or not isinstance(result[name], int):
            raise ValueError(f"dispatch field {name} must be an integer")
    if not isinstance(result["ce_cube_time_fused"], bool):
        raise ValueError("dispatch field ce_cube_time_fused must be boolean")
    if build.get("ce_nl_source") != "active.rank":
        raise ValueError("unsupported CE layer-selection source")
    if runtime.get("rx_bucket") != receiver.get("rx_bucket"):
        raise ValueError("dispatch/runtime RX bucket mismatch")
    if runtime.get("layer_bucket") != receiver.get("layer_bucket"):
        raise ValueError("dispatch/runtime layer bucket mismatch")
    if result["ce_nr"] != runtime.get("rx_bucket"):
        raise ValueError("CE build NR does not match runtime RX bucket")
    if result["detector_rx_capacity"] != runtime.get("rx_bucket"):
        raise ValueError("detector RX capacity does not match runtime RX bucket")
    if result["detector_layer_capacity"] != runtime.get("layer_bucket"):
        raise ValueError("detector layer capacity does not match runtime layer bucket")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    args = parser.parse_args()
    result = resolve(args.plan, args.rank)
    if args.format == "json":
        print(json.dumps(result, sort_keys=True))
    else:
        for name in FIELDS:
            value = result[name]
            if isinstance(value, bool):
                value = "ON" if value else "OFF"
            print(value)


if __name__ == "__main__":
    main()
