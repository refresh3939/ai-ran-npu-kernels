#!/usr/bin/env python3
"""Sweep common full-digital MIMO shapes through planning and target dispatch."""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

from mimo_profile_compiler import compile_profile, evaluate_capabilities


TX_COUNTS = (1, 2, 4, 8, 16)
RX_COUNTS = (1, 2, 4, 8, 16, 32, 64)
RANKS = (1, 2, 3, 4)


def candidate(base: dict, tx_antennas: int, rx_antennas: int, rank: int) -> dict:
    value = copy.deepcopy(base)
    value["profile_name"] = f"coverage_fd{tx_antennas}x{rx_antennas}_rank{rank}"
    value["topology"] = {
        "architecture": "full_digital",
        "tx_antennas": tx_antennas,
        "tx_rf_chains": tx_antennas,
        "rx_antennas": rx_antennas,
        "rx_rf_chains": rx_antennas,
    }
    value["spatial"]["supported_ranks"] = [rank]
    if rank == 3:
        value["spatial"]["port_policy"] = {
            "mode": "explicit", "ports_by_rank": {"3": 4}}
        value["spatial"]["precoding"] = {
            "default_mode": "bypass",
            "by_rank": {"3": {"mode": "codebook", "tpmi": 6,
                                "prg_size_rb": 4}},
        }
    else:
        value["spatial"]["port_policy"] = {"mode": "identity"}
        value["spatial"]["precoding"] = {"default_mode": "bypass"}
    value["dmrs"]["ports_by_rank"] = {
        str(rank): ([1000, 1002] if rank == 2 else
                    list(range(1000, 1000 + rank)))}
    value["channel"]["gain_by_rank"] = {str(rank): 0.32}
    return value


def build_matrix(profile_path: Path, capabilities: Path) -> list[dict]:
    base = json.loads(profile_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        for tx_antennas in TX_COUNTS:
            for rx_antennas in RX_COUNTS:
              for rank in RANKS:
                if rank > min(tx_antennas, rx_antennas):
                    continue
                path = scratch / f"fd{tx_antennas}x{rx_antennas}_rank{rank}.json"
                path.write_text(json.dumps(candidate(base, tx_antennas,
                                                     rx_antennas, rank)),
                                encoding="utf-8")
                authored = compile_profile(path, rank)
                evaluated = evaluate_capabilities(authored, capabilities).value
                receiver = evaluated["receiver"]
                rows.append({
                    "tx": tx_antennas,
                    "rx": rx_antennas,
                    "rank": rank,
                    "requested_bucket": [
                        receiver.get("requested_rx_bucket", receiver["rx_bucket"]),
                        receiver.get("requested_layer_bucket", receiver["layer_bucket"]),
                    ],
                    "execution_bucket": [receiver["rx_bucket"],
                                         receiver["layer_bucket"]],
                    "variant": evaluated.get("dispatch", {}).get("name"),
                    "eligibility": evaluated["validation"]["execution_eligibility"],
                    "blockers": evaluated["validation"]["blocking_operators"],
                })
    return rows


def markdown(rows: list[dict]) -> str:
    lines = [
        "| TX×RX | Rank | requested bucket | execution bucket | variant | result |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        requested = "×".join(map(str, row["requested_bucket"]))
        execution = "×".join(map(str, row["execution_bucket"]))
        variant = row["variant"] or "—"
        if row["eligibility"] == "eligible":
            result = "eligible"
        else:
            result = "blocked: " + ", ".join(row["blockers"])
        lines.append(
            f"| {row['tx']}×{row['rx']} | {row['rank']} | {requested} | "
            f"{execution} | {variant} | {result} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    common = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path,
                        default=common / "profiles/fd8x8_rank1_4.json")
    parser.add_argument("--capabilities", type=Path,
                        default=common / "mimo_operator_capabilities.json")
    parser.add_argument("--format", choices=("json", "markdown"),
                        default="markdown")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = build_matrix(args.profile, args.capabilities)
    if args.check:
        eligible = {(row["tx"], row["rx"], row["rank"])
                    for row in rows if row["eligibility"] == "eligible"}
        expected = {(row["tx"], row["rx"], row["rank"]) for row in rows}
        if eligible != expected:
            raise SystemExit(
                f"coverage regression: eligible={sorted(eligible)}, expected={sorted(expected)}")
        print("[PASS] installed target covers common 1..16TX x 1..64RX, PUSCH Rank1-4")
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(markdown(rows), end="")


if __name__ == "__main__":
    main()
