#!/usr/bin/env python3
"""Summarize the rank-8/16 MoE-LoRA graduation guardrail.

The input root contains one directory per device.  Each raw report is produced
by ``bench_algorithm_families.py`` and must include both measurement orders.
This script deliberately keeps logical rank and the implementation's masked
physical tile separate: rank 8 remains the reported semantic rank even when a
Triton tensor-core dot uses a 16-wide compile-time tile.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

DEVICES = {
    "h200": "rank8_16_guardrail_v1_h200",
    "gb300": "rank8_16_guardrail_v1_gb300",
}
EXPECTED_TOKENS = (1, 32, 256, 2048)
EXPECTED_RANKS = (8, 16)
SITES = ("gate_a", "gate_consumer", "down_a", "down_finalize")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _raw_files(device_root: Path) -> list[Path]:
    return sorted(
        path
        for path in device_root.glob("*.json")
        if path.name != "smoke.json" and not path.name.startswith("smoke_")
    )


def _key(run: dict, device: str) -> tuple:
    case = run["case"]
    return (
        device,
        run["scope"],
        case["adapter_mode"],
        case["tokens"],
        case["rank"],
        run["site"],
        run["accumulation"],
        run["execution"],
        run["cache_state"],
        run["family"],
    )


def _correctness_metrics(value: dict) -> tuple[float, float]:
    """Return worst max-error and error/signal across scalar or named outputs."""
    if "max_abs_error" in value:
        return float(value["max_abs_error"]), float(value["error_over_signal"])
    leaves = [
        _correctness_metrics(child)
        for child in value.values()
        if isinstance(child, dict)
    ]
    if not leaves:
        raise ValueError(f"unrecognized correctness record: {value}")
    return max(item[0] for item in leaves), max(item[1] for item in leaves)


def _aggregate(root: Path) -> tuple[list[dict], list[dict], dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    statuses: Counter[str] = Counter()
    disqualifications: Counter[str] = Counter()
    sources: list[str] = []
    correctness_rows = 0
    graph_oracle_rows = 0

    for device, dirname in DEVICES.items():
        device_root = root / dirname
        files = _raw_files(device_root)
        if not files:
            raise ValueError(f"no raw reports under {device_root}")
        for path in files:
            report = _load(path)
            order = report.get("measurement_order")
            if order not in ("forward", "reverse"):
                raise ValueError(f"{path}: missing measurement order")
            sources.append(str(path.relative_to(root)))
            for row in report.get("disqualifications", []):
                disqualifications[row["reason"]] += 1
            for run in report["runs"]:
                statuses[run["status"]] += 1
                if run["status"] != "ok":
                    continue
                case = run["case"]
                if case["tokens"] not in EXPECTED_TOKENS:
                    raise ValueError(f"{path}: unexpected token count {case['tokens']}")
                if case["rank"] not in EXPECTED_RANKS:
                    raise ValueError(f"{path}: unexpected rank {case['rank']}")
                if run["site"] not in SITES:
                    raise ValueError(f"{path}: unexpected site {run['site']}")
                if run.get("correctness") is None:
                    raise ValueError(f"{path}: successful row lacks correctness")
                correctness_rows += 1
                if run["execution"] == "cuda_graph":
                    graph = run.get("graph_correctness")
                    if not graph or not graph.get("replay_completed"):
                        raise ValueError(f"{path}: graph row lacks replay oracle")
                    graph_oracle_rows += 1
                grouped[_key(run, device)].append(
                    {
                        "order": order,
                        "p20_us": run["timing"]["p20_us"],
                        "p50_us": run["timing"]["p50_us"],
                        "p80_us": run["timing"]["p80_us"],
                        "correctness": run["correctness"],
                    }
                )

    family_rows: list[dict] = []
    for key, samples in grouped.items():
        orders = sorted({sample["order"] for sample in samples})
        if orders != ["forward", "reverse"]:
            raise ValueError(f"comparison {key} lacks counterbalanced orders: {orders}")
        errors = [_correctness_metrics(x["correctness"]) for x in samples]
        family_rows.append(
            {
                "device": key[0],
                "scope": key[1],
                "adapter_mode": key[2],
                "tokens": key[3],
                "logical_rank": key[4],
                "physical_dot_rank": max(16, key[4]),
                "site": key[5],
                "accumulation": key[6],
                "execution": key[7],
                "cache_state": key[8],
                "family": key[9],
                "replicates": len(samples),
                "orders": orders,
                "median_p20_us": statistics.median(x["p20_us"] for x in samples),
                "median_p50_us": statistics.median(x["p50_us"] for x in samples),
                "median_p80_us": statistics.median(x["p80_us"] for x in samples),
                "worst_max_abs_error": max(item[0] for item in errors),
                "worst_error_over_signal": max(item[1] for item in errors),
            }
        )

    comparison_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in family_rows:
        comparison_groups[
            (
                row["device"],
                row["scope"],
                row["adapter_mode"],
                row["tokens"],
                row["logical_rank"],
                row["site"],
                row["accumulation"],
                row["execution"],
                row["cache_state"],
            )
        ].append(row)

    winners: list[dict] = []
    for key, rows in comparison_groups.items():
        ordered = sorted(rows, key=lambda row: row["median_p50_us"])
        winner = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        winners.append(
            {
                "device": key[0],
                "scope": key[1],
                "adapter_mode": key[2],
                "tokens": key[3],
                "logical_rank": key[4],
                "physical_dot_rank": max(16, key[4]),
                "site": key[5],
                "accumulation": key[6],
                "execution": key[7],
                "cache_state": key[8],
                "winner": winner["family"],
                "winner_p50_us": winner["median_p50_us"],
                "runner_up": runner_up["family"] if runner_up else None,
                "runner_up_slower_pct": (
                    (runner_up["median_p50_us"] / winner["median_p50_us"] - 1.0) * 100.0
                    if runner_up
                    else None
                ),
            }
        )

    metadata = {
        "sources": sorted(sources),
        "run_status": dict(statuses),
        "disqualifications": dict(disqualifications),
        "successful_rows_with_oracle": correctness_rows,
        "successful_graph_rows_with_replay_oracle": graph_oracle_rows,
        "family_rows": len(family_rows),
        "winner_rows": len(winners),
    }
    return family_rows, winners, metadata


def _canonical(winners: Iterable[dict]) -> list[dict]:
    return sorted(
        (
            row
            for row in winners
            if row["adapter_mode"] == "multi"
            and row["execution"] == "cuda_graph"
            and row["cache_state"] == "hot"
            and row["accumulation"] == "bf16"
        ),
        key=lambda row: (
            row["device"],
            row["scope"],
            row["site"],
            row["tokens"],
            row["logical_rank"],
        ),
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _readme(canonical: list[dict], metadata: dict) -> str:
    by_key = {
        (
            row["device"],
            row["scope"],
            row["site"],
            row["tokens"],
            row["logical_rank"],
        ): row
        for row in canonical
    }
    lines = [
        "# Rank-8/16 MoE-LoRA graduation guardrail",
        "",
        "Logical rank 8 initially failed three Triton tensor-core paths because their",
        "compile-time K tile was 8. The serving expand path, the aligned C2 consumer,",
        "and the benchmark segmented consumer now use a masked physical-16 dot tile",
        "while retaining logical-rank storage and zero-masked tail lanes.",
        "",
        "The retained matrix covers H200 and GB300, T={1,32,256,2048}, logical",
        "R={8,16}, K0/O0, eager/CUDA graph, hot/cold cache, all four sites,",
        "forward/reverse order, all-active rows, and mixed base/adapter rows. Every",
        "successful row has an independent numerical oracle; every graph row also",
        "has an independent replay oracle.",
        "",
        f"Successful timed rows: {metadata['run_status'].get('ok', 0)}; graph replay oracles: {metadata['successful_graph_rows_with_replay_oracle']}.",
        "",
        "## Canonical all-active CUDA-graph/hot BF16 winners",
        "",
        "Each cell is `family p50-us` after taking the median of forward and reverse",
        "orders. T=1 uses the all-active fixture; the mixed fixture intentionally makes",
        "its only row base-only and is retained as a separate semantic control.",
        "",
    ]
    for device in DEVICES:
        for scope in ("K0", "O0"):
            for site in SITES:
                lines.extend(
                    [
                        f"### {device.upper()} {scope} `{site}`",
                        "",
                        "| T | R8 (physical dot 16) | R16 |",
                        "|---:|---:|---:|",
                    ]
                )
                for tokens in EXPECTED_TOKENS:
                    cells = []
                    for rank in EXPECTED_RANKS:
                        row = by_key.get((device, scope, site, tokens, rank))
                        cells.append(
                            "n/a"
                            if row is None
                            else f"{row['winner']} {row['winner_p50_us']:.2f}"
                        )
                    lines.append(f"| {tokens} | {cells[0]} | {cells[1]} |")
                lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Rank 8 is now compile-legal and numerically covered without padding the stored factors.",
            "- Rank 8 and rank 16 share a physical tensor-core K floor, but their logical factor bytes and non-dot work remain distinct; the selector must not conflate the two ranks.",
            "- Winner changes with site, work size, scope, and device. These rows are planner/autotune evidence, not a universal family cutoff.",
            "- `raw_family_rows.json` retains every counterbalanced family result; `winners.csv` retains every exact comparison winner.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_hashes(root: Path) -> None:
    output = root / "SHA256SUMS"
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path != output
    )
    output.write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
            for path in paths
        )
        + "\n"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    root = _parse_args(argv).results_root.resolve()
    family_rows, winners, metadata = _aggregate(root)
    canonical = _canonical(winners)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "schema": "sgl_lora_rank8_16_guardrail_v1",
                "logical_to_physical_dot_rank": {"8": 16, "16": 16},
                "metadata": metadata,
                "canonical_winners": canonical,
            },
            indent=2,
        )
        + "\n"
    )
    (root / "raw_family_rows.json").write_text(json.dumps(family_rows, indent=2) + "\n")
    _write_csv(root / "winners.csv", winners)
    (root / "README.md").write_text(_readme(canonical, metadata))
    _write_hashes(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
