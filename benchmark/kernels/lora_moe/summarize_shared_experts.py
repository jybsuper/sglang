#!/usr/bin/env python3
"""Summarize shared-expert MoE-LoRA benchmark JSON as JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

VARIANT_ORDER = (
    "separate_serial",
    "separate_overlap",
    "fused_global",
    "fused_per_rank",
)


def _load(paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in paths:
        candidates = sorted(item.rglob("*.json")) if item.is_dir() else [item]
        for path in candidates:
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("scope") != "SH/local_M0" or data.get("timing") is None:
                continue
            data["artifact"] = str(path)
            records.append(data)
    return records


def _rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, list[dict[str, object]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        key = (
            str(record["environment"]["gpu"]),
            str(record["case"]["case_id"]),
            str(record["execution"]),
        )
        grouped[key][str(record["variant"])].append(record)

    rows: list[dict[str, object]] = []
    for (device, case_id, execution), variants in sorted(grouped.items()):
        if not all(name in variants for name in VARIANT_ORDER):
            continue
        baseline = median(
            float(item["timing"]["p50_us"]) for item in variants["separate_serial"]
        )
        p50 = {
            name: median(float(item["timing"]["p50_us"]) for item in variants[name])
            for name in VARIANT_ORDER
        }
        winner = min(p50, key=p50.get)
        case = variants[winner][0]["case"]
        rows.append(
            {
                "device": device,
                "case_id": case_id,
                "phase": case["phase"],
                "tokens": case["tokens"],
                "rank": case["rank"],
                "active_adapters": case["active_adapters"],
                "base_rows": case["include_base_rows"],
                "shared_experts": case["num_shared_experts"],
                "execution": execution,
                "repeats": {name: len(variants[name]) for name in VARIANT_ORDER},
                "p50_us": p50,
                "vs_separate_serial_pct": {
                    name: 100.0 * (value - baseline) / baseline
                    for name, value in p50.items()
                },
                "winner": winner,
                "winner_margin_vs_serial_pct": 100.0
                * (p50[winner] - baseline)
                / baseline,
                "per_rank_map_vs_global_pct": 100.0
                * (p50["fused_per_rank"] - p50["fused_global"])
                / p50["fused_global"],
            }
        )
    return rows


def _correctness(records: list[dict[str, object]]) -> dict[str, object]:
    checks = [record["correctness"] for record in records]
    return {
        "records": len(checks),
        "all_passed": bool(checks) and all(bool(item["passed"]) for item in checks),
        "minimum_cosine": min(float(item["cosine"]) for item in checks),
        "maximum_abs_error": max(float(item["max_abs"]) for item in checks),
        "maximum_mean_abs_error": max(float(item["mean_abs"]) for item in checks),
        "oracle": sorted({str(item["oracle"]) for item in checks}),
        "oracle_uses_physical_ids": any(
            bool(item["oracle_uses_physical_ids"]) for item in checks
        ),
        "shared_slots_receive_lora": any(
            bool(item["shared_slots_receive_lora"]) for item in checks
        ),
    }


def _markdown(rows: list[dict[str, object]], correctness: dict[str, object]) -> str:
    lines = [
        "# Shared-expert MoE-LoRA matrix summary",
        "",
        "Negative percentages mean faster than conventional serial shared experts. "
        "Values are medians of each artifact's p50; use the raw distributions and "
        "timeline traces before promoting a production policy.",
        "",
        f"Correctness: `{correctness['all_passed']}` across "
        f"`{correctness['records']}` records; minimum cosine "
        f"`{correctness['minimum_cosine']:.8f}`, maximum absolute error "
        f"`{correctness['maximum_abs_error']:.6g}`. The independent oracle used "
        "logical IDs only, and no shared slot received LoRA.",
        "",
        "| Device | Case | Exec | Runs | R | S | Serial us | Overlap us (%) | "
        "Fused-global us (%) | Fused-per-rank us (%) | Winner | Map vs global |",
        "|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|---:|",
    ]
    for row in rows:
        p50 = row["p50_us"]
        delta = row["vs_separate_serial_pct"]
        lines.append(
            f"| {row['device']} | {row['case_id']} | {row['execution']} | "
            f"{min(row['repeats'].values())} | "
            f"{row['rank']} | {row['shared_experts']} | "
            f"{p50['separate_serial']:.3f} | "
            f"{p50['separate_overlap']:.3f} ({delta['separate_overlap']:+.2f}%) | "
            f"{p50['fused_global']:.3f} ({delta['fused_global']:+.2f}%) | "
            f"{p50['fused_per_rank']:.3f} ({delta['fused_per_rank']:+.2f}%) | "
            f"{row['winner']} | {row['per_rank_map_vs_global_pct']:+.2f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    records = _load(args.inputs)
    if not records:
        raise SystemExit("no SH/local_M0 timing records found")
    rows = _rows(records)
    correctness = _correctness(records)
    payload = {
        "schema_version": 1,
        "records": len(records),
        "complete_comparisons": len(rows),
        "correctness": correctness,
        "rows": rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(_markdown(rows, correctness))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
