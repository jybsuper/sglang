#!/usr/bin/env python3
"""Summarize one or more C3 matrix JSON directories as JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def _status(values: list[float]) -> str:
    if not values or min(values) <= 0.0 <= max(values):
        return "inconclusive within dispersion"
    effect = abs(median(values))
    dispersion_floor = max(1.0, max(values) - min(values))
    if effect <= dispersion_floor:
        return "inconclusive: effect <= dispersion floor"
    if max(values) < 0.0:
        return "C3 faster (all matched repeats)"
    if min(values) > 0.0:
        return "C3 slower (all matched repeats)"
    raise AssertionError("unreachable comparison status")


def _summarize_file(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    case = data["case"]
    rows = []
    comparisons = data.get("counterbalanced_comparisons", [])
    for execution in ("eager", "cuda_graph"):
        matched = [row for row in comparisons if row["execution"] == execution]
        if not matched:
            continue
        timings = {
            variant: median(
                row["p50_us"][variant]
                for row in matched
                if variant in row["p50_us"]
            )
            for variant in ("C0", "C1", "C2P", "C2F", "C3")
            if any(variant in row["p50_us"] for row in matched)
        }
        for baseline in ("C0", "C1", "C2P", "C2F"):
            values = [
                row["c3_vs_pct"][baseline]
                for row in matched
                if baseline in row["c3_vs_pct"]
            ]
            if not values:
                continue
            rows.append(
                {
                    "artifact": path.name,
                    "device": data["environment"]["gpu"],
                    "case_id": case["case_id"],
                    "tokens": case["T"],
                    "rank": case["R"],
                    "active_adapters": case["L_active"],
                    "has_base_rows": bool(case["B_base"]),
                    "execution": execution,
                    "baseline": baseline,
                    "baseline_median_p50_us": timings[baseline],
                    "c3_median_p50_us": timings["C3"],
                    "c3_vs_baseline_median_pct": median(values),
                    "c3_vs_baseline_min_pct": min(values),
                    "c3_vs_baseline_max_pct": max(values),
                    "status": _status(values),
                    "matched_repeats": len(values),
                }
            )
    return rows


def _markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# C3 overlap matrix summary",
        "",
        "Negative percentages mean C3 is faster. A winner is named only when all "
        "matched counterbalanced repeats agree on the sign and the median effect "
        "exceeds both 1% and the matched-repeat range.",
        "",
        "| Device | T | R | Base rows | Execution | Baseline | Baseline us | "
        "C3 us | C3 vs baseline median [min, max] | Result |",
        "|:---|---:|---:|:---:|:---|:---|---:|---:|:---|:---|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            item["device"],
            item["tokens"],
            item["rank"],
            item["has_base_rows"],
            item["execution"],
            item["baseline"],
        ),
    ):
        lines.append(
            "| {device} | {tokens} | {rank} | {has_base_rows} | {execution} | {baseline} | "
            "{baseline_median_p50_us:.3f} | {c3_median_p50_us:.3f} | "
            "{c3_vs_baseline_median_pct:+.2f}% "
            "[{c3_vs_baseline_min_pct:+.2f}, {c3_vs_baseline_max_pct:+.2f}] | "
            "{status} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    paths = []
    for item in args.inputs:
        paths.extend(sorted(item.glob("*.json")) if item.is_dir() else [item])
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        if data.get("scope") == "M0_local_bf16_c3_gate_a_overlap_experimental":
            rows.extend(_summarize_file(path))
    payload = {"schema_version": 1, "rows": rows}
    if args.json_output:
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.write_text(_markdown(rows))
    if not args.json_output and not args.markdown_output:
        print(_markdown(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
