#!/usr/bin/env python3
"""Summarize counterbalanced architecture-auto versus PDL-off K0 artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def _load_directory(path: Path, order: str) -> list[dict[str, object]]:
    rows = []
    for artifact in sorted(path.glob("*.json")):
        report = json.loads(artifact.read_text())
        timing = report["results"][0]["timing"]
        case = report["case"]
        rows.append(
            {
                "artifact": str(artifact),
                "order": order,
                "device": report["environment"]["gpu"],
                "case_id": case["case_id"],
                "tokens": case["T"],
                "site": report["site"],
                "execution": report["environment"]["cli"]["execution"],
                "policy": report["pdl_policy"],
                "p20_us": timing["p20_us"],
                "p50_us": timing["p50_us"],
                "p80_us": timing["p80_us"],
                "max_abs_error": report["results"][0]["correctness"]["max_abs_error"],
            }
        )
    return rows


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["device"], row["tokens"], row["site"], row["execution"])].append(
            row
        )

    summary = []
    for (device, tokens, site, execution), group in sorted(groups.items()):
        by_order: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
        for row in group:
            by_order[str(row["order"])][str(row["policy"])] = row
        comparisons = []
        for order, policies in sorted(by_order.items()):
            if policies.keys() < {"auto", "off"}:
                continue
            auto = policies["auto"]
            off = policies["off"]
            comparisons.append(
                {
                    "order": order,
                    "auto_p50_us": auto["p50_us"],
                    "off_p50_us": off["p50_us"],
                    "auto_vs_off_pct": (
                        float(auto["p50_us"]) / float(off["p50_us"]) - 1.0
                    )
                    * 100.0,
                }
            )
        deltas = [float(item["auto_vs_off_pct"]) for item in comparisons]
        auto_rows = [row for row in group if row["policy"] == "auto"]
        off_rows = [row for row in group if row["policy"] == "off"]
        summary.append(
            {
                "device": device,
                "tokens": tokens,
                "site": site,
                "execution": execution,
                "orders": [item["order"] for item in comparisons],
                "auto_median_p50_us": median(float(row["p50_us"]) for row in auto_rows),
                "off_median_p50_us": median(float(row["p50_us"]) for row in off_rows),
                "auto_vs_off_median_pct": median(deltas),
                "auto_vs_off_min_pct": min(deltas),
                "auto_vs_off_max_pct": max(deltas),
                "order_stable": min(deltas) <= 0.0
                and max(deltas) <= 0.0
                or min(deltas) >= 0.0
                and max(deltas) >= 0.0,
                "max_correctness_error": max(
                    float(row["max_abs_error"]) for row in group
                ),
                "comparisons": comparisons,
            }
        )
    return summary


def _markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# PDL control matrix summary",
        "",
        "Negative percentages mean architecture-auto PDL is faster than forced-off. "
        "Each row combines forward and reverse process order.",
        "",
        "| GPU | T | Site | Execution | Auto us | Off us | Auto vs off median [min, max] | Stable sign |",
        "|:---|---:|:---|:---|---:|---:|:---|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {device} | {tokens} | {site} | {execution} | "
            "{auto_median_p50_us:.3f} | {off_median_p50_us:.3f} | "
            "{auto_vs_off_median_pct:+.2f}% [{auto_vs_off_min_pct:+.2f}, "
            "{auto_vs_off_max_pct:+.2f}] | {order_stable} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", action="append", type=Path, required=True)
    parser.add_argument("--reverse", action="append", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.forward:
        rows.extend(_load_directory(path, "forward"))
    for path in args.reverse:
        rows.extend(_load_directory(path, "reverse"))
    summary = _summarize(rows)
    payload = {"schema_version": 1, "rows": summary}
    if args.json_output:
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.write_text(_markdown(summary))
    if not args.json_output and not args.markdown_output:
        print(_markdown(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
