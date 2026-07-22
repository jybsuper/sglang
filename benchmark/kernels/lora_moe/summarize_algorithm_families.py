#!/usr/bin/env python3
"""Summarize durable MoE-LoRA algorithm-family benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

SITES = ("gate_a", "gate_consumer", "down_a", "down_finalize")
TOKENS = (1, 32, 256, 2048)
RANKS = (16, 32, 64, 128)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _core_files(device_root: Path, scope: str) -> list[Path]:
    if device_root.name == "h200":
        return sorted((device_root / "final").glob(f"{scope}_r*.json"))
    return sorted((device_root / "final").glob(f"{scope}*.json"))


def _aggregate_core(root: Path) -> tuple[list[dict], dict[str, object]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    source_files: set[str] = set()
    status = Counter()
    for device in ("h200", "gb300"):
        device_root = root / device
        for scope in ("k0", "o0"):
            for path in _core_files(device_root, scope):
                report = _load(path)
                order = report.get("measurement_order", "forward")
                source_files.add(str(path.relative_to(root)))
                for run in report["runs"]:
                    status[run["status"]] += 1
                    if run["status"] != "ok":
                        continue
                    case = run["case"]
                    key = (
                        device,
                        scope.upper(),
                        case["tokens"],
                        case["rank"],
                        run["site"],
                        run["accumulation"],
                        run["execution"],
                        run["cache_state"],
                        run["family"],
                    )
                    grouped[key].append(
                        {
                            "order": order,
                            "p20_us": run["timing"]["p20_us"],
                            "p50_us": run["timing"]["p50_us"],
                            "p80_us": run["timing"]["p80_us"],
                        }
                    )

    family_rows = []
    for key, samples in grouped.items():
        values = [sample["p50_us"] for sample in samples]
        family_rows.append(
            {
                "device": key[0],
                "scope": key[1],
                "tokens": key[2],
                "rank": key[3],
                "site": key[4],
                "accumulation": key[5],
                "execution": key[6],
                "cache_state": key[7],
                "family": key[8],
                "replicates": len(samples),
                "orders": sorted({sample["order"] for sample in samples}),
                "median_p50_us": statistics.median(values),
                "min_p50_us": min(values),
                "max_p50_us": max(values),
                "median_p20_us": statistics.median(
                    sample["p20_us"] for sample in samples
                ),
                "median_p80_us": statistics.median(
                    sample["p80_us"] for sample in samples
                ),
            }
        )

    comparison_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in family_rows:
        comparison_groups[
            (
                row["device"],
                row["scope"],
                row["tokens"],
                row["rank"],
                row["site"],
                row["accumulation"],
                row["execution"],
                row["cache_state"],
            )
        ].append(row)

    winners = []
    for key, rows in comparison_groups.items():
        ordered = sorted(rows, key=lambda row: row["median_p50_us"])
        winner = ordered[0]
        per_order_winners = {}
        for order in ("forward", "reverse"):
            order_values = []
            for row in rows:
                sample_values = [
                    sample["p50_us"]
                    for sample in grouped[
                        (
                            row["device"],
                            row["scope"],
                            row["tokens"],
                            row["rank"],
                            row["site"],
                            row["accumulation"],
                            row["execution"],
                            row["cache_state"],
                            row["family"],
                        )
                    ]
                    if sample["order"] == order
                ]
                if sample_values:
                    order_values.append(
                        (statistics.median(sample_values), row["family"])
                    )
            if order_values:
                per_order_winners[order] = min(order_values)[1]
        winners.append(
            {
                "device": key[0],
                "scope": key[1],
                "tokens": key[2],
                "rank": key[3],
                "site": key[4],
                "accumulation": key[5],
                "execution": key[6],
                "cache_state": key[7],
                "winner": winner["family"],
                "winner_median_p50_us": winner["median_p50_us"],
                "runner_up": ordered[1]["family"] if len(ordered) > 1 else None,
                "runner_up_median_p50_us": (
                    ordered[1]["median_p50_us"] if len(ordered) > 1 else None
                ),
                "runner_up_slower_pct": (
                    (ordered[1]["median_p50_us"] / winner["median_p50_us"] - 1.0)
                    * 100.0
                    if len(ordered) > 1
                    else None
                ),
                "per_order_winners": per_order_winners,
                "order_stable": len(set(per_order_winners.values())) <= 1,
            }
        )
    metadata = {
        "source_files": sorted(source_files),
        "run_status": dict(status),
        "family_rows": len(family_rows),
        "winner_rows": len(winners),
    }
    return sorted(winners, key=lambda row: tuple(map(str, row.values()))), metadata


def _route_anchor_summary(root: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for device in ("h200", "gb300"):
        rows = []
        disqualified = []
        for path in sorted((root / device / "anchors").glob("[ko]0_*.json")):
            report = _load(path)
            rows.extend(report.get("runs", []))
            disqualified.extend(report.get("disqualifications", []))
        reasons = Counter(row["reason"] for row in disqualified)
        cvs = [
            row["route_metrics"].get("cv_m_g")
            for row in disqualified
            if row.get("route_metrics", {}).get("cv_m_g") is not None
        ]
        result[device] = {
            "successful_runs": sum(row.get("status") == "ok" for row in rows),
            "disqualification_count": len(disqualified),
            "disqualification_reasons": dict(reasons),
            "disqualified_route_cv_m_g_min": min(cvs) if cvs else None,
            "disqualified_route_cv_m_g_max": max(cvs) if cvs else None,
            "bmm_policy": "admit only >=2 nonempty groups with exactly equal M_g",
        }
    return result


def _cutedsl_summary(root: Path) -> dict[str, object]:
    h200 = _load(root / "h200" / "cutedsl_capability.json")
    gb300 = _load(root / "gb300" / "anchors" / "cutedsl_indexed.json")
    rows = gb300["runs"]
    comparisons = []
    keys = {
        (
            row["case"]["tokens"],
            row["case"]["rank"],
            row["site"],
            row["execution"],
            row["cache_state"],
        )
        for row in rows
        if row["status"] == "ok"
    }
    for key in sorted(keys):
        matched = {
            row["family"]: row["timing"]["p50_us"]
            for row in rows
            if row["status"] == "ok"
            and (
                row["case"]["tokens"],
                row["case"]["rank"],
                row["site"],
                row["execution"],
                row["cache_state"],
            )
            == key
        }
        if {"cutedsl", "indexed"} <= matched.keys():
            comparisons.append(
                {
                    "tokens": key[0],
                    "rank": key[1],
                    "site": key[2],
                    "execution": key[3],
                    "cache_state": key[4],
                    "cutedsl_us": matched["cutedsl"],
                    "triton_indexed_us": matched["indexed"],
                    "cutedsl_slower_x": matched["cutedsl"] / matched["indexed"],
                }
            )
    return {
        "h200": {
            "status": "unavailable",
            "disqualification": h200["disqualifications"][0]["reason"],
        },
        "gb300": {
            "status": "compiled_correct_eager_and_graph",
            "implementation": "one_thread_per_output_scalar_capability_probe",
            "comparisons": comparisons,
            "min_slowdown_x": min(row["cutedsl_slower_x"] for row in comparisons),
            "max_slowdown_x": max(row["cutedsl_slower_x"] for row in comparisons),
        },
    }


def _accumulation_summary(root: Path) -> dict[str, object]:
    result = {}
    for device in ("h200", "gb300"):
        anchors = root / device / "anchors"
        gate = _load(anchors / "fp32_gate_consumer.json")
        down = _load(anchors / "fp32_down_finalize.json")
        one_shot = [
            run
            for run in down["runs"]
            if run["status"] == "ok"
            and run["family"] in ("one_shot_bf16", "one_shot_fp32")
            and run["cache_state"] == "hot"
        ]
        pairs = defaultdict(dict)
        for run in one_shot:
            pairs[(run["case"]["tokens"], run["case"]["rank"])][run["family"]] = run[
                "timing"
            ]["p50_us"]
        one_shot_rows = []
        for key, values in sorted(pairs.items()):
            if len(values) == 2:
                one_shot_rows.append(
                    {
                        "tokens": key[0],
                        "rank": key[1],
                        **values,
                        "fp32_vs_bf16_pct": (
                            values["one_shot_fp32"] / values["one_shot_bf16"] - 1.0
                        )
                        * 100.0,
                    }
                )
        hot_by_shape: dict[tuple, dict[str, float]] = defaultdict(dict)
        for run in down["runs"]:
            if (
                run["status"] == "ok"
                and run["execution"] == "cuda_graph"
                and run["cache_state"] == "hot"
            ):
                hot_by_shape[(run["case"]["tokens"], run["case"]["rank"])][
                    run["family"]
                ] = run["timing"]["p50_us"]
        one_shot_ratios = []
        for values in hot_by_shape.values():
            non_one_shot = [
                value
                for family, value in values.items()
                if not family.startswith("one_shot")
            ]
            for family in ("one_shot_bf16", "one_shot_fp32"):
                if family in values and non_one_shot:
                    one_shot_ratios.append(values[family] / min(non_one_shot))
        result[device] = {
            "gate_fp32_successful_runs": sum(
                run.get("status") == "ok" for run in gate["runs"]
            ),
            "gate_fp32_disqualifications": gate.get("disqualifications", []),
            "down_fp32_successful_runs": sum(
                run.get("status") == "ok" for run in down["runs"]
            ),
            "one_shot_accumulation": one_shot_rows,
            "one_shot_vs_best_non_one_shot": {
                "comparison_count": len(one_shot_ratios),
                "min_slowdown_x": min(one_shot_ratios),
                "median_slowdown_x": statistics.median(one_shot_ratios),
                "max_slowdown_x": max(one_shot_ratios),
            },
        }
    return result


def _tuning_config(category: str, family: str, tuning: dict) -> str:
    if category in ("indexed", "aligned", "segmented"):
        return (
            f"BN{tuning[f'{category}_bn']}/BK{tuning[f'{category}_bk']}"
            f"/W{tuning[f'{category}_warps']}"
        )
    if category == "consumer":
        bn_key = {
            "indexed": "pair_consumer_bn",
            "aligned": "aligned_consumer_bn",
            "segmented": "segmented_consumer_bn",
        }[family]
        return f"BN{tuning[bn_key]}/W{tuning['consumer_warps']}"
    if category == "finalize":
        bh_key = "direct_finalize_bh" if family == "indexed" else "reduce_finalize_bh"
        return f"BH{tuning[bh_key]}"
    raise ValueError(f"unknown tuning category: {category}")


def _tuning_summary(root: Path) -> dict[str, object]:
    """Summarize the independent tile sweep used before family comparison."""

    result = {}
    categories = ("indexed", "aligned", "segmented", "consumer", "finalize")
    for device in ("h200", "gb300"):
        device_root = root / device
        selected_path = (
            device_root / "final" / "k0_r16.json"
            if device == "h200"
            else device_root / "final" / "k0.json"
        )
        selected = _load(selected_path)["kernel_tuning"]
        device_result: dict[str, object] = {
            "selected_core_config": selected,
            "categories": {},
        }
        for category in categories:
            candidates: set[str] = set()
            grouped: dict[tuple, list[dict]] = defaultdict(list)
            source_files = sorted((device_root / "tuning").glob(f"{category}_*.json"))
            for path in source_files:
                report = _load(path)
                tuning = report["kernel_tuning"]
                for run in report["runs"]:
                    if run["status"] != "ok":
                        continue
                    config = _tuning_config(category, run["family"], tuning)
                    candidates.add(config)
                    key = (
                        run["site"],
                        run["family"],
                        run["case"]["tokens"],
                        run["case"]["rank"],
                        run["execution"],
                        run["cache_state"],
                    )
                    grouped[key].append(
                        {
                            "config": config,
                            "p50_us": run["timing"]["p50_us"],
                            "source": str(path.relative_to(root)),
                        }
                    )

            anchors = []
            win_counts = Counter()
            for key, rows in sorted(grouped.items()):
                ordered = sorted(rows, key=lambda row: row["p50_us"])
                winner = ordered[0]
                win_counts[winner["config"]] += 1
                anchors.append(
                    {
                        "site": key[0],
                        "family": key[1],
                        "tokens": key[2],
                        "rank": key[3],
                        "execution": key[4],
                        "cache_state": key[5],
                        "winner_config": winner["config"],
                        "winner_p50_us": winner["p50_us"],
                        "runner_up_config": (
                            ordered[1]["config"] if len(ordered) > 1 else None
                        ),
                        "runner_up_slower_pct": (
                            (ordered[1]["p50_us"] / winner["p50_us"] - 1.0) * 100.0
                            if len(ordered) > 1
                            else None
                        ),
                    }
                )
            device_result["categories"][category] = {
                "candidate_configs": sorted(candidates),
                "anchor_win_counts": dict(sorted(win_counts.items())),
                "anchors": anchors,
                "source_files": [str(path.relative_to(root)) for path in source_files],
            }
        result[device] = device_result
    return result


def _stability_summary(winners: list[dict]) -> dict[str, object]:
    stable = [row for row in winners if row["order_stable"]]
    unstable = [row for row in winners if not row["order_stable"]]
    canonical = [
        row
        for row in winners
        if row["execution"] == "cuda_graph"
        and row["cache_state"] == "hot"
        and row["accumulation"] == "bf16"
    ]
    canonical_unstable = [row for row in canonical if not row["order_stable"]]
    margins = [
        row["runner_up_slower_pct"]
        for row in winners
        if row["runner_up_slower_pct"] is not None
    ]
    unstable_margins = [
        row["runner_up_slower_pct"]
        for row in unstable
        if row["runner_up_slower_pct"] is not None
    ]
    return {
        "stable_count": len(stable),
        "unstable_count": len(unstable),
        "canonical_count": len(canonical),
        "canonical_unstable_count": len(canonical_unstable),
        "all_runner_up_margin_median_pct": statistics.median(margins),
        "unstable_runner_up_margin_median_pct": statistics.median(unstable_margins),
        "unstable_runner_up_margin_max_pct": max(unstable_margins),
        "cells_with_runner_up_within_1_pct": sum(margin <= 1.0 for margin in margins),
        "cells_with_runner_up_within_3_pct": sum(margin <= 3.0 for margin in margins),
        "cells_with_runner_up_within_5_pct": sum(margin <= 5.0 for margin in margins),
        "unstable_breakdown": dict(
            sorted(
                Counter(
                    f"{row['device']}/{row['scope']}/{row['site']}/"
                    f"{row['execution']}/{row['cache_state']}"
                    for row in unstable
                ).items()
            )
        ),
    }


def _correction_summary(root: Path) -> dict[str, object]:
    return _load(root / "correction_manifest.json")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            serialized = {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _canonical_map(
    winners: list[dict], device: str, scope: str, site: str
) -> list[str]:
    lines = [
        f"#### {device.upper()} {scope} — `{site}`",
        "",
        "| T \\ R | 16 | 32 | 64 | 128 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for tokens in TOKENS:
        cells = []
        for rank in RANKS:
            matched = [
                row
                for row in winners
                if row["device"] == device
                and row["scope"] == scope
                and row["site"] == site
                and row["tokens"] == tokens
                and row["rank"] == rank
                and row["execution"] == "cuda_graph"
                and row["cache_state"] == "hot"
                and row["accumulation"] == "bf16"
            ]
            if not matched:
                cells.append("—")
            else:
                row = matched[0]
                cells.append(f"{row['winner']} {row['winner_median_p50_us']:.2f} us")
        lines.append(f"| {tokens} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _readme(
    winners: list[dict],
    metadata: dict[str, object],
    route: dict[str, object],
    cutedsl: dict[str, object],
    accumulation: dict[str, object],
    tuning: dict[str, object],
    stability_summary: dict[str, object],
    correction: dict[str, object],
) -> str:
    canonical = [
        row
        for row in winners
        if row["execution"] == "cuda_graph"
        and row["cache_state"] == "hot"
        and row["accumulation"] == "bf16"
    ]
    stability = sum(row["order_stable"] for row in winners)
    lines = [
        "# SGL LoRA MoE algorithm-family benchmark",
        "",
        "This evidence is benchmark-only; no serving dispatch imports these candidates.",
        "All final winners use the median p50 of forward and reverse measurement orders.",
        "`K0` prebuilds routing metadata; `O0` charges route/segment construction,",
        "padding/packing, clears/casts, and every consumer.",
        "",
        "## Coverage and validity",
        "",
        f"- Final successful run records: {metadata['run_status'].get('ok', 0)}.",
        f"- Counterbalanced winner cells: {len(winners)}; order-stable: {stability}.",
        "- Canonical graph/hot cells with order-dependent winners:"
        f" {stability_summary['canonical_unstable_count']}/"
        f"{stability_summary['canonical_count']}; treat these as autotune",
        "  boundaries, not hard dispatch thresholds.",
        "- Devices: H200 (SM90) and GB300 (SM103).",
        "- Core dimensions: T={1,32,256,2048}, R={16,32,64,128}, four sites,",
        "  eager/CUDA graph, hot/cold cache, K0/O0.",
        "- Adapter anchors: four active adapters plus separate mixed base/LoRA rows;",
        "  regular equal-M_g, true IID, and Zipf-skewed routes.",
        "- Every eager candidate and graph replay is checked against an independent",
        "  chunked FP32 PyTorch oracle.",
        "",
        "## Main decisions",
        "",
        "- With prebuilt routing (K0), padded aligned/grouped is the default winner",
        "  for gate-A and all prefill-scale sites. Raw indexed remains best for many",
        "  T<=32 down-A/finalize cells; segmented is competitive for the fused gate",
        "  consumer at decode sizes.",
        "- When route building is charged (O0), raw indexed wins nearly every T<=32",
        "  cell and many T=256 cells. Aligned/grouped overtakes it at T=2048.",
        "- BMM is useful only on recorded exactly-equal M_g routes. It wins selected",
        "  large gate-A cells when routing is prebuilt, but no canonical O0 fused",
        "  gate-consumer or down-finalize cell after route construction is charged.",
        "  IID/skewed mixed routes are rejected rather than padded into a misleading",
        "  BMM result.",
        "- The one-shot down A+B probe is decisively rejected: each H tile recomputes",
        "  the A reduction. Its median slowdown versus the best decomposed arm is",
        f"  {accumulation['h200']['one_shot_vs_best_non_one_shot']['median_slowdown_x']:.2f}x"
        " on H200 and"
        f" {accumulation['gb300']['one_shot_vs_best_non_one_shot']['median_slowdown_x']:.2f}x"
        " on GB300.",
        "  Shared-A factorization requires a different cooperative schedule.",
        "- Current aligned C2 routing omits base-only sentinel blocks. The fair mixed",
        "  benchmark charges an explicit base-SwiGLU prepass; serving promotion must",
        "  implement equivalent semantics.",
        "- CuTe DSL 4.6 compiles and graph-replays the GB300 raw-indexed probe, but",
        f"  this scalar capability version is {cutedsl['gb300']['min_slowdown_x']:.2f}–"
        f"{cutedsl['gb300']['max_slowdown_x']:.2f}x slower than tuned Triton.",
        "  The H200 image cannot import `cutlass`, despite installed package",
        "  metadata.",
        "",
        "## How candidate implementations were kept honest",
        "",
        "- Indexed, aligned, segmented, fused-consumer, and finalizer tiles were",
        "  swept independently before cross-family comparison. The exact candidate",
        "  sets and per-anchor winners are in `summary.json` under `tuning`.",
        "- Each sweep used identical weights, routes, output contracts, CUDA-graph",
        "  replay, cache state, and independent oracle. Slower families therefore",
        "  were not compared using a single arbitrary tile inherited from a winner.",
        "- Forward/reverse counterbalancing exposes close boundary cells: the median",
        "  runner-up margin in order-unstable cells is"
        f" {stability_summary['unstable_runner_up_margin_median_pct']:.2f}%.",
        "  Keep those cells autotuned rather than encoding exact table transitions.",
        "- Selected core settings are included below; they are robust defaults, not",
        "  a claim that one tile is optimal for every site and shape.",
        "",
        "```json",
        json.dumps(
            {
                device: details["selected_core_config"]
                for device, details in tuning.items()
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## O0 BMM cost-accounting correction",
        "",
        "- The initial O0 BMM gate-consumer and down-finalize arms reused a prebuilt",
        "  order. The corrected closures rebuild virtual-group ids and stable-sort",
        "  the route inside every timed invocation, matching the O0 contract.",
        f"- {correction['corrected_run_records']} affected records were rerun across",
        "  both devices, both measurement orders, eager/graph, and hot/cold cache;",
        "  every corrected run and independent graph-oracle check passed.",
        f"- {correction['winner_change_count']} of 1,024 winner cells changed, including"
        f" {correction['canonical_winner_change_count']} canonical graph/hot cells.",
        f"  All {correction['canonical_winner_change_count']} canonical changes remove"
        " previously understated BMM winners from",
        "  O0 down-finalize. No corrected canonical O0 fused-consumer/finalize cell",
        "  selects BMM.",
        "- Exact before/after transitions and raw reruns are retained in",
        "  `correction_manifest.json` and each device's `corrections/` directory.",
        "",
        "## Canonical CUDA-graph/hot winner maps",
        "",
        "Each cell is `family median-p50` across forward/reverse order.",
        "",
    ]
    for device in ("h200", "gb300"):
        for scope in ("K0", "O0"):
            for site in SITES:
                lines.extend(_canonical_map(canonical, device, scope, site))
    lines.extend(
        [
            "## Route and accumulation evidence",
            "",
            "```json",
            json.dumps(
                {"route": route, "accumulation": accumulation}, indent=2, sort_keys=True
            ),
            "```",
            "",
            "## Files",
            "",
            "- `winners.csv`: every exact winner across device/scope/T/R/site/",
            "  eager-or-graph/hot-or-cold.",
            "- `summary.json`: machine-readable conclusions, route admission, CuTe,",
            "  tile tuning, stability, and FP32/BF16 evidence.",
            "- `correction_manifest.json`: exact O0 BMM cost-accounting rerun and",
            "  before/after winner transitions.",
            "- `h200/` and `gb300/`: raw final, reverse-order, tuning, correctness,",
            "  route, accumulation, correction, and CuTe artifacts.",
            "- `SHA256SUMS`: hashes for every evidence file.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_hashes(root: Path, output: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path != output
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root)}")
    output.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.results_root.resolve()
    winners, metadata = _aggregate_core(root)
    route = _route_anchor_summary(root)
    cutedsl = _cutedsl_summary(root)
    accumulation = _accumulation_summary(root)
    tuning = _tuning_summary(root)
    stability_summary = _stability_summary(winners)
    correction = _correction_summary(root)
    summary = {
        "schema_version": 1,
        "metadata": metadata,
        "route": route,
        "cutedsl": cutedsl,
        "accumulation": accumulation,
        "tuning": tuning,
        "stability": stability_summary,
        "correction": correction,
        "winners": winners,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(root / "winners.csv", winners)
    (root / "README.md").write_text(
        _readme(
            winners,
            metadata,
            route,
            cutedsl,
            accumulation,
            tuning,
            stability_summary,
            correction,
        )
    )
    _write_hashes(root, root / "SHA256SUMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
