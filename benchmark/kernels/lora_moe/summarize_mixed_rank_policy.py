#!/usr/bin/env python3
"""Validate and summarize mixed-rank policy raw artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.kernels.lora_moe.matrix import mixed_rank_cases
from benchmark.kernels.lora_moe.profiling import summarize_timings_us

POLICIES = ("padded_rmax", "packed_bucket")
ORDERS = ("forward", "reverse")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _occupancy(case: dict[str, object]) -> str:
    if int(case["L_active"]) == 0:
        return "all-base"
    if int(case["B_base"]) == 1:
        return "mixed"
    return "full-lora"


def _load_results(root: Path) -> list[tuple[Path, dict[str, object]]]:
    results = []
    for path in sorted((root / "raw").glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("schema") != "sgl_lora_mixed_rank_policy_v2":
            raise ValueError(f"unexpected schema in {path}")
        correctness = data["correctness"]
        if not correctness["outputs_finite"]:
            raise ValueError(f"non-finite output in {path}")
        active_delta = correctness["active_delta"]
        if active_delta is not None and (
            not active_delta["nonzero"] or not float(active_delta["max_abs"]) > 0.0
        ):
            raise ValueError(f"active delta is zero in {path}")
        for policy in POLICIES:
            graph = data["policies"][policy]["graph_correctness"]
            if data["environment"]["cli"]["execution"] == "cuda_graph":
                if graph["memory_allocated_delta_bytes"] != 0:
                    raise ValueError(
                        f"graph replay allocation changed for {policy} in {path}"
                    )
                if graph["rank_metadata_allocations_per_replay_by_construction"] != 0:
                    raise ValueError(f"rank metadata allocation in {path}")
        results.append((path, data))
    if not results:
        raise ValueError(f"no raw JSON files under {root / 'raw'}")
    return results


def _raw_correctness_key(data: dict[str, object]) -> tuple[object, ...]:
    case = data["case"]
    return (
        case["device"],
        data["environment"]["cli"]["execution"],
        data["order"],
        case["model"],
        case["phase"],
        case["T"],
        case["K"],
        case["H_moe"],
        case["I"],
        case["E_local"],
        case["R_max"],
        _occupancy(case),
        case["L_active"],
        case["B_base"],
        case["L_capacity"],
    )


def _validate_active_correctness(
    results: list[tuple[Path, dict[str, object]]],
) -> list[dict[str, object]]:
    """Validate active deltas against path-local and matched numerical oracles.

    Distinct physical-rank reductions are not bitwise identical in BF16, and
    even the same-shape R=Rmax controls show a small run-to-run envelope.  The
    output bound therefore includes the exactly matched Rmax error plus two
    BF16-epsilon units at the observed output scale, but is capped at half of
    the LoRA signal so a dropped delta cannot pass.  Independent delta-vector
    cosine and relative-L2 gates protect against a directionally wrong update.
    """

    nulls: dict[tuple[object, ...], tuple[Path, dict[str, object]]] = {}
    for path, data in results:
        case = data["case"]
        if int(case["L_active"]) == 0 or int(case["R"]) != int(case["R_max"]):
            continue
        key = _raw_correctness_key(data)
        if key in nulls:
            raise ValueError(f"duplicate correctness null for {key}")
        nulls[key] = (path, data)

    validations = []
    for path, data in results:
        case = data["case"]
        if int(case["L_active"]) == 0:
            continue
        key = _raw_correctness_key(data)
        if key not in nulls:
            raise ValueError(f"missing correctness R=Rmax null for {key}")
        null_path, null = nulls[key]
        correctness = data["correctness"]
        active = correctness["active_delta"]
        signal = float(active["max_abs"])
        error = float(correctness["padded_vs_packed_max_abs"])
        null_error = float(null["correctness"]["padded_vs_packed_max_abs"])
        output_scale = float(correctness["output_max_abs"])
        bf16_envelope = 2.0 * (2.0**-7) * output_scale
        allowed_error = min(
            0.5 * signal,
            null_error + max(bf16_envelope, 0.1 * signal),
        )
        relative_l2 = float(active["difference"]["relative_l2"])
        cosine = float(active["cosine_similarity"])
        base_error = float(active["base_reference_max_abs_difference"])
        if error > allowed_error:
            raise ValueError(
                f"active output error exceeds numerical oracle in {path}: "
                f"error={error}, allowed={allowed_error}, signal={signal}, "
                f"matched_null={null_error}"
            )
        if relative_l2 > 0.1 or cosine < 0.995 or base_error > 5e-3:
            raise ValueError(
                f"active delta-vector oracle failed in {path}: "
                f"relative_l2={relative_l2}, cosine={cosine}, "
                f"base_error={base_error}"
            )
        validations.append(
            {
                "file": str(path),
                "matched_null_file": str(null_path),
                "output_error": error,
                "allowed_output_error": allowed_error,
                "signal": signal,
                "matched_null_error": null_error,
                "bf16_output_scale_envelope": bf16_envelope,
                "delta_relative_l2": relative_l2,
                "delta_cosine_similarity": cosine,
                "base_reference_max_abs_difference": base_error,
            }
        )
    return validations


def _validate_complete_matrix(
    results: list[tuple[Path, dict[str, object]]],
) -> dict[str, object]:
    devices = sorted({str(data["case"]["device"]) for _, data in results})
    expected = {
        (device, case.case_id, execution, order)
        for device in devices
        for case in mixed_rank_cases(device)
        for execution in ("eager", "cuda_graph")
        for order in ("forward", "reverse")
    }
    observed: dict[tuple[str, str, str, str], list[Path]] = defaultdict(list)
    samples_by_device: dict[str, set[int]] = defaultdict(set)
    for path, data in results:
        device = str(data["case"]["device"])
        execution = str(data["environment"]["cli"]["execution"])
        order = str(data["order"])
        key = (device, str(data["case"]["case_id"]), execution, order)
        observed[key].append(path)
        declared_samples = int(data["environment"]["cli"]["samples"])
        samples_by_device[device].add(declared_samples)
        policy_lengths = {
            policy: len(data["policies"][policy]["timing"]["raw_samples_us"])
            for policy in POLICIES
        }
        if set(policy_lengths.values()) != {declared_samples}:
            raise ValueError(
                f"sample count mismatch in {path}: declared={declared_samples}, "
                f"policies={policy_lengths}"
            )
    duplicates = {key: paths for key, paths in observed.items() if len(paths) != 1}
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            "matrix is incomplete or non-unique: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )
    if any(len(samples) != 1 for samples in samples_by_device.values()):
        raise ValueError(f"sample count changed within a device: {samples_by_device}")
    return {
        "devices": devices,
        "expected_raw_files_per_device": len(mixed_rank_cases(devices[0])) * 4,
        "observed_raw_files_per_device": {
            device: sum(key[0] == device for key in observed) for device in devices
        },
        "samples_per_policy_order_by_device": {
            device: next(iter(samples)) for device, samples in samples_by_device.items()
        },
        "exact_case_execution_order_coverage": True,
        "unique_artifact_per_cell_order": True,
    }


def _combine(results: list[tuple[Path, dict[str, object]]]):
    grouped = defaultdict(list)
    for path, data in results:
        case = data["case"]
        execution = data["environment"]["cli"]["execution"]
        grouped[(case["device"], case["case_id"], execution)].append((path, data))

    cells = []
    manifest_rows = []
    for (device, case_id, execution), members in sorted(grouped.items()):
        if len(members) != 2:
            raise ValueError(
                f"{device}/{case_id}/{execution} needs exactly two order artifacts"
            )
        orders = {data["order"] for _, data in members}
        if orders != {"forward", "reverse"}:
            raise ValueError(
                f"{device}/{case_id}/{execution} needs forward and reverse, got {orders}"
            )
        case = members[0][1]["case"]
        if any(data["case"] != case for _, data in members[1:]):
            raise ValueError(
                f"case metadata changed across orders for {device}/{case_id}/{execution}"
            )
        policy_rows = {}
        for policy in POLICIES:
            raw = []
            order_p50 = {}
            setup_by_order = {}
            for path, data in members:
                policy_data = data["policies"][policy]
                samples = [
                    float(value) for value in policy_data["timing"]["raw_samples_us"]
                ]
                raw.extend(samples)
                order_p50[data["order"]] = float(
                    policy_data["timing"]["timing"]["p50_us"]
                )
                setup_by_order[data["order"]] = float(
                    policy_data["load_time_setup_ms_one_shot"]
                )
                manifest_rows.append(
                    {
                        "device": device,
                        "case_id": case_id,
                        "execution": execution,
                        "order": data["order"],
                        "policy": policy,
                        "file": str(path.relative_to(path.parents[1])),
                        "samples": len(samples),
                        "p50_us": order_p50[data["order"]],
                    }
                )
            stats = summarize_timings_us(raw, launches_per_batch=1)
            order_drift = 100.0 * (
                max(order_p50.values()) / min(order_p50.values()) - 1.0
            )
            policy_rows[policy] = {
                "timing": {
                    "p20_us": stats.p20_us,
                    "p50_us": stats.p50_us,
                    "p80_us": stats.p80_us,
                    "min_us": stats.min_us,
                    "max_us": stats.max_us,
                    "num_samples": stats.num_samples,
                },
                "order_p50_us": order_p50,
                "order_drift_percent": order_drift,
                "resident_bytes": members[0][1]["policies"][policy]["resident_bytes"],
                "load_time_setup_ms_one_shot_by_order": setup_by_order,
                "load_time_setup_ms_descriptive_median": statistics.median(
                    setup_by_order.values()
                ),
                "load_time_setup_measurement": (
                    "two one-shot values in counterbalanced transform order; "
                    "descriptive only, not a canonical latency claim"
                ),
                "physical_rank": members[0][1]["policies"][policy]["physical_rank"],
            }
        padded = policy_rows["padded_rmax"]["timing"]["p50_us"]
        packed = policy_rows["packed_bucket"]["timing"]["p50_us"]
        order_effect_percent = {
            order: 100.0
            * (
                policy_rows["packed_bucket"]["order_p50_us"][order]
                / policy_rows["padded_rmax"]["order_p50_us"][order]
                - 1.0
            )
            for order in ORDERS
        }
        oracle = min(padded, packed)
        raw_winner = "padded_rmax" if padded <= packed else "packed_bucket"
        cells.append(
            {
                "device": device,
                "case_id": case_id,
                "execution": execution,
                "model": case["model"],
                "phase": case["phase"],
                "T": case["T"],
                "K": case["K"],
                "H_moe": case["H_moe"],
                "I": case["I"],
                "E_local": case["E_local"],
                "R": case["R"],
                "R_max": case["R_max"],
                "occupancy": _occupancy(case),
                "L_active": case["L_active"],
                "B_base": case["B_base"],
                "L_capacity": case["L_capacity"],
                "policies": policy_rows,
                "raw_p50_winner": raw_winner,
                "packed_vs_padded_percent": 100.0 * (packed / padded - 1.0),
                "order_effect_percent": order_effect_percent,
                "regret_percent": {
                    "padded_rmax": 100.0 * (padded / oracle - 1.0),
                    "packed_bucket": 100.0 * (packed / oracle - 1.0),
                },
                "correctness_max_abs": max(
                    float(data["correctness"]["padded_vs_packed_max_abs"])
                    for _, data in members
                ),
                "delta_error_over_signal_max": max(
                    (
                        (
                            float(data["correctness"]["padded_vs_packed_max_abs"])
                            / float(data["correctness"]["active_delta"]["max_abs"])
                        )
                        if data["correctness"]["active_delta"] is not None
                        else 0.0
                    )
                    for _, data in members
                ),
            }
        )
    return cells, manifest_rows


def _null_key(cell: dict[str, object]) -> tuple[object, ...]:
    """Return every non-rank identity needed for a matched null control."""

    return (
        cell["device"],
        cell["execution"],
        cell["model"],
        cell["phase"],
        cell["T"],
        cell["K"],
        cell["H_moe"],
        cell["I"],
        cell["E_local"],
        cell["R_max"],
        cell["occupancy"],
        cell["L_active"],
        cell["B_base"],
        cell["L_capacity"],
    )


def _effect_sign(effect: float) -> int:
    return (effect > 0.0) - (effect < 0.0)


def _apply_noise_bounds(cells: list[dict[str, object]]) -> None:
    """Apply a literal matched-null/order-consistency evidence heuristic.

    R=Rmax active rows execute identical physical shapes. A lower-rank row is
    labeled as beyond the observed bound only when forward and reverse order
    agree on the winner and each order-specific effect exceeds its fully
    shape-matched null effect. This is not a statistical significance test;
    raw timings and effect sizes remain the primary evidence.
    """

    nulls: dict[tuple[object, ...], dict[str, object]] = {}
    for cell in cells:
        if int(cell["L_active"]) == 0 or int(cell["R"]) != int(cell["R_max"]):
            continue
        key = _null_key(cell)
        if key in nulls:
            raise ValueError(f"duplicate matched R=Rmax null control for {key}")
        nulls[key] = cell

    for cell in cells:
        cell["evidence_winner"] = None
        if int(cell["L_active"]) == 0:
            cell["evidence_class"] = "all-base-bypass"
            cell["observed_null_bound_percent"] = None
            continue
        if int(cell["R"]) == int(cell["R_max"]):
            cell["evidence_class"] = "matched-null-control"
            cell["observed_null_bound_percent"] = {
                order: abs(float(cell["order_effect_percent"][order]))
                for order in ORDERS
            }
            continue

        key = _null_key(cell)
        if key not in nulls:
            raise ValueError(f"missing matched R=Rmax null control for {key}")
        null = nulls[key]
        effects = {
            order: float(cell["order_effect_percent"][order]) for order in ORDERS
        }
        null_effects = {
            order: float(null["order_effect_percent"][order]) for order in ORDERS
        }
        signs = {_effect_sign(effect) for effect in effects.values()}
        pooled_sign = _effect_sign(float(cell["packed_vs_padded_percent"]))
        consistent_order_winner = len(signs) == 1 and 0 not in signs
        pooled_agrees = consistent_order_winner and pooled_sign in signs
        each_order_exceeds_null = all(
            abs(effects[order]) > abs(null_effects[order]) for order in ORDERS
        )
        cell["matched_null_case_id"] = null["case_id"]
        cell["matched_null_order_effect_percent"] = null_effects
        cell["observed_null_bound_percent"] = {
            order: abs(null_effects[order]) for order in ORDERS
        }
        cell["evidence_consistent_order_winner"] = consistent_order_winner
        cell["evidence_pooled_winner_agrees"] = pooled_agrees
        cell["evidence_each_order_exceeds_null"] = each_order_exceeds_null
        cell["evidence_margin_vs_null_percent"] = {
            order: abs(effects[order]) - abs(null_effects[order]) for order in ORDERS
        }
        if pooled_agrees and each_order_exceeds_null:
            cell["evidence_class"] = "effect-beyond-observed-null-and-order-bound"
            cell["evidence_winner"] = cell["raw_p50_winner"]
        else:
            cell["evidence_class"] = "inconclusive-observed-bound"


def _static_regret(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    groupings: list[tuple[str, Callable[[dict[str, object]], tuple[object, ...]]]] = [
        (
            "device_execution_global",
            lambda cell: (cell["device"], cell["execution"]),
        ),
        (
            "device_execution_rank_signature",
            lambda cell: (
                cell["device"],
                cell["execution"],
                cell["R"],
                cell["R_max"],
            ),
        ),
    ]
    for scope, key_fn in groupings:
        grouped = defaultdict(list)
        for cell in cells:
            # All-base rows bypass LoRA entirely. R == R_max rows are active
            # same-physical-shape null controls. Neither is evidence for a
            # deployable rank-storage policy, so retain them in the cell table
            # while excluding them from policy regret and winner counts.
            if int(cell["L_active"]) == 0 or int(cell["R"]) == int(cell["R_max"]):
                continue
            grouped[key_fn(cell)].append(cell)
        for key, members in sorted(grouped.items()):
            policy_scores = {}
            for policy in POLICIES:
                regrets = [cell["regret_percent"][policy] for cell in members]
                ratios = [1.0 + regret / 100.0 for regret in regrets]
                policy_scores[policy] = {
                    "mean_regret_percent": statistics.fmean(regrets),
                    "p95_regret_percent": _percentile(regrets, 0.95),
                    "max_regret_percent": max(regrets),
                    "geometric_mean_oracle_ratio": math.exp(
                        statistics.fmean(math.log(value) for value in ratios)
                    ),
                }
            raw_best_fixed = min(
                POLICIES,
                key=lambda policy: policy_scores[policy]["geometric_mean_oracle_ratio"],
            )
            evidence_winners = [
                cell["evidence_winner"]
                for cell in members
                if cell["evidence_winner"] is not None
            ]
            if (
                len(evidence_winners) == len(members)
                and len(set(evidence_winners)) == 1
            ):
                evidence_recommendation = evidence_winners[0]
                evidence_recommendation_status = "all_cells_beyond_bound_and_agree"
            elif not evidence_winners:
                evidence_recommendation = None
                evidence_recommendation_status = "no_cells_beyond_bound"
            elif len(set(evidence_winners)) > 1:
                evidence_recommendation = None
                evidence_recommendation_status = "beyond_bound_cells_disagree"
            else:
                evidence_recommendation = None
                evidence_recommendation_status = "some_cells_inconclusive"
            summaries.append(
                {
                    "scope": scope,
                    "key": list(key),
                    "num_cells": len(members),
                    "raw_best_fixed_policy": raw_best_fixed,
                    "raw_best_fixed": policy_scores[raw_best_fixed],
                    "raw_policy_scores": policy_scores,
                    "evidence_policy_recommendation": evidence_recommendation,
                    "evidence_policy_recommendation_status": (
                        evidence_recommendation_status
                    ),
                    "evidence_winner_counts": {
                        policy: sum(
                            cell["evidence_winner"] == policy for cell in members
                        )
                        for policy in POLICIES
                    },
                    "inconclusive_cells": sum(
                        cell["evidence_winner"] is None for cell in members
                    ),
                }
            )
    return summaries


def _write_manifest(root: Path, rows: list[dict[str, object]]) -> None:
    path = root / "raw_manifest.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _readme(cells: list[dict[str, object]], regret: list[dict[str, object]]) -> str:
    lines = [
        "# SGL LoRA MoE mixed-rank policy benchmark",
        "",
        "The full M0 BF16 runner is identical in both arms. `padded_rmax` executes "
        "zero-tailed R_max factors; `packed_bucket` executes load-time packed "
        "R_phys factors. Packing, immutable rank-plan construction, and graph-family "
        "selection are outside forward timing. Every cell combines both forward-order "
        "and reverse-order samples per policy; exact counts are in the raw manifest.",
        "",
        "## Combined counterbalanced results",
        "",
        "| Device | Execution | Phase/T | Occupancy | R/Rmax | padded us | packed us | packed vs padded | Evidence | Packed/padded factor bytes |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for cell in sorted(
        cells,
        key=lambda item: (
            item["device"],
            item["execution"],
            item["phase"],
            item["occupancy"],
            item["R"],
        ),
    ):
        padded = cell["policies"]["padded_rmax"]
        packed = cell["policies"]["packed_bucket"]
        padded_bytes = int(padded["resident_bytes"])
        packed_bytes = int(packed["resident_bytes"])
        byte_ratio = packed_bytes / padded_bytes if padded_bytes else 1.0
        lines.append(
            f"| {cell['device']} | {cell['execution']} | "
            f"{cell['phase']}/{cell['T']} | {cell['occupancy']} | "
            f"{cell['R']}/{cell['R_max']} | "
            f"{padded['timing']['p50_us']:.3f} | "
            f"{packed['timing']['p50_us']:.3f} | "
            f"{cell['packed_vs_padded_percent']:+.2f}% | "
            f"{cell['evidence_winner'] or cell['evidence_class']} | {byte_ratio:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Raw static-policy regret (noise-qualified)",
            "",
            "A fixed policy is scored against the oracle policy for every included "
            "batch cell. This is the deployable graph-planner question; oracle-per-"
            "batch results are not presented as a production policy. Raw regret "
            "ratios remain visible, but categorical winner counts include only "
            "effects with order-consistent winners beyond both matched per-order "
            "null effects. This is an observed-bound heuristic, not a statistical "
            "significance test. All-base and "
            "R=Rmax null-control cells are excluded.",
            "",
            "| Scope | Key | Cells | Raw best fixed | Mean regret | P95 regret | Max regret | Evidence wins padded/packed | Inconclusive |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in regret:
        best = row["raw_best_fixed"]
        wins = row["evidence_winner_counts"]
        lines.append(
            f"| {row['scope']} | `{row['key']}` | {row['num_cells']} | "
            f"{row['raw_best_fixed_policy']} | {best['mean_regret_percent']:.2f}% | "
            f"{best['p95_regret_percent']:.2f}% | "
            f"{best['max_regret_percent']:.2f}% | "
            f"{wins['padded_rmax']}/{wins['packed_bucket']} | "
            f"{row['inconclusive_cells']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Core cells have a uniform active rank, including base+LoRA mixed rows; "
            "heterogeneous resident ranks are represented by the static planner but "
            "need multi-bucket execution before production dispatch can use them.",
            "- All-base cells resolve to N0 for both policies. Their small deltas are "
            "counterbalanced timing noise, not rank compute, and are excluded from "
            "static rank-policy regret.",
            "- R128/Rmax128 active cells are same-physical-rank null controls. They "
            "are matched on the complete non-rank shape and occupancy identity. A "
            "lower-rank row gets an evidence winner only when forward and reverse "
            "order agree and each order-specific effect exceeds its matched null.",
            "- `resident_bytes` and the factor-byte ratio cover the four LoRA factor "
            "tensors only; they exclude base weights, intermediates, routing metadata, "
            "and CUDA-graph pools.",
            "- This is WS1 local M0 evidence. It says nothing about distributed "
            "communication or canonical adapter load/eviction latency. Reported "
            "transform samples are descriptive one-shot observations only.",
            "- Each topology is captured and replayed independently. The benchmark "
            "serializes the graph key but does not validate a production graph cache "
            "or selection transition across slot/rank changes.",
            "- Active graph keys include the exact slot/rank assignment. Production "
            "can avoid recapture churn only after defining stable bucket-pool graph "
            "ownership; this planner intentionally does not claim that lifecycle.",
            "- Current padded kernels do not consume `lora_ranks`; zero tails are a "
            "semantic requirement. Packed execution makes physical rank explicit.",
            "- Rank-metadata allocation-free replay is a reviewed construction "
            "invariant: immutable tuple metadata is bound before capture/launch. The "
            "numeric zero is not presented as allocator-instrumented evidence.",
            "- Active correctness uses a matched same-shape R128 numerical control, "
            "a two-BF16-epsilon output-scale envelope capped below half the LoRA "
            "signal, delta relative-L2 <= 0.1, and delta cosine >= 0.995. Exact "
            "per-artifact values are in `summary.json`.",
            "",
            "See `summary.json`, `raw_manifest.csv`, raw JSON files, and "
            "`SHA256SUMS` for complete samples and provenance.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_hashes(root: Path, paths: Iterable[Path]) -> None:
    rows = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    root = args.root
    results = _load_results(root)
    correctness_validation = _validate_active_correctness(results)
    completeness = _validate_complete_matrix(results)
    cells, manifest = _combine(results)
    _apply_noise_bounds(cells)
    regret = _static_regret(cells)
    environments = defaultdict(set)
    for _, data in results:
        device = data["case"]["device"]
        environments[device].add(
            json.dumps(data["environment"]["source"], sort_keys=True)
        )
    if any(len(values) != 1 for values in environments.values()):
        raise ValueError("source provenance changed within a device matrix")

    summary = {
        "schema": "sgl_lora_mixed_rank_policy_summary_v2",
        "raw_files": len(results),
        "combined_cells": len(cells),
        "matrix_completeness": completeness,
        "active_correctness_validation": correctness_validation,
        "cells": cells,
        "static_policy_regret": regret,
        "source_provenance_by_device": {
            device: json.loads(next(iter(values)))
            for device, values in environments.items()
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_manifest(root, manifest)
    (root / "README.md").write_text(_readme(cells, regret))
    artifact_paths = [path for path, _ in results] + [
        root / "summary.json",
        root / "raw_manifest.csv",
        root / "README.md",
    ]
    _write_hashes(root, artifact_paths)
    print(
        f"summarized {len(results)} raw files into {len(cells)} counterbalanced cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
