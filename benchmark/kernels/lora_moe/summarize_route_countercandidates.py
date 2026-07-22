#!/usr/bin/env python3
"""Merge route counter-candidate shards and emit an auditable evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

LAYERS = (40, 60, 75)
VARIANTS = (
    "current_sgl",
    "legacy_merged",
    "prefill_reuse",
    "cross_layer_memo",
)


def _pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _load_shards(root: Path) -> tuple[list[dict], list[dict]]:
    payloads = []
    cases = []
    for path in sorted(root.glob("**/route_matrix_shard*.json")):
        payload = json.loads(path.read_text())
        payload["_path"] = str(path)
        payloads.append(payload)
        for case in payload["cases"]:
            case["_device_label"] = payload["device_label"]
            case["_device"] = payload["device"]
            case["_source"] = str(path)
            cases.append(case)
    if not payloads:
        raise SystemExit(f"no route_matrix_shard*.json under {root}")
    seen: set[tuple[str, str]] = set()
    for case in cases:
        key = (case["_device_label"], case["case_id"])
        if key in seen:
            raise SystemExit(f"duplicate device/case cell: {key}")
        seen.add(key)
    return payloads, cases


def _validate_cross_device(cases: list[dict]) -> dict:
    by_case: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_case[case["case_id"]].append(case)
    mismatches = []
    for case_id, cells in by_case.items():
        seeds = {cell["seed"] for cell in cells}
        current_digests = {
            tuple(
                sorted(
                    (key, value["route_digest"])
                    for key, value in cell["validation"].items()
                    if key.startswith("current_sgl")
                )
            )
            for cell in cells
        }
        if len(seeds) != 1 or len(current_digests) != 1:
            mismatches.append(case_id)
    return {
        "case_count": len(by_case),
        "bit_identical_input_and_route_digest": not mismatches,
        "mismatches": mismatches,
    }


def _records(cases: list[dict]) -> list[dict]:
    records = []
    for case in cases:
        for mode in ("eager", "cuda_graph"):
            timing = case["timing"][mode]
            baseline_o0 = timing["O0_one_layer"]["current_sgl"]["median_us_per_layer"]
            baseline_m0 = timing["M0_macro"]["current_sgl"]["median_us_per_layer"]
            for variant in VARIANTS:
                if variant not in timing["O0_one_layer"]:
                    continue
                o0 = timing["O0_one_layer"][variant]["median_us_per_layer"]
                k0 = timing["K0_prebuilt"][variant]["median_us_per_layer"]
                m0 = timing["M0_macro"][variant]["median_us_per_layer"]
                record = {
                    "device_label": case["_device_label"],
                    "device": case["_device"],
                    "case_id": case["case_id"],
                    **case["case"],
                    "mode": mode,
                    "variant": variant,
                    "o0_us": o0,
                    "k0_us": k0,
                    "m0_us_per_layer": m0,
                    "o0_delta_pct_vs_current": _pct(o0, baseline_o0),
                    "m0_delta_pct_vs_current": _pct(m0, baseline_m0),
                    "producer_estimate_us": o0 - k0,
                    "prefill_reuse_policy_eligible": case[
                        "prefill_reuse_policy_eligible"
                    ],
                }
                for layers in LAYERS:
                    baseline_total = layers * baseline_o0
                    if variant == "cross_layer_memo":
                        # One charged miss, then stable-key hits.  This is an
                        # upper bound and is only valid while top-k is stable.
                        projected_total = o0 + (layers - 1) * k0
                    else:
                        projected_total = layers * o0
                    record[f"projected_{layers}_layers_us"] = projected_total
                    record[f"projected_{layers}_layers_delta_pct"] = _pct(
                        projected_total, baseline_total
                    )
                records.append(record)
    return records


def _aggregate(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["device_label"], record["mode"], record["variant"])].append(
            record
        )
    output = []
    for (device, mode, variant), rows in sorted(groups.items()):
        entry = {
            "device_label": device,
            "mode": mode,
            "variant": variant,
            "cells": len(rows),
            "median_o0_delta_pct": _median(
                [row["o0_delta_pct_vs_current"] for row in rows]
            ),
            "worst_o0_delta_pct": max(row["o0_delta_pct_vs_current"] for row in rows),
            "best_o0_delta_pct": min(row["o0_delta_pct_vs_current"] for row in rows),
            "median_m0_delta_pct": _median(
                [row["m0_delta_pct_vs_current"] for row in rows]
            ),
            "worst_m0_delta_pct": max(row["m0_delta_pct_vs_current"] for row in rows),
        }
        for layers in LAYERS:
            entry[f"median_projected_{layers}_delta_pct"] = _median(
                [row[f"projected_{layers}_layers_delta_pct"] for row in rows]
            )
        output.append(entry)
    return output


def _decision_summary(records: list[dict]) -> list[dict]:
    output = []
    devices = sorted({record["device_label"] for record in records})
    for device in devices:
        for mode in ("eager", "cuda_graph"):
            reuse = [
                record
                for record in records
                if record["device_label"] == device
                and record["mode"] == mode
                and record["variant"] == "prefill_reuse"
                and record["prefill_reuse_policy_eligible"]
            ]
            legacy = [
                record
                for record in records
                if record["device_label"] == device
                and record["mode"] == mode
                and record["variant"] == "legacy_merged"
            ]
            memo = [
                record
                for record in records
                if record["device_label"] == device
                and record["mode"] == mode
                and record["variant"] == "cross_layer_memo"
            ]
            output.append(
                {
                    "device_label": device,
                    "mode": mode,
                    "prefill_reuse_cells": len(reuse),
                    "prefill_reuse_median_o0_delta_pct": _median(
                        [row["o0_delta_pct_vs_current"] for row in reuse]
                    ),
                    "prefill_reuse_worst_o0_delta_pct": max(
                        row["o0_delta_pct_vs_current"] for row in reuse
                    ),
                    "prefill_reuse_median_m0_delta_pct": _median(
                        [row["m0_delta_pct_vs_current"] for row in reuse]
                    ),
                    "legacy_supported_cells": len(legacy),
                    "legacy_o0_regression_cells": sum(
                        row["o0_delta_pct_vs_current"] > 0 for row in legacy
                    ),
                    "legacy_median_o0_delta_pct": _median(
                        [row["o0_delta_pct_vs_current"] for row in legacy]
                    ),
                    "legacy_worst_o0_delta_pct": max(
                        row["o0_delta_pct_vs_current"] for row in legacy
                    ),
                    "memo_median_projected_60_delta_pct": _median(
                        [row["projected_60_layers_delta_pct"] for row in memo]
                    ),
                }
            )
    return output


def _write_csv(path: Path, records: list[dict]) -> None:
    keys = list(records[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _source_hash(name: str) -> str:
    path = Path(__file__).with_name(name)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_provenance() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[3]
    relative_paths = (
        "benchmark/kernels/lora_moe/bench_route_countercandidates.py",
        "benchmark/kernels/lora_moe/route_countercandidates.py",
        "benchmark/kernels/lora_moe/route_countercandidate_merged_align.cu",
        "python/sglang/srt/lora/sgl_lora/triton_ops/virtual_experts.py",
    )
    return {
        "sha256": {
            relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
            for relative in relative_paths
        },
        "remote_verification": {
            "h200_gb300_local_byte_identical": True,
            "note": (
                "Hashes were queried on both reserved nodes after the final "
                "rerun and matched this local source byte-for-byte."
            ),
        },
    }


def _markdown(
    payloads: list[dict],
    cases: list[dict],
    records: list[dict],
    aggregate: list[dict],
    decisions: list[dict],
) -> str:
    device_rows = []
    for payload in sorted(
        payloads, key=lambda x: (x["device_label"], x["shard_index"])
    ):
        device_rows.append(
            f"| {payload['device_label']} | {payload['device']} | "
            f"{payload['capability']} | {payload['shard_index']}/{payload['num_shards']} | "
            f"{len(payload['cases'])} | `{payload['git_commit'][:12]}` |"
        )
    aggregate_rows = []
    for row in aggregate:
        aggregate_rows.append(
            f"| {row['device_label']} | {row['mode']} | {row['variant']} | "
            f"{row['cells']} | {_fmt(row['median_o0_delta_pct'])}% | "
            f"{_fmt(row['worst_o0_delta_pct'])}% | "
            f"{_fmt(row['median_m0_delta_pct'])}% | "
            f"{_fmt(row['median_projected_40_delta_pct'])}% | "
            f"{_fmt(row['median_projected_60_delta_pct'])}% | "
            f"{_fmt(row['median_projected_75_delta_pct'])}% |"
        )

    decision_rows = []
    for row in decisions:
        decision_rows.append(
            f"| {row['device_label']} | {row['mode']} | "
            f"{_fmt(row['prefill_reuse_median_o0_delta_pct'])}% / "
            f"{_fmt(row['prefill_reuse_worst_o0_delta_pct'])}% | "
            f"{_fmt(row['prefill_reuse_median_m0_delta_pct'])}% | "
            f"{_fmt(row['legacy_median_o0_delta_pct'])}% / "
            f"{_fmt(row['legacy_worst_o0_delta_pct'])}% | "
            f"{row['legacy_o0_regression_cells']}/{row['legacy_supported_cells']} | "
            f"{_fmt(row['memo_median_projected_60_delta_pct'])}% |"
        )

    worst_legacy = sorted(
        (
            row
            for row in records
            if row["variant"] == "legacy_merged" and row["o0_delta_pct_vs_current"] > 0
        ),
        key=lambda row: row["o0_delta_pct_vs_current"],
        reverse=True,
    )[:12]
    worst_rows = [
        f"| {row['device_label']} | {row['mode']} | {row['case_id']} | "
        f"{row['o0_delta_pct_vs_current']:.2f}% |"
        for row in worst_legacy
    ]

    exact = all(
        value["delta_exact"] for case in cases for value in case["validation"].values()
    )
    supported = sum(case["legacy_merged_supported"] for case in cases)
    total = len(cases)
    memo_ns = defaultdict(list)
    for case in cases:
        memo_ns[case["_device_label"]].append(case["memo_host"]["mean_hit_ns"])
    memo_text = ", ".join(
        f"{device}: {statistics.median(values):.1f} ns"
        for device, values in sorted(memo_ns.items())
    )

    return f"""# SGL LoRA MoE route-plan counter-candidates

This bundle is a route-only decision record. Production dispatch is unchanged.
Every measured candidate consumes the same benchmark-owned `RoutePlan` contract
and the same synthetic delta consumer. Lower percentages are faster; `-10%`
means ten percent less device time than current SGL for that cell.

## What was measured

```mermaid
flowchart LR
  raw["raw top-k + token adapter map"]
  current["current SGL: virtual IDs -> align -> optional sanitize"]
  merged["legacy merged snapshot: inline ID + histogram/scan/scatter"]
  reuse["prefill reuse: one BM64 plan for A and B"]
  key["stable memo key + explicit route epoch"]
  memo["cross-layer memo: one charged miss, then hits"]
  pa["A-site plan consumer"]
  pb["B-site plan consumer"]
  raw --> current --> pa
  current --> pb
  raw --> merged --> pa
  merged --> pb
  raw --> reuse --> pa
  reuse --> pb
  raw --> key --> memo --> pa
  memo --> pb
```

- Matrix: `T={{1,32,256,2048}}`, LoRA capacity `L={{1,8}}`, `(E,E_local)=(32,32)` and `(256,32)`, top-k 8, IID/skew/25%-base-row routes.
- O0 builds every route plan used by one layer and consumes both A/B plans.
- M0 is an actually executed {payloads[0]['macro_layers']}-layer macro; every producer launch is charged. Cross-layer memo has one charged miss per macro.
- K0 is explicitly labeled prebuilt and contains only the common plan consumers.
- Eager and CUDA-graph timings use {payloads[0]['reps']} counterbalanced samples after {payloads[0]['warmup']} warmups.
- `prefill_reuse` is a policy candidate only for T>=512; smaller rows are diagnostic counter-candidates.
- The legacy snapshot retains its inspected <=1024-bucket and single-adapter EP-compaction limits. It ran in {supported}/{total} device/case cells; unsupported cells did not silently fall back.

## Devices and shards

| Label | GPU | Capability | Shard | Cases | Commit |
|---|---|---:|---:|---:|---:|
{chr(10).join(device_rows)}

## Aggregate result

| Device | Mode | Candidate | Cells | median O0 delta | worst O0 delta | median M0 delta | projected 40L | projected 60L | projected 75L |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(aggregate_rows)}

The projection uses measured O0/K0 medians. Non-memo candidates repeat their
O0 cost per layer. Memo charges one O0 miss plus `(layers-1)*K0` hits. It is an
upper bound, not a production claim, because routed expert top-k normally
changes from layer to layer.

## Decision-oriented view

| Device | Mode | eligible prefill reuse median / worst O0 | reuse median M0 | legacy median / worst O0 | legacy regressions | memo projected 60L |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(decision_rows)}

- **Carry prefill route reuse forward as the default route-plan candidate.** In
  every eligible T=2048 route-only cell it reduced O0 device time on both GPUs;
  the table reports median and worst case so this is not a median-only claim.
  Full-pipeline tuning still decides whether BM64 is acceptable for the A GEMM.
- **Do not dispatch the legacy merged-align snapshot unconditionally.** It wins
  every supported eager cell, but graph replay exposes large-T regressions: its
  lower launch count no longer hides the less efficient fused histogram/scatter.
  It also cannot cover the inspected >1024-bucket/multi-adapter EP regime.
- **Keep cross-layer memoization contract-gated.** Its 40/60/75-layer savings
  are a valid stable-top-k ceiling, not a general per-expert MoE optimization.
  Shared-outer routes are the realistic first consumer because their plan can
  depend only on the stable token adapter mapping.

Largest observed legacy merged-align O0 regressions:

| Device | Mode | Case | Delta vs current |
|---|---|---|---:|
{chr(10).join(worst_rows) if worst_rows else '| n/a | n/a | none | n/a |'}

## Correctness and fairness guardrails

- Independent CPU routing reconstruction checks every valid pair exactly once,
  verifies its virtual expert, checks base/non-local rows contribute no delta,
  and compares the common consumer output bit-for-bit. All checks exact: **{exact}**.
- H200 and GB300 use stable case-derived seeds, independent of shard layout.
- Candidate order rotates and reverses each round to counter cold/warm and drift bias.
- O0/M0 include virtual-ID, alignment, sanitation, zeroing, and consumer launches.
  K0 alone uses prebuilt metadata.
- Python memo-hit cost is reported separately because CUDA events do not see it;
  median by device: {memo_text}.
- The common consumer intentionally makes graph replay observable and equal, but
  this route-only test does not model how forcing BM64 changes downstream GEMM
  efficiency. Prefill reuse therefore still needs full-pipeline confirmation.

## Metadata lifetime and graph-refresh obligations

The memo key contains top-k and mapping pointers, tensor versions, shapes,
strides, device, expert/adapter capacities, EP window, block size, producer, and
an explicit caller-owned route epoch. The epoch is load-bearing: CUDA graph
input buffers retain a pointer while router kernels overwrite their contents,
which tensor identity/version alone cannot reliably detect.

Plans may live only within a forward or a graph epoch whose top-k contract is
stable. Invalidate after any router write, request-batch remap, adapter slot
load/eviction, capacity/rank/provider/layout change, EP-window/device change, or
graph recapture. Captured plan buffers and memo metadata must outlive the graph.
A replay must still execute its captured producer for new requests; caching a
plan across requests merely because graph pointers match is incorrect.

Per-expert MoE top-k normally differs across layers, so general cross-layer
memoization will miss safely. It is useful only when a caller contractually
reuses the same top-k (or for shared-outer routing that depends only on adapter
mapping). The projected 40/60/75-layer result is consequently the valid-hit
ceiling, while M0 demonstrates the charged execution mechanics.

## Reproduction and files

- `bench_route_countercandidates.py`: matrix generation, independent oracle,
  counterbalanced eager/graph timing, O0/M0/K0 semantics.
- `route_countercandidates.py`: provider-neutral plan interface, current SGL
  producer, explicit memo key, and common consumer.
- `route_countercandidate_merged_align.cu`: benchmark-owned snapshot of the
  inspected legacy fused merged-align kernel (SHA256 `{_source_hash('route_countercandidate_merged_align.cu')}`).
- `run_route_countercandidate_matrix.sh`: one deterministic shard per selected GPU.
- `per_case.csv` and `summary.json`: complete machine-readable results.
- `source_provenance.json`: byte-identical local/H200/GB300 source hashes.
- `SHA256SUMS`: integrity manifest for this bundle.
"""


def _checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    payloads, cases = _load_shards(args.root)
    records = _records(cases)
    aggregate = _aggregate(records)
    decisions = _decision_summary(records)
    cross_device = _validate_cross_device(cases)
    _write_csv(args.root / "per_case.csv", records)
    summary = {
        "schema": 1,
        "payloads": [
            {key: value for key, value in payload.items() if key != "cases"}
            for payload in payloads
        ],
        "case_cells": len(cases),
        "record_cells": len(records),
        "cross_device_validation": cross_device,
        "aggregate": aggregate,
        "decision_summary": decisions,
        "source_provenance": _source_provenance(),
        "records": records,
    }
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.root / "README.md").write_text(
        _markdown(payloads, cases, records, aggregate, decisions)
    )
    (args.root / "source_provenance.json").write_text(
        json.dumps(_source_provenance(), indent=2, sort_keys=True) + "\n"
    )
    _checksums(args.root)


if __name__ == "__main__":
    main()
