#!/usr/bin/env python3
"""Aggregate process-isolated C2 cross-model guardrail artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_STEM = re.compile(
    r"^(?P<model>.+)_T(?P<tokens>\d+)_R(?P<rank>\d+)_"
    r"(?P<schedule>pair|aligned)_BN(?P<block_n>\d+)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _classify_failure(log: str) -> str:
    if "illegal memory access" in log:
        return "cuda_illegal_memory_access"
    if "activation differs under CUDA graph replay" in log:
        return "cuda_graph_activation_mismatch"
    if "down rank differs under CUDA graph replay" in log:
        return "cuda_graph_down_rank_mismatch"
    if "AssertionError" in log:
        return "correctness_assertion"
    return "other"


def _parse_status(path: Path) -> dict[str, object]:
    match = _STEM.match(path.stem)
    if match is None:
        raise ValueError(f"unexpected result stem {path.stem!r}")
    metadata = {
        "model": match.group("model"),
        "tokens": int(match.group("tokens")),
        "rank": int(match.group("rank")),
        "schedule": match.group("schedule"),
        "block_n": int(match.group("block_n")),
    }
    status = int(path.read_text().strip())
    json_path = path.with_suffix(".json")
    log_path = path.with_suffix(".log")
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    if status != 0 or not json_path.exists():
        return {
            **metadata,
            "status": "failed",
            "exit_code": status,
            "failure_class": _classify_failure(log),
            "log_path": str(log_path),
            "log_sha256": _sha256(log_path) if log_path.exists() else None,
            "log_tail": log.splitlines()[-20:],
        }
    payload = json.loads(json_path.read_text())
    record = payload["records"][0]
    run = record["runs"][0]
    check = next(iter(record["checks"].values()))
    pairs = record["route"]["pairs"]
    pairs_post_padded = record["route"]["pairs_post_padded"]
    return {
        **metadata,
        "status": "passed",
        "p20_us": run["timing"]["p20_us"],
        "p50_us": run["timing"]["p50_us"],
        "p80_us": run["timing"]["p80_us"],
        "activation_max_abs_error": check["activation_max_abs_error"],
        "activation_error_over_signal": check["activation_error_over_signal"],
        "down_rank_max_abs_error": check["down_rank_max_abs_error"],
        "down_rank_error_over_signal": check["down_rank_error_over_signal"],
        "invalid_pairs": check["invalid_pairs"],
        "base_only_tokens": check["base_only_tokens"],
        "routed_scaling": check["routed_scaling"],
        "graph_correctness": run["graph_correctness"],
        "kernel_family": record["semantic_contract"]["kernel_family"],
        "logical_i": record["case"]["logical_i"],
        "physical_i": record["case"]["physical_i"],
        "pairs": pairs,
        "pairs_post_padded": pairs_post_padded,
        "route_padding_ratio": pairs_post_padded / pairs,
        "json_path": str(json_path),
        "json_sha256": _sha256(json_path),
        "log_path": str(log_path),
        "log_sha256": _sha256(log_path),
    }


def _aggregate(records: list[dict[str, object]], device: str) -> dict[str, object]:
    passed = [record for record in records if record["status"] == "passed"]
    failed = [record for record in records if record["status"] == "failed"]
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for record in passed:
        grouped[(record["model"], record["tokens"], record["rank"])].append(record)
    selections = []
    for (model, tokens, rank), candidates in sorted(grouped.items()):
        fastest = min(candidates, key=lambda candidate: candidate["p50_us"])
        best_by_schedule = {}
        for schedule in ("pair", "aligned"):
            schedule_candidates = [
                candidate
                for candidate in candidates
                if candidate["schedule"] == schedule
            ]
            if schedule_candidates:
                best_by_schedule[schedule] = min(
                    schedule_candidates, key=lambda candidate: candidate["p50_us"]
                )
        ratio = None
        if set(best_by_schedule) == {"pair", "aligned"}:
            ratio = (
                best_by_schedule["aligned"]["p50_us"]
                / best_by_schedule["pair"]["p50_us"]
            )
        selections.append(
            {
                "model": model,
                "tokens": tokens,
                "rank": rank,
                "selected_schedule": fastest["schedule"],
                "selected_block_n": fastest["block_n"],
                "selected_p50_us": fastest["p50_us"],
                "best_pair": best_by_schedule.get("pair"),
                "best_aligned": best_by_schedule.get("aligned"),
                "aligned_over_pair": ratio,
                "candidate_count_passed": len(candidates),
            }
        )

    model_summary = {}
    for model in sorted({record["model"] for record in records}):
        model_selections = [row for row in selections if row["model"] == model]
        model_failures = [record for record in failed if record["model"] == model]
        model_summary[model] = {
            "selection_counts": dict(
                Counter(row["selected_schedule"] for row in model_selections)
            ),
            "passed_configs": sum(record["model"] == model for record in passed),
            "failed_configs": len(model_failures),
            "failure_classes": dict(
                Counter(record["failure_class"] for record in model_failures)
            ),
        }

    return {
        "schema_version": 1,
        "device": device,
        "counts": {
            "status_files": len(records),
            "passed": len(passed),
            "failed": len(failed),
            "resolved_shape_cells": len(selections),
        },
        "correctness_maxima": {
            "activation_max_abs_error": max(
                (record["activation_max_abs_error"] for record in passed),
                default=None,
            ),
            "activation_error_over_signal": max(
                (record["activation_error_over_signal"] or 0.0 for record in passed),
                default=None,
            ),
            "down_rank_max_abs_error": max(
                (record["down_rank_max_abs_error"] for record in passed),
                default=None,
            ),
            "down_rank_error_over_signal": max(
                (record["down_rank_error_over_signal"] or 0.0 for record in passed),
                default=None,
            ),
            "routed_scaling_max_abs_error": max(
                (
                    record["routed_scaling"]["max_abs_error"]
                    for record in passed
                    if record["routed_scaling"] is not None
                ),
                default=None,
            ),
        },
        "model_summary": model_summary,
        "selections": selections,
        "failures": failed,
        "records": records,
    }


def _write_markdown(payload: dict[str, object], path: Path) -> None:
    counts = payload["counts"]
    lines = [
        f"# C2 cross-model guardrail summary: {payload['device']}",
        "",
        (
            f"Passed {counts['passed']}/{counts['status_files']} isolated schedule "
            f"configurations; {counts['failed']} failed. Pair timing includes the "
            "down-rank zero plus one consumer launch; aligned timing includes the "
            "zero, masked base-only activation fill, and aligned consumer under "
            "CUDA graph replay."
        ),
        "",
        "| Model | T | R | Selected | BN | p50 us | aligned/pair |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["selections"]:
        ratio = row["aligned_over_pair"]
        lines.append(
            f"| {row['model']} | {row['tokens']} | {row['rank']} | "
            f"{row['selected_schedule']} | {row['selected_block_n']} | "
            f"{row['selected_p50_us']:.3f} | "
            f"{ratio:.3f} |"
            if ratio is not None
            else f"| {row['model']} | {row['tokens']} | {row['rank']} | "
            f"{row['selected_schedule']} | {row['selected_block_n']} | "
            f"{row['selected_p50_us']:.3f} | n/a |"
        )
    if payload["failures"]:
        lines.extend(
            [
                "",
                "## Failed experimental configurations",
                "",
                "| Model | T | R | Schedule | BN | Failure |",
                "|---|---:|---:|---|---:|---|",
            ]
        )
        for record in payload["failures"]:
            lines.append(
                f"| {record['model']} | {record['tokens']} | {record['rank']} | "
                f"{record['schedule']} | {record['block_n']} | "
                f"{record['failure_class']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--results-dir", type=Path, action="append", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    status_paths = sorted(
        path for directory in args.results_dir for path in directory.glob("*.status")
    )
    records = [_parse_status(path) for path in status_paths]
    payload = _aggregate(records, args.device)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, args.markdown_output)
    print(json.dumps(payload["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
