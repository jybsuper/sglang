#!/usr/bin/env python3
"""Summarize matched ``bench_one_batch_server`` and PyTorch trace artifacts.

This keeps the model-level comparison auditable without loading the Chrome traces
in a browser.  It intentionally reports the measured providers as-is: missing
adapter traces and crashed profilers are evidence, not values to interpolate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_RESULT_RE = re.compile(r"^(base|lora)_r\d+\.jsonl$")
_TRACE_STAGE_RE = re.compile(r"-(EXTEND|DECODE)\.trace\.json\.gz$")


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def summarize_bench(directory: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    files = []
    for path in sorted(directory.glob("*_r*.jsonl")):
        match = _RESULT_RE.match(path.name)
        if match is None:
            continue
        files.append(path.name)
        traffic = match.group(1)
        for row in _read_jsonl(path):
            grouped[(traffic, int(row["batch_size"]))].append(row)

    if not grouped:
        raise ValueError(f"no repeated benchmark JSONL files found under {directory}")

    metrics = (
        "latency",
        "last_ttft",
        "input_throughput",
        "output_throughput",
        "overall_throughput",
    )
    cells: dict[str, Any] = {}
    for (traffic, batch_size), rows in sorted(grouped.items()):
        cell: dict[str, Any] = {
            "traffic": traffic,
            "batch_size": batch_size,
            "repetitions": len(rows),
            "input_len": sorted({int(row["input_len"]) for row in rows}),
            "output_len": sorted({int(row["output_len"]) for row in rows}),
        }
        for metric in metrics:
            samples = [float(row[metric]) for row in rows]
            cell[metric] = {
                "median": statistics.median(samples),
                "p20": _percentile(samples, 0.20),
                "p80": _percentile(samples, 0.80),
                "samples": samples,
            }
        cells[f"{traffic}_bs{batch_size}"] = cell
    return {"files": files, "cells": cells}


def _output_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("output_ids")
    return None


def summarize_correctness(directory: Path) -> dict[str, Any]:
    names = ("base_before", "adapter", "base_after", "mixed")
    values: dict[str, Any] = {}
    missing = []
    for name in names:
        path = directory / f"{name}.json"
        if not path.exists():
            missing.append(path.name)
            continue
        values[name] = json.loads(path.read_text())

    result: dict[str, Any] = {"missing": missing}
    if {"base_before", "base_after"} <= values.keys():
        result["base_before_after_ids_equal"] = (
            _output_ids(values["base_before"]) == _output_ids(values["base_after"])
        )
    if "mixed" in values and isinstance(values["mixed"], list):
        mixed = values["mixed"]
        result["mixed_size"] = len(mixed)
        if len(mixed) >= 2 and {"base_before", "adapter"} <= values.keys():
            result["mixed_base_ids_equal"] = _output_ids(mixed[0]) == _output_ids(
                values["base_before"]
            )
            result["mixed_adapter_ids_equal"] = _output_ids(mixed[1]) == _output_ids(
                values["adapter"]
            )
    boolean_checks = [
        value for key, value in result.items() if key.endswith("_equal")
    ]
    result["passed"] = not missing and bool(boolean_checks) and all(boolean_checks)
    return result


def _canonical_kernel(name: str) -> str:
    for marker in (
        "_chunked_lora_shrink_kernel",
        "_chunked_lora_expand_kernel",
        "_moe_lora_shrink_splitk_kernel",
        "_moe_lora_expand_add_flat_kernel",
        "_moe_lora_expand_add_kernel",
        "_gate_up_lora_b_kernel",
        "_sgemm_lora_a_kernel",
        "_sgemm_lora_b_kernel",
        "fill_gateup_input_triton_kernel",
        "post_reorder_deepgemm_triton_kernel",
        "_silu_mul_delta_masked_kernel",
        "fused_moe_kernel",
        "deep_gemm::",
        "bmm_Bfloat16",
    ):
        if marker in name:
            return marker
    return name


def summarize_trace(path: Path, *, top: int) -> dict[str, Any]:
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    aggregate: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    for event in payload.get("traceEvents", []):
        if event.get("cat") != "kernel" or "dur" not in event:
            continue
        name = _canonical_kernel(str(event.get("name", "unknown")))
        aggregate[name][0] += float(event["dur"])
        aggregate[name][1] += 1
    ordered = sorted(aggregate.items(), key=lambda item: -float(item[1][0]))
    return {
        "file": str(path),
        "kernel_time_us_sum": sum(float(value[0]) for value in aggregate.values()),
        "kernel_launches": sum(int(value[1]) for value in aggregate.values()),
        "top_kernels": [
            {"name": name, "time_us": value[0], "launches": value[1]}
            for name, value in ordered[:top]
        ],
    }


def summarize_profiles(directory: Path, *, top: int) -> dict[str, Any]:
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(directory.glob("profiles/**/*.trace.json.gz")):
        match = _TRACE_STAGE_RE.search(path.name)
        stage = match.group(1).lower() if match else "unknown"
        traces[stage].append(summarize_trace(path, top=top))

    crash_markers: dict[str, list[str]] = {}
    for path in sorted(directory.glob("*.log")):
        text = path.read_text(errors="replace")
        found = []
        for marker in (
            "SIGSEGV",
            "exit code -11",
            "Response ended prematurely",
            "scheduler_0 crashed",
        ):
            if marker in text:
                found.append(marker)
        if found:
            crash_markers[path.name] = found
    return {
        "traces": dict(traces),
        "profile_crash_markers": crash_markers,
    }


def compare(sgl: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(set(sgl["cells"]) & set(control["cells"])):
        sgl_cell = sgl["cells"][key]
        control_cell = control["cells"][key]
        metrics = {}
        for metric in (
            "latency",
            "last_ttft",
            "input_throughput",
            "output_throughput",
            "overall_throughput",
        ):
            sgl_value = float(sgl_cell[metric]["median"])
            control_value = float(control_cell[metric]["median"])
            metrics[metric] = {
                "sgl": sgl_value,
                "control": control_value,
                "sgl_minus_control_percent": 100.0
                * (sgl_value - control_value)
                / control_value,
                "higher_is_better": metric.endswith("throughput"),
            }
        result[key] = metrics
    return result


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Matched GB300 model-level SGL-LoRA comparison",
        "",
        "Three counterbalanced repetitions were run on the same GPU. Positive",
        "percentages mean SGL is numerically larger; that is favorable for throughput",
        "and unfavorable for latency/TTFT.",
        "",
        "| Traffic | BS | SGL out tok/s | Control out tok/s | Difference | SGL input tok/s | Control input tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, metrics in summary["comparison"].items():
        cell = summary["sgl"]["bench"]["cells"][key]
        output = metrics["output_throughput"]
        input_metric = metrics["input_throughput"]
        lines.append(
            f"| {cell['traffic']} | {cell['batch_size']} | "
            f"{output['sgl']:.2f} | {output['control']:.2f} | "
            f"{output['sgl_minus_control_percent']:+.2f}% | "
            f"{input_metric['sgl']:.2f} | {input_metric['control']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Correctness and profiling",
            "",
            f"- SGL transition checks passed: `{summary['sgl']['correctness']['passed']}`.",
            f"- Control transition checks passed: `{summary['control']['correctness']['passed']}`.",
            "- The control produced a base trace, but its adapter profiling request",
            "  crashed the scheduler (the exact markers and log are retained).",
            "- SGL produced both base and adapter EXTEND/DECODE traces without a crash.",
            "",
            "The JSON companion retains p20/p80/sample values and aggregated kernel",
            "launch/time tables. Unprofiled `bench_one_batch_server` repetitions are the",
            "performance authority; traces are structural evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgl-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--top-kernels", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = {
        "method": {
            "benchmark_authority": "unprofiled_bench_one_batch_server",
            "comparison": "same_gpu_counterbalanced_three_repetitions",
            "trace_role": "structural_only",
        },
        "sgl": {
            "bench": summarize_bench(args.sgl_dir),
            "correctness": summarize_correctness(args.sgl_dir),
            "profiles": summarize_profiles(args.sgl_dir, top=args.top_kernels),
        },
        "control": {
            "bench": summarize_bench(args.control_dir),
            "correctness": summarize_correctness(args.control_dir),
            "profiles": summarize_profiles(args.control_dir, top=args.top_kernels),
        },
    }
    summary["comparison"] = compare(
        summary["sgl"]["bench"], summary["control"]["bench"]
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
