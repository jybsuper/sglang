#!/usr/bin/env python3
"""Extract stream topology and overlap metrics from an Nsight SQLite export."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def _merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def _duration(intervals) -> int:
    return sum(end - start for start, end in intervals)


def _intersection_duration(lhs, rhs) -> int:
    i = j = total = 0
    while i < len(lhs) and j < len(rhs):
        start = max(lhs[i][0], rhs[j][0])
        end = min(lhs[i][1], rhs[j][1])
        if end > start:
            total += end - start
        if lhs[i][1] <= rhs[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(path)
    strings = dict(connection.execute("SELECT id, value FROM StringIds"))
    rows = list(
        connection.execute(
            "SELECT start, end, streamId, shortName, graphNodeId, graphId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        )
    )
    if not rows:
        raise ValueError(f"no CUDA kernels in {path}")

    by_stream = {}
    for start, end, stream, name_id, graph_node, graph_id in rows:
        item = by_stream.setdefault(
            int(stream), {"intervals": [], "names": Counter(), "nodes": set()}
        )
        item["intervals"].append((int(start), int(end)))
        item["names"][strings[name_id]] += 1
        if graph_node is not None:
            item["nodes"].add(int(graph_node))

    stream_rows = []
    merged_by_stream = {}
    for stream, item in by_stream.items():
        intervals = _merged(item["intervals"])
        merged_by_stream[stream] = intervals
        stream_rows.append(
            {
                "stream_id": stream,
                "kernel_count": len(item["intervals"]),
                "unique_graph_nodes": len(item["nodes"]),
                "kernel_sum_us": _duration(item["intervals"]) / 1e3,
                "busy_union_us": _duration(intervals) / 1e3,
                "top_kernels": [
                    {"name": name, "count": count}
                    for name, count in item["names"].most_common(12)
                ],
            }
        )
    stream_rows.sort(key=lambda row: row["kernel_sum_us"], reverse=True)

    all_union = _merged([(row[0], row[1]) for row in rows])
    kernel_sum_ns = sum(row[1] - row[0] for row in rows)
    union_ns = _duration(all_union)
    pair_overlap = None
    if len(stream_rows) >= 2:
        first, second = stream_rows[0]["stream_id"], stream_rows[1]["stream_id"]
        overlap_ns = _intersection_duration(
            merged_by_stream[first], merged_by_stream[second]
        )
        shorter_ns = min(
            _duration(merged_by_stream[first]),
            _duration(merged_by_stream[second]),
        )
        pair_overlap = {
            "stream_ids": [first, second],
            "overlap_us": overlap_ns / 1e3,
            "overlap_fraction_of_shorter_stream": (
                overlap_ns / shorter_ns if shorter_ns else 0.0
            ),
        }
    return {
        "schema_version": 1,
        "source": str(path),
        "kernel_count": len(rows),
        "stream_count": len(stream_rows),
        "graph_ids": sorted({row[5] for row in rows if row[5] is not None}),
        "kernel_sum_us": kernel_sum_ns / 1e3,
        "gpu_busy_union_us": union_ns / 1e3,
        "concurrent_kernel_time_us": (kernel_sum_ns - union_ns) / 1e3,
        "concurrent_fraction_of_busy_union": (
            (kernel_sum_ns - union_ns) / union_ns if union_ns else 0.0
        ),
        "primary_stream_pair": pair_overlap,
        "streams": stream_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = analyze(args.sqlite)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.json_output:
        args.json_output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
