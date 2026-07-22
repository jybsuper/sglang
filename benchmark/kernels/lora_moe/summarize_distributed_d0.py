#!/usr/bin/env python3
"""Validate and summarize the distributed SGL LoRA MoE D0 evidence bundle."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

EXECUTED_HARNESS_SHA256 = (
    "5a2dc6b1ceca280934870f16d1c32ba3bd016ddbe1b5cae6d000410a457c507d"
)
METADATA_CONTRACT = [
    "global_token_id",
    "topk_slot",
    "global_expert_id",
    "adapter_id",
    "origin_ep_rank",
]
ADAPTER_MODES = ("active", "mixed", "base")
EP_ROUTE_MODES = ("all_local", "mixed", "no_local")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_groups(rank: int, topology: dict[str, Any]) -> dict[str, list[int]]:
    tp = int(topology["tp"])
    ep = int(topology["ep"])
    dp = int(topology["moe_dp"])
    moe_tp = tp // (ep * dp)
    dp_rank = rank // (ep * moe_tp)
    ep_rank = (rank // moe_tp) % ep
    moe_tp_rank = rank % moe_tp
    moe_tp_base = (dp_rank * ep + ep_rank) * moe_tp
    ep_base = dp_rank * ep * moe_tp + moe_tp_rank
    dp_base = ep_rank * moe_tp + moe_tp_rank
    return {
        "tp": list(range(tp)),
        "moe_tp": list(range(moe_tp_base, moe_tp_base + moe_tp)),
        "moe_ep": [ep_base + index * moe_tp for index in range(ep)],
        "moe_dp": [dp_base + index * ep * moe_tp for index in range(dp)],
    }


def _load_case(path: Path, platform: str, root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    assert payload["status"] == "passed", path
    topology = payload["topology"]
    tp = int(topology["tp"])
    ep = int(topology["ep"])
    dp = int(topology["moe_dp"])
    assert tp % (ep * dp) == 0, path
    moe_tp = tp // (ep * dp)
    ranks = payload["ranks"]
    assert len(ranks) == tp, path
    assert sorted(record["rank"] for record in ranks) == list(range(tp)), path

    for record in ranks:
        rank = int(record["rank"])
        assert record["status"] == "passed", (path, rank)
        assert record["groups"] == _expected_groups(rank, topology), (path, rank)
        assert record["transport"]["metadata_contract"] == METADATA_CONTRACT
        assert (
            record["transport"]["global_to_local_boundary"]
            == "after_EP_dispatch_before_local_weight_index"
        )
        evidence = record["collective_evidence"]
        assert evidence["ep_all_to_all_world_size"] == ep
        assert evidence["moe_tp_all_reduce_world_size"] == moe_tp
        assert evidence["moe_dp_all_gather_world_size"] == dp
        graph = record["graph"]
        assert graph["status"] == "passed", (path, rank, graph)
        assert graph["scope"] == "post_dispatch_local_compute"
        assert graph["collectives_in_graph"] is False
        route_mode = payload["route_mode"]
        if ep > 1 and route_mode == "all_local":
            assert record["transport"]["nonlocal_send_pairs"] == 0
        elif ep > 1 and route_mode == "no_local":
            assert record["transport"]["self_send_pairs"] == 0
        elif ep > 1 and route_mode == "mixed":
            assert record["transport"]["self_send_pairs"] > 0
            assert record["transport"]["nonlocal_send_pairs"] > 0

    return {
        "platform": platform,
        "artifact": str(path.relative_to(root)),
        "device": payload["device"],
        "torch_version": payload["torch_version"],
        "cuda_version": payload["cuda_version"],
        "topology": topology["name"],
        "tp": tp,
        "ep": ep,
        "moe_dp": dp,
        "moe_tp": moe_tp,
        "route_mode": payload["route_mode"],
        "adapter_mode": payload["adapter_mode"],
        "rank_reports": len(ranks),
        "hosts": sorted({record["host"] for record in ranks}),
        "graph_passes": sum(record["graph"]["status"] == "passed" for record in ranks),
        "max_abs": max(record["oracle"]["max_abs"] for record in ranks),
        "max_rel_l2": max(record["oracle"]["rel_l2"] for record in ranks),
        "min_cosine": min(record["oracle"]["cosine"] for record in ranks),
        "self_send_pairs": sum(
            record["transport"]["self_send_pairs"] for record in ranks
        ),
        "nonlocal_send_pairs": sum(
            record["transport"]["nonlocal_send_pairs"] for record in ranks
        ),
        "rank_host_map": {str(record["rank"]): record["host"] for record in ranks},
        "rank_groups": {str(record["rank"]): record["groups"] for record in ranks},
        "command": payload["command"],
    }


def _expected_matrix() -> dict[str, set[tuple[str, str, str]]]:
    return {
        "h200_single_node_4rank": {
            *(("tp4_ep1_dp1", "standard", adapter) for adapter in ADAPTER_MODES),
            *(("tp4_ep1_dp4", "standard", adapter) for adapter in ADAPTER_MODES),
            *(
                ("tp4_ep4_dp1", route, adapter)
                for route in EP_ROUTE_MODES
                for adapter in ADAPTER_MODES
            ),
        },
        "h200_single_node_8rank": {
            *(
                ("tp8_ep2_dp2", route, adapter)
                for route in EP_ROUTE_MODES
                for adapter in ADAPTER_MODES
            )
        },
        "gb300_two_node_8rank": {
            *(
                ("tp8_ep2_dp2", route, adapter)
                for route in EP_ROUTE_MODES
                for adapter in ADAPTER_MODES
            )
        },
    }


def _load_matrix(root: Path) -> list[dict[str, Any]]:
    sources = {
        "h200_single_node_4rank": root / "raw/h200/matrix",
        "h200_single_node_8rank": root / "raw/h200/tp8_matrix",
        "gb300_two_node_8rank": root / "raw/gb300_2node",
    }
    cases: list[dict[str, Any]] = []
    expected = _expected_matrix()
    for platform, directory in sources.items():
        paths = sorted(directory.glob("*.json"))
        if platform == "gb300_two_node_8rank":
            paths = [path for path in paths if path.name.endswith("_2node.json")]
        records = [_load_case(path, platform, root) for path in paths]
        actual = {
            (record["topology"], record["route_mode"], record["adapter_mode"])
            for record in records
        }
        assert actual == expected[platform], (platform, actual ^ expected[platform])
        assert len(records) == len(actual), (platform, "duplicate cases")
        cases.extend(records)
    return cases


def _trace_summary(root: Path) -> dict[str, Any]:
    trace_dir = root / "raw/gb300_2node/traces"
    trace_case_path = trace_dir / "tp8_ep2_dp2_no_local_active_trace.json"
    trace_case = _load_case(trace_case_path, "gb300_two_node_nsys", root)
    assert trace_case["topology"] == "tp8_ep2_dp2"
    assert trace_case["route_mode"] == "no_local"
    assert trace_case["adapter_mode"] == "active"

    kernel_rows = []
    for path in sorted(trace_dir.glob("*_stats_cuda_gpu_kern_sum.csv")):
        node = "node0" if "node0" in path.name else "node1"
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                if row["Name"].startswith("ncclDevKernel_"):
                    kernel_rows.append(
                        {
                            "node": node,
                            "name": row["Name"].split("(", 1)[0],
                            "instances": int(row["Instances"]),
                            "total_time_ns": int(row["Total Time (ns)"]),
                        }
                    )
    required = ("AllReduce", "SendRecv", "AllGather")
    for name in required:
        assert any(name in row["name"] for row in kernel_rows), name

    logs = sorted(trace_dir.glob("*_nsys_log.txt"))
    p2p_lines: list[str] = []
    clique_lines: list[str] = []
    for path in logs:
        for line in path.read_text(errors="replace").splitlines():
            if "via P2P/MNNVL" in line:
                p2p_lines.append(line)
            if "MNNVL 1 cliqueId" in line:
                clique_lines.append(line)
    assert len(p2p_lines) > 0
    assert len(clique_lines) > 0

    def samples(lines: list[str], count: int) -> list[str]:
        unique: list[str] = []
        seen = set()
        for line in lines:
            suffix = line.split("NCCL INFO ", 1)[-1]
            if suffix not in seen:
                unique.append(line)
                seen.add(suffix)
            if len(unique) == count:
                break
        return unique

    return {
        "case": trace_case,
        "reports": [
            str(path.relative_to(root)) for path in sorted(trace_dir.glob("*.nsys-rep"))
        ],
        "kernel_rows": kernel_rows,
        "p2p_mnnvl_line_count": len(p2p_lines),
        "mnnvl_clique_line_count": len(clique_lines),
        "p2p_mnnvl_samples": samples(p2p_lines, 12),
        "mnnvl_clique_samples": samples(clique_lines, 8),
    }


def _full_server_summary(root: Path) -> dict[str, Any]:
    directory = root / "full_server/h200"
    base = json.loads((directory / "base_response.json").read_text())
    adapter = json.loads((directory / "adapter_response.json").read_text())
    base_after = json.loads(
        (directory / "base_after_adapter_response.json").read_text()
    )
    mixed = json.loads((directory / "mixed_batch_response.json").read_text())
    assert base["output_ids"] == base_after["output_ids"]
    assert adapter["output_ids"] != base["output_ids"]
    assert len(mixed) == 2
    assert mixed[0]["output_ids"] == base["output_ids"]
    assert mixed[1]["output_ids"] == adapter["output_ids"]
    server_log = directory / "tp4_ep4_server_triton_attention.txt"
    log_text = server_log.read_text(errors="replace")
    for marker in (
        "sgl_lora execution is enabled",
        "tp_size=4",
        "ep_size=4",
        "lora_execution_engine='sgl_lora'",
        "lora_use_virtual_experts=True",
        "loaded weights for target modules ['down_proj', 'gate_up_proj']",
    ):
        assert marker in log_text, marker
    return {
        "status": "passed",
        "model": "Qwen/Qwen1.5-MoE-A2.7B",
        "topology": "TP4/EP4",
        "attention_backend": "triton",
        "lora_backend": "csgmv",
        "lora_execution_engine": "sgl_lora",
        "virtual_experts": True,
        "cuda_graph": False,
        "base_before_output_ids": base["output_ids"],
        "adapter_output_ids": adapter["output_ids"],
        "base_after_output_ids": base_after["output_ids"],
        "mixed_base_output_ids": mixed[0]["output_ids"],
        "mixed_adapter_output_ids": mixed[1]["output_ids"],
        "base_stable_across_adapter_transition": True,
        "mixed_batch_matches_individual_requests": True,
        "first_adapter_request_includes_jit_warmup": True,
        "flashinfer_control": (
            "The first launch stopped before model execution because installed "
            "flashinfer_python 0.6.12 was below required 0.6.15; the retained "
            "successful run selected Triton attention explicitly."
        ),
    }


def _write_cases_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = [
        "platform",
        "device",
        "topology",
        "tp",
        "ep",
        "moe_dp",
        "moe_tp",
        "route_mode",
        "adapter_mode",
        "rank_reports",
        "graph_passes",
        "max_abs",
        "max_rel_l2",
        "min_cosine",
        "self_send_pairs",
        "nonlocal_send_pairs",
        "hosts",
        "artifact",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            row = {field: case[field] for field in fields}
            row["hosts"] = ";".join(case["hosts"])
            writer.writerow(row)


def _platform_summary(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for platform in sorted({case["platform"] for case in cases}):
        selected = [case for case in cases if case["platform"] == platform]
        rows.append(
            {
                "platform": platform,
                "device": selected[0]["device"],
                "cases": len(selected),
                "rank_reports": sum(case["rank_reports"] for case in selected),
                "graph_passes": sum(case["graph_passes"] for case in selected),
                "topologies": sorted({case["topology"] for case in selected}),
                "route_modes": sorted({case["route_mode"] for case in selected}),
                "adapter_modes": sorted({case["adapter_mode"] for case in selected}),
                "hosts": sorted({host for case in selected for host in case["hosts"]}),
                "worst_max_abs": max(case["max_abs"] for case in selected),
                "worst_rel_l2": max(case["max_rel_l2"] for case in selected),
                "minimum_cosine": min(case["min_cosine"] for case in selected),
            }
        )
    return rows


def _write_topology_evidence(path: Path, trace: dict[str, Any]) -> None:
    case = trace["case"]
    lines = [
        "D0 two-node GB300 topology evidence",
        "===================================",
        "",
        "Physical rank placement:",
    ]
    for rank, host in sorted(
        case["rank_host_map"].items(), key=lambda item: int(item[0])
    ):
        lines.append(f"  rank {rank}: {host}")
    lines.extend(
        [
            "",
            "Rank-0 SGL groups:",
            *(f"  {name}: {ranks}" for name, ranks in case["rank_groups"]["0"].items()),
            "",
            f"NCCL 'via P2P/MNNVL' lines in retained Nsight logs: "
            f"{trace['p2p_mnnvl_line_count']}",
            f"NCCL MNNVL clique lines in retained Nsight logs: "
            f"{trace['mnnvl_clique_line_count']}",
            "",
            "MNNVL clique samples:",
            *(f"  {line}" for line in trace["mnnvl_clique_samples"]),
            "",
            "P2P/MNNVL channel samples:",
            *(f"  {line}" for line in trace["p2p_mnnvl_samples"]),
            "",
            "Nsight CUDA kernel rows:",
            *(
                "  {node}: {name}, instances={instances}, total_time_ns={total_time_ns}".format(
                    **row
                )
                for row in trace["kernel_rows"]
            ),
            "",
            "Interpretation:",
            "  EP groups are intra-node under contiguous torchrun rank placement.",
            "  MoE-DP groups [0,4], [1,5], [2,6], [3,7] cross the physical nodes.",
            "  The trace therefore proves real inter-node MNNVL transport for the",
            "  DP all-gather, while SendRecv proves the real EP all-to-all path.",
            "  MoE-TP all-reduce groups are intra-node in this topology.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _markdown(summary: dict[str, Any]) -> str:
    platform_rows = []
    for row in summary["platforms"]:
        display = {**row, "topologies": ", ".join(row["topologies"])}
        platform_rows.append(
            "| {platform} | {device} | {topologies} | {cases} | {rank_reports} | "
            "{graph_passes}/{rank_reports} | {worst_rel_l2:.7g} | "
            "{minimum_cosine:.9f} |".format(**display)
        )
    trace = summary["nsight_trace"]
    server = summary["full_server"]
    return f"""# Distributed D0 validation: SGL LoRA MoE

## Outcome

All {summary['case_count']} planned matrix cases passed across
{summary['rank_report_count']} rank reports. Every rank passed the fixed-shape
post-dispatch CUDA-graph replay, every case matched the independent full-weight
oracle, and the two-node GB300 run retained both per-node Nsight Systems reports
and readable NCCL summaries. This is correctness and topology evidence; the cold
diagnostic timings in raw JSON are deliberately not performance claims.

| Platform | Device | Topologies | Cases | Rank reports | Graph passes | Worst rel-L2 | Min cosine |
|:---|:---|:---|---:|---:|---:|---:|---:|
{chr(10).join(platform_rows)}

## What the harness actually executes

```mermaid
flowchart LR
    A["DP-owned BF16 tokens + GLOBAL expert IDs"] --> B["Real NCCL EP all_to_all_single"]
    B --> C["Ownership check; GLOBAL to LOCAL expert ID"]
    C --> D["Local expert buffers + serving SGL virtual-expert gate/up LoRA A+B"]
    D --> E["BF16 base gate/up + SwiGLU"]
    E --> F["Serving SGL virtual-expert down LoRA A+B + BF16 base down"]
    F --> G["Real SGL MoE-TP all-reduce"]
    G --> H["Real reverse EP all_to_all_single"]
    H --> I["Top-k weighted reduction in origin-token domain"]
    I --> J["Real SGL MoE-DP padded all-gather"]
```

The transported metadata is exactly `global_token_id`, `topk_slot`,
`global_expert_id`, `adapter_id`, and `origin_ep_rank`. Global IDs remain attached
through dispatch. Conversion to local expert IDs happens once, after ownership is
known and before local-buffer indexing. Local expert weights have `E / EP` expert
rows; gate/up and down intermediate dimensions are sharded by MoE-TP.

The independent oracle uses the unsharded full weights and explicitly recreates
the BF16 boundaries of each MoE-TP partial. It does not call the serving LoRA
kernels under test.

## Coverage

- H200, TP4/EP1/MoE-DP1: standard routing, active/mixed/base adapters.
- H200, TP4/EP4/MoE-DP1: all-local/mixed/no-local routing crossed with
  active/mixed/base adapters.
- H200, TP4/EP1/MoE-DP4: uneven DP token domains `(1, 3, 7, 13)`, crossed with
  active/mixed/base adapters.
- H200, TP8/EP2/MoE-DP2: all-local/mixed/no-local crossed with
  active/mixed/base adapters; uneven DP token domains `(5, 11)`.
- Two physical GB300 nodes, TP8/EP2/MoE-DP2: the same nine route/adapter cells.
- Rank-16, two adapters, top-k 2, BF16, eight global experts. Rank 8/16 broader
  performance guardrails and quantized providers are separate plan lanes; D0 is
  the distributed contract lane.

`cases.csv` has one row per matrix cell. `summary.json` retains every rank's host
and exact SGL group membership.

## CUDA-graph claim and collective boundary

Each rank prewarms route metadata, captures only the fixed-shape local compute
from post-dispatch gate/up through down output, poisons the destination, and
replays once before comparing with eager output. All
{summary['graph_pass_count']} rank captures passed.

NCCL collectives are intentionally outside that graph:

- EP variable-split forward/reverse all-to-all is eager;
- MoE-TP all-reduce is eager;
- uneven MoE-DP padded all-gather is eager.

The report field `collectives_in_graph=false` prevents this evidence from being
misread as a full distributed-pipeline graph capture.

## Two-node GB300 topology and Nsight evidence

Torchrun placed ranks 0-3 on `rdx-gb300-r01-c012.rdx.local` and ranks 4-7 on
`rdx-gb300-r01-c018.rdx.local`. With SGL's rank order, EP and MoE-TP groups are
within a physical node, while MoE-DP groups `[0,4]`, `[1,5]`, `[2,6]`, and
`[3,7]` cross nodes. Thus the EP all-to-all is real NCCL but intra-node in this
specific placement; the MoE-DP all-gather is the exercised inter-node collective.

The retained two-node no-local/active trace passed graph replay and the oracle.
Its logs contain {trace['p2p_mnnvl_line_count']} `via P2P/MNNVL` channel lines and
{trace['mnnvl_clique_line_count']} MNNVL clique lines. The per-node CUDA kernel
summaries contain NCCL `SendRecv`, `AllReduce`, and `AllGather` kernels. The
binary `.nsys-rep` files are the authoritative timeline; `topology_evidence.txt`
extracts the rank placement, representative channel lines, and kernel totals.

NCCL may print `nNodes 1` for an eight-GPU MNNVL clique even though the rank hosts
above are two physical GB300 nodes. That is NCCL's fabric-clique topology view,
not evidence that the launch collapsed to one host.

## Full-server smoke

A real `Qwen/Qwen1.5-MoE-A2.7B` server ran TP4/EP4 on H200 with
`lora_execution_engine=sgl_lora`, virtual experts, the csgmv LoRA backend, and a
Qwen MoE adapter targeting `gate_up_proj` and `down_proj`.

- base -> adapter -> base returned stable base IDs around the adapter transition;
- a two-request mixed base/adapter batch matched the two individual results;
- all requests returned HTTP 200 and the adapter loaded on TP0/EP0 through
  TP3/EP3;
- the first adapter latency includes JIT compilation and is not a latency sample.

The initial launch selected the installed FlashInfer attention package and stopped
before model execution because version 0.6.12 was below the required 0.6.15. The
successful control selected Triton attention explicitly. Server CUDA graphs were
disabled; graph evidence belongs to the distributed contract harness described
above.

Base output IDs: `{server['base_before_output_ids']}`

Adapter output IDs: `{server['adapter_output_ids']}`

## Reproduction

Single-node example:

```bash
PYTHONPATH=python torchrun --standalone --nproc-per-node=4 \\
  benchmark/kernels/lora_moe/bench_distributed.py \\
  --topology tp4_ep4_dp1 --route-mode mixed --adapter-mode mixed \\
  --graph-replay --output /tmp/tp4_ep4_dp1_mixed_mixed.json
```

Two-node execution used one `srun` task per node, then `torchrun --nnodes=2
--nproc-per-node=4` with node ranks from `SLURM_NODEID`. The no-local/active trace
wrapped that torchrun command with:

```bash
nsys profile --trace=cuda,nvtx,nccl --sample=none --cpuctxsw=none \\
  --force-overwrite=true --output=<per-node-output> torchrun <two-node-args>
```

Regenerate the bundle summaries after placing raw artifacts under this directory:

```bash
python benchmark/kernels/lora_moe/summarize_distributed_d0.py \\
  --root benchmark_results/sgl_lora_moe_20260722/distributed_d0
```

The exact GPU-executed harness snapshot SHA256 is
`{summary['source_provenance']['executed_harness_sha256']}`. It was checked
byte-for-byte against the isolated H200 source and the shared two-node GB300
source before the allocation was released. The repository harness is formatted
with the current project hooks; the summarizer verifies that its parsed Python
AST is identical to the retained executed snapshot.

## Limits and interpretation

- This is not an end-to-end throughput benchmark. Setup, first-JIT, and profiler
  overhead dominate the recorded timing fields.
- The harness uses real SGL process groups and serving-owned virtual-expert LoRA
  kernels, but it constructs an explicit PyTorch NCCL EP dispatcher to make the
  global/local-ID boundary directly auditable. It does not claim to be the final
  production dispatcher implementation.
- The full server uses SGL's normal replicated-input EP path (`moe_a2a_backend=none`)
  rather than the harness's explicit A2A transport.
- DeepEP/FlashInfer A2A, graph-captured collectives, lifecycle/eviction, shared
  experts, TP8/EP layouts with EP spanning physical nodes, and quantized providers
  remain separate validation lanes.
- `SHA256SUMS` covers every retained artifact except itself. `MANIFEST.json`
  records file sizes and hashes; ignored binary Nsight reports must be added with
  `git add -f` when publishing this evidence directory.
"""


def _manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.json", "SHA256SUMS"}:
            continue
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    (root / "MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "files": entries}, indent=2) + "\n"
    )
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{_sha256(path)}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[3]
    harness = repo / "benchmark/kernels/lora_moe/bench_distributed.py"
    executed_snapshot = root / "source/bench_distributed_gpu_executed.py.txt"
    harness_hash = _sha256(harness)
    executed_hash = _sha256(executed_snapshot)
    assert executed_hash == EXECUTED_HARNESS_SHA256
    harness_ast = ast.dump(ast.parse(harness.read_text()), include_attributes=False)
    executed_ast = ast.dump(
        ast.parse(executed_snapshot.read_text()), include_attributes=False
    )
    assert harness_ast == executed_ast, "repository harness differs semantically"

    cases = _load_matrix(root)
    trace = _trace_summary(root)
    server = _full_server_summary(root)
    platforms = _platform_summary(cases)
    source_provenance = {
        "isolated_execution_tree": "sglang-distributed-d0-8e4114",
        "execution_base_revision_short": "8e4114e2c0",
        "harness_path": "benchmark/kernels/lora_moe/bench_distributed.py",
        "harness_sha256": harness_hash,
        "executed_snapshot_path": ("source/bench_distributed_gpu_executed.py.txt"),
        "executed_harness_sha256": executed_hash,
        "repository_harness_ast_matches_executed_snapshot": True,
        "remote_hash_checks": {
            "h200:/sgl-workspace/sglang-distributed-d0-8e4114": executed_hash,
            "gb300:/data/home/yjiang/sglang-distributed-d0-8e4114": executed_hash,
        },
    }
    summary = {
        "schema_version": 1,
        "status": "passed",
        "scope": "distributed_d0_sgl_lora_moe_contract",
        "case_count": len(cases),
        "rank_report_count": sum(case["rank_reports"] for case in cases),
        "graph_pass_count": sum(case["graph_passes"] for case in cases),
        "platforms": platforms,
        "matrix": cases,
        "nsight_trace": trace,
        "full_server": server,
        "source_provenance": source_provenance,
        "assertions": {
            "planned_matrix_complete": len(cases) == 33,
            "all_rank_statuses_passed": True,
            "all_local_graph_replays_passed": True,
            "independent_oracle_passed": True,
            "global_to_local_boundary_verified": True,
            "real_ep_all_to_all_executed": True,
            "real_moe_tp_all_reduce_executed": True,
            "real_moe_dp_all_gather_executed": True,
            "two_physical_gb300_hosts_verified": len(
                next(
                    row["hosts"]
                    for row in platforms
                    if row["platform"] == "gb300_two_node_8rank"
                )
            )
            == 2,
            "nsight_contains_nccl_sendrecv_allreduce_allgather": True,
            "full_server_base_adapter_mixed_passed": True,
        },
    }
    assert all(summary["assertions"].values())
    _write_cases_csv(root / "cases.csv", cases)
    _write_topology_evidence(root / "topology_evidence.txt", trace)
    (root / "source_provenance.json").write_text(
        json.dumps(source_provenance, indent=2, sort_keys=True) + "\n"
    )
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (root / "README.md").write_text(_markdown(summary))
    _manifest(root)
    print(
        f"validated {len(cases)} cases / {summary['rank_report_count']} rank "
        f"reports; wrote {root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
