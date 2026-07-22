#!/usr/bin/env python3
"""Run the auditable quantized SGL-LoRA production-plan matrix.

Each cell is a fresh process so provider setup, JIT failures, and allocator
state cannot contaminate later results.  Setup is excluded from the CUDA-event
latency inside ``bench_quantized_moe_pipeline.py`` but retained in every JSON.
The runner continues after failures and writes one manifest at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = REPO_ROOT / "benchmark/kernels/lora_moe/bench_quantized_moe_pipeline.py"


@dataclass(frozen=True, slots=True)
class MatrixCell:
    key: str
    tokens: int
    experts: int
    top_k: int
    hidden: int
    intermediate: int
    rank: int
    occupancy: str
    phase: str
    execution: str
    output_dtype: str = "fp32"
    model: str = "qwen3.5-35b-a3b"


def _full_cells() -> list[MatrixCell]:
    cells = []
    for rank in (16, 32, 64, 128):
        for occupancy in ("active", "mixed", "base"):
            for execution in ("eager", "cuda_graph"):
                cells.append(
                    MatrixCell(
                        key=f"qwen-decode-r{rank}-{occupancy}-{execution}",
                        tokens=32,
                        experts=8,
                        top_k=8,
                        hidden=2048,
                        intermediate=512,
                        rank=rank,
                        occupancy=occupancy,
                        phase="decode",
                        execution=execution,
                    )
                )
    cells.append(
        MatrixCell(
            key="qwen-decode-r64-active-graph-bf16-dst",
            tokens=32,
            experts=8,
            top_k=8,
            hidden=2048,
            intermediate=512,
            rank=64,
            occupancy="active",
            phase="decode",
            execution="cuda_graph",
            output_dtype="bf16",
        )
    )
    cells.extend(
        MatrixCell(
                key=f"qwen-prefill-r64-mixed-{execution}",
                tokens=128,
                experts=8,
                top_k=8,
                hidden=2048,
                intermediate=512,
                rank=64,
                occupancy="mixed",
                phase="prefill",
                execution=execution,
        )
        for execution in ("eager", "cuda_graph")
    )
    cells.extend(
        [
            MatrixCell(
                key="kimi-k2.5-decode-r64-mixed-graph",
                tokens=16,
                experts=8,
                top_k=8,
                hidden=7168,
                intermediate=2048,
                rank=64,
                occupancy="mixed",
                phase="decode",
                execution="cuda_graph",
                model="kimi-k2.5",
            ),
            MatrixCell(
                key="glm-5.2-decode-r64-mixed-graph",
                tokens=16,
                experts=8,
                top_k=8,
                hidden=6144,
                intermediate=2048,
                rank=64,
                occupancy="mixed",
                phase="decode",
                execution="cuda_graph",
                model="glm-5.2",
            ),
        ]
    )
    return cells


def _representative_cells() -> list[MatrixCell]:
    full = _full_cells()
    selected = {
        # All occupancy and execution modes at the production core rank.
        *(
            f"qwen-decode-r64-{occupancy}-{execution}"
            for occupancy in ("active", "mixed", "base")
            for execution in ("eager", "cuda_graph")
        ),
        # Direct and runtime-general rank schedules.
        "qwen-decode-r16-active-cuda_graph",
        "qwen-decode-r32-active-cuda_graph",
        "qwen-decode-r128-mixed-cuda_graph",
        # Destination, phase, and large-model geometry anchors.
        "qwen-decode-r64-active-graph-bf16-dst",
        "qwen-prefill-r64-mixed-eager",
        "qwen-prefill-r64-mixed-cuda_graph",
        "kimi-k2.5-decode-r64-mixed-graph",
        "glm-5.2-decode-r64-mixed-graph",
    }
    return [cell for cell in full if cell.key in selected]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("h200", "gb300"), required=True)
    parser.add_argument(
        "--provider",
        choices=("all", "fp8", "nvfp4", "marlin"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=("full", "representative"), default="full"
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    supported = {
        "h200": ("fp8", "marlin"),
        "gb300": ("fp8", "nvfp4", "marlin"),
    }[args.device]
    providers = supported if args.provider == "all" else (args.provider,)
    unknown = [provider for provider in providers if provider not in supported]
    if unknown:
        raise SystemExit(
            f"{args.device} does not support matrix provider(s): {', '.join(unknown)}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records = []
    for provider in providers:
        provider_dir = output_dir / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        cells = (
            _full_cells() if args.suite == "full" else _representative_cells()
        )
        for cell in cells:
            result_path = provider_dir / f"{cell.key}.json"
            stdout_path = provider_dir / f"{cell.key}.stdout.log"
            stderr_path = provider_dir / f"{cell.key}.stderr.log"
            command = [
                sys.executable,
                str(BENCHMARK),
                "--provider",
                provider,
                "--tokens",
                str(cell.tokens),
                "--experts",
                str(cell.experts),
                "--top-k",
                str(cell.top_k),
                "--hidden",
                str(cell.hidden),
                "--intermediate",
                str(cell.intermediate),
                "--rank",
                str(cell.rank),
                "--occupancy",
                cell.occupancy,
                "--phase",
                cell.phase,
                "--execution",
                cell.execution,
                "--output-dtype",
                cell.output_dtype,
                "--warmups",
                str(args.warmups),
                "--iterations",
                str(args.iterations),
                "--json",
                str(result_path),
            ]
            cell_started = time.time()
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            stdout_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)
            records.append(
                {
                    "provider": provider,
                    "cell": asdict(cell),
                    "returncode": completed.returncode,
                    "status": "pass" if completed.returncode == 0 else "fail",
                    "result": str(result_path.relative_to(output_dir)),
                    "stdout": str(stdout_path.relative_to(output_dir)),
                    "stderr": str(stderr_path.relative_to(output_dir)),
                    "elapsed_seconds": time.time() - cell_started,
                }
            )
            print(
                f"[{len(records):03d}] {provider} {cell.key}: "
                f"{'pass' if completed.returncode == 0 else 'FAIL'}",
                flush=True,
            )

    artifact_hashes = {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    failures = [record for record in records if record["returncode"]]
    manifest = {
        "schema": "sgl_lora_quantized_production_matrix_v1",
        "device_class": args.device,
        "suite": args.suite,
        "providers": list(providers),
        "status": "pass" if not failures else "fail",
        "python": platform.python_version(),
        "benchmark": str(BENCHMARK.relative_to(REPO_ROOT)),
        "benchmark_sha256": _sha256(BENCHMARK),
        "matrix_is_full_cartesian_for": (
            [
                "qwen decode ranks 16/32/64/128",
                "active/mixed/base occupancy",
                "eager/cuda_graph",
            ]
            if args.suite == "full"
            else []
        ),
        "representative_coverage": (
            [
                "qwen rank64 active/mixed/base eager/cuda_graph",
                "ranks 16/32/128 graph anchors",
                "BF16 destination",
                "prefill eager/cuda_graph",
                "Kimi K2.5 and GLM-5.2 local-shape graph anchors",
            ]
            if args.suite == "representative"
            else []
        ),
        "additional_anchors": [
            "BF16 destination",
            "Qwen prefill eager/cuda_graph",
            "Kimi K2.5 local shape",
            "GLM-5.2 local shape",
        ],
        "explicit_model_gap": {
            "nemotron-3": (
                "relu2/non-gated activation is outside the current swiglu provider "
                "contract; running its dimensions with swiglu would not validate "
                "Nemotron semantics"
            )
        },
        "records": records,
        "failure_count": len(failures),
        "artifact_sha256": artifact_hashes,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
