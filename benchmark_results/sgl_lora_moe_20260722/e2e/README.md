# Model-level LoRA evidence

This directory contains real-server checks for
`Qwen/Qwen1.5-MoE-A2.7B` with adapter
`jonahbernard/sglang-lora-moe-test-qwen1.5-MoE-A2.7B`.  The controlled tool is
`sglang.benchmark.one_batch_server`; `bench_serving` is deliberately not used.

## Authoritative GB300 bracket

`experimental_trtllm_gb300_v4/` and `sgl_dual_gb300_repeat/` are a same-GPU,
counterbalanced bracket at input/output lengths 128/32 and batch sizes 1/16/32.
Each traffic class was measured three times.  Both servers passed
base -> adapter -> base and simultaneous mixed base/adapter token-ID checks.

The control required two compatibility observations:

1. current FlashInfer adds `permutedIdxToBiasRowIdx` to the generic batched-GEMM
   runner ABI; commit `29b1ad7581` supplies the null pointer required by this BF16
   control;
2. the experimental backend is SM100-only in this source tree and therefore is
   not a runnable H200 control.

Median output-throughput results are:

| Traffic | BS | SGL-LoRA | experimental TRTLLM | SGL difference |
|---|---:|---:|---:|---:|
| base | 1 | 448.40 | 421.15 | +6.47% |
| base | 16 | 3451.40 | 3502.64 | -1.46% |
| base | 32 | 5914.83 | 5894.69 | +0.34% |
| adapter | 1 | 346.90 | 406.73 | -14.71% |
| adapter | 16 | 2966.59 | 3469.76 | -14.50% |
| adapter | 32 | 5190.14 | 5935.44 | -12.56% |

This directly answers the earlier gating question: the current production SGL
path is **not** yet faster than the optimized experimental TRTLLM adapter path at
all tested token counts.  Base-only traffic is at parity or better except the
small BS16 difference, while adapter decode remains about 12.6-14.7% behind.
The result is a target for the production execution planner and fused consumers,
not a reason to discard the cleaner provider boundary.

The matched input-throughput result is less concerning after warm-up: SGL is
`+20.0/+2.4/+0.1%` at BS1/16/32.  Earlier single-run H200 SGL adapter TTFT values
were dominated by first-use compilation and are retained but not used as a
performance conclusion.

## Trace interpretation

`bench_one_batch_server --profile --profile-by-stage` produced independent
EXTEND and DECODE traces.  Unprofiled repetitions above remain authoritative;
the traces establish topology and locate costs.

- SGL emitted base and adapter traces in both stages without crashing.
- Its base decode uses the dedicated no-LoRA graph and stock `fused_moe_kernel`.
- Its adapter decode shows the expected DeepGEMM gate/down pair, two MoE A and B
  launches per layer, activation/fill/reorder kernels, and ordinary dense LoRA
  shrink/expand launches.  The trace therefore explains why graph separation
  removes base tax without by itself closing active-adapter latency.
- The experimental control emitted a valid base trace.  Starting the adapter
  profile then crashed its scheduler with SIGSEGV/exit `-11`; the server log and
  incomplete profile directory are retained.  Its unprofiled adapter timing and
  correctness completed before the crash and remain valid.

`gb300_matched_summary.json` stores every sample, p20/p80/median metrics, token-ID
checks, crash markers, trace launch counts, and aggregated top-kernel durations.
Regenerate it with:

```bash
python3 benchmark/kernels/lora_moe/analyze_e2e_model.py \
  --sgl-dir benchmark_results/sgl_lora_moe_20260722/e2e/sgl_dual_gb300_repeat \
  --control-dir benchmark_results/sgl_lora_moe_20260722/e2e/experimental_trtllm_gb300_v4 \
  --json-output benchmark_results/sgl_lora_moe_20260722/e2e/gb300_matched_summary.json \
  --markdown-output benchmark_results/sgl_lora_moe_20260722/e2e/gb300_matched_summary.md
```

## H200 directories

`h200_dual_graph/` validates the separate no-LoRA/LoRA decode graph key and
base/adapter/mixed replay.  `h200_legacy_control/` is the stock legacy-LoRA
comparison.  Those first measurements contain one repetition per cell and are
useful correctness/smoke evidence only; they are not promoted to the same level
as the counterbalanced GB300 bracket.  A production-planner graduation run must
repeat them in both orders.
