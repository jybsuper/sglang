# Production BF16 MoE-LoRA planner validation

This bundle records the final graduation check for the measured C2/C3 BF16
MoE-LoRA execution plans.  It covers the real Qwen1.5-MoE server lifecycle on
H200 and GB300: model and adapter load, CUDA-graph capture, base-only requests,
adapter requests, mixed base/adapter rows, prompt processing, and decode.

## Production policy

The planner consumes explicit forward phase, graph mode, token count, rank,
and whether the admitted batch can contain base rows.  It never guesses
prefill versus decode from token count.

| Forward | Condition | Default plan | Two-stream request |
|---|---|---|---|
| Captured decode | `T <= 128`, BF16, virtual experts, rank 1-128 | C2F | C3 |
| Captured decode | `T > 128` | C2P | C2P |
| Eager decode | `T <= 256`, BF16, virtual experts, rank 1-128 | C2F | C2F |
| Eager decode | `T > 256` | C2P | C2P |
| Prefill / extend | measured BF16 envelope | C2P | C2P |
| Other phase/provider, non-virtual experts, fully sharded, or rank outside 1-128 | C0 | C0 |

C2P is the scalable partial-fusion tail. C2F adds the complete fused down
finalizer. C3 is C2F plus overlap of gate/up LoRA-A shrink with base prepare
and gate/up GEMM. C3 remains opt-in because its model-level advantage is
strongest at small and medium decode batches and is not a portable eager or
large-token rule.

## End-to-end setup

- Model: `Qwen/Qwen1.5-MoE-A2.7B`.
- Adapter: `jonahbernard/sglang-lora-moe-test-qwen1.5-MoE-A2.7B`.
- Execution: TP1, EP1, MoE-DP1, BF16, virtual experts, SGL-LoRA engine.
- Graph: full decode CUDA graph through batch size 32; prefill graph disabled
  by the existing LoRA/breakable-graph compatibility rule.
- Workload: `bench_one_batch_server`, prompt length 128, output length 32,
  batches 1/16/32.
- Sampling: three complete repetitions for adapter and default C2 base paths.
  C3 has three adapter repetitions; its GB300 base control has one repetition.
- Ordering: base correctness, adapter correctness, mixed base/adapter
  correctness, rotated throughput runs, then base correctness again.
- Devices: NVIDIA H200 GPU 5 and NVIDIA GB300 GPU 1, each using an isolated
  source tree. No multi-GPU communication result is implied by this test.

The C2 GB300 launch used the default graph tiers through 32, which also include
12 and 24. The C3 GB300 and both H200 launches used explicit tiers
1/2/4/8/16/32. Every reported batch has an exact captured tier in both paths.

## Median throughput

Input and output values are tokens/s. Input throughput is the prompt-processing
measurement; output throughput is decode.

### GB300

| Path | Metric | BS1 | BS16 | BS32 |
|---|---|---:|---:|---:|
| C2 default | LoRA input | 2,073.94 | 28,024.23 | 50,797.83 |
| C2 default | LoRA output | 368.38 | 3,142.36 | 5,318.59 |
| C3 opt-in | LoRA input | 2,045.53 | 28,291.07 | 53,194.60 |
| C3 opt-in | LoRA output | 385.89 | 3,277.41 | 5,313.18 |
| C2 default | base output | 456.42 | 3,437.42 | 5,915.09 |
| C3 opt-in control | base output | 457.77 | 3,466.10 | 5,918.58 |

Compared with the matched pre-planner SGL-LoRA result, default C2 improves
decode by **6.19% / 5.92% / 2.47%** at BS1/16/32. Prompt throughput changes by
**+5.50% / +0.61% / +1.52%**, so the graduated default has no observed prefill
regression in this workload. C3 improves decode over C2 by
**4.75% / 4.30% / -0.10%**; that supports keeping C3 opt-in and shape-aware.

The optimized experimental TRTLLM BF16 reference remains faster in decode.
Default C2 trails it by 9.43-10.39%; C3 narrows the gap to 5.12-5.54% at BS1/16
but remains 10.48% behind at BS32. The planner closes a real part of the gap,
not the entire backend gap.

### H200

| Path | Metric | BS1 | BS16 | BS32 |
|---|---|---:|---:|---:|
| C2 default | LoRA input | 2,217.56 | 32,831.34 | 61,218.24 |
| C2 default | LoRA output | 315.17 | 2,230.93 | 3,633.45 |
| C3 opt-in | LoRA input | 2,347.36 | 34,059.78 | 61,094.11 |
| C3 opt-in | LoRA output | 322.32 | 2,248.50 | 3,719.50 |
| C2 default | base output | 398.05 | 2,512.92 | 4,170.75 |

C3 improves H200 decode over C2 by **2.27% / 0.79% / 2.37%**. The earlier
H200 SGL result was a single smoke repetition, so it is retained only as a
non-authoritative check; the three-repetition C2/C3 comparison above is the
decision evidence.

## Correctness and integration result

Base, adapter, and mixed requests produce the expected token-ID sequences on
both architectures. Base-before and base-after IDs match, and the mixed batch
contains the independently expected base and adapter sequences. This checks
that graph replay does not leak adapter state into base rows.

The final integrated suite passed on both architectures:

```text
36 passed on NVIDIA H200
36 passed on NVIDIA GB300
```

The suite covers the host planner, typed quant provider boundaries, unified
runner, C2 finalizer, C3 event/stream ownership, rank 8 and 16, all-active and
mixed base rows, BF16 and FP32 output destinations, eager execution, and CUDA
graph replay.

The `server.log` files end with scheduler exit `-15` because the benchmark
servers were deliberately terminated after evidence collection. No request,
capture, or execution crash occurred during the measured runs.

## Production Nsight Systems trace

`nsys_h200/production_planner_c2_c3_h200.nsys-rep` is an exclusive capture of
the final production plan executor, not an experimental runner copy. It uses
the Qwen3.5-35B-A3B local MoE shape at `T=32`, rank 64, and captured decode.
Ten C2F and ten C3 graph replays are marked separately with NVTX ranges.
Eager-to-graph maximum absolute error is `1.220703125e-4` for each path. C2F
owns no graph-scoped event; C3 owns exactly one, as required by its join.

The trace contains 220 kernels and exactly two streams. C3 produces 150.463 us
of overlap between the primary stream pair across ten replays, or 6.04% of the
shorter stream's busy interval. C2F and C3 use the same production fused
consumer/finalizer kernels; the extra stream is confined to C3 producer
overlap. The kernel summary shows 20 invocations each of the fused gate-B plus
activation plus down-A consumer and fused down-B finalizer, proving one of each
per captured replay across the two paths.

Capture command (after copying `profile_driver.py` to the repository root):

```text
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=python:. nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --force-overwrite=true \
  -o production_planner_c2_c3_h200 \
  python3 profile_driver.py
```

## Baselines and conclusions

The authoritative matched GB300 baselines (BS1/16/32) were:

| Backend | Input tok/s | Output tok/s |
|---|---|---|
| Pre-planner SGL-LoRA | 1,965.80 / 27,853.65 / 50,039.21 | 346.90 / 2,966.59 / 5,190.14 |
| Experimental TRTLLM BF16 | not used for the prefill decision | 406.73 / 3,469.76 / 5,935.44 |

The production decision is therefore:

1. Use C2 as the default measured BF16 execution family.
2. Use C2P for prefill and large captured decode to avoid the known large-T
   complete-finalizer regression.
3. Expose C3 only through the existing explicit two-stream request and only
   for captured decode through 128 tokens.
4. Keep C0 for contracts outside the measured fused envelope.
5. Continue backend optimization: dense LoRA launches and base DeepGEMM work
   still dominate the remaining 5-10% GB300 gap to the TRTLLM reference.

## Artifact map

- `raw/gb300_c2`, `raw/h200_c2`: default C2 server logs, three LoRA runs,
  three base runs, and base/adapter/mixed correctness responses.
- `raw/gb300_c3`, `raw/h200_c3`: opt-in C3 server logs, three LoRA runs, and
  correctness responses; GB300 also includes a base control.
- `summary.json`: machine-readable medians, comparisons, policy, and test
  result.
- `nsys_h200`: final production C2F/C3 trace, SQLite export, kernel/NVTX CSV
  summaries, overlap analysis, exact driver, and correctness/resource result.

The server logs are the authoritative launch-configuration record, including
resolved CUDA graph tiers and all SGLang arguments. Throughput JSONL files are
the unmodified `bench_one_batch_server` outputs.
