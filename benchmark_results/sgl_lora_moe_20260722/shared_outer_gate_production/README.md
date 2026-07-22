# Shared-outer gate-A production promotion evidence

Date: 2026-07-22

This artifact validates the production static selector and segmented
token-deduplicated shared-outer gate/up LoRA-A kernel on an NVIDIA H200 and an
NVIDIA GB300.  It is intentionally a compact promotion matrix, not a repeat of
the larger exploratory shared-outer study.

## Compared region

Both variants execute the same matched gate-A boundary:

1. prepare the per-expert gate-B virtual-expert route needed by the downstream
   gate-B consumer;
2. produce the pair-major BF16 gate-A intermediate `[T, top_k, 2R]`.

The forced-generic control also prepares the shared-A virtual-expert route and
runs the repeated-pair grouped shrink.  The selected implementation skips that
route, computes `hidden @ shared_A.T` once per token/request segment, and
broadcasts the accumulator to every top-k slot in the same launch.  No runtime
tile descriptor, token sort, token-owned scratch tensor, or materialization
launch is hidden from the measurement.

Each timing is 20 warmups plus 200 measured iterations.  CUDA-graph numbers
are the primary launch-overhead-controlled comparison.  Eager numbers include
the same GPU route/allocation work but are more sensitive to host allocator
variance.

## Selected CUDA-graph results

`speedup` is `generic / selected - 1`; positive means token dedup is faster.

| Case | H200 generic / selected (us) | H200 speedup | GB300 generic / selected (us) | GB300 speedup |
|---|---:|---:|---:|---:|
| T32 H2048 R128, 8 adapter spans | 26.209 / 13.909 | +88.4% | 26.638 / 14.363 | +85.5% |
| T32 H2048 R128, 32 one-token segments | 26.372 / 15.303 | +72.3% | 26.636 / 16.399 | +62.4% |
| T32 H2048 R128, mixed adapter/base | 25.370 / 12.861 | +97.3% | 24.603 / 14.352 | +71.4% |
| T256 H2048 R64, 4 adapter spans | 36.490 / 16.775 | +117.5% | 35.960 / 18.451 | +94.9% |
| T32 H4096 R64 K10, 4 adapter spans | 30.072 / 18.408 | +63.4% | 30.735 / 18.454 | +66.6% |

All selected eager cells also won: +71.1% to +78.8% on H200 and +105.7% to
+120.7% on GB300.  The maximum BF16 absolute difference from the generic path
was 0.00146484375.

## Static generic fallbacks

The selector preserved the generic virtual-expert implementation for all
eager and captured forms of these evidence-bounded cells:

- T32 H2048 R64 active (cross-device noisy in the exploratory matrix);
- T32 H2048 R64 mixed/base;
- T32 H7168 R64 (wide Kimi-like decode);
- T1 H2048 R32 (tiny decode).

It also falls back outside Hopper/Blackwell, for top-k one, incomplete segment
metadata, unknown phases, hidden sizes above 4096, and ranks outside 1–128.

The integrated selector is intentionally stricter than that exploratory
envelope. It promotes only the exact cross-device win cells listed above,
only for decode, and rejects segment grids whose masked row capacity exceeds
the measured 16x one-token-fragmentation envelope. Other hidden sizes, ranks,
top-k values, token counts, prefill shapes, and more pathological segment skew
remain on the generic path pending evidence.

## Correctness, graph, and integration coverage

- host selector/planner tests: 9 passed;
- H200 integrated shared-LoRA aggregate: 56 passed plus 10 subtests;
- GB300 integrated shared-LoRA aggregate: 56 passed plus 10 subtests;
- active and mixed/base segments, eager and CUDA-graph replay are covered;
- the dispatch test verifies that selected routing omits the shared-A route,
  retains the gate-B route, and matches the generic intermediate.
- rank-64 direct expand and rank-128 generic expand are covered with PDL forced
  on in eager and captured full A+B chains; graph capture starts with an empty
  routing cache, so the intervening gate-B route is part of the captured chain;
- the selected segment bound follows the same indptr domain used by the
  backend, including TorchNative's consecutive-adapter request merging;
- physical shared-expert fusion is enabled only for the Standard dispatcher;
  per-rank fused shared-slot layouts with EP>1 remain rejected because the
  dispatcher remaps those physical IDs before LoRA mapping;
- the selected shared-A route is directly composed with a physical
  shared-expert ID map on both devices: shared slots are excluded from gate-B
  routing while the token-deduplicated shared-A producer remains unchanged.

The H200 node-level Nsight Systems trace contains nine observed generic shrink
instances and nine token-dedup instances across setup/timing/profile replay.
It records 27 fused virtual-ID launches, exactly the expected aggregate of two
routes per generic execution plus one gate-B route per selected execution.  It
therefore confirms that selection removes the shared-A route and replaces the
generic shrink with one token-dedup launch; it is not merely a host-side label.

GB300 timing and correctness completed, but its Nsight report exporter hung
after the workload finished.  The profiler and Python PIDs were terminated and
no partial report is treated as evidence.

## Files

- `h200/matrix.json`: complete H200 selected/fallback eager+graph matrix;
- `gb300/matrix.json`: complete GB300 selected/fallback eager+graph matrix;
- `h200/selected_graph_nodes.nsys-rep`: node-level CUDA-graph trace;
- `h200/selected_graph_nodes.sqlite`: queryable export of that trace;
- `h200/selected_graph_nodes_kernel_sum.csv`: compact kernel-count summary;
- `h200/selected_graph.nsys-rep` and `.sqlite`: compact non-node trace retained
  as a secondary raw artifact.

Reproduce with:

```bash
python benchmark/kernels/lora_moe/bench_shared_outer_gate_dispatch.py \
  --case all --execution all --warmup 20 --iterations 200 \
  --json-output /tmp/shared_outer_gate_matrix.json
```
