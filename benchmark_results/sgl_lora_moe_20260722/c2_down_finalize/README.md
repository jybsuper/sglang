# C2 down-B + base-finalize benchmark evidence

This directory contains the raw H200 and GB300 evidence for the benchmark-only
complete BF16 C2 tail. Production dispatch is unchanged.

## Compared topologies

- **C0 / split**: existing base-W2 post-reorder/finalize followed by the
  existing routed LoRA-B expansion.
- **C2P**: the partial C2 gate/up producer and existing split down tail.
- **C2I / indexed_split**: the base finalizer remains separate, but LoRA-B is
  pair-indexed directly from canonical `[T, K, R]`; this removes route-plan
  construction without fusing the two branches.
- **C2F / fused**: one token-and-hidden-tile-owned finalizer accumulates masked
  base-W2 and LoRA-B in FP32, applies `topk_weight * routed_scaling_factor`
  exactly once to both branches, and converts once to the requested output
  dtype.
- **Shared outer C2F**: first reduces routed rank vectors over top-k, then loads
  the one shared down-B matrix once per token/hidden tile.

The C2I control is important: it distinguishes the effect of removing route
construction from the effect of actually fusing base finalization and LoRA-B.

## Strict correctness coverage

`test_sgl_lora_c2_down_finalize.py` passed all six rank/layout cases on each
GPU: ranks 32/64/128 crossed with per-expert/shared-outer down-B. The test uses
nonzero base output, scaling 1.75, nonzero global expert offsets, invalid and
non-local expert IDs, mixed adapter/base-only rows, out-of-range adapter IDs,
FP32 and BF16 outputs, and CUDA graph replay.

The M0 JSON files also check full output and a base-subtracted LoRA-delta
oracle. For the representative Qwen decode case (`T=32`, `K=8`, `H=2048`,
`I=512`, `R=64`), C2F's maximum delta error is 0.00048828125 on both GPUs,
below the explicit 10%-of-signal gate:

| Case | GPU | delta signal | C2F max delta error | error / signal |
|---|---:|---:|---:|---:|
| per-expert | H200 | 0.00620842 | 0.00048828 | 7.86% |
| per-expert | GB300 | 0.00598145 | 0.00048828 | 8.16% |
| shared outer | H200 | 0.00592041 | 0.00036621 | 6.19% |
| shared outer | GB300 | 0.00625610 | 0.00048828 | 7.80% |

## Main performance results

Negative percentages mean lower latency than C0/split. Results use two
counterbalanced repeats with 100 samples per arm.

| Scope / case | H200 C2F | GB300 C2F | Interpretation |
|---|---:|---:|---|
| K0 per-expert, R64 | -28.76% | -17.23% | Fused kernel itself wins |
| O0 per-expert, R64 | -72.63% | -76.67% | Route/allocation plus tail boundary strongly favors C2F |
| M0 CUDA graph, per-expert R64 | -0.69% | -0.09% | End-to-end local MoE result is neutral-to-small win |
| K0 shared outer, R64 | -25.38% | -45.83% | Rank-first shared-B specialization wins |
| M0 CUDA graph, shared outer R64 | -3.23% | -4.13% | Material full-path win on both GPUs |

The route-free but unfused C2I arm loses at M0: +1.62% on H200 and +2.66% on
GB300 for per-expert R64. Thus C2F's result is not explained merely by deleting
the route plan. C2I does substantially reduce O0 relative to production split,
but pair-owned atomic accumulation makes its isolated K0 slower.

Rank tuning found different best tiles. For the measured Qwen K0 cases:

- R32: H200 and GB300 favored `BH=64, warps=4` among the tested choices.
- R64: both favored `BH=64, warps=2`.
- R128: H200 favored `BH=32, warps=2`; GB300 was best around `BH=32,
  warps=4` in this sweep.

R128's raw finalizer and synthetic K0/O0 correctness are covered.  After
`88de33bf49` tiled the production gate-A rank axis, the formerly blocked R128
M0 endpoint was rerun from a real C2 producer (not synthetic input): C2F beat
C0 by a paired median **2.99% on H200** and **7.56% on GB300**.  The strict
base-subtracted error was 0.00048828125 on both devices, respectively 5.04%
and 4.94% of the measured LoRA signal.  See
`*/qwen_mixed_r128/m0_graph_production_a_fixed.json`.

All full C2 results here use the exact **pair** gate/up consumer schedule. The
experimental aligned consumer cannot currently be promoted for mixed
base-only rows: native alignment excludes sentinel blocks, so it requires a
charged base-only activation prepass (or equivalent handling) before its
correctness/performance can be compared fairly.

## Profile evidence

Nsight Systems reports and exported summary CSVs are under each GPU's
`qwen_mixed_r64/` directory. C2F changes the measured three-iteration profile
from 36 to 27 `cuLaunchKernelEx` calls: three fewer launch calls per iteration.
The fused finalizer averaged about 14.7 us on H200 and 13.3 us on GB300 in the
profile capture.

Nsight Compute reports and exported detail CSVs are in the same directories.
At `BH=64, warps=2`, the R64 per-expert kernel used 107 registers/thread on
H200 and 110 on GB300, with no spill observed. Theoretical occupancy was 25%
and register-limited; achieved occupancy was about 20.7% and 18.4%,
respectively. Adding warps was therefore the wrong direction in the measured
sweep. A future kernel pass should target register footprint and B-load
latency.

## Artifact map

- `*/qwen_mixed_r64/m0_graph_fair_strict.json`: strict C0/C2P/C2I/C2F CUDA
  graph comparison.
- `*/qwen_shared_r64/m0_graph_fair_strict.json`: strict shared-outer full-path
  comparison.
- `*/qwen_mixed_r64/k0_fair.json` and `o0_fair.json`: split/indexed/fused
  fairness arms.
- `*/qwen_mixed_r{32,128}/k0.json`: rank/tile sweeps.
- `*/qwen_mixed_r64_shared/k0.json`: shared-outer tile sweep.
- `*/qwen_mixed_r64/nsys_*.nsys-rep`: timeline/launch evidence.
- `*/qwen_mixed_r64/ncu_*.ncu-rep`: kernel-level profile evidence.
- `*/pytest_c2_down_finalize.log`: final six-case registered GPU test run.

All JSON files include the exact CLI, GPU/software environment, correctness
metrics, per-repeat timings, and paired comparisons used for the summaries.
