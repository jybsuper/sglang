# Shared-expert profiling summary

The timing matrix is in `summary.md`.  This file records the timeline and
kernel-level evidence used to interpret those timings.  Every profile uses the
same Qwen3.5-shaped BF16 routed MoE block (`H=2048`, `I=512`, `E=256`, `K=8`)
with four active adapters plus base rows and two model shared/sink experts.

## Nsight Systems: graph topology and overlap

The CUDA-graph captures compare the conventional separate shared MLP on a side
stream with the fused per-rank physical-slot provider.  H200 traces contain
five replays; the GB300 profiler required `stop-shutdown`, so the final fused
and prefill traces contain one replay.  Raw `.nsys-rep`, exported `.sqlite`,
and per-trace `.analysis.json` files are under `raw/<device>/nsys/`.

| Device | Phase | Variant | Replays | Kernels | Streams | GPU busy / replay (us) | Cross-stream overlap / replay (us) | Shorter-stream overlap |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| H200 | decode, T=32 R=128 | separate overlap | 5 | 100 | 2 | 545.663 | 8.928 | 62.7% |
| H200 | decode, T=32 R=128 | fused per-rank | 5 | 80 | 1 | 544.176 | 0.653 | n/a |
| H200 | prefill, T=256 R=128 | separate overlap | 5 | 105 | 2 | 1114.462 | 12.294 | 66.6% |
| H200 | prefill, T=256 R=128 | fused per-rank | 5 | 85 | 1 | 1118.282 | 0.883 | n/a |
| GB300 | decode, T=32 R=128 | separate overlap | 5 | 100 | 2 | 386.991 | 6.835 | 55.6% |
| GB300 | decode, T=32 R=128 | fused per-rank | 1 | 16 | 1 | 396.386 | 0.640 | n/a |
| GB300 | prefill, T=256 R=128 | separate overlap | 1 | 21 | 2 | 820.676 | 0.000 | 0.0% |
| GB300 | prefill, T=256 R=128 | fused per-rank | 1 | 17 | 1 | 813.220 | 0.384 | n/a |

The side-stream implementation is wired correctly: both decode traces and the
H200 prefill trace show real concurrent work, with 56–67% of the shorter stream
hidden.  That overlap is too small to overcome the extra shared-MLP launches in
most timing cells.  At the high-rank prefill point the two choices are within
roughly one percent, which is why production policy should not claim a stable
winner there from this evidence alone.

## Nsight Compute: physical-ID map specialization

The new per-rank path performs its physical-to-routed lookup inside the
existing virtual-ID kernel; it adds no launch.  The control is the same kernel
specialized for contiguous global slots.  Reports and human-readable details
are under `raw/<device>/ncu/`.

| Device | Variant | Route duration (us) | Registers/thread | Grid | Full graph map-vs-global median |
|:---|:---|---:|---:|---:|---:|
| H200 | fused global | 2.94 | 26 | 1 CTA | control |
| H200 | fused per-rank map | 3.55 | 32 | 1 CTA | +0.19% at T=32 R=128; -0.02% at T=256 R=128 |
| GB300 | fused global | 5.31 | 26 | 1 CTA | control |
| GB300 | fused per-rank map | 5.44 | 32 | 1 CTA | +0.56% at T=32 R=128; -0.06% at T=256 R=128 |

The lookup specialization costs six registers and 0.13–0.61 microseconds in
this one-CTA route kernel.  It is under 0.2% of the H200 full pipeline and under
0.1% of the GB300 full pipeline at the profiled shapes.  Keeping the lookup
fused is therefore preferable to a standalone layout-conversion launch.

## Interpretation boundary

- Nsight measurements are profiler runs, not the source of latency claims;
  latency comes from unprofiled CUDA events in `summary.json`.
- `fused_per_rank` is a one-GPU identity/layout proxy.  It validates the
  DeepEP/MegaMOE physical-ID contract but is not a real distributed D0 result.
- The shared-expert implementation is BF16 C0.  Current C2/C3 consumers assume
  equal base and LoRA expert domains, so fused-shared serving safely falls back
  to C0 until those kernels accept separate base-physical and routed-factor
  identities.
