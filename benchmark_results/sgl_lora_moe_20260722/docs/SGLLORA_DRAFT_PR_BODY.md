## What this draft contains

This is the commit-by-commit Phase-1 MoE execution refactor for the new
`sgl_lora` engine. It keeps legacy LoRA selectable, puts provider-private
layouts behind typed execution plans, and records the H200/GB300 benchmark
campaign in a repository-local, checksummed evidence bundle.

The series is intentionally reviewable by boundary: scaffold and semantic
contracts; benchmark methodology fixes; BF16 kernels/planner; graph/PDL;
cross-model/rank/shared/mixed-rank studies; quant providers; real distributed
and lifecycle validation; alternate CuTe DSL evidence; and final audit docs.

## Important result

This PR does **not** claim that SGL is universally faster. In the matched
GB300 Qwen1.5-MoE `one_batch_server` bracket, the historical pre-planner SGL path
trailed experimental TRTLLM by 12.56%-14.71%. After planner promotion, default C2
trails by 9.43%, 9.44%, and 10.39%; opt-in C3 trails by 5.12%, 5.54%, and 10.48% at
BS1/16/32. Base decode traffic was near parity in the original bracket. These
negative results remain the next provider-fusion/launch-reduction target.

## Implemented in this phase

- virtual-expert-only MoE LoRA with local/global expert-domain handling;
- strict logical activation, slice, output-dtype, scaling, workspace, and
  distributed assignment contracts;
- device/phase/graph/rank/route-aware BF16 C0/C2/C3 planning;
- fused gate consumer and fused down finalization, with measured serial versus
  two-stream policy;
- ranks 8/16/32/64/128, active/mixed/base rows, and an evidence-bounded shared-outer
  selector. The shared policy passed 9 host tests and 56 tests plus 10 subtests on
  each of H200 and GB300, including TorchNative merged-bound repair, physical
  shared-ID composition, and PDL pairing;
- physical shared IDs only for Standard contiguous/EP1 or otherwise safe
  non-per-rank-remapped layouts. Per-rank physical shared EP>1 and advanced A2A are
  not promoted;
- bounded provider-neutral C0 seams for FP8 W8A8, synthetic native CuTe DSL NVFP4
  W4A4, and attachable Marlin W4A16. No-LoRA invokes the resident quant method.
  Active FP8 rejects static activation and Blackwell requires resident packed UE8M0
  scales. Native CuTe DSL W4A4 is not serving-reachable; broad quantized
  checkpoint/server attachment is not claimed. Base GEMMs remain provider-native,
  while LoRA arithmetic and the activation/down-A bridge remain BF16/materialized;
- separate base/adapter CUDA-graph families, explicit PDL controls, adapter
  lifecycle/eviction/slot-recycle validation;
- route-view reuse only within one MoE layer invocation; there is no whole-forward
  or cross-layer memoization claim;
- real TP/EP/MoE-DP D0, including a two-node TP8/EP2/DP2 MNNVL run;
- independently tuned indexed/segmented/grouped/BMM/one-shot/shared studies;
  the optimized Blackwell CuTe DSL/TMA candidate wins isolated static K0
  (`73.152 us` versus Triton's `87.520 us`, `-16.42%`) but fails the
  dynamic-route M0 validity gate and therefore does not enter production. This is
  evidence-only and is separate from the synthetic W4A4 provider testbed.

## Review map

Final rebased source: `f2f406e056` (replace after the last upstream rebase).
The final commit hashes and recommended review groups are filled after that rebase:

1. Engine and semantic ABI.
2. Benchmark methodology and correctness repairs.
3. BF16 execution kernels and production planner.
4. Rank, route, shared, mixed-rank, graph, PDL, and alternate-family studies.
5. Quantized providers and production execution.
6. Distributed execution and adapter lifecycle.
7. Durable evidence, third-party feedback disposition, and final docs.

## Validation

The final PR update includes the post-rebase smoke, D0/E0/lifecycle matrices,
profiler artifacts, source state, and SHA-256 manifest. Shared closure already records
9 host tests plus 56 tests and 10 subtests on each device. The Marlin per-invocation
workspace/dirty-destination repair passed the terminal `f2f406e056` H200 and GB300 smoke.
Canonical results are unprofiled; Nsight Systems verifies stream/graph structure and
Nsight Compute explains shortlisted kernels.

## Deliberate follow-on scope

This draft does not claim Phase-2 dense/special-layer completion, Phase-3
adapter residency/control-plane redesign, `kv_b_lora_absorbed`, MTP/EAGLE,
advanced A2A, sink-shared-expert optimization, or broad model-level production
graduation. Benchmark-only mixed-rank packing is not represented as a complete
multi-rank serving residency system.
