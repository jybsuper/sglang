# SGL LoRA refactor plan

Date: 2026-07-22<br>
Public execution-engine name: `sgl_lora`<br>
Current implementation phase: Phase 1 MoE execution graduation candidate<br>
Final rebased campaign head: `f2f406e056`

Architecture and full lifecycle companion:
`sgl_lora_lifecycle_and_orchestration_design.md`. It contains Mermaid views of
the request lifecycle, target control/execution-plane architecture, model-schema
compilation, and distributed adapter load transaction. That document is the
authoritative target design; if this sequencing summary ever conflicts with it,
update this plan rather than inventing a second ABI.

Current implementation status, known gaps, and the development-time validation
policy live in `sgl_lora_worklog.md`. In Phase 1a, broad unsupported-combination
guards are deferred; the worklog records the obligations until support lands or the
final graduation boundary is defined.

## 0. Campaign checkpoint (2026-07-22)

The benchmark/implementation campaign that followed the Phase-1a scaffold has now
advanced the MoE execution slice through the following independently reviewable
milestones:

1. corrected shared-expert IDs, signal-relative correctness gates, real IID/skewed
   routes, cold/loaded-host controls, counterbalanced ordering, and durable evidence;
2. neutral stock/legacy and experimental-TRTLLM model-level controls using
   `sglang.benchmark.one_batch_server`;
3. BF16 C2/C3 production planning, fused down finalization, rank 8/16/128 support,
   activation/provider guardrails, and separate base/adapter graph families;
4. bounded provider-neutral C0 seams for BF16, FP8 W8A8, synthetic/testbed NVFP4
   W4A4, and attachable Marlin W4A16, without claiming broad server attachment;
5. real TP/EP/MoE-DP execution, a two-node TP8/EP2/DP2 MNNVL run, and adapter
   lifecycle/eviction/replay tests;
6. independent shared-outer, mixed-rank, PDL, algorithm-family, and Triton studies,
   plus an evidence-only optimized Blackwell CuTe DSL/TMA challenge. The latter wins
   static K0 (`73.152 us` versus `87.520 us`, `-16.42%`) but is invalid under dynamic
   route mutation at M0 and is not integrated into production dispatch.

Physical shared-expert composition is promoted only for the bounded layouts proved
by the campaign: the standard contiguous `StandardDispatcher`/EP1 path and other
safe layouts that do not remap physical shared IDs per rank. Per-rank physical-shared
EP>1 layouts and advanced A2A dispatchers are explicitly unpromoted and must reject
or remain on a validated legacy path. This is narrower than a claim of general model
shared-expert graduation.

Shared-outer LoRA is also promoted only through its evidence-bounded selector. Its
host policy passed 9 focused tests, and the combined shared-outer/physical-ID/PDL
matrix passed 56 tests plus 10 subtests on each of H200 and GB300. The promoted work
includes the TorchNative merged-segment-bound repair, physical shared-ID
composition, and the validated producer/consumer PDL pairing; it does not imply
arbitrary factor-sharing signatures.

The key negative result is part of the plan, not hidden. On the matched GB300
Qwen1.5-MoE bracket, the pre-planner SGL path trailed experimental TRTLLM by
12.56%-14.71%. Those percentages are historical pre-planner results only. After
planner promotion, default C2 trails by 9.43%, 9.44%, and 10.39%; opt-in C3 trails by
5.12%, 5.54%, and 10.48% at BS1/16/32. Base decode traffic was near parity in the
original bracket. This remains the target bar for subsequent provider fusion and
launch reduction.

Quant support in this phase is a provider-neutral C0 execution seam, not broad
checkpoint/server attachment. A no-active-adapter invocation calls the resident base
layer quant method instead of constructing or forcing a Triton provider. Active FP8
rejects static-activation scaling, and Blackwell FP8 requires the attached provider's
resident packed UE8M0 scale ABI. Marlin W4A16 is attachable; its per-invocation
workspace/dirty-destination repair passed the terminal `f2f406e056` H200 and GB300
smoke. The native CuTe DSL NVFP4 W4A4 implementation is a
synthetic/testbed provider and is not serving-reachable.

“Completed benchmark milestone” and “complete product support” remain different.
Phase 2 dense/special layers and Phase 3 adapter control plane are still later phases;
advanced A2A, MTP/EAGLE, broad model checkpoints, and production mixed-rank residency
must not be inferred from local kernel evidence. The exact passed matrix, retained
limitations, source hashes, and profiler artifacts are recorded in
`sgl_lora_moe_kernel_benchmark_architecture_audit.md` and
`benchmark_results/sgl_lora_moe_20260722/`.

## 1. Goal and ordering

Build one maintainable LoRA execution plane for Hopper and Blackwell supporting:

- BF16 base MoE;
- FP8 block-scale base MoE;
- NVFP4 W4A4 base MoE;
- Marlin W4A16;
- dense and special-layer LoRA;
- CUDA graphs, two-stream overlap, TP, EP, DP attention, and MTP/speculative
  execution where explicitly validated.

The work stays ordered by impact and risk:

1. MoE LoRA execution and providers.
2. Dense and special-layer execution, fusion, and overlap.
3. Adapter control plane: identity, loading, residency, scheduling, and updates.

The current adapter control plane remains in place while the execution ABI is
stabilized.

## 2. Accepted design decisions

1. `sgl_lora` names the new LoRA execution engine, not a particular GEMM kernel.
2. The new MoE core supports virtual experts only.
3. Canonical objects define semantics; providers own physical tensor layouts.
4. LoRA weights are local-expert-sized under EP even when routing initially uses
   global expert IDs.
5. Static normalization happens at adapter load; dynamic conversion is fused at
   routing or stage boundaries when profitable.
6. Dense `--lora-backend` values such as CSGMV and Triton remain provider choices.
7. Two-stream overlap is an independent execution policy and is off by default.
8. Legacy execution remains available until the complete support matrix graduates.
9. At graduation, explicit `sgl_lora` selection is strict and fails clearly for any
   remaining unsupported configuration; automatic selection may fall back only with
   a stable reason and telemetry.
10. CuTe DSL is a provider technology, not the shared abstraction boundary.
11. Batch-local adapter ordinals are the canonical planning identity. Physical GPU
    slots stay behind a generation-checked residency snapshot, and a provider may
    expose either ordinals or slots through its stable device assignment.
12. The LoRA execution engine and base MoE provider are independent selectors in the
    target design. The Phase-1a MoE-runner spelling is only a compatibility shorthand.
13. Model exceptions use a per-site tri-state extension: override, exclude, or fall
    back to the centralized structural compiler.
14. Adapter-framework dialect parsing is centralized and separate from model-specific
    topology/alias overrides; providers consume only canonical bound weights.
15. Kernel algorithm and implementation technology are independent axes. Raw indexed,
    true segmented SGMV, grouped GEMM, qualifying BMM, token-owned reduction, and
    fused-consumer schedules may each be implemented in Triton, CuTe DSL, CUDA, or a
    validated provider library. Do not equate “indexed” with Triton or “CuTe DSL” with
    quantized/grouped work.
16. Canonical routing semantics do not require one universal aligned row plan. The
    selected schedule may consume canonical IDs directly, request a segmented or
    aligned view, reuse producer-packed metadata, or build provider-private metadata.
    Preparation that a selected kernel does not consume must not be paid.
17. The existing BF16 gate-first contiguous SwiGLU representation is a valid Phase-1a
    provider contract. The target separates logical activation math, logical projection
    slices/target masks, and provider-private physical input/output contracts so that
    this representation does not become the ABI for non-gated, interleaved, biased,
    FP8, or NVFP4 providers.
18. Shared-outer LoRA is routed-MoE kernel work distinct from model shared experts.
    Its promoted selector uses token/adapter-deduplicated shared gate/up A and
    weighted-rank-reduced shared down B only in the measured domain. Physical model
    shared IDs compose only through the bounded Standard contiguous/EP1 or otherwise
    non-per-rank-remapped layouts described above; per-rank physical shared EP>1 and
    advanced A2A remain unpromoted.
19. Initial model-scale BF16 tuning prioritizes ranks 32, 64, and 128. Rank 16
    correctness remains early, while rank 8/16 performance tuning is required at BF16
    graduation rather than on the first architecture screen.
20. CuTe DSL may compete for selected raw indexed/SGMV-style, segmented, grouped, and
    fused kernels in BF16, FP8, and NVFP4. cuTile remains an optional,
    capability-gated experiment; neither requires implementing every
    algorithm-by-technology combination.
21. No cross-backend performance claim or production selector promotion is made
    without matched `N0/C0/C1` and neutral experimental-TRTLLM/stock-provider
    baselines at the same semantic boundary. The GB300 model-level bracket now
    supplies that control and shows that SGL adapter decode is still slower; it is a
    target bar, not evidence for automatic default selection.
22. Available H200 and GB300 devices should be used as independent workers for safe
    case sharding. Parallel throughput is an experiment-running policy, not evidence
    of distributed TP/EP/MoE-DP behavior.
23. Route-view memoization is scoped to one MoE layer invocation. Reuse between the
    gate/up and down sites requires an exact route-view match; whole-forward and
    cross-layer caching are not promoted without separate lifetime and invalidation
    evidence.

## 3. Public selector contract

### 3.1 Top-level flag

```text
--lora-execution-engine auto|legacy|sgl_lora
```

Phase 1a default is `auto`, which resolves to effective `legacy` unless the
Phase-1a MoE shorthand is selected explicitly. Capability-based automatic selection
does not begin until the MoE provider matrix has graduated.

### 3.2 Phase 1a compatibility shorthand

```text
--moe-runner-backend sgl_lora
```

This remains sufficient to select the new MoE implementation during Phase 1a. It
normalizes the top-level execution engine to `sgl_lora` and the base-provider selector
back to `auto`. `sgl_lora` therefore never remains a runtime `MoeRunnerBackend` value.

### 3.3 Phase-1a compatibility resolution matrix

| Execution engine | MoE runner | Effective result |
|---|---|---|
| `auto` | `auto` or a stock provider | `legacy`; runner unchanged |
| `auto` | `sgl_lora` | `sgl_lora` engine; runner normalized to `auto` |
| `legacy` | stock provider | strict legacy |
| `legacy` | `sgl_lora` | configuration error |
| `sgl_lora` | `auto` | `sgl_lora` engine; provider remains `auto` |
| `sgl_lora` | `sgl_lora` | `sgl_lora` engine; runner normalized to `auto` |
| `sgl_lora` | another explicit provider | `sgl_lora` engine; provider selector preserved |

This matrix describes the current compatibility bridge, not the target selector
coupling. The Phase-1a cleanup retains the shorthand as an input alias but removes
`sgl_lora` from the normalized base-provider domain so the two axes can evolve
independently.

### 3.4 Overlap flag

```text
--enable-lora-two-stream
```

It is disabled by default and never selects an execution engine implicitly. Outside
`sgl_lora` it is currently a no-op; final CLI validation is deferred until the overlap
policy is stable.

## 4. Historical Phase 1a — clean BF16 MoE foundation

This section preserves the Phase-1a branch scope before the 2026-07-22 campaign.
Every use of “current,” “planned,” or “before graduation” in sections 4.1-4.7 refers
to that historical snapshot, not to the implementation status in section 0.

### 4.1 Naming and package boundary

- Use `sgl_lora` in the LoRA execution-engine enum and compatibility CLI alias; do
  not retain it as a base `MoeRunnerBackend` enum value after normalization.
- Move the implementation to `python/sglang/srt/lora/sgl_lora/`.
- Own the runtime bridge, virtual-expert routing, shrink/expand kernels, and
  side-stream resources in that package. The new engine must not import from or
  dispatch through `trtllm_lora_temp`; keep the legacy fork unchanged while both
  paths coexist.
- Use explicit operation names such as `init_sgl_lora_moe`,
  `dispatch_sgl_lora_moe`, and `run_sgl_lora_moe`.
- Keep generic existing names such as `FusedMoEWithLoRA`, `fused_moe_lora`, and
  `merged_experts_fused_moe_lora_add`; they describe algorithms shared by other
  paths, not the backend identity.
- Remove unused FP8/NVFP4 provider structs from the BF16-only PR.

### 4.2 Isolation from legacy experimental behavior

- Remove the positive dependency on `SGLANG_EXPERIMENTAL_LORA_OPTI`.
- Keep the new engine independent of the legacy master env. If that env still creates
  a real conflict at graduation, validate it at the final selector boundary.
- Represent gate/up as two explicit output slices. Retain a flat midpoint schedule
  when both halves are tensor-core tileable, and use an independent two-slice grid
  with masked local tails when they are not; never reduce the tile below the
  tensor-core floor merely to divide the midpoint.
- Keep serial execution as the default.
- Keep side-stream resources engine-owned. Initialize them for explicit two-stream
  execution and, during migration only, when an installed legacy dense patch still
  consumes the same named stream; the legacy master switch must never select the new
  engine.

### 4.3 Historical vertical-slice assumptions and graduation envelope

The current Phase 1a implementation has the following assumptions and gaps. They are
tracked in `sgl_lora_worklog.md` rather than encoded as a large launch-time rejection
matrix while the backend is still under construction:

- LoRA enabled;
- virtual experts enabled;
- BF16 unquantized MoE weights;
- DeepGEMM available on NVIDIA Hopper or Blackwell;
- EP size 1;
- DP attention disabled until LoRA token metadata follows the gathered MoE
  token domain explicitly;
- `moe_a2a_backend=none` and standard dispatch;
- ordinary gated SwiGLU without alpha, clamp, or `swiglu_limit`;
- no GEMM biases;
- no storage/Parameter-replacing online base-weight update until provider rebinding
  and CUDA-graph recapture are implemented; pointer-preserving updates follow an
  explicit model-weight epoch;
- no `apply_router_weight_on_input`;
- ordinary expert combine;
- routed scaling factor absent or 1;
- maximum LoRA rank in the range 1 through 64;
- gate/up and down MoE bindings present together for every adapter that actually
  targets routed experts. Global dense suffix matches are not proof of a routed MoE
  binding.

Before the new MoE backend graduates, every item must either be supported or rejected
clearly at a stable boundary. Generic high-rank gate/up expansion, for example, must
be made split-aware or receive a provider-level limitation once strategy selection is
settled.

The rank range above describes the current production caller policy, not a hardware or
target-contract ceiling. Benchmark-only evidence already exercises rank 128, and the
graduated provider must select whole-rank, looped-rank, or reduction schedules from
device and shape evidence rather than one hard rank cutoff.

### 4.4 Internal execution strategies

The engine may use more than one strategy without changing its public identity:

- no active adapter: call the base layer's resident quant method/provider so its
  packed weights, scale ABI, and graph contract remain authoritative; do not force a
  replacement Triton strategy;
- active adapter or capture: new DeepGEMM pipeline;
- optional two-stream strategy when explicitly enabled and shape-qualified.

Large-prefill workspace planning remains a tracked gap. It should become a
device/model-aware strategy decision, not a universal token threshold or a temporary
fixed-byte launch guard.

### 4.5 Phase 1a tests

- Selector resolution matrix and contradictory selector combinations.
- CLI choice parsing.
- SGL LoRA execution-engine enum and shared-expert-fusion policy.
- Legacy-path import isolation.
- S3 activation parity for gate-first/up-first, contiguous/interleaved, with and
  without delta, and invalid routed pairs.
- Gate/up A+B reference parity at ranks 16 and 64.
- Direct expand parity for one and two output slices, including the ordinary
  `H=192` midpoint schedule, production down-collapse behavior, and an odd
  half-width such as `H=193` that requires independently masked slice tails.
- An isolated CUDA-graph-aware schedule benchmark comparing flat midpoint and
  two-slice grids at equal tile sizes, with an optional wide-tile experiment and an
  independent numerical reference.
- Base-only, active adapter, multiple adapters, and slot recycling.
- Eager, capture, replay, and recapture.
- Engine-only and MoE-shorthand selection must be bit-equivalent.
- Legacy default must not import or initialize the new execution path.
- Near graduation, add the settled A2A, EP, DP-attention, quantization, rank, bias,
  activation, adapter-binding, graph, workspace, and base-weight update validation
  matrix.

### 4.6 Phase 1a exit criteria

- No stale backend-name references in tracked source.
- Local CPU/config tests pass.
- Hopper and Blackwell kernel tests pass.
- Existing GB300 accuracy result reproduces without the experimental master env.
- Serial `sgl_lora` is no worse than legacy BF16 when the experimental master env and
  its gated optimizations are disabled on both sides, within agreed benchmark noise.
- Enabling two-stream is separately measurable and never changes correctness.

### 4.7 Current Phase-1a status

Maintain the live done/in-progress/gap list only in `sgl_lora_worklog.md`; do not
duplicate commit hashes and transient review status in this sequencing plan.

### 4.8 Independent correctness fixes to land early

These do not wait for the Phase-3 control-plane rewrite:

- make catalog resolution plus request-lease acquisition atomic with unload;
- release every acquired lease in `finally`, including pre-dispatch failures;
- preserve identity on implicit reload only after immutable revision verification;
- roll back every locally staged artifact and aggregate every required-rank prepare
  result before publish;
- forward exact multimodal language-site filtering until schema compilation replaces
  the current unused hooks;
- implement gate/up/down binding through the typed rank-local schema rather than a
  one-off Phase 1a adapter validator or global dense suffixes.

## 5. Phase 1b — canonical MoE execution contracts

### 5.1 Phase 1b-core

Introduce only the stable semantic and residency handoff types while adapting from
the existing manager and memory pool:

```text
BatchAdapterTable
  ordinal -> immutable AdapterKey
  domain: local batch or execution group

LoRABatchAssignment
  request adapter ordinals
  token adapter ordinals
  request/token transform metadata

AssignmentView
  named token/request domain
  selection/permutation for the current MoE token transform

SlotRef / ResidencySnapshot
  ordinal -> (physical slot, generation)
  snapshot lease and residency epoch

DeviceAdapterAssignment
  stable max-capacity device buffers
  declared kernel index space: ORDINAL or SLOT
  ordinal-to-slot/generation/valid and per-site rank/scale/enabled views
  host mapping epoch

MoeGeometry
  global and local expert counts
  local offset or expert map
  expert-ID input and kernel domains

LoRAExecutionContext
  batch table and assignment
  residency snapshot and device assignment
  provider state and completion lease

ExecutionCompletionTracker
  terminal stream fences
  detached slot and graph-lane leases
```

Rules:

- Do not infer request assignment from CSGMV chunk segmentation.
- Derive canonical assignment from request adapter keys in `ForwardBatch`; use the
  residency snapshot to lower ordinals to physical slots only where required.
- Refresh stable device mapping buffers in place. Adapter keys, ordinal permutations,
  slot generations, and mapping epochs are graph data, not graph-executable keys.
- Before DP-attention redistribution, ranks agree on an execution-group adapter table
  or carry an explicit `(source_rank, local_ordinal)` pair until remapping.
- Initially wrap existing memory-pool tensors; do not duplicate adapter loading.
- Hold snapshot slot leases through completion events from every stream that may read
  adapter weights; never recycle a slot merely because Python forward returned.
- Require every consuming stream to wait on activation readiness before copying,
  packing, or reading a published slot.
- Use `-1` adapter slot/index sentinels only in legacy bridges and validate them
  before every indirect adapter-weight load. Canonical expert localization retains
  `-1` for an absent/nonlocal expert.
- Do not add unused `StageValue`, exhaustive capabilities, lm-head pass fields, or
  speculative-only fields. Add each with its first consumer.

### 5.2 Phase 1b-BF16 provider boundary

- Build `LoRASiteCandidate` values from actual rank-local `FusedMoE` objects and
  compile the initial `MoeLoRASite` through model-override-first, central-fallback
  resolution.
- Derive dimensions, expert geometry, activation, and physical partitions from the
  constructed module/runner configuration rather than repeating HF-config formulas,
  while keeping adapter-logical/rank-local dimensions separate from provider-padded
  storage dimensions.
- Add the initial PEFT/tensor `AdapterFormatReader` bridge and resolve routed-expert
  weights to exact typed sites/slices; do not make longest-substring matching part of
  the new ABI.
- Partition binding ownership explicitly: the new compiler claims routed-MoE weights,
  legacy claims dense/special weights, and strict rejection happens only after both
  decline or on an ambiguous double claim.
- Reverse the dependency on `trtllm_lora_temp` through neutral or engine-owned ops.
- Define reusable `ProviderPlan` and opaque stable workspace ownership separately
  from per-forward `ProviderInvocation` counts, routing IDs, virtual IDs, and current
  stage views.
- Keep canonical logical weights separate from BF16 provider-packed views.
- Preserve the current BF16 contract explicitly: standard gate-first contiguous W13,
  ordinary gated SwiGLU, canonical pair-domain LoRA delta
  `[T,K,2I] = [GATE | UP]`, BF16 W2 input, and the pair-domain BF16 activation
  bridge consumed by down-A. This is implemented behavior, not a missing contract.
- Compile three independent specifications instead of exposing that physical form as
  the universal ABI:
  1. `ActivationSpec`: logical activation equation and parameters, including gated
     versus non-gated, alpha/clamp/bias, and exact numerical policy;
  2. `ProjectionSliceSpec`: logical `GATE`/`UP`, one `VALUE`, and `DOWN` slices,
     adapter target masks, factor sharing, rank, and scale without mandatory zero
     materialization for untargeted slices;
  3. `ProviderIOContract`: provider-private physical W13 layout and exact consumer
     outputs, including W2 input dtype/layout, scale/swizzle tensors, the pre-quant
     source view for down-A, invalid-row policy, destination dtype, and buffer
     ownership.
- The provider owns physical order/interleave, zero-fill when it is actually useful,
  max-rank component strides, quant packing, and optional prepared route views;
  `stacked_multiply` and one aligned row plan are not site contracts.
- Do not add duplicate Phase-1a runtime containers merely to rename existing BF16
  fields. Introduce each specification with its first alternative provider or fused
  consumer, adapting the current BF16 plan into it.
- Declare sharing, routing-weight, expert-ID, bias, and scaling requirements in the
  logical site; add only the minimal BF16 capability checks consumed by the selector.
- Replace the fixed workspace budget with a model/device-aware planner.

## 6. Phase 1c — graphs and quantized MoE providers

Land providers independently behind the same pipeline.

### 6.1 Full decode graphs and breakable prefill

- Add distinct LoRA-off and LoRA-on graph families.
- Refresh stable `DeviceAdapterAssignment` and workspace buffers before replay.
- Give each concurrently in-flight forward one graph lane; fence all provider streams
  before lane or slot reuse.
- Keep the existing full CUDA graph as the primary decode backend.
- Make breakable CUDA graph the primary Hopper/Blackwell prefill/extend target. Keep
  graph-safe LoRA kernels captured and place explicit eager breaks only around stages
  that use capture-stable arguments, fixed bridge outputs, and refreshed stable device
  metadata. A stage that branches on live Python adapter IDs or changes output shape
  forces the whole forward eager.
- Keep the BCG-captured transformer body and eager embedding/lm-head/logits prefix or
  tail on one assignment, residency snapshot, and completion lease.
- Remove the current BCG-LoRA guard only after capture-pool/host memory and segment
  count/latency gates pass against eager and base-only BCG.
- Refresh mappings after final EXTEND transforms and clear padded adapter-metadata
  tails. Until this is complete, reject a capture that used `lora_ids=None` but would
  replay with real adapters.
- Defer `tc_piecewise` LoRA support as a separate compatibility/platform path; its
  fake, mutation, alias, and split-op contracts are not required for initial BCG.
- Recapture before publishing a base-weight update that replaces captured storage.

### 6.2 FP8 block scale

- DeepGEMM base GEMM provider for Hopper and Blackwell.
- Preserve BF16 LoRA inputs while the base provider consumes quantized views.
- Keep BF16 activated output alive for down LoRA.
- Fuse activation plus base-input quantization only when both views remain correct.

### 6.3 NVFP4 W4A4

Status: **historical provider-native fusion backlog.** Provider-neutral C0 has since
been validated; the native bridge-removal work below remains follow-on scope. The
private workspace note is local provenance, not a required input to this design.

Measured B200 reference after symmetric-memory configuration:

| Batch size | W4A4 LoRA tok/s | Standalone W4A4 retained | Versus optimized W4A16 |
|---:|---:|---:|---:|
| 1 | 101.06 | 67.6% | -7.6% |
| 16 | 1171.42 | 77.4% | +7.7% |
| 32 | 2097.76 | 82.3% | +20.5% |

S4 retained 98.4%/97.8% of S1 at BS16/32. The BS1 weakness is the decomposed LoRA
topology: it added about 3.21 ms over W4A4 base versus 1.66 ms for optimized Marlin.
The dominant costs are BF16 gate/up materialization, the separate delta-aware
activation+quant step, the BF16 activated bridge into down LoRA, the serial down
tail, top-k repacking, and attention/dense LoRA launch count—not the routed FP4 BMM.

Provider work packages, in order:

1. Define explicit NVFP4 LoRA injection seams immediately before activation/FP4
   quantization and before finalize/combine/collective. Keep FP4 packing, scales,
   workspace, and physical route layout provider-private. CuTe DSL is the primary
   self-authored implementation candidate, while validated TRT-LLM/CUDA/provider
   kernels remain controls and may be wrapped first.
2. Add the consumed NVFP4 `StageValue` views; do not force full BF16 bridge tensors
   into the shared ABI.
3. Fuse gate/up B expansion/add into activation+FP4 quantization.
4. Fuse down-A shrink with activation production, then feed rank output directly to
   weighted top-k reduction.
5. Fuse rank reduction/shared down-B into finalize and, where valid, the final TP
   collective while preserving base rows, adapter mapping, routed scaling, EP-local
   ownership, and one final reduction.
6. Preserve packed top-k through routing/provider lowering; remove the standalone
   repack launch.

Shared graph/execution dependencies:

- Restore separate no-LoRA and LoRA-on decode graphs so all-base batches replay the
  stock fused TRT-LLM path.
- Make capture-aware skipping consistent for routed MoE, shared sink, qkvr,
  `wo_ud`, embeddings, and lm head.
- Restore `wo_ud` direct output to the symmetric all-reduce buffer, add its LoRA
  delta in place before the collective, and reduce qkvr/`wo_ud` launch count.
- Evaluate constrained LoRA BCG prefill with explicit capture-pool and host-memory
  gates.
- Select symmetric memory per device/driver/provider capability with a conservative
  fallback; do not auto-enable it globally.

Rejected isolated optimization: a shared gate/up-A plus post-reduction down-B
factorization prototype passed GPU correctness but was throughput-neutral
(-0.7% to +1.1%). Its source was intentionally not retained. Do not repeat or
cherry-pick it as a standalone optimization; revisit the algebra only together with
the deeper bridge/epilogue fusion above.

Graduation requires S1-S4, mixed/base-only rows, poisoned intermediates, graph
eager/capture/replay and pointer stability, repeated graph transitions, adapter
lifecycle/eviction during forwarding, TP8, and ordinary EP>1 global/local expert-ID
coverage. Benchmark NVFP4 on B200 and use H200 shared-execution/Marlin controls for
common graph/collective changes. This first NVFP4 milestone excludes MTP/EAGLE and
advanced A2A/EPLB/elastic EP. Use input 1024/output 512, BS1/16/24/32, medians of
three trials, standalone and same-server base comparisons, and whole-batch
critical-path profiling with Perfetto-compatible traces.

### 6.4 Marlin W4A16

- Adapt the existing Marlin hook points or add a provider implementation.
- Preserve pre-activation gate/up and pre-combine down injection semantics.

Each provider needs independent reference, graph, memory, and performance gates.

## 7. Phase 1d — EP and distributed execution

1. Allocate LoRA expert weights as `[max_loras, E_local, ...]`.
2. Keep global routing identity at the dispatch boundary.
3. Localize exactly once using contiguous offset arithmetic or `expert_map`.
4. Build virtual expert IDs from local IDs.
5. Carry token adapter ordinals through A2A with the routed tokens; lower them only
   in the destination provider's declared device index domain.
6. Validate ordinary EP first.
7. Add DeepEP/Mooncake/Mori/NIXL only when their payload and graph contracts are
   explicit.
8. Add DP-attention only with execution-group adapter-table agreement before token
   redistribution.
9. Add MTP/speculative assignment views and an explicit target/draft adapter policy
   after token ownership and mapping are proven.

EP is not complete until CPU loading, GPU pool size, routing, graph metadata, and
provider kernels all use the same ID-space contract.

## 8. Phase 1e — MoE optimization and graduation

Benchmark and select provider tiers rather than hard-coding one kernel:

- direct `(token, top-k)` sparse decode;
- aligned/grouped execution;
- one-shot A+B with register-resident rank intermediate;
- split-phase FP32 intermediate fallback;
- optional PDL/TMA only where end-to-end benefit is measured;
- two-stream `off`, then measured `auto`, with `force` reserved for testing.

### 8.1 MoE-LoRA benchmark source of truth

The benchmark implementation belongs under `benchmark/kernels/lora_moe/`, not in a
production kernel module. Use these benchmark-only components:

```text
cases.py             immutable resolved case records and model presets
matrix.py            curated case expansion; never a blind Cartesian product
bench_local.py       kernel and single-rank local-pipeline scopes
profiling.py         shared CUDA-event, graph, NVTX and cudaProfiler mechanics
bench_shrink_schedules.py  benchmark-only A tile/split schedule laboratory
bench_distributed.py dispatcher -> runner -> combine scopes
profile_kernels.py   optional Nsight Compute/nsight-python shortlist profiler
```

`MoeLoraBenchCase` is one fully resolved concrete run. It does not contain lists of
candidate ranks, token counts, providers, or schedules. The first implementation uses
small immutable model, adapter, factor-shape and resolved-case records; it performs only
essential occupancy and shape checks. Keep future provider/layout/schedule support gaps
in the worklog until those implementations exist instead of front-loading review-heavy
validation. Timing iterations and reporting options remain outside semantic case
identity.

Use these terms consistently:

| Symbol | Meaning |
|---|---|
| `WS1` | Physical world-size-one execution: `TP=1`, `EP=1`, and `MoE-DP=1`. A one-GPU local-shape proxy for another topology is not WS1 and is not distributed evidence. |
| `T` | Local input token rows before top-k expansion. |
| `K` | Router top-k. |
| `P_capacity` | Pair capacity, normally `T * K`; graph padding may make it larger. |
| `P_valid` | Valid rank-local routed pairs after ID localization/masking. |
| `P_aligned` | Pair slots after the selected row-plan alignment/padding. |
| `P_work` | Rows actually processed by one stage: `P_aligned`, indexed `P_valid`, or token-deduplicated `T_valid`. This is a stage property, not a synonym for `T*K`. |
| `G` | Nonempty virtual groups `(adapter, local expert, factor/layout signature)`. |
| `M_g` | Valid row count for virtual group `g`; its histogram is a primary schedule input. |
| `E_hit` | Local experts whose factor rows are touched in this run; use it for factor-byte and bandwidth accounting. |
| `H_model` | Outer model hidden size. |
| `H_moe` | Hidden size actually consumed by routed experts; Nemotron Super uses a latent value. |
| `I` / `I_phys` | Logical rank-local expert intermediate width / provider-padded physical width. |
| `R` / `R_max` / `R_phys` | Active factor rank / allocated slot rank / kernel-or-provider padded physical rank. For example, logical rank 8 may execute as physical rank 16. |
| `L_active` | Distinct non-base LoRA adapters with at least one row in the run. |
| `B_base` | Whether base-only rows are present (`0` or `1`). Base is a real batch/pool identity even though its LoRA rank is zero. |
| `L_groups` | Distinct batch identities, `L_active + B_base`; it must be no greater than `L_capacity`. |
| `L_capacity` | Configured `max_loras_per_batch` (`mlpb`) and allocated or graph-captured slots. SGLang's default is 8 and the limit includes a base-only request. `L_capacity`, not only `L_active`, determines virtual-expert histogram and buffer size. |
| `L_resident` | Non-base adapters occupying initialized device-pool slots, including adapters not used by this forward. `L_active <= L_resident`; keep device residency, host registration/loading, churn, active work and slot capacity separate. |
| `S` | Logical input-projection slice count: two for gated gate/up and one for non-gated value. |

Derived values are not independent benchmark axes:

```text
moe_tp       = TP / (EP * moe_DP)
E_local      = E_global / EP, plus provider-resolved shared slots
I_local      = I_global / moe_tp
I_phys       = provider.resolve_or_pad(I_local)
P_capacity   = T_local * K
P_valid      = count(valid rank-local IDs)
P_aligned    = selected row-plan result
P_work       = stage.resolve_work_domain(T_valid, P_valid, P_aligned)
L_groups     = L_active + B_base
G            = count(nonempty virtual groups)
E_hit        = count(local experts with at least one valid routed pair)
R_phys       = kernel_or_provider.resolve_rank_padding(R, R_max)
factor_views = topology.resolve_local_A_B_shapes_and_sharding(site)
```

The resolved case includes a per-site `FactorShardSpec`; deriving only `E_local` and
`I_local` is insufficient. Under TP, gate/up-A and down-B can retain full outer `H`
while gate/up-B output and down-A input are intermediate-sharded. Under EP, expert
count changes instead. Report the exact local A/B shapes and touched bytes for every
stage.

### 8.2 Stage and kernel comparison matrix

Each row defines an exact comparison boundary. A candidate must produce every output
listed for that boundary; removing a bridge is fair only when the candidate also
performs its consumer.

| ID | Stage and semantic contract | Variants and schedules to compare | Required outputs and measurements | Invalid or unfair comparisons |
|---|---|---|---|---|
| `G0` | Router logits/bias -> canonical top-k IDs, weights and optional provider/LoRA packed epilogue. | Existing model router; AOT/JIT CUDA/Triton; small-token versus large-token schedule; FP32/BF16/FP16 logits and correction bias with host casts versus in-register widening; fused top-k/scale transpose/LoRA pack versus separate epilogue; sigmoid/noaux, bias, grouping, renormalization and EPLB variants. | Canonical IDs/weights plus declared optional packed outputs; gate-only and gate+epilogue latency, removed casts/launches, graph allocation behavior and numerical pairing of IDs with weights. | Producer fusion is valid only for the exact router semantics. Never compare a fused pack producer with a separate-pack consumer without charging the removed work. Preserve ID-weight pairing, scaling and renormalization. |
| `R0` | Canonical top-k plus token adapter assignment -> localized expert identity and only the optional route views requested by selected consumers. Localize expert identity exactly once. | Canonical raw IDs/no plan; segmented offsets; native/aligned row plan; current JIT/global-histogram, local dense histogram, active-group compact histogram, single-CTA fused ID+histogram+scan+scatter, and multi-CTA plans; producer-side packed-top-k; local/global IDs; load-time EP-local packed factors versus global factors plus mask; nonzero offset and `-1` sentinel; gate/down view reuse hit/miss; `L_capacity` virtual-expert thresholds and graph-capacity plans. | Route-only and full-prepare latency, requested view set, `P_valid/P_aligned`, bucket count, padding, shared memory, scan barriers, allocations/workspace, plan cache hit/miss, metadata refresh/replay scope and launches. | Never compare prebuilt routing with route-inclusive timing. Do not build or expose an aligned plan for a raw-indexed, producer-packed, token-owned, or provider-private consumer that does not use it. Include repack unless its producer emits it. Global IDs require a map/offset. A single-CTA limit is a dispatch condition, not a semantic assertion. Every consumer must reject sentinels before dereference. Per-forward noncontiguous expert slicing/copy is not a valid free optimization. |
| `Q1` | Canonical routes plus token-domain hidden input -> provider-native W13 input rows and quant scales. | Separate permute then quantize; fused permute+quantize; process `P_aligned`, indexed `P_valid`, or quantize once per source token then scatter to pairs; future provider-cooperative indexed W13 that avoids duplicate scattering; BF16/FP8/NVFP4; exact versus fast quant math; provider scale swizzles. | Prepared rows/scales at the exact G1 ABI; `P_work`, padding amplification, bytes, launches, touched rows, scale-layout fingerprint and graph-capacity cost. | Dedup is valid only when all destinations consume the identical source-domain quantization. Compare through G1 if layouts differ. Poison padded/sentinel rows. Charge scatter or indexed-GEMM gather. |
| `A1` | Gate/up LoRA-A shrink: token-domain input times selected A factors -> pair-domain packed-rank result. Router weight is not applied here. | Raw-route indexed/direct row ownership; true segmented SGMV; aligned expert-grouped GEMM; cuBLAS/cuBLASLt/ordinary GEMM only when algebra truly collapses, such as shared A; BMM only for regular equal `M_g`; whole-K/looped-K; split-K; BF16 versus FP32 workspace/accumulation; cast in a separate kernel versus consumer on-load cast; compute shared A once per token/adapter versus pair repetition; static-capacity-one specialization; gated two-factor and non-gated one-factor plans. Implement shortlisted algorithms in Triton and selected CuTe DSL/CUDA/provider technologies independently. | Hot-plan and route-inclusive latency, `P_work`, `R/R_phys`, workspace clear/reduction/cast cost, touched factor bytes, determinism, registers/spills and exact local factor shape. | A handwritten raw indexed kernel is SGMV-like but is not evidence for a true segmented SGMV implementation. One active adapter does not turn per-expert A into a dense GEMM. Do not deduplicate per-expert A across top-k. BMM includes padding. Split-K clear is timed. One-shot A+B is incompatible with A-only overlap or a delayed fused-B consumer. |
| `G1` | Base W13/gate-up provider: routed input -> provider-native raw gate/up/value output immediately before LoRA injection. | BF16 masked DeepGEMM; FP8 provider; NVFP4 W4A4 provider; Marlin W4A16 provider where applicable; grouped/masked/persistent/provider BMM; materialized producer versus provider-cooperative consumer fusion. | W13-only and prepare+W13 latency, ready timestamp under overlap, physical layout/scales, raw-output bytes and utilization. | A raw-output timing is not stage-equivalent to a provider that also performs activation/quantization. Compare at the next common boundary. |
| `B1` | Gate/up LoRA-B: packed A result times selected B slices -> pre-activation delta, added only to targeted logical slices. | Materialized delta baseline; raw pair-owned/indexed, true segmented SGMV, or expert-grouped ownership; aligned-flat, uniform-sliced, compiled-ragged, descriptor-ragged, compiled full union with existing zero factors, and multi-launch diagnostic; whole-rank versus looped-rank; flattened/2-D/3-D program mapping; overwrite, ordered load-add-store or true reduction atomic; `BN` through 512 and `GROUP_M`. Screen algorithms in Triton, then use selected CuTe DSL/CUDA/provider implementations to test whether a different technology or fusion boundary wins. | Delta correctness/latency when materialized, or inclusion in `S3`; `P_work`, `E_hit`, touched factor bytes/effective bandwidth, `R/R_phys`, target preservation, tail class, layout metadata and materialized bytes. | Flat tiles may not cross semantic slice boundaries. Atomics do not make an unsynchronized base-GEMM store race safe. Use atomics only for true multi-writer reduction; otherwise establish ownership/order. Multi-launch includes all launches. Full-union includes zero-factor costs. |
| `S3/A2` | Common fused consumer: base W13 output + gate/up A result/B + maps -> base add -> activation -> provider-native W2 input; optionally compute down LoRA-A from the exact pre-quant activation. | Current `B1 -> activation -> BF16 activation bridge -> down-A`; fused B+activation; activation+down-A with materialized delta; full B+activation+optional FP8/NVFP4 quant+down-A. B raw/direct versus segmented/grouped; down-A loop-I, split-I atomic/partial reduction, or cluster reduction. Sweep scalar/vector loads, elements or scale-blocks per thread, grid width, CTA waves, register versus shared activation cache, scale reduction, exact quant mode and scale swizzle. Triton is the first schedule laboratory; selected CuTe DSL prototypes may cover raw, segmented, grouped, or fused ownership. cuTile is optional and capability-gated. | Time until provider W2 input/scales and down-rank result are both ready; `P_work`; launches; removed `[T,K,2I]` and `[T,K,I]` bytes; scale traffic; exact activation/rounding; registers/shared memory; `ceil(I/quant_block)` versus CTA coverage. | A candidate removing the BF16 bridge must perform down-A. BF16 and NVFP4 are different boundaries. Down-A executes exactly once. Split-I clear/reduce is timed. Full fused B cannot coexist with side-stream A+B. Never assume one scale block/thread covers arbitrary `I`; loop, enlarge legally or fall back. |
| `G2` | Base W2/down provider: provider-native activated/quantized rows -> routed expert output. | BF16 masked DeepGEMM; FP8; NVFP4 W4A4; Marlin W4A16; grouped/masked/persistent/provider BMM; start from materialized or fused-`S3` ready boundary. | W2-only and W2-from-common-ready-boundary latency; input/scale/output bytes; utilization. | Use identical prepared input contracts. Quantized providers are compared as complete provider pipelines, not against an uncharged BF16 conversion. |
| `B2/F` | Down LoRA-B plus reduction/finalize: down-rank result times B, router weight and routed scaling exactly once, pair->token reduction, add to base in destination dtype before/inside finalize/collective. | Current expert-grouped atomic expansion; raw indexed/direct; true segmented SGMV; grouped expansion then reduce; token-owned top-k loop with one FP32-accumulated store; shared-B weighted rank reduction then one B; fresh overwrite versus ordered RMW versus true reduction atomic; fuse B into finalize or TP collective; down-B overlap with W2 only as an experiment. | Latency from `{W2 output, down-rank}` ready to final token output; `E_hit`, touched B bytes/effective bandwidth, hidden temporaries, atomics/contention, launches, event waits, collective bytes and BF16/FP16/FP32 destination correctness. | Shared-B rank reduction is invalid for per-expert B. Never apply top-k/routed scale twice. Atomics are not cross-stream synchronization. Compare fork-to-join. Match TP/EP/collective/symmetric-memory policy and initialize the base output nonzero in correctness tests. |
| `SH` | Shared-expert contribution and interaction with routed finalization. | Disabled, separate shared expert, fused global slot, fused per-rank slot; serial add versus measured shared-add overlap; LoRA targets routed only versus explicitly bound shared site. | Final output, slot placement, `1/EP` scaling, launches, overlap critical path. | Do not infer shared-slot placement from model config alone; use the instantiated provider. Do not fold shared results into routed scaling incorrectly. |
| `M0` | Complete local MoE block: router input/logits or hidden+canonical top-k -> final hidden through preparation, base provider and both LoRA injection points. | Stock Triton/legacy, experimental TRTLLM, current SGL materialized, fused SGL winners; base-only control; exact serial/overlap window; side A-only versus A+B; stream priority/resource partition diagnostic; PDL off/on at valid producer-consumer edges. | Critical-path latency, routed pairs/s, active/base retention, route/LoRA tax, memory, launches, graph nodes, allocation owner, fork/join timestamps, graph replay and numerical reference. | Changing provider/layout is a whole-backend comparison. Report matched base-only retention. Sum of isolated kernels does not establish overlap. A PDL-enabled repeated standalone kernel is not a production chain. |

Scheduler applicability is stage-specific:

| Schedule | Gate/up A | Fused gate/up B | Down A | Down B/finalize |
|---|---|---|---|---|
| Raw indexed/direct | Primary fragmented-decode candidate; may consume canonical IDs without an aligned plan. | Primary small-`M_g` candidate; naturally owns row-wide activation/quant. | Primary small-`M_g` or fused-consumer candidate. | Possible, but token-owned top-k is more natural. |
| True segmented SGMV | Competes when request/group segment metadata is already available and useful. | Competes for ragged small groups without aligned padding. | Competes when activation rows form reusable segments. | Possible, but pair-to-token reduction still needs explicit ownership. |
| Expert-grouped GEMM | Current path and likely substantial-group/prefill candidate. | B reuse and contiguous provider rows; likely substantial-group candidate. | Likely large-group standalone winner. | Current atomic path; B reuse but cross-group reduction remains. |
| BMM | Only fixed, regular equal group shapes with padding charged. | Same restriction. | Same restriction. | Poor fit for ragged top-k reduction. |
| Token-owned top-k | Not applicable. | Not required. | May be internal to a fused activation consumer. | High-priority decode candidate; avoids atomics and writes once. |
| Ordinary GEMM | One substantial group. | One substantial group. | One substantial group. | Rare specialization. |
| Split/stream-K or persistent | Orthogonal occupancy option when output tiles underfill the device. | Orthogonal option after row schedule is chosen. | Orthogonal reduction option. | Only if the complete reduction/finalize benefits. |

BMM is not implemented merely to fill the matrix. Admit it when a saved route trace
has regular equal `M_g` or measured padding efficiency high enough to justify it.

### 8.3 Whole-pipeline execution variants

| ID | Execution topology | Priority |
|---|---|---|
| `N0` | Provider-matched no-LoRA path with no LoRA launches and that provider's appropriate no-LoRA graph/output contract. Every SGL, legacy, TRTLLM, and quantized whole-provider comparison supplies its own matched `N0`. | Required control; never reuse an unmatched stock-provider base number. |
| `C0` | LoRA route + gate/up A+B -> base prepare/W13 -> activation plus BF16 bridge -> W2/finalize -> down A+B. | Current serial/materialized control. |
| `C1` | Main route prewarm; side gate/up A+B in parallel with main prepare/W13; join before activation; bridge -> W2/finalize -> serial down A+B. | Current two-stream control, including event and main routing overhead. |
| `C2` | Route -> gate/up A -> base W13 -> fused gate/up B+activation(+quant)+down-A -> W2 -> down-B+finalize. | First production target: serial path and initial prefill candidate. |
| `C3` | Main route prewarm; side gate/up A in parallel with main prepare/W13; join -> fused B+activation(+quant)+down-A -> W2 -> down-B+finalize. | Second production target: measured decode/A-only-overlap candidate. |
| `C4` | `C2`/`C3` plus down-B overlap with W2, joining before finalize. | Experimental fallback only; prior down overlap was neutral/unsafe. |
| `C5` | One-shot materialized gate A+B in parallel with W13 and one-shot down A+B after finalize. | Decode fallback/diagnostic; not the fused target ABI. |

Mutual exclusions are explicit:

- `C1` side-stream A+B cannot use fused gate-B consumers `C2`/`C3`;
- A-only overlap requires a materialized A result and cannot use one-shot A+B;
- a materialized activation bridge and fused down-A are alternative boundaries;
- post-finalize down A+B and precomputed down-A/fused finalize are alternative paths;
- serial/overlap topology is fixed per captured graph family, not switched inside a
  replay by live Python state.

### 8.4 Resolved case dimensions

Model presets use global logical geometry. Rank-local logical and physical shapes are
derived through the chosen topology/provider.

| Preset | `H_model` | `H_moe` | `I_global` | Routed experts | Top-k | Expert form | MoE layers | Primary role |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Qwen3.5-35B-A3B | 2048 | 2048 | 512 | 256 | 8 | Two-slice SwiGLU | 40 | BF16 correctness/E2E anchor; existing adapter. |
| Qwen3.5-397B-A17B | 4096 | 4096 | 1024 | 512 | 10 | Two-slice SwiGLU | 60 | Large-expert FP8/NVFP4 shape. |
| Kimi-K2.5 | 7168 | 7168 | 2048 | 384 | 8 | Two-slice SwiGLU | 60 | Largest-H W4A16/NVFP4 anchor. |
| GLM-5.2 | 6144 | 6144 | 2048 | 256 | 8 | Two-slice SwiGLU | 75 | Large-H FP8/NVFP4 and DSA-family anchor. |
| Nemotron-3 Super | 4096 | 1024 latent | 2688 | 512 | 22 | One-slice non-gated ReLU2 | 40 | Mandatory latent-H, high-top-k, non-gated case. |
| Nemotron-3 Nano | 2688 | 2688 | 1856 | 128 | 6 | One-slice non-gated ReLU2 | 23 | Odd-I/provider-padding graduation edge. |

| Axis | Required core values | Extended/graduation values and rule |
|---|---|---|
| Active/max/physical rank | Initial model-scale benchmark/schedule performance: active `32,64,128`, with explicit `(R,R_max)=(32,128),(64,128),(128,128)`. Pre-graduation R128 is benchmark-only evidence until a production A path exists. Keep rank-16 correctness in early tests. | BF16 graduation adds model-scale performance at `8,16`, a supported production R128 path, `(R,R_max)=(8,128)`, `(R,R_phys)=(8,16)`, then `256`, non-power-of-two `12,24,48,96,257`, and heterogeneous ranks. Keep semantic rank, allocated rank and tensor-core/provider padding separate. Poison inactive tails. |
| Local token rows | Decode `1,4,16,32,64,128,256`; prefill `64,128,256,512,1024,2048,4096`. | Decode `512,1024`; prefill `8192,16384` if memory permits; recorded real traces. Derive pair counts. |
| Requests | Decode one token/request; prefill equal and ragged contiguous segments. | Zipf lengths, chunk boundaries, mixed/verify. |
| Router top-k | Every model-native `K`; isolated Qwen anchor sweep `K={1,2,4,8,16,22}` at fixed `T`, plus matched-`P_valid` cases to separate top-k fragmentation from total pair work. | Captured model-native/noaux traces and larger future `K`; do not compare different `K` values only at fixed `T` and call the result a kernel-schedule effect. |
| Adapters, base rows and slot capacity (`mlpb`) | Use `(L_active,B_base,L_capacity)`, where `L_active` excludes base and `L_active+B_base<=L_capacity`. Compact core cells are `(0,1,8)`, `(1,0,1)`, `(1,0,8)`, `(1,1,8)`, `(3,0,8)`, `(4,1,5)`, `(7,1,8)`, and `(8,0,8)`. They cover the production default, base-only, single-adapter, mixed, odd active count, non-power-of-two full occupancy, sparse capacity and saturated capacity without a Cartesian sweep. | At fixed `(L_active,B_base)=(1,0)`, isolate capacity with `L_capacity={1,2,3,4,5,8,16,32}`; large capacities may be `R0/O0`-only when full factors are impractical. At capacity 8, sweep occupancy with no-base active `{1,2,3,4,7,8}` and mixed active `{1,4,7}`. Add odd capacities `3,5`, sparse/high/permuted slot IDs, `L_resident >> L_active`, heterogeneous rank and churn. Derive virtual-expert boundary cells around `E_local*L_capacity=1024` and `8192` rather than treating arbitrary capacities as magic values; examples include `E_local=256,C=3/4/5`, `E_local=512,C=1/2/3`, `E_local=512,C=15/16/17`, and `E_local=256,C=31/32/33`. |
| Expert/adaptor routing | Exact balanced, seeded iid uniform, 80/20 hotset, one no-local-route correctness case. | Captured routes, severe Zipf/single-hot, correlated expert/adapter hotsets, worst singleton fragmentation, empty experts/ranks. |
| Work domain and padding | Record `T_valid`, `P_valid`, `P_aligned`, stage-specific `P_work`, `E_hit`, bucket count and `P_aligned/P_valid`. | Compare aligned-full, indexed-valid and token-deduplicated work where semantics allow; include graph capacity much larger than active rows and provider max-padding amplification. |
| Top-k weights | Equal normalized and seeded random normalized FP32, identical across candidates. | Real sigmoid/noaux traces, dominant weights, routed-scaling variants. |
| Factor sharing | Fully per-expert and current shared-outer pattern: gate/up A shared, down B shared. Exercise both before freezing fused interfaces; compare token/adapter-deduplicated shared A and weighted-rank-reduced shared B with repeated-pair controls. | Arbitrary sharing signatures only with a real consumer. Never apply shared-B rank reduction to per-expert B. Model shared experts are a separate later dimension. |
| Factor sharding/storage | Resolve every site’s local A/B shape under TP/EP/MoE-DP; global-ID/global-factor mask is the initial control. | Load-time EP-local contiguous factor packing/local IDs versus global resident factors; report adapter-load/update cost, per-rank memory and forward bytes. Never slice/copy noncontiguous expert factors every forward. |
| Target mask/slices | All routed experts; both gate/up slices; non-gated value slice. | Gate-only, up-only, random slice/expert subsets, no-target groups, arbitrary active slices without zero materialization. |
| Gate/activation semantics | Gate-first contiguous SwiGLU and non-gated ReLU2. | Up-first, gate/up interleave, provider alpha/clamp/bias and future declared activations. |
| LoRA/input/output dtype | BF16 input/A/B/output; FP32 reference; explicit BF16 versus FP32 split accumulation. | FP16 factors where supported; provider BF16 bridge; caller-selected FP32 destination; mixed accumulation only as declared policy. |
| Base provider/quant | BF16 DeepGEMM first. | FP8 W8A8 block-128; NVFP4 W4A4 group-16; Marlin W4A16 checkpoint group; Nemotron mixed expert/outer quant. |
| DSL/implementation | PyTorch/FP32 reference plus Triton schedule laboratory. Treat algorithm and technology as independent labels. | Selected CuTe DSL implementations for raw indexed/SGMV-style, segmented, grouped, and fused boundaries where they resolve implementation-quality or fusion uncertainty; CUDA/provider libraries where appropriate; cuTile only as an optional capability-gated experiment. Existing TRTLLM/CUDA remains a numerical and whole-provider baseline. Do not implement every algorithm in every technology. |
| Kernel tile/launch policy | Kernel-legal candidates drawn from `BM={1,2,4,8,16,32,64}`, `BN={16,32,64,128,256,512}`, `GROUP_M={1,4,8}`, rank/`BK` chunks `{8,16,32,64}`, warps `{1,2,4,8}`, stages `{1,2,3,4,5}`, vector width, CTA/grid width, whole/looped rank, and split/stream-K. | Add divisibility/tail classes `N%BN`, `K%BK`, masked versus padded versus safe affine addressing, flattened/2-D/3-D program mapping, and independent H200/GB300 tuning. A tile is a performance choice, never a support condition. |
| Writeback and output state | Fresh overwrite, same-stream ordered load-add-store, or atomic only for true multi-writer reduction; nonzero base destination. | Partial-slice output gaps, caller-selected destination dtype, collective-owned output. Unsynchronized base store plus LoRA atomic RMW is invalid. |
| Cache/input residency | Rotated-buffer producer-realistic cold L2 is the primary `K0` performance number; report working-set bytes actually touched, rotation footprint/L2 size and effective bandwidth. | Hot-L2 diagnostic and real upstream-producer chain. Do not size rotation from full allocated weights when only `E_hit` rows are touched. |
| Metadata lifetime | Build/update once per adapter load, batch, graph refresh, layer, or replay as declared. | Cache hit/miss and gate/down plan reuse; quantify graph-node amplification across 40–75 MoE layers. |
| Forward/graph/streams | Real decode and extend/prefill; eager and fixed-shape raw CUDA graph; serial prefill; decode overlap toggle; explicit main/side work region, fork/join, stream priority and output allocation owner. | Mixed/verify, separate no-LoRA/LoRA graphs, recapture/transitions, breakable prefill, allocator reuse pressure and resource-partition diagnostics. Correctness runs under sustained replay, not capture alone. |
| PDL | `off` and valid producer-consumer `on` chains, orthogonal to eager/graph and serial/two-stream mode. | Producer subset `{G1 only,G2 only,both}` where supported. Keep only when a timeline shows reduced critical path; never infer benefit from back-to-back identical standalone launches. |
| Device | H200 SM90 and GB300 SM103, independently tuned. The available 8 H200s and 4 GB300s may run deterministic independent case shards. | H100, B200/GB200, real 8-GPU and two-node MNNVL. Parallel single-GPU workers are not distributed evidence. |

Do not run a full Cartesian performance matrix. Use two model-coverage levels:

- **early guardrails**, before the Qwen fused ABI is hardened: one or two cheap
  correctness/local-shape cells for large-H Qwen/Kimi/GLM, latent-H high-top-k
  Nemotron Super, odd-I Nemotron Nano, non-gated ReLU2, partial slice targets, and
  shared-outer factors. These are compatibility alarms, not tuning claims;
- **later model-scale tuning**, after `C2/C3` are stable: each model at `R=32`, one
  adapter, seeded iid routing, eager+graph, decode `T={1,32,128}` and prefill
  `T={256,2048}`; then rank sweeps on Qwen3.5-397B and Nemotron Super at
  `R={8,16,32,64,128}`, `T={1,32,256,2048}`;
- adapter/fragmentation sweep on Kimi at `T={32,256,2048}`. At the `T=32,R=64`
  anchor, run the compact core `(L_active,B_base,L_capacity)` cells
  `{(0,1,8),(1,0,1),(1,0,8),(1,1,8),(3,0,8),(4,1,5),(7,1,8),(8,0,8)}`;
  use balanced and skewed adapter rows where meaningful and 50/50 plus skewed
  base/LoRA rows for mixed cases. Replay a capacity-8 graph through
  `(L_active,B_base)={(0,1),(1,0),(1,1),(3,0),(7,1),(8,0)}`, using contiguous,
  sparse/high and permuted slot IDs plus slot eviction/reassignment. Run the
  capacity-5 mixed-full case under graph replay as a non-power-of-two check;
- routing sweep on GLM over balanced/iid/80-20/no-local cases;
- topology proxies for every model at `R=32`, `T={32,512}`;
- add stress/non-power-of-two cells to correctness, not every performance sweep.

### 8.5 Topology matrix

A world-size-one case models local geometry only. Label it
`<resolved topology> local-shape proxy`, for example `TP4/EP4/DP1 local-shape proxy`,
never distributed performance.

| Local proxy | Resolved local effect | Missing until real distributed execution |
|---|---|---|
| `TP1/EP1/DP1` | Full experts/intermediate/tokens; no-communication control. | All communication. |
| `TP4/EP1/DP1` | Full experts, `I/4`. | TP down/finalize all-reduce or reduce-scatter and overlap. |
| `TP4/EP4/DP1`, rank 0 and last rank | `E/4`, full `I`, rank-specific offsets and valid/`-1` routes. | Dispatch/A2A/combine, cross-rank reduction and imbalance. |
| `TP4/EP1/DP4` | Full experts/full `I`, approximately `T/4`. | Token gather/scatter, uneven ranks and graph buckets. |
| `TP8/EP2/DP2` | `E/2`, `I/2`, approximately `T/2`. | All real collectives and overlap. |

Required real-distributed runs:

1. `TP4/EP1/DP1`, standard dispatcher;
2. `TP4/EP4/DP1`, ordinary standard EP with all-local/mixed/no-local routes;
3. `TP4/EP1/DP4`, including uneven local token counts;
4. graduation: `TP8/EP2/DP2`, FlashInfer A2A, DeepEP normal/low-latency where
   supported, then two-node EP/MNNVL.

Every distributed correctness subset covers local/global/sentinel IDs,
separate/fused shared experts, no-LoRA/all-LoRA/mixed rows, and graph replay. Bracket
core time separately from dispatch/combine/collectives.

For every topology, materialize a per-site factor-sharding table before allocating
the benchmark. At minimum record local gate/up-A, gate/up-B, down-A and down-B shapes,
whether each factor is replicated, TP-sharded or expert-sharded, `E_hit`, touched bytes
and destination collective. Add a matched-provider TP8/EP1 versus TP8/EP8 sweep over
batch size: the optimal topology can cross over with batch and must also pass
cross-rank autotune/capture-stability checks.

### 8.6 Measurement scopes, metrics, and advancement rule

| Scope | Included work | Purpose |
|---|---|---|
| `K0 kernel` | Exact primitive with only its implementation-required preparation prebuilt; this may be raw IDs and no row plan. | Schedule and tile attribution. |
| `O0 operator` | Canonical inputs -> only the selected implementation's requested preparation, possibly none -> primitive. | Routing, padding, descriptor and avoided-preparation tax. |
| `M0 local MoE` | Base prepare/W13 + both LoRA injection points + activation/quant + W2 + finalize, excluding communication. | Critical-path provider comparison and fusion selection. |
| `D0 distributed MoE` | Actual dispatcher -> runner -> combine and collectives. | TP/EP/MoE-DP correctness, communication and overlap. |
| `E0 server` | Full model, scheduler, graph, adapter residency/lifecycle and fixed-batch requests driven by `sglang.benchmark.one_batch_server`. | Product-level graduation. Do not substitute `bench_serving` for this refactor's controlled E0 comparison. |

For every timed stage, publish correctness reference, eager and graph-replay timing,
prebuilt and route-inclusive timing where applicable, cold/hot/producer cache state,
median plus dispersion, `P_work/E_hit`, touched bytes/effective bandwidth, launches,
graph nodes, workspace/peak memory, compilation/capture separately, and representative
tiny/medium/large profiles with event stalls, SM/tensor utilization, DRAM/L2 traffic,
registers and spills. Poison inactive ranks, missing slices, non-owned/sentinel lanes,
padded rows and recycled high-rank-to-low-rank slots with NaN/Inf; use a nonzero base
destination to detect overwrite. Correctness must include sustained graph replay and
allocation pressure because throughput can remain unchanged while replay is corrupt.

A variant advances only if it wins or is neutral at the next wider boundary. A `K0`
win that loses after routing, quant conversion, synchronization, graph replay, or
provider finalization does not become production policy. Tune Hopper and Blackwell
independently.

A trace review is mandatory before advancing any shortlisted candidate that changes a
fusion boundary, routing work domain, launch count, stream/event placement, allocation
owner, CUDA graph structure, PDL edge, or collective. Tile-only candidates do not each
need a trace; trace the shortlist and any suspicious shifted-work win.

### 8.7 Phased benchmark implementation and final disposition

| Priority | Goal and candidate set | 2026-07-22 campaign disposition |
|---|---|---|
| `P0` | Establish matched `N0`, materialized `C0`, two-stream `C1`, stock/legacy, and experimental TRTLLM controls at the same semantic boundary. | **Complete.** Same-device internal controls and a real Qwen `one_batch_server` bracket ran. Historical pre-planner SGL trailed by 12.56%-14.71%; final default C2 trails by 9.43%, 9.44%, and 10.39%, while opt-in C3 trails by 5.12%, 5.54%, and 10.48% at BS1/16/32. No universal winner is claimed. |
| `P1` | Complete the BF16 local target across indexed/direct, segmented, grouped, qualifying BMM, one-shot, shared-factor, token-owned, and fused-consumer candidates. | **Complete as a measured local decision campaign.** Both devices cover K0/O0/M0, ranks 16/32/64/128 plus R8 guardrails, eager/graph, hot/cold, IID/skew, active/mixed/base, and shared-outer. C2/C3 are promoted only in their measured domains; C0 remains the safe fallback. |
| `P2` | Guard the semantic ABI across large/latent H, high top-k, odd I, non-gated ReLU2, partial slices, and provider padding. | **Complete for local correctness/compile boundaries.** Qwen, Kimi, GLM, and Nemotron-derived shapes are retained. Nemotron ReLU2 is benchmark-only and must fail fast at production attachment until promoted; this is not broad model-server graduation. |
| `P3` | Harden BF16 policy, rank support, topology proxies, graph transitions, and an E0 server bracket. | **Complete for the Phase-1 scope.** Production A is legal through R128; R8 uses masked physical-R16 execution; no-LoRA/LoRA graphs are separate; local proxies remain labeled WS1; real D0 is recorded separately. |
| `P4` | Add FP8 W8A8, NVFP4 W4A4, and Marlin W4A16 behind the same logical contracts and measure exact provider boundaries. | **Provider-neutral C0 seam validated, with bounded attachment.** H200 covers FP8/Marlin; GB300 covers FP8/Marlin plus a native-NVFP4 synthetic provider fixture at the provider execution boundary. Active FP8 rejects static activation; Blackwell requires resident packed UE8M0 scales. Marlin W4A16 is attachable, and its per-call workspace/dirty-destination repair passed the terminal H200/GB300 smoke. Native CuTe DSL NVFP4 W4A4 remains synthetic/testbed-only and is not serving-reachable. Base GEMMs are provider-native while LoRA/bridge arithmetic remains BF16/materialized. Broad checkpoint/server attachment and provider-native fused LoRA are not graduated. |
| `P5` | Validate real TP/EP/MoE-DP dispatch/combine/collectives and model physical shared experts. | **Distributed topology complete; physical shared support bounded.** Real H200 topologies and two-node GB300 TP8/EP2/DP2 pass, and global/local ID transport is exercised. Physical shared IDs are promoted only for Standard contiguous/EP1 or safe non-per-rank-remapped layouts. Per-rank physical shared EP>1, advanced A2A, and sink-shared-expert optimization remain unpromoted. Shared-outer's evidence-bounded selector passed 9 host tests and 56 tests plus 10 subtests on each device, including physical-ID composition and PDL pairing. |
| `P6` | Validate the controlled server and lifecycle boundary with `sglang.benchmark.one_batch_server`, never `bench_serving`. | **Complete for the controlled Qwen BF16 campaign.** Base/active/mixed traffic, load/unload/eviction/slot reuse, eager/graph transitions, H200/GB300, and multi-node D0 are retained. Broad quantized/model E0, MTP/EAGLE, and the Phase-3 adapter control plane are not implied. |

A terminal GB300 control also found a pre-existing unmasked per-token FP8 JIT/AOT
bit-exact mismatch and reproduced one failing case against official main. The
branch-added masked provider-padding cases pass. This is not a LoRA-provider
regression or a Phase-1 acceptance blocker, but it is now an explicit later general
quant-kernel backlog item; raw control logs are retained under `final_validation/`.

Triton is the first schedule laboratory, not the definition of any algorithm. Do not
implement every algorithm in every technology. Select independent CuTe DSL prototypes
where they can test raw indexed/SGMV-style ownership, segmented/grouped tensor-core
mapping, or a fused consumer that Triton cannot express efficiently. CUDA and provider
libraries remain valid competitors. cuTile is optional and only enters when its
capability surface covers the chosen endpoint. Promote only winners that survive the
same M0 and server boundary.

#### Parallel worker policy

The available 8 H200 GPUs and 4 GB300 GPUs should be used concurrently for independent
WS1 work whenever experiments resume:

- deterministically shard correctness cases, coarse schedule screens, compile probes,
  and model guardrails across devices;
- give each worker unique artifact, cache, temporary, and port namespaces and record
  physical GPU UUID, software image, source hash, case hash, and resolved configuration;
- keep each candidate and its controls on the same physical GPU;
- rerun canonical timing winners isolated and counterbalanced on a quiet GPU; do not
  publish a fine delta measured only during a saturated fan-out;
- reserve Nsight Systems/Compute profiling for an exclusive device and bounded range;
- suspend local fan-out during D0, because real multi-GPU communication owns the
  participating devices.

Parallel workers increase coverage and reduce wall time. They do not turn WS1 runs into
TP/EP/MoE-DP evidence.

### 8.8 Profiling, CUDA graph, cache, and PDL protocol

Overall latency answers whether a candidate wins; it does not prove why it wins or
that the intended dependency graph is correct. Use this ladder:

| Pass | Tool and mode | Question answered | Rules |
|---|---|---|---|
| Correctness | PyTorch/provider oracle in eager, capture and sustained replay | Are IDs, weights, tails, output ownership and quant semantics correct? | Run before timing. Include poisoned inactive/sentinel/padded regions, nonzero base output, slot reuse and allocation pressure. |
| Canonical timing | CUDA events or graph-replay timing without a profiler | What is the deployable latency and dispersion? | Publish eager and graph separately. Use cold-L2 rotation for memory-bound `K0`, hot-L2 as a diagnostic, and the real producer chain at `O0/M0`. Do not use a traced run as the canonical number. |
| Timeline | Nsight Systems with stage NVTX ranges | Are launch order, gaps, events, streams, overlap, graph nodes, casts/memsets/copies and collectives correct? | Trace shortlisted structural changes with `--cuda-graph-trace=node`; use graph-level tracing for lower-overhead sanity. Compare eager and graph. On Blackwell, evaluate hardware-based trace when available for many short kernels. |
| Kernel counters | Nsight Compute directly or `NVIDIA/nsight-python` | Why is a kernel limited: DRAM/L2, occupancy/waves, tensor use, atomics, registers/spills, branches or tails? | Use for `K0/P1` parameter sweeps and CSV/regression analysis. Prefer one kernel per annotation; use range replay/combined metrics only when supported. NCU replay is not evidence of real overlap. |
| Distributed/server | Nsight Systems plus server metrics | Does the complete graph/dispatcher/collective/lifecycle retain throughput and correctness? | Profile bounded windows; report steady decode, latency, graph transitions and cross-rank stability separately from trace overhead. |

`nsight-python` is useful because it automates Nsight Compute across resolved tile and
schedule configurations and can return a DataFrame. It is optional benchmark tooling,
not a runtime dependency and not a replacement for Nsight Systems. Its NVTX annotation
is thread-local, nested annotations are unsupported, and range replay supports only a
subset of CUDA APIs; multi-thread or multi-stream server traces therefore stay in
Nsight Systems.

Every production-shaped trace uses stable NVTX names for `G0`, `R0`, `Q1`, `A1`,
`G1`, `B1`, `S3/A2`, `G2`, `B2/F`, shared expert, collectives and graph refresh. Record
producer-ready, side-ready, join and final-ready timestamps so overlap is measured as
the fork-to-join critical path rather than the sum of kernel durations.

CUDA Graph is a separate execution mode, not merely a timing trick:

1. Establish eager correctness/timing first.
2. Capture each fixed `N0/C0-C5` topology as its own graph family with stable buffers,
   streams, events and allocation ownership.
3. Report capture/instantiate/upload separately from steady replay.
4. Include metadata refresh and static-input copy cost at `O0/M0` when production pays
   it.
5. Inspect node-level traces for lost two-stream overlap or hidden serialization.

PDL is orthogonal to CUDA Graph and is tested as a dependency-edge policy. The
campaign added an explicit off/on seam and paired producer/consumer protocol. A
producer signals only when its selected direct consumer executes the matching
device-side wait; generic consumers retain ordinary stream ordering. The retained
matrix covers isolated and chained eager/graph modes on both devices. PDL remains an
opt-in measured policy, not a global architecture switch.

| Measurement | PDL setting | Interpretation |
|---|---|---|
| Isolated/profile-comparable `K0` | Off by default | Prevent repeated identical graph nodes from opportunistically overlapping launch tails and understating the real consumer wait. |
| Real producer -> consumer chain | Off/on | Primary PDL experiment. Producer triggers only after completing all reads needed for safe overlap; consumer synchronizes before dependent reads. |
| `M0` eager | Off/on | Measures launch/headroom benefit without graph effects. |
| `M0` graph | Off/on with captured programmatic edge | Deployable result; verify the edge and critical path in the timeline. |

PDL requires compute capability 9.0 or newer, offers opportunistic rather than
guaranteed overlap, and can be represented in CUDA Graph stream capture or explicit
programmatic edge data. Do not enable it globally merely because the device supports
it. Fusion may remove the producer-consumer edge entirely; two-stream execution may
already expose independent work; resource contention can erase the gain. Promote PDL
only when correctness passes and the `M0` timeline/latency improve on that device.

Profiling references:

- https://docs.nvidia.com/nsight-python/
- https://docs.nvidia.com/nsight-systems/UserGuide/
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html

Only after the matrix is stable should `--lora-execution-engine=auto`
opportunistically select `sgl_lora`.

Graduation gates:

- no-LoRA overhead at or below 1%;
- new route matches or beats legacy throughput/latency within agreed noise;
- local expert memory scaling under EP;
- stable selection/fallback telemetry;
- validated H100/H200 and B200/GB200/GB300 coverage.

## 9. Phase 2 — dense and special-layer execution

Move layer families behind the same top-level execution engine while preserving
`--lora-backend` as a provider choice.

1. Split the monolithic legacy monkeypatch installer, or keep it wholly enabled until
   all coupled families migrate.
2. Extend the central live-module site compiler and tri-state `ModelLoRAExtension`
   bridge across dense/special layer classes. Existing `get_hidden_dim`,
   `get_stacked_multiply`, `should_apply_lora`, and `supported_lora_modules` remain
   compatibility inputs, not the new ABI.
3. Define backend-neutral split A/B calls and workspace ownership.
4. Migrate ordinary column/row/replicated linear layers.
5. Replace hardcoded string normalizers family by family with structured
   `ResolvedWeightBinding` and declarative fusion/slice rules.
6. Introduce explicit `LoRASliceSpec`, `CanonicalSliceWeights`, and
   `LinearDeltaPlan`: allow absence at any schema-optional position, independent
   per-slice rank/scale, and shared factor references without eager A repetition.
7. Replace semantic `run_qkv_lora`/`run_gate_up_lora` APIs with one sliced-linear
   provider operation. Retain single-slice, equal-two, and generic N-slice kernels as
   provider-selected fast paths.
8. Fold row-parallel LoRA partials into the existing base output collective where
   algebra and sharding permit one all-reduce.
9. Add measured shrink overlap and consumer-boundary expand.
10. Migrate QKV and merged column sites, plus MLA/Mamba/GDN logical-weight binding
   groups with explicit execution variants.
11. Migrate embeddings and ordered lm-head/multi-pass assignment views.
12. Remove legacy hooks one model/family at a time only after schema comparison tests
   prove the central compiler plus sparse extension is equivalent.
13. Remove legacy monkeypatches one layer family at a time only after the installer is
   separable and that family passes parity and graph tests.

## 10. Phase 3 — adapter control plane

Use “adapter control plane” rather than “request management.” It owns:

- alias-independent immutable `AdapterKey(artifact identity, immutable revision)`;
- mutable public aliases that resolve to immutable adapter keys;
- KV-cache identity and invalidation;
- artifact resolution and validation;
- centralized adapter-format readers that produce structured raw weight records for
  the schema compiler;
- prepare/publish/abort distributed updates, with every fallible action in prepare
  and a no-fail pointer flip at publish;
- CPU/GPU byte budgets and residency;
- pinned-slot admission;
- residency-aware scheduling and prefetch;
- native canonical batch assignment;
- explicit register/activate/deactivate/evict/unregister lifecycle operations.

Legacy bridges are removed only after the execution engine consumes native control-
plane objects and all production configurations have graduated.

## 11. PR sequence

The phase-numbered sections above group capabilities; they do not require quantized or
distributed work to land before the BF16 kernel target. The review-approved landing
order follows section 8.7 and keeps each change independently reviewable:

1. **1a scaffold (current):** naming, selector/provider decoupling, implied virtual
   experts, package isolation, current BF16 materialized runner, tests, and docs.
2. **BF16 neutral baseline:** matched `N0/C0/C1` plus stock/legacy and experimental
   TRTLLM whole-M0 controls at the same output contract. No production selector change.
3. **BF16 `C2`:** optional route views, selected A schedules, fused gate/up-B plus
   activation and down-A, then down-B/finalize; include shared-outer controls and
   selected CuTe DSL competitors where they answer a concrete uncertainty.
4. **BF16 `C3`:** A-only overlap and graph topology, retaining `C0/C1` controls and
   leaving `C4` down-overlap for a later isolated experiment.
5. **Cross-model guardrails:** large/latent/odd dimensions, non-gated activation,
   partial slice targets, shared-outer, and provider-padding correctness before the
   fused ABI hardens.
6. **BF16 graduation:** rank 8/16 performance, supported rank 128, wider adapters and
   routes, local topology proxies, LoRA-off/on graphs, and single-rank server evidence.
7. **Minimal execution-contract bridge:** add canonical assignment, residency,
   completion, typed site, provider-plan, and stable-buffer objects only as their first
   quantized/distributed consumers require them; do not pull Phase-3 control-plane work
   forward wholesale.
8. **Independent quant providers:** FP8, NVFP4 W4A4, and Marlin W4A16 in separate
   commits/PRs, each with native provider-output contracts and its own graduation.
9. **Distributed and model-shared execution:** ordinary EP/TP/MoE-DP, standard A2A,
   execution-group agreement, collectives, then model shared experts and advanced A2A.
10. **Full server/multi-node graduation:** real lifecycle correctness, graph
    transitions, supported A2A, and two-node MNNVL.
11. **2.x:** dense and special-layer families.
12. **3.x:** full adapter control plane and legacy retirement.

Each PR must leave the default production path usable and preserve an explicit legacy
escape hatch until its replacement has graduated.
