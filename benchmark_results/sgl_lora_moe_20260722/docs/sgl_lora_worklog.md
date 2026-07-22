# SGL LoRA refactor worklog

Updated: 2026-07-22  
Branch snapshot: active `sgl-lora` at `f2f406e056` after the final upstream rebase;
archived post-Phase-1a prototype branch
`sgl-lora-pre-redesign-backup-20260721` at `72055b46fd`  
Current phase: **Phase 1 — MoE execution graduation candidate**

Authoritative references:

- `sgl_lora_lifecycle_and_orchestration_design.md` — target architecture and contracts.
- `sgl_lora_refactor_plan.md` — phase ordering and graduation goals.
- This worklog — current implementation status, known gaps, and immediate next work.

This is a status ledger, not another architecture specification. A gap listed here
is neither supported nor guaranteed to fail cleanly in the current prototype.
During development we keep only small kernel-local invariants and focused tests for
implemented behavior. Broad support-matrix validation belongs near graduation, once
the intended matrix is actually implemented and stable.

## Status legend

- **DONE** — implemented; evidence is recorded below.
- **BOUNDED** — implemented or exercised only inside the explicitly named attachment,
  layout, or provider envelope; it is not broad product graduation.
- **IN PROGRESS** — part of the current commit or active cleanup.
- **PLANNED** — accepted next work.
- **DEFERRED** — intentionally later or waiting for explicit approval.
- **BLOCKED** — cannot proceed without an external dependency.

`WS1` means world-size-one local execution. It can reproduce rank-local tensor
shapes, but it is not evidence for distributed collectives, communication overlap,
or multi-rank graph safety.

## 2026-07-22 execution checkpoint — IMPLEMENTED AND MEASURED

The “planning only” section below is retained as the pre-campaign decision record.
Its missing-work statements are superseded by this checkpoint and by the final audit.

| Milestone | Status | Result / boundary |
|---|---|---|
| Third-party critical/validity review | DONE | Fixed fused shared-expert virtual IDs, vacuous delta checks, route labeling/sampling, O0 attribution, loaded-host/cold controls, counterbalancing, silent-wrong contracts, PDL control, and per-lane artifact durability. The final root manifest and finding-by-finding disposition are bundled with the evidence. |
| BF16 local execution | DONE | Production host planner selects measured C0/C2/C3 paths by provider, phase, graph mode, rank, row contract, and schedule; rank 8/16/32/64/128, FP32 destination, non-unit routed scaling, and gated SwiGLU are covered. ReLU2 has benchmark-contract guardrails only and must fail fast at production attachment until promoted. |
| Neutral E0 comparison | DONE | `one_batch_server` real-model bracket completed. SGL is not universally faster: the historical pre-planner path was 12.56%-14.71% behind experimental TRTLLM; final default C2 is 9.43%, 9.44%, and 10.39% behind, while opt-in C3 is 5.12%, 5.54%, and 10.48% behind at BS1/16/32. Base traffic is near parity. |
| Graph/lifecycle | DONE | Separate base/adapter graphs plus base→adapter→base, mixed replay, load/unload, eviction, slot recycling, and same-name reload passed on H200 and GB300. |
| Distributed D0 | DONE | Real H200 TP/EP/MoE-DP collectives and two-node GB300 TP8/EP2/DP2 MNNVL execution passed; local-shape proxies remain labeled separately. |
| Quant providers | BOUNDED | Provider-neutral C0 seams were exercised, but broad checkpoint/server attachment is not graduated. No-LoRA calls the resident quant method. Active FP8 rejects static activation; Blackwell FP8 requires resident packed UE8M0 scales. Marlin W4A16 is attachable; its per-call workspace/dirty-destination repair passed the terminal `f2f406e056` H200 and GB300 smoke. Native CuTe DSL NVFP4 W4A4 remains synthetic/testbed-only and is not serving-reachable. |
| Shared semantics | DONE — BOUNDED | The evidence-bounded shared-outer selector passed 9 host tests and 56 tests plus 10 subtests on each of H200 and GB300, including the TorchNative merged-bound fix, physical shared-ID composition, and PDL pairing. Physical shared experts are promoted only for Standard contiguous/EP1 or otherwise safe non-per-rank-remapped layouts; per-rank physical shared EP>1 and advanced A2A remain unpromoted. |
| Algorithm technologies | DONE | Indexed/aligned/segmented/BMM/one-shot/fused-finalize candidates were screened independently from Triton/CuTe DSL/CUDA technology. The evidence-only Blackwell CuTe DSL/TMA K0 result is 73.152 us versus Triton 87.520 us (-16.42%); static M0 is invalid under route mutation, so no production integration is claimed. |
| Mixed-rank policy | MEASURED | Load-time physical-rank packing wins materially for R32/R128 and R64/R128. The benchmark static rank planner serializes graph keys; production serving residency and multi-bucket ownership remain a Phase-3 control-plane obligation. |

Durable evidence lives under
`benchmark_results/sgl_lora_moe_20260722/` with per-directory manifests and
SHA-256 files. The original H200 pod and user-provided GB300 devbox were reused; the
temporary two-node Slurm job used for D0 was explicitly cancelled after collection.

Route-plan memoization in the promoted runner is deliberately local to one MoE layer
invocation. Gate/up and down sites may reuse an exactly matching route view during
that invocation; the result is not a whole-forward or cross-layer cache. Extending
the lifetime would require independent ownership and invalidation evidence.

## 2026-07-22 design-review checkpoint — PLANNING ONLY

No kernel, runner, benchmark, branch, GPU node, or retained evidence changed in this
checkpoint. It records the reviewed plan before execution resumes.

| Decision | Recorded plan |
|---|---|
| Algorithm versus implementation | Raw indexed/direct, true segmented SGMV, grouped GEMM, qualifying BMM, token-owned reduction, shared-factor specialization, and fused-consumer ownership are algorithms. Triton, CuTe DSL, CUDA, cuTile, and provider libraries are implementation technologies. CuTe DSL may implement raw indexed/SGMV-style work as well as segmented, grouped, fused, FP8, and NVFP4 work. cuTile is optional and capability-gated. |
| Routing metadata | Canonical IDs and assignments are stable semantics. The current aligned virtual-expert plan is only one implementation view. A selected kernel may consume raw IDs, segmented offsets, an aligned plan, producer-packed metadata, or a provider-private view; unused preparation must not be built. |
| Current gate/up contract | Phase 1a already has one valid BF16 contract: standard gate-first contiguous W13, ordinary gated SwiGLU, canonical `[T,K,2I]` LoRA delta with the gate slice followed by the up slice, BF16 W2 input, and a pair-domain BF16 activation bridge for down-A. |
| Target gate/up/provider contracts | Name logical activation math, logical projection slices/target masks, and provider-private physical IO separately. This is to avoid turning the current BF16 representation into the universal ABI for non-gated, interleaved, biased, FP8, or NVFP4 providers—not to replace a missing current contract. |
| Shared work | Shared-outer LoRA enters the early BF16 routed-MoE matrix. Compare shared gate/up-A pair repetition with once-per-token/adapter computation, and shared down-B pair expansion with weighted rank reduction followed by one B. Model shared experts remain a later, separate integration. |
| Rank order | Initial model-scale performance work uses rank 32/64/128. Rank-16 correctness remains early; model-scale rank 8/16 performance is a BF16 graduation requirement. |
| Baselines | The active campaign compared internal SGL variants only. A matched experimental-TRTLLM and stock/legacy whole-M0 bracket remains required; no current result establishes SGL superiority over TRTLLM at any shape or token count. |
| Execution order | Establish matched `N0/C0/C1` controls; build BF16 Qwen `C2`, then `C3`; run early cross-model guardrails; finish BF16 graduation; add FP8/NVFP4/W4A16 independently; then real distributed execution, model shared experts, and full server/multi-node graduation. `C5` remains a diagnostic/fallback; `C4` down-overlap is last. |
| Topology evidence | One GPU may simulate a rank-local TP/EP/MoE-DP tensor shape, labeled for example `TP4 local-shape proxy`. It cannot establish collectives, A2A, imbalance, communication overlap, or D0 graph safety. |
| Available GPU parallelism | When work resumes, deterministically shard independent WS1 correctness, compile, coarse-screen, and model-guardrail cases across the available 8 H200 and 4 GB300 GPUs. Keep candidate/control pairs on the same physical GPU, use unique artifacts/caches/ports, rerun canonical timings isolated and counterbalanced, profile exclusively, and stop local fan-out during D0. |

The implementation sequence and complete comparison matrix live in sections 8.1–8.8
of `sgl_lora_refactor_plan.md`. The audit document records what has and has not actually
been measured; this checkpoint must not be cited as new GPU evidence.

## Phase 1a objective

Establish the smallest understandable MoE LoRA execution path:

1. one explicit execution-engine selector;
2. one isolated package;
3. virtual-expert semantics;
4. one standard-layout BF16 base provider;
5. one semantic MoE pipeline with the two LoRA injection points;
6. focused selector, kernel, and runner tests.

Phase 1a is a development vertical slice. It is not the final model, quantization,
distributed, graph, or adapter-format support matrix.

## Done in Phase 1a

| Area | Status | Current result |
|---|---|---|
| Execution selection | DONE | `--lora-execution-engine=sgl_lora`; legacy remains the default; `--moe-runner-backend=sgl_lora` is normalized as a temporary input alias. |
| Virtual experts | DONE | Selecting `sgl_lora` implies virtual-expert semantics; no second semantic flag is required. |
| Package boundary | DONE | Runtime, virtual-expert routing, shrink/expand kernels, and side-stream resources are owned under `python/sglang/srt/lora/sgl_lora/`; the new engine has no import or runtime reference to `trtllm_lora_temp`. |
| Legacy isolation | DONE | The default legacy path does not import the new package. The legacy experimental master switch does not select the new engine. |
| Provider seam | DONE | `MoeLoraBaseGemm` separates the semantic pipeline from the concrete base GEMM stages. |
| BF16 provider | DONE | Standard-layout BF16 `w13`/`w2` weights use DeepGEMM masked grouped GEMMs. |
| Semantic pipeline | DONE | Prepare/route → base gate-up → add gate/up LoRA before activation → activation → base down → finalize → routed down LoRA. |
| Activation boundary | DONE | The S3 Triton kernel joins masked base-GEMM rows with canonical expanded LoRA rows and produces the down-LoRA activation input. |
| Gate/up slices | DONE | Gate and up are explicit logical slices. Active direct B selects aligned-flat or a two-slice masked-tail grid; active generic B launches matching equal-width zero-copy A/B/output views per slice. The broader uniform/compiled-ragged/descriptor-ragged study remains archived benchmark evidence, not active source. |
| Execution policy | DONE | Serial execution is default; two-stream gate/up overlap is an explicit, default-off policy. |
| No-adapter eager path | DONE | Eager batches with no active adapter invoke the base layer's resident quant method/provider. They do not synthesize or force a Triton quant provider, so resident packing and scale contracts remain authoritative. |
| Latest-main integration | DONE | Rebased to the relocated EP-MoE kernels and the current `post_reorder_deepgemm` finalize wrapper. |
| Focused tests | DONE | Selector/isolation, activation, direct expand (including odd half-width fallback), gate/up A+B, unified-runner, and isolated schedule-benchmark coverage exist. |

## Current routing and kernel boundary — clarified

Phase 1a intentionally does **not** replace model routing, top-k selection, or the
standard EP dispatcher. Its current normalized input contract is:

| Value | Phase 1a contract |
|---|---|
| Hidden states | `[T, H]` BF16 |
| Top-k IDs | `[T, K]` INT32 local expert IDs; `-1` means non-local/invalid |
| Top-k weights | `[T, K]` FP32 |
| Token adapter map | `[T]` INT32 memory-pool slot; `-1` means base-only |
| Logical output | `[T, H]`, input dtype |

The existing `StandardDispatcher` translates global expert IDs to local IDs once.
The base path then reuses `moe_ep_deepgemm_preprocess`, masked DeepGEMM, and
`post_reorder_deepgemm`; it represents routing with `hidden_permuted`, `masked_m`,
and `src2dst`. The current grouped/direct/generic LoRA path forms
`virtual_expert = adapter_slot * E_local + local_expert`, then reuses the existing
`moe_align_block_size` primitive to produce
`sorted_token_ids`, one `expert_id` per padded block, and
`num_tokens_post_padded`. The isolated `sgl_lora` package owns this virtual-ID
transformation, its routing cache/staging, and the LoRA kernels, but it is not a new
server-level router or EP transport.

This aligned route plan is current-source behavior, not the target kernel/backend
contract. Raw indexed/direct, true segmented SGMV, token-owned down, shared-factor
deduplication, BMM on qualifying regular groups, and provider-fused consumers may
request different views or consume canonical IDs directly. The planner must not pay
alignment or descriptor costs for a selected family that does not consume them.

This is a logical normalization boundary, not a claim that every existing
quantization/GPU backend already has the same physical tensors. Packed activations,
scales, weight layouts, global-ID providers, and internal permutation formats remain
provider-private. New providers should preserve the common logical top-k/pair
contract or add an explicit adapter at this boundary rather than leaking their
physical layouts into LoRA orchestration.

For Phase 1a the semantic base result happens to use the BF16 input dtype. The
general contract is stricter: the caller-owned additive destination defines the
final LoRA-B result dtype. LoRA-B accumulates in FP32 and converts once into that
destination for store/add/reduce. LoRA-A split-K workspace precision is an
independent choice. If no base destination exists, the future dense planner may
request an explicit output dtype; otherwise it resolves the site's floating compute
dtype rather than guessing from a packed or quantized input tensor.

## Current cleanup result

- **DONE:** Removed the broad Phase 1a launch capability validator from
  `server_args.py`.
- **DONE:** Removed adapter-wide routed target-pair validation and its unrelated
  management-layer changes from this commit.
- **DONE:** Removed rank, model-metadata, and fixed workspace-admission guards from
  the execution path.
- **DONE:** Kept only selector contradictions and immediate low-level kernel
  invocation invariants.
- **DONE:** Replaced the large rejection-test matrix with focused behavior and
  numerical tests.
- **DONE:** Focused CPU selector/wiring suite: 9 tests plus 10 subtests passed.
- **DONE:** Final changed-file pre-commit checks and `git diff --check` passed.
- **DONE:** GB300 GPU correctness and the isolated expand schedule testbed ran on
  `rxs` job 1293 (`rdx-gb300-r01-c017`); the job was cancelled afterward and
  `rxs ls --mine` reported no remaining jobs.
- **DONE:** 25 focused GPU kernel/runner tests passed. The combined run reported
  33 tests plus 10 subtests passed and one environment-only failure: the image's
  `pyzmq` utility modules are zero-byte files, and the spawned config-test process
  could not inherit the in-process test stub.
- **DONE:** Amended the single Phase 1a commit as `c33e26adad`.
- **DONE:** The broad A/B matrix and focused 42-test suite passed on the retained
  H200 pod, then the same suite and matrix passed on the user-provided 4xGB300
  pod `yanbin-jiang-gb300-4gpu`.
- **DONE:** Representative mixed/base rows, multiple active adapters, rank-128
  default-capacity cells, and full-runner serial/two-stream eager and CUDA-graph
  correctness now cover BF16 WS1 on H200 and GB300.
- **PLANNED:** Broader model/shape coverage, quantized providers, distributed D0,
  and server E0 remain before end-to-end graduation.

## Historical Phase-1a gaps at the pre-campaign snapshot

This section through the historical change log preserves the state and decisions at
the named snapshot. Words such as “current,” “planned,” and “next” below are
historical and must not override the 2026-07-22 execution checkpoint above.

These are development obligations, not reasons to preserve broad guard code in the
Phase 1a commit.

### 1. Base provider and execution contract

- Current concrete provider is BF16 only.
- Current weight contract assumes standard `w13 = [E_local, 2I, H]` and
  `w2 = [E_local, H, I]` layouts.
- Provider plans, persistent workspaces, and per-forward metadata are not yet cleanly
  separated.
- The current masked DeepGEMM workspace can be large for prefill; device/model-aware
  planning and stable graph workspaces are not implemented.
- Base-weight rebinding, model-weight epochs, pointer replacement, and graph recapture
  are not implemented.
- The new runner now owns an environment-free local virtual-expert core, routing
  behavior, expand kernels, and side-stream helper. Duplication with the legacy
  experimental fork is temporary; deduplicate later through a neutral compatibility
  layer only after the explicit policy inputs stabilize.
- The legacy shared-add overlap experiment is intentionally not copied into the new
  engine. Reintroduce overlap only as an explicit producer/consumer execution policy.
- The base-only eager fallback and active/capture pipeline are separate concrete
  strategies. The promoted eager no-LoRA path invokes the resident base quant method
  while skipping virtual routing and LoRA tails; `_sgl_lora_triton_qi` is not the
  no-adapter provider contract. Separate captured base/adapter graph families are
  covered by the final campaign, while richer provider-selection policy remains
  future architecture work.

### 2. Model semantics and layouts

The current BF16 contract is already explicit and valid: standard gate-first,
non-interleaved W13; ordinary gated SiLU/SwiGLU; canonical pair-domain LoRA delta
`[T,K,2I]=[GATE|UP]`; BF16 W2 input; and a pair-domain BF16 activation bridge for
down-A. The gap is not “no contract.” The gap is that this physical form must not
become the contract for every future provider.

The target compiler therefore separates:

1. logical activation math and parameters;
2. logical projection slices, target masks, ranks, scales, and sharing;
3. provider-private physical W13 layout and exact output views: W2 dtype/layout,
   quant scales/swizzles, down-A pre-quant view, invalid rows, destination dtype, and
   buffer ownership.

Do not add parallel Phase-1a metadata classes just to restate existing BF16 fields.
Introduce each distinction with its first alternative provider or fused consumer and
adapt the current BF16 plan into it.

- Plain gated SiLU/SwiGLU is the implemented activation path.
- Non-gated experts, activation variants, clamp/alpha/`swiglu_limit`, and MoE biases
  need explicit semantic support.
- Up-first, interleaved, blocked, and quantized gate/up representations need
  provider-owned layout plans rather than model-specific branches spread through
  kernels; gate-first contiguous remains the valid current BF16 plan.
- S3 must pair corresponding gate and up columns in one **logical** tile, loading
  from their provider-native start offsets. Do not physically interleave or relayout
  tensors merely for S3. S1 only permutes `[T,H]` and has no gate/up dimension; a
  packed provider may choose a native blocked/interleaved W13 layout only when its
  load-time packing and benchmarks justify it.
- Routed scaling, `apply_router_weight_on_input`, no-combine variants, shared experts,
  and model-specific finalization semantics need end-to-end definitions.
- Gate, up, and down must become typed logical slices. A missing optional slice may
  occur at any position; support must not depend on tail-only zero filling.
- QKV, merged-column, replicated, and other stacked dense projections need the same
  slice model later, without forcing them through the MoE kernel ABI.
- Per-layer and per-expert hidden/intermediate dimensions must come from compiled live
  sites, not fragile global name tables or one model-wide hidden-dimension lookup.

### 3. Adapter formats and binding

- PEFT and other framework key dialects still need a centralized parser.
- Exact qualified weight-to-site binding and useful unmatched-key diagnostics are not
  implemented in the new execution plane.
- Gate/up/down A/B completeness, optional slices, independently ranked/scaled slices,
  shared-outer factors, and partial fused projections need a typed schema policy.
- Dense suffixes such as `gate_up_proj` and `down_proj` are ambiguous with routed MoE
  sites; actual rank-local module binding must resolve the distinction.
- Startup and dynamic load must eventually share the same compile, validation, pack,
  publish, and rollback flow.
- Model-specific exceptions should use sparse `ModelLoRAExtension` overrides with
  centralized fallback, so old models do not require a mass migration.

### 4. Rank and expansion strategies

- Production dispatch currently selects the direct expand/add provider strategy only
  through rank 64, but this is a historical policy rather than a kernel or hardware
  limit. The H200 whole-rank probe compiled and passed correctness through rank 1024.
- The current generic rank-greater-than-64 fallback is now slice-correct: gated
  gate/up launches one generic fused-MoE operation per equal output slice using
  zero-copy views, while down and non-gated sites remain one launch. Its remaining
  questions are launch-count/performance and the direct-versus-generic policy.
  Production rank-128 gate/up remains blocked separately by the production A
  schedule's resource use. Benchmark-only M0 can exercise rank 128 through indexed
  A, but this does not make the production rank-128 path supported.
- For rank at most 64, production keeps the flat midpoint schedule when each half is
  divisible by a tensor-core-sized tile. Otherwise it uses the two-slice grid with an
  independent masked tail per half and caps the default tile at 64.
- Implement whole-rank and looped-rank sliced expand as the core algorithms. Keep
  true split-rank/expand split-K only as an optional candidate for unusually large
  rank plus an underfilled output grid; it must not define large-rank correctness.
- Final support should select among direct sparse, segmented, aligned-grouped,
  whole-rank, looped-rank, optional split-rank, and later one-shot A+B strategies by
  shape/device, with rank limits kept provider-private.
- Mixed ranks and independently ranked gate/up/down slices need correctness and
  performance coverage.

### 5. Assignment, residency, and per-forward state

- Current execution consumes legacy `LoRAInfo`/batch metadata as a bridge.
- Batch-local adapter ordinals, immutable adapter identity, residency snapshots,
  slot generations, and provider device assignments are not implemented.
- Mutable singleton batch state must become a per-forward execution context.
- Slot leases and graph-lane leases must remain held until every provider stream is
  complete.
- Mapping refresh, tail clearing, eviction, slot recycling, and stale-generation
  handling need explicit tests.

### 6. CUDA graphs

- Separate no-LoRA and LoRA-active decode graph families are not implemented for the
  new engine.
- Capture/replay/recapture, graph transitions, stable metadata addresses, event
  lifetime, and workspace ownership need full coverage.
- Breakable CUDA graph is the preferred prefill design; LoRA metadata refresh and
  constrained prefill capture are not implemented yet.
- Piecewise CUDA graph support is not a target unless a later requirement justifies it.
- Large-prefill workspace planning must replace the removed fixed admission limit.

### 7. Distributed execution

- Phase 1a has only been designed around standard dispatch with local expert IDs.
- LoRA expert weights must be stored at local-expert size under EP.
- Global expert IDs must be translated to local IDs exactly once at a named boundary.
- Adapter ordinals must travel with tokens through EP/A2A dispatch and be reconstructed
  in the post-dispatch token domain.
- TP, ordinary EP greater than one, DP attention, advanced A2A backends, EPLB/expert
  maps, and elastic EP are not implemented in the new contracts.
- DP-attention ranks must agree on one execution-group adapter table before token
  redistribution.
- Speculative/MTP domains need explicit assignment and graph behavior; they are not
  part of the current vertical slice.

### 8. Quantized providers

- FP8 block-scale base MoE provider is planned.
- NVFP4 W4A4 base MoE provider is planned.
- Marlin W4A16 provider is planned.
- Quantized providers must preserve the BF16/FP16 values consumed by LoRA shrink at
  the correct stage while keeping packed/quantized layouts provider-private.
- Introduce `StageValue` or multiple physical views only with the first provider that
  genuinely requires them; do not complicate the BF16 slice preemptively.

### 9. Performance

- The isolated GB300 CUDA-graph testbed shows that the two-slice grid is a correctness
  generalization, not a universal speedup. At equal tile sizes its median change was
  +0.17% for `BN=16`, +0.59% for `BN=32`, and -2.26% for `BN=64`, with shape-dependent
  results on both sides of zero.
- A two-slice `BN=128` policy was consistently worse for ranks 32 and 64 (roughly
  11-18% slower) and was neutral-to-worse in the longer rank-16 rerun. Therefore the
  Phase 1a policy keeps the legal flat schedule and uses the two-slice schedule only
  as the otherwise-untileable fallback; no per-rank dispatch table is added yet.
- A follow-up H200 testbed covered the previously untested `H=N/2=48` case. The
  current midpoint schedule is legal only at `BN=16`; the best existing two-slice
  schedule was a median 4.36% faster across ranks 16/32/64 and token batches 1/16/32
  (range -0.02% to +6.11%). This demonstrates that divisibility is a correctness
  condition, not a sufficient performance policy. The current Phase 1a production
  dispatch remains unchanged while the broader device/shape planner is designed.
- The H200 benchmark-only generalization uses one Triton source compiled into
  `ALIGNED_FLAT`, `UNIFORM_RAGGED`, and `GENERAL_RAGGED` schedules. The arithmetic
  equal-slice mode beat the existing two-slice implementation by a median 5.68% for
  H=48 in the longer run. In contrast, forcing all shapes through the runtime-prefix
  general mode added a median 8.19% over the equal-slice mode for H=48 and larger
  overhead on the single/equal-two cases. Keep one semantic operation and one source
  per routing domain, but retain compile-time fast schedules; do not use one
  always-runtime-generic binary.
- Additional H200 sweeps at half widths 80, 96, 112, and 160 found cases where a
  slice-aware `BN=32/64` schedule was roughly 8-18% faster than a flat schedule
  constrained to `BN=16`. Tile selection must ultimately depend on device, width,
  rank, and routed-row count, not only boundary divisibility.
- A current-source GB300 best-policy sweep at `72055b46fd` supersedes the earlier
  equal-BN-only Phase-1a conclusion. It covered half widths 48/80/96/112/160/176/
  192/224/256/384/512/768, total token counts 1/16/32/128/256, ranks 16/32/64/128,
  and each schedule's best legal `BN=16/32/64/128` (plus 256 in the later/larger
  sweeps). Uniform-sliced median gains were 7.39% at H=80, 15.07% at H=112,
  22.51% at H=176, and 9.56% at H=224. H=192/384/768 were effectively tied;
  aligned-flat led by about 1-1.4% on average at H=256/512. Long reruns confirmed
  +40.27% for H=112/T=256/R=128 and +57.41% for H=176/T=256/R=128, while flat
  retained a 3.89% lead for H=512/T=256/R=128. Therefore flat is not a universal
  gate/up schedule. Keep the existing constexpr `UNIFORM_SLICED` mode, but stop
  standalone-kernel policy work after a simple interim rule: prefer flat only when
  the half width admits at least BN=64; otherwise use uniform-sliced initially at
  BN=64. Defer finer `(device, H, rows, rank)` selection to end-to-end tuning.
  Raw artifacts and the summary are in
  `benchmark_results/gb300_gate_up_bn_sweep_20260721/`.
- Raw benchmark artifacts are stored under
  `benchmark_results/gb300_job1293/`. They cover one GB300, BF16, `H=192`, top-k 8,
  64 experts, hot-cache expand-only replay. The result is not Hopper or end-to-end
  evidence. Those saved files identify the clean base revision of the isolated
  remote checkout because the Phase 1a sources were copied into it before execution;
  the final benchmark now records the dirty-worktree bit as well as the revision.
- H200 artifacts and the exact benchmark-source snapshot are stored under
  `benchmark_results/h200_sliced_20260720/`. These are BF16, top-k 8, 64-expert,
  hot-cache expand-only CUDA-graph measurements; they are not end-to-end results and
  do not establish the Blackwell policy.
- Establish a neutral legacy comparison with unrelated experimental optimizations
  disabled on both sides.
- Re-home useful routing and overlap optimizations independently instead of depending
  on the legacy master switch.
- Reduce materialized bridge tensors and launch count at the semantic injection
  boundaries.
- The target gate/up overlap is base prepare/W13 concurrent with LoRA-A shrink only.
  After the join, fuse LoRA-B expansion directly into S3 activation and optional
  FP8/NVFP4 quantization, eliminating `[T,K,2I]` `gate_up_delta` write/read traffic.
  Keep the current materialized A+B overlap as a benchmark fallback until the fused
  path wins the relevant shape/device matrix; it can sometimes hide B behind W13.
- Do not recompute gate/up LoRA-A per S3 output tile: materialize the small logical
  `[T,K,2,R]` (`[T,K,2R]` physical) result once. A later S3 fusion may consume each
  activation tile into down-LoRA-A, with an explicit looped-I, split-I atomic, partial
  reduction, or Hopper/Blackwell cluster reduction policy.
- Measure eager and graph paths, all-base overhead, active/mixed adapters, memory,
  launch count, latency, and throughput on Hopper and Blackwell.
- Final goal remains no meaningful all-base tax and parity-or-better active-LoRA
  throughput within agreed benchmark noise.

## Archived general A/B kernel prototype — EVIDENCE ONLY

The seven commits after `c33e26adad` are no longer on the active `sgl-lora` branch.
They are preserved losslessly on `sgl-lora-pre-redesign-backup-20260721` at
`72055b46fd`. Their correctness results, schedules, and raw measurements remain useful
benchmark evidence, but their universal routed A/B API mixed MoE routing, slice layout,
weighting, reduction, and destination behavior before the stage-specific provider
contracts were settled. Do not cherry-pick these commits into the active branch.
Rebuild useful candidates behind the new benchmark-only case/matrix harness, then select
production gate/up and down endpoints from `O0/M0` evidence.

The following list describes what the archived prototype established. The current plan
supersedes direct production integration: first implement the resolved benchmark
harness, compare stage-specific candidates, and only then land new BF16/FP16 production
primitives after the Phase-1a vertical slice.

1. **DONE for the indexed-group primitive:** Commits `a8a2e1f222`, `e86158e963`,
   and `72055b46fd` provide one semantic `sliced_lora_b_expand_add` source with
   aligned-flat, uniform-sliced, compiled-ragged, and descriptor-ragged slice
   schedules; independent A/compact-B/output offsets; arbitrary active slices
   without zero-filled factors; graph-stable per-virtual-expert layout selection;
   and bounded whole/looped-rank execution. The descriptor path supports different
   target subsets in one launch, `-1` no-target groups, routed sentinels, and
   in-place layout-map refresh across graph replay. The active integration adds
   destination-typed store, add-to-base, and routed pair-to-token reduction. Dense
   row integration and fused provider epilogues remain.
2. **DONE for `INDEXED_GROUP`; other row schedules remain:** Commit `32fe6f74d8`
   adds a packed-factor LoRA-A shrink over graph-stable indexed rows, token/pair input
   domains, bounded K tiles, looped-K, and explicit FP32 versus destination-dtype
   split-K accumulation. The active integration connects FP32 A workspaces safely to
   BF16 B factors, clears per-expert-A rows before broader shared-B consumption, and
   covers ranks 16/64/128. Factor deduplication across top-k and heterogeneous factor
   signatures remain planner work.
3. **IN PROGRESS:** Share GEMM tile math, slice descriptors, weight packing, and
   tuning infrastructure while retaining compile-time row schedules:
   `CONTIGUOUS_SEGMENT`, `INDEXED_GROUP`, and optionally `DIRECT_SPARSE`.
   Dense rows use adapter groups; MoE rows use `(adapter, local_expert)` virtual-expert
   groups. Do not force both through one runtime-branch-heavy binary.
4. **PARTLY DONE:** B now implements routed pair store, add-to-existing-destination,
   and routed-weighted pair-to-token reduction under one destination-dtype contract.
   Ordinary dense segment/scatter row planning and provider-owned activation/quant
   and collective fusion remain.
5. Replace the long-term materialized gate/up-delta boundary: overlap base W13 with
   gate/up LoRA-A only, then let the provider fuse LoRA-B + base add + activation +
   optional quantization. Represent gate/up as logical paired slices while preserving
   native physical layouts. Keep the current materialized schedule only as a measured
   fallback, not as the semantic ABI.
   The first end-to-end checkpoint deliberately retains the already working BF16
   materialized sequence: gate/up A+B store -> base W13 -> delta-aware activation ->
   base W2/finalize -> routed down A+B reduction. Do not block that checkpoint on the
   final fused schedule.
   Long-term public endpoints are separate: dense sliced expand/add; MoE gate/up
   B+base-add+activation(+optional quant); and MoE down routed-weighted pair-to-token
   reduce/add before or inside finalize. They may share B tile math and descriptors,
   but dense calls must not carry top-k metadata or down-reduction flags.
   Prefill starts serial. Decode should later compare serial, side-stream gate/up A+B,
   and side-stream A-only followed by fused B+S3. Carry explicit forward phase into
   the execution plan instead of using only `num_tokens <= 256` as a decode proxy.
6. Compare segmented/chunked, aligned-grouped, and direct row schedules using both
   precomputed-routing kernel timing and route-inclusive timing. Do not remove either
   segmented or grouped execution until end-to-end data shows one dominates.

The retained H200 is sufficient for API/correctness work, compiler-resource checks,
CUDA-graph replay, Hopper tuning, and the first row-schedule comparison. Cover dense
decode/prefill distributions; MoE top-k/expert distributions; single/equal/unequal
and arbitrary-active slices; common and non-power-of-two ranks; store/add/reduce
epilogues; and mixed/base rows. The same saved ragged matrix has now run on H200 and
GB300. Keep independent Hopper and Blackwell schedule/config selections; neither
device's winner is automatically a default for the other.

The first production B checkpoint passed H200 correctness for rank 8/16/64, one,
two, three, and ragged active slices, compact B with an untouched output hole,
routed weighted reduction, sentinel rows, fixed-capacity grouped routing, and CUDA
graph replay. Against the exact prior production expand, the final 24-case BF16
CUDA-graph sweep (top-k 8, 64 virtual experts, tokens 1/16/32, ranks 16/64) improved
every measured one/two-slice case by 1.23% to 9.93% (median 3.72%) on H200. This is a
Hopper result; the independent Blackwell evidence is recorded below.

The full H200 A-to-B tuning sweep covered gate/up and down, top-k 8, 64 virtual
experts, ranks 16/32/64/96/128, and token counts 1/16/32 under hot-routing CUDA-graph
timing. The best bounded BF16-workspace A schedule beat the old full chain in every
case where the old gate/up path compiled: gate/up by 7.33%-53.47% and down by
4.60%-38.71%. Gate/up ranks 96/128 also compiled and ran where the old A kernel hit
shared-memory resource errors. Winning configs depend materially on row count,
packed rank, and projection K: decode usually selects split-K 8, while larger
rank/batch shapes frequently select split-K 1. Preserve these measurements as Hopper
autotune evidence; do not turn one shape's winner into a universal cutoff or apply it
to Blackwell. Raw result and exact script:
`benchmark_results/h200_a_tune_20260720/lora_ab_hopper_a_tune_h200.json` and
`bench_lora_ab_hopper_a_tune_h200.py`.

The matching GB300 sweep covered the same 30 BF16 A-to-B shapes. All bounded
schedules passed validation with maximum absolute difference `0.03125`; all six
legacy gate/up rank-96/128 shapes remained out of resources while the bounded path
ran. Across the 24 legacy-comparable shapes, the best bounded schedule improved
gate/up by 5.97%-59.78% and down by 7.84%-46.63% (22.90% overall median). Hopper's
winner was within 1% of the GB300 winner in only 17/30 shapes and was as much as
8.45% slower, principally on gate/up, so Blackwell retains an independent policy.
Small winner differences are not table entries: the median winning p20-p80 span was
0.98%, and configs within that range are treated as ties.

The broad matrix measured BF16 A workspaces, so a second GB300 sweep timed the exact
production rank-greater-than-64 path: indexed A, FP32 split-K workspace, and the
direct B input cast. It found that the runner supplied lowercase `num_stages=3` but
the selector ignored it and silently used stage 1. Commit `ec8a4eccb9` now preserves
that existing stage setting. In the controlled before/after runs this improved
high-rank gate/up by 18.72%-62.38%; down decode was neutral and down batches improved
4.49%-11.67%. The focused GB300 suite still passes all 42 tests.

After the fix, exact production FP32 remains 0.19%-17.76% behind the best swept FP32
schedule (7.12% median) across gate/up/down, ranks 96/128, and tokens 1/16/32. A
tuned BF16 workspace is a further median 3.51% faster than tuned FP32, with a range
from FP32 winning by 1.70% to BF16 winning by 19.34%. This is evidence for an
explicit workspace-precision policy, not permission to change numerical behavior
implicitly. Preserve FP32 and the temporary rank bridge until the planner can bind
device/shape configs and workspace dtype together.

Raw GB300 results and exact scripts are under
`benchmark_results/gb300_a_tune_20260720/`: the broad matrix, pre-fix and post-fix
exact-production dtype sweeps, and both benchmark source snapshots are retained.

The final H200 ragged-schedule study is under
`benchmark_results/h200_ragged_20260720/`. It compares independently tuned
`BLOCK_SIZE_N` choices using BF16, top-k 8, CUDA-graph timing, ranks 16/32/64, and
decode-like routed row counts. Across 27 homogeneous irregular-layout points, the
runtime descriptor was a median 13.73% slower than a layout compiled into the
kernel (range 7.94%-23.92%), but separate per-slice launches were a median 47.09%
slower than the descriptor (range 15.22%-121.88%); the descriptor beat separate
launches in all 27 points. For genuinely mixed Q/QV/KV/QKV layouts, one descriptor
launch was a median 20.51% slower than a compiled full-QKV union with zero factors,
while four layout-partitioned launches were a median 181.91% slower for batches
that exercised multiple layouts. Larger routed blocks showed the same result:
descriptor latency was 16.64%-19.84% above the zero union, while partitioned-launch
latency was 137.80%-188.72% above the descriptor. Packing/zero-materialization and
routing repartition costs are intentionally outside these kernel-only timings.

Therefore selection remains hybrid rather than descriptor-only:

- use aligned-flat or uniform-sliced for canonical layouts;
- use compiled-ragged for stable homogeneous irregular hot layouts when its compile
  and cache cost is justified;
- use descriptor-ragged for mixed or runtime-changing per-group layouts;
- retain compiled full-union plus zero factors as an autotune candidate only when
  those factors already exist or their packing/materialization cost is included.

Descriptor construction and layout-map updates are setup work, not per-forward
work. The runtime cost is cache-resident metadata and dependent address generation
per CTA, not the few kilobytes of storage. The packed table uses a 16-byte-aligned
header and 16-byte per-tile records; this reduced the measured descriptor penalty
materially. Device/shape autotuning should choose the schedule and tile size.

The matching GB300 rerun is stored under
`benchmark_results/gb300_ragged_20260721/`. The exact transferred kernel and test
files match commit `72055b46fd` by SHA-256. The full A/B/runner suite passed 45/45,
including the five ragged cases, CUDA-graph replay, and in-place per-virtual-expert
layout-map mutation. Across the same 27 homogeneous irregular-layout points, the
runtime descriptor was a median 16.00% slower than compiled-exact layout
(range 3.19%-24.11%), while separate per-slice launches were a median 60.29% slower
than the descriptor (range 14.95%-137.51%); descriptor won all 27 points. In mixed
decode-like batches, descriptor latency was a median 20.74% above the zero union,
while partitioned-launch latency was a median 192.71% above descriptor when multiple
layouts were active. For the larger routed-block cases those medians were 21.74%
and 177.04%, respectively. This confirms the same hybrid ordering on Blackwell.

With each device tuned independently, GB300 reduced median descriptor latency by
14.73% for the homogeneous matrix, 5.16% for mixed decode, and 6.57% for the larger
routed-block cases versus H200. Descriptor `BLOCK_SIZE_N` winners matched H200 in
23/27, 11/12, and 4/4 points, respectively, but the misses and other schedule
differences still justify an independent Blackwell table. H200 used Torch
`2.11.0+cu129` while GB300 used `2.11.0+cu130`, so these absolute cross-device
percentages are operational comparisons, not a CUDA-minor-controlled attribution to
the GPU architecture alone.

The archived `72055` prototype runner used logical rank greater than 64 as a temporary
migration bridge for selecting indexed A plus an FP32 workspace. It was not a design
limit or the intended final selector. The Hopper and GB300 sweeps both show that tuned
bounded A also wins at rank 16/32/64. Replace the bridge with offline-generated,
device-specific policy keyed by row domain/count, projection K, packed rank, and
workspace precision. Do not hard-code each noisy per-shape winner or reuse one
device's table on the other.

## MoE-LoRA benchmark matrix — IMPLEMENTATION IN PROGRESS

The canonical stage, schedule, dimension, topology, provider, measurement-scope,
profiling, graph/PDL, and phased-priority tables are now in sections 8.1 through 8.8 of
`sgl_lora_refactor_plan.md`. They are intentionally not duplicated here.

Key decisions recorded there:

- `MoeLoraBenchCase` is one immutable, fully resolved run; `matrix.py` expands
  curated presets instead of storing lists or creating a blind Cartesian product;
- benchmark code lives in a neutral `benchmark/kernels/lora_moe/` package and is
  never imported by production kernels;
- comparison boundaries are named `K0` primitive, `O0` route-inclusive operator,
  `M0` local full MoE, `D0` distributed MoE, and `E0` server;
- `C0/C1` remain matched materialized controls. `C2` fused serial is the first target,
  `C3` A-only overlap plus fused consumer is second, `C5` remains a
  diagnostic/fallback, and `C4` down overlap is deferred until last;
- raw indexed/direct, true segmented SGMV, grouped GEMM, qualifying BMM, token-owned
  reduction, and fused ownership are algorithm choices. Triton is the first schedule
  laboratory; selected CuTe DSL implementations may compete for raw
  indexed/SGMV-style, segmented, grouped, or fused endpoints. CUDA/provider libraries
  remain valid competitors; cuTile is optional and capability-gated. Do not implement
  every algorithm in every technology;
- one-GPU TP/EP/MoE-DP cases are labeled local-shape proxies. None has been executed
  in the active evidence campaign yet. Actual dispatcher, ID mapping, collectives,
  imbalance, communication overlap, and graph safety require `D0` runs;
- run cheap early cross-model guardrails before hardening the Qwen fused ABI; perform
  broader model performance tuning only after `C2/C3` stabilize;
- router gate/top-k and provider W13-input permute/quant are explicit boundaries rather
  than hidden inside routing or GEMM timing;
- logical/allocated/physical rank, active/slot-capacity adapters, per-stage work domain,
  factor sharding, touched expert bytes, cache state and writeback ownership are
  separate resolved fields;
- initial model-scale performance uses rank 32/64/128. Rank-16 correctness remains
  early; model-scale rank 8/16 performance is required at BF16 graduation. `mlpb` is
  tested independently from
  active work using explicit `(active_nonbase,base_present,capacity)` cells. The
  compact core is `(0,1,8)`, `(1,0,1)`, `(1,0,8)`, `(1,1,8)`, `(3,0,8)`,
  `(4,1,5)`, `(7,1,8)`, and `(8,0,8)`. Capacity 8 is the production-default
  anchor, capacity 5 checks a full non-power-of-two mixed batch, and capacities
  `16/32` are targeted stress or routing-only coverage when factor memory is large;
- CUDA graph and PDL are orthogonal benchmark modes; isolated PDL timing is not accepted
  as chain-level evidence;
- Nsight Systems timeline review is an advancement gate for structural changes, while
  Nsight Compute/nsight-python is used for shortlisted kernel counter sweeps;
- a kernel winner advances only if it remains competitive at the next wider scope.
- shared-outer LoRA is early kernel work: compare shared gate/up-A deduplication and
  shared down-B weighted rank reduction with repeated-pair controls. Model shared
  experts are later provider/model integration;
- matched experimental-TRTLLM and stock/legacy whole-M0 controls have not yet run in
  the active campaign. Internal SGL wins are not cross-backend superiority evidence;
- `E0` server graduation uses `sglang.benchmark.one_batch_server` with controlled
  fixed batches and real adapters; `bench_serving` is not used for this refactor's
  performance comparison.

Implementation status:

1. **DONE (`6fb1c25266`):** added small immutable case records, the six model
   presets, exact `P0` cells, and dependency-free CPU tests. Future provider and
   schedule constraints remain in this worklog instead of becoming premature
   launch assertions.
2. **DONE (`6fb1c25266`, `2023fe1b52`):** added reusable unprofiled CUDA-event,
   CUDA-graph, NVTX/cudaProfiler helpers and a production-launcher-based BF16
   `K0/O0` A/B driver. It supports direct/generic B, staged/combined correctness,
   eager/graph execution and precise source/environment JSON.
3. **DONE (`db60a724a8`):** added a separate matched-base `N0`, serial `C0` and
   true two-stream `C1` local full-MoE driver. World-size-one NCCL/model-parallel
   lifecycle, input reset, eager execution, CUDA-graph capture/replay and C0/C1
   correctness pass on H200 and GB300. BF16 gated SwiGLU is the current M0 scope;
   non-gated, quantized and distributed M0 remain explicit gaps.
4. **IN PROGRESS:** run curated `P0` cells independently on H200 and GB300. Hot/cold
   K0, isolated allocation-inclusive O0 and matched M0 evidence now cover the Qwen
   sparse `T=32/R=64` cell, the odd capacity-five mixed/base cell, and both default
   capacity-eight rank-128 full-occupancy cells, tiny decode, and the first T=2048
   prefill anchor. Threshold-neighbor and forced-prefill overlap cells are complete;
   other model geometries, producer-chain cache states and distributed shapes remain.
5. **STARTED:** captured Nsight Systems route and graph traces and Nsight Compute A
   counters on both devices. `nsight-python` remains optional automation; direct
   `ncu` is sufficient for the current shortlist.
6. **DONE FOR FIRST A SHORTLIST (`8116e12b5f` through `17ccc1b0a2`):** measured
   grouped tiled/split-K and raw-route indexed A under hot/cold K0, fair A-only O0 and
   matched M0. In an operator-isolated A-side O0 measurement, indexed removes
   virtualize/sort/align and cuts latency by 55.5%-71.9%; this saving is not
   pipeline-realizable while a downstream B path still consumes the shared plan,
   and improves serial C0 about 10 us on both GPUs. It is tied/modest in C1 because
   gate A+B is already hidden and down is B-dominated. Keep the candidate
   benchmark-only while the next shortlist targets fused gate/up B + activation
   (+ optional quant/down-A) and down-B + finalize.
7. **DONE FOR THE FIRST PREFILL ANCHOR (`88b7dcc0e2` through `a513e8dd59`):** production dispatch still
   applies the unchanged 256-token auto policy, while the runner now obeys an
   explicit resolved topology. M0 can force C1 without an environment flag or runner
   copy. T=128/256/257/2048 eager/graph timing and an H200 T=2048 structural trace
   are recorded below; real server prefill graphs remain.
8. **DONE (`af57edadfd`, `d74173b7d2`):** added a benchmark-only LoRA-B
   configuration laboratory. `logical-t` leaves production resolution untouched,
   `flat-tk` resolves at `T*K`, and `explicit` controls BM/BN/BK/group/warps/stages.
   K0 and route-inclusive O0-B record requested, resolved and effective configs,
   including fields normalized or ignored by direct B. Standalone runs now initialize
   the same minimal server context as the official fused-MoE benchmarks; the first
   fallback-only artifacts are retained but are not selector evidence.
9. **DONE (`20da51d0b7`):** fixed gated generic B so each output slice consumes its
   matching LoRA-A rank slice through zero-copy strided views. The old one-launch
   generic gate measurements were semantically invalid and are discarded. Rank-128
   identity oracles pass on H200 and GB300, rank-16/64 direct coverage remains green,
   and corrected generic gate CUDA-graph checks use direct B as an independent oracle.
10. **DONE FOR THE FIRST B M0 (`4ffaee0b41`):** added benchmark-only,
    independently configurable gate/up and down B families and launch configs,
    composed beneath indexed A without changing production dispatch. Checked
    rank-128 `T=32` full-active and mixed/base CUDA-graph M0 brackets cover
    direct/direct on H200 and direct-gate/generic-down on GB300. Matched mixed/base
    C1 node traces pass the launch-count, routing, stream-placement, and
    activation-join review.
11. **DONE FOR THE FIRST WIDER ROUTING-HOT ANCHOR:** BF16 WS1
    `T=256/R=128` shortlist K0/O0 and composite M0 timing plus matched C1
    structural traces are complete independently on H200 and GB300.
12. **PLANNED — neutral baseline:** repeat matched `N0/C0/C1` and add stock/legacy
    plus experimental-TRTLLM whole-M0 controls at identical semantic boundaries on
    H200 and GB300.
13. **PLANNED — BF16 Qwen target:** implement and compare the missing raw/segmented/
    grouped/BMM-when-qualified algorithm families and selected Triton/CuTe DSL/CUDA
    realizations; include shared-outer; build `C2` first and `C3` second. Initial
    model-scale ranks are 32/64/128.
14. **PLANNED — early guardrails:** run cheap large-H, latent-H, high-top-k, odd-I,
    non-gated ReLU2, partial-slice, shared-outer, and provider-padding correctness/
    local-shape cases before freezing the Qwen fused interface.
15. **PLANNED — BF16 graduation:** add model-scale rank 8/16 performance, a supported
    production rank-128 A path, wider adapters/routes, explicitly labeled one-GPU
    topology proxies, graph transitions, and single-rank
    `sglang.benchmark.one_batch_server` evidence.
16. **PLANNED — independent quant providers:** FP8, NVFP4 W4A4, and Marlin W4A16,
    each with provider-native activation/output contracts and separate graduation.
17. **PLANNED — real distributed/model shared execution:** D0 TP/EP/MoE-DP,
    dispatch/combine/collectives, standard then advanced A2A, and later model shared
    experts. Local proxy runs do not satisfy this item.
18. **PLANNED — full E0/multi-node:** lifecycle correctness, graph transitions,
    supported A2A, and two-node MNNVL. The broader adapter control-plane redesign
    remains Phase 3.

### First saved BF16 evidence (`2023fe1b52`)

- Both H200 SM90 and GB300 SM103 pass every synthetic A/B target, direct/generic
  correctness, staged/combined correctness, eager execution and CUDA-graph replay.
- Historical single-cell result: Qwen3.5-35B, `T=1`, `R=32`, capacity 1, direct B
  beats generic B on both devices. Policy conclusions are superseded by the broader
  Ninth evidence below.
  Gate/down B are `2.467/3.474 us` direct versus `2.707/3.565 us` generic on H200,
  and `3.218/4.378 us` versus `3.350/4.579 us` on GB300.
- Qwen3.5-35B, `T=32`, one active adapter, `R=64`: prebuilt A+B is effectively
  unchanged between capacity 1 and 8 on H200 (`49.261/36.229 us` gate/down versus
  `49.046/36.205 us`) and changes by only about 2% on GB300
  (`41.722/30.282 us` versus `42.546/30.458 us`). Capacity is therefore not an
  intrinsic cached A/B compute tax at this cell.
- Route-inclusive O0 does expose capacity cost. Capacity 1 -> 8 routing grows
  `144.832 -> 172.160 us` on H200 and `182.384 -> 206.624 us` on GB300. H200 full
  gate/down O0 grows 13–15%; GB300 down O0 grows 16%. The isolated GB300 gate O0
  result is noisy and is not used as a conclusion.
- The O0 route trace has eight eager kernels: virtual-ID, count/sort, align and
  sanitize are each launched once for A's row block and once for B's. Their combined
  GPU duration is only about 13 us; host/API launch gaps dominate the 145–207 us
  operator time. Shared graph-stable route planning and launch fusion outrank
  micro-tuning these tiny kernels.
- The cached `T=32` gate A+B graph has exactly two nodes. A shrink consumes 70–75%
  of traced GPU time: `38.944 us` A + `13.216 us` B on H200 and `31.424 us` A +
  `13.536 us` B on GB300.
- Nsight Compute reports the current gate-A shrink at `BM=16, BN=128, BK=256`, grid
  256, 167 registers/thread and 147.46 KiB dynamic shared memory/block. Both devices
  are limited to one block/SM, 6.25% occupancy; compute utilization is about 12%.
  The row plan expands 256 valid pairs over 211 hit experts to 3,376 A slots (13.2x)
  and 13,504 B slots (52.8x). First candidates are smaller N tiles and direct/indexed
  fragmented-row schedules, tuned separately on Hopper and Blackwell.
- Raw JSON, `.nsys-rep`, SQLite and `.ncu-rep` artifacts live under
  `/workspace/artifacts/sgl_lora_p0/` on the retained H200 hostPath and
  `/mirror/artifacts/sgl_lora_p0/` on the GB300 devbox.

### Second saved BF16 evidence (`8116e12b5f` through `db60a724a8`)

- The benchmark-only shrink schedule driver uses the production Triton source with
  explicit BM/BN/BK/split-K/warp/stage launch metadata. Its independent chunked FP32
  oracle keeps rank-128 correctness testable when the production schedule cannot
  launch; unsupported sweep entries are recorded rather than aborting the matrix.
- Qwen3.5-35B, `T=32`, one adapter, gate A, CUDA-graph batches of ten: current
  `BN128/BK256/SK1` is `35.824 us` H200 / `25.955 us` GB300. The best curated grouped
  schedule (`BN64/BK64/SK8`) is `31.083/21.454 us`; the tuned raw-route indexed
  schedule is `27.877/18.370 us`. Isolated one-replay confirmation is
  `39.488/30.496 us` current, `33.696/26.016 us` grouped, and `31.232/23.968 us`
  indexed. Graph batching and isolated replay are reported separately because PDL
  and back-to-back launch behavior differ.
- For the matching down A cell, current production is `6.819/6.088 us` H200/GB300,
  `BN32/BK256/SK1` grouped is `6.333/5.885 us`, and indexed is `3.987/3.629 us`.
  Gate and down therefore require separate schedule policy: split-K helps the long
  gate reduction under graphs, while down's shorter reduction favors no split.
- Gate rank 128 exposes a real current-path gap on both devices: production selects
  `BN=256` and requests 278,528 bytes of shared memory against a 232,448-byte limit.
  The bounded grouped path runs at `66.179 us` H200 / `45.002 us` GB300, and indexed
  runs at `63.042/42.950 us`. For rank-128 down A, bounded `BN32/SK1` improves current
  production from `15.706 -> 11.395 us` H200 and `13.258 -> 10.595 us` GB300.
- Nsight Compute explains the schedule change. Current gate A uses 167 registers per
  thread, 147.46 KiB dynamic shared memory and about 6.25% occupancy. The grouped
  `BK64/SK8` candidate uses 56 registers and 20.48 KiB at 50.95% achieved occupancy
  on H200, and 49 registers/20.48 KiB at 49.83% on GB300. The indexed winners use
  48 registers and 256 bytes at 46.17% on H200, and 32 registers/64 bytes at 70.78%
  on GB300. Profiler durations are diagnostic; unprofiled CUDA events remain
  authoritative.
- The full-MoE M0 driver passes matched N0, serial C0, two-stream C1, zero-LoRA base
  parity, C0/C1 parity and CUDA-graph replay on both GPUs. Qwen `T=32/R=64` graph
  p50 is `401.392/464.368/450.272 us` for N0/C0/C1 on H200 and
  `285.088/334.336/323.424 us` on GB300. C1 therefore reduces the current active-LoRA
  overhead from 62.976 to 48.880 us on H200 and from 49.248 to 38.336 us on GB300.
- Nsight Systems confirms the C1 structure. On H200 about 60 us of gate-LoRA kernels
  overlap base preprocessing/gate GEMM and the side path ends about 2.3 us before the
  activation join. On GB300 about 23 us of kernels overlap and the side path leaves a
  roughly 4.6 us tail before activation. Activation begins only after both paths on
  both traces. Saved reports are `h200_qwen_cap1_m0_c1_graph.nsys-rep` and
  `gb300_qwen_cap1_m0_c1_graph.nsys-rep` in the artifact roots above.
- These are still development diagnostics: M0 uses default (not offline-tuned)
  DeepGEMM configs, the indexed kernel is not wired into production, and BF16 WS1 is
  the only completed M0 provider/topology. D0 TP/EP/MoE-DP, quantized providers,
  mixed/base replay, producer-chain cache state and E0 via
  `sglang.benchmark.one_batch_server` remain required before production selection.

### Third saved BF16 evidence (`041a82a2c4`)

- Both shrink drivers now support explicit `hot` or forced-`cold` timing. Cold mode
  preallocates a same-stream eviction buffer of twice the CUDA-reported L2, touches it
  before each warmup/sample start event, excludes eviction from timing and graph
  capture, and requires one logical invocation per sample. JSON records the method,
  detected L2 and flush bytes. H200 detected 60 MiB L2 and flushed 120 MiB; GB300
  detected 129.25 MiB and flushed 258.5 MiB.
- The resolved cell is Qwen3.5-35B, `T=32`, `K=8`, `H=2048`, `I=512`, `E=256`,
  one active adapter in capacity 8, and `R=64`. Every curated grouped and all 27
  indexed schedules were swept independently on each GPU under one-replay CUDA-graph
  cold timing; winners were then rerun with 200 fresh samples.
- Gate A confirmation p50 on H200 is `46.000 us` production,
  `40.384 us` grouped `BN64/BK64/SK8`, and `38.720 us` indexed
  `BN32/BK128/W4`. Indexed is 15.8% faster than production and 4.1% faster than the
  grouped winner. On GB300 the values are `40.384`, `32.224`, and `32.064 us`
  (`BN32/BK128/W8` indexed): both candidates are about 20% faster than production,
  while indexed and grouped differ by only 0.5%.
- Down A confirmation p50 on H200 is `12.912 us` production,
  `12.640 us` grouped `BN32/BK256/SK1`, and `10.368 us` indexed
  `BN16/BK128/W8`. GB300 is `14.016`, `13.408`, and `11.072 us` with indexed
  `BN8/BK128/W8`. Indexed improves production by 19.7% on H200 and 21.0% on GB300.
- Cold versus hot materially changes both absolute latency and some winners. In
  particular, the large hot-cache indexed advantage for GB300 gate collapses to a
  tie once factor data is evicted. Cache state and device must therefore be schedule
  keys; no single K0 result is sufficient for production dispatch. Next evidence is
  benchmark-only M0 substitution plus producer-chain and route-inclusive timing.

### Fourth saved BF16 evidence (`c08d0db80e`)

- O0 now measures one isolated allocation-inclusive A operator from an idle GPU with
  synchronized wall time. Cache eviction/reset completes before the timed boundary;
  route allocation, Python/CUDA API work, route kernels, the selected A shrink and
  device completion are included. Grouped O0 rebuilds only the A route and reuses its
  preallocated output; it does not accidentally build or prewarm the B route. Indexed
  O0 remains its single raw-route kernel. K0 CUDA-event timing is unchanged.
- Every grouped and indexed schedule was swept for the same Qwen sparse
  `T=32/K=8/R=64` gate/down cell under forced-cold inputs, followed by 200-sample
  confirmation runs. Gate A production/grouped/indexed p50 is
  `143.111/133.232/63.645 us` on H200 and `170.145/171.825/63.072 us` on GB300.
  Indexed reduces the operator-isolated A-side O0 baseline by 55.5% on H200 and
  62.9% on GB300.
- Down A production/grouped/indexed confirmation is
  `105.512/107.889/34.621 us` on H200 and `154.512/141.713/43.376 us` on GB300.
  Indexed reduces the operator-isolated A-side O0 baseline by 67.2% and 71.9%,
  respectively. These savings are not pipeline-realizable if B still requires the
  aligned plan. GB300 isolated
  wall timing has visible host jitter, but the indexed margin is much larger than the
  run-to-run movement and repeats in the full configuration sweep.
- The conclusion differs from K0: grouped schedule tuning changes only a few percent
  once allocation/routing dominates, while indexed address resolution removes the
  virtualize/sort/align/sanitize plan entirely. The indexed design now has sufficient
  K0, cold-L2 and O0 evidence to enter benchmark-only M0; production dispatch remains
  unchanged until chain correctness, timing and trace review pass.

### Fifth saved BF16 evidence (`3435e909b9`)

- The M0 driver can now substitute benchmark-only indexed A at both gate and down
  sites while retaining production B, SwiGLU activation, DeepGEMM base stages,
  finalize, stream/event handling and graph capture. Production remains the default
  and no production file or dispatch policy changed. The indexed path passes
  production-C0 parity, base-only parity, C0/C1 parity and eager-versus-graph checks
  on both H200 and GB300.
- Exact sparse-case CUDA-graph p50 on H200 is `401.296/465.392/452.256 us` for
  production N0/C0/C1 and `399.264/454.944/450.736 us` with indexed A. C0 improves
  10.448 us (2.2% total; about 13% of measured LoRA overhead), while C1 is tied within
  base-run variation. GB300 production is `284.272/332.544/321.344 us`; indexed is
  `283.392/322.304/316.768 us`. C0 improves 10.240 us (3.1% total; about 19% of LoRA
  overhead) and C1 improves 4.576 us (1.4% total).
- H200 eager timing also favors indexed A: C0 `598.592 -> 553.408 us` and C1
  `707.312 -> 643.088 us`. GB300 eager totals move in the same direction, but the N0
  control shifted substantially between processes, so only the graph comparison is
  treated as advancement evidence there.
- Matched H200 Nsight Systems node traces explain the C1 result. Production gate A
  shrink plus B expand completes about 3.6 us before activation because its B kernel
  is delayed/resource-overlapped behind the 242 us base gate GEMM. Indexed gate A+B
  finishes about 239.5 us before activation. The serial down tails are effectively
  identical: about 30.7 us production versus 30.0 us indexed. Thus faster A is useful
  for serial/eager/O0 paths, but it cannot materially improve this C1 cell while gate
  A+B is already hidden and the down tail is B-dominated.
- GB300 timing and correctness completed, but two Nsight Systems graph-node captures
  hung before the profiler range and produced no report. Only those profiler process
  trees were terminated; the devbox recovered cleanly. The next C1 work is fused
  gate-B + activation (+ optional quant/down-A) and down-B + finalize, not further A
  micro-tuning. Remove the conservative unused A-route prewarm later as cleanup and
  for shapes where the base gate GEMM is too short to hide the side chain.
- The non-power-of-two mixed/base cell (`T=32`, four active adapters plus one base
  identity, capacity five, rank 64) also passes production-C0 parity, base-only
  parity, C0/C1 parity and graph replay on both devices. H200 production
  N0/C0/C1 is `400.800/463.424/451.024 us`; indexed A is
  `401.008/453.504/450.496 us`, improving C0 by 9.920 us (2.1%) while C1 is tied.
  GB300 production is `284.928/332.544/321.280 us`; indexed is
  `285.744/323.408/317.216 us`, improving C0 by 9.136 us (2.7%) and C1 by
  4.064 us (1.3%). This closes the first mixed/base and odd-capacity semantic cell;
  tiny decode and prefill remain separate coverage items.

### Sixth saved BF16 evidence (`17ccc1b0a2`)

- The M0 harness now treats only Triton's expected `OutOfResources` from the
  production C0 oracle as a structured unsupported reference; every other exception
  still stops the run. The indexed-A wrapper is fail-closed: external calls may use
  only production `routing` or indexed `all`. This is benchmark-only handling and
  does not relax a production launch check.
- Both default-capacity rank-128 cells were independently swept on H200 and GB300:
  eight active adapters, and seven active adapters plus base-only rows. Every grouped
  and indexed cold-K0 schedule was measured, all candidates within about 1% of the
  winner were confirmed with 200 fresh samples, and checked M0 used explicit
  per-device winners rather than the rank-64 `auto` shortlist.
- Production gate A cannot launch at rank 128 on either device: its `BN=256` schedule
  requests 278,528 bytes of shared memory against a 232,448-byte block limit.
  Production down A remains supported. Confirmed H200 full-active gate A is
  `76.864 us` grouped versus `74.816 us` indexed; down A is
  `25.904/19.904/16.608 us` production/grouped/indexed. The mixed/base values are
  `76.864/74.720 us` gate grouped/indexed and
  `25.824/19.968/16.640 us` down production/grouped/indexed.
- On GB300, confirmed full-active gate A is `60.128 us` grouped versus `61.152 us`
  indexed; down A is `24.000/18.144/15.200 us`. The mixed/base values are
  `60.208/61.968 us` gate grouped/indexed and
  `24.048/18.128/15.104 us` down production/grouped/indexed. The small grouped gate
  K0 advantage does not include route-plan allocation; indexed still removes that
  work, so K0 alone is not a production selector.
- Checked rank-128 CUDA-graph M0 N0/C0/C1 is
  `401.360/533.872/519.408 us` H200 full-active and
  `401.328/518.720/504.080 us` H200 mixed/base. GB300 is
  `283.360/384.800/371.456 us` full-active and
  `283.424/374.240/359.488 us` mixed/base. Two-stream saves 13.3-14.752 us over serial;
  all finite-output, zero-LoRA, C0/C1 and eager/graph checks pass. These are indexed
  absolute latencies and overheads versus matched N0, not production-active
  speedups, because the production rank-128 gate path is unsupported.
- The results reinforce a device/shape policy rather than one universal schedule.
  Next core cells are tiny decode and prefill; the next execution optimization remains
  fused gate-B + activation (+ optional quant/down-A) and down-B + finalize.

### Seventh saved BF16 evidence (`17ccc1b0a2`, additional GPU artifacts)

- Tiny `T=1/R=32/capacity=1` exposes a wider-scope reversal. On H200, confirmed
  gate A is `9.568 us` production, `9.216 us` tuned grouped and `13.952 us` indexed;
  down A is `8.224/7.712/5.872 us` production/grouped/indexed. Nevertheless checked
  M0 production N0/C0/C1 is `95.136/97.616/94.080 us`, while indexed is
  `94.912/98.128/90.240 us`: indexed loses 0.512 us in serial but wins 3.840 us in
  two-stream. GB300 gate is `11.104 us` production versus `17.472 us` indexed and
  down is `9.952/9.056/7.584 us`; production M0 is
  `84.784/88.640/83.440 us` and indexed is `85.120/87.008/82.992 us`.
- Matched H200 tiny C1 graph-node traces explain the result. Production and indexed
  gate A+B finish 8.736 and 8.960 us before activation, so the slower indexed gate
  is fully hidden. The critical post-reorder down span falls from 9.472 to 6.848 us,
  and total traced kernel span falls from 91.968 to 88.544 us. A site/provider choice
  therefore depends on overlap placement, not just the isolated gate winner.
- T=2048/R=64 prefill gives the opposite row-schedule result. H200 gate A is
  `143.776 us` production, `135.488 us` tuned grouped and `1,037.840 us` indexed;
  down is `33.664 us` production versus `136.512 us` indexed. Production M0
  N0/C0/C1 is `520.240/892.864/891.824 us`; indexed is
  `521.568/1,853.776/1,852.944 us`. GB300 gate is
  `124.384/109.280/639.744 us` production/grouped/indexed; down is
  `29.712 us` production versus `95.120 us` indexed. Production M0 is
  `367.328/670.720/672.496 us`; indexed is
  `369.408/1,234.192/1,233.632 us`. The tested Triton raw-route indexed A is a decode
  schedule, not a universal replacement; the current prefill path should retain
  grouped/tensor-core work. This does not reject a future CuTe/CUDA raw-indexed or
  hybrid tensor-core implementation.
- All tiny and prefill finite-output, production/indexed parity, zero-LoRA, C0/C1 and
  eager/graph checks pass. Prefill C1 is not two-stream evidence: both providers
  report `two_stream_overlap_effective=false` and execute the serial threshold
  fallback. M0 numbers are steady-state graph replay even though the matrix cell's
  K0 cache state is cold; do not compare those cache labels directly.
- The H200 production prefill trace is entirely serial. Gate routing+A+B occupies
  about the first 221.6 us; after the base stages, down A+B remains an approximately
  156.4 us tail. This strengthens the existing priority for fused gate-B consumers
  and down-B/finalize rather than more raw indexed-A tuning.
- The prefill B-config audit found no `T` versus `T*K` layout requirement. Generic B
  correctly uses kernel `top_k=1` for its flattened `[T*K,R]` row domain, while the
  inherited fused-MoE resolver looks up logical `T`. `af57edadfd` now exposes
  `logical-t`, `flat-tk` and explicit schedules, and the Ninth evidence shows that
  the lookup coordinate is a performance-policy choice rather than a correctness
  rule. Neither coordinate is universal across site, family and device.

### Eighth saved BF16 evidence (`88b7dcc0e2`)

- Two-stream policy and execution are now separate without changing the server
  default. `lora_layer` resolves the existing requested-and-`T<=256` production
  policy; `run_sgl_lora_moe(two_stream_enabled=...)` executes that fixed decision.
  The M0-only `--c1-overlap-policy=force` option records both the production-auto
  decision and the forced effective topology, and uses the same decision for eager
  correctness, graph capture, replay, timing and profile labels.
- Forced T=2048 production-A graph N0/C0/C1 on H200 is
  `519.680/892.672/886.736 us`, versus the auto-policy baseline
  `520.240/892.864/891.824 us`. Forced overlap saves only 5.936 us (0.67%) against
  its matched C0; eager is `529.728/911.744/896.096 us`, a 15.648 us (1.72%) gain.
  GB300 forced graph is `367.360/672.512/658.176 us`, saving 14.336 us (2.13%)
  against matched C0. GB300 eager is `381.776/759.776/839.136 us`, where force
  regresses 79.360 us (10.45%). All numerical and graph checks pass, and metadata
  confirms force changed an otherwise-serial production-auto decision.
- The H200 forced C1 node trace explains the limited graph benefit. Routing occupies
  the main stream through about 24.9 us; gate A runs 24.7-167.9 us and mostly
  saturates the GPU, while base-stream work first appears at 161.5 us. Gate B runs
  167.8-243.9 us concurrently with base preparation, but the base gate GEMM does not
  start until 243.5 us, essentially when gate B ends. The side chain finishes far
  before activation at 530.8 us, yet only its tail overlaps useful base work. The
  155.9 us down tail is unchanged from serial.
- This first anchor does not justify changing production's 256-token cutoff. Forced
  overlap is a small graph win, a small H200 eager win, and a material GB300 eager
  loss. Keep the default unchanged; next compare threshold neighbors and actual
  breakable/constrained prefill graphs, and prioritize fused gate-B consumers and
  the serial down-B/finalize tail over broader two-stream enablement.
- Matched threshold-neighbor cells confirm there is no performance discontinuity at
  256. H200 graph C1 versus C0 is -0.22% at T128, +0.37% at T256, a serial +0.06%
  repeat at T257 auto, and +0.43% at T257 forced. GB300 graph is -0.21%, +0.86%,
  -0.003% and +0.75%, respectively. All correctness and policy checks pass. The
  cutoff is therefore a compatibility heuristic, not a kernel constraint or tuned
  universal boundary.
- Sequential eager results are not accepted for fine policy selection yet. On GB300,
  the identical resolved-serial T257 auto C0/C1 paths differed by 4.64%, proving an
  order/thermal confound larger than the graph effects. Add interleaved or randomized
  eager timing before using sub-5% eager deltas; graph replay remains the stable
  evidence for this boundary matrix.

### Ninth saved BF16 evidence (`af57edadfd` through `20da51d0b7`)

The saved selector and explicit-grid artifacts are under
`/workspace/artifacts/sgl_lora_p0/bconfig_*` on H200 and
`/mirror/artifacts/sgl_lora_p0/bconfig_*` on GB300. All recorded candidates kept
correctness checks enabled. The first `af57` selector pass hit
`Global server args is not set yet!` and resolved both policies to the same local
fallback; those near-ties are diagnostic only. `d74173b7d2` initializes the minimal
standalone server context and makes missing context fail closed.

Direct B consumes BM/group/warps for these Qwen shapes, forces effective BN128, and
does not consume generic BK/stage fields. The K0 grid selected:

| Device | Decode gate | Decode down | Prefill gate | Prefill down |
|---|---:|---:|---:|---:|
| H200 | `BM16/G1`, 5.944 us | `BM16/G1`, 13.059 us | `BM32/G8`, 43.675 us | `BM32/G8`, 117.240 us |
| GB300 | `BM16/G1`, 5.267 us | `BM16/G1`, 11.174 us | `BM32/G8`, 32.890 us | `BM16/G8`, 95.614 us |

Decode has 2,048 valid pairs; BM16/32/64/128 route plans materialize
4,096/8,192/16,384/32,768 padded rows. Prefill has 16,384 valid pairs; BM16 and
BM32 both materialize 24,576 rows, while BM64/128 materialize 49,152/98,304.
Therefore decode pays immediately for BM32, while prefill gate can use BM32's better
tile efficiency at no extra padding. Prefill down remains site/device dependent.
Group size is secondary in this matrix.

The valid generic down-B core grid selected `BN64/BK32` on both devices, BM16/G1
for decode and BM32/G8 for prefill:

| Device | Decode winner | Prefill winner | Reduction versus inherited logical config |
|---|---:|---:|---:|
| H200 | 21.718 us | 147.467 us | 51.1% / 48.0% |
| GB300 | 17.702 us | 119.882 us | 51.5% / 50.3% |

This grid proves that BM-only tuning is insufficient for generic B: BN64 consistently
beats BN32 and BK32 generally beats BK64. The row tile still follows routed density:
BM16 for sparse decode, BM32 for prefill. Generic down is forced benchmark coverage
at ranks 32/64; production rank-greater-than-64 and broader occupancy cells still
need their own grid.

The audit also found a real gated generic-B correctness bug. A gated shrink stores
`[gate_A; up_A]`, but the old one-launch generic kernel read `gate_A` for both output
halves. `20da51d0b7` now launches the generic kernel over matching zero-copy A/B/C
slices. The affected production path was gated gate/up with rank greater than 64;
down and non-gated single-slice paths are unchanged. Three registered gate/up GPU
tests pass on each device, including an exact rank-128 `[zeros, ones]` identity
oracle. Corrected generic gate logical/flat K0 is 21.882/17.805 us decode and
132.638/69.547 us prefill on H200, and 18.173/25.338 us decode and
105.619/166.442 us prefill on GB300. A graph-node trace contains 20
`fused_moe_kernel` instances for ten captured B calls, confirming exactly two
slice launches per gated generic B operation.

Route-inclusive O0-B holds A fixed, rebuilds only the B route each sample, and uses
synchronized host wall timing through device completion. H200 explicit direct
prefill gate/down winners
beat the bracketed logical baseline by 4.6%/2.2%; generic down decode/prefill winners
beat it by 13.1%/32.4%. GB300 generic down wins cleanly by 11.2%/30.5%. GB300 direct
prefill gate is only directional because its two logical baselines drifted 9.4%, and
direct down is a marginal 1.0% O0 result. K0 winners therefore advanced as benchmark
candidates, not production policy. The first selected rank-128 candidates have now
passed matched M0 and structural trace review in the Tenth evidence below. Broader
ranks, token counts, routing occupancy and model geometries remain. A surviving
local WS1 candidate still requires E0 `sglang.benchmark.one_batch_server`; D0
remains independently required before distributed production selection.

### Tenth saved BF16 evidence (`4ffaee0b41`)

- M0 now has a benchmark-only per-site B substitution seam. Gate/up and down can
  independently select production, direct, or generic B and hold an explicit launch
  config only around that site's routing/B work. The override composes beneath
  indexed A, records requested and effective families/configs, and explicitly
  records `production_policy_changed=false`. Production dispatch and production
  defaults are unchanged.
- Correctness now compares the active LoRA delta after subtracting the matched
  base-only result, so the much larger base output cannot hide a missing or incorrect
  B contribution. The mixed rank-128 smoke reports maximum candidate/reference
  delta errors of `0.0001831` on H200 and `0.0002441` on GB300, against LoRA signals
  of `0.0055647` and `0.005493`. Maximum eager-versus-graph differences are
  `0.000305` and `0.000244`, respectively.
- The omitted H200 generic schedule space was screened before accepting the B
  comparison. The best added `BK=128` gate/down configs were `29.856/45.952 us`,
  both slower than the retained `BK=64` winners at `25.371/39.758 us`. A separate
  `BK=64`, two-warp screen was also slower at `30.448/45.152 us`. The first H200
  rank-128 candidate is therefore direct gate plus direct down; the GB300 candidate
  is direct gate plus its independently tuned generic down.
- Counterbalanced `generic -> candidate -> generic` 200-sample CUDA-graph brackets
  cover the Qwen `T=32/R=128` full-active and mixed/base cells. After normalizing
  each run by its matched N0 base control, H200 direct/direct reduces active-LoRA
  overhead by `5.728/2.064 us` (`5.0%/2.1%`) for mixed C0/C1; the full-active
  reductions are `5.808/3.496 us` (`4.5%/3.1%`).
  GB300 direct-gate/generic-down reduces mixed C0/C1 overhead by `2.74/2.27 us`
  (`3.5%/3.5%`) and full-active overhead by `1.95/2.23 us` (`2.2%/2.9%`). H200
  is a clear first-shape win; the GB300 hybrid win is smaller but repeats in both
  occupancy cells. Unprofiled bracket timing remains authoritative.
- Matched H200 node traces contain 14 candidate versus 15 generic-baseline kernels
  per replay. One direct gate B plus direct down B takes `47.280 + 36.451 us` versus
  two generic gate slices plus generic down at `51.418 + 37.817 us`, removing one
  launch and `5.504 us` of summed B device work. Profiled GPU span falls from
  `496.575` to `493.814 us`. The base gate GEMM still overlaps gate LoRA, activation
  joins the slower path, down remains serial, and each LoRA routing kernel appears
  exactly once per replay with no second rebuild before down.
- Matched GB300 software-CUDA traces likewise contain 14 candidate versus 15
  baseline kernels per replay. Direct gate B takes `21.203 us` versus two generic
  slices at `22.438 us`; generic down is unchanged at `27.280/27.296 us`. The
  candidate removes one launch, starts activation `0.710 us` earlier, and reduces
  profiled GPU span from `343.902` to `343.518 us`, while preserving routing counts,
  equal-priority stream placement, overlap, and the join. On Blackwell, plain
  `--trace=cuda` selected Nsight's hardware-event path and retained only graph
  construction records; accepted executable traces therefore use
  `--trace=cuda-sw,nvtx` and `--cuda-graph-trace=node:host-only`.
- Initial captures that ran correctness/setup inside the profiled process or enabled
  symbol resolution could stall before the range or during report construction.
  Correctness is run separately; the stable structural recipe uses `--skip-check`,
  disabled symbol/CPU sampling, ten graph replays, and a hard 120-second timeout.
  Accepted reports are under
  `/workspace/artifacts/sgl_lora_p0/rank128_m0_4ffa/h200/traces/` and
  `/mirror/artifacts/sgl_lora_p0/rank128_m0_4ffa/gb300/traces/`.
- These T=32 results select benchmark candidates only; the T=256 anchor is recorded
  below. Remaining gates include broader token/routing distributions and additional
  model-shape coverage, quantized and non-gated providers, distributed D0, and E0
  through `sglang.benchmark.one_batch_server`.

### Eleventh saved BF16 evidence (`4ffaee0b41`, T=256/R=128 artifacts)

- The routing-hot BF16 WS1 `T=256/K=8/R=128` matrix holds all eight adapters active
  while increasing the routed pair count from the completed T=32 anchor. It keeps
  benchmark-only indexed A fixed, screens per-site direct and generic B candidates
  at K0 and route-inclusive O0, then advances the resulting composite plans into
  matched M0 brackets and C1 node traces. Production dispatch and production
  defaults are unchanged.
- Within the screened shortlist, H200 favors direct gate `BM16/G1/W8` and direct
  down `BM16/G8/W8`. At K0 they
  beat the best bounded generic configs by `1.98%/4.46%`; O0 strengthens the result
  to `9.17%/5.17%`. In the generic -> direct/direct -> generic M0 bracket, matched
  N0-normalized C0/C1 overhead falls by `16.808/16.912 us`
  (`1.733%/1.765%`). All strict LoRA-delta and graph-replay checks pass.
- The H200 C1 traces contain 16 generic versus 15 direct kernels per replay. Direct
  collapses gate B from two launches to one and reduces its summed device work by
  `9.952 us`, but gate B remains exposed after the base gate GEMM and start-to-join
  improves only `2.944 us`. The decisive saving is the serial down tail: direct down
  reduces down-B time by `17.088 us` and the post-base-down critical tail by
  `17.456 us`. Median profiled GPU span falls from `1431.759` to `1413.231 us`.
  Routing is one identical four-kernel build per replay and stream/join placement is
  unchanged.
- Within the screened shortlist, GB300 M0 favors generic gate
  `BM16/BN128/BK32/G1/W4/S3` and generic down `BM16/BN128/BK32/G8/W4/S3`, but
  only after a scope reversal. K0 favors generic/generic; O0 favors
  direct-gate/generic-down because its eager allocation-and-launch boundary favors
  one gate launch; full M0 favors generic/generic. Carrying the O0 direct-gate
  choice into M0 increases matched N0-normalized C0/C1 overhead by
  `8.568/10.960 us` (`1.41%/1.86%`).
- The GB300 C1 traces explain the reversal. Generic gate B uses two kernels totaling
  `97.610 us`; the one-launch direct gate takes `110.330 us`. Gate B controls the
  activation join, so the direct path starts the downstream chain about `12 us`
  later even though it has one fewer launch. Mean profiled GPU span is
  `918.078 us` generic versus `931.393 us` direct. The four routing kernels, routing
  span, down B, down tail, stream placement, and join semantics are matched. O0
  includes allocation and host-launch overhead that is amortized or absent in graph
  replay; in M0, the direct gate's slower measured device work dominates.
- Production rank-128 gate A still cannot launch: its selected `BN=256` schedule
  requests 278,528 bytes of shared memory against the 232,448-byte per-block limit.
  These comparisons use benchmark-only indexed A and establish B-family selection
  only; they do not make production rank-128 execution supported. The bounded
  schedule neighborhood is sufficient for this checkpoint, while broader offline
  autotuning remains planner work.
- Raw K0/O0/M0/trace artifacts are under
  `/workspace/artifacts/sgl_lora_p0/rank128_t256_4ffa/h200/` and
  `/mirror/artifacts/sgl_lora_p0/rank128_t256_4ffa/gb300/`. Next gaps are
  producer-chain cache states, additional model geometries and routing distributions,
  quantized/non-gated providers, a supported production rank-128 A path, distributed
  D0, and E0 through `sglang.benchmark.one_batch_server`.

## External PR/commit optimization audit — COMPLETE

The author PR list, every commit in PR #21, and all 57 commits in the linked gate/up
comparison were inspected at their exact public revisions. The durable audit is:

`/Users/yanbin.jiang/Desktop/lora_refactor/lora_external_optimization_audit_2026-07-21.md`

The audit found no architectural reversal, but it added these previously missing or
under-specified dimensions to the canonical plan:

- `G0` router gate/top-k and `Q1` provider input permute/quant boundaries;
- `P_work`, `E_hit`, `R_phys`, active versus slot-capacity adapters, and per-site factor
  sharding/storage;
- direct/native/global/local/compact/single-CTA routing schedules and gate/down plan
  reuse;
- cold/hot/producer cache state, actual touched bytes, tails/addressing/program mapping,
  large `BN`/`GROUP_M`, writeback ownership and nonzero-base correctness;
- activation/quant vector/CTA/scale-layout coverage, metadata amplification, consumer-
  stream allocation lifetime, exact overlap windows and chain-level PDL;
- mandatory trace review for fusion, routing-domain, launch-count, stream/event,
  allocation, graph, PDL or collective changes.

Notable evidence: one PR preserved throughput while CUDA-graph output was corrupted;
broad PDL was E2E-neutral; warm-L2 microbenchmarks were 15–20% optimistic; and a
per-layer redundant rank mask produced roughly 1,980 unnecessary captured kernels.
These results are why aggregate latency alone is not an acceptance criterion.

## Work remaining after the general A/B primitives

Completing general sliced A and B finishes the ordinary **linear LoRA algebra**, not
the complete execution backend. The remaining work is explicitly:

- migrate dense column, row, replicated, QKV, merged, Mamba/GDN, ordinary MLA, and
  LM-head sites onto the shared operations and typed slice plans;
- preserve each site's semantic result dtype during dense migration. CUDA DSA
  `indexer.weights_proj`, in particular, intentionally computes BF16 x BF16 into
  FP32, while today's LoRA wrapper forms base+delta in BF16 and only then upcasts.
  Its migrated caller must create the FP32 base destination and run LoRA-B directly
  into it; legacy segmented/chunked/Torch B kernels do not yet honor this contract;
- retain a specialized embedding-A token lookup; embedding-B can use general B;
- define added-vocabulary embedding publication and replacement if that capability is
  brought into scope;
- implement absorbed-MLA `kv_b_lora` head-aware q-side and v-side corrections;
- finish dense adapter-row planning and MoE virtual-expert construction, routing,
  alignment, sanitization, EP localization, and direct-versus-grouped selection;
- implement provider epilogues and bridge removal: gate/up delta before activation,
  activation plus FP8/NVFP4 quantization, down-A consumption, and down-B insertion
  before finalize/combine/collective;
- place row-parallel/column-parallel collectives correctly and fuse LoRA contributions
  into an existing collective only where the algebra permits it;
- add ordered LM-head pruning/multi-pass assignment views; this reuses A/B math but
  has a distinct execution contract;
- finish no-LoRA/LoRA graph families, stable metadata/workspaces, breakable prefill,
  stream/event lifetime, and graph transitions;
- broaden BF16, FP8, NVFP4 W4A4, and Marlin W4A16 provider attachment beyond the
  bounded C0 seams. Active FP8 must retain dynamic activation semantics and the
  provider's scale ABI; native CuTe DSL W4A4 is still testbed-only; Marlin is
  attachable and its per-call workspace/dirty-destination repair passed the terminal
  `f2f406e056` H200 and GB300 smoke. Base quantization does not create new LoRA algebra while
  factors remain BF16/FP16, but it does require provider-specific physical views and
  fused stage boundaries;
- complete TP/EP/A2A/DP-attention/speculative assignment domains and the final
  correctness/performance graduation matrix.

Absorbed `kv_b_lora` and embedding-A are the remaining genuinely different LoRA
contractions. Routing, activation/quant, collectives, and graph work are execution
infrastructure or provider fusions and must not become additional semantic A/B APIs.

## Planned commit sequence

The architectural phase numbers group capabilities. The following is the reviewed
landing order for small, auditable commits:

| Order | Planned outcome |
|---|---|
| 1 | Preserve the clean Phase-1a BF16 virtual-expert vertical slice and current materialized controls. |
| 2 | Add the missing neutral stock/legacy and experimental-TRTLLM matched-M0 baseline. |
| 3 | Land BF16 Qwen `C2`: optional route views, selected A schedules, fused gate/up-B + activation + down-A, and down-B/finalize, including shared-outer. |
| 4 | Land BF16 Qwen `C3`: A-only overlap and graph topology; keep `C4` down overlap deferred. |
| 5 | Add early cross-model semantic guardrails before hardening the fused ABI. |
| 6 | Finish BF16 graduation: rank 8/16 performance, supported rank 128, graph families, topology proxies, and single-rank E0. |
| 7 | Add the minimal canonical assignment/residency/completion/typed-site/provider-plan objects required by the first quantized or distributed consumer. |
| 8 | Add FP8, NVFP4 W4A4, and Marlin W4A16 independently behind the same logical contracts. |
| 9 | Add real TP/EP/MoE-DP, standard then advanced A2A, collectives, and later model shared experts. |
| 10 | Finish full server/multi-node graduation, then Phase 2 dense/special execution and Phase 3 adapter control-plane refactoring. |

## NVFP4 W4A4 handoff — DEFERRED

Do not start this kernel implementation until Yanbin asks. Preserve these Phase 1c
priorities:

1. Add an explicit NVFP4 provider with LoRA injection immediately before
   activation/FP4 quantization and before final combine/collective.
2. Fuse gate-B expansion/add into activation plus FP4 quantization.
3. Fuse down-A shrink with activation production to remove the full BF16 bridge.
4. Fuse weighted rank reduction/shared down-B into finalize or the final collective.
5. Preserve packed top-k instead of launching a standalone repack.
6. Restore distinct no-LoRA and LoRA-active CUDA graphs.
7. Restore `wo_ud` direct all-reduce-buffer output and reduce qkvr/`wo_ud` launches.
8. Evaluate constrained breakable-prefill graphs and a per-device symmetric-memory
   policy.

Measured B200 W4A4 S1 throughput after symmetric-memory configuration was
101.06/1171.42/2097.76 tok/s at BS1/16/32, retaining 67.6%/77.4%/82.3% of the
standalone W4A4 base. S4 retained 98.4%/97.8% of S1 at BS16/32. W4A4 lost to
optimized Marlin W4A16 only at BS1, then led by 7.7% at BS16 and 20.5% at BS32.

Do not repeat the isolated shared-A/down-B factorization by itself. It passed GPU
correctness but was throughput-neutral (-0.7% to +1.1%) because it did not remove
the dominant bridge traffic, epilogue boundaries, or launches. The isolated source
was intentionally not retained; the durable result and conclusion are recorded here.

## Deferred validation and graduation matrix

Once the implementations and support matrix settle, add clear validation at stable
boundaries and run the full matrix:

- selector and automatic-fallback behavior;
- eager, capture, replay, recapture, and no-LoRA/LoRA graph transitions;
- base-only, one adapter, mixed base/adapter, S1-S4, multiple adapters, slot recycling,
  load/unload/eviction, and poisoned workspaces;
- activation, layout, logical-slice, rank, scale, bias, and nonuniform-dimension cases;
- H100/H200 and B200/GB200/GB300;
- BF16, FP8, NVFP4 W4A4, and Marlin W4A16;
- TP8, ordinary EP greater than one, supported A2A, and DP attention;
- correctness, memory, launch count, graph stability, throughput, and latency;
- no MTP/EAGLE in the first NVFP4 graduation milestone.

When experiments resume, use the available 8 H200 and 4 GB300 devices as independent
WS1 workers for deterministic case shards. Give every worker unique artifact/cache/
temporary/port namespaces; record GPU UUID, image/software, source hash, case hash,
and resolved config; keep candidate and controls on the same GPU; and rerun promoted
timings isolated and counterbalanced. Nsight profiling gets an exclusive device. D0
temporarily suspends local fan-out and owns all participating ranks. This policy
changes wall-clock throughput only and must never be reported as distributed evidence.

New Slurm allocations use `rxs` and require an explicit `rxs cancel` check at
completion; user-owned `rx` devboxes may be reused only when explicitly authorized.
GB300 job 1293 was cancelled after the earlier 2026-07-20 run. The H200 follow-up
reused the user's retained `sglang-29157-v0514-h200-exact` pod and neither acquired
nor released its reserved node. On 2026-07-21 the user explicitly authorized pod
replacement; the same pod was recreated on `gpu1-10-220-51-64` with only
`CAP_SYS_ADMIN` added so Nsight Compute could access hardware counters. The persistent
hostPath preserved the source and all artifacts; a fully privileged container was not
needed. For the later Blackwell matrices, the user explicitly provided the
already-running rx-managed pod `yanbin-jiang-gb300-4gpu`; the earlier ragged rerun used
GPU 0 for correctness and GPU 1 for serial performance measurements and performed
no acquire or release action. The rank-128 T=32 and T=256 M0 brackets and traces also
reused the same devbox without acquisition, release, or extension. The user
authorized extending it near its TTL if a long matrix needs it; use
`rx devbox extend yanbin-jiang-gb300-4gpu --by 8h`
only when needed. User-owned retained pods must not be released unless the user asks.

## Change log

Rows are grouped by campaign and review sequence rather than strict chronological
order.

| Date | Snapshot | Change | Evidence / remaining work |
|---|---|---|---|
| 2026-07-22 | `f2f406e056` shared/quant closure | Promoted the exact shared-outer selector; repaired TorchNative merged-segment bounds; composed bounded physical shared IDs and PDL; preserved resident no-LoRA provider dispatch; tightened active FP8 and Blackwell UE8M0 contracts. | Host shared policy: 9 passed. H200 and GB300 combined shared matrices: 56 passed plus 10 subtests on each device. Terminal post-format validation passed 128 GPU tests with 6 expected skips on H200, 129 with 5 expected skips on GB300, and 185 host tests plus 10 subtests on each source snapshot; real H200 TP8/EP2/MoE-DP2 and GB300 TP4/EP4 NCCL replays passed. Per-rank physical shared EP>1, advanced A2A, broad quant checkpoint/server attachment, and native CuTe DSL W4A4 serving remain unpromoted. |
| 2026-07-22 | official-main GB300 control | A broader, non-LoRA unmasked per-token FP8 JIT/AOT bit-exact diagnostic exposed 66 GB300 mismatches. One representative mismatch reproduced against the official-main AOT source, while all branch-added masked-layout and LoRA provider-plan cases passed. | Archived in `final_validation/gb300/optional_general_jit_aot_gb300.log` and `upstream_general_jit_control.log`. This is excluded from LoRA acceptance but retained as a later general quant-kernel investigation; H200's broader diagnostic passed. |
| 2026-07-22 | documentation-only design review | Separated algorithm families from implementation technologies; allowed CuTe DSL to compete for raw indexed/SGMV-style as well as segmented/grouped/fused work; made aligned routing an optional consumer view; documented the existing BF16 gate/up contract versus target logical/provider contracts; moved shared-outer early and model shared experts later; reordered delivery around neutral baselines, `C2`, `C3`, guardrails, BF16 graduation, quant, and D0/E0; added the 8-H200/4-GB300 parallel-worker policy. | No source, benchmark, GPU job, branch, or evidence changed. Matched TRTLLM/stock whole-M0, shared-outer M0, CuTe competitors, topology proxies, cross-model guardrails, quant providers, D0, and E0 remain planned. |
| 2026-07-21 | `4ffaee0b41` + H200/GB300 T=256/R=128 routing-hot artifacts | Completed bounded BF16 WS1 per-site K0/O0 screening, composite M0 brackets, and matched C1 traces at the wider routing anchor while holding benchmark-only indexed A fixed. | Benchmark-only H200 M0 favors direct/direct by `16.808/16.912 us` after matched-N0 normalization (`1.733%/1.765%`), with the serial direct-down tail providing most of the gain. The screened GB300 plan changes from generic/generic at K0 to direct-gate/generic-down at O0 and back to generic/generic at M0, where direct gate loses `8.568/10.960 us`; graph traces localize this to slower direct gate device work once host launch cost is amortized. Production is unchanged and production rank-128 gate A remains out of resources. |
| 2026-07-21 | `4ffaee0b41` + H200/GB300 rank-128 M0 brackets and traces | Added benchmark-only independent gate/down B family and config substitution beneath indexed A, plus a LoRA-delta correctness oracle; closed the omitted H200 generic schedule space and ran counterbalanced full/mixed graph brackets. | H200 direct/direct reduces N0-normalized LoRA overhead by about 2.1%-5.0%; GB300 direct-gate/generic-down reduces it by about 2.2%-3.5%. The mixed-C1 candidate/baseline trace pair on each device removes one gate-B launch without changing routing or overlap. Production dispatch is unchanged; cover wider shapes before E0 or any production selector. |
| 2026-07-21 | `20da51d0b7` + H200/GB300 gate-up oracles and corrected graph artifacts | Fixed generic gated B to launch matching zero-copy A/B/output slices and made rank-64 benchmark checks use direct B as an independent oracle. | The old generic gate timings were semantically invalid and are discarded. Rank-128 identity, rank-16/64 direct, eager and graph checks pass on both devices; the correct generic fallback is two launches. Benchmark whole-rank direct versus sliced generic above rank 64 before changing the caller policy. |
| 2026-07-21 | `af57edadfd`, `d74173b7d2` + H200/GB300 B-config K0/O0 artifacts | Added selector/explicit B configuration instrumentation, repaired standalone production resolution, swept direct BM/group and generic down BM/BN/BK/group, then advanced winners through counterbalanced O0-B. | Direct schedules are phase/site/device dependent; generic down converges on BN64/BK32 with roughly 48%-52% K0 reduction from the inherited logical config and clean 11%-32% O0 wins. Keep production dispatch unchanged until matched M0 and trace review. |
| 2026-07-21 | `a513e8dd59` + H200/GB300 T128/T256/T257 boundary artifacts | Added matched small-prefill and both sides of the production overlap threshold, then ran auto/forced eager and graph policies. | No 256-token performance cliff exists; graph effects are within -0.2% to +0.9%. Preserve compatibility default but replace the long-term scalar cutoff with measured phase/device/graph policy. Randomize/interleave eager comparisons before trusting small deltas. |
| 2026-07-21 | `88b7dcc0e2` + forced-prefill H200/GB300 artifacts and H200 node trace | Moved the unchanged production auto cutoff to dispatch, made the runner obey one explicit topology decision, and added a benchmark-only forced C1 policy with consistent graph metadata. | Forced T2048 is only 0.7%-2.1% faster in graph replay and is mixed in eager (H200 +1.7%, GB300 -10.5%). Keep production auto unchanged; test threshold neighbors and real prefill graph modes before policy changes. |
| 2026-07-21 | `17ccc1b0a2` + H200/GB300 tiny/prefill artifacts and H200 node traces | Completed cold-K0 and checked-M0 anchors at T1/R32 and T2048/R64, with structural traces for tiny C1 and serial prefill C0. | Indexed A is decode-oriented and can win C1 even when its gate K0 loses because gate is hidden and down is critical; it is 84%-108% slower end-to-end for prefill. Add a forced-overlap benchmark seam and retain grouped prefill scheduling. |
| 2026-07-21 | `17ccc1b0a2` + H200/GB300 rank-128 artifacts | Made expected production-oracle resource failure a structured benchmark result, made indexed substitution fail closed, and swept/confirmed the default full-active and mixed/base rank-128 cells through checked M0. | Production gate A is out of shared memory; indexed M0 remains correct and two-stream saves 13.3-14.752 us. Continue with tiny decode/prefill and fused B consumer endpoints; do not claim a production-active rank-128 speedup. |
| 2026-07-21 | `3435e909b9` + matched H200/GB300 M0 artifacts | Added a benchmark-only indexed-A substitution around the unchanged production B/activation/DeepGEMM pipeline, with production-C0 parity and graph checks; captured matched H200 production/indexed C1 node traces. | Indexed improves serial C0 by about 10 us on both devices, but C1 is tied/modest because gate A+B is already hidden and down is B-dominated. Advance fused gate-B/activation/down-A and down-B/finalize; keep production dispatch unchanged. |
| 2026-07-21 | `c08d0db80e` + H200/GB300 O0 artifacts | Added isolated synchronized wall timing and an allocation-inclusive A-only grouped O0 path; corrected indexed/local O0 to use the same timing domain. Swept and confirmed gate/down schedules on both GPUs. | Indexed raw-route A cuts operator-isolated A-side O0 by 55.5%-71.9%; the saving is not pipeline-realizable while B consumes the shared plan. Grouped tile tuning is secondary once route allocation dominates. Advance indexed A to benchmark-only M0 and inspect the resulting stream graph before production policy changes. |
| 2026-07-21 | `041a82a2c4` + H200/GB300 cold-cache artifacts | Added audited same-stream forced-L2 eviction to both A schedule drivers, swept all grouped/indexed configurations for the first Qwen sparse gate/down cell, and reran winners with 200 fresh samples. | Indexed remains best for H200 gate and both down cells; cold GB300 gate is tied with tuned grouped. Cache state/device must be explicit schedule keys. Compare benchmark-only substitution inside M0 and add producer-chain/O0 evidence before production wiring. |
| 2026-07-21 | `db60a724a8` + H200/GB300 M0 traces | Added and GPU-validated the matched DeepGEMM N0, serial C0 and two-stream C1 full-MoE driver in eager and CUDA-graph modes; captured the real Qwen C1 stream timelines. | C1 reduces current Qwen T32/R64 LoRA overhead on both devices and the join is correctly placed. Wire shortlisted A only after cold-L2/O0 evidence, then repeat M0; D0 and `one_batch_server` E0 remain. |
| 2026-07-21 | `8116e12b5f`, `f05d0d7fa9`, `01a3079819`, `d77f87bfff` | Added grouped schedule and raw-route indexed A benchmark drivers, independent rank-128 oracle, unsupported-schedule reporting, and H200/GB300 correctness/tuning/counter sweeps. | Indexed A wins current Qwen gate/down K0 cells and avoids A routing metadata; production rank-128 gate A is out of shared memory. Keep candidates benchmark-only until producer-chain, cold-L2 and M0 integration evidence. |
| 2026-07-21 | `2023fe1b52` + H200/GB300 artifacts | Started the resolved benchmark implementation and ran the first BF16 `K0/O0` cells plus Nsight Systems/Compute profiles. | Direct B wins the first real Qwen cell; capacity tax localizes to routing; route launch/API gaps and fragmented A shrink are the first measured bottlenecks. Add cold-L2 rotation, A schedule candidates and M0 before production policy changes. |
| 2026-07-21 | `6fb1c25266`, `2023fe1b52` | Added the benchmark-only case/matrix, timing/profiling helpers and local production-launcher A/B driver in two reviewable commits. | 22 dependency-free CPU tests and pre-commit pass. Both GPU architectures pass correctness, eager and graph smoke. No production kernel changed. |
| 2026-07-21 | server benchmark selection | Fixed `E0` graduation on `sglang.benchmark.one_batch_server`; excluded `bench_serving` from controlled refactor comparisons. | Keep unprofiled runs authoritative. Use the profiling flag only for PyTorch quick traces or `CUDA_PROFILER`-delimited Nsight Systems capture. |
| 2026-07-21 | branch restart | Preserved the complete post-Phase-1a A/B prototype at remote branch `sgl-lora-pre-redesign-backup-20260721` (`72055b46fd`) and reset/force-with-lease updated active `sgl-lora` to `c33e26adad`. | Keep the accepted Phase-1a selector/package/provider scaffold. Do not cherry-pick the seven archived commits; next active commit is the benchmark-only `benchmark/kernels/lora_moe/` harness aligned with the stage-specific design. |
| 2026-07-21 | adapter-capacity matrix correction | Clarified that `max_loras_per_batch` includes base-only as a real slot identity; split `L_active`, `B_base`, `L_groups`, `L_capacity`, and `L_resident`. Replaced the example-driven 4/5 matrix with a compact default-capacity-8 core, non-power-of-two coverage, fixed-active capacity isolation through 32, and slot-placement/reassignment replay. | Implement tuple validation `L_active+B_base<=L_capacity`; large capacity 16/32 cells may stop at routing/operator scope when full per-expert factor allocation is not representative or feasible. |
| 2026-07-21 | benchmark-matrix clarification | Promoted rank 128 to core performance coverage and first separated actual active adapters from configured `max_loras_per_batch`; proposed paired `mlpb=4/5` cells. | Superseded by the adapter-capacity correction above after confirming that base-only consumes a batch/pool identity and the production default capacity is 8. |
| 2026-07-21 | external optimization audit | Audited all four zcnrex PRs, all 73 PR #21 commits, and all 57 commits in the linked gate/up branch; added `G0/Q1`, work-domain/cache/sharding/writeback/graph-lifetime/PDL dimensions and a mandatory structural trace gate. | Durable report: `lora_external_optimization_audit_2026-07-21.md`. The findings refine the benchmark harness before implementation; no external code was transplanted. |
| 2026-07-21 | planning checkpoint | Defined the complete resolved MoE-LoRA benchmark matrix across routing, A/B stages, activation/quant, base providers, finalize/collectives, execution overlap, model dimensions, TP/EP/MoE-DP, devices, graph modes, and measurement scopes. | Await Yanbin's review; then implement benchmark-only `cases.py`, `matrix.py`, and the BF16 `K0/O0/M0` harness before the next production fusion. |
| 2026-07-17 | `65e69838d9` | Rebased Phase 1a on OSS main and removed premature broad validation from the review surface. | 9 CPU tests + 10 subtests and full pre-commit passed; GPU rerun pending an `rxs` node. |
| 2026-07-20 | `c33e26adad` | Isolated the new engine from `trtllm_lora_temp`, added explicit one/two-slice expand semantics, and benchmarked the masked two-slice schedule. | GB300 correctness passed; graph replay showed the grid mapping is shape-dependent, so midpoint remains the legal fast path and two-slice is the `BN<=64` fallback. Hopper, EP>1, and rank>64 remain open. |
| 2026-07-20 | `c33e26adad` + benchmark-only worktree | Prototyped one source-level sliced expand with aligned-flat, uniform-ragged, and general-ragged compiled schedules. | H200 correctness passed for 1/2/3 equal slices, unequal QKV-like slices, and H=48. H=48 favors slice-aware scheduling, while runtime-prefix generality has measurable overhead. Production dispatch is unchanged pending a proper planner and broader matrix. |
| 2026-07-20 | `c33e26adad` + isolated H200 compile probe | Confirmed that the direct whole-rank BF16 expand has no intrinsic rank-64 limit: with `BM=16, BN=32`, ranks 64/128/256/512/1024 compiled and passed correctness with 32/32/40/40/72 registers per thread, zero spills, and 6/12/24/48/96 KiB shared memory. | The current `max_lora_rank <= 64` branch is only a caller policy. Implement a sliced looped-rank fallback before claiming rank-greater-than-64 gate/up support, then autotune whole-rank versus looped-rank by Hopper/Blackwell geometry. True expand split-K remains an optional under-occupancy candidate, not the default large-rank path. Production dispatch is unchanged. Raw results: `benchmark_results/h200_sliced_20260720/rank_above64_direct_h200.json` and `rank_very_high_direct_h200.json`. |
| 2026-07-20 | planning checkpoint | Recorded the implementation-ready general sliced A/B scope, retained segmented/grouped/direct row schedules behind one semantic API, and enumerated the work remaining after the linear LoRA algebra is complete. | Begin implementation and Hopper validation in new post-Phase-1a commits; defer Blackwell schedule selection until the same saved matrix can run on GB300. |
| 2026-07-20 | `650f8b1a7f` | Checkpointed the benchmark-only general sliced-expand testbed separately from production. | Pre-commit and syntax checks passed; the source matches the retained H200 benchmark snapshot. |
| 2026-07-20 | `a8a2e1f222` | Replaced the separate production flat/two-slice kernels with one semantic sliced LoRA-B source compiled as aligned-flat, uniform-sliced, or descriptor-ragged; added compact-B/output-gap descriptors without changing the current runner API. | H200 correctness and graph replay passed, including rank 8 and fixed-capacity `GROUP_SIZE_M=4`. The final old-vs-new 24-case sweep improved all cases by 1.23%–9.93% (median 3.72%). Raw result: `benchmark_results/h200_sliced_20260720/production_sliced_b_ab_h200.json`. Looped rank, dense row schedules, and provider epilogues remain open. |
| 2026-07-20 | planning clarification | Froze the Phase-1 routing boundary around existing standard top-k, EP localization, DeepGEMM preprocessing/finalize, and block alignment; recorded logical rather than physical gate/up pairing. | Focus kernel work on LoRA GEMMs and provider epilogues. Target LoRA-A-only overlap followed by fused LoRA-B + activation + optional quantization; retain materialized A+B overlap only as a measured fallback. |
| 2026-07-20 | `e86158e963` | Added bounded whole-rank/looped-rank sliced LoRA-B scheduling without a rank-64 correctness ceiling. | H200 correctness covers rank 8 through 257; per-device selection remains offline-tuned policy rather than a hard rank condition. |
| 2026-07-20 | `32fe6f74d8` | Added the packed-factor indexed LoRA-A primitive with typed row plans and explicit split-K precision. | H200 correctness and graph replay passed for token/pair domains, packed rank 257, BF16/FP16, dirty-output clearing, and FP32 split-K. |
| 2026-07-20 | `9c54c147fb` | Connected bounded A to sliced B, added destination-typed store/add/reduce epilogues, fixed local-ID EP per-expert-A to shared-B stale rows, and ran the broad Hopper schedule sweep. | Focused H200 suite: 42 passed. Best tuned new A+B beats the old compilable chain across all 24 comparable shapes; rank-96/128 gate/up now runs. DSA FP32 destination is recorded for dense migration. Blackwell tuning, dense row schedules, and fused provider epilogues remain. |
| 2026-07-20 | `ec8a4eccb9` | Ran the matching broad and exact-production GB300 A/B sweeps, then fixed indexed A to honor the runner's existing lowercase stage setting. | Focused GB300 suite: 42 passed. Stage propagation removes 18.72%-62.38% of high-rank gate/up time and is neutral-to-11.67% faster for down. The remaining best-FP32 gap is 0.19%-17.76%; device-specific schedule and workspace-precision policy, mixed/multi-adapter rows, and full-runner two-stream timing remain. |
| 2026-07-20 | `72055b46fd` | Completed the ragged sliced-B ABI with compiled layouts, aligned per-tile runtime descriptors, per-virtual-expert layout maps, no-target groups, and graph-stable in-place map updates; removed benchmark-only wrappers from the production API. | H200 and GB300 full suites: 45 passed on each. Both tuned matrices confirm the same hybrid ordering: one eligible compiled-layout/full-union launch remains fastest, while descriptor avoids the large multi-launch tax; tile choices remain device-specific. Raw results: `benchmark_results/h200_ragged_20260720/` and `benchmark_results/gb300_ragged_20260721/`. |
| 2026-07-21 | `72055b46fd` + current-source GB300 benchmark | Compared aligned-flat and uniform-sliced gate/up schedules at each schedule's independently best legal BN across 12 half widths, 5 token counts, and ranks 16/32/64/128; then long-reran H=112/176/512. | Flat is not universal: sliced gains are large when flat is forced to BN16, while wide aligned shapes are tied or modestly favor flat. Retain the two constexpr schedules under one source, use only a simple interim selector, and move optimization effort to the end-to-end gate/up activation/quant and down reduction/finalize endpoints. Raw results: `benchmark_results/gb300_gate_up_bn_sweep_20260721/`. |
