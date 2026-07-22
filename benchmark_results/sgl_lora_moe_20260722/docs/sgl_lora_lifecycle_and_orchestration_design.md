# SGLang LoRA lifecycle and orchestration design

Date: 2026-07-22<br>
Repository snapshot: final `sgl-lora` campaign tip `f2f406e056`, also recorded by the
repository-local benchmark manifest; historical design-review point `4ffaee0b41`<br>
Status: **authoritative target design and migration contract**<br>
Latest migration-order review: implemented through the Phase-1 MoE execution slice;
Phase 2/3 remain target architecture.<br>
Companion plans:

- `sgl_lora_refactor_plan.md` — implementation order and provider roadmap
- `sgl_lora_worklog.md` — current done/in-progress/gap ledger and validation policy
- `vllm_lora_study_final.md` — comparative vLLM study and transferable lessons

## 0. 2026-07-22 implementation-alignment checkpoint

The target object model in this document is still authoritative, but the MoE
execution slice is no longer the original Phase-1a materialized prototype. The
current branch now has the following concrete boundary:

~~~mermaid
flowchart LR
    FB["ForwardBatch<br/>request IDs + phase"] --> LM["Existing LoRAManager<br/>host assignment"]
    LM --> BI["LoRABatchInfo<br/>stable device assignment"]
    BI --> LL["FusedMoEWithLoRA<br/>layer boundary"]
    LL --> RP["Immutable MoeLoraExecutionPlan<br/>provider + phase + graph + rank + row policy"]
    RP --> N0["No-active adapter<br/>base-provider graph"]
    RP --> C0["Provider-neutral C0<br/>serial LoRA injection"]
    RP --> C2["BF16 C2<br/>fused consumer + finalize"]
    RP --> C3["BF16 C3<br/>gate-A-only overlap"]
    C0 --> BP["Base provider contract<br/>BF16 / FP8 / NVFP4 / Marlin"]
    C2 --> BP
    C3 --> BP
    BP --> CI["StandardCombineInput<br/>dispatcher / collective ownership"]
~~~

The execution plan is host-resolved before launch and is safe to close over in a
CUDA graph. It never infers decode versus prefill from a token-count threshold.
The base MoE provider owns packed/quantized weight and activation layouts; the LoRA
layer owns only the two semantic injection sites. Packed top-k, local/global expert
translation, bounded physical shared-expert IDs, activation contracts, destination
dtype, routed scaling, workspace admission, and collective ownership are explicit
contracts rather than incidental kernel parameters.

The final Phase-1 implementation boundary is deliberately narrower than the target
architecture in later sections:

- shared-outer uses an exact evidence-bounded selector. Its host policy passed 9
  focused tests, and its combined TorchNative-bound/physical-ID/PDL suite passed 56
  tests plus 10 subtests on each of H200 and GB300;
- physical shared IDs are promoted for the Standard contiguous/EP1 path and other
  safe layouts that do not remap physical shared IDs per rank. Per-rank physical
  shared EP>1 and advanced A2A remain unpromoted;
- a no-active-adapter call invokes the base layer's resident quant method/provider;
  it does not construct or force a Triton quant path;
- active FP8 rejects static-activation scaling. Blackwell FP8 requires resident
  packed UE8M0 scales from the attached quant provider;
- Marlin W4A16 is attachable, and its per-invocation workspace/dirty-destination
  repair passed the terminal `f2f406e056` H200 and GB300 smoke;
- native CuTe DSL NVFP4 W4A4 remains a synthetic/testbed provider and is not
  serving-reachable. The CuTe DSL/TMA measurements are evidence-only, not a serving
  provider promotion;
- provider-neutral C0 validation is not broad quantized checkpoint/server
  graduation.

The controlled server result remains a design constraint: historical pre-planner
SGL trailed experimental TRTLLM by 12.56%-14.71%, while final C2 trails by
9.43%/9.44%/10.39% and C3 by 5.12%/5.54%/10.48% at BS1/16/32. The historical range
must not be presented as the final planner result.

Route-plan reuse is similarly bounded: gate/up and down may reuse a matching plan
inside one MoE layer invocation. It is not whole-forward or cross-layer memoization.

What remains intentionally outside this Phase-1 execution boundary:

- immutable `AdapterKey`, aliases, leases, and two-phase distributed publication;
- a native ordinal-based `BatchAdapterTable` replacing physical slots in request
  semantics;
- byte-budgeted multi-tier residency and stable mixed-rank bucket ownership;
- the general dense/special-layer site compiler and sparse per-model extensions;
- MTP/EAGLE and advanced A2A graduation.

Those are not silently claimed by the kernel campaign. The retained manager supplies
the current assignment and residency bridge, while lifecycle tests exercise load,
eviction, slot recycling, graph replay, and unload/reload so the later control-plane
replacement has a behavioral oracle.

## 1. Executive recommendation

The cleanest split is:

1. **Adapter control plane** — identity, catalog, lifecycle, leases, distributed
   publication, admission, host artifacts, and device residency.
2. **LoRA execution plane** — model-site semantics, canonical batch assignment,
   provider planning, stable graph workspaces, and kernels.

“Adapter control plane” is a better subsystem name than “request management.”
Within it, “adapter lifecycle and orchestration” describes the work. The boundary
between the two planes is not a Python layer wrapper; it is a small set of versioned,
typed objects:

~~~text
AdapterKey + AdapterArtifact + CanonicalSliceWeights
ModelLoRASchema + LoRASliceSpec
BatchAdapterTable + LoRABatchAssignment
ResidencySnapshot
DeviceAdapterAssignment
LoRAExecutionContext
ProviderPlan
ProviderInvocation
ExecutionCompletionTracker
~~~

The current three-step priority remains right:

1. MoE execution and providers.
2. Dense and special-layer execution.
3. Adapter control-plane refactor.

However, the semantic contracts above should be introduced during the kernel work.
Otherwise the new kernels will inherit the current manager, slot, segmentation, and
model-name coupling and will need another rewrite later.

The strongest refinement to the previous proposal is:

> Canonical batch metadata should use a **batch-local adapter ordinal**, not a
> physical GPU slot. Physical `SlotRef(slot, generation)` values belong to a
> separate residency snapshot and provider plan.

Ordinals are the semantic/planning identity. They are not required to be the final
index read by every kernel. Before launch or CUDA-graph replay, each provider
materializes a stable-address `DeviceAdapterAssignment` whose index space is
declared as `ORDINAL` or `SLOT`. Dense providers will usually resolve ordinals
to physical slots once; a virtual-expert MoE provider may retain compact ordinals
plus an ordinal-to-slot table when that is faster. All mapping contents are refreshed
in place before replay.

That keeps slot reuse, EP/A2A, DP attention, and future residency policies out of
request semantics without forcing one physical indexing strategy on every kernel.

This file is authoritative when another planning note disagrees on object shape,
ownership, or lifecycle. The implementation plan may summarize these contracts but
must not redefine them.

## 2. The essential lifecycle

Stripped of implementation details, LoRA serving only needs this flow:

~~~mermaid
flowchart LR
    A["Resolve adapter source"] --> B["Compile immutable artifact"]
    B --> C["Publish AdapterKey"]
    C --> D["Acquire request lease"]
    D --> E["Admission and device activation"]
    E --> F["Build one batch assignment"]
    F --> G["Build provider plans"]
    G --> H["Execute all LoRA sites"]
    H --> I["Release execution snapshot"]
    I --> J["Release request lease"]

    C --> K["Unpublish or replace"]
    K --> L["Drain leases"]
    L --> M["Evict device and host state"]
~~~

The important separations are:

- A name is not an immutable adapter identity.
- A published adapter is not necessarily resident on GPU.
- A GPU slot is not an adapter identity.
- Request assignment is not provider segmentation.
- Logical gate/up, Q/K/V, expert, and scaling semantics are not physical layouts.
- Model support is not a list of string suffixes.

## 3. Current SGLang lifecycle

### 3.1 Server launch

Current startup has two independently initialized views:

1. `ServerArgs.check_lora_server_args()` parses startup paths, capacities,
   targets, and execution compatibility. Startup adapters receive deterministic
   IDs derived from name and path so separately initialized nodes agree.
2. `TokenizerManager.init_lora()` creates the logical `LoRARegistry`.
3. Each model rank constructs a separate `LoRAManager` after the base model is
   loaded.
4. `LoRAManager.init_state()` performs all of the following:
   - load adapter config and weights to CPU;
   - normalize checkpoint names and fused weights;
   - infer the union of targets and maximum rank;
   - detect one shared-outer format for the whole pool;
   - replace matching model modules with LoRA wrappers;
   - allocate all GPU adapter buffers;
   - load the base/no-adapter entry;
   - bind stable pool tensor views into every wrapper.
5. CUDA-graph setup separately allocates shared MoE workspaces and dense batch
   metadata.

This works, but `LoRAManager` currently owns artifact parsing, model compilation,
layer mutation, CPU caching, GPU residency, batch planning, provider state, and graph
initialization. Those are different lifetimes and should not share one owner.

### 3.2 Request entry and identity

The public request field named `lora_path` is normally interpreted as a registered
adapter **name**:

1. Tokenizer-side validation checks that LoRA is enabled.
2. A name missing from the live registry may be implicitly reloaded from
   `lora_ref_cache`.
3. `LoRARegistry.acquire()` resolves name to `lora_id` and increments an
   in-flight counter.
4. The scheduler-side `Req` stores `lora_id`.
5. `Req` concatenates the ID into `extra_key`, so radix-cache prefix matching
   is adapter-sensitive.
6. Completion or abort is expected to release the registry counter.

The cache identity idea is correct, but the representation is weak:

- startup identity is name plus path, not an immutable revision or digest;
- dynamic reload of the same artifact gets a new UUID;
- concatenating an ID into an arbitrary string `extra_key` is not a structured
  namespace;
- user name, artifact revision, and residency identity are conflated.

### 3.3 Scheduling and device activation

Before admitting a new request, the scheduler:

1. collects adapters used by running and chunked requests;
2. applies `LoRADrainer` fairness;
3. checks whether the adapter is already active;
4. either starts asynchronous GPU loading through `LoRAOverlapLoader` or asks
   `LoRAManager.validate_lora_batch()`;
5. delays admission until the load event is ready.

`LoRAMemoryPool` then owns the rank-local UID-to-slot map and device LRU/FIFO.
This is distinct from tokenizer-side `max_loaded_loras`, which governs registered
CPU artifacts. The two tiers are useful, but their state transitions are implicit.

Today `None` is loaded as the base-model entry and can consume a normal adapter
slot. A cleaner design gives base/no-adapter a semantic ordinal 0 and, only when a
kernel needs it, a dedicated immutable zero view that does not consume user adapter
capacity.

### 3.4 Forward preparation

`ForwardBatch.init_new()` copies request IDs into `ForwardBatch.lora_ids`.
Immediately before forward:

1. missing adapters are copied into GPU slots;
2. logical IDs are translated to physical slot indices;
3. rank and scale arrays are built;
4. the selected dense backend creates segments, permutations, or chunks;
5. MoE token-to-adapter metadata is derived;
6. every wrapper reads the same mutable `backend.batch_info`.

The current `LoRABatchInfo` mixes two kinds of data:

- semantic request/token assignment;
- Triton/CSGMV-specific segmentation and ordering.

CSGMV consequently needs duplicate `req_seg_indptr` and
`req_weight_indices` fields because its physical chunk segments cannot describe
the original request assignment. That is direct evidence that these objects should
be separated.

### 3.5 Layer execution

The current wrapper family covers:

- vocab embedding;
- lm head;
- column-parallel linear;
- merged-column linear;
- QKV-parallel linear;
- row-parallel linear;
- replicated linear;
- fused MoE.

Dense wrappers reproduce substantial parts of their base layer forward logic.
`FusedMoEWithLoRA` constructs a `LoRAInfo` from global mutable backend state on
every layer call, dispatches tokens, calls a LoRA-aware runner, and combines tokens.

The semantic MoE injection points are correct:

1. gate/up LoRA delta after GEMM1 and before activation;
2. down LoRA delta after GEMM2 and before router-weighted reduction/combine.

Those two points, plus the base runner’s activation and routing semantics, should be
the stable model contract. The temporary packing, quant-info class, top-k alignment,
gate/up interleave, and provider runner calls should not be layer-facing contracts.

### 3.6 Dynamic load and unload

Dynamic load currently:

1. creates a random `LoRARef`;
2. fans out load to model workers;
3. registers the name in the tokenizer catalog after worker success;
4. if over `max_loaded_loras`, unregisters and unloads an LRU adapter.

Dynamic unload currently:

1. unregisters the name so no new request can acquire it;
2. waits for its active request counter to reach zero;
3. fans out backend unload.

This is directionally correct, but it is not a transaction. Worker staging,
cross-rank validation, publication, rollback, revocation, CPU eviction, and GPU
eviction are not explicit operations.

### 3.7 CUDA graphs

LoRA graph support relies on:

- stable maximum-sized metadata buffers;
- in-place updates before replay;
- stable pool addresses;
- shared MoE intermediate workspaces;
- capture-time dummy base assignments;
- global mutable backend metadata.

This is viable for one serial execution lane. It is fragile for two-stream overlap,
TBO/DBO, DP attention, concurrent graph families, slot reuse, and prefill graphs.
Persistent graph storage needs explicit lane/epoch ownership, while an execution
context must retain its residency snapshot until every stream has finished.

## 4. Concrete current risks

These are architecture findings, not reasons to delay the kernel-first plan. They
should become control-plane tests and later commits.

### 4.1 Lifecycle and transaction risks

1. **Acquire/unload race.** Registry lookup releases its lock before incrementing
   the request counter. Unload can unregister, observe zero, delete the counter, and
   remove the backend adapter before acquire increments it.
2. **Lease leak before dispatch.** Adapter acquisition occurs before all
   tokenization/validation/send work. Some exception paths remove request state
   without a guaranteed LoRA release, which can block unload forever.
3. **Partial worker load.** `LoRAManager.load_lora_adapter()` inserts config state
   before loading weights and does not roll back every partial dictionary on failure.
   Tensor-based loading has the same shape.
4. **Partial distributed publication.** Load/unload consumers rely on the first
   gathered result. DP size 1 hides rank-aggregation ambiguity today.
5. **Unregister-before-unload failure.** A backend unload failure has no catalog
   rollback, leaving the adapter unavailable even though rank-local state may remain.
6. **Capacity eviction after publication.** A new adapter can be registered before
   LRU eviction completes; eviction failure can leave partially committed capacity.
7. **Unload means two different things.** Because `lora_ref_cache` is retained, a
   later inference can silently reload an explicitly unloaded adapter. Evict and
   revoke need separate APIs.
8. **Lazy device removal.** Worker dictionaries are deleted, but stale GPU
   UID-to-slot entries remain until normal pool eviction.

### 4.2 Identity and invalidation risks

1. Name/path UUIDs do not prove artifact immutability.
2. Same-content implicit reload generates a new dynamic ID and loses safe cache reuse.
3. Same-path mutated content can retain a deterministic startup ID after restart.
4. Slot reuse is not validated through an immutable generation-checked execution
   snapshot and refreshed device mapping.
5. A cached provider path can therefore be structurally reusable yet read a
   different physical payload after eviction without detecting the rebinding.

### 4.3 Batch-boundary risks

1. Backend metadata is built before some later padding/gather transformations.
2. DP-attention and A2A can change token ownership/order without carrying adapter
   assignment through the same transform.
3. Speculative verification expands tokens per request and needs an explicit mapping.
4. Mutable singleton `backend.batch_info` has no in-flight lane ownership.
5. Prefill CUDA-graph paths do not yet have the same complete LoRA replay contract as
   decode.

### 4.4 Model-contract risks

1. `should_apply_lora()` exists in twelve model files but is not called by
   `LoRAManager.init_lora_modules()`; comments refer to a gate that is absent.
   Vision/projector modules can therefore collide with intended text-only suffixes.
2. `supported_lora_modules` is not authoritative. Declarations include Baichuan,
   ChatGLM, GPT-BigCode, Gemma3n, and MiniCPM names that are removed by the current
   canonical allow-set.
3. Module matching relies on suffixes, substrings, and longest-match behavior.
4. Physical meaning is inferred from names such as the presence of “moe”.
5. Gate/up interleave is sometimes inferred indirectly from activation parameters
   even though the base runner already has an explicit layout field.
6. One pool-wide shared-outer choice prevents mixed compatible adapter formats.
7. Pipeline-parallel ranks allocate by the global layer count instead of compiling
   only their local sites.
8. `get_hidden_dim` and `get_stacked_multiply` transfer all names to a model override
   when the method exists; there is no per-site `NOT_HANDLED` fallback, so special
   models duplicate common formulas.
9. External checkpoint-dialect parsing, model aliasing, tensor fusion, and final
   runtime binding are mixed in string-replacement/substring code instead of separate
   typed stages.
10. Fused projection coverage is positional and family-specific. Generic merged
    columns can represent only a prefix of base partitions, while QKV, gate/up, GDN,
    Mamba, and MLA each have different anchor and zero-fill rules.
11. Public methods named `run_qkv_lora` and `run_gate_up_lora` imply different
    semantics even though most providers execute the same stacked shrink and
    offset-driven expand operation. A Triton equal-two-slice fast path has leaked into
    the layer-facing API.

## 5. Proposed architecture

~~~mermaid
flowchart TB
    subgraph CP["Adapter control plane - server-wide logical state"]
        API["API / request parser"]
        Catalog["AdapterCatalog"]
        Lease["AdapterLeaseManager"]
        Replica["AdapterReplicaCoordinator"]
        Admission["AdapterAdmissionPolicy"]
        API --> Catalog
        Catalog --> Lease
        Catalog --> Replica
        Lease --> Admission
    end

    subgraph WC["Worker compilation and residency - rank-local physical state"]
        Model["Loaded rank-local model graph"]
        Schema["ModelLoRASchema"]
        Source["PEFT / tensor source"]
        Compiler["AdapterArtifact compiler"]
        Store["Immutable AdapterArtifactStore"]
        Residency["DeviceResidencyCache"]
        Model --> Schema
        Schema --> Compiler
        Source --> Compiler
        Compiler --> Store
        Store --> Residency
    end

    subgraph FP["Per-forward orchestration - immutable execution snapshot"]
        Batch["ForwardBatch"]
        Assignment["BatchAdapterTable + LoRABatchAssignment"]
        Snapshot["ResidencySnapshot"]
        Planner["LoRAExecutionPlanner"]
        DeviceMap["DeviceAdapterAssignment"]
        Invocation["ProviderInvocation"]
        Context["LoRAExecutionContext"]
        Completion["ExecutionCompletionTracker"]
        Batch --> Assignment
        Assignment --> Planner
        Snapshot --> Planner
        Planner --> DeviceMap
        Planner --> Invocation
        Assignment --> Context
        Snapshot --> Context
        DeviceMap --> Context
        Invocation --> Context
        Context --> Completion
        Completion -- "release after fences" --> Snapshot
    end

    subgraph EX["Execution providers - provider-owned physical layouts"]
        Dense["Dense / special providers"]
        Moe["MoE virtual-expert providers"]
        Graph["GraphLane + GraphMetadataArena"]
        Context --> Dense
        Context --> Moe
        Context --> Graph
    end

    Replica -- "prepare / commit / abort" --> Compiler
    Admission -- "activate AdapterKey set" --> Residency
    Lease -- "immutable AdapterKey" --> Assignment
    Residency --> Snapshot
    Schema -- "typed SiteHandle set" --> Planner
~~~

### 5.1 Ownership table

| Object | Owner | Lifetime | Must not own |
|---|---|---|---|
| `AdapterCatalog` | tokenizer/control coordinator | server | GPU slots, kernel layouts |
| `AdapterLeaseManager` | control coordinator | request | provider metadata |
| `AdapterFormatReader` | artifact compiler | one source/load attempt | model topology, provider packing |
| `ModelLoRAExtension` | model implementation/worker compiler | model instance | framework-dialect parsing, hot-path state |
| `ModelLoRASchema` | model worker | model instance | adapter residency |
| `AdapterArtifactStore` | model worker | registered/host-cached artifact | request mapping |
| `DeviceResidencyCache` | execution group/rank | device cache | user-visible names |
| `BatchAdapterTable` | scheduler/forward builder | one forward | physical slots |
| `LoRABatchAssignment` | forward builder | one forward | segments/chunks |
| `ResidencySnapshot` | residency cache | one in-flight execution | target resolution |
| `DeviceAdapterAssignment` | provider planner/graph lane | one execution or stable graph arena | user identity, artifact loading |
| `ProviderPlan` | provider | one forward or reusable graph key | logical identity |
| `ProviderInvocation` | provider/graph lane | one submitted execution | reusable kernel choice, user identity |
| `GraphLane` / `GraphMetadataArena` | model executor/provider | one in-flight lane / graph lifetime | catalog state |
| `LoRAExecutionContext` | forward context plus completion lease | one submitted execution | global mutable singleton |
| `ExecutionCompletionTracker` | model executor/residency reclaimer | submission through GPU completion | catalog/request identity |

`AdapterAdmissionPolicy` is the target byte-aware admission/prefetch seam. Until it
adds behavior beyond the current drainer, overlap loader, and batch validator, use a
thin bridge rather than creating a second policy owner.

### 5.2 Independent selection axes

The target configuration has independent axes. A LoRA engine must not be encoded as
a base MoE runner:

| Axis | Selects | Does not select |
|---|---|---|
| `--lora-execution-engine` | legacy versus `sgl_lora` orchestration and injection contracts | base GEMM implementation or quant layout |
| base MoE provider selector | Triton, DeepGEMM, CuTe DSL, Marlin, or another compatible provider | adapter lifecycle or request identity |
| dense LoRA provider selector | CSGMV, Triton, or a future provider | MoE provider |
| overlap policy | serial, two-stream, or later measured automatic scheduling | execution-engine identity |

Overlap policy is resolved by orchestration before entering the execution runner;
the runner obeys one explicit topology decision and must not reinterpret it through a
hidden token-count branch. Token count is one policy feature, not the policy itself.
The compatibility implementation may preserve the existing `T <= 256` automatic
decision while migration is in progress, but the target selector is keyed by forward
phase, graph/eager mode, device/provider, work geometry, and measured critical-path
slack. H200/GB300 T128/T256/T257/T2048 evidence shows no discontinuity at 256 and
different graph/eager behavior, so that scalar cutoff must not become semantic ABI.

During Phase 1a, `--moe-runner-backend=sgl_lora` may remain a compatibility input.
Argument normalization turns it into `--lora-execution-engine=sgl_lora` and the
runtime representation must eventually resolve the base provider separately.
Selecting the new engine also selects virtual-expert semantics internally; there is
no second positive semantic flag for users to coordinate.

The Phase-1a selector checkpoint `65e69838d9` reaches the intended selector shape: the compatibility
input is normalized out of the runtime `MoeRunnerBackend` domain, explicit base-provider
values remain independent, and `sgl_lora` implies virtual experts. Capability-based
provider selection remains later work.

## 6. Canonical data model

### 6.1 Identity and artifact

~~~text
AdapterKey
  immutable_revision       # content digest, resolved HF commit, or caller version
  base_model_compatibility # optional immutable base/config identity

AdapterAlias
  namespace
  public_name
  adapter_key

LoRACacheIdentity
  model_revision_or_weight_epoch
  adapter_key_or_BASE

AdapterArtifact
  key
  source_metadata
  validated PEFT features
  logical_weights: LogicalWeightId -> CanonicalLoRAWeights
  resolved_site_coverage: SiteId -> LogicalWeightId + slice/view
  per-site rank and scaling policy
  target coverage
  checkpoint layout description

CanonicalLoRAWeights
  slices: CanonicalSliceWeights

CanonicalSliceWeights
  slice_id -> PRESENT(A_ref, B_ref, rank, scale) | NOT_TARGETED
  immutable tensor/view references; slices may share one factor reference
  DECLARED_BUT_MISSING never enters a valid artifact
  no provider padding, interleave, max-rank stride, or synthesized zero tensor
~~~

`AdapterKey` is alias-independent and belongs in structured KV/radix cache
identity. A mutable user-facing name belongs to `AdapterAlias`, not to artifact or
execution identity. Replacing an adapter under the same public name stages a new
key, atomically switches the alias binding, and lets old requests drain against the
old key. Multiple aliases may intentionally point at the same immutable key.

Every radix/KV-cache implementation uses the structured `LoRACacheIdentity` (or an
equivalent typed field), never a mutable alias, concatenated free-form string, or GPU
slot. This keeps cache correctness independent of residency and makes the same
contract reusable across cache implementations. A base-weight publication advances
the model revision/weight epoch; an adapter replacement changes the adapter key.

The artifact compiler should parse at least:

- ordinary LoRA versus unsupported DoRA;
- standard scaling and rsLoRA scaling;
- global and per-site rank/alpha patterns;
- split versus already-fused checkpoint tensors;
- per-expert versus shared-outer factors;
- missing optional slices and explicit zero-fill;
- local expert selection;
- tied embedding/head and vocabulary metadata.

### 6.2 Model schema

~~~text
ModelLoRASchema
  model_revision
  local_pipeline_stage
  sites: SiteId -> LoRASite
  weight_bindings: LogicalWeightId -> WeightBindingGroup
  checkpoint_alias_rules
  cross-site fusion constraints

LoRASite
  id                         # namespace + global layer + scope + logical role
  module_path
  kind                       # COLUMN, ROW, MERGED, QKV, REPLICATED,
                             # EMBEDDING, LM_HEAD, MOE, CORRECTION
  logical_slices
  adapter-logical input/output dimensions
  rank-local logical dimensions and ownership
  TP/EP/PP ownership
  injection point
  activation/routing semantics
  expert topology and sharing scope
  base/provider physical and padded-layout descriptor
  required semantic features
  logical_weight_id
  execution modes and slice view

LoRASliceSpec
  id and semantic role          # Q, K, V, GATE, UP, Z, VALUE, ...
  logical input/output dimensions
  base logical output range
  rank-local output mapping and TP replication/shard policy
  presence policy               # REQUIRED or OPTIONAL
  factor-sharing group, when slices intentionally share A or B
  base physical-layout view     # input to provider packing, not kernel ABI

WeightBindingGroup
  logical_weight_id
  canonical artifact weight
  one or more consuming SiteId values
  per-site slice/view transform
  per-forward-mode activation predicate

LoRASiteCandidate
  exact rank-local module path and module object
  global layer and language/vision/routed/shared scope
  native module class
  logical dimensions/partitions and sharding metadata
  separate physical/padded storage metadata

ModelLoRAExtension
  resolve_site(candidate) -> OVERRIDE(spec) | SKIP(reason) | NOT_HANDLED
  resolve_weight_alias(raw_key, schema) -> binding | NOT_HANDLED
  declarative binding/scope overrides for true model exceptions
~~~

The generic compiler should recognize the eight existing supported layer classes.
A model extension gets first refusal for each candidate, but `NOT_HANDLED` falls back
to the centralized module-class compiler. This is per-site rather than all-or-nothing:
a hybrid model can override one Mamba, GDN, MLA, or multimodal site while ordinary
QKV/MLP sites continue through shared logic. Model files should return a small
declarative extension object, not add model-name conditionals to hot kernels.

For generic sites, dimensions and partitions come primarily from the actual
constructed module (`input_size`, `output_size`, semantic fused partition sizes,
embedding metadata, or `FusedMoE` geometry) rather than being blindly recomputed from
the HF config. The compiler must preserve **adapter-logical**, rank-local logical, and
provider-physical/padded dimensions separately: a padded expert weight or a provider-
mutated physical `hidden_size` never changes the checkpoint's logical LoRA shape.
Artifact validation uses logical dimensions; provider packing owns padding. The
HF-config dimension table remains a legacy fallback or semantic cross-check until
every existing site class exposes unambiguous structural metadata.

The schema declares required semantics; it does not know which provider supports
them. The planner compares those requirements with provider capabilities and returns
either a supported plan or a stable, explicit rejection/fallback reason.

A logical adapter weight is not necessarily consumed at one physical call site.
DeepSeek MLA is the concrete counterexample: one `kv_b_proj` weight can feed the
ordinary prefix projection and multiple absorbed-attention correction sites, with
different slices selected by forward mode. `WeightBindingGroup` represents this
one-artifact-to-many-sites relation without introducing a general graph IR.

### 6.3 Batch-local assignment

~~~text
BatchAdapterTable
  domain                    # LOCAL_BATCH or EXECUTION_GROUP
  ordinal 0 -> base / no adapter
  ordinal 1 -> AdapterKey A
  ordinal 2 -> AdapterKey B
  ...

LoRABatchAssignment
  batch_adapter_table
  request_indptr
  request_adapter_ordinals
  token_adapter_ordinals
  speculative token expansion metadata
  has_active_adapter

AssignmentView
  source_assignment
  token selection/permutation
  consumer-specific adapter ordinals
~~~

Why ordinals instead of slots:

- request semantics remain valid if a GPU slot is evicted or reused;
- each EP destination can map the same ordinal to its own local slot;
- the ordinal can travel with tokens through A2A;
- virtual-expert IDs can use a compact batch-local adapter dimension;
- base/no-adapter does not consume normal residency capacity.

Ordinals need only be unique within the table that defines them; sorting by
`AdapterKey` is useful for reproducibility but is not a local correctness
requirement. DP-attention ranks can start with different local tables, so they must
build one identical `EXECUTION_GROUP` table before gathering tokens. The forward
preparer owns a small metadata exchange over active `AdapterKey` values, constructs
the union using a coordinator-broadcast order or a canonical deterministic order,
and remaps each rank's local ordinals into that group table. A provider that cannot
perform this agreement may carry `(source_rank, local_ordinal)` as an explicit
fallback payload; plain local ordinals must never cross that boundary ambiguously.

`AssignmentView` represents consumer-specific transforms without mutating the
canonical request/token assignment. Speculative expansion, lm-head pruning, and
multi-pass logits processing produce one or more views with explicit token indices.

During migration, a legacy bridge can materialize the current physical
`req_weight_indices` from an assignment plus residency snapshot. New providers
must not make that physical field canonical.

### 6.4 Device assignment and CUDA-graph refresh

~~~text
KernelAdapterIndexSpace = ORDINAL | SLOT

DeviceAdapterAssignment
  index_space
  persistent token/request index buffers
  persistent ordinal_to_slot
  persistent ordinal_to_generation # optional device assertion/debug view
  persistent ordinal_valid
  per-site or provider-family rank/scale/enabled views
  host mapping_epoch
  capacity
~~~

Canonical ordinals may remain device-visible; they are not forced to disappear at
the host boundary. Each provider chooses its kernel-facing index space:

- `SLOT`: translate token ordinals to local physical slots once during plan refresh;
- `ORDINAL`: retain compact ordinal-based virtual experts and read a stable
  ordinal-to-slot or pointer table;
- a provider may compact active weights into batch-local views when the copy cost is
  justified.

The mapping table cannot assume one adapter-wide rank or scale: PEFT rank/alpha
patterns and partially supported sites can make these values site-specific.
Provider/site plans therefore own stable per-site or family views.

All device buffers have maximum-capacity stable addresses. Before eager launch or
graph replay, the planner refreshes their contents in place from
`LoRABatchAssignment + ResidencySnapshot`. CUDA-graph replay contains no Python,
but it reads the refreshed buffers. Captured graph keys include provider, site
signature, architecture/quantization, graph lane, shape/capacity bucket, and kernel
index space; they do **not** include adapter keys, ordinal permutation, slot
generation, or mapping epoch. Adapter churn therefore refreshes data rather than
causing recapture.

Before launch, the context verifies every leased `SlotRef.generation` against the
residency cache and verifies that the device mapping was refreshed for this execution.
The lease then prevents either binding from changing until completion. A provider may
keep `ordinal_to_generation` for device assertions, but correctness must not require a
generation branch on every weight load. `mapping_epoch` detects stale host refreshes
and invalidates data-dependent routing cache contents. Neither epoch nor generation is
a reason to recapture a graph when addresses and shapes are unchanged.

### 6.5 Residency

~~~text
SlotRef
  slot
  generation

ResidencySnapshot
  batch_adapter_ordinal -> SlotRef or DISABLED
  slot_table_epoch
  references to static per-slot site-enabled bitsets
  readiness events
  SlotLease values
~~~

Activation is transactional:

1. reserve a slot;
2. increment its generation;
3. reset all site views and enabled masks;
4. copy or pack every required local site;
5. record readiness;
6. publish the `SlotRef`;
7. retain it until all execution streams release the snapshot.

Failure before publication clears the reserved payload/masks and returns the slot to
the free state without exposing its new generation.

Readiness/completion events are allocated lazily for provider lanes that actually
overlap loading or execution; serial lanes may use the already-ordered main stream.

Reset-before-write and per-site enabled masks are mandatory so a partially targeted
adapter cannot observe stale weights from the previous occupant. Site-enabled
bitsets are built or reset during slot activation and stored with the slot payload;
forward preparation references or compacts them and must not iterate over hundreds
of sites in Python on every step. Zeroing may be skipped only when the compiler
proves that the new payload overwrites every byte consumed by that provider/site.
`slot_table_epoch` advances whenever a physical binding changes and is captured by
the snapshot; the derived device mapping receives its own `mapping_epoch`. Both are
refresh/validation versions, not graph executable keys.

Slot release is GPU-completion-based, not Python-scope-based:

1. acquiring a snapshot increments a `SlotLease` for each referenced slot;
2. before a provider copies, packs, or reads a published slot payload, every consuming
   stream waits on that slot's readiness event; a serial provider may rely on the
   already-ordered stream that performed activation;
3. each provider declares every stream that can read those slots;
4. after the last read, execution records one completion event per independent
   stream, or one main-stream event after explicit joins;
5. after CUDA-graph replay, the executor records the completion event outside the
   graph; a multi-stream captured graph must join its internal streams before that
   event or return additional provider fences;
6. closing the Python execution context transfers leases and fences to an
   `ExecutionCompletionTracker` rather than making slots immediately reusable;
7. the tracker/reclaimer decrements leases only after all events complete;
8. activation and eviction exclude slots that are pinned, loading, or have nonzero
   execution leases.

Timeout never permits unsafe slot reuse. It leaves the adapter or lane in a draining
state and surfaces an operational error; a force operation must cancel affected
requests and still wait for GPU completion, or escalate to worker restart.

### 6.6 Execution context, graph lanes, and provider plans

~~~text
LoRAExecutionContext
  model_schema
  batch_adapter_table
  batch_assignment
  residency_snapshot
  device_assignments
  provider_plans
  provider_invocations
  graph_lane_lease

GraphLane
  lane_id
  lane_epoch
  GraphMetadataArena
  completion fence

DenseProviderPlan
  segments or chunks
  permutations
  site-specific LinearDeltaPlan values
  stable workspace

LinearDeltaPlan
  ordered active logical slices and output ranges
  rank/scale/factor-sharing views
  A reduction precision and final additive-destination dtype
  input/output TP shard policy and collective placement
  selected provider pack and kernel family

MoeProviderPlan
  MoeGeometry
  provider and kernel index space
  routing/kernel strategy and capacity buckets
  stable routing/stage buffer addresses
  base and LoRA physical-view descriptors
  stream/event contract

MoeProviderInvocation
  current token/expert counts
  localized expert-ID buffer contents
  token adapter indices in the declared kernel domain
  virtual-expert-ID buffer contents
  current stage-buffer views
~~~

The context should live in SGLang’s existing per-forward context mechanism rather
than a module-global mutable backend object. Its Python scope may end after forward
submission; `ExecutionCompletionTracker` retains its slot and graph-lane leases until
the declared completion fences fire.

A graph lane is one concurrently in-flight forward/replay lane, such as a TBO
microbatch or decode stream index—not each auxiliary CUDA stream. Side, communication,
and main streams used by one forward belong to the same lane. Each lane owns stable
metadata/workspace addresses and cannot be acquired by another batch until its
completion fence fires. `lane_epoch` increments on acquisition and catches stale
host handles; it is not a kernel input or a graph-recapture key. Add lanes only when
the executor actually permits concurrent in-flight forwards.

Reusable `ProviderPlan` objects contain kernel choices, capacities, descriptors,
and stable buffer addresses. Per-forward adapter mapping lives in
`DeviceAdapterAssignment`; routed expert IDs, counts, virtual IDs, and current stage
views live in `ProviderInvocation`. Both refresh stable arena storage before replay.
In the target architecture this distinction makes immutable host-plan memoization
independent of adapter composition and token routing while preserving correct slot
resolution. This does not enlarge the current device route-view cache: that cache is
valid only within one MoE layer invocation, never across layers or a whole forward.

Numerical precision is part of the plan, not inferred inside a kernel from whichever
tensor is convenient. LoRA-A reduction/workspace precision and final LoRA-B result
precision are independent fields. B converts A tiles to the B-factor compute dtype,
accumulates the dot in FP32, and converts once in its epilogue. When B adds into an
existing base result, that caller-owned destination tensor and its dtype are
authoritative. When a provider requests a standalone delta, an optional resolved
output dtype is carried by the plan; otherwise it defaults to the site's semantic
floating compute dtype. It must not default from a packed FP8/NVFP4 input tensor.
Passing both a destination tensor and a conflicting scalar dtype is invalid because
it creates two sources of truth.

Kernel schedule selection is also a `ProviderPlan` decision, not a rank cutoff hidden
inside the launcher. Offline tuning emits typed, device-specific records selected by
an exact or explicitly validated bucket match. The tuning key includes the actual
tensor device architecture (not merely the process current device), compiler/Triton
version, site semantics (`gate/up`, `down`, or a later fused consumer), kernel family
(`direct`, `generic`, or fused), input/weight and A-workspace precision, output width
and logical slice schedule, row schedule and physical GEMM row domain, logical token
count and router top-k as separate coordinates, valid and padded row capacity,
alignment block size, projection K, packed rank, active adapter/expert occupancy, and
shared-outer/factor-sharing layout. Virtual-expert capacity remains benchmark metadata;
once its routing effect is fully represented by occupancy and the row plan, it is not
a redundant dispatch key. Hopper, SM100, and SM103 records are independent, and one
device's nearest winner is not silently used for another. An exact table miss uses the
bounded correctness-first heuristic.

Resolver lookup coordinates, the physical GEMM row domain, and the effective launch
constants are distinct values. A lookup at logical `T` does not change a B kernel's
physical `[T*K,R]` input, and a lookup at `T*K` is not automatically more faithful:
routed rows are partitioned across virtual experts and may incur very different
padding. Persisted records store both the nominal resolver dictionary and the
effective family-specific configuration. Direct B, for example, does not consume
generic BK/stage fields and may normalize or replace BN; tuning only nominal fields
would create false distinctions. Benchmark overrides are tuning instrumentation, not
part of the production execution ABI.

The same resolved record controls both workspace allocation dtype and kernel launch;
the runner must not allocate FP32 from one rank-based rule while the launcher selects
a BF16 schedule from another. Tuning generation records correctness, p20/p50/p80,
compiler version, and neighboring candidates. Differences within the measured noise
band are ties and choose the simpler schedule. A record becomes automatic policy only
after repeat runs plus neighboring row/rank/routing distributions; raw single-node
winners remain worklog evidence rather than handwritten branches.

## 7. Model schema and artifact compilation

### 7.1 Compilation flow

~~~mermaid
flowchart LR
    M["Actual rank-local model modules"] --> C["Build LoRASiteCandidate values"]
    X["ModelLoRAExtension"] --> S["Resolve site: override, skip, or fallback"]
    C --> S
    G["Central module-class compiler"] -- "on NOT_HANDLED" --> S
    S --> V["Validate unique bindings and topology"]
    W["Adapter checkpoint or tensor input"] --> F["AdapterFormatReader"]
    F --> K["Structured RawAdapterWeightKey values"]
    K --> R["Resolve exact SiteId + logical slice"]
    X --> R
    AR["Central common alias/fusion rules"] -- "on NOT_HANDLED" --> R
    V --> R
    R --> N["Normalize canonical A/B tensors"]
    N --> L["Shard local TP/EP/PP ownership"]
    L --> ART["Immutable canonical AdapterArtifact"]
    ART --> P["Optional provider-packed cache"]
~~~

Important rules:

1. Scan the actual rank-local graph, not only the HF config.
2. Bind by exact typed sites and qualified paths, never bare substring alone.
3. Apply a model’s language/vision scope filter during discovery.
4. Keep logical slices explicit even if the base module is physically fused.
5. Validate all requested targets. Do not silently drop an unmatched target.
6. Normalize checkpoint layouts once at load.
7. Let providers pack from canonical weights; never expose their layout flags to a
   request or model class.
8. Log or expose the compiled schema for debugging.
9. Resolve ambiguous aliases such as bare `wk` or `wq_b` against qualified
   model sites; do not apply a model-family-specific global rename unconditionally.
10. Enforce paired gate/up/down requirements against actual routed-expert bindings
    or loaded routed-expert weights, not ambiguous normalized dense/MoE suffixes.

Before the full schema lands, the current unused `should_apply_lora` multimodal
filter and the Phase-1a dense-versus-routed target-pair ambiguity are standalone
correctness bugs. Fix them early with exact module paths/actual expert keys rather
than waiting for the complete refactor.

### 7.2 Model extensions, dimensions, and weight-name resolution

The user's proposed precedence is correct when it is **per candidate**:

1. Central code walks the actual rank-local graph and creates a
   `LoRASiteCandidate`; model code does not own traversal.
2. `ModelLoRAExtension.resolve_site(candidate)` gets first refusal.
3. `OVERRIDE` supplies only the exceptional fields, `SKIP` excludes the site with a
   reason, and `NOT_HANDLED` invokes the centralized module-class compiler.
4. The compiler emits one typed site whose dimensions, partitions, and ownership are
   validated against the live module.

The optional model-file hook is `get_lora_model_extension() -> ModelLoRAExtension`;
its absence means an empty extension and therefore pure central handling. Keep the
protocol and compiler in the LoRA package—the model file only constructs declarative
matchers/spec patches or a narrowly scoped resolver when per-layer logic is truly
dynamic.

The current `get_hidden_dim()` already calls a model method before shared logic, but
it treats the presence of that method as ownership of **every** module name. An
unrecognized name raises instead of falling back, and common QKV/MLP formulas are
duplicated in model files. The tri-state contract keeps the useful precedence while
removing that all-or-nothing behavior.

Checkpoint compatibility is a separate layered problem:

~~~text
AdapterFormatReader (PEFT first; tensor API uses the same output)
  -> RawAdapterWeightKey
       source dialect
       qualified module path segments
       A or B factor
       optional expert ID
       optional logical slice hint
       tensor and source metadata
  -> model-extension alias override, when claimed
  -> otherwise centralized structural resolution
       exact qualified schema binding first
       q/k/v -> QKV logical slices
       gate/up and w1/w3 -> gate/up logical slices
       w2 -> down
       common embedding/head aliases
  -> one ResolvedWeightBinding
       SiteId / LogicalWeightId / slice / factor / expert
  -> shape and coverage validation
  -> canonical tensor assembly
~~~

Do not normalize with unrestricted substring replacement and then guess a target from
the longest matching suffix. A sparse model override gets first chance after dialect
parsing; `NOT_HANDLED` enters the central resolver, where exact qualified binding wins
before common aliases. Every alias rule consumes structured path components;
zero-fill is allowed only for a slice the compiled site marks optional. An unmatched
or multiply matched weight produces an actionable error containing the raw key,
parsed form, candidate sites, and the rule that failed. The new explicit `sgl_lora`
path is strict; the legacy path may retain warning behavior during migration.

Strictness is scoped during staged migration. Every structured weight receives one
owner: `SGL_MOE`, `LEGACY_DENSE_OR_SPECIAL`, or `UNCLAIMED`. Phase 1b offers actual
routed-expert bindings to the new compiler and other supported weights to the legacy
bridge. Reject a weight only after both applicable resolvers decline it; dual
ownership is an ambiguity error. Coverage telemetry records which path claimed each
weight. When a layer family migrates, its ownership moves atomically to the new
schema compiler.

Framework compatibility and model compatibility remain separate axes:

- an `AdapterFormatReader` understands how a framework serialized LoRA metadata and
  keys;
- `ModelLoRAExtension` describes exceptional SGLang model topology and aliases;
- the central compiler owns ordinary SGLang module classes and common PEFT fusion
  rules;
- providers see only canonical bound weights, never external checkpoint names.

The base model's existing HF-to-SGLang mapper may seed qualified aliases when its
semantics match, but it is not automatically authoritative for LoRA: A/B factors,
optional slices, expert axes, and fused logical weights require LoRA-specific binding
descriptors.

Migration does not require editing every old model at once. A
`LegacyModelLoRAExtension` adapts existing hooks:

- `should_apply_lora` becomes an effective scope decision for ordinary discovered
  sites, while the compatibility bridge preserves today's separately opted-in
  embedding and lm-head handling before that legacy gate;
- `get_hidden_dim` and `get_stacked_multiply` are used when they handle the candidate,
  with `NotImplementedError` translated to `NOT_HANDLED`;
- `supported_lora_modules` is treated only as a discovery/alias hint, not proof of
  executable support.

New or migrated models declare only their exceptional matchers and bindings. Support
is proven by `compiled site + resolved artifact weights + selected provider`, never by
membership in a model-name list.

Consequently, remove the need for two drifting global registries (CLI target choices
and `_KNOWN_LORA_TARGET_MODULES`). The CLI may retain compatibility validation during
migration, but the compiled schema becomes the single authority for auto-detection,
explicit target validation, wrapping, dimensions, buffer allocation, and loading.

### 7.3 Avoid a general graph IR

The requirement set is finite. A small typed site system is easier to maintain than
a general adapter graph:

- linear site with sharding and logical slices;
- embedding/head site;
- MoE site with two injection points;
- absorbed/fusion correction site.

Model extensions should be declarative:

- exact path filters;
- checkpoint aliases;
- slice descriptors;
- dimension providers;
- activation and sharing properties;
- absorbed-projection correction hooks.

## 8. Model requirement inventory

There is no trustworthy model allow-list today. Support is structural: a model works
when its targets resolve to known roles, its modules are one of the eight wrapper
types, its dimensions/sharding are handled, and its adapter format matches the
normalizer. The following matrix captures the flexibility the new design must retain.

| Family / requirement | Current examples | Semantic requirement | Keep out of the semantic API |
|---|---|---|---|
| Standard dense decoder | Llama, Qwen, Gemma | Q/K/V and gate/up logical slices; TP ownership | concatenated B order, repeated A |
| Unequal/replicated QKV | GQA/MQA when TP exceeds KV heads | per-slice widths and replication groups | one hard-coded fused stride |
| Partial/asymmetric fused targets | adapters omitting K or up | optional slice with explicit zero contribution | implicit key absence heuristics in kernels |
| DeepSeek MLA | fused q_a + kv_a, q_b/kv_b | unequal slice boundary plus one logical weight bound to mode-dependent prefix/q-side/v-side execution variants | manual `first_output_dim` mutation |
| DSA indexer | `indexer.wq_b`, `indexer.wk`, `indexer.weights_proj` | qualified namespace; fusion compatibility | bare `wk` or `wq_b` matching |
| Standard gated MoE | Qwen MoE, DeepSeek/Kimi routed experts | two injection points, router-weight placement | runner-specific hook booleans |
| GPT-OSS MoE | GPT-OSS | clamped/alpha SwiGLU, explicit gate/up interleave, biases where supported | inferring layout from alpha |
| Non-gated MoE | Nemotron-H | one projection, activation kind such as ReLU2 | synthetic two-way gate/up shape |
| Shared-outer adapter factors | MoE adapters with expert dimension 1 | sharing scope per factor and per adapter | one pool-wide format flag |
| Model shared experts | DeepSeek-style routed plus dense shared expert | distinguish routed, model-shared, ordinary dense scope | ambiguous `gate_up_proj` suffix |
| EP/global expert IDs | standard local-ID and global-ID providers | global identity, local ownership map, `-1` absent sentinel | global-sized LoRA buffers |
| Mamba hybrid | Nemotron-H | partial `in_proj` slices, layer-dependent site kinds, and latent-MoE dimensions such as `moe_latent_size` | “first N partitions” inferred from tensor shape |
| GDN/linear attention | Qwen3.5 | Q/K/V/Z fused slices, output gate, mixed layer kinds; leave room for the currently unsupported `in_proj_ba` site | unconditional four-way repeated A |
| Nonuniform layer shapes | Gemma3n/Gemma4 and hybrid models | per-site dimensions, not one config-wide intermediate size | allocating one uniform shape policy |
| Multimodal wrapper | Qwen VL, Gemma MM, Phi4MM, Llama4, Ernie VL | language-only namespace filter unless explicitly targeted | unused regex hooks and suffix collisions |
| Embedding | tested Llama embedding LoRA | vocab ownership, TP reduction, token lookup | treating embedding as ordinary GEMM |
| lm head | tied Qwen head, pruned logprobs | vocab shard, tied semantics, and an ordered per-pass sequence of assignment views matching pruning | backend-global multipass metadata |
| Added vocabulary | currently rejected | explicit tokenizer plus embedding/head publication protocol | silently sized extra buffers |
| Legacy naming families | Baichuan, ChatGLM, GPT-BigCode, MiniCPM declarations | model-specific aliases to typed sites | expanding a global string allow-list |
| Pipeline parallel | local subsets of decoder layers | global SiteId with rank-local schema/materialization | buffers for every global layer on every rank |
| DP attention/A2A | gathered or redistributed tokens | carry adapter ordinal through identical token transform | reconstructing from pre-transform order |
| Speculative/MTP | target verify, future EAGLE/MTP | explicit target/draft adapter policy and expanded-token map | assuming one token per request |

### 8.1 Important detailed cases

#### Fused QKV and gate/up

An adapter may be trained against split PEFT modules or an already-fused SGLang
module. Q, K, and V can have unequal dimensions, and K/V can be replicated across TP
ranks. Gate and up can be split, concatenated gate-first, concatenated up-first, or
physically interleaved. The schema should preserve logical slice roles; artifact and
provider compilers decide repetition, concatenation, zero-fill, slicing, and layout.

The current implementation confirms that the apparent layer diversity is mostly
representation and API debt:

| Current path | Partial coverage actually representable | Physical kernel truth |
|---|---|---|
| Generic `MergedColumnParallelLinearWithLoRA` | Slice count is inferred from `A.shape / rank`, and offsets use the **first** base partitions. Only a trailing suffix can therefore be structurally absent. | Generic stacked A plus either generic N-slice B or the equal-two fast path. |
| QKV | Split normalization assumes Q and V exist; only missing K is synthesized as a zero slice. QKV B has a special load-time TP/KV replication rule. | The Triton “QKV” B kernel is already an arbitrary N-slice offset kernel. |
| Gate/up | Gate is the anchor; missing up is synthesized as zero. An orphan up works only if the model exposes a separate site. | Triton has an equal-two-slice B specialization; CSGMV and torch use a generic sliced operation. |
| GDN Q/K/V/Z | Split form fuses only when all four exist; already-fused A is repeated four times. | The same generic N-slice computation is sufficient. |
| Mamba gate/x and MLA q_a/kv_a | Mamba requires both split pieces; MLA anchors on q_a and attempts to zero-fill kv_a. | These are two-slice plans with different dimensions/bindings, not new LoRA algebra. |

So the remembered “only trailing parts may be missing” rule is accurate for the
generic merged-column packed representation, but it is not universal. Other
families preserve a full physical stack and encode a few hard-coded missing cases as
zeros. Neither behavior is a desirable semantic contract.

The audit also exposes concrete correctness gaps that the migration tests must pin
down: missing-K currently sizes K from V; the QKV wrapper uses K's shard width for V
even though the base layer supports a distinct V head width; MLA sizes a missing
kv_a B factor from q_a despite their unequal output dimensions; and a half-present
A/B pair can fall through as a zero contribution rather than a structured error.

The canonical contract is instead:

1. Every fused base projection has an ordered tuple of `LoRASliceSpec` values.
2. Artifact binding produces `CanonicalSliceWeights` with explicit coverage per
   slice. Any optional slice may be untargeted regardless of position; a required
   slice or a declared-but-missing A/B pair fails validation. Absence is not inferred
   from tensor size, key order, or all-zero contents.
3. Slices may have independent rank and scale. Already-fused artifacts may let
   several slices reference one shared A factor without eagerly repeating it.
4. A provider may materialize a dense full-slice pack with zeros, a compact
   active-slice pack plus mapping, or a specialized common-mask pack. Zero tensors
   are created only after coverage validation and from that slice's exact dimensions.
5. Dense execution exposes one sliced-linear operation. A provider can select a
   single-slice kernel, an equal-two-slice fast path, or a generic N-slice kernel;
   `run_qkv_lora` and `run_gate_up_lora` are not semantic APIs.
6. QKV replication, row-parallel collective placement, and physical gate/up
   interleave remain real concerns, but they live in slice shard policy,
   `LinearDeltaPlan`, and provider packing—not checkpoint-name branches or separate
   mathematical operations.

MoE remains a distinct executor because it has an expert/routing domain and two
non-final injection points. Its input projection still uses the same logical slice
model (`GATE`/`UP`, or one `VALUE` slice for a non-gated expert), while the provider
owns virtual-expert packing and may reuse one routed A/B primitive for input and down
deltas.

#### DeepSeek MLA and DSA

The q_a and kv_a projection is physically fused but has an unequal semantic boundary.
Absorbed MLA paths can bypass an ordinary linear and need explicit correction sites.
One logical `kv_b_proj` weight can be consumed by multiple physical variants:
ordinary prefix projection, q-side correction, and v-side correction. They share one
`LogicalWeightId`, while `WeightBindingGroup` selects the site, slice, and
execution variant for the current forward mode.
DSA indexer leaves collide with unrelated model leaves, so the qualified parent path
is part of the logical target. Enabling a fusion that removes those sites must fail
at startup rather than silently discard the adapter.

DSA also demonstrates why output dtype belongs to the site contract. On CUDA,
`indexer.weights_proj` intentionally performs BF16-by-BF16 GEMM with FP32 output.
Its LoRA-enabled execution must first create that same FP32 base destination, then
run LoRA-B directly into it so the FP32 accumulator is not rounded to BF16 before
the add. A post-add `.float()` is not equivalent. Other quantized linear providers
normally expose BF16/FP16 semantic results, but the actual semantic destination—not
the raw packed activation dtype or a global LoRA default—remains authoritative.

#### MoE

The schema must declare:

- gated versus non-gated;
- activation and alpha/clamp/limit;
- gate/up logical order;
- router weight application point;
- routed scaling application point;
- correction or GEMM biases;
- local/global expert geometry;
- per-expert versus shared factor scope;
- model-shared expert behavior;
- exact gate/up and down injection points.

Virtual experts remain the only execution model for the new engine. A provider is
free to choose a direct sparse kernel, grouped alignment, pointer table, or compact
batch-local pack; that choice does not change virtual-expert semantics.

#### Hybrid and multimodal models

Hybrid models prove that `module_name -> one global shape` is insufficient.
Different layers can contain attention, GDN, Mamba, dense MLP, gated MoE, or non-gated
MoE sites. Multimodal models prove that leaf suffixes are insufficient because vision
and language subgraphs can share the same names and layer numbers.

The compiled `SiteId` therefore needs namespace, global layer identity, scope,
logical role, and optional slice identity.

#### Embedding and lm head

An lm-head call is not always described by one sampled-token assignment. Logprob
pruning and multipass execution can produce an ordered sequence of token selections.
The logits processor produces `LmHeadPassPlan.passes: list[AssignmentView]`; every
pass contains the selected hidden-state indices and matching adapter ordinals. The
lm-head provider consumes those views rather than maintaining hidden mutable
multipass state. Tied embedding/head weight identity remains a schema binding, while
their LoRA execution sites and TP ownership stay distinct.

## 9. Clean lifecycle protocols

### 9.1 Load and publish

~~~mermaid
sequenceDiagram
    participant API
    participant C as AdapterCatalog
    participant R as ReplicaCoordinator
    participant W as All model ranks

    API->>C: load(name, source, expected revision)
    C->>C: reserve name; state=STAGING
    C->>R: prepare(AdapterKey)
    R->>W: resolve, validate schema, compile artifact, reserve host budget
    W-->>R: prepared or failure
    alt every rank prepared
        R->>W: publish prepared handle (no-fail local pointer flip)
        R-->>C: commit epoch
        C->>C: atomically publish name -> AdapterKey
        C-->>API: READY
    else any rank failed
        R->>W: abort and rollback
        C->>C: state=FAILED or remove reservation
        C-->>API: complete aggregated error
    end
~~~

Publication occurs only after every required rank has validated the same immutable
revision. Host caching and GPU activation are separate. All fallible work—source
resolution, allocation, validation, compilation, packing, and readiness checks—must
finish in `prepare`. `commit` is an infallible local pointer/table flip. A process
failure during commit fails the serving worker/failure domain; the protocol does not
claim durable database-style recovery from a half-dead distributed server.

This is a lightweight prepare/publish/abort protocol, not a reason to postpone
today's fixes. Current DP1 paths should immediately add local try/finally rollback
and aggregate every required TP/PP rank result. The fuller replica coordinator lands
when multi-rank dynamic publication needs one owner.

### 9.2 Acquire and execute

1. Atomically resolve a published name and acquire an `AdapterLease`.
2. Store the immutable `AdapterKey` in structured request/cache identity.
3. Scheduler asks `AdapterAdmissionPolicy` for one of:
   - resident and ready;
   - activatable with estimated bytes/load event;
   - blocked with a stable reason.
4. Forward preparation builds the local or execution-group adapter table and
   canonical assignment, transforming it alongside every token operation.
5. Residency returns a leased snapshot for that table.
6. Each provider refreshes its stable `DeviceAdapterAssignment` and obtains a
   reusable provider plan.
7. All layers read the same explicit execution context.
8. GPU completion fences asynchronously release slot/lane leases.
9. Request leases release in `finally` on success, cancellation, and every
   pre-dispatch failure path.

The scheduler should not call through model runner internals. It should depend on the
admission interface and retain the existing drainer as one fairness policy.

### 9.3 Unpublish, evict, and revoke

These operations must be distinct:

- `evict_device(key)`: retain publication and host artifact; reload on demand.
- `evict_host(key)`: retain publication/source metadata; resolve again on demand.
- `unpublish(name)`: stop new leases but allow a later explicit republish.
- `revoke(key)`: permanently reject implicit reload of that revision.
- `delete_artifact(key)`: remove cached compiled artifacts when no lease/snapshot
  remains.

Safe unload:

1. atomically unpublish the name;
2. mark the key draining;
3. wait for request leases and execution snapshots;
4. deactivate GPU slots with a generation bump;
5. evict host state as requested;
6. mark retired/revoked;
7. aggregate all-rank completion.

Unload has a bounded administrative wait. Timeout leaves the adapter unpublished
and `DRAINING`, without reusing its slots. A force request first cancels/aborts
affected requests, then still waits for their execution fences; it never bypasses a
live lease. If cleanup cannot establish safety, restart the worker instead of
recycling potentially in-use memory.

### 9.4 Replace in place

To update a public name:

1. stage a new immutable revision without touching the old binding;
2. commit it on every rank;
3. atomically switch name to the new key;
4. allow existing old-key requests and KV entries to finish;
5. retire the old key later.

This gives deterministic cache behavior and avoids a global stop-the-world update.

## 10. Distributed execution contracts

### 10.1 Expert IDs

Canonical routing keeps global expert identity. Localization happens exactly once:

~~~text
global top-k expert IDs
  -> dispatch / routing preparation
  -> global_to_local offset or expert_map
  -> local expert ID or -1
  -> local base and LoRA weights
~~~

Global route IDs never require global-sized resident LoRA buffers.

### 10.2 EP and A2A

Carry `token_adapter_ordinal` with every dispatched token. On the destination:

1. localize expert ID;
2. map batch ordinal through the destination residency snapshot;
3. build `virtual_expert = adapter_ordinal * E_local + local_expert`;
4. use the provider's stable device binding to reach physical slots or weight views.

This works even if physical slot numbers differ across ranks. A legacy or
slot-indexed provider may instead lower the ordinal to a local slot once after
dispatch; that physical representation remains provider-owned.

### 10.3 DP attention

DP-attention ranks may schedule different local batches, so equal local ordinals do
not imply equal adapters. Before hidden-state gather/repartition:

1. the forward preparer exchanges the small active `AdapterKey` lists over the
   DP-attention execution group;
2. every rank builds the same deduplicated group table;
3. admission ensures every destination can activate that union;
4. local ordinals remap to group ordinals;
5. the identical token transformation applies to hidden states and group ordinals;
6. each destination residency snapshot maps group ordinal to its own local slot.

Request-level IDs or unqualified local ordinals are insufficient after token
ownership changes. The metadata exchange is proportional to the active adapter
capacity, not the token count, and its cost must be measured.

### 10.4 TP and PP

Each site declares input/output ownership and collective semantics. The artifact
compiler slices local weights once. PP ranks compile only local sites, while global
SiteIds preserve distributed validation and artifact coverage.

## 11. Quantization and physical layouts

The stable MoE pipeline is semantic:

~~~text
route
  -> base GEMM1
  -> add gate/up LoRA
  -> activation
  -> base GEMM2
  -> add down LoRA
  -> router-weighted combine
~~~

Providers own:

- base weight format and scales;
- token/expert packing;
- top-k alignment;
- gate/up physical layout;
- quantize/requantize placement;
- workspace shapes;
- direct versus grouped strategy;
- one-shot versus split LoRA A/B;
- PDL, TMA, stream, and event details.

When a provider needs more than one representation of the same semantic stage,
use a typed stage value:

~~~text
StageValue
  semantic BF16 view when required by LoRA
  provider quantized view
  scale view
  ownership/lifetime
~~~

Do not introduce `StageValue` as an unused Phase-1b abstraction. Land it with the
first FP8/NVFP4 consumer that must keep BF16 LoRA and quantized base views alive
simultaneously. BF16 providers may use their simpler concrete workspace.

Layout conversion should be:

1. static at adapter compile/load when it is adapter-specific;
2. fused at a provider stage boundary when it depends on dynamic routing or removes a
   real memory pass;
3. never repeated inside every GEMM tile merely to repair an unclear contract.

CuTe DSL is useful for the second case, but correctness and model flexibility come
from the schema and provider boundary, not from choosing a kernel language.

### 11.1 Base-weight update and address stability

Providers may cache base weight handles and CUDA graphs capture physical addresses.
The supported update protocol is:

- pointer-preserving in-place copies may update contents and advance a
  `ModelWeightEpoch`;
- replacing a Parameter, its storage, packed provider object, or scale buffer must
  rebind provider weights and invalidate/recapture every graph that captured the old
  address before the new model epoch is published;
- a provider that cannot prove either path rejects online base-weight update with a
  clear requirement to recapture/restart.

Phase 1a records this as an explicit worklog gap rather than adding a temporary
provider-wide guard. Before graduation, the implementation must support the protocol
or validate it at the stable update boundary, covering both in-place copy and explicit
storage replacement with `recapture_cuda_graph`.

### 11.2 NVFP4 W4A4 provider contract and measured evidence

This subsection records the moonrise PR #89 handoff as evidence for Phase 1c. It is
a backlog and design constraint, not authorization to start kernel implementation.
The durable measurements and conclusions needed by this design are summarized here;
the original private handoff note is local provenance and is not required to review
or reproduce the design decision.

#### Measured B200 baseline

The measured implementation used the decomposed `experimental_sgl_trtllm` W4A4
topology: TRT-LLM NVFP4 base GEMMs, Triton LoRA, virtual experts, and one to four
resident adapters. With symmetric memory configured on a validated node, corrected
S1 throughput was:

| Batch size | W4A4 LoRA tok/s | Retained versus standalone W4A4 base |
|---:|---:|---:|
| 1 | 101.06 | 67.6% |
| 16 | 1171.42 | 77.4% |
| 32 | 2097.76 | 82.3% |

S4 retained 98.4% of S1 at BS16 and 97.8% at BS32. W4A4 was 7.6% slower than the
optimized Marlin W4A16 path at BS1, but 7.7% faster at BS16 and 20.5% faster at
BS32. At BS1 the decomposed W4A4 LoRA graph added about 3.21 ms over its standalone
base, versus about 1.66 ms for optimized Marlin; that roughly 1.55 ms excess LoRA
tax erased W4A4's faster base GEMM. A matched BS16 profile attributed only about
2.82 ms/step to routed FP4 BMM, while the complete LoRA topology added materially
more work across streams and collectives.

This establishes three design conclusions:

1. W4A4 is already the stronger throughput path at medium and large decode batches;
   it should not be replaced wholesale by W4A16 because of the BS1 result.
2. The main opportunity is the LoRA stage topology—bridge traffic, epilogue
   boundaries, launches, repacking, and graph selection—not the routed FP4 GEMM.
3. Adapter count is not the dominant remaining scaling problem: S4 stays close to
   S1 once the graph is active.

Base-only requests sent through the LoRA-enabled server also ran almost identically
to active-LoRA requests at BS16/24. The captured topology was therefore charging
all-base batches for the LoRA-capable graph. This is direct evidence for distinct
no-LoRA and LoRA-on graph families rather than only capture-time empty adapter IDs.

#### Required provider seams and optimization order

The semantic pipeline in this section remains unchanged, but an NVFP4 provider must
make two LoRA injection seams explicit:

~~~text
packed routing + hidden
  -> provider gate/up base work
  -> add/consume gate/up LoRA delta immediately before activation + FP4 quantization
  -> provider down base work
  -> add/consume down LoRA delta before finalize/combine/collective
~~~

The initial implementation may wrap an existing TRT-LLM base kernel or use a new
CuTe DSL kernel. The shared ABI does not depend on that choice. The provider owns FP4
packing, scales, workspace, and physical route representation; the shared execution
plane owns the two semantic injection points and their correctness requirements.

Work should proceed in this order after explicit implementation approval:

1. Add the provider injection contract and consumed `StageValue` views. Avoid a
   shared runner contract that requires a complete raw BF16 gate/up tensor or a
   complete BF16 activated bridge.
2. Fuse gate/up B expansion and addition into activation plus FP4 quantization so
   the full raw BF16 gate/up result is not materialized and reread when avoidable.
3. Fuse down-A shrink with activation production and feed the rank result directly
   into weighted top-k rank reduction, eliminating the full BF16
   `activation_lora_input` write/read bridge where possible.
4. Fuse weighted rank reduction plus shared down-B expansion into provider finalize
   and, when collective ownership permits, into the final TP collective. Preserve
   base rows, per-adapter mapping, routed scaling, EP-local ownership, and exactly one
   final reduction.
5. Preserve the base provider's packed top-k. Either emit canonical and packed views
   together at routing, or lower the canonical representation once inside the
   provider; do not add a standalone repack launch on every forward.
6. Reduce the remaining routed/shared and qkvr/`wo_ud` launch count through their own
   typed site executors. These are execution-plane dependencies, not fields in the
   NVFP4 provider ABI.

The existing shared-A/down-B algebraic factorization is negative evidence against an
isolated factorization PR. The isolated prototype passed S1-S4, mixed rows,
poisoned-buffer, eager, repeated graph replay, and TP8 transition correctness, but
changed throughput by only -0.7% to +1.1%. Its source was intentionally not retained.
Do not repeat it alone; revisit the algebra only as part of fusion that removes
bridge traffic or an epilogue/collective boundary.

#### Graph, collective, and device policy

- Decode needs separate no-LoRA and LoRA-on graph families. The no-LoRA family must
  replay the stock fused base-provider topology with zero LoRA launches; mixed and
  active batches use the LoRA-on family.
- Capture-aware skipping must be consistent across routed MoE, shared sink, qkvr,
  `wo_ud`, embeddings, and lm head. A LoRA-on graph still keeps fixed topology during
  capture and changes behavior through stable device metadata, not Python branching.
- `wo_ud` should write its base result directly to the symmetric all-reduce buffer,
  add the LoRA delta in place, and then launch the collective. Reducing base first
  and adding a rank-local delta afterward is incorrect.
- BCG remains the preferred constrained LoRA prefill experiment. Its capture-pool
  memory and host-memory envelope must be measured; the current path disables it for
  headroom rather than because the architecture requires eager prefill.
- Symmetric memory is a device/driver/collective capability policy, not a universal
  flag. It improved B200 W4A4 S4 from 934.85 to 1152.35 tok/s at BS16 and 1685.28 to
  2050.85 at BS32, but another B200 node previously hung during rendezvous. Select it
  per provider and validated device class, with a conservative fallback.

#### NVFP4 graduation envelope

Before the provider is eligible for automatic selection, validate S1-S4 with true
round-robin adapters, mixed base/adapter rows, base-only batches, poisoned
intermediates, repeated base-to-LoRA transitions, load/unload/eviction during
forwarding, eager/capture/replay parity, graph pointer stability, TP8, and ordinary
EP greater than one with global/local expert-ID translation. Validate B200 NVFP4 and
run H200 shared-execution/Marlin controls for common graph, collective, and
symmetric-memory changes. Advanced A2A, EPLB/elastic EP, and MTP/EAGLE/speculation
are outside this first NVFP4 graduation envelope and must fail clearly or remain on a
validated legacy path. Performance qualification uses input 1024/output 512 and
BS1/16/24/32, reports medians of three trials against both standalone W4A4 base and
same-server base requests, and profiles the whole batch. Attribute critical-path
wall time and stream dependencies rather than treating summed kernel duration as the
latency result; retain Perfetto-compatible traces.

## 12. Layer boundary

The long-term layer API should express stage-scoped injection sites, not copy each
base layer’s entire forward:

~~~text
LoRASiteExecutor.execute(
  site_id,
  injection_stage,
  StageOperands,
  execution_context,
) -> StageResult

MoeLoRAExecutor
  gate_up_delta(hidden_states, routing, context)
      -> gate_up_delta + activation_lora_input
  down_delta(activation_lora_input, routing, context)
      -> pre-combine delta
~~~

The generic signature is a typed protocol, not a demand that all sites have one
tensor in and one tensor out. MoE has two coupled seams and a side value carried
between them; lm head has an ordered pass plan; absorbed MLA has forward-mode
execution variants. Their concrete executors expose those facts.

Practical migration:

1. Keep wrappers initially.
2. Give each wrapper a compiled `SiteHandle`.
3. Replace reads of global `backend.batch_info` with the execution context.
4. Move segmentation/packing into provider plans.
5. Where possible, expose narrow injection seams in base layers.
6. Remove wrapper copies only after parity for that layer family.

MoE needs the two explicit internal seams above. Ordinary dense linear needs one
delta-add seam. Embedding, lm head, and absorbed MLA corrections remain typed special
sites backed by the same assignment/residency contracts.

## 13. CUDA-graph execution: full decode and breakable prefill

Use phase-specific graph backends:

- **decode:** the existing full CUDA graph remains primary;
- **prefill/extend on Hopper and Blackwell:** breakable CUDA graph (BCG) is the
  primary `sgl_lora` target;
- **`tc_piecewise`:** defer LoRA integration and retain it as a platform or
  compatibility fallback;
- **eager:** remain the correctness fallback for an unsupported graph envelope.

BCG and `tc_piecewise` are independent backends. BCG explicitly captures CUDA-graph
segments around eager break points without `torch.compile`; `tc_piecewise` traces and
FX-splits through `torch.compile`. “Piecewise” should therefore not be used as an
ambiguous umbrella name in the LoRA roadmap.

Graph integration is a named deliverable, not an incidental provider detail:

1. Capture separate base/no-LoRA and LoRA-on graph families so base replay launches
   zero LoRA kernels.
2. Every LoRA-on family reads stable `DeviceAdapterAssignment` and provider
   workspace addresses refreshed before replay.
3. Capture with realistic nonzero cyclic adapter mappings, not only
   `lora_ids=None`.
4. Decode, extend/prefill, target-verify, and each concurrent graph lane declare
   their assignment token domain and capacity.
5. BCG keeps graph-safe LoRA kernels inside captured segments. Use
   `eager_on_graph`/an explicit break only around a provider stage that cannot yet be
   captured **and** still receives capture-stable tensor references, returns a
   fixed-shape/fixed-address bridge value, and reads changing adapter state only from
   refreshed stable device buffers. It must not branch on live Python `lora_ids` or
   allocate adapter-dependent output shapes. Otherwise the whole forward falls back
   to eager; a graph break does not repair an invalid replay ABI.
6. The current BCG prefill envelope captures the transformer body while embedding
   preparation and lm-head/logits processing may remain eager. The captured body and
   every eager prefix/tail consume the same `LoRAExecutionContext`, assignment views,
   leased `ResidencySnapshot`, and graph-lane completion contract.
7. A later `tc_piecewise` integration registers the required custom/split ops with
   correct fake, mutation, alias, and carried-value lifetime contracts. Those
   torch-compile contracts are not a prerequisite for the first BCG path.
8. EXTEND replay refreshes LoRA assignment after its final padding/gather/pruning
   decisions and before graph launch.
9. Clear padded request/token tails so replay cannot observe the prior batch's adapter
   mapping.
10. Base-weight address replacement follows §11.1 and recaptures before publication.
11. Remove BCG's LoRA compatibility guard only after capture-pool GPU memory, host
    memory, segment/break count, and prefill latency/throughput are measured against
    eager and base-only BCG.

Current prefill capture can construct a dummy batch with `lora_ids=None` while
replay carries real adapters. Until the stable LoRA metadata refresh contract above
is implemented and tested, keep the current LoRA auto-disable for both BCG and
`tc_piecewise`, and reject an explicitly forced unsupported configuration rather than
silently capture an incomplete graph. Selecting BCG is the target design, not a claim
that the reviewed branch already supports LoRA prefill replay.

## 14. Migration aligned with the current plan

### Phase 0/1a containment and independent correctness fixes

Keep the BF16 vertical slice narrow, but do not preserve known correctness bugs:

1. Track gate/up/down pairing as a known gap and implement it through typed
   rank-local binding in Phase 1b-BF16; do not add a one-off Phase 1a adapter
   validator based on ambiguous suffixes.
2. Treat current `LoRAInfo` and physical slot metadata as a named legacy bridge,
   not the new execution ABI.
3. Treat the current DeepGEMM workspace/weight structs as concrete BF16 provider
   types, not a finalized all-quant interface.
4. Keep `--lora-execution-engine` authoritative. The Phase-1a spelling
   `--moe-runner-backend=sgl_lora` may remain as an input alias, but normalize it
   away and resolve the actual base MoE provider independently. Selecting
   `sgl_lora` implies virtual-expert semantics internally; users should not need a
   second semantic flag.
5. Keep unsupported prefill graphs and base-weight replacement out of the advertised
   matrix and record them in the worklog. Add boundary validation near graduation if
   their refresh/recapture contracts have not landed.
6. Compare Phase-1a performance against legacy with the experimental master env and
   its gated optimizations disabled on both sides. The hard graduation comparison
   uses the best supported legacy path only after neutral routing knobs are re-homed.

Land these control-plane bugfixes as independent small PRs rather than waiting for
Phase 3:

- atomic catalog resolve plus lease increment versus unload;
- guaranteed lease release in `finally` on every pre-dispatch failure;
- implicit reload preserving identity only after immutable revision verification;
- local staged-load rollback and complete required-rank result aggregation;
- exact multimodal language-site filtering until schema compilation replaces it.

The architectural sections above define capabilities; the reviewed landing order is
deliberately kernel-first and is the following.

### Landing step 1: neutral BF16 controls and the current contract

1. Preserve the implemented standard BF16 contract: gate-first contiguous W13,
   ordinary gated SwiGLU, canonical gate-followed-by-up LoRA delta, BF16 W2 input,
   and the pair-domain BF16 activation bridge for down-A.
2. Establish provider-matched `N0`, materialized serial `C0`, and materialized
   two-stream `C1` controls on H200 and GB300.
3. Add neutral stock/legacy and experimental-TRTLLM whole-M0 brackets at the same
   output boundary. Do not make a cross-backend claim from the existing internal-SGL
   comparisons.
4. Keep every implementation behind the explicit experimental selector; landing an
   incremental implementation commit does not require production automatic selection.

### Landing step 2: BF16 Qwen C2/C3 and shared-outer kernels

1. Treat raw indexed/direct, true segmented SGMV, grouped GEMM, qualifying BMM,
   token-owned reduction, and fused-consumer ownership as algorithms. Triton, CuTe
   DSL, CUDA, and provider libraries are independent implementation choices; cuTile
   is optional and capability-gated.
2. Make prepared routing conditional on the selected consumer. A kernel may consume
   canonical IDs directly, segmented offsets, an aligned view, producer-packed
   metadata, or provider-private metadata. Do not build unused alignment.
3. Validate shared-outer LoRA early: compare once-per-token/adapter shared gate/up-A
   against pair repetition, and shared down-B weighted rank reduction followed by one
   B against repeated pair expansion. This is not model shared-expert execution.
4. Build `C2` first: gate/up-A, base W13, fused gate/up-B + base add + activation
   (+ provider quant where applicable) + down-A, base W2, then down-B/finalize.
5. Build `C3` second by overlapping only gate/up-A with base preparation/W13. Keep
   `C0/C1` as controls, `C5` as a diagnostic/fallback, and leave `C4` down overlap
   until last because prior evidence was neutral/unsafe.
6. Use ranks 32/64/128 for initial model-scale benchmark/schedule work. R128 remains
   benchmark-only until a production-supported A path graduates.
7. Advance candidates from K0 through their exact requested-preparation O0 boundary
   to matched M0 plus a structural trace; no isolated win skips a scope.

### Landing step 3: early model guardrails and BF16 graduation

1. Before hardening the Qwen fused ABI, run cheap correctness/local-shape guardrails
   for large H, latent H, high top-k, odd I, non-gated ReLU2, partial slice targets,
   shared-outer factors, and provider padding across the retained model presets.
2. Add model-scale rank 8/16 performance at graduation, a supported production R128
   path, wider adapter/routing coverage, and device-specific policies.
3. Add distinct LoRA-off/LoRA-on decode graph families and leased graph arenas only
   for actual concurrent lanes. Validate replay, recapture, overlap streams, EXTEND,
   and base-weight update.
4. Make BCG the primary Hopper/Blackwell prefill target and implement its stable
   metadata refresh, tail clearing, and selective eager-break contract from §13.
   Keep `tc_piecewise` behind its separate compatibility/custom-op contract.
5. A one-GPU TP/EP/MoE-DP case is only a `<resolved topology> local-shape proxy`.
   It does not validate dispatch, A2A, collectives, imbalance, communication overlap,
   or distributed graph safety.
6. Run controlled single-rank E0 with `sglang.benchmark.one_batch_server` after local
   BF16 correctness and M0 gates pass.

### Landing step 4: minimal execution and provider-contract bridge

Add these objects only with the first quantized or distributed consumer that needs
them; do not pull the full Phase-3 adapter control plane forward:

1. `BatchAdapterTable`, `LoRABatchAssignment`, `AssignmentView`, `MoeGeometry`,
   `SlotRef`, `ResidencySnapshot`, `DeviceAdapterAssignment`,
   `LoRAExecutionContext`, and `ExecutionCompletionTracker`.
2. A bridge from `ForwardBatch`, the current memory pool, and current backend
   metadata, with in-place stable device mapping refresh, snapshot leases, completion
   fencing, and activation-readiness waits on every consuming stream.
3. Reusable provider executable/opaque workspace ownership separate from per-forward
   `ProviderInvocation`.
4. Live rank-local `LoRASiteCandidate` compilation, the tri-state model extension,
   centralized adapter-format parsing, and exact typed routed-MoE site/slice binding.
5. Separate `ActivationSpec`, logical projection slices/target masks, and the
   provider-private physical output contract. The latter declares W2 input
   dtype/layout, scale/swizzle tensors, down-A's pre-quant source view, invalid rows,
   destination dtype, and ownership. Do not duplicate current BF16 containers until
   the first alternative provider/fused consumer needs these distinctions.
6. Keep `stacked_multiply`, gate-first physical packing, max-rank component strides,
   and one aligned route plan outside the semantic site contract.

Do not add unused `StageValue`, exhaustive provider capabilities, lm-head pass types,
or speculative-only fields. Add each with its first consumer.

### Landing step 5: independent quantized providers

1. Add FP8 block-scale through DeepGEMM and introduce `StageValue` only when both
   BF16 LoRA and quantized base views are first consumed.
2. Add NVFP4 W4A4 with the explicit pre-activation and pre-finalize seams in §11.2.
   CuTe DSL is a primary kernel technology, but the contract may first wrap a
   validated TRT-LLM base implementation.
3. Add or adapt Marlin W4A16.
4. Keep every provider's packing, scales, workspace, physical route view, and kernel
   index space private, and graduate each quant independently.

### Landing step 6: real distributed and model-shared execution

1. Use local-sized expert weights and one global-to-local expert translation.
2. Carry adapter ordinals with ordinary EP/A2A tokens and add execution-group table
   agreement before DP-attention redistribution.
3. Validate real TP/EP/MoE-DP dispatch, combine, collectives, imbalance,
   communication overlap, and graph behavior before calling any proxy result D0.
4. Add conventional/fused/sink model shared experts here as a later, separate
   model/provider feature; do not conflate them with shared-outer LoRA from step 2.
5. Add each advanced A2A backend only after its payload and graph contract is
   explicit, then add target-verify/MTP assignment views and target/draft policy.

### Landing step 7: full server and multi-node graduation

Graduate every supported path through unprofiled controlled E0, real adapter
lifecycle correctness, graph transitions, supported A2A, and two-node MNNVL. Automatic
selection begins only after correctness, memory, graph, performance, D0, and E0 gates
pass. The broader adapter identity/loading/residency/scheduling refactor remains Phase
3 rather than being pulled into the kernel program.

### Phase 2: dense and special sites

1. Split the monolithic legacy monkeypatch installer, or keep it wholly enabled until
   all coupled families migrate; do not claim per-family removal before that split.
2. Extend the live-module site compiler and sparse tri-state `ModelLoRAExtension`
   across dense and special classes; adapt old hooks without requiring a mass model
   migration.
3. Add ordinary column/row/replicated sites.
4. Replace string normalizers family by family with structured weight bindings and
   declarative fusion/slice rules.
5. Add `LoRASliceSpec`, `CanonicalSliceWeights`, and `LinearDeltaPlan`; migrate QKV
   and merged sites without retaining prefix-only coverage or `stacked_multiply` as
   semantic ABI.
6. Replace semantic `run_qkv_lora`/`run_gate_up_lora` APIs with one sliced-linear
   provider operation while retaining single-slice, equal-two, and generic N-slice
   kernels as provider-selected fast paths.
7. Make every dense provider honor the plan's additive-destination dtype. Migrate DSA
   `indexer.weights_proj` with an FP32 base destination and add zero/nonzero-delta
   parity tests; do not rely on the legacy BF16-then-upcast path.
8. Add Mamba/GDN and MLA `WeightBindingGroup` execution variants.
9. Add embeddings and `LmHeadPassPlan` assignment views.
10. Replace global backend batch state provider by provider.
11. Retire legacy `get_hidden_dim`/`get_stacked_multiply`/target hints only after
   schema comparison tests cover their models.

### Phase 3: replace the control-plane producer

1. Add the prepare/publish/abort replica coordinator.
2. Replace the interim catalog with immutable alias-independent `AdapterKey`
   identity and structured cache keys.
3. Make centralized adapter-format readers and structured raw-weight records the
   native artifact source for the schema compiler.
4. Separate host artifact and device residency budgets.
5. Add explicit evict/unpublish/revoke/delete operations.
6. Add byte-aware residency admission and prefetch.
7. Produce the same canonical assignment/context contracts natively.

The kernel ABI should not change when Phase 3 lands.

## 15. Validation matrix

### Lifecycle

- injected failure at every fallible prepare step with complete local rollback;
- one-rank failure during distributed prepare, followed by group-wide abort;
- process failure during the no-fail publish step, followed by fail-stop recovery;
- acquire racing unload, with atomic resolve-plus-lease behavior;
- exception after acquire but before scheduler dispatch, with `finally` release;
- replace the same alias with a new immutable revision while old requests run;
- implicit reload with verified-same and changed immutable revisions;
- radix/KV-cache reuse across device eviction for the same immutable key, and forced
  miss after alias replacement or model-weight-epoch publication;
- explicit revoke versus evict/implicit reload;
- host and device budget pressure with pinned artifacts;
- stale slot, slot recycling, generation invalidation, and mapping-epoch refresh;
- unload timeout remaining in `DRAINING`, plus forced cancellation and fence drain;
- in-place base-weight copy and storage-replacing update with required recapture.

### Assignment and graphs

- base-only, one adapter, mixed base/adapter, maximum adapters;
- request batches with different sequence lengths;
- chunked prefill, decode, extend, target verify, and future MTP;
- padding/gather/reordering after initial request creation;
- request- and token-domain `AssignmentView` construction for lm-head passes;
- two DP ranks assigning ordinal one to different adapters, then agreeing on one
  execution-group table before redistribution;
- adapter-key and ordinal permutations across replays without graph recapture;
- slot recycling with unchanged graph executable but refreshed generation buffers;
- realistic LoRA-on capture mappings rather than only `lora_ids=None`;
- decode and extend/prefill graph capture, replay, recapture, and rejection of an
  unsupported prefill path;
- BCG segment replay, selective eager breaks, padded-tail clearing, and captured
  LoRA-kernel parity;
- BCG capture-pool/host memory, segment count, hidden eager-fallback telemetry, and
  prefill latency/throughput versus eager and base-only BCG;
- deferred `tc_piecewise` custom-op fake, mutation, alias, and carried-value lifetime
  contracts when that compatibility path lands;
- concurrent graph lanes, lane reuse only after completion fencing, and stale host
  handle rejection by lane epoch;
- overlap loading and auxiliary-stream completion fencing before slot reuse;
- concurrent activation with every consumer stream waiting on readiness before its
  first copy, pack, or weight read;
- failure after partial multi-stream launch, with leases held until every declared
  provider stream is complete.

### Models

- standard Llama/Qwen/Gemma dense;
- QKV with TP greater than KV head count;
- partial Q/K/V and gate/up coverage;
- every single-slice and multi-slice presence mask allowed by its schema, including
  missing first/middle/last slices rather than only missing suffixes;
- independently ranked/scaled slices and shared-A fused artifacts;
- canonical absence lowered to dense-zero and compact-active provider packs with
  parity against an unfused reference;
- generic N-slice execution versus the equal-two fast path, including unequal Q/K/V
  and MLA boundaries;
- a dense `gate_up_proj`/`down_proj` adapter that must not satisfy or fail the routed
  MoE target-pair check;
- DeepSeek MLA one-logical-weight-to-multiple-execution-site binding and DSA indexer
  target qualification;
- DSA `indexer.weights_proj` zero-delta parity with the FP32 no-LoRA base result and
  nonzero-delta parity with an FP32-accumulated reference;
- gated and non-gated MoE;
- GPT-OSS interleaved clamped SwiGLU;
- shared-outer and per-expert adapters in the same server;
- model shared expert plus routed experts;
- Nemotron-H Mamba/latent/non-gated layers;
- Qwen3.5 GDN and mixed layer types;
- multimodal language-only filtering;
- a hybrid model overriding one special site while common sites fall through to the
  central live-module compiler;
- legacy `get_hidden_dim`/`get_stacked_multiply` raising `NotImplementedError` for an
  unknown site and correctly falling through to central handling;
- PEFT split/fused, w1/w2/w3, expert-indexed, shared-outer, and tensor-API key dialects
  resolving to identical canonical bindings;
- exact qualified-path resolution, model alias override, ambiguous suffix rejection,
  and actionable unmatched-weight diagnostics;
- adapter-logical dimensions disagreeing with provider-padded live storage, with both
  represented separately and provider padding excluded from canonical tensor shape;
- a mixed dense-plus-MoE adapter in Phase 1b, with every weight claimed exactly once
  across the new MoE compiler and legacy dense bridge;
- tied embedding/lm-head, ordered multi-pass assignments, and pruned logprob paths;
- nonuniform per-layer dimensions;
- PP-local site materialization.

### Distributed and quantized

- TP, ordinary EP, expert maps/EPLB, and each A2A provider;
- EP/A2A with adapter ordinals carried alongside tokens and one global-to-local
  expert translation;
- DP attention with execution-group table agreement before token redistribution;
- BF16 Hopper and Blackwell;
- FP8 block scale Hopper and Blackwell;
- NVFP4 W4A4 Blackwell;
- Marlin W4A16;
- provider-local ordinal and slot kernel index spaces producing identical results;
- no-LoRA graph with zero LoRA launches.

### Selection, compatibility, and performance

- `sgl_lora` execution-engine selection independent of the base MoE provider after
  Phase-1a shorthand normalization;
- selecting `sgl_lora` implies virtual experts without a second user-facing flag;
- legacy remains the default and does not import, initialize, or launch the new path;
- startup and dynamic load run the same schema and actual-bound-site validation;
- Phase-1a comparison with the legacy master env and its gated optimizations disabled
  on both sides;
- graduation comparison against the best supported legacy path after neutral routing
  optimizations are re-homed;
- memory, graph-launch, throughput, and latency gates for every provider/device pair.

## 16. Immediate decisions

1. Keep the kernel-first refactor order.
2. Use **Adapter Control Plane** as the high-level name.
3. Use **ModelLoRASchema** as the model-flexibility boundary.
4. Make adapter ordinals canonical in planning and semantics; let each provider's
   `DeviceAdapterAssignment` declare whether its kernels consume ordinals or slots.
5. Keep stable device metadata addresses and refresh mappings in place. Adapter keys,
   ordinal permutations, slot generations, and mapping epochs are data, not graph keys.
6. Put `SlotRef(slot, generation)` in residency snapshots and provider device
   bindings, never in the canonical request identity.
7. Agree on one execution-group adapter table before DP-attention redistribution;
   rank-local ordinals alone are not portable across ranks.
8. Hold slot leases until completion events cover every provider stream, and give
   each concurrently in-flight forward one fenced graph lane.
9. Keep virtual experts as the only new MoE semantic model; selecting the new engine
   implies them internally.
10. Compile model quirks, checkpoint layouts, logical-weight binding groups, and site
    requirements once, before serving.
11. Carry adapter assignment through every token transform and expose ordered views
    where one logical operation performs multiple passes.
12. Replace mutable singleton batch state with a per-forward execution context.
13. Keep all fallible distributed load work in prepare; publish is a no-fail pointer
    flip, and process failure during publish is fail-stop.
14. Land acquire/unload, lease-finally, reload-identity, staged-rollback, and
    multimodal filtering as independent correctness changes when scheduled. Implement
    routed-target validation with the Phase 1b typed binding schema, not as Phase 1a
    review noise.
15. Keep full graphs for decode and make BCG the primary Hopper/Blackwell prefill
    target, with explicit metadata refresh, tail clearing, completion, and recapture
    contracts. Defer `tc_piecewise` LoRA support and reject incomplete configurations.
16. Fix or replace the currently unused `should_apply_lora` hooks before claiming
    comprehensive multimodal support.
17. Treat the caller-owned additive destination dtype as the final LoRA-B dtype;
    keep LoRA-A split-K accumulation precision independent and make DSA's FP32 site
    requirement explicit in its `LinearDeltaPlan`.
18. Do not use `supported_lora_modules` as an allow-list; validate a compiled
    rank-local schema plus explicit end-to-end tests.
19. Do not introduce one general graph IR. Use typed `WeightBindingGroup`,
    `AssignmentView`, and stage-specific executor contracts only where model behavior
    requires them.
20. Give a sparse `ModelLoRAExtension` first refusal per site/key, with explicit
    `NOT_HANDLED` fallback to centralized structural logic; never use `hasattr` as an
    all-or-nothing ownership transfer.
21. Parse external adapter dialects centrally, resolve structured keys to exact typed
    sites, and keep model aliases separate from provider packing. Longest-substring
    matching is legacy compatibility only.
22. Treat fused projection components as explicit logical slices. Canonical absence
    may occur at any schema-optional position; providers may zero-fill, compact, or
    specialize it, but `stacked_multiply`, prefix-only coverage, and semantic
    `run_qkv_lora`/`run_gate_up_lora` methods are legacy details.

## 17. Source map

Primary current-code anchors:

- `python/sglang/srt/server_args.py` — startup parsing and compatibility
- `python/sglang/srt/lora/lora_registry.py` — logical registry and request counters
- `python/sglang/srt/managers/tokenizer_manager.py` — request resolution and release
- `python/sglang/srt/managers/tokenizer_control_mixin.py` — dynamic update protocol
- `python/sglang/srt/managers/scheduler.py` — adapter-aware admission
- `python/sglang/srt/lora/lora_drainer.py` — fairness/draining
- `python/sglang/srt/lora/lora_overlap_loader.py` — asynchronous activation
- `python/sglang/srt/lora/lora_manager.py` — current worker orchestrator
- `python/sglang/srt/lora/lora.py` — artifact loading and normalization
- `python/sglang/srt/lora/mem_pool.py` — GPU residency and physical shapes
- `python/sglang/srt/lora/utils.py` — targets, dimensions, and batch metadata
- `python/sglang/srt/lora/backend/` — provider segmentation and graph buffers
- `python/sglang/srt/lora/layers.py` — execution wrappers and MoE boundary
- `python/sglang/srt/model_executor/forward_batch_info.py` — forward handoff
- `python/sglang/srt/model_executor/runner/` — CUDA-graph capture/replay
- `python/sglang/srt/models/` — model declarations and special requirements

This document is a design snapshot, not a claim that the current branch implements
the proposed control plane. The current branch should continue landing the execution
refactor commit by commit behind the `sgl_lora` selector.
