# Third-party benchmark/architecture review disposition

Source review:
`THIRD_PARTY_REVIEW_FEEDBACK.md`
(Claude, 2026-07-22, branch snapshot `4ffaee0b41`). The feedback is copied
verbatim into this bundle. This disposition includes later campaign and
production-integration work; the final rebased source identity is recorded in
`SOURCE_STATE.json` after the upstream rebase.

This ledger maps every actionable review group to the implementation or evidence
that addresses it.  A benchmark result is not marked as production support merely
because the corresponding testbed exists.

## Critical and benchmark-validity findings

| Finding | Disposition | Evidence / implementation |
|---|---|---|
| C1 fused shared-expert IDs enter the LoRA factor domain | Fixed before broader model runs. IDs outside the explicit factor-expert interval become sentinels, and static physical-to-routed maps prevent shared/sink slots from aliasing another adapter's factors. Production scope is intentionally limited to validated Standard-dispatcher layouts; per-rank physical shared layouts with EP>1 and advanced all-to-all dispatchers remain rejected because they can remap physical IDs before LoRA. | `1844c4d634`; `shared_experts/`; `shared_outer_gate_production/`. |
| V1 vacuous M0 and B-family tolerances | Fixed. All active candidates are checked on the base-subtracted LoRA signal with signal-relative gates; direct and down targets use the same rule. | `ea968ed204`, `930d1e8aac`; corrected algorithm-family and rank-guardrail summaries. |
| V2 deterministic lattice mislabeled IID | Fixed. The old route is labeled lattice; true-IID seeds and an explicit skewed draw are separate route policies. | `8c38a7c0d4`; `algorithm_families/`; `h200|gb300/.../p0_routes/`. |
| V3 O0 double-charged shared route plan | Corrected. O0 is operator-isolated evidence only; production decisions use marginal shared preparation and M0/E0. | `4b16d215b9`, `32b57ba014`; final audit text. |
| V4 quiet-host and hot-L2 bias | Measured explicitly. Loaded-host eager and forced-cold M0 remain separate evidence modes; unprofiled isolated runs stay authoritative. | `6ec45fc26d`, `8e6dd04477`; `loaded_host/`, `c2_down_finalize/`. |
| V5 graph dispersion and fixed execution order | Fixed for promoted comparisons with counterbalanced forward/reverse processes and p20/p50/p80 reporting. | `6ec45fc26d`, `050a8d6661`, `8e4114e2c0`; model E0 summary. |

## Evidence-to-conclusion corrections

| Review correction | Disposition |
|---|---|
| R128 M0 was benchmark-only before a legal production A schedule | The rank-tiled A path now supports R128; old rows remain labeled historical/conditional and new production rows are separate. See `88de33bf49`, `e08a9b1964`, and `production_planner/`. |
| GB300 T2048 eager overlap sign flip required tracing and execution-mode policy | Traced and retained; execution mode and forward phase are immutable planner keys, not a token-threshold alias. See `566c82d8cb`, `47dc9bf7b8`, and device traces. |
| Missing T64/T128 R128 cells | Added before planner promotion. See `e08a9b1964` and the archived device trees. |
| Capacity-one versus capacity-eight confound | Both native and large virtual-expert routes are retained and presented; capacity changes the route implementation and is never treated as a free proxy. See `c08d0db80e`, `4b16d215b9`, and the device archive. |
| Serial prefill LoRA tax underclaimed | Promoted to a primary motivation for C2/C3 and quant consumer fusion in the final audit. |
| Rows-per-hit-group fixed at one anchor | The algorithm-family campaign adds true-IID/skewed and denser local-expert regimes; distributed evidence labels local topology separately. See `algorithm_families/` and `distributed_d0/`. |

## Silent-wrong and delayed-fatal paths

| Finding | Disposition | Commit |
|---|---|---|
| Non-unit routed scaling omitted from down delta | Fixed by applying the factor exactly once to the down-LoRA top-k weights; non-unit tests cover BF16/FP32 destinations. | `930d1e8aac` |
| Unsupported activation/layout semantics silently ignored | Replaced by typed activation/provider contracts; SwiGLU and non-gated ReLU2 have distinct kernels, with padded/odd dimensions and partial-slice oracle coverage. | `be5aee63eb`, `47dc9bf7b8` |
| DP-attention assignment token domain mismatch | Rejected at the runner boundary and exercised in real distributed topology tests. | `930d1e8aac`, `04b65d2af0` |
| Unbounded provider workspace | Preflighted before provider allocation with graph-aware admission. | `e74fe23843` |
| Rank greater than 64 failed mid-serving | Replaced by legal rank tiling through 128, including R8 physical-16 masking. | `88de33bf49`, `707832fde0` |
| Quantized provider failed only on first active adapter | Quant provider resolution is explicit and independently tested before production dispatch. Marlin W4A16 has a resident attach path with local expert IDs. Dynamic-activation FP8 requires the resident weight-scale ABI and static-activation FP8 is rejected. Native CuTe DSL NVFP4 W4A4 is validated through the production-plan testbed, but compatible checkpoint/server attachment is not graduated. | `a8e698cdf4`; `quant_providers/`; `quantized_production_plan/`. |
| Broad exception masked align failures | Narrow fallback/error semantics are covered by route counter-candidate tests. | `32b57ba014` |
| Side-stream route prewarm and unbounded event lifetime | Routing/allocation ownership moved to the main stream and events are tied to CUDA-graph resources. | `23154589c0` |
| PDL producer could launch an unwaiting consumer | Fixed by pairing producer signaling only with direct consumers that execute `gdc_wait`; generic consumers retain stream ordering. | production quant follow-up |

## Plan and sequencing findings

| Review item | Disposition |
|---|---|
| Neutral TRTLLM/legacy model baseline before claiming a winner | Completed with `one_batch_server`. SGL is explicitly **not** universally faster. The **historical pre-planner** path trailed experimental TRTLLM by 12.56%–14.71%. In the final matched bracket, default C2 trails by **9.43% / 9.44% / 10.39%** and opt-in C3 trails by **5.12% / 5.54% / 10.48%** at BS1/16/32; base traffic was near parity in the original bracket. See `b6e9a6ace7`, `e2e/`, and `production_planner/`. |
| Build down-B/finalize before hardening gate consumer | Done: fused down finalization (`aa91c441ea`) preceded production planner promotion; fused gate consumer remains an independently selectable boundary. |
| Add one-shot A+B and FP32 accumulation arms | Implemented as algorithm-family competitors and retained as negative/conditional evidence rather than forced production paths. See `050a8d6661` and `algorithm_families/`. |
| Add legacy fused-align/prefill-reuse controls and cross-layer route reuse | Legacy controls were added. Production consumers reuse route metadata **within one MoE layer**. A route plan is not memoized across model layers; cross-layer reuse remains an explicitly unpromoted optimization. See `32b57ba014`, `47dc9bf7b8`, and `route_countercandidates/`. |
| Run all-base tax and separate no-LoRA/LoRA graphs | Completed in local, graph, and model-server scopes. See `84f8f5be7d`, `b6e9a6ace7`, `lifecycle/`. |
| Add concrete shared-outer cells | Completed across shared gate-A and shared down-B factorizations. The guarded gate-A selector is production-integrated only in its measured static envelope. Host selector coverage passed 9 tests; the integrated shared-LoRA aggregate passed 56 tests plus 10 subtests on each of H200 and GB300. See `83452e7c81`, `shared_outer/`, and `shared_outer_gate_production/`. |
| Add mixed-rank `(R,R_max)` cells and static-policy regret | Completed for R32/R128 and R64/R128 on both GPUs, eager/graph, active/mixed/base. Packed rank wins materially; production residency/multi-bucket ownership is kept distinct from the measured kernel policy. See `8e4114e2c0`, `mixed_rank_policy/`. |
| Pull R8/R16 guardrails forward | Completed, including logical R8 with physical tensor-core R16 masking. See `707832fde0`, `acdfd9ce18`, `rank8_16_guardrail/`. |
| Add a production per-site seam | Completed through the immutable host execution/provider plan; benchmark overrides no longer define the promoted path. See `47dc9bf7b8`. |
| Measure static-policy regret | Completed for mixed-rank and algorithm-family choices; captured graph topology remains a static key. See `mixed_rank_policy/` and `algorithm_families/`. |
| Add real PDL-off control | Completed for isolated and chained eager/graph modes on both devices. PDL remains opt-in evidence, not a global architecture toggle. See `02579672d2`, `pdl_control/`. |
| Resolve TMA compatibility | The Blackwell CuTe Tensor Core/TMA lane is recorded independently in `cutedsl_tensorcore/`. It is evidence-only: the static-route candidate wins one large-rank prefill boundary but fails route-mutation CUDA-graph replay, and no serving code imports it. A dynamic GPU route/descriptor design is required before reconsidering dispatch. |

## Late production graduation and explicit boundaries

| Area | Final disposition |
|---|---|
| BF16 server gap | Final default C2 remains 9.43% / 9.44% / 10.39% behind the matched experimental TRTLLM control at BS1/16/32. Opt-in C3 narrows BS1/16 to 5.12% / 5.54% but remains 10.48% behind at BS32. The 12.56%–14.71% range is historical pre-planner evidence only. |
| Physical shared experts | Shared/sink IDs are isolated from the routed LoRA factor domain. Validated production scope is the Standard dispatcher and its safe physical-ID layouts. Per-rank shared layouts with EP>1 and advanced all-to-all remapping are rejected/unpromoted; the proxy benchmark is not a serving-support claim. |
| Shared-outer | The gate-A token-dedup selector is integrated behind a conservative static policy. It has H200 and GB300 eager/graph coverage, mixed/base segments, PDL composition, physical shared-ID-map composition, and launch-structure evidence. Unmeasured or noisy cells use generic shrink. |
| FP8 | Dynamic-activation FP8 may attach only when resident scale tensors satisfy the expected ABI (including packed UE8M0 where required). Static-activation FP8 is explicitly rejected because resident input-scale support is not implemented. |
| Marlin W4A16 | Resident Marlin weights are attachable. The output is zero-initialized before split-K atomic accumulation, per-invocation lock workspaces prevent cross-graph aliasing, and the terminal regression smoke passed on H200 and GB300 at `f2f406e056`. Global-ID EP layouts remain rejected until an explicit provider mapping contract exists. |
| NVFP4 W4A4 | The native CuTe DSL provider passes the synthetic production C0 testbed on GB300. Current resident checkpoint/provider layouts are not assumed compatible, so checkpoint/server attachment remains a gap. |
| CuTe Tensor Core lane | Benchmark-only upper-bound evidence. Static route descriptors are semantically stale when graph replay mutates routing, so the candidate is not production eligible. |

## Reproducibility and document hygiene

| Finding | Disposition |
|---|---|
| TTL-only artifacts | All promoted H200/GB300 results, Nsight reports, raw logs, manifests, and SHA-256 files are archived below this directory. |
| Docs/tree drift and stale changelog | Corrected in the final plan, worklog, architecture audit, and lifecycle design update. |
| Dead `/tmp` and amended-commit pointers | Removed from normative instructions; rejected experiments remain historical evidence only. |
| Stale `design.md` | Marked superseded by the final lifecycle/orchestration design. |
| Missing software/device provenance | Current drivers record source hashes, CUDA/Torch/Triton/provider versions, device identity, and resolved configs; cross-image percentages are treated as operational, not architectural. Some early retained artifacts predate the complete schema, so their lane manifests/source hashes—not a retroactively inferred environment—are authoritative. |
| U0/WS1/symbol terminology drift | Unified in the final audit glossary and plan. |
| Poisoning/sustained replay/touched-byte protocol | Implemented where the kernel owns an inactive/sentinel/padded domain; unsupported provider-private poisoning remains explicitly scoped rather than claimed. |
| Benchmark monkeypatch/source drift risk | Production planner tests call the production runner and provider seams; benchmark-only historical results remain labeled as such. |

The terminal clean-snapshot validation is archived in `final_validation/`: H200
passed 128 branch-relevant GPU tests with 6 expected skips, GB300 passed 129 with 5 expected skips,
and each device passed 185 host tests plus 10 subtests. Real H200
TP8/EP2/MoE-DP2 and GB300 TP4/EP4 NCCL graph replays also passed. The final rebased
review sequence is listed in `REBASED_COMMIT_MAP.md`; older short hashes retained
above identify the campaign provenance that the original review discussed.

## Final interpretation

The review changed both code and conclusions. In particular, this campaign does
not claim that one kernel family wins everywhere, that SGL already beats TRTLLM
at every server point, that a one-GPU topology proxy is distributed evidence,
that the native CuTe DSL NVFP4 testbed is checkpoint/server attachment, that
static-route CuTe Tensor Core evidence is dispatch eligible, or that
benchmark-only packed-rank storage is already a complete serving residency
system. Promoted policies are keyed by device/provider, phase, graph mode, rank,
route/occupancy contract, activation/layout, and output/finalize ownership.
Close, incompatible, or unimplemented cells retain the established path or are
rejected explicitly.
