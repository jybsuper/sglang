# SGL LoRA MoE benchmark artifacts — 2026-07-22

This is the durable evidence bundle for the H200 and GB300 SGL LoRA MoE
campaign. It replaces the TTL-sensitive paths on the GPU nodes and keeps raw
timings, correctness records, profiler reports, software/device provenance,
negative experiments, and generated summaries together.

The bundle is deliberately an evidence archive, not a blanket support claim.
Each lane's README defines the semantic boundary it measured and the limits of
the resulting production policy.

## Final interpretation

- The production BF16 planner promotes measured C2 and opt-in C3 regions and
  retains C0 as the explicit fallback. It does **not** beat experimental TRTLLM
  at every server point. In the final matched GB300 Qwen bracket, default C2
  trails by **9.43% / 9.44% / 10.39%** and opt-in C3 trails by
  **5.12% / 5.54% / 10.48%** at BS1/16/32. The larger
  **12.56%–14.71%** deficit belongs only to the historical pre-planner SGL
  path.
- Route metadata can be reused by consumers within one MoE layer. The current
  implementation does not memoize one route plan across model layers;
  cross-layer reuse remains an unpromoted optimization.
- The shared-outer gate-A selector is production-integrated only inside its
  measured static envelope. Final focused coverage passed on each device
  (**56 tests plus 10 subtests on H200 and on GB300**) and its host selector
  suite passed **9 tests**. Wider or noisier cells retain the generic path.
- Physical shared experts are production-safe only for the validated Standard
  dispatcher layouts. Shared/sink physical IDs are mapped out of the routed
  LoRA factor domain. Per-rank physical shared layouts with EP>1 and advanced
  all-to-all dispatchers remain rejected/unpromoted because those paths can
  remap physical IDs before LoRA mapping.
- Marlin W4A16 has a resident production attachment path (with local expert-ID
  requirements), and its terminal dirty-destination/per-invocation-workspace smoke
  passed on H200 and GB300. Dynamic-activation FP8 attaches only when its resident weight
  scales match the required ABI; static-activation FP8 is explicitly rejected.
  The native CuTe DSL NVFP4 W4A4 provider is validated through a synthetic
  production-plan testbed, but compatible checkpoint/server attachment remains
  a gap.
- The Blackwell CuTe Tensor Core study is evidence-only. Its static-route
  candidate wins one large-rank prefill boundary but fails route-mutation graph
  replay, so no production serving code imports it.

## Provenance

- Integration branch: `sgl-lora`.
- Tested source: `f2f406e0560bb8c95479eeb164bf93b65fb6a0c1`, rebased directly
  on OSS main `4eaa5ca6510622cb0006bcfee5947b17859ac8c7`. The exact state is
  recorded in `SOURCE_STATE.json`; later evidence/docs-only commits do not alter it.
- H200 source: `default/sglang-29157-v0514-h200-exact`.
- GB300 source: `yanbin-jiang-gb300-4gpu`.
- Retained formats include benchmark JSON, process logs, JUnit reports,
  Nsight Systems reports/SQLite exports, Nsight Compute reports/CSVs, and
  generated summaries.
- Triton/compiler caches, generated JIT cache metadata, PID files, and other
  machine-local transient state are not evidence and must not be included.

Individual lane manifests and source-hash files preserve the exact measured
snapshots where the campaign predates the final integration commit. Remote Git
revisions embedded in copied worktrees are not authoritative when source was
explicitly synchronized; use each lane's source hashes and README.

## Evidence map

| Topic | Directory | What it establishes |
|---|---|---|
| Core families and route controls | `algorithm_families/`, `rank8_16_guardrail/`, `route_countercandidates/` | Per-site candidate comparisons, strict delta correctness, IID/skew controls, low-rank legality, and negative candidates |
| BF16 planner and fused tails | `production_planner/`, `c2_down_finalize/`, `c3_overlap/` | Production C0/C2/C3 policy, fused down finalization, and bounded two-stream overlap |
| Model/server bracket | `e2e/` plus the final planner records in `production_planner/` | Historical pre-planner and final C2/C3 gaps versus the matched experimental TRTLLM control |
| Cross-model contracts | `cross_model_c2/` | Qwen/Kimi/GLM and odd-width non-gated contract guardrails; benchmark coverage is not automatic model attachment |
| Distributed topology | `distributed_d0/` | Real TP/EP/MoE-DP execution; local-shape simulations elsewhere remain proxies |
| Graph and adapter lifecycle | `lifecycle/` | Separate base/adapter graphs, graph transitions, eviction, slot reuse, and reload behavior |
| Host/cold controls | `loaded_host/` | Loaded-host eager and forced-cold evidence kept separate from hot replay results |
| Mixed-rank policy | `mixed_rank_policy/` | Packed physical-rank benefit and static-policy regret; not a complete residency manager |
| PDL controls | `pdl_control/` | Paired producer/consumer PDL behavior and explicit PDL-off controls |
| Quant provider boundary | `quant_providers/` | Direct provider-stage BF16/FP8/Marlin/NVFP4 testbed evidence |
| Quantized production C0 | `quantized_production_plan/` | Provider-neutral production entrypoint; Marlin attachment and explicit FP8/NVFP4 limits |
| Physical shared experts | `shared_experts/` | Shared-ID isolation and C0 behavior; production scope is Standard safe layouts only |
| Shared-outer exploration | `shared_outer/` | Broad gate-A/down-B factorization measurements and fallback evidence |
| Shared-outer production selector | `shared_outer_gate_production/` | Integrated static selector, H200/GB300 correctness, graph behavior, and launch-structure trace |
| Blackwell CuTe Tensor Core study | `cutedsl_tensorcore/` | Static-route upper bound and the route-mutation failure that blocks production integration |
| Terminal post-rebase validation | `final_validation/` | Exact H200/GB300 GPU and host suites plus real NCCL distributed graph replays at the tested source commit |
| Independent review | `THIRD_PARTY_REVIEW_FEEDBACK.md`, `THIRD_PARTY_REVIEW_DISPOSITION.md` | Original review and finding-by-finding disposition |
| Original device trees | `h200/`, `gb300/` | Early P0/P1 raw runs and profiler artifacts retained for audit |

## Integrity

The root `MANIFEST.txt`, `SHA256SUMS`, and `SOURCE_STATE.json` are generated
only after the final upstream rebase and artifact refresh. Verify the finished
bundle from this directory with:

```bash
shasum -a 256 -c SHA256SUMS
```

Per-lane checksum files remain useful for detecting changes to the copied raw
evidence. The root checksums are authoritative for the final assembled bundle.

The interpretation rules, evidence ladder, exact tables, and reproduction
commands live in the companion architecture audit. Benchmark drivers live
under `benchmark/kernels/lora_moe/` in the repository snapshot identified by
`SOURCE_STATE.json`.
