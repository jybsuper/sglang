# Shared-expert phase handoff

## Exact validation

- H200, GPU 6: `20 passed, 5 skipped` in `9.35s`.
- GB300, GPU 2: `20 passed, 5 skipped` in `13.49s`.
- Post-integration H200/GB300 map + planner suites: `13 passed` on each device.
- Post-integration H200 runner/production-plan regression: `15 passed`.
- Suite:
  - `test/registered/unit/lora/test_sgl_lora_shared_experts.py`
  - `test/registered/lora/test_sgl_lora_shared_expert_ids.py`
  - `test/registered/lora/test_virtual_experts_kernels.py`
- Final fully-synced `fused_per_rank` CUDA-graph smoke passed on both devices.

## Matrix accounting

- 96 primary records: 6 Qwen3.5-shaped cases × 4 variants × 2 execution
  modes × 2 devices.
- 48 reverse-order repeats: 6 cases × 4 variants × CUDA graph × 2 devices.
- 16 tiny-shape smoke records: 4 variants × 2 execution modes × 2 devices.
- Primary/repeat correctness: 144/144 passed, minimum cosine
  `0.99998337`, maximum absolute error `3.05176e-05`, and zero records in
  which a shared slot received LoRA.

## Winner boundaries

- Decode CUDA graph (`T=1/32`, `R=32/64/128`): fused physical slots are
  `1.4–10.1%` faster than serial shared experts on H200/GB300.
- Prefill CUDA graph (`T=256/512`, `R=32/64`): fused slots are roughly
  `0.3–1.6%` faster.
- High-rank two-sink prefill (`T=256`, `R=128`, `S=2`): all variants are tied
  within `0.35%`; do not encode a winner.
- Eager usually favors fused slots by `2.5–20.8%`, but host launch contention
  makes graph evidence the stronger policy input.
- Per-rank map vs global fused layout: sub-percent in repeated graph results;
  the physical layout follows the provider and is not an autotune decision.

## Timeline artifacts

Each path below exists under both `raw/h200/nsys/` and `raw/gb300/nsys/` with
`.nsys-rep`, `.sqlite`, and `.analysis.json` where applicable:

- `qwen35-decode-t32-r128-two-sink-mixed__separate_overlap__cuda_graph`
- `qwen35-decode-t32-r128-two-sink-mixed__fused_per_rank__cuda_graph`
- `qwen35-prefill-t256-r128-two-sink-mixed__separate_overlap__cuda_graph`
- `qwen35-prefill-t256-r128-two-sink-mixed__fused_per_rank__cuda_graph`

Nsight Compute controls are under `raw/<device>/ncu/`:

- `<device>-fused-global-route.ncu-rep`
- `<device>-fused-per-rank-route.ncu-rep`
- matching `.details.txt` files.

## Production boundary

Production support is limited to the Standard dispatcher and its validated
safe physical-ID layouts. Fused shared slots use C0. The `fused_per_rank`
matrix is a mapping/layout proxy, not EP>1 serving graduation: current
MegaMOE/DeepEP-style paths can remap per-rank physical IDs before LoRA mapping,
so per-rank shared layouts with EP>1 and advanced all-to-all dispatchers remain
rejected.

Current C2/C3 consumers use one raw top-k ID as both the base physical row
(which includes shared slots) and routed LoRA factor row (which excludes
them); fused down-finalize also requires equal expert dimensions. Therefore
the planner must explicitly select C0 whenever `num_fused_shared_experts > 0`.

The deferred mapped-C2/C3 extension needs separate `base_physical_expert_id`
and `routed_factor_expert_id` inputs in the fused consumer and finalizer.  It
must preserve base activation/reduction for shared pairs while masking their
gate/up-B, down-A, and down-B LoRA work.  Do not remove the C0 guard until that
ABI has independent-oracle, mixed/base-row, one/two-slot, eager/graph, H200,
GB300, and real D0 evidence.
