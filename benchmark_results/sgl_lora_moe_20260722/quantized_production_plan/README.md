# Quantized SGL-LoRA production-plan graduation evidence

Date: 2026-07-22

## Scope and result

This bundle validates the **production SGL-LoRA MoE execution entrypoint**, not
only isolated provider microbenchmarks. `run_sgl_lora_moe_plan` consumes the
host-resolved plan and executes the complete C0 topology using provider-native
base GEMMs plus BF16 LoRA arithmetic:

```text
packed/local top-k + hidden
  -> gate/up LoRA A+B
  -> provider gate/up GEMM
  -> delta-aware SwiGLU
  -> provider W2 input conversion/quantization
  -> provider down GEMM
  -> down LoRA A+B
  -> requested BF16/FP32 destination
```

The representative graduation matrix passed **28/28 cells on H200** and
**42/42 cells on GB300**. It covers FP8 W8A8 and Marlin W4A16 on H200, plus
FP8 W8A8, native NVFP4 W4A4, and Marlin W4A16 on GB300. Every provider was
checked with active, mixed, and base-only adapter occupancy; eager and CUDA
graph execution; direct ranks 16/32/64 and runtime-general rank 128; BF16 and
FP32 destinations; decode and prefill shapes; and local Kimi K2.5/GLM-5.2
geometry anchors.

## Production changes validated

- Provider-neutral execution dispatch keeps C0 common to BF16, FP8, NVFP4,
  and Marlin while leaving the measured fused C2/C3 tails BF16-only.
- The resolved execution plan records the provider contract key; execution
  does not infer a second policy after planning.
- Packed top-k remains a standard semantic carrier regardless of the
  experimental producer switch.
- Blackwell FP8 fixtures use the same packed UE8M0 weight-scale ABI as the
  production FP8 load path.
- Masked, already-activated FP8 input now selects `MaskedLayoutScheduler`.
  The scheduler dynamically uses 16/8/4/2/1 subwarps so hidden sizes such as
  512, 2048, and 7168 are legal without quantizing padded expert rows.
- The virtual-expert shrink PDL signal is emitted only for the same-call
  direct expand path, and the direct flat/two-slice consumers launch with PDL
  and execute a matching device-side wait. Runtime-general and split-stage
  paths keep ordinary stream ordering.

The two correctness fixes were found by this production harness. Before the
paired PDL wait, GB300 FP8 Kimi-shaped CUDA-graph output diverged by roughly
3529. After the PDL fix, a remaining sub-unit error exposed the masked FP8
quantizer selecting the flat scheduler. With both fixes, the Kimi-shaped FP8
case passes eager and graph execution, base rows remain provider-N0 identical,
and the representative matrices are clean.

## Matrix definition

Each provider runs the following 14 cells in a fresh process:

| Dimension | Representative cells |
| --- | --- |
| Core rank-64 decode | active/mixed/base x eager/CUDA graph |
| Rank schedules | rank 16 active graph, rank 32 active graph, rank 128 mixed graph |
| Destination | rank-64 active graph with BF16 destination; other cells use FP32 |
| Phase | rank-64 mixed prefill, eager and graph |
| Large local geometry | Kimi K2.5 (`H=7168`, `I=2048`) and GLM-5.2 (`H=6144`, `I=2048`) graph anchors |
| Routing | 8 local experts, top-k 8, virtual experts, two active adapters unless occupancy is base-only |

The model anchors simulate one rank's local tensor geometry. They do not
measure TP/EP/MoE-DP communication. Distributed execution is validated by a
separate lane in this refactor.

## Correctness summary

| Device | Provider | Cells | Max graph-vs-eager abs diff | Max base-row vs provider-N0 abs diff |
| --- | --- | ---: | ---: | ---: |
| H200 | FP8 W8A8 | 14 | 0.01667 | 0.00000 |
| H200 | Marlin W4A16 | 14 | 0.01563 | 0.00760 |
| GB300 | FP8 W8A8 | 14 | 0.03457 | 0.00000 |
| GB300 | native NVFP4 W4A4 | 14 | 0.02808 | 0.00000 |
| GB300 | Marlin W4A16 | 14 | 0.01172 | 0.00814 |

All active/mixed cases produced a nonzero LoRA delta, all outputs were finite,
and all base-only rows stayed inside the provider-specific N0 tolerance.

The final focused regression suites also passed:

- local CPU planner tests: `6 passed`;
- H200: `14 passed, 1 skipped, 196 deselected` (NVFP4 is intentionally
  skipped on Hopper);
- GB300: `15 passed, 196 deselected`.

JUnit reports are `h200/pytest_quantized.xml` and
`gb300/pytest_quantized.xml`.

## Matched-N0 timing anchor

The table below is the rank-64 mixed CUDA-graph cell. `plan/N0` measures the
complete LoRA production plan against the same provider's no-LoRA base path;
it is an overhead ratio, not a cross-provider speed comparison.

| Device | Provider | Provider N0 (ms) | Production plan (ms) | Plan/N0 |
| --- | --- | ---: | ---: | ---: |
| H200 | FP8 W8A8 | 0.03880 | 0.08107 | 2.090x |
| H200 | Marlin W4A16 | 0.04022 | 0.07618 | 1.894x |
| GB300 | FP8 W8A8 | 0.04196 | 0.07662 | 1.826x |
| GB300 | native NVFP4 W4A4 | 0.03217 | 0.06647 | 2.066x |
| GB300 | Marlin W4A16 | 0.03786 | 0.07236 | 1.911x |

Provider setup, weight conversion, and JIT compilation are excluded from the
CUDA-event latency and retained separately as `provider_setup_ms_excluded`.
The harness intentionally used default MoE configs where no tuned config file
existed, so these numbers graduate correctness and executable topology; they
are not the final performance ceiling.

## Structural profiling

Compact Nsight Systems captures contain exactly two active eager forwards;
the corresponding `*_active_cuda_gpu_trace.csv` files are directly readable,
and the small `*_active_range.nsys-rep` files preserve the raw timeline.
Nsight Compute `--set basic` reports one active forward in `*_ncu_raw.csv`.

The traces confirm the expected ordered chain for every provider:

1. virtual-top-k construction, alignment, and sanitize;
2. gate/up LoRA shrink and paired direct expand;
3. provider dispatch/input preparation and provider gate/up GEMM;
4. `_silu_mul_delta_masked_kernel`;
5. provider W2 conversion/quantization and provider down GEMM;
6. reorder/finalize followed by down-LoRA shrink and expand.

The FP8 trace specifically shows the post-activation W2 conversion using
`MaskedLayoutScheduler`; padded expert rows are not sent through the flat
scheduler. NVFP4 shows its BF16-to-FP4 conversion before each provider GEMM,
and Marlin shows both Marlin GEMMs with the BF16 delta-aware activation bridge.

## Reproduction

Representative matrix:

```bash
python3 benchmark/kernels/lora_moe/run_quantized_production_matrix.py \
  --device gb300 --provider all --suite representative \
  --warmups 2 --iterations 5 --output-dir /tmp/quantized_production
```

Compact structural trace (replace provider/device as needed):

```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none -o /tmp/fp8_active \
  python3 benchmark/kernels/lora_moe/bench_quantized_moe_pipeline.py \
  --provider fp8 --tokens 32 --experts 8 --top-k 8 --hidden 2048 \
  --intermediate 512 --rank 64 --adapters 2 --occupancy mixed \
  --phase decode --execution eager --warmups 0 --iterations 1 \
  --profile-range --profile-iterations 2
```

Basic per-kernel report:

```bash
ncu --set basic --profile-from-start off --target-processes all \
  -o /tmp/fp8_ncu \
  python3 benchmark/kernels/lora_moe/bench_quantized_moe_pipeline.py \
  --provider fp8 --tokens 32 --experts 8 --top-k 8 --hidden 2048 \
  --intermediate 512 --rank 64 --adapters 2 --occupancy mixed \
  --phase decode --execution eager --warmups 1 --iterations 1 \
  --profile-range --profile-iterations 1
```

## Artifact map

- `h200/manifest.json`, `gb300/manifest.json`: complete matrix definition,
  command outputs, per-file hashes, and pass/fail records.
- `h200/<provider>/*.json`, `gb300/<provider>/*.json`: contract, plan,
  correctness, and matched-N0 latency for each cell.
- `profiles/*/*_active_cuda_gpu_trace.csv`: ordered active-forward timelines.
- `profiles/*/*_active_range.nsys-rep`: compact raw Nsight Systems reports.
- `profiles/*/*_ncu_raw.csv`: per-kernel Nsight Compute basic metrics.
- `SHA256SUMS`: bundle integrity hashes.

## Explicit limits and follow-up seams

- Native NVFP4 evidence uses the full synthetic production provider pipeline.
  Server attachment still requires a resident compatible provider payload.
  Current server-side NVFP4 variants differ materially: ModelOpt CuteDSL-v2
  uses interleaved/fused storage, the TRTLLM fused path owns activation/A2
  quantization, and canonical CuteDSL-v1 is tied to DeepEP masked dispatch.
  This change does not claim checkpoint conversion or lifecycle support for an
  incompatible resident cache.
- Marlin currently consumes local expert IDs; a global-ID EP provider needs an
  explicit provider mapping contract rather than a larger global-sized LoRA
  buffer.
- Kimi and GLM entries validate local shapes only. Nemotron-3's non-gated
  ReLU2 semantics remain an explicit activation-contract gap; running those
  dimensions through SwiGLU would not validate the model.
- Shared experts, adapter loading/eviction, graph recapture policy, and
  multi-rank communication are separate planned lanes.
- Experimental TRTLLM is not reported as a neutral baseline because it cannot
  consume these resident provider payloads at equal ABI without load-time
  conversion. The matched-N0 ratios above are the honest comparison available
  at this boundary.
- Fusing the BF16 activation bridge, delta application, and provider W2
  quantization is the next provider-ABI optimization seam; this graduation
  commit deliberately establishes the correct common execution contract first.
