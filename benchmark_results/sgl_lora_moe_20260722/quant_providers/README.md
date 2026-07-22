# SGL LoRA MoE quant-provider evidence

This directory contains the focused evidence for the independent SGL LoRA MoE
base-provider lane. The harness drives the provider stages directly:

1. routing and optional input quantization;
2. provider-native W13;
3. a nonzero BF16 gate/up LoRA delta followed by SwiGLU;
4. an explicit BF16 activation bridge and optional W2 input quantization;
5. provider-native W2 and a weighted FP32 final destination.

Every recorded run also passes the caller's opaque `packed_topk_ids` object
through `prepare()` and checks Python object identity. This demonstrates that a
provider does not reconstruct or replace packed routing metadata; it is not a
validation of a particular bit-packing algorithm.

## Validated matrix

All measurements use `T=16, E=4, top_k=2, H=256, I=256`, five warmups, and 20
timed iterations. Latencies are direct provider-stage microbenchmarks, not model
or server latency.

| Provider contract | GPU | Mode | max abs | cosine | latency |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 `deepgemm_bf16` | H200 | eager | 0.000593 | 1.000000 | 0.3951 ms |
| BF16 `deepgemm_bf16` | H200 | graph replay | 0.000593 | 1.000000 | 0.01349 ms |
| FP8 W8A8 `deepgemm_fp8_w8a8` | H200 | eager | 0.085216 | 0.999074 | 0.5586 ms |
| FP8 W8A8 `deepgemm_fp8_w8a8` | H200 | graph replay | 0.085216 | 0.999074 | 0.01813 ms |
| Marlin W4A16 `marlin_w4a16` | H200 | eager | 0.013063 | 0.999984 | 0.3888 ms |
| Marlin W4A16 `marlin_w4a16` | H200 | graph replay | 0.013063 | 0.999984 | 0.01800 ms |
| NVFP4 W4A4 `cutedsl_nvfp4_w4a4` | GB300 | eager | 0.472935 | 0.973475 | 0.9140 ms |
| NVFP4 W4A4 `cutedsl_nvfp4_w4a4` | GB300 | graph replay | 0.472935 | 0.973475 | 0.02094 ms |

The JSON files are the authoritative results and include the exact thresholds,
software versions, device capability, semantic dtypes, shape, and wall time.

## Provider contracts and current attach support

| Contract | Physical row domain | W2 input after BF16 LoRA seam | Production attach boundary |
| --- | --- | --- | --- |
| BF16 | expert-masked | BF16 | Canonical unquantized `[E,N,K]` weights |
| FP8 W8A8 | expert-masked | FP8 plus explicit per-token-group scale | Canonical DeepGEMM FP8 weights/scales |
| NVFP4 W4A4 | expert-masked | packed FP4 plus swizzled E4M3 scale | Provider-native canonical gate-first, non-interleaved payload |
| Marlin W4A16 | routed-pair | BF16 | Resident Marlin W4 payload with local expert IDs |

The native NVFP4 provider itself is validated on GB300. The currently resident
ModelOpt standard formats are deliberately not auto-attached to it: CuteDSL-v2
stores `[Up,Gate]` interleaved weights/MMA scales, while the TRT-LLM form owns
activation and A2 quantization inside a fused kernel. Treating either as the
canonical provider payload would silently violate the LoRA injection contract.
Canonical CuteDSL-v1 is presently tied to DeepEP masked dispatch rather than the
unified Standard dispatcher. These cases fail explicitly at attach time.

Marlin currently requires already-local expert IDs. A non-null global-ID
`expert_map` is rejected instead of being silently misrouted.

## What the evidence does and does not establish

The eager and graph artifacts validate provider-native base stages, nonzero BF16
gate/up delta injection, the BF16 activation bridge, FP32 finalization, and
packed-routing identity at the tested shape. Graph mode captures and replays the
direct provider pipeline after warming JIT and workspace state.

This is not a full checkpoint, server, or end-to-end CUDA graph run. It excludes
virtual-expert LoRA A/B GEMM latency and does not validate every model geometry,
expert-parallel mapping, or fused stock weight representation. The runner's FP32
destination plumbing is covered separately by the focused runner test.

The BF16 oracle uses the source weights directly. FP8 and Marlin use dequantized
provider-weight references. NVFP4 uses the source BF16 weights because the
grouped scale swizzle is provider-private; cosine similarity is therefore its
primary quantized correctness check (`0.973475`, threshold `0.90`).

## Reproduction

Focused tests:

```bash
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 CUDA_VISIBLE_DEVICES=7 PYTHONPATH=python \
  python -m pytest -q \
  test/registered/lora/test_sgl_lora_runner.py \
  test/manual/cpu/test_sgl_lora_quant_providers.py \
  test/manual/cpu/test_sgl_lora_execution_plan.py \
  test/registered/lora/test_sgl_lora_production_plan.py
```

Provider harness (replace the provider and GPU as needed):

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=python \
  python benchmark/kernels/lora_moe/bench_quant_providers.py \
  --provider fp8 --mode eager --warmups 5 --iterations 20

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python \
  python benchmark/kernels/lora_moe/bench_quant_providers.py \
  --provider nvfp4 --mode graph --warmups 5 --iterations 20
```

See `SOURCE_PROVENANCE.md` for the executed content hashes and
`SHA256SUMS` for artifact integrity.
