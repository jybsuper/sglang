# SGL LoRA MoE mixed-rank policy benchmark

The full M0 BF16 runner is identical in both arms. `padded_rmax` executes zero-tailed R_max factors; `packed_bucket` executes load-time packed R_phys factors. Packing, immutable rank-plan construction, and graph-family selection are outside forward timing. Every cell combines both forward-order and reverse-order samples per policy; exact counts are in the raw manifest.

## Combined counterbalanced results

| Device | Execution | Phase/T | Occupancy | R/Rmax | padded us | packed us | packed vs padded | Evidence | Packed/padded factor bytes |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| h200 | cuda_graph | decode/32 | all-base | 32/128 | 318.880 | 319.040 | +0.05% | all-base-bypass | 1.000 |
| h200 | cuda_graph | decode/32 | all-base | 64/128 | 318.624 | 318.864 | +0.08% | all-base-bypass | 1.000 |
| h200 | cuda_graph | decode/32 | all-base | 128/128 | 319.312 | 319.568 | +0.08% | all-base-bypass | 1.000 |
| h200 | cuda_graph | decode/32 | full-lora | 32/128 | 470.528 | 355.248 | -24.50% | packed_bucket | 0.250 |
| h200 | cuda_graph | decode/32 | full-lora | 64/128 | 469.728 | 390.576 | -16.85% | packed_bucket | 0.500 |
| h200 | cuda_graph | decode/32 | full-lora | 128/128 | 469.136 | 469.184 | +0.01% | matched-null-control | 1.000 |
| h200 | cuda_graph | decode/32 | mixed | 32/128 | 389.888 | 336.400 | -13.72% | packed_bucket | 0.250 |
| h200 | cuda_graph | decode/32 | mixed | 64/128 | 390.224 | 351.120 | -10.02% | packed_bucket | 0.500 |
| h200 | cuda_graph | decode/32 | mixed | 128/128 | 389.216 | 389.408 | +0.05% | matched-null-control | 1.000 |
| h200 | cuda_graph | prefill/2048 | all-base | 32/128 | 556.080 | 556.320 | +0.04% | all-base-bypass | 1.000 |
| h200 | cuda_graph | prefill/2048 | all-base | 64/128 | 555.520 | 555.712 | +0.03% | all-base-bypass | 1.000 |
| h200 | cuda_graph | prefill/2048 | all-base | 128/128 | 555.760 | 555.936 | +0.03% | all-base-bypass | 1.000 |
| h200 | cuda_graph | prefill/2048 | full-lora | 32/128 | 2012.160 | 913.264 | -54.61% | packed_bucket | 0.250 |
| h200 | cuda_graph | prefill/2048 | full-lora | 64/128 | 2013.904 | 1229.200 | -38.96% | packed_bucket | 0.500 |
| h200 | cuda_graph | prefill/2048 | full-lora | 128/128 | 2011.952 | 2013.264 | +0.07% | matched-null-control | 1.000 |
| h200 | cuda_graph | prefill/2048 | mixed | 32/128 | 1091.984 | 744.656 | -31.81% | packed_bucket | 0.250 |
| h200 | cuda_graph | prefill/2048 | mixed | 64/128 | 1092.832 | 823.536 | -24.64% | packed_bucket | 0.500 |
| h200 | cuda_graph | prefill/2048 | mixed | 128/128 | 1091.840 | 1091.584 | -0.02% | matched-null-control | 1.000 |
| h200 | eager | decode/32 | all-base | 32/128 | 326.272 | 325.072 | -0.37% | all-base-bypass | 1.000 |
| h200 | eager | decode/32 | all-base | 64/128 | 325.568 | 324.512 | -0.32% | all-base-bypass | 1.000 |
| h200 | eager | decode/32 | all-base | 128/128 | 326.528 | 325.200 | -0.41% | all-base-bypass | 1.000 |
| h200 | eager | decode/32 | full-lora | 32/128 | 784.864 | 634.576 | -19.15% | packed_bucket | 0.250 |
| h200 | eager | decode/32 | full-lora | 64/128 | 772.048 | 645.056 | -16.45% | packed_bucket | 0.500 |
| h200 | eager | decode/32 | full-lora | 128/128 | 734.624 | 734.384 | -0.03% | matched-null-control | 1.000 |
| h200 | eager | decode/32 | mixed | 32/128 | 749.216 | 642.688 | -14.22% | packed_bucket | 0.250 |
| h200 | eager | decode/32 | mixed | 64/128 | 737.296 | 643.152 | -12.77% | packed_bucket | 0.500 |
| h200 | eager | decode/32 | mixed | 128/128 | 739.728 | 737.248 | -0.34% | matched-null-control | 1.000 |
| h200 | eager | prefill/2048 | all-base | 32/128 | 562.688 | 562.560 | -0.02% | all-base-bypass | 1.000 |
| h200 | eager | prefill/2048 | all-base | 64/128 | 563.264 | 562.992 | -0.05% | all-base-bypass | 1.000 |
| h200 | eager | prefill/2048 | all-base | 128/128 | 562.320 | 562.320 | +0.00% | all-base-bypass | 1.000 |
| h200 | eager | prefill/2048 | full-lora | 32/128 | 2033.216 | 933.232 | -54.10% | packed_bucket | 0.250 |
| h200 | eager | prefill/2048 | full-lora | 64/128 | 2033.136 | 1248.768 | -38.58% | packed_bucket | 0.500 |
| h200 | eager | prefill/2048 | full-lora | 128/128 | 2035.056 | 2035.104 | +0.00% | matched-null-control | 1.000 |
| h200 | eager | prefill/2048 | mixed | 32/128 | 1113.408 | 764.576 | -31.33% | packed_bucket | 0.250 |
| h200 | eager | prefill/2048 | mixed | 64/128 | 1112.000 | 842.640 | -24.22% | packed_bucket | 0.500 |
| h200 | eager | prefill/2048 | mixed | 128/128 | 1112.864 | 1112.592 | -0.02% | matched-null-control | 1.000 |

## Raw static-policy regret (noise-qualified)

A fixed policy is scored against the oracle policy for every included batch cell. This is the deployable graph-planner question; oracle-per-batch results are not presented as a production policy. Raw regret ratios remain visible, but categorical winner counts include only effects with order-consistent winners beyond both matched per-order null effects. This is an observed-bound heuristic, not a statistical significance test. All-base and R=Rmax null-control cells are excluded.

| Scope | Key | Cells | Raw best fixed | Mean regret | P95 regret | Max regret | Evidence wins padded/packed | Inconclusive |
|---|---|---:|---|---:|---:|---:|---:|---:|
| device_execution_global | `['h200', 'cuda_graph']` | 8 | packed_bucket | 0.00% | 0.00% | 0.00% | 0/8 | 0 |
| device_execution_global | `['h200', 'eager']` | 8 | packed_bucket | 0.00% | 0.00% | 0.00% | 0/8 | 0 |
| device_execution_rank_signature | `['h200', 'cuda_graph', 32, 128]` | 4 | packed_bucket | 0.00% | 0.00% | 0.00% | 0/4 | 0 |
| device_execution_rank_signature | `['h200', 'cuda_graph', 64, 128]` | 4 | packed_bucket | 0.00% | 0.00% | 0.00% | 0/4 | 0 |
| device_execution_rank_signature | `['h200', 'eager', 32, 128]` | 4 | packed_bucket | 0.00% | 0.00% | 0.00% | 0/4 | 0 |
| device_execution_rank_signature | `['h200', 'eager', 64, 128]` | 4 | packed_bucket | 0.00% | 0.00% | 0.00% | 0/4 | 0 |

## Interpretation boundary

- Core cells have a uniform active rank, including base+LoRA mixed rows; heterogeneous resident ranks are represented by the static planner but need multi-bucket execution before production dispatch can use them.
- All-base cells resolve to N0 for both policies. Their small deltas are counterbalanced timing noise, not rank compute, and are excluded from static rank-policy regret.
- R128/Rmax128 active cells are same-physical-rank null controls. They are matched on the complete non-rank shape and occupancy identity. A lower-rank row gets an evidence winner only when forward and reverse order agree and each order-specific effect exceeds its matched null.
- `resident_bytes` and the factor-byte ratio cover the four LoRA factor tensors only; they exclude base weights, intermediates, routing metadata, and CUDA-graph pools.
- This is WS1 local M0 evidence. It says nothing about distributed communication or canonical adapter load/eviction latency. Reported transform samples are descriptive one-shot observations only.
- Each topology is captured and replayed independently. The benchmark serializes the graph key but does not validate a production graph cache or selection transition across slot/rank changes.
- Active graph keys include the exact slot/rank assignment. Production can avoid recapture churn only after defining stable bucket-pool graph ownership; this planner intentionally does not claim that lifecycle.
- Current padded kernels do not consume `lora_ranks`; zero tails are a semantic requirement. Packed execution makes physical rank explicit.
- Rank-metadata allocation-free replay is a reviewed construction invariant: immutable tuple metadata is bound before capture/launch. The numeric zero is not presented as allocator-instrumented evidence.
- Active correctness uses a matched same-shape R128 numerical control, a two-BF16-epsilon output-scale envelope capped below half the LoRA signal, delta relative-L2 <= 0.1, and delta cosine >= 0.995. Exact per-artifact values are in `summary.json`.

See `summary.json`, `raw_manifest.csv`, raw JSON files, and `SHA256SUMS` for complete samples and provenance.
