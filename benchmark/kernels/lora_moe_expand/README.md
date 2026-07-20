# Gate/up LoRA expand schedule benchmark

`bench_gate_up_schedules.py` compares the legacy flat midpoint schedule with
the sgl_lora two-slice schedule. Routing metadata is built once per shape and
is shared by both kernels; routing alignment is intentionally outside the timed
region.

The default run covers BF16 H=192, top-k 8, ranks 16/32/64, and decode batches
1/16/32:

```bash
python benchmark/kernels/lora_moe_expand/bench_gate_up_schedules.py \
  --json-output /tmp/sgl_lora_gate_up_expand.json
```

The output includes two comparisons:

- `ISO_TILE`: both grids use `BLOCK_SIZE_N=64`, isolating grid construction.
- `POLICY`: each grid uses its default policy. At H=192, the midpoint-safe
  flat grid and the two-slice grid both use 64. The two-slice default is capped
  there because the GB300 sweep found that a 128-column masked tile regresses
  the larger tested ranks.
- `WIDE_TILE` is optional (`--comparisons wide_tile`): flat uses its safe
  default while two-slice is forced to 128, reproducing the fewer-CTA versus
  masked-work experiment.

Each result reports p20/p50/p80 kernel latency, `sliced_speedup_p50` as
`flat_latency / sliced_latency`, and `sliced_delta_pct` as
`100 * (flat_latency / sliced_latency - 1)`. Positive delta means sliced is
faster; negative means flat is faster. The JSON also records the GPU, CUDA,
Torch, Triton, host, timestamp, Git revision, and command arguments. It also
records whether that revision has local changes; preserve an external source
snapshot when benchmarking a dirty checkout. No performance threshold is
enforced; compare results on the same idle GPU and software image. The benchmark
repeatedly uses the same inputs, so these are hot-cache kernel timings rather
than end-to-end request latency.

CUDA-graph timing is the default: after the correctness launches JIT-compile
each schedule, the benchmark captures `--inner-iterations` launches in a
separate graph for each implementation, times one replay, and divides by the
launch count. This keeps Python enqueue gaps out of the small-kernel result.
Use `--execution eager` only when diagnosing the ordinary launch path.

## General sliced-expand prototype

`bench_general_sliced_expand.py` is a benchmark-only prototype for one routed
LoRA-B kernel source compiled into three schedules:

- `ALIGNED_FLAT` uses arithmetic slice offsets when every equal-width slice
  boundary is aligned to `BLOCK_SIZE_N`.
- `UNIFORM_RAGGED` restarts the N grid for each of any number of equal-width
  slices and masks each slice tail independently.
- `GENERAL_RAGGED` uses graph-stable runtime prefixes for unequal-width slices;
  its ragged grid launches only real slice tiles.

The first two schedules contain no runtime slice-metadata loads after Triton
specialization. The benchmark covers one slice, equal two- and three-slice
cases, unequal three-slice QKV-like shapes, and the focused 48-column-per-slice
gate/up case (`h48` is retained as the CLI case name for artifact continuity).

The focused slice-width-48 sweep compares the legal flat `BN=16` midpoint
schedule with slice-aware `BN=16/32/64/128` schedules:

```bash
python benchmark/kernels/lora_moe_expand/bench_general_sliced_expand.py \
  --cases h48 \
  --tokens 1 16 32 \
  --ranks 16 32 64 \
  --block-sizes 16 32 64 128 \
  --json-output /tmp/general_sliced_h48.json
```

The prototype does not change the production kernel. Its purpose is to measure
the metadata/scheduling cost before replacing the aligned-flat and equal-two
fast paths with a common source abstraction. Compare variants at the same
`BLOCK_SIZE_N` to compare the aggregate compiled implementations, and compare
the best block size per variant to evaluate a dispatch policy. Prototype versus
production also changes constexpr specialization and row-grid arithmetic, so it
is not a pure measurement of metadata-load overhead. Do not treat a slice-width
divisibility check as a performance policy: it proves only that a flat tile does
not cross a logical boundary.
