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

Each result reports p20/p50/p80 kernel latency and p50 speedup. The JSON also
records the GPU, CUDA, Torch, Triton, host, timestamp, Git revision, and command
arguments. It also records whether that revision has local changes; preserve an
external source snapshot when benchmarking a dirty checkout. No performance
threshold is enforced; compare results on the same idle GPU and software image.
The benchmark repeatedly uses the same inputs, so these are hot-cache kernel
timings rather than end-to-end request latency.

CUDA-graph timing is the default: after the correctness launches JIT-compile
each schedule, the benchmark captures `--inner-iterations` launches in a
separate graph for each implementation, times one replay, and divides by the
launch count. This keeps Python enqueue gaps out of the small-kernel result.
Use `--execution eager` only when diagnosing the ordinary launch path.
