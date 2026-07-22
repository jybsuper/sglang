# Distributed D0 validation: SGL LoRA MoE

## Outcome

All 33 planned matrix cases passed across
204 rank reports. Every rank passed the fixed-shape
post-dispatch CUDA-graph replay, every case matched the independent full-weight
oracle, and the two-node GB300 run retained both per-node Nsight Systems reports
and readable NCCL summaries. This is correctness and topology evidence; the cold
diagnostic timings in raw JSON are deliberately not performance claims.

| Platform | Device | Topologies | Cases | Rank reports | Graph passes | Worst rel-L2 | Min cosine |
|:---|:---|:---|---:|---:|---:|---:|---:|
| gb300_two_node_8rank | NVIDIA GB300 | tp8_ep2_dp2 | 9 | 72 | 72/72 | 0.0002554186 | 0.999999821 |
| h200_single_node_4rank | NVIDIA H200 | tp4_ep1_dp1, tp4_ep1_dp4, tp4_ep4_dp1 | 15 | 60 | 60/60 | 0.001066405 | 0.999999523 |
| h200_single_node_8rank | NVIDIA H200 | tp8_ep2_dp2 | 9 | 72 | 72/72 | 0.0002554186 | 0.999999821 |

## What the harness actually executes

```mermaid
flowchart LR
    A["DP-owned BF16 tokens + GLOBAL expert IDs"] --> B["Real NCCL EP all_to_all_single"]
    B --> C["Ownership check; GLOBAL to LOCAL expert ID"]
    C --> D["Local expert buffers + serving SGL virtual-expert gate/up LoRA A+B"]
    D --> E["BF16 base gate/up + SwiGLU"]
    E --> F["Serving SGL virtual-expert down LoRA A+B + BF16 base down"]
    F --> G["Real SGL MoE-TP all-reduce"]
    G --> H["Real reverse EP all_to_all_single"]
    H --> I["Top-k weighted reduction in origin-token domain"]
    I --> J["Real SGL MoE-DP padded all-gather"]
```

The transported metadata is exactly `global_token_id`, `topk_slot`,
`global_expert_id`, `adapter_id`, and `origin_ep_rank`. Global IDs remain attached
through dispatch. Conversion to local expert IDs happens once, after ownership is
known and before local-buffer indexing. Local expert weights have `E / EP` expert
rows; gate/up and down intermediate dimensions are sharded by MoE-TP.

The independent oracle uses the unsharded full weights and explicitly recreates
the BF16 boundaries of each MoE-TP partial. It does not call the serving LoRA
kernels under test.

## Coverage

- H200, TP4/EP1/MoE-DP1: standard routing, active/mixed/base adapters.
- H200, TP4/EP4/MoE-DP1: all-local/mixed/no-local routing crossed with
  active/mixed/base adapters.
- H200, TP4/EP1/MoE-DP4: uneven DP token domains `(1, 3, 7, 13)`, crossed with
  active/mixed/base adapters.
- H200, TP8/EP2/MoE-DP2: all-local/mixed/no-local crossed with
  active/mixed/base adapters; uneven DP token domains `(5, 11)`.
- Two physical GB300 nodes, TP8/EP2/MoE-DP2: the same nine route/adapter cells.
- Rank-16, two adapters, top-k 2, BF16, eight global experts. Rank 8/16 broader
  performance guardrails and quantized providers are separate plan lanes; D0 is
  the distributed contract lane.

`cases.csv` has one row per matrix cell. `summary.json` retains every rank's host
and exact SGL group membership.

## CUDA-graph claim and collective boundary

Each rank prewarms route metadata, captures only the fixed-shape local compute
from post-dispatch gate/up through down output, poisons the destination, and
replays once before comparing with eager output. All
204 rank captures passed.

NCCL collectives are intentionally outside that graph:

- EP variable-split forward/reverse all-to-all is eager;
- MoE-TP all-reduce is eager;
- uneven MoE-DP padded all-gather is eager.

The report field `collectives_in_graph=false` prevents this evidence from being
misread as a full distributed-pipeline graph capture.

## Two-node GB300 topology and Nsight evidence

Torchrun placed ranks 0-3 on `rdx-gb300-r01-c012.rdx.local` and ranks 4-7 on
`rdx-gb300-r01-c018.rdx.local`. With SGL's rank order, EP and MoE-TP groups are
within a physical node, while MoE-DP groups `[0,4]`, `[1,5]`, `[2,6]`, and
`[3,7]` cross nodes. Thus the EP all-to-all is real NCCL but intra-node in this
specific placement; the MoE-DP all-gather is the exercised inter-node collective.

The retained two-node no-local/active trace passed graph replay and the oracle.
Its logs contain 1536 `via P2P/MNNVL` channel lines and
48 MNNVL clique lines. The per-node CUDA kernel
summaries contain NCCL `SendRecv`, `AllReduce`, and `AllGather` kernels. The
binary `.nsys-rep` files are the authoritative timeline; `topology_evidence.txt`
extracts the rank placement, representative channel lines, and kernel totals.

NCCL may print `nNodes 1` for an eight-GPU MNNVL clique even though the rank hosts
above are two physical GB300 nodes. That is NCCL's fabric-clique topology view,
not evidence that the launch collapsed to one host.

## Full-server smoke

A real `Qwen/Qwen1.5-MoE-A2.7B` server ran TP4/EP4 on H200 with
`lora_execution_engine=sgl_lora`, virtual experts, the csgmv LoRA backend, and a
Qwen MoE adapter targeting `gate_up_proj` and `down_proj`.

- base -> adapter -> base returned stable base IDs around the adapter transition;
- a two-request mixed base/adapter batch matched the two individual results;
- all requests returned HTTP 200 and the adapter loaded on TP0/EP0 through
  TP3/EP3;
- the first adapter latency includes JIT compilation and is not a latency sample.

The initial launch selected the installed FlashInfer attention package and stopped
before model execution because version 0.6.12 was below the required 0.6.15. The
successful control selected Triton attention explicitly. Server CUDA graphs were
disabled; graph evidence belongs to the distributed contract harness described
above.

Base output IDs: `[32671, 624, 59604, 198, 39572, 198, 94409, 198]`

Adapter output IDs: `[220, 22, 18, 15, 24, 488, 220, 19]`

## Reproduction

Single-node example:

```bash
PYTHONPATH=python torchrun --standalone --nproc-per-node=4 \
  benchmark/kernels/lora_moe/bench_distributed.py \
  --topology tp4_ep4_dp1 --route-mode mixed --adapter-mode mixed \
  --graph-replay --output /tmp/tp4_ep4_dp1_mixed_mixed.json
```

Two-node execution used one `srun` task per node, then `torchrun --nnodes=2
--nproc-per-node=4` with node ranks from `SLURM_NODEID`. The no-local/active trace
wrapped that torchrun command with:

```bash
nsys profile --trace=cuda,nvtx,nccl --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=<per-node-output> torchrun <two-node-args>
```

Regenerate the bundle summaries after placing raw artifacts under this directory:

```bash
python benchmark/kernels/lora_moe/summarize_distributed_d0.py \
  --root benchmark_results/sgl_lora_moe_20260722/distributed_d0
```

The exact GPU-executed harness snapshot SHA256 is
`5a2dc6b1ceca280934870f16d1c32ba3bd016ddbe1b5cae6d000410a457c507d`. It was checked
byte-for-byte against the isolated H200 source and the shared two-node GB300
source before the allocation was released. The repository harness is formatted
with the current project hooks; the summarizer verifies that its parsed Python
AST is identical to the retained executed snapshot.

## Limits and interpretation

- This is not an end-to-end throughput benchmark. Setup, first-JIT, and profiler
  overhead dominate the recorded timing fields.
- The harness uses real SGL process groups and serving-owned virtual-expert LoRA
  kernels, but it constructs an explicit PyTorch NCCL EP dispatcher to make the
  global/local-ID boundary directly auditable. It does not claim to be the final
  production dispatcher implementation.
- The full server uses SGL's normal replicated-input EP path (`moe_a2a_backend=none`)
  rather than the harness's explicit A2A transport.
- DeepEP/FlashInfer A2A, graph-captured collectives, lifecycle/eviction, shared
  experts, TP8/EP layouts with EP spanning physical nodes, and quantized providers
  remain separate validation lanes.
- `SHA256SUMS` covers every retained artifact except itself. `MANIFEST.json`
  records file sizes and hashes; ignored binary Nsight reports must be added with
  `git add -f` when publishing this evidence directory.
