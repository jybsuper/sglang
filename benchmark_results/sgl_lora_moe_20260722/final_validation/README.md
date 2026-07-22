# Terminal post-rebase validation

These logs are the terminal validation of the clean source snapshot
`f2f406e0560bb8c95479eeb164bf93b65fb6a0c1`, rebased on OSS main
`4eaa5ca6510622cb0006bcfee5947b17859ac8c7`. The identical tree was copied to
`/sgl-workspace/sglang-final-f2f406e` on both devices before the runs.

## Result

| Device | Runtime | GPU pytest shards | Host/planner suite | Real NCCL replay |
|---|---|---:|---:|---|
| NVIDIA H200 (SM90) | PyTorch 2.11.0+cu129, CUDA 12.9, Triton 3.6.0 | 128 passed, 6 expected architecture skips | 185 passed + 10 subtests | TP8 / EP2 / MoE-DP2 passed |
| NVIDIA GB300 (SM103) | PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0 | 129 passed, 5 expected architecture skips | 185 passed + 10 subtests | TP4 / EP4 / MoE-DP1 passed |

The branch-relevant pytest warnings are dependency/config deprecations, and no
branch-relevant test failed. The one
additional H200 skip is the Blackwell-only native NVFP4 synthetic provider fixture.
Marlin's dirty-destination and distinct per-invocation lock-workspace regression
test passed in `g2.log` on both devices.

The distributed JSON files are the machine-readable source of truth. They record
the exact benchmark command and per-rank topology, collective groups, global-to-local
expert-ID boundary, mixed adapter/route modes, graph replay scope, transport counts,
and numerical oracle. NCCL collectives execute eagerly outside the captured local
post-dispatch compute graph; the JSON labels that boundary explicitly.

## Files

- `g0.log` through `g3.log`: disjoint GPU pytest shards.
- `cpu.log`: host planner, policy, contract, and orchestration tests.
- `distributed.log`: complete launcher/rank output for the real NCCL replay.
- `distributed_*.json`: structured, per-rank distributed result and exact command.

## Broader non-LoRA control

An intentionally broader general quant-kernel diagnostic is retained separately.
The H200 combined diagnostic passed 200 cases with one expected Blackwell-only skip.
On GB300, 66 cases in the pre-existing unmasked
`test_v2_jit_matches_aot` bit-exact comparison failed while 135 cases passed. The
new masked-layout cases and all LoRA provider-plan cases passed in the focused
`g2.log` (41 passed). One representative failing case was then run against the
official-main AOT source and failed identically; see
`gb300/upstream_general_jit_control.log`. Therefore this is an upstream GB300
JIT-versus-AOT numerical-equivalence issue, not a regression from the LoRA branch.
It is archived transparently in `optional_general_jit_aot_*.log` but excluded from
the branch acceptance count.

These are correctness and integration logs, not performance measurements. Canonical
unprofiled timings and profiler reports remain in their dedicated campaign lanes.
