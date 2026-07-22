#!/usr/bin/env python3
"""Benchmark model shared experts together with routed-only SGL LoRA.

This is the ``SH``/local-M0 graduation driver.  It compares the same logical
MoE block in four execution layouts:

* ``separate_serial``: routed MoE+LoRA, then a conventional dense shared MLP;
* ``separate_overlap``: the same shared MLP overlaps the routed block;
* ``fused_global``: shared experts are appended as global physical slots;
* ``fused_per_rank``: DeepEP/MegaMOE-style per-rank shared slots are interleaved
  in the global physical expert space (a WS1 identity/layout proxy, not D0).

Every adapter targets routed experts only.  Shared slots are therefore mapped
to ``-1`` before virtual-expert LoRA routing.  The independent PyTorch oracle
starts from logical routed/shared weights and never consumes production route
plans or physical expert IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from benchmark.kernels.lora_moe.bench_local import (  # noqa: E402
    _detect_device,
    _ensure_benchmark_server_args,
)
from benchmark.kernels.lora_moe.profiling import (  # noqa: E402
    RunConfig,
    cuda_profile_range,
    make_batch,
    time_cuda_events,
)

Variant = Literal[
    "separate_serial",
    "separate_overlap",
    "fused_global",
    "fused_per_rank",
]
VARIANTS: tuple[Variant, ...] = (
    "separate_serial",
    "separate_overlap",
    "fused_global",
    "fused_per_rank",
)


@dataclass(frozen=True, slots=True)
class SharedExpertBenchCase:
    case_id: str
    phase: Literal["decode", "prefill"]
    tokens: int
    hidden_size: int
    intermediate_size: int
    num_routed_experts: int
    routed_topk: int
    rank: int
    active_adapters: int
    include_base_rows: bool
    lora_capacity: int
    num_shared_experts: int
    ep_proxy_size: int = 4
    ep_proxy_rank: int = 1

    def __post_init__(self) -> None:
        for name in (
            "tokens",
            "hidden_size",
            "intermediate_size",
            "num_routed_experts",
            "routed_topk",
            "rank",
            "lora_capacity",
            "num_shared_experts",
            "ep_proxy_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.routed_topk > self.num_routed_experts:
            raise ValueError("routed_topk cannot exceed num_routed_experts")
        if self.active_adapters <= 0 or self.active_adapters > self.lora_capacity:
            raise ValueError("active_adapters must be in [1, lora_capacity]")
        if self.num_routed_experts % self.ep_proxy_size:
            raise ValueError("num_routed_experts must divide ep_proxy_size")
        if not 0 <= self.ep_proxy_rank < self.ep_proxy_size:
            raise ValueError("ep_proxy_rank must be in [0, ep_proxy_size)")


CASES: tuple[SharedExpertBenchCase, ...] = (
    SharedExpertBenchCase(
        "shared-smoke-r32",
        "decode",
        4,
        128,
        128,
        8,
        2,
        32,
        1,
        True,
        2,
        1,
        2,
        1,
    ),
    SharedExpertBenchCase(
        "qwen35-decode-t1-r32",
        "decode",
        1,
        2048,
        512,
        256,
        8,
        32,
        1,
        False,
        1,
        1,
    ),
    SharedExpertBenchCase(
        "qwen35-decode-t32-r64-mixed",
        "decode",
        32,
        2048,
        512,
        256,
        8,
        64,
        4,
        True,
        8,
        1,
    ),
    SharedExpertBenchCase(
        "qwen35-decode-t32-r128-two-sink-mixed",
        "decode",
        32,
        2048,
        512,
        256,
        8,
        128,
        4,
        True,
        8,
        2,
    ),
    SharedExpertBenchCase(
        "qwen35-prefill-t256-r32-mixed",
        "prefill",
        256,
        2048,
        512,
        256,
        8,
        32,
        3,
        True,
        8,
        1,
    ),
    SharedExpertBenchCase(
        "qwen35-prefill-t512-r64",
        "prefill",
        512,
        2048,
        512,
        256,
        8,
        64,
        4,
        False,
        8,
        1,
    ),
    SharedExpertBenchCase(
        "qwen35-prefill-t256-r128-two-sink-mixed",
        "prefill",
        256,
        2048,
        512,
        256,
        8,
        128,
        4,
        True,
        8,
        2,
    ),
)
CASES_BY_ID = {case.case_id: case for case in CASES}


@triton.jit
def _shared_swiglu_scale_kernel(
    gate_up_ptr,
    scales_ptr,
    act_ptr,
    total,
    I: tl.constexpr,  # noqa: E741
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    i = offsets % I
    ts = offsets // I
    gate = tl.load(gate_up_ptr + ts * (2 * I) + i, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up_ptr + ts * (2 * I) + I + i, mask=mask, other=0.0).to(
        tl.float32
    )
    scale = tl.load(scales_ptr + ts, mask=mask, other=0.0).to(tl.float32)
    value = gate * tl.sigmoid(gate) * up * scale
    tl.store(act_ptr + ts * I + i, value, mask=mask)


def _invoke_shared_activation(
    gate_up: torch.Tensor, scales: torch.Tensor, activation: torch.Tensor
) -> None:
    total = activation.numel()
    _shared_swiglu_scale_kernel[(triton.cdiv(total, 256),)](
        gate_up,
        scales,
        activation,
        total,
        I=activation.shape[-1],
        S=activation.shape[1],
        BLOCK=256,
        num_warps=4,
    )


def _random_bf16(
    shape: tuple[int, ...], *, generator: torch.Generator, scale: float
) -> torch.Tensor:
    result = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    result.uniform_(-scale, scale, generator=generator)
    return result


def _physical_routed_ids(
    logical_ids: torch.Tensor, case: SharedExpertBenchCase, variant: Variant
) -> torch.Tensor:
    if variant != "fused_per_rank":
        return logical_ids
    local = case.num_routed_experts // case.ep_proxy_size
    return (
        logical_ids
        + torch.div(logical_ids, local, rounding_mode="floor") * case.num_shared_experts
    )


def _shared_physical_ids(
    case: SharedExpertBenchCase, variant: Variant, device: torch.device
) -> torch.Tensor:
    slots = torch.arange(case.num_shared_experts, dtype=torch.int32, device=device)
    if variant == "fused_global":
        return slots + case.num_routed_experts
    if variant == "fused_per_rank":
        local = case.num_routed_experts // case.ep_proxy_size
        return slots + case.ep_proxy_rank * (local + case.num_shared_experts) + local
    raise ValueError(f"{variant} has no fused shared IDs")


def _make_physical_weights(
    routed_w13: torch.Tensor,
    routed_w2: torch.Tensor,
    shared_w13: torch.Tensor,
    shared_w2: torch.Tensor,
    case: SharedExpertBenchCase,
    variant: Variant,
) -> tuple[torch.Tensor, torch.Tensor]:
    if variant.startswith("separate"):
        return routed_w13, routed_w2
    if variant == "fused_global":
        return (
            torch.cat((routed_w13, shared_w13), dim=0).contiguous(),
            torch.cat((routed_w2, shared_w2), dim=0).contiguous(),
        )

    local = case.num_routed_experts // case.ep_proxy_size
    physical_count = (
        case.num_routed_experts + case.ep_proxy_size * case.num_shared_experts
    )
    w13 = torch.empty(
        (physical_count, *routed_w13.shape[1:]),
        dtype=routed_w13.dtype,
        device=routed_w13.device,
    )
    w2 = torch.empty(
        (physical_count, *routed_w2.shape[1:]),
        dtype=routed_w2.dtype,
        device=routed_w2.device,
    )
    for rank in range(case.ep_proxy_size):
        logical = slice(rank * local, (rank + 1) * local)
        physical = rank * (local + case.num_shared_experts)
        w13[physical : physical + local].copy_(routed_w13[logical])
        w2[physical : physical + local].copy_(routed_w2[logical])
        w13[physical + local : physical + local + case.num_shared_experts].copy_(
            shared_w13
        )
        w2[physical + local : physical + local + case.num_shared_experts].copy_(
            shared_w2
        )
    return w13, w2


@dataclass(slots=True)
class SharedExpertFixture:
    case: SharedExpertBenchCase
    variant: Variant
    hidden_seed: torch.Tensor
    hidden_work: torch.Tensor
    logical_topk_ids: torch.Tensor
    topk_output: object
    token_lora_mapping: torch.Tensor
    routed_w13: torch.Tensor
    routed_w2: torch.Tensor
    shared_w13: torch.Tensor
    shared_w2: torch.Tensor
    shared_w2_combined: torch.Tensor
    shared_scales: torch.Tensor
    lora_weights: tuple[torch.Tensor, ...]
    runner_config: object
    quant_info: object
    base: object
    lora_info: object
    shared_gate_up: torch.Tensor
    shared_activation: torch.Tensor
    shared_output: torch.Tensor
    final_output: torch.Tensor
    side_stream: torch.cuda.Stream | None
    shared_ready: torch.cuda.Event | None
    last_output: torch.Tensor | None = None

    def reset(self) -> None:
        if self.hidden_work.shape != self.hidden_seed.shape:
            self.hidden_work = self.hidden_seed.clone()
        else:
            self.hidden_work.copy_(self.hidden_seed)

    def _dispatch(self):
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )

        return StandardDispatchOutput(self.hidden_work, None, self.topk_output)

    def _shared_forward(self) -> None:
        # Flatten S independent gate/up projections into one GEMM.  The W2
        # matrix concatenates the slot input axes, so its one GEMM performs the
        # sum over shared/sink experts.
        torch.mm(
            self.hidden_work,
            self.shared_w13.reshape(-1, self.case.hidden_size).t(),
            out=self.shared_gate_up.view(self.case.tokens, -1),
        )
        _invoke_shared_activation(
            self.shared_gate_up, self.shared_scales, self.shared_activation
        )
        torch.mm(
            self.shared_activation.view(self.case.tokens, -1),
            self.shared_w2_combined.t(),
            out=self.shared_output,
        )

    def _routed_forward(self) -> torch.Tensor:
        from sglang.srt.lora.sgl_lora.moe_lora_runner import run_sgl_lora_moe

        return run_sgl_lora_moe(
            self._dispatch(),
            self.quant_info,
            self.runner_config,
            self.lora_info,
            self.base,
            two_stream_enabled=False,
        ).hidden_states

    def invoke(self) -> None:
        if self.variant == "separate_serial":
            routed = self._routed_forward()
            self._shared_forward()
            torch.add(routed, self.shared_output, out=self.final_output)
            self.last_output = self.final_output
            return
        if self.variant == "separate_overlap":
            assert self.side_stream is not None and self.shared_ready is not None
            main = torch.cuda.current_stream()
            self.side_stream.wait_stream(main)
            with torch.cuda.stream(self.side_stream):
                self._shared_forward()
                self.shared_ready.record()
            routed = self._routed_forward()
            main.wait_event(self.shared_ready)
            torch.add(routed, self.shared_output, out=self.final_output)
            self.last_output = self.final_output
            return
        self.last_output = self._routed_forward()


def _build_fixture(
    case: SharedExpertBenchCase, variant: Variant
) -> SharedExpertFixture:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.lora.lora_moe_runners import LoRAInfo
    from sglang.srt.lora.sgl_lora.base_gemm import resolve_base_gemm
    from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo
    from sglang.srt.lora.sgl_lora.shared_experts import (
        MoeLoraExpertTopology,
        build_routed_expert_id_map,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260722)
    hidden = _random_bf16(
        (case.tokens, case.hidden_size), generator=generator, scale=0.2
    )
    tokens = torch.arange(case.tokens, dtype=torch.int32, device=device)
    slots = torch.arange(case.routed_topk, dtype=torch.int32, device=device)
    logical_ids = (tokens[:, None] * 13 + slots[None, :] * 7) % case.num_routed_experts
    routed_weights = torch.rand(
        (case.tokens, case.routed_topk),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    routed_weights /= routed_weights.sum(dim=1, keepdim=True)
    shared_scales = (
        torch.linspace(
            0.75,
            1.25,
            case.num_shared_experts,
            dtype=torch.float32,
            device=device,
        )
        .expand(case.tokens, -1)
        .contiguous()
    )

    identities = list(range(case.active_adapters))
    if case.include_base_rows:
        identities.append(-1)
    identity_tensor = torch.tensor(identities, dtype=torch.int32, device=device)
    token_lora_mapping = identity_tensor[
        torch.arange(case.tokens, device=device) % len(identities)
    ]

    routed_w13 = _random_bf16(
        (
            case.num_routed_experts,
            2 * case.intermediate_size,
            case.hidden_size,
        ),
        generator=generator,
        scale=0.02,
    )
    routed_w2 = _random_bf16(
        (
            case.num_routed_experts,
            case.hidden_size,
            case.intermediate_size,
        ),
        generator=generator,
        scale=0.02,
    )
    shared_w13 = _random_bf16(
        (
            case.num_shared_experts,
            2 * case.intermediate_size,
            case.hidden_size,
        ),
        generator=generator,
        scale=0.02,
    )
    shared_w2 = _random_bf16(
        (
            case.num_shared_experts,
            case.hidden_size,
            case.intermediate_size,
        ),
        generator=generator,
        scale=0.02,
    )
    # The S shared/sink outputs are reduced by one GEMM.  Pack this static
    # matrix once with the fixture; doing ``permute(...).reshape(...)`` in the
    # forward path copies for S > 1 and would unfairly charge layout packing to
    # the conventional separate-expert variants.
    shared_w2_combined = (
        shared_w2.permute(1, 0, 2)
        .contiguous()
        .view(case.hidden_size, case.num_shared_experts * case.intermediate_size)
    )
    physical_w13, physical_w2 = _make_physical_weights(
        routed_w13, routed_w2, shared_w13, shared_w2, case, variant
    )

    topk_ids = _physical_routed_ids(logical_ids, case, variant)
    topk_weights = routed_weights
    fused_shared = 0
    if variant.startswith("fused"):
        fused_shared = case.num_shared_experts
        shared_ids = _shared_physical_ids(case, variant, device).expand(case.tokens, -1)
        topk_ids = torch.cat((topk_ids, shared_ids), dim=1)
        # Base finalize applies routed_scaling_factor to the whole reduction.
        # Inverse-compensate shared weights so their net scale stays exactly
        # the model's shared gate, independent of routed scaling.
        topk_weights = torch.cat((routed_weights, shared_scales / 1.7), dim=1)
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()
    topk_output = StandardTopKOutput(
        topk_weights,
        topk_ids,
        torch.empty(0, dtype=torch.float32, device=device),
    )

    gate_a = _random_bf16(
        (
            case.lora_capacity,
            case.num_routed_experts,
            2 * case.rank,
            case.hidden_size,
        ),
        generator=generator,
        scale=0.01,
    )
    gate_b = _random_bf16(
        (
            case.lora_capacity,
            case.num_routed_experts,
            2 * case.intermediate_size,
            case.rank,
        ),
        generator=generator,
        scale=0.01,
    )
    down_a = _random_bf16(
        (
            case.lora_capacity,
            case.num_routed_experts,
            case.rank,
            case.intermediate_size,
        ),
        generator=generator,
        scale=0.01,
    )
    down_b = _random_bf16(
        (
            case.lora_capacity,
            case.num_routed_experts,
            case.hidden_size,
            case.rank,
        ),
        generator=generator,
        scale=0.01,
    )

    num_physical = physical_w13.shape[0]
    total_topk = case.routed_topk + fused_shared
    config = MoeRunnerConfig(
        num_experts=num_physical,
        num_local_experts=num_physical,
        hidden_size=case.hidden_size,
        intermediate_size_per_partition=case.intermediate_size,
        top_k=total_topk,
        activation="silu",
        is_gated=True,
        inplace=False,
        no_combine=False,
        routed_scaling_factor=1.7,
        num_fused_shared_experts=fused_shared,
        gate_up_interleaved=False,
    )
    quant_info = SglLoraBf16QuantInfo(
        w13_weight=physical_w13,
        w2_weight=physical_w2,
        num_local_experts=num_physical,
        intermediate_size=case.intermediate_size,
        hidden_size=case.hidden_size,
    )
    base = resolve_base_gemm(quant_info, config)
    if variant == "fused_per_rank":
        topology = MoeLoraExpertTopology(
            num_routed_experts=case.num_routed_experts,
            num_fused_shared_experts=case.num_shared_experts,
            ep_size=case.ep_proxy_size,
            ep_rank=case.ep_proxy_rank,
            id_layout="global_per_rank_shared",
            factor_domain="global",
        )
        base.lora_expert_id_maps[case.num_routed_experts] = build_routed_expert_id_map(
            topology, device=device
        )

    segment_indptr = torch.arange(case.tokens + 1, dtype=torch.int32, device=device)
    lora_info = LoRAInfo(
        gate_up_lora_a_weights=gate_a,
        gate_up_lora_b_weights=gate_b,
        down_lora_a_weights=down_a,
        down_lora_b_weights=down_b,
        seg_indptr=segment_indptr,
        req_to_lora=token_lora_mapping.clone(),
        lora_ranks=torch.full(
            (case.lora_capacity,), case.rank, dtype=torch.int32, device=device
        ),
        adapter_enabled=torch.arange(
            case.lora_capacity, dtype=torch.int32, device=device
        )
        < case.active_adapters,
        token_lora_mapping=token_lora_mapping,
        max_lora_rank=case.rank,
        num_experts=case.num_routed_experts,
        has_active_lora=True,
        experts_shared_outer_loras=False,
        tp_size=1,
        tp_rank=0,
        hidden_size=case.hidden_size,
        lora_use_virtual_experts=True,
    )

    shared_gate_up = torch.empty(
        (case.tokens, case.num_shared_experts, 2 * case.intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    shared_activation = torch.empty(
        (case.tokens, case.num_shared_experts, case.intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    shared_output = torch.empty(
        (case.tokens, case.hidden_size), dtype=torch.bfloat16, device=device
    )
    final_output = torch.empty_like(shared_output)
    side_stream = (
        torch.cuda.Stream(priority=0) if variant == "separate_overlap" else None
    )
    shared_ready = torch.cuda.Event() if variant == "separate_overlap" else None
    return SharedExpertFixture(
        case,
        variant,
        hidden,
        hidden.clone(),
        logical_ids,
        topk_output,
        token_lora_mapping,
        routed_w13,
        routed_w2,
        shared_w13,
        shared_w2,
        shared_w2_combined,
        shared_scales,
        (gate_a, gate_b, down_a, down_b),
        config,
        quant_info,
        base,
        lora_info,
        shared_gate_up,
        shared_activation,
        shared_output,
        final_output,
        side_stream,
        shared_ready,
    )


@torch.no_grad()
def _independent_oracle(
    fixture: SharedExpertFixture, chunk_size: int = 16
) -> torch.Tensor:
    """Logical PyTorch oracle; no production physical IDs or route plan."""

    case = fixture.case
    gate_a, gate_b, down_a, down_b = fixture.lora_weights
    result = torch.zeros(
        (case.tokens, case.hidden_size), dtype=torch.float32, device="cuda"
    )
    logical_ids = fixture.logical_topk_ids.long()
    routed_weights = fixture.topk_output.topk_weights[:, : case.routed_topk]
    mapping = fixture.token_lora_mapping

    for start in range(0, case.tokens, chunk_size):
        stop = min(start + chunk_size, case.tokens)
        x = fixture.hidden_seed[start:stop]
        ids = logical_ids[start:stop]
        adapters = mapping[start:stop]
        active = adapters >= 0
        routed_sum = torch.zeros(
            (stop - start, case.hidden_size), dtype=torch.float32, device="cuda"
        )
        for k in range(case.routed_topk):
            expert = ids[:, k]
            w13 = fixture.routed_w13[expert]
            gate_up = torch.bmm(w13, x.unsqueeze(-1)).squeeze(-1)
            if bool(active.any()):
                rows = torch.nonzero(active, as_tuple=False).flatten()
                adapter = adapters[rows].long()
                expert_active = expert[rows]
                a_rank = torch.bmm(
                    gate_a[adapter, expert_active], x[rows].unsqueeze(-1)
                ).squeeze(-1)
                rank = case.rank
                gate_delta = torch.bmm(
                    gate_b[adapter, expert_active, : case.intermediate_size],
                    a_rank[:, :rank].unsqueeze(-1),
                ).squeeze(-1)
                up_delta = torch.bmm(
                    gate_b[adapter, expert_active, case.intermediate_size :],
                    a_rank[:, rank:].unsqueeze(-1),
                ).squeeze(-1)
                gate_up[rows, : case.intermediate_size] += gate_delta
                gate_up[rows, case.intermediate_size :] += up_delta
            gate, up = gate_up.split(case.intermediate_size, dim=-1)
            activation = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                torch.bfloat16
            )
            pair = torch.bmm(
                fixture.routed_w2[expert], activation.unsqueeze(-1)
            ).squeeze(-1)
            if bool(active.any()):
                rows = torch.nonzero(active, as_tuple=False).flatten()
                adapter = adapters[rows].long()
                expert_active = expert[rows]
                rank_value = torch.bmm(
                    down_a[adapter, expert_active],
                    activation[rows].unsqueeze(-1),
                ).squeeze(-1)
                pair[rows] += torch.bmm(
                    down_b[adapter, expert_active], rank_value.unsqueeze(-1)
                ).squeeze(-1)
            weight = routed_weights[start:stop, k].float() * 1.7
            routed_sum += pair.float() * weight[:, None]

        shared_sum = torch.zeros_like(routed_sum)
        for shared in range(case.num_shared_experts):
            gate_up = torch.mm(x, fixture.shared_w13[shared].t())
            gate, up = gate_up.split(case.intermediate_size, dim=-1)
            activation = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                torch.bfloat16
            )
            shared_value = torch.mm(activation, fixture.shared_w2[shared].t())
            shared_sum += (
                shared_value.float() * fixture.shared_scales[start:stop, shared, None]
            )
        result[start:stop] = routed_sum + shared_sum
    return result.to(torch.bfloat16)


def _check(fixture: SharedExpertFixture) -> dict[str, object]:
    fixture.reset()
    fixture.invoke()
    torch.cuda.synchronize()
    assert fixture.last_output is not None
    got = fixture.last_output.clone()
    oracle = _independent_oracle(fixture)
    diff = (got.float() - oracle.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        got.float().flatten(), oracle.float().flatten(), dim=0
    ).item()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    finite = bool(torch.isfinite(got).all())
    passed = finite and cosine >= 0.995 and max_abs <= 2.0
    if not passed:
        raise AssertionError(
            f"shared-expert oracle mismatch: max_abs={max_abs}, "
            f"mean_abs={mean_abs}, cosine={cosine}, finite={finite}"
        )
    return {
        "passed": True,
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
        "oracle": "logical_chunked_pytorch_bmm_fp32_reduction",
        "oracle_uses_physical_ids": False,
        "shared_slots_receive_lora": False,
    }


def _git_value(args: list[str], fallback: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def _environment(args: argparse.Namespace) -> dict[str, object]:
    cli = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "git_revision": os.getenv("SGLANG_LORA_BENCH_REVISION")
        or _git_value(["rev-parse", "HEAD"], "unknown"),
        "git_dirty": bool(_git_value(["status", "--porcelain"], "unknown")),
        "cli": cli,
    }


def _result(
    fixture: SharedExpertFixture,
    run_config: RunConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    # Compile every kernel and establish side-stream/event ownership before
    # correctness, capture, or timing.
    for _ in range(3):
        fixture.reset()
        fixture.invoke()
    torch.cuda.synchronize()
    correctness = _check(fixture)
    fixture.reset()
    prepared = make_batch(
        fixture.invoke,
        execution=run_config.execution,
        inner_iterations=run_config.inner_iterations,
    )
    timing = None
    if run_config.mode == "time":
        timing = asdict(
            time_cuda_events(
                prepared.run,
                launches_per_batch=prepared.launches_per_batch,
                warmup=run_config.warmup,
                samples=run_config.samples,
                before_sample=fixture.reset,
            )
        )
    else:
        label = (
            f"sgl_lora_moe::SH::{fixture.variant}::{fixture.case.case_id}::"
            f"{run_config.execution}"
        )
        with cuda_profile_range(label):
            for _ in range(run_config.profile_iterations):
                fixture.reset()
                prepared.run()
        torch.cuda.synchronize()

    case = fixture.case
    physical_count = fixture.quant_info.num_local_experts
    return {
        "schema_version": 1,
        "scope": "SH/local_M0",
        "case": asdict(case),
        "variant": fixture.variant,
        "execution": run_config.execution,
        "correctness": correctness,
        "timing": timing,
        "contract": {
            "logical_routed_experts": case.num_routed_experts,
            "physical_experts": physical_count,
            "routed_topk": case.routed_topk,
            "shared_topk": (
                0 if fixture.variant.startswith("separate") else case.num_shared_experts
            ),
            "shared_expert_form": (
                "conventional_separate"
                if fixture.variant.startswith("separate")
                else (
                    "fused_global_slots"
                    if fixture.variant == "fused_global"
                    else "fused_per_rank_interleaved_slots"
                )
            ),
            "shared_execution": (
                "overlap"
                if fixture.variant == "separate_overlap"
                else "serial_or_in_provider"
            ),
            "shared_scaling": "inverse_routed_scale_in_fused_topk_then_post_scale",
            "routed_scaling_factor": 1.7,
            "lora_target_scope": "routed_experts_only",
            "expert_id_map": fixture.variant == "fused_per_rank",
            "ep_proxy": fixture.variant == "fused_per_rank",
            "ep_proxy_is_not_d0": fixture.variant == "fused_per_rank",
            "cuda_graph_replay": run_config.execution == "cuda_graph",
            "mixed_adapter_base_rows": case.include_base_rows,
        },
        "environment": _environment(args),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--case-id", default="shared-smoke-r32")
    parser.add_argument("--variant", choices=VARIANTS, default="fused_global")
    parser.add_argument("--device", choices=("auto", "h200", "gb300"), default="auto")
    parser.add_argument(
        "--execution", choices=("eager", "cuda_graph"), default="cuda_graph"
    )
    parser.add_argument("--mode", choices=("time", "nsys", "ncu"), default="time")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--inner-iterations", type=int, default=1)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_cases:
        for case in CASES:
            print(
                f"{case.case_id:<48} {case.phase:<7} T={case.tokens:<4} "
                f"H={case.hidden_size:<4} I={case.intermediate_size:<4} "
                f"E={case.num_routed_experts:<3} K={case.routed_topk} "
                f"R={case.rank:<3} shared={case.num_shared_experts} "
                f"L={case.active_adapters}+{int(case.include_base_rows)}/{case.lora_capacity}"
            )
        return
    try:
        case = CASES_BY_ID[args.case_id]
    except KeyError as exc:
        raise ValueError(f"unknown case {args.case_id!r}") from exc
    _detect_device(args.device)
    _ensure_benchmark_server_args()
    run_config = RunConfig(
        mode=args.mode,
        execution=args.execution,
        warmup=args.warmup,
        samples=args.samples,
        inner_iterations=args.inner_iterations,
        profile_iterations=args.profile_iterations,
    )
    from benchmark.kernels.lora_moe.bench_moe_pipeline import _single_rank_runtime

    with _single_rank_runtime():
        fixture = _build_fixture(case, args.variant)
        result = _result(fixture, run_config, args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
