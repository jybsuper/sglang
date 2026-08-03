"""GPU numerical coverage for the production BF16 SGL MoE-LoRA runner.

This test deliberately starts at the real selector/provider/runner boundary.
It does not reuse a benchmark reference kernel: the expected result is an
explicit PyTorch FP32 implementation of per-expert SWIGLU MoE + LoRA.  The
three token-slot patterns cover fully active traffic, mixed active/base
traffic, and a base-only batch that still traverses the unified LoRA-capable
topology.

Shared-outer factors require the serving ``segment_info`` contract and are
covered separately at that boundary.  Supplying invented segment metadata
here would make this focused test validate a synthetic seam instead of the
production contract.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.lora.sgl_lora.execution_plan import ActivationFamily
from sglang.srt.lora.sgl_lora.moe_lora_runner import (
    SglMoeLoraBatch,
    SglMoeLoraRunner,
)
from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo
from sglang.srt.lora.sgl_lora.selector import (
    PER_EXPERT_LAYOUT,
    PolicyInput,
    PolicyMode,
    architecture_for_device,
    resolve_base_gemm_provider,
    select_policy,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="production SGL MoE-LoRA requires CUDA"
)

_EXPERTS = 2
_TOP_K = 2
_HIDDEN = 128
_INTERMEDIATE = 128
_PHYSICAL_RANK = 16
_SLOTS = 2
_ROUTED_SCALING = 0.75


def _rand_bf16(
    shape: tuple[int, ...], *, generator: torch.Generator, scale: float
) -> torch.Tensor:
    return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)


def _make_cpu_tensors(num_tokens: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0x5A17 + num_tokens)
    tensors = {
        "hidden_states": _rand_bf16(
            (num_tokens, _HIDDEN), generator=generator, scale=0.20
        ),
        "w13_weight": _rand_bf16(
            (_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN),
            generator=generator,
            scale=0.08,
        ),
        "w2_weight": _rand_bf16(
            (_EXPERTS, _HIDDEN, _INTERMEDIATE),
            generator=generator,
            scale=0.08,
        ),
        "gate_up_lora_a": _rand_bf16(
            (_SLOTS, _EXPERTS, 2 * _PHYSICAL_RANK, _HIDDEN),
            generator=generator,
            scale=0.15,
        ),
        "gate_up_lora_b": _rand_bf16(
            (_SLOTS, _EXPERTS, 2 * _INTERMEDIATE, _PHYSICAL_RANK),
            generator=generator,
            scale=0.15,
        ),
        "down_lora_a": _rand_bf16(
            (_SLOTS, _EXPERTS, _PHYSICAL_RANK, _INTERMEDIATE),
            generator=generator,
            scale=0.15,
        ),
        "down_lora_b": _rand_bf16(
            (_SLOTS, _EXPERTS, _HIDDEN, _PHYSICAL_RANK),
            generator=generator,
            scale=0.15,
        ),
    }

    alternating_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    tensors["topk_ids"] = alternating_ids.repeat((num_tokens + 1) // 2, 1)[
        :num_tokens
    ].contiguous()
    tensors["topk_weights"] = torch.tensor([[0.65, 0.35]], dtype=torch.float32).repeat(
        num_tokens, 1
    )
    tensors["router_logits"] = torch.zeros((num_tokens, _EXPERTS), dtype=torch.float32)
    tensors["adapter_enabled"] = torch.tensor([1, 0], dtype=torch.int32)
    return tensors


def _token_slots(traffic: str, num_tokens: int) -> torch.Tensor:
    if traffic == "active":
        return torch.zeros(num_tokens, dtype=torch.int32)
    if traffic == "mixed":
        # Batch preparation canonicalizes every inactive resident assignment
        # to -1 before any MoE layer consumes the mapping.
        pattern = torch.tensor([0, -1, -1, 0, -1, -1], dtype=torch.int32)
        return pattern.repeat((num_tokens + pattern.numel() - 1) // pattern.numel())[
            :num_tokens
        ].contiguous()
    if traffic == "base_only":
        # Keep has_active_lora=True at the batch level so selector choice and
        # graph shape stay fixed while every token follows the base sentinel.
        return torch.full((num_tokens,), -1, dtype=torch.int32)
    raise AssertionError(f"unknown traffic pattern {traffic}")


def _fp32_reference(
    tensors: dict[str, torch.Tensor], token_slots: torch.Tensor
) -> torch.Tensor:
    """Independent token/expert reference with no production route helpers."""

    hidden_states = tensors["hidden_states"].float()
    w13_weight = tensors["w13_weight"].float()
    w2_weight = tensors["w2_weight"].float()
    gate_up_lora_a = tensors["gate_up_lora_a"].float()
    gate_up_lora_b = tensors["gate_up_lora_b"].float()
    down_lora_a = tensors["down_lora_a"].float()
    down_lora_b = tensors["down_lora_b"].float()
    topk_ids = tensors["topk_ids"]
    topk_weights = tensors["topk_weights"].float()
    adapter_enabled = tensors["adapter_enabled"]

    output = torch.zeros((hidden_states.shape[0], _HIDDEN), dtype=torch.float32)
    for token_idx, hidden in enumerate(hidden_states):
        slot = int(token_slots[token_idx])
        slot_is_active = slot >= 0 and bool(adapter_enabled[slot])
        for topk_idx in range(_TOP_K):
            expert = int(topk_ids[token_idx, topk_idx])
            gate_up = torch.mv(w13_weight[expert], hidden)
            gate = gate_up[:_INTERMEDIATE]
            up = gate_up[_INTERMEDIATE:]

            if slot_is_active:
                gate_up_a = gate_up_lora_a[slot, expert]
                gate_up_b = gate_up_lora_b[slot, expert]
                gate_rank = torch.mv(gate_up_a[:_PHYSICAL_RANK], hidden)
                up_rank = torch.mv(gate_up_a[_PHYSICAL_RANK:], hidden)
                gate = gate + torch.mv(gate_up_b[:_INTERMEDIATE], gate_rank)
                up = up + torch.mv(gate_up_b[_INTERMEDIATE:], up_rank)

            activation = F.silu(gate) * up
            pair_output = torch.mv(w2_weight[expert], activation)
            if slot_is_active:
                down_rank = torch.mv(down_lora_a[slot, expert], activation)
                pair_output = pair_output + torch.mv(
                    down_lora_b[slot, expert], down_rank
                )

            output[token_idx].add_(
                pair_output, alpha=float(topk_weights[token_idx, topk_idx])
            )

    return output * _ROUTED_SCALING


def _standalone_output_allocation(
    runner: SglMoeLoraRunner,
    *,
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Match eager output geometry without requiring a serving TP group."""

    return torch.empty(
        (num_tokens, runner.provider.hidden_size), dtype=dtype, device=device
    )


@pytest.mark.parametrize(
    ("mode", "num_tokens"),
    ((PolicyMode.DECODE, 8), (PolicyMode.PREFILL, 64)),
    ids=("decode", "lab-prefill"),
)
def test_selector_chosen_per_expert_swiglu_matches_fp32_reference(
    monkeypatch: pytest.MonkeyPatch, mode: PolicyMode, num_tokens: int
) -> None:
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    device_name = torch.cuda.get_device_name(device)
    try:
        architecture = architecture_for_device(device_name, capability)
    except NotImplementedError:
        pytest.skip(
            f"SGL MoE-LoRA policy does not support {device_name} "
            f"(SM{capability[0]}{capability[1]})"
        )
    base_gemm_provider = resolve_base_gemm_provider("auto", architecture)

    # Serving wraps this allocation in the optional symmetric-memory context,
    # which requires an initialized TP group.  This standalone numerical test
    # has no collective and needs only the identical eager tensor geometry.
    monkeypatch.setattr(
        SglMoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )

    cpu = _make_cpu_tensors(num_tokens)
    gpu = {name: tensor.to(device) for name, tensor in cpu.items()}

    choice = select_policy(
        PolicyInput(
            architecture=architecture,
            base_gemm_provider=base_gemm_provider,
            factor_layout=PER_EXPERT_LAYOUT,
            activation=ActivationFamily.SWIGLU,
            mode=mode,
            num_tokens=num_tokens,
            active_rank=_PHYSICAL_RANK,
        )
    )
    assert choice.provider is not None
    assert choice.plan is not None
    assert choice.launch_config is not None

    provider_cls = SglMoeLoraRunner.select_provider_cls(choice.provider)
    provider = provider_cls(
        SglLoraBf16QuantInfo(
            w13_weight=gpu["w13_weight"],
            w2_weight=gpu["w2_weight"],
            num_local_experts=_EXPERTS,
            intermediate_size=_INTERMEDIATE,
            hidden_size=_HIDDEN,
        )
    )
    runner = SglMoeLoraRunner(
        provider=provider,
        top_k=_TOP_K,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFamily.SWIGLU,
        execution_plan=choice.plan,
        launch_config=choice.launch_config,
    )
    runner.validate_factors(
        gate_up_lora_a=gpu["gate_up_lora_a"],
        gate_up_lora_b=gpu["gate_up_lora_b"],
        down_lora_a=gpu["down_lora_a"],
        down_lora_b=gpu["down_lora_b"],
        factor_layout=PER_EXPERT_LAYOUT,
    )

    dispatch = StandardDispatchOutput(
        hidden_states=gpu["hidden_states"],
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=gpu["topk_weights"],
            topk_ids=gpu["topk_ids"],
            router_logits=gpu["router_logits"],
        ),
    )
    references: dict[str, torch.Tensor] = {}
    for traffic in ("active", "mixed", "base_only"):
        cpu_slots = _token_slots(traffic, num_tokens)
        references[traffic] = _fp32_reference(cpu, cpu_slots)
        batch = SglMoeLoraBatch(
            gate_up_lora_a=gpu["gate_up_lora_a"],
            gate_up_lora_b=gpu["gate_up_lora_b"],
            down_lora_a=gpu["down_lora_a"],
            down_lora_b=gpu["down_lora_b"],
            token_slots=cpu_slots.to(device),
            adapter_enabled=gpu["adapter_enabled"],
            physical_rank=_PHYSICAL_RANK,
            factor_layout=PER_EXPERT_LAYOUT,
            use_cuda_graph=False,
            is_prefill=mode is PolicyMode.PREFILL,
            has_active_lora=True,
        )
        actual = runner.run(dispatch, batch, output_dtype=torch.float32)
        torch.testing.assert_close(
            actual.hidden_states.detach().cpu(),
            references[traffic],
            atol=0.018,
            rtol=0.06,
            msg=f"{choice.key}: {traffic} traffic",
        )

    # Make the test sensitive to accidentally bypassing all LoRA math despite
    # the unavoidable BF16-vs-FP32 comparison tolerance above.
    assert (references["active"] - references["base_only"]).abs().max().item() > 0.02
