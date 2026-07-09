"""Pluggable base-GEMM implementations for the unified MoE-LoRA runner.

``MoeLoraBaseGemm`` is the per-quant seam: the unified runner drives
prepare (S1: permute + optional input quant) -> gateup (S2) -> act_with_delta
(S3, the LoRA injection point) -> down (S4) -> finalize (S5), all launched from
Python so the cross-stream LoRA event join lands between S2 and S3.

Phase 1a implements the standard-layout BF16 path with DeepGEMM masked grouped
GEMM. FP8, NVFP4, and W4A16 providers will plug into the same stage boundary in
later commits.
All buffers are allocated by the RUNNER on the main stream (cuda-graph
allocator safety) — implementations must not allocate on the side stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.sgl_lora.quant_info import (
    SglLoraBf16QuantInfo,
    SglLoraQuantInfo,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


class BaseGemmWorkspace(msgspec.Struct, kw_only=True):
    """Per-forward tensors produced by prepare() and consumed by S2-S5.

    All in the DeepGEMM masked layout: ``[E_local, m_max, ·]`` with
    ``src2dst[t * top_k + k] = expert * m_max + offset`` and validity carried
    by ``topk_ids >= 0`` (post_reorder convention).
    """

    hidden_permuted: torch.Tensor  # [E_local, m_max, hidden] (quant dtype per impl)
    masked_m: torch.Tensor  # [E_local] int32
    expected_m: int
    src2dst: torch.Tensor  # [num_tokens * top_k] int32
    m_max: int


class MoeLoraBaseGemm:
    """Interface. One instance per (layer, quant type), bound to quant_info."""

    def prepare(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, top_k: int
    ) -> BaseGemmWorkspace:
        raise NotImplementedError

    def gateup(self, ws: BaseGemmWorkspace, out: torch.Tensor) -> None:
        raise NotImplementedError

    def act_with_delta(
        self,
        ws: BaseGemmWorkspace,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def down(
        self, ws: BaseGemmWorkspace, act_out: torch.Tensor, out: torch.Tensor
    ) -> None:
        raise NotImplementedError

    def finalize(
        self,
        ws: BaseGemmWorkspace,
        down_out: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    # Buffer shape helpers so the runner owns every allocation.
    def gateup_out_shape(self, ws: BaseGemmWorkspace) -> tuple[int, ...]:
        raise NotImplementedError

    def act_out_shape(self, ws: BaseGemmWorkspace) -> tuple[int, ...]:
        raise NotImplementedError

    def down_out_shape(self, ws: BaseGemmWorkspace) -> tuple[int, ...]:
        raise NotImplementedError


class DeepGemmBf16BaseGemm(MoeLoraBaseGemm):
    """bf16 base GEMMs via DeepGEMM masked grouped GEMM.

    Reuses the exact preprocessing (``moe_ep_deepgemm_preprocess``) and
    finalize (``post_reorder_deepgemm``) the stock deep_gemm MoE runner
    uses for standard dispatch — only the activation between the GEMMs is
    replaced by the LoRA-aware S3 kernel.
    """

    def __init__(self, quant_info: SglLoraBf16QuantInfo, config: MoeRunnerConfig):
        self.quant_info = quant_info
        self.config = config
        # gpt-oss-class layouts arrive here via constexpr flags (design §3).
        self.gate_first = True
        self.interleaved = False

        # Bind callees once (this instance is constructed at LoRA-attach time
        # and lives for the layer's lifetime — no per-forward imports).
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            moe_ep_deepgemm_preprocess,
            post_reorder_deepgemm,
        )
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.lora.sgl_lora.triton_ops import silu_mul_delta_masked

        self._grouped_gemm_bf16_masked = deep_gemm_wrapper.grouped_gemm_nt_bf16_masked
        self._preprocess = moe_ep_deepgemm_preprocess
        self._post_reorder = post_reorder_deepgemm
        self._act_kernel = silu_mul_delta_masked

    def prepare(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, top_k: int
    ) -> BaseGemmWorkspace:
        masked_m, expected_m, src2dst, hidden_permuted, _ = self._preprocess(
            topk_ids,
            self.quant_info.num_local_experts,
            hidden_states,
            top_k,
            None,
            output_dtype=torch.bfloat16,
        )
        return BaseGemmWorkspace(
            hidden_permuted=hidden_permuted,
            masked_m=masked_m,
            expected_m=expected_m,
            src2dst=src2dst,
            m_max=hidden_permuted.shape[1],
        )

    def gateup(self, ws: BaseGemmWorkspace, out: torch.Tensor) -> None:
        self._grouped_gemm_bf16_masked(
            ws.hidden_permuted,
            self.quant_info.w13_weight,
            out,
            ws.masked_m,
            ws.expected_m,
        )

    def act_with_delta(
        self,
        ws: BaseGemmWorkspace,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
    ) -> None:
        self._act_kernel(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            ws.src2dst,
            topk_ids,
            gate_first=self.gate_first,
            interleaved=self.interleaved,
        )

    def down(
        self, ws: BaseGemmWorkspace, act_out: torch.Tensor, out: torch.Tensor
    ) -> None:
        self._grouped_gemm_bf16_masked(
            act_out,
            self.quant_info.w2_weight,
            out,
            ws.masked_m,
            ws.expected_m,
        )

    def finalize(
        self,
        ws: BaseGemmWorkspace,
        down_out: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
    ) -> None:
        num_tokens, hidden = output.shape
        self._post_reorder(
            down_out.view(-1, hidden),
            output,
            ws.src2dst,
            topk_ids,
            topk_weights,
            topk_ids.shape[1],
            num_tokens,
            hidden,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
        )

    def gateup_out_shape(self, ws: BaseGemmWorkspace) -> tuple[int, ...]:
        E, m_max = self.quant_info.num_local_experts, ws.m_max
        return (E, m_max, 2 * self.quant_info.intermediate_size)

    def act_out_shape(self, ws: BaseGemmWorkspace) -> tuple[int, ...]:
        E, m_max = self.quant_info.num_local_experts, ws.m_max
        return (E, m_max, self.quant_info.intermediate_size)

    def down_out_shape(self, ws: BaseGemmWorkspace) -> tuple[int, ...]:
        E, m_max = self.quant_info.num_local_experts, ws.m_max
        return (E, m_max, self.quant_info.hidden_size)


def resolve_base_gemm(
    quant_info: SglLoraQuantInfo, config: MoeRunnerConfig
) -> MoeLoraBaseGemm:
    if isinstance(quant_info, SglLoraBf16QuantInfo):
        return DeepGemmBf16BaseGemm(quant_info, config)
    raise NotImplementedError(
        f"sgl_lora has no base-GEMM provider for {type(quant_info).__name__}."
    )
