"""Pluggable base-GEMM implementations for the unified MoE-LoRA runner.

``MoeLoraBaseGemm`` is the per-quant seam: the unified runner drives
prepare (S1: permute + optional input quant) -> gateup (S2) -> act_with_delta
(S3, the LoRA injection point) -> down (S4) -> finalize (S5), all launched from
Python so the cross-stream LoRA event join lands between S2 and S3.

The semantic pipeline owns LoRA and the final destination.  Each base provider
owns its physical row layout, input quantization, and optional post-activation
quantization.  Those ownership boundaries are recorded in
:class:`MoeLoraProviderContract`; they are deliberately not inferred from a
weight dtype or tensor width.

No provider allocates on the LoRA side stream.  Provider-private quantization
outputs may be allocated while the provider runs on the main stream when the
underlying production primitive has no caller-provided-output form.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import msgspec
import torch

from sglang.srt.lora.sgl_lora.quant_info import (
    SglLoraBf16QuantInfo,
    SglLoraFp8QuantInfo,
    SglLoraMarlinQuantInfo,
    SglLoraNvFp4QuantInfo,
    SglLoraQuantInfo,
)
from sglang.srt.lora.sgl_lora.workspace import (
    MoeLoraWorkspacePlanner,
    estimate_bf16_moe_lora_workspace,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class MoeLoraProviderContract:
    """Stable semantic/ownership description for one physical base provider."""

    key: str
    weight_format: str
    row_domain: Literal["expert_masked", "routed_pair"]
    gate_up_slices: tuple[str, ...]
    gate_first: bool
    interleaved: bool
    activation: Literal["swiglu"]
    gate_up_output_dtype: torch.dtype
    w2_input_dtype: torch.dtype
    w2_scale_format: str | None
    lora_delta_dtype: torch.dtype
    lora_activation_dtype: torch.dtype
    supported_output_dtypes: tuple[torch.dtype, ...]
    packed_topk_policy: Literal["preserve_if_present"] = "preserve_if_present"
    borrowed_buffers: tuple[str, ...] = (
        "hidden_states",
        "topk_ids",
        "topk_weights",
        "packed_topk_ids",
        "base_weights",
    )
    runner_owned_buffers: tuple[str, ...] = (
        "gate_up_output",
        "activation_bf16",
        "activation_lora_input",
        "down_output",
        "final_output",
    )
    provider_owned_buffers: tuple[str, ...] = ()

    def validate_output_dtype(self, dtype: torch.dtype) -> None:
        if dtype not in self.supported_output_dtypes:
            supported = ", ".join(str(item) for item in self.supported_output_dtypes)
            raise ValueError(
                f"{self.key} cannot write sgl_lora output dtype {dtype}; "
                f"supported dtypes: {supported}"
            )


class BaseGemmWorkspace(msgspec.Struct, kw_only=True):
    """Per-forward tensors produced by prepare() and consumed by S2-S5.

    Masked providers use ``[E_local, m_max, ·]`` and
    ``src2dst[t * top_k + k] = expert * m_max + offset``. Routed-pair providers
    use ``[num_tokens * top_k, ·]`` and an identity ``src2dst``. In both row
    domains, validity is carried by ``topk_ids >= 0`` for final reordering.
    """

    hidden_permuted: torch.Tensor  # [E_local, m_max, hidden] (quant dtype per impl)
    masked_m: torch.Tensor  # [E_local] int32
    expected_m: int
    src2dst: torch.Tensor  # [num_tokens * top_k] int32
    m_max: int
    # The canonical routing tensors are borrowed from the dispatcher.  Keeping
    # the optional packed representation here is intentional: providers that
    # do not consume it still preserve its identity for a later injected
    # provider instead of rebuilding (and potentially changing) the route.
    topk_weights: torch.Tensor | None = None
    packed_topk_ids: torch.Tensor | None = None
    hidden_scale: torch.Tensor | None = None
    activation_quant: torch.Tensor | None = None
    activation_scale: torch.Tensor | None = None
    sorted_token_ids: torch.Tensor | None = None
    expert_ids: torch.Tensor | None = None
    num_tokens_post_padded: torch.Tensor | None = None
    moe_block_size: int = 0
    hidden_permuted_owned: bool = True


class MoeLoraBaseGemm:
    """Interface. One instance per (layer, quant type), bound to quant_info."""

    contract: MoeLoraProviderContract
    # Static incoming-physical-ID -> routed-factor-ID maps keyed by the factor
    # expert dimension. Most layouts need no entry; per-rank fused shared
    # slots bind one map at LoRA attach time and reuse it for every forward.
    lora_expert_id_maps: dict[int, torch.Tensor]

    def admit_workspace(
        self,
        *,
        num_tokens: int,
        top_k: int,
        rank: int,
        max_loras: int,
        dtype: torch.dtype,
        device: torch.device,
        capture: bool,
        memory_query_safe: bool,
    ) -> None:
        raise NotImplementedError

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        *,
        topk_weights: torch.Tensor | None = None,
        packed_topk_ids: torch.Tensor | None = None,
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

    def validate_runtime_inputs(
        self,
        hidden_states: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> None:
        """Validate the semantic boundary shared by every shipped provider."""
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                f"{self.contract.key} requires BF16 MoE/LoRA activations, got "
                f"{hidden_states.dtype}"
            )
        self.contract.validate_output_dtype(output_dtype)


class _MaskedBaseGemm(MoeLoraBaseGemm):
    """Common masked-row routing, LoRA activation, finalize, and sizing."""

    def __init__(
        self,
        quant_info: SglLoraQuantInfo,
        config: MoeRunnerConfig,
        workspace_planner: MoeLoraWorkspacePlanner | None,
    ) -> None:
        self.quant_info = quant_info
        self.config = config
        self.workspace_planner = workspace_planner or MoeLoraWorkspacePlanner()
        self.lora_expert_id_maps = {}

        from sglang.kernels.ops.moe.ep_moe_kernels import (
            moe_ep_deepgemm_preprocess,
            post_reorder_deepgemm,
        )
        from sglang.srt.lora.sgl_lora.triton_ops import silu_mul_delta_masked

        self._preprocess = moe_ep_deepgemm_preprocess
        self._post_reorder = post_reorder_deepgemm
        self._act_kernel = silu_mul_delta_masked

    def admit_workspace(
        self,
        *,
        num_tokens: int,
        top_k: int,
        rank: int,
        max_loras: int,
        dtype: torch.dtype,
        device: torch.device,
        capture: bool,
        memory_query_safe: bool,
    ) -> None:
        # The BF16 estimator is a conservative bound for FP8/NVFP4 providers:
        # it prices both provider inputs as BF16 even though their quantized
        # payloads are one byte or less per element.  The explicit provider
        # fields (scales and packed data) are small compared with that margin.
        estimate_element_size = max(dtype.itemsize, torch.bfloat16.itemsize)
        if self.contract.w2_input_dtype in (
            torch.float8_e4m3fn,
            torch.uint8,
        ):
            # Quantized providers keep the BF16 activation bridge and the
            # provider-private W2 quantization live together. Pricing each
            # logical element at four bytes safely covers both plus scales.
            estimate_element_size = max(estimate_element_size, 4)
        estimate = estimate_bf16_moe_lora_workspace(
            num_tokens=num_tokens,
            top_k=top_k,
            hidden_size=self.quant_info.hidden_size,
            intermediate_size=self.quant_info.intermediate_size,
            rank=rank,
            num_local_experts=self.quant_info.num_local_experts,
            max_loras=max_loras,
            element_size=estimate_element_size,
        )
        self.workspace_planner.admit(
            estimate=estimate,
            device=device,
            capture=capture,
            geometry_key=(
                self.contract.key,
                self.quant_info.num_local_experts,
                self.quant_info.hidden_size,
                self.quant_info.intermediate_size,
                rank,
                top_k,
                max_loras,
                dtype,
            ),
            memory_query_safe=memory_query_safe,
        )

    def _prepare_masked(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        *,
        topk_weights: torch.Tensor | None,
        packed_topk_ids: torch.Tensor | None,
        block_shape: tuple[int, int] | None,
        output_dtype: torch.dtype,
        use_mxfp8: bool = False,
    ) -> BaseGemmWorkspace:
        masked_m, expected_m, src2dst, hidden_permuted, hidden_scale = self._preprocess(
            topk_ids,
            self.quant_info.num_local_experts,
            hidden_states,
            top_k,
            list(block_shape) if block_shape is not None else None,
            output_dtype=output_dtype,
            use_mxfp8=use_mxfp8,
        )
        return BaseGemmWorkspace(
            hidden_permuted=hidden_permuted,
            hidden_scale=hidden_scale,
            masked_m=masked_m,
            expected_m=expected_m,
            src2dst=src2dst,
            m_max=hidden_permuted.shape[1],
            topk_weights=topk_weights,
            packed_topk_ids=packed_topk_ids,
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
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
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
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            2 * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, ws: BaseGemmWorkspace) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            self.quant_info.intermediate_size,
        )

    def down_out_shape(self, ws: BaseGemmWorkspace) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            self.quant_info.hidden_size,
        )


class DeepGemmBf16BaseGemm(_MaskedBaseGemm):
    """bf16 base GEMMs via DeepGEMM masked grouped GEMM.

    Reuses the exact preprocessing (``moe_ep_deepgemm_preprocess``) and
    finalize (``post_reorder_deepgemm``) the stock deep_gemm MoE runner
    uses for standard dispatch — only the activation between the GEMMs is
    replaced by the LoRA-aware S3 kernel.
    """

    contract = MoeLoraProviderContract(
        key="deepgemm_bf16",
        weight_format="bf16_e_n_k",
        row_domain="expert_masked",
        gate_up_slices=("gate", "up"),
        gate_first=True,
        interleaved=False,
        activation="swiglu",
        gate_up_output_dtype=torch.bfloat16,
        w2_input_dtype=torch.bfloat16,
        w2_scale_format=None,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        supported_output_dtypes=(torch.bfloat16, torch.float32),
        provider_owned_buffers=(
            "masked_m",
            "src2dst",
            "hidden_permuted_bf16",
        ),
    )

    def __init__(
        self,
        quant_info: SglLoraBf16QuantInfo,
        config: MoeRunnerConfig,
        workspace_planner: MoeLoraWorkspacePlanner | None = None,
    ):
        super().__init__(quant_info, config, workspace_planner)
        # gpt-oss-class layouts arrive here via constexpr flags (design §3).
        self.gate_first = True
        self.interleaved = False

        # Bind callees once (this instance is constructed at LoRA-attach time
        # and lives for the layer's lifetime — no per-forward imports).
        from sglang.srt.layers import deep_gemm_wrapper

        self._grouped_gemm_bf16_masked = deep_gemm_wrapper.grouped_gemm_nt_bf16_masked

    def admit_workspace(
        self,
        *,
        num_tokens: int,
        top_k: int,
        rank: int,
        max_loras: int,
        dtype: torch.dtype,
        device: torch.device,
        capture: bool,
        memory_query_safe: bool,
    ) -> None:
        estimate = estimate_bf16_moe_lora_workspace(
            num_tokens=num_tokens,
            top_k=top_k,
            hidden_size=self.quant_info.hidden_size,
            intermediate_size=self.quant_info.intermediate_size,
            rank=rank,
            num_local_experts=self.quant_info.num_local_experts,
            max_loras=max_loras,
            element_size=dtype.itemsize,
        )
        self.workspace_planner.admit(
            estimate=estimate,
            device=device,
            capture=capture,
            geometry_key=(
                self.quant_info.num_local_experts,
                self.quant_info.hidden_size,
                self.quant_info.intermediate_size,
                rank,
                top_k,
                max_loras,
                dtype,
            ),
            memory_query_safe=memory_query_safe,
        )

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        *,
        topk_weights: torch.Tensor | None = None,
        packed_topk_ids: torch.Tensor | None = None,
    ) -> BaseGemmWorkspace:
        return self._prepare_masked(
            hidden_states,
            topk_ids,
            top_k,
            topk_weights=topk_weights,
            packed_topk_ids=packed_topk_ids,
            block_shape=None,
            output_dtype=torch.bfloat16,
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


class DeepGemmFp8BaseGemm(_MaskedBaseGemm):
    """FP8 W8A8 provider using the production DeepGEMM masked primitives."""

    contract = MoeLoraProviderContract(
        key="deepgemm_fp8_w8a8",
        weight_format="fp8_e4m3_block_scaled_e_n_k",
        row_domain="expert_masked",
        gate_up_slices=("gate", "up"),
        gate_first=True,
        interleaved=False,
        activation="swiglu",
        gate_up_output_dtype=torch.bfloat16,
        w2_input_dtype=torch.float8_e4m3fn,
        w2_scale_format="per_token_group_fp32_or_packed_ue8m0",
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        supported_output_dtypes=(torch.bfloat16, torch.float32),
        provider_owned_buffers=(
            "masked_m",
            "src2dst",
            "hidden_permuted_fp8",
            "hidden_scale",
            "activation_quant_fp8",
            "activation_scale",
        ),
    )

    def __init__(
        self,
        quant_info: SglLoraFp8QuantInfo,
        config: MoeRunnerConfig,
        workspace_planner: MoeLoraWorkspacePlanner | None = None,
    ) -> None:
        super().__init__(quant_info, config, workspace_planner)
        if quant_info.w13_weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "DeepGEMM FP8 provider requires float8_e4m3fn W13 weights, "
                f"got {quant_info.w13_weight.dtype}"
            )
        if quant_info.w2_weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "DeepGEMM FP8 provider requires float8_e4m3fn W2 weights, "
                f"got {quant_info.w2_weight.dtype}"
            )
        if len(quant_info.block_shape) != 2 or min(quant_info.block_shape) <= 0:
            raise ValueError(
                f"invalid FP8 block_shape {quant_info.block_shape}; expected (N, K)"
            )

        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_8bit,
        )
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.moe.moe_runner.deep_gemm import (
            _cast_to_e8m0_with_rounding_up,
        )

        if quant_info.use_mxfp8 and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            raise ValueError("MXFP8 requires DeepGEMM packed UE8M0 scale support")
        self._deep_gemm = deep_gemm_wrapper
        self._cast_to_e8m0 = _cast_to_e8m0_with_rounding_up
        self._quantize_activation = sglang_per_token_group_quant_8bit
        self._grouped_gemm = deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked

    @property
    def _recipe(self) -> tuple[int, int] | None:
        return self.quant_info.block_shape if self.quant_info.use_mxfp8 else None

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        *,
        topk_weights: torch.Tensor | None = None,
        packed_topk_ids: torch.Tensor | None = None,
    ) -> BaseGemmWorkspace:
        ws = self._prepare_masked(
            hidden_states,
            topk_ids,
            top_k,
            topk_weights=topk_weights,
            packed_topk_ids=packed_topk_ids,
            block_shape=self.quant_info.block_shape,
            output_dtype=torch.float8_e4m3fn,
            use_mxfp8=self.quant_info.use_mxfp8,
        )
        assert ws.hidden_scale is not None
        if self._deep_gemm.DEEPGEMM_SCALE_UE8M0:
            if ws.hidden_scale.dtype != torch.int32:
                b, scale_m, scale_k = ws.hidden_scale.shape
                if scale_m % 4 or scale_k % 4:
                    raise ValueError(
                        "DeepGEMM UE8M0 scales require dimensions aligned to 4, "
                        f"got {(b, scale_m, scale_k)}"
                    )
                ws.hidden_scale = self._cast_to_e8m0(ws.hidden_scale)
        elif self._deep_gemm.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            ws.hidden_scale = self._deep_gemm.get_mn_major_tma_aligned_tensor(
                ws.hidden_scale
            )
        return ws

    def gateup(self, ws: BaseGemmWorkspace, out: torch.Tensor) -> None:
        assert ws.hidden_scale is not None
        self._grouped_gemm(
            (ws.hidden_permuted, ws.hidden_scale),
            (self.quant_info.w13_weight, self.quant_info.w13_scale),
            out,
            ws.masked_m,
            ws.expected_m,
            recipe_a=self._recipe,
            recipe_b=self._recipe,
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
        # The LoRA semantic boundary stays BF16.  Only after the exact
        # post-delta activation has been materialized do we create W2's FP8
        # provider input and scale.
        super().act_with_delta(
            ws,
            gateup_out,
            gate_up_delta,
            topk_ids,
            act_out,
            activation_lora_input,
        )
        group_size = self.quant_info.block_shape[1]
        packed_ue8m0 = self._deep_gemm.DEEPGEMM_SCALE_UE8M0
        activation_quant, activation_scale = self._quantize_activation(
            x=act_out,
            group_size=group_size,
            dst_dtype=torch.float8_e4m3fn,
            masked_m=ws.masked_m,
            # Hopper follows the stock route: write ordinary row-major FP32
            # scales, then transform them with DeepGEMM's TMA-layout helper.
            # Blackwell's quantizer writes packed column-major UE8M0 directly.
            column_major_scales=packed_ue8m0,
            scale_tma_aligned=packed_ue8m0,
            scale_ue8m0=packed_ue8m0,
        )
        if self._deep_gemm.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            # Keep this conversion explicit even when the quantizer returned a
            # column-major view. DeepGEMM's helper also applies the exact TMA
            # padding/descriptor layout expected by masked W2 on Hopper.
            activation_scale = self._deep_gemm.get_mn_major_tma_aligned_tensor(
                activation_scale
            )
        ws.activation_quant = activation_quant
        ws.activation_scale = activation_scale

    def down(
        self, ws: BaseGemmWorkspace, act_out: torch.Tensor, out: torch.Tensor
    ) -> None:
        if ws.activation_quant is None or ws.activation_scale is None:
            raise RuntimeError("FP8 W2 input was not produced by act_with_delta")
        self._grouped_gemm(
            (ws.activation_quant, ws.activation_scale),
            (self.quant_info.w2_weight, self.quant_info.w2_scale),
            out,
            ws.masked_m,
            ws.expected_m,
            recipe_a=self._recipe,
            recipe_b=self._recipe,
        )


class CuteDslNvFp4BaseGemm(_MaskedBaseGemm):
    """Native NVFP4 W4A4 provider with an exposed BF16 LoRA seam.

    This provider deliberately targets the canonical CuteDSL masked
    representation.  Standard CuteDSL-v2 and TRT-LLM weights are transformed
    for fused kernels that own activation and A2 quantization; accepting those
    tensors here would silently swap gate/up or misread scale layout.
    """

    contract = MoeLoraProviderContract(
        key="cutedsl_nvfp4_w4a4",
        weight_format="nvfp4_e2m1_group16_canonical",
        row_domain="expert_masked",
        gate_up_slices=("gate", "up"),
        gate_first=True,
        interleaved=False,
        activation="swiglu",
        gate_up_output_dtype=torch.bfloat16,
        w2_input_dtype=torch.uint8,
        w2_scale_format="nvfp4_e4m3_group16_swizzled",
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        supported_output_dtypes=(torch.bfloat16, torch.float32),
        provider_owned_buffers=(
            "masked_m",
            "src2dst",
            "hidden_permuted_nvfp4",
            "hidden_scale",
            "activation_quant_nvfp4",
            "activation_scale",
        ),
    )

    def __init__(
        self,
        quant_info: SglLoraNvFp4QuantInfo,
        config: MoeRunnerConfig,
        workspace_planner: MoeLoraWorkspacePlanner | None = None,
    ) -> None:
        super().__init__(quant_info, config, workspace_planner)
        if not quant_info.gate_first or quant_info.interleaved:
            raise ValueError(
                "injectable NVFP4 requires canonical gate-first, non-interleaved W13"
            )
        if quant_info.w13_weight.dtype != torch.uint8:
            raise TypeError("NVFP4 W13 must contain packed uint8 values")
        if quant_info.w2_weight.dtype != torch.uint8:
            raise TypeError("NVFP4 W2 must contain packed uint8 values")
        if quant_info.w13_blockscale.dtype != torch.float8_e4m3fn:
            raise TypeError("NVFP4 W13 blockscale must be float8_e4m3fn")
        if quant_info.w2_blockscale.dtype != torch.float8_e4m3fn:
            raise TypeError("NVFP4 W2 blockscale must be float8_e4m3fn")

        from flashinfer import scaled_fp4_grouped_quantize
        from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

        from sglang.srt.layers.moe.flashinfer_cutedsl_moe import get_cute_dtype

        self._quantize = scaled_fp4_grouped_quantize
        self._grouped_gemm = grouped_gemm_nt_masked
        self._get_cute_dtype = get_cute_dtype

    def _validate_device(self) -> None:
        device = self.quant_info.w13_weight.device
        if device.type != "cuda":
            raise RuntimeError("NVFP4 CuteDSL provider requires a CUDA device")
        major, _ = torch.cuda.get_device_capability(device)
        if major < 10:
            raise RuntimeError("native NVFP4 W4A4 provider requires Blackwell (SM100+)")

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        *,
        topk_weights: torch.Tensor | None = None,
        packed_topk_ids: torch.Tensor | None = None,
    ) -> BaseGemmWorkspace:
        self._validate_device()
        ws = self._prepare_masked(
            hidden_states,
            topk_ids,
            top_k,
            topk_weights=topk_weights,
            packed_topk_ids=packed_topk_ids,
            block_shape=None,
            output_dtype=torch.bfloat16,
        )
        hidden_bf16 = ws.hidden_permuted
        hidden_quant, hidden_scale = self._quantize(
            hidden_bf16,
            ws.masked_m,
            self.quant_info.w13_input_scale,
        )
        ws.hidden_permuted = hidden_quant
        ws.hidden_scale = hidden_scale
        return ws

    def _run_grouped(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        masked_m: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._grouped_gemm(
            (activation, activation_scale),
            (weight.permute(1, 2, 0), weight_scale),
            out.permute(1, 2, 0),
            masked_m,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            alpha=alpha.view(1, 1, self.quant_info.num_local_experts),
            alpha_dtype=self._get_cute_dtype(alpha),
        )

    def gateup(self, ws: BaseGemmWorkspace, out: torch.Tensor) -> None:
        assert ws.hidden_scale is not None
        self._run_grouped(
            ws.hidden_permuted,
            ws.hidden_scale,
            self.quant_info.w13_weight,
            self.quant_info.w13_blockscale,
            self.quant_info.w13_alpha,
            ws.masked_m,
            out,
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
        super().act_with_delta(
            ws,
            gateup_out,
            gate_up_delta,
            topk_ids,
            act_out,
            activation_lora_input,
        )
        activation_quant, activation_scale = self._quantize(
            act_out,
            ws.masked_m,
            self.quant_info.w2_input_scale,
        )
        ws.activation_quant = activation_quant
        ws.activation_scale = activation_scale

    def down(
        self, ws: BaseGemmWorkspace, act_out: torch.Tensor, out: torch.Tensor
    ) -> None:
        if ws.activation_quant is None or ws.activation_scale is None:
            raise RuntimeError("NVFP4 W2 input was not produced by act_with_delta")
        self._run_grouped(
            ws.activation_quant,
            ws.activation_scale,
            self.quant_info.w2_weight,
            self.quant_info.w2_blockscale,
            self.quant_info.w2_alpha,
            ws.masked_m,
            out,
        )


class MarlinW4A16BaseGemm(MoeLoraBaseGemm):
    """Marlin W4A16 provider in canonical routed-pair row order."""

    contract = MoeLoraProviderContract(
        key="marlin_w4a16",
        weight_format="marlin_w4_packed",
        row_domain="routed_pair",
        gate_up_slices=("gate", "up"),
        gate_first=True,
        interleaved=False,
        activation="swiglu",
        gate_up_output_dtype=torch.bfloat16,
        w2_input_dtype=torch.bfloat16,
        w2_scale_format="marlin_group_scale",
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        supported_output_dtypes=(torch.bfloat16, torch.float32),
        provider_owned_buffers=(
            "src2dst_identity",
            "sorted_token_ids",
            "expert_ids",
            "num_tokens_post_padded",
            "marlin_workspace",
            "marlin_internal_scratch",
        ),
    )

    def __init__(
        self,
        quant_info: SglLoraMarlinQuantInfo,
        config: MoeRunnerConfig,
        workspace_planner: MoeLoraWorkspacePlanner | None = None,
    ) -> None:
        self.quant_info = quant_info
        self.config = config
        self.workspace_planner = workspace_planner or MoeLoraWorkspacePlanner()
        self.lora_expert_id_maps = {}
        if quant_info.weight_bits != 4:
            raise ValueError(
                f"sgl_lora Marlin lane is W4A16; got {quant_info.weight_bits} bits"
            )
        if quant_info.expert_map is not None:
            raise NotImplementedError(
                "sgl_lora Marlin provider does not yet accept global-ID expert_map; "
                "dispatch local expert IDs before entering the provider"
            )

        from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            get_scalar_type,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            moe_align_block_size,
        )
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
        from sglang.srt.lora.sgl_lora.triton_ops import silu_mul_delta_masked

        self._gemm = moe_wna16_marlin_gemm
        self._post_reorder = post_reorder_deepgemm
        self._get_scalar_type = get_scalar_type
        self._align = moe_align_block_size
        self._make_workspace = marlin_make_workspace
        self._act_kernel = silu_mul_delta_masked
        self._workspace: torch.Tensor | None = (
            self._make_workspace(quant_info.w13_weight.device, max_blocks_per_sm=4)
            if quant_info.w13_weight.device.type == "cuda"
            else None
        )

    def _get_workspace(self, device: torch.device) -> torch.Tensor:
        if self._workspace is None or self._workspace.device != device:
            self._workspace = self._make_workspace(device, max_blocks_per_sm=4)
        return self._workspace

    def admit_workspace(
        self,
        *,
        num_tokens: int,
        top_k: int,
        rank: int,
        max_loras: int,
        dtype: torch.dtype,
        device: torch.device,
        capture: bool,
        memory_query_safe: bool,
    ) -> None:
        estimate = estimate_bf16_moe_lora_workspace(
            num_tokens=num_tokens,
            top_k=top_k,
            hidden_size=self.quant_info.hidden_size,
            intermediate_size=self.quant_info.intermediate_size,
            rank=rank,
            num_local_experts=self.quant_info.num_local_experts,
            max_loras=max_loras,
            element_size=max(dtype.itemsize, torch.bfloat16.itemsize),
        )
        self.workspace_planner.admit(
            estimate=estimate,
            device=device,
            capture=capture,
            geometry_key=(
                self.contract.key,
                self.quant_info.num_local_experts,
                self.quant_info.hidden_size,
                self.quant_info.intermediate_size,
                rank,
                top_k,
                max_loras,
                dtype,
            ),
            memory_query_safe=memory_query_safe,
        )

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        *,
        topk_weights: torch.Tensor | None = None,
        packed_topk_ids: torch.Tensor | None = None,
    ) -> BaseGemmWorkspace:
        if hidden_states.device.type != "cuda":
            raise RuntimeError("Marlin W4A16 provider requires a CUDA device")
        major, _ = torch.cuda.get_device_capability(hidden_states.device)
        if major < 9:
            raise RuntimeError("sgl_lora Marlin provider requires SM90 or newer")
        if topk_weights is None:
            raise ValueError("Marlin provider requires topk_weights during prepare")
        num_tokens = hidden_states.shape[0]
        for block_size_m in (8, 16, 32, 48, 64):
            if (
                num_tokens * top_k / self.quant_info.num_local_experts / block_size_m
                < 0.9
            ):
                break
        sorted_token_ids, expert_ids, num_tokens_post_padded = self._align(
            topk_ids, block_size_m, self.quant_info.num_local_experts
        )
        pair_count = topk_ids.numel()
        src2dst = torch.arange(
            pair_count, device=hidden_states.device, dtype=torch.int32
        )
        return BaseGemmWorkspace(
            hidden_permuted=hidden_states,
            masked_m=torch.empty(0, device=hidden_states.device, dtype=torch.int32),
            expected_m=pair_count,
            src2dst=src2dst,
            m_max=pair_count,
            topk_weights=topk_weights,
            packed_topk_ids=packed_topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            moe_block_size=block_size_m,
            hidden_permuted_owned=False,
        )

    def _run_marlin(
        self,
        *,
        activation: torch.Tensor,
        out: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        sort_indices: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        bias: torch.Tensor | None,
        ws: BaseGemmWorkspace,
        top_k: int,
        size_m: int,
        size_n: int,
        size_k: int,
    ) -> None:
        if (
            ws.sorted_token_ids is None
            or ws.expert_ids is None
            or ws.num_tokens_post_padded is None
            or ws.topk_weights is None
        ):
            raise RuntimeError("Marlin routing plan is incomplete")
        self._gemm(
            activation,
            out,
            weight,
            bias,
            scales,
            global_scale,
            qzeros,
            g_idx,
            sort_indices,
            self._get_workspace(activation.device),
            ws.sorted_token_ids,
            ws.expert_ids,
            ws.num_tokens_post_padded,
            ws.topk_weights,
            moe_block_size=ws.moe_block_size,
            top_k=top_k,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=self._get_scalar_type(
                self.quant_info.weight_bits,
                qzeros is not None,
                scales,
                global_scale,
            ),
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
            is_k_full=self.quant_info.is_k_full,
            use_atomic_add=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    def gateup(self, ws: BaseGemmWorkspace, out: torch.Tensor) -> None:
        self._run_marlin(
            activation=ws.hidden_permuted,
            out=out,
            weight=self.quant_info.w13_weight,
            scales=self.quant_info.w13_scales,
            qzeros=self.quant_info.w13_qzeros,
            g_idx=self.quant_info.w13_g_idx,
            sort_indices=self.quant_info.w13_g_idx_sort_indices,
            global_scale=self.quant_info.w13_global_scale,
            bias=self.quant_info.w13_bias,
            ws=ws,
            top_k=ws.topk_weights.shape[1] if ws.topk_weights is not None else 0,
            size_m=ws.hidden_permuted.shape[0],
            size_n=2 * self.quant_info.intermediate_size,
            size_k=self.quant_info.hidden_size,
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
            gate_first=True,
            interleaved=False,
        )

    def down(
        self, ws: BaseGemmWorkspace, act_out: torch.Tensor, out: torch.Tensor
    ) -> None:
        self._run_marlin(
            activation=act_out,
            out=out,
            weight=self.quant_info.w2_weight,
            scales=self.quant_info.w2_scales,
            qzeros=self.quant_info.w2_qzeros,
            g_idx=self.quant_info.w2_g_idx,
            sort_indices=self.quant_info.w2_g_idx_sort_indices,
            global_scale=self.quant_info.w2_global_scale,
            bias=self.quant_info.w2_bias,
            ws=ws,
            top_k=1,
            size_m=ws.expected_m,
            size_n=self.quant_info.hidden_size,
            size_k=self.quant_info.intermediate_size,
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
            down_out,
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
        return (ws.expected_m, 2 * self.quant_info.intermediate_size)

    def act_out_shape(self, ws: BaseGemmWorkspace) -> tuple[int, ...]:
        return (ws.expected_m, self.quant_info.intermediate_size)

    def down_out_shape(self, ws: BaseGemmWorkspace) -> tuple[int, ...]:
        return (ws.expected_m, self.quant_info.hidden_size)


BaseGemmFactory = Callable[
    [SglLoraQuantInfo, "MoeRunnerConfig", MoeLoraWorkspacePlanner | None],
    MoeLoraBaseGemm,
]

_BASE_GEMM_PROVIDERS: dict[type, BaseGemmFactory] = {
    SglLoraBf16QuantInfo: DeepGemmBf16BaseGemm,
    SglLoraFp8QuantInfo: DeepGemmFp8BaseGemm,
    SglLoraNvFp4QuantInfo: CuteDslNvFp4BaseGemm,
    SglLoraMarlinQuantInfo: MarlinW4A16BaseGemm,
}


def register_base_gemm_provider(
    quant_info_type: type,
    factory: BaseGemmFactory,
    *,
    replace: bool = False,
) -> None:
    """Register an out-of-tree provider without changing the semantic runner."""
    if quant_info_type in _BASE_GEMM_PROVIDERS and not replace:
        raise ValueError(
            f"a base-GEMM provider is already registered for {quant_info_type.__name__}"
        )
    _BASE_GEMM_PROVIDERS[quant_info_type] = factory


def resolve_base_gemm(
    quant_info: SglLoraQuantInfo,
    config: MoeRunnerConfig,
    workspace_planner: MoeLoraWorkspacePlanner | None = None,
    *,
    provider_factory: BaseGemmFactory | None = None,
) -> MoeLoraBaseGemm:
    factory = provider_factory or _BASE_GEMM_PROVIDERS.get(type(quant_info))
    if factory is None:
        raise NotImplementedError(
            f"sgl_lora has no base-GEMM provider for {type(quant_info).__name__}."
        )
    return factory(quant_info, config, workspace_planner)


__all__ = [
    "BaseGemmFactory",
    "BaseGemmWorkspace",
    "CuteDslNvFp4BaseGemm",
    "DeepGemmBf16BaseGemm",
    "DeepGemmFp8BaseGemm",
    "MarlinW4A16BaseGemm",
    "MoeLoraBaseGemm",
    "MoeLoraProviderContract",
    "register_base_gemm_provider",
    "resolve_base_gemm",
]
