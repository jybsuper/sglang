"""CuTeDSL GEMMs for per-channel FP8 weights with per-token activation scales.

Weights keep their checkpoint values and one FP32 scale per output row;
tokens quantize to one scale per row during dispatch; GEMM outputs remain
BF16 for LoRA.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.lora.moe.activation_delta import act_delta
from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
    dispatch_fill_rows_contiguous_fp8,
)
from sglang.kernels.ops.lora.moe.dispatch_masked import dispatch_fill_masked_fp8
from sglang.kernels.ops.lora.moe.fp8_quant import quantize_rows_per_token_fp8
from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProviderContract,
    expected_rows_per_expert,
    prepare_buffer,
)
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
    ContiguousRowState,
)
from sglang.srt.lora.moe.base_gemm_provider.cutedsl_fp8 import CuteDslFp8Mixin
from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowState,
    masked_m_max,
)
from sglang.srt.lora.moe.quant_info import MoeLoraFp8ChannelQuantInfo

_CONTRACT = dict(
    quant_info_cls=MoeLoraFp8ChannelQuantInfo,
    gate_first=True,
    interleaved=False,
    gate_up_output_dtype=torch.bfloat16,
    lora_delta_dtype=torch.bfloat16,
    lora_activation_dtype=torch.bfloat16,
)


def _admit_tile_geometry(quant_info: MoeLoraFp8ChannelQuantInfo) -> None:
    """The kernel stages one weight-row scale per 128-row tile and tiles K by 128."""
    for name, value in (
        ("hidden_size", quant_info.hidden_size),
        ("intermediate_size", quant_info.intermediate_size),
    ):
        if value % 128:
            raise NotImplementedError(
                "per-channel FP8 MoE LoRA on CuTeDSL requires "
                f"{name} % 128 == 0, got {value}"
            )


class CuteDslFp8ChannelMixin(CuteDslFp8Mixin):
    _DTYPE_TAG = "fp8_channel"

    def _bind_weights(self, quant_info: MoeLoraFp8ChannelQuantInfo) -> None:
        # [E, rows, 1] checkpoint scales -> [E, rows] for the kernel.
        self._sf_w13 = quant_info.w13_scale.flatten(1).contiguous()
        self._sf_w2 = quant_info.w2_scale.flatten(1).contiguous()

    def act_with_delta(
        self,
        row_state,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
    ) -> None:
        # One scale per row, written while activating: no separate quant pass.
        act_delta(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            row_state.pair_to_row,
            topk_ids,
            num_local_experts=self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            act_quant=(
                row_state.act_fp8,
                row_state.act_scale,
                self.quant_info.intermediate_size,
            ),
        )
        row_state.act_quant_ready = True

    def _launch(
        self,
        stage: str,
        a_fp8: torch.Tensor,
        sf_tokens: torch.Tensor,
        c: torch.Tensor,
        row_state,
        schedule: torch.Tensor,
        tiles: torch.Tensor,
    ) -> None:
        call = self._compiled[row_state.token_width][stage]
        # Runtime wrapping must match the compiled argument's MLIR type.
        dyn = self._as_dynamic_cute_tensor
        a_rows = self._rows_arg(a_fp8)
        call.compiled_fn(
            dyn(a_rows, leading_dim=2),
            call.b_arg,
            dyn(sf_tokens.view(*a_rows.shape[:2]), leading_dim=1),
            dyn(call.sf_weights, leading_dim=1),
            dyn(self._rows_arg(c), leading_dim=2),
            dyn(self._group_arg(row_state), leading_dim=0),
            dyn(schedule, leading_dim=0),
            dyn(tiles, leading_dim=0),
            self._stream(a_fp8.device),
        )


class CuteDslFp8ChannelMaskedRowState(MaskedRowState, kw_only=True):
    token_width: int
    sf_rows: torch.Tensor  # [E_local, m_max] fp32 token scales
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor
    act_fp8: torch.Tensor  # [E_local, m_max, inter]
    act_scale: torch.Tensor  # [E_local, m_max]
    act_quant_ready: bool = False


class CuteDslFp8ChannelMaskedProvider(CuteDslFp8ChannelMixin, MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(key="cutedsl_fp8_channel_masked", **_CONTRACT)

    _ROW_MODE = "masked"

    def __init__(self, quant_info: MoeLoraFp8ChannelQuantInfo):
        super().__init__(quant_info)
        _admit_tile_geometry(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_masked_fp8_rowwise
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_masked,
            dual_stage_schedule_capacities_masked,
        )

        self._prepare_masked_gemm = prepare_masked_fp8_rowwise
        self._build_schedules = build_dual_stage_schedules_masked
        self._schedule_capacities = dual_stage_schedule_capacities_masked
        self._init_tiles(quant_info)

    def _prepare_dummy(self, stage: str, config):
        weight, weight_sf = self._stage_weight(stage), self._stage_scale(stage)
        device = weight.device
        experts, n, k = weight.shape
        dummy_a = torch.zeros(
            (experts, 256, k), dtype=torch.float8_e4m3fn, device=device
        )
        dummy_sf = torch.zeros((experts, 256), dtype=torch.float32, device=device)
        dummy_c = torch.empty((experts, 256, n), dtype=torch.bfloat16, device=device)
        dummy_masked = torch.zeros(experts, dtype=torch.int32, device=device)
        return self._prepare_masked_gemm(
            dummy_a,
            weight,
            dummy_sf,
            weight_sf,
            dummy_c,
            dummy_masked,
            config=config,
        )

    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def _input_rows(row_state: CuteDslFp8ChannelMaskedRowState) -> torch.Tensor:
        return row_state.hidden_permuted

    @staticmethod
    def _group_arg(row_state: CuteDslFp8ChannelMaskedRowState) -> torch.Tensor:
        return row_state.masked_m

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> CuteDslFp8ChannelMaskedRowState:
        experts = self.quant_info.num_local_experts
        k = hidden_states.size(1)
        inter = self.quant_info.intermediate_size
        expected_m = expected_rows_per_expert(topk_ids.numel(), experts)
        token_width = self._token_width_for(hidden_states.size(0), expected_m)
        # Token-scale loads are unpredicated across the whole selected token tile.
        m_max = masked_m_max(hidden_states.size(0), token_width)
        device = hidden_states.device

        def _buf(name, shape, dtype):
            return prepare_buffer(workspace, name, shape, dtype=dtype, device=device)

        masked_m = _buf("masked:masked_m", (experts,), torch.int32)
        pair_to_row = _buf("masked:pair_to_row", (topk_ids.numel(),), torch.int32)
        rows_fp8 = self._prepare_input_rows(workspace, (experts, m_max, k), device)
        sf_rows = _buf("masked:sf_tokens", (experts, m_max), torch.float32)
        act_fp8 = _buf("masked:act_fp8", (experts, m_max, inter), torch.float8_e4m3fn)
        act_scale = _buf("masked:act_token_scale", (experts, m_max), torch.float32)
        geometry = dict(
            token_width=token_width,
            n_gemm1=self.gate_up_slices * inter,
            n_gemm2=self.quant_info.hidden_size,
            output_width=self.OUTPUT_WIDTH,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
        )
        capacities = self._schedule_capacities(
            num_experts=experts, m_max=m_max, **geometry
        )
        dispatch_fill_masked_fp8(
            hidden_states,
            topk_ids,
            top_k,
            masked_m_out=masked_m,
            pair_to_row_out=pair_to_row,
            rows_fp8_out=rows_fp8,
            scale_out=sf_rows,
            per_token=True,
        )
        schedule_outputs = {}
        if workspace is not None:
            schedule_outputs = self._schedule_buffers(
                workspace, f"cutedsl_masked:tw{token_width}", capacities, device
            )
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules(
            masked_m, m_max=m_max, **geometry, **schedule_outputs
        )
        return CuteDslFp8ChannelMaskedRowState(
            hidden_permuted=rows_fp8,
            masked_m=masked_m,
            expected_m=expected_m,
            pair_to_row=pair_to_row,
            m_max=m_max,
            input_buffer_reuse=workspace is not None,
            token_width=token_width,
            sf_rows=sf_rows,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
            act_fp8=act_fp8,
            act_scale=act_scale,
        )

    def _down_operands(
        self, row_state: CuteDslFp8ChannelMaskedRowState, act_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not row_state.act_quant_ready:
            # Fused-B activation stays BF16.
            quantize_rows_per_token_fp8(
                act_out,
                q_out=row_state.act_fp8,
                scale_out=row_state.act_scale,
                masked_m=row_state.masked_m,
            )
        return row_state.act_fp8, row_state.act_scale


class CuteDslFp8ChannelContiguousRowState(ContiguousRowState, kw_only=True):
    token_width: int
    sf_rows: torch.Tensor  # [m_pad_ceiling] fp32 token scales
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor
    act_fp8: torch.Tensor  # [m_pad_ceiling, inter]
    act_scale: torch.Tensor  # [m_pad_ceiling]
    act_quant_ready: bool = False


class CuteDslFp8ChannelContiguousProvider(
    CuteDslFp8ChannelMixin, ContiguousRowDomainProvider
):
    contract = MoeBaseProviderContract(
        key="cutedsl_fp8_channel_contiguous", **_CONTRACT
    )

    M_ALIGNMENT = 128

    _ROW_MODE = "contiguous"

    def __init__(self, quant_info: MoeLoraFp8ChannelQuantInfo):
        super().__init__(quant_info, m_alignment=self.M_ALIGNMENT)
        _admit_tile_geometry(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import (
            prepare_contiguous_fp8_rowwise,
        )
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_contiguous,
            dual_stage_schedule_capacities_contiguous,
            validate_tile_geometry_contiguous,
        )

        self._prepare_contiguous_gemm = prepare_contiguous_fp8_rowwise
        self._validate_tile_geometry = validate_tile_geometry_contiguous
        self._build_schedules_contiguous = build_dual_stage_schedules_contiguous
        self._schedule_capacities = dual_stage_schedule_capacities_contiguous
        self._init_tiles(quant_info)

    def _admit_tile_width(self, token_width: int) -> None:
        self._validate_tile_geometry(token_width, self._m_alignment)

    def _prepare_dummy(self, stage: str, config):
        weight, weight_sf = self._stage_weight(stage), self._stage_scale(stage)
        device = weight.device
        num_experts, n, k = weight.shape
        dummy_a = torch.zeros((256, k), dtype=torch.float8_e4m3fn, device=device)
        dummy_sf = torch.zeros((1, 256), dtype=torch.float32, device=device)
        dummy_c = torch.empty((256, n), dtype=torch.bfloat16, device=device)
        dummy_seg = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        return self._prepare_contiguous_gemm(
            dummy_a,
            weight,
            dummy_sf,
            weight_sf,
            dummy_c,
            dummy_seg,
            config=config,
        )

    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        # Flat rows occupy one expert slot in the swap_ab wrapper.
        return tensor.unsqueeze(0)

    @staticmethod
    def _input_rows(row_state: CuteDslFp8ChannelContiguousRowState) -> torch.Tensor:
        return row_state.hidden_compact

    @staticmethod
    def _group_arg(row_state: CuteDslFp8ChannelContiguousRowState) -> torch.Tensor:
        return row_state.seg_offsets

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> CuteDslFp8ChannelContiguousRowState:
        base = super().prepare(
            hidden_states, topk_ids, top_k, workspace, fill_rows=False, finalize=False
        )
        num_experts = self.quant_info.num_local_experts
        hidden = hidden_states.size(1)
        inter = self.quant_info.intermediate_size
        device = hidden_states.device
        prefix = f"contiguous:a{self._m_alignment}"

        def _buf(name, shape, dtype):
            return prepare_buffer(
                workspace, f"{prefix}:{name}", shape, dtype=dtype, device=device
            )

        rows_fp8 = self._prepare_input_rows(
            workspace, (base.m_pad_ceiling, hidden), device
        )
        sf_rows = _buf("sf_tokens", (base.m_pad_ceiling,), torch.float32)
        act_fp8 = _buf("act_fp8", (base.m_pad_ceiling, inter), torch.float8_e4m3fn)
        act_scale = _buf("act_token_scale", (base.m_pad_ceiling,), torch.float32)
        dispatch_fill_rows_contiguous_fp8(
            hidden_states,
            topk_ids,
            base.pair_to_row,
            rows_fp8_out=rows_fp8,
            scale_out=sf_rows,
            per_token=True,
            seg_offsets=base.seg_offsets,
        )
        # Distinct top-k experts bound each expert to one row per token.
        max_expert_rows = max(int(hidden_states.size(0)), 1)
        expected_m = expected_rows_per_expert(topk_ids.numel(), num_experts)
        token_width = self._token_width_for(max_expert_rows, expected_m)
        geometry = dict(
            m_pad_ceiling=base.m_pad_ceiling,
            max_expert_rows=max_expert_rows,
            m_alignment=self._m_alignment,
            token_width=token_width,
            n_gemm1=self.gate_up_slices * inter,
            n_gemm2=self.quant_info.hidden_size,
            output_width=self.OUTPUT_WIDTH,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
        )
        schedule_outputs = {}
        if workspace is not None:
            schedule_outputs = self._schedule_buffers(
                workspace,
                f"cutedsl_contiguous:a{self._m_alignment}:tw{token_width}",
                self._schedule_capacities(num_experts=num_experts, **geometry),
                device,
            )
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules_contiguous(
            base.seg_counts, **geometry, **schedule_outputs
        )
        return CuteDslFp8ChannelContiguousRowState(
            hidden_compact=rows_fp8,
            seg_counts=base.seg_counts,
            seg_offsets=base.seg_offsets,
            pair_to_row=base.pair_to_row,
            m_pad_ceiling=base.m_pad_ceiling,
            input_buffer_reuse=base.input_buffer_reuse,
            token_width=token_width,
            sf_rows=sf_rows,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
            act_fp8=act_fp8,
            act_scale=act_scale,
        )

    def _down_operands(
        self, row_state: CuteDslFp8ChannelContiguousRowState, act_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not row_state.act_quant_ready:
            # Fused-B activation stays BF16; the segment total bounds the rows.
            quantize_rows_per_token_fp8(
                act_out,
                q_out=row_state.act_fp8,
                scale_out=row_state.act_scale,
                live_rows=row_state.seg_offsets[self.quant_info.num_local_experts :],
            )
        return row_state.act_fp8, row_state.act_scale
