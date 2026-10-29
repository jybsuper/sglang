"""CuTeDSL GEMMs on block-scaled checkpoints: MXFP8 (E4M3 x E8M0 per 32),
MXFP4 (E2M1 pairs x E8M0 per 32) and NVFP4 (E2M1 pairs x E4M3 per 16 under
per-expert global scales).

Activations are quantized per row into the weights' format with the scales
packed in the tcgen05 atom order; GEMM outputs remain BF16 for LoRA.
"""

from __future__ import annotations

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProviderContract,
    expected_rows_per_expert,
    prepare_buffer,
)
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
    ContiguousRowState,
)
from sglang.srt.lora.moe.base_gemm_provider.cutedsl_common import CuteDslTileMixin
from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowState,
    masked_m_max,
)
from sglang.srt.lora.moe.quant_info import (
    MoeLoraMxfp4QuantInfo,
    MoeLoraMxfp8QuantInfo,
    MoeLoraMxQuantInfo,
    MoeLoraNvFp4QuantInfo,
)


def _contract(key: str, quant_info_cls) -> MoeBaseProviderContract:
    return MoeBaseProviderContract(
        key=key,
        quant_info_cls=quant_info_cls,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )


class _CuteDslBlockScaledMixin(CuteDslTileMixin):
    # Token tiles cover half or all of a 128-row scale atom.
    TOKEN_WIDTHS = (
        CuteDslTileMixin.WIDE_TOKEN_WIDTH,
        CuteDslTileMixin.XWIDE_TOKEN_WIDTH,
    )
    WIDE_EXPECTED_M_THRESHOLD = 32
    SF_VEC_SIZE = 32

    def _bind_weights(self, quant_info: MoeLoraMxQuantInfo) -> None:
        from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
        from sglang.kernels.ops.lora.moe.quant_blockscaled import (
            pack_scale_atoms,
            quant_mx_rows,
        )

        self._scale_atom_bytes = scale_atom_bytes
        self._quant_rows = quant_mx_rows
        self._pack = quant_info.elements_per_byte
        # FP4 pairs may load as int8; the kernels take packed uint8 storage.
        store_dtype = torch.uint8 if self._pack == 2 else quant_info.w13_weight.dtype
        self._w13 = quant_info.w13_weight.view(store_dtype)
        self._w2 = quant_info.w2_weight.view(store_dtype)
        self._sf_w13 = pack_scale_atoms(quant_info.w13_scale)
        self._sf_w2 = pack_scale_atoms(quant_info.w2_scale)
        # NVFP4 fills these with per-expert global scales and activation scales.
        self._alpha = {"gemm1": None, "gemm2": None}
        self._alpha_rows_per_half = {"gemm1": 1, "gemm2": 1}
        self._act_scale = {"gemm1": None, "gemm2": None}

    def _stage_weight(self, stage: str) -> torch.Tensor:
        if stage == "gemm1":
            return self._w13
        if stage == "gemm2":
            return self._w2
        raise ValueError(f"unknown CuTeDSL base stage {stage!r}")

    def _stage_scale(self, stage: str) -> torch.Tensor:
        return self._sf_w13 if stage == "gemm1" else self._sf_w2

    def _quant_buffers(self, workspace, prefix, slabs, rows, k, device):
        """Quantized rows plus their scale atoms for ``slabs`` slabs of ``rows``."""
        shape = (slabs, rows, k // self._pack) if slabs > 1 else (rows, k // self._pack)
        data = prepare_buffer(
            workspace, f"{prefix}:q", shape, dtype=self._w13.dtype, device=device
        )
        scales = prepare_buffer(
            workspace,
            f"{prefix}:sf",
            (self._scale_atom_bytes(rows, k, self.SF_VEC_SIZE, slabs),),
            dtype=torch.uint8,
            device=device,
        )
        return data, scales

    def _quantize(self, stage, rows, data, scales, pair_to_row):
        self._quant_rows(
            rows.reshape(-1, rows.shape[-1]),
            out=data.reshape(-1, data.shape[-1]),
            sf_out=scales,
            sf_vec_size=self.SF_VEC_SIZE,
            rows_per_slab=rows.shape[-2],
            pair_to_row=pair_to_row,
        )

    def _launch(
        self,
        stage: str,
        a_q: torch.Tensor,
        sf_tokens: torch.Tensor,
        c: torch.Tensor,
        row_state,
        schedule: torch.Tensor,
        tiles: torch.Tensor,
    ) -> None:
        call = self._compiled[row_state.token_width][stage]
        # Runtime wrapping must match the compiled argument's MLIR type.
        dyn = self._as_dynamic_cute_tensor
        call.compiled_fn(
            dyn(self._rows_arg(a_q), leading_dim=2),
            call.b_arg,
            dyn(sf_tokens, leading_dim=0),
            dyn(call.sf_weights, leading_dim=0),
            dyn(self._rows_arg(c), leading_dim=2),
            dyn(self._alpha_arg(stage), leading_dim=1),
            dyn(self._group_arg(row_state), leading_dim=0),
            dyn(schedule, leading_dim=0),
            dyn(tiles, leading_dim=0),
            self._stream(a_q.device),
        )

    def _alpha_arg(self, stage: str) -> torch.Tensor:
        alpha = self._alpha[stage]
        if alpha is None:
            alpha = self._alpha[stage] = torch.ones(
                (self.quant_info.num_local_experts, 1),
                dtype=torch.float32,
                device=self._w13.device,
            )
        return alpha

    def _stage_kwargs(self, stage: str) -> dict:
        return dict(
            sf_vec_size=self.SF_VEC_SIZE,
            alpha=self._alpha[stage],
            alpha_rows_per_half=self._alpha_rows_per_half[stage],
        )

    def gateup(self, row_state, out: torch.Tensor) -> None:
        self._launch(
            "gemm1",
            row_state.rows_q,
            row_state.sf_rows,
            out,
            row_state,
            row_state.gemm1_schedule,
            row_state.gemm1_tiles,
        )

    def down(self, row_state, act_out: torch.Tensor, out: torch.Tensor) -> None:
        self._quantize(
            "gemm2",
            act_out,
            row_state.act_q,
            row_state.act_sf,
            row_state.pair_to_row,
        )
        self._launch(
            "gemm2",
            row_state.act_q,
            row_state.act_sf,
            out,
            row_state,
            row_state.gemm2_schedule,
            row_state.gemm2_tiles,
        )


class _CuteDslNvFp4Mixin(_CuteDslBlockScaledMixin):
    SF_VEC_SIZE = 16

    def _bind_weights(self, quant_info: MoeLoraNvFp4QuantInfo) -> None:
        super()._bind_weights(quant_info)
        from sglang.kernels.ops.lora.moe.quant_blockscaled import quant_nvfp4_rows

        self._quant_rows = quant_nvfp4_rows
        # The epilogue applies activation x weight global scales per expert;
        # gate and up rows of W13 may carry separate weight scales.
        w13_global = quant_info.w13_global_scale.reshape(
            quant_info.num_local_experts, -1
        )
        self._alpha["gemm1"] = (w13_global * quant_info.w13_input_scale).contiguous()
        self._alpha_rows_per_half["gemm1"] = (
            quant_info.intermediate_size if w13_global.shape[1] == 2 else 1 << 30
        )
        self._alpha["gemm2"] = (
            quant_info.w2_global_scale.reshape(-1, 1) * quant_info.w2_input_scale
        ).contiguous()
        self._alpha_rows_per_half["gemm2"] = 1 << 30
        self._act_scale = {
            "gemm1": quant_info.w13_input_scale,
            "gemm2": quant_info.w2_input_scale,
        }

    def _quantize(self, stage, rows, data, scales, pair_to_row):
        self._quant_rows(
            rows.reshape(-1, rows.shape[-1]),
            out=data.reshape(-1, data.shape[-1]),
            sf_out=scales,
            global_scale=self._act_scale[stage],
            rows_per_slab=rows.shape[-2],
            pair_to_row=pair_to_row,
        )


class CuteDslBlockScaledMaskedRowState(MaskedRowState, kw_only=True):
    token_width: int
    rows_q: torch.Tensor  # [E_local, m_max, hidden / pack]
    sf_rows: torch.Tensor  # uint8 scale atoms
    act_q: torch.Tensor  # [E_local, m_max, inter / pack]
    act_sf: torch.Tensor
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class _CuteDslBlockScaledMaskedProvider(
    _CuteDslBlockScaledMixin, MaskedRowDomainProvider
):
    _ROW_MODE = "masked"

    def __init__(self, quant_info: MoeLoraMxQuantInfo):
        super().__init__(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_masked_blockscaled
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_masked,
            dual_stage_schedule_capacities_masked,
        )

        self._prepare_masked_gemm = prepare_masked_blockscaled
        self._build_schedules = build_dual_stage_schedules_masked
        self._schedule_capacities = dual_stage_schedule_capacities_masked
        self._init_tiles(quant_info)

    def _prepare_dummy(self, stage: str, config):
        weight, weight_sf = self._stage_weight(stage), self._stage_scale(stage)
        device = weight.device
        experts, n, k_store = weight.shape
        dummy_a = torch.zeros(
            (experts, 256, k_store), dtype=weight.dtype, device=device
        )
        dummy_sf = torch.zeros(
            self._scale_atom_bytes(
                256, k_store * self._pack, self.SF_VEC_SIZE, experts
            ),
            dtype=torch.uint8,
            device=device,
        )
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
            **self._stage_kwargs(stage),
        )

    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def _group_arg(row_state: CuteDslBlockScaledMaskedRowState) -> torch.Tensor:
        return row_state.masked_m

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> CuteDslBlockScaledMaskedRowState:
        experts = self.quant_info.num_local_experts
        k = hidden_states.size(1)
        inter = self.quant_info.intermediate_size
        expected_m = expected_rows_per_expert(topk_ids.numel(), experts)
        token_width = self._token_width_for(hidden_states.size(0), expected_m)
        m_max = masked_m_max(hidden_states.size(0), token_width)
        device = hidden_states.device

        def _buf(name, shape, dtype):
            return prepare_buffer(workspace, name, shape, dtype=dtype, device=device)

        masked_m = _buf("masked:masked_m", (experts,), torch.int32)
        pair_to_row = _buf("masked:pair_to_row", (topk_ids.numel(),), torch.int32)
        hidden_permuted = _buf(
            "masked:hidden_permuted", (experts, m_max, k), torch.bfloat16
        )
        rows_q, sf_rows = self._quant_buffers(
            workspace, "masked:rows", experts, m_max, k, device
        )
        act_q, act_sf = self._quant_buffers(
            workspace, "masked:act", experts, m_max, inter, device
        )
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
        self._preprocess(
            hidden_states,
            topk_ids,
            top_k,
            masked_m_out=masked_m,
            pair_to_row_out=pair_to_row,
            rows_out=hidden_permuted,
        )
        self._quantize("gemm1", hidden_permuted, rows_q, sf_rows, pair_to_row)
        schedule_outputs = {}
        if workspace is not None:
            schedule_outputs = self._schedule_buffers(
                workspace, f"cutedsl_masked:tw{token_width}", capacities, device
            )
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules(
            masked_m, m_max=m_max, **geometry, **schedule_outputs
        )
        return CuteDslBlockScaledMaskedRowState(
            hidden_permuted=hidden_permuted,
            masked_m=masked_m,
            expected_m=expected_m,
            pair_to_row=pair_to_row,
            m_max=m_max,
            input_buffer_reuse=workspace is not None,
            token_width=token_width,
            rows_q=rows_q,
            sf_rows=sf_rows,
            act_q=act_q,
            act_sf=act_sf,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
        )


class CuteDslBlockScaledContiguousRowState(ContiguousRowState, kw_only=True):
    token_width: int
    rows_q: torch.Tensor  # [m_pad_ceiling, hidden / pack]
    sf_rows: torch.Tensor
    act_q: torch.Tensor  # [m_pad_ceiling, inter / pack]
    act_sf: torch.Tensor
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class _CuteDslBlockScaledContiguousProvider(
    _CuteDslBlockScaledMixin, ContiguousRowDomainProvider
):
    # Segments start on scale-atom boundaries.
    M_ALIGNMENT = 128

    _ROW_MODE = "contiguous"

    def __init__(self, quant_info: MoeLoraMxQuantInfo):
        super().__init__(quant_info, m_alignment=self.M_ALIGNMENT)
        from sglang.kernels.ops.lora.moe.cutedsl.api import (
            prepare_contiguous_blockscaled,
        )
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_contiguous,
            dual_stage_schedule_capacities_contiguous,
            validate_tile_geometry_contiguous,
        )

        self._prepare_contiguous_gemm = prepare_contiguous_blockscaled
        self._validate_tile_geometry = validate_tile_geometry_contiguous
        self._build_schedules_contiguous = build_dual_stage_schedules_contiguous
        self._schedule_capacities = dual_stage_schedule_capacities_contiguous
        self._init_tiles(quant_info)

    def _admit_tile_width(self, token_width: int) -> None:
        self._validate_tile_geometry(token_width, self._m_alignment)

    def _prepare_dummy(self, stage: str, config):
        weight, weight_sf = self._stage_weight(stage), self._stage_scale(stage)
        device = weight.device
        num_experts, n, k_store = weight.shape
        dummy_a = torch.zeros((256, k_store), dtype=weight.dtype, device=device)
        dummy_sf = torch.zeros(
            self._scale_atom_bytes(256, k_store * self._pack, self.SF_VEC_SIZE),
            dtype=torch.uint8,
            device=device,
        )
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
            **self._stage_kwargs(stage),
        )

    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        # Flat rows occupy one expert slot in the swap_ab wrapper.
        return tensor.unsqueeze(0)

    @staticmethod
    def _group_arg(row_state: CuteDslBlockScaledContiguousRowState) -> torch.Tensor:
        return row_state.seg_offsets

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> CuteDslBlockScaledContiguousRowState:
        base = super().prepare(hidden_states, topk_ids, top_k, workspace)
        num_experts = self.quant_info.num_local_experts
        hidden = hidden_states.size(1)
        inter = self.quant_info.intermediate_size
        device = hidden_states.device
        prefix = f"contiguous:a{self._m_alignment}"
        rows_q, sf_rows = self._quant_buffers(
            workspace, f"{prefix}:rows", 1, base.m_pad_ceiling, hidden, device
        )
        act_q, act_sf = self._quant_buffers(
            workspace, f"{prefix}:act", 1, base.m_pad_ceiling, inter, device
        )
        self._quantize("gemm1", base.hidden_compact, rows_q, sf_rows, base.pair_to_row)
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
        return CuteDslBlockScaledContiguousRowState(
            hidden_compact=base.hidden_compact,
            seg_counts=base.seg_counts,
            seg_offsets=base.seg_offsets,
            pair_to_row=base.pair_to_row,
            m_pad_ceiling=base.m_pad_ceiling,
            input_buffer_reuse=base.input_buffer_reuse,
            token_width=token_width,
            rows_q=rows_q,
            sf_rows=sf_rows,
            act_q=act_q,
            act_sf=act_sf,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
        )


class CuteDslMxfp8MaskedProvider(_CuteDslBlockScaledMaskedProvider):
    contract = _contract("cutedsl_mxfp8_masked", MoeLoraMxfp8QuantInfo)
    _DTYPE_TAG = "mxfp8"


class CuteDslMxfp8ContiguousProvider(_CuteDslBlockScaledContiguousProvider):
    contract = _contract("cutedsl_mxfp8_contiguous", MoeLoraMxfp8QuantInfo)
    _DTYPE_TAG = "mxfp8"


class CuteDslMxfp4MaskedProvider(_CuteDslBlockScaledMaskedProvider):
    contract = _contract("cutedsl_mxfp4_masked", MoeLoraMxfp4QuantInfo)
    _DTYPE_TAG = "mxfp4"


class CuteDslMxfp4ContiguousProvider(_CuteDslBlockScaledContiguousProvider):
    contract = _contract("cutedsl_mxfp4_contiguous", MoeLoraMxfp4QuantInfo)
    _DTYPE_TAG = "mxfp4"


class CuteDslNvFp4MaskedProvider(_CuteDslNvFp4Mixin, _CuteDslBlockScaledMaskedProvider):
    contract = _contract("cutedsl_nvfp4_masked", MoeLoraNvFp4QuantInfo)
    _DTYPE_TAG = "nvfp4"


class CuteDslNvFp4ContiguousProvider(
    _CuteDslNvFp4Mixin, _CuteDslBlockScaledContiguousProvider
):
    contract = _contract("cutedsl_nvfp4_contiguous", MoeLoraNvFp4QuantInfo)
    _DTYPE_TAG = "nvfp4"
