"""BF16 MoE provider backed by the DeepGEMM masked grouped GEMM.

The masked row domain (S1 preprocess, S3 activation join, S5 finalize,
geometry, workspace) lives in :mod:`masked_row_domain`; this class supplies
only the GEMM engine: the raw ``grouped_gemm_nt_bf16_masked`` primitive for
S2/S4. No stock ``MoeRunner`` core participates.
"""

from __future__ import annotations

import torch

from sglang.srt.lora.sgl_lora.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.sgl_lora.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowWorkspace,
)
from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo


class DeepGemmBf16Provider(MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="deepgemm_bf16",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        supported_output_dtypes=(torch.bfloat16, torch.float32),
    )

    def __init__(self, quant_info: SglLoraBf16QuantInfo):
        super().__init__(quant_info)
        self._require_supported_geometry(quant_info)
        from sglang.srt.layers import deep_gemm_wrapper

        self._grouped_gemm_bf16_masked = deep_gemm_wrapper.grouped_gemm_nt_bf16_masked

    @staticmethod
    def _require_supported_geometry(quant_info: SglLoraBf16QuantInfo) -> None:
        """Reject contraction dimensions unsupported by DeepGEMM on SM90.

        Hopper's BF16 kernel requires ``K % 64 == 0``. Gate/up contracts over
        ``hidden_size`` and down contracts over ``intermediate_size``, so both
        dimensions must qualify. The SM100 implementation has no such limit.
        """
        major, _minor = torch.cuda.get_device_capability(quant_info.w2_weight.device)
        if major >= 10:
            return
        offenders = {
            name: value
            for name, value in (
                ("hidden_size", quant_info.hidden_size),
                ("intermediate_size", quant_info.intermediate_size),
            )
            if value % 64 != 0
        }
        if offenders:
            detail = ", ".join(
                f"{name}={value}" for name, value in sorted(offenders.items())
            )
            raise ValueError(
                f"deepgemm_bf16 on SM{major}x requires every GEMM contraction "
                f"dimension to be a multiple of 64, but {detail}. gate/up "
                "contracts over hidden_size and down over intermediate_size, "
                "so both must qualify; SM100 has no such constraint"
            )

    def gateup(
        self,
        ws: MaskedRowWorkspace,
        out: torch.Tensor,
    ) -> None:
        self._grouped_gemm_bf16_masked(
            ws.hidden_permuted,
            self.quant_info.w13_weight,
            out,
            ws.masked_m,
            ws.expected_m,
        )

    def down(
        self,
        ws: MaskedRowWorkspace,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._grouped_gemm_bf16_masked(
            act_out,
            self.quant_info.w2_weight,
            out,
            ws.masked_m,
            ws.expected_m,
        )
