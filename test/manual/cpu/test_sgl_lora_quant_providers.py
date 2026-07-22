from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.sgl_lora.base_gemm import (
    BaseGemmWorkspace,
    CuteDslNvFp4BaseGemm,
    DeepGemmBf16BaseGemm,
    DeepGemmFp8BaseGemm,
    MarlinW4A16BaseGemm,
    resolve_base_gemm,
)
from sglang.srt.lora.sgl_lora.lora_layer import build_sgl_lora_quant_info
from sglang.srt.lora.sgl_lora.quant_info import (
    SglLoraBf16QuantInfo,
    SglLoraFp8QuantInfo,
    SglLoraMarlinQuantInfo,
    SglLoraNvFp4QuantInfo,
)


def _empty(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype)


def test_quant_payloads_carry_semantic_dimensions_for_packed_weights():
    marlin = SglLoraMarlinQuantInfo(
        w13_weight=_empty(2, 16, 32, dtype=torch.int32),
        w2_weight=_empty(2, 16, 32, dtype=torch.int32),
        w13_scales=_empty(2, 4, 64),
        w2_scales=_empty(2, 4, 64),
        weight_bits=4,
        num_local_experts=2,
        intermediate_size=768,
        hidden_size=1024,
    )
    assert marlin.w13_weight.shape[1] != 2 * marlin.intermediate_size
    assert (marlin.num_local_experts, marlin.intermediate_size, marlin.hidden_size) == (
        2,
        768,
        1024,
    )

    nvfp4 = SglLoraNvFp4QuantInfo(
        w13_weight=_empty(2, 1536, 512, dtype=torch.uint8),
        w2_weight=_empty(2, 1024, 384, dtype=torch.uint8),
        w13_blockscale=_empty(1, dtype=torch.float8_e4m3fn),
        w2_blockscale=_empty(1, dtype=torch.float8_e4m3fn),
        w13_alpha=_empty(2, dtype=torch.float32),
        w2_alpha=_empty(2, dtype=torch.float32),
        w13_input_scale=_empty(2, dtype=torch.float32),
        w2_input_scale=_empty(2, dtype=torch.float32),
        num_local_experts=2,
        intermediate_size=768,
        hidden_size=1024,
    )
    assert nvfp4.w13_weight.shape[-1] * 2 == nvfp4.hidden_size


@pytest.mark.parametrize(
    ("provider", "key", "row_domain", "w2_dtype"),
    [
        (DeepGemmBf16BaseGemm, "deepgemm_bf16", "expert_masked", torch.bfloat16),
        (
            DeepGemmFp8BaseGemm,
            "deepgemm_fp8_w8a8",
            "expert_masked",
            torch.float8_e4m3fn,
        ),
        (
            CuteDslNvFp4BaseGemm,
            "cutedsl_nvfp4_w4a4",
            "expert_masked",
            torch.uint8,
        ),
        (MarlinW4A16BaseGemm, "marlin_w4a16", "routed_pair", torch.bfloat16),
    ],
)
def test_provider_contract_is_explicit(provider, key, row_domain, w2_dtype):
    contract = provider.contract
    assert contract.key == key
    assert contract.row_domain == row_domain
    assert contract.w2_input_dtype == w2_dtype
    assert contract.gate_up_slices == ("gate", "up")
    assert contract.gate_first and not contract.interleaved
    assert contract.lora_delta_dtype == torch.bfloat16
    assert contract.lora_activation_dtype == torch.bfloat16
    assert contract.packed_topk_policy == "preserve_if_present"
    assert torch.float32 in contract.supported_output_dtypes
    contract.validate_output_dtype(torch.float32)
    with pytest.raises(ValueError, match="supported dtypes"):
        contract.validate_output_dtype(torch.float64)


def test_workspace_preserves_caller_packed_topk_identity():
    packed = torch.tensor([17, 23], dtype=torch.int32)
    workspace = BaseGemmWorkspace(
        hidden_permuted=_empty(2, 4),
        masked_m=torch.tensor([1], dtype=torch.int32),
        expected_m=1,
        src2dst=torch.tensor([0], dtype=torch.int32),
        m_max=1,
        packed_topk_ids=packed,
    )
    assert workspace.packed_topk_ids is packed
    assert workspace.hidden_permuted_owned


def test_resolver_accepts_an_explicit_injected_factory():
    quant_info = SglLoraBf16QuantInfo(
        w13_weight=_empty(1, 16, 8),
        w2_weight=_empty(1, 8, 8),
        num_local_experts=1,
        intermediate_size=8,
        hidden_size=8,
    )
    sentinel = object()
    calls = []

    def factory(received_quant_info, config, planner):
        calls.append((received_quant_info, config, planner))
        return sentinel

    config = SimpleNamespace(top_k=2)
    assert resolve_base_gemm(quant_info, config, provider_factory=factory) is sentinel
    assert calls == [(quant_info, config, None)]


def test_fp8_payload_keeps_block_shape_and_borrowed_scales():
    w13_scale = _empty(2, 4, 4, dtype=torch.float32)
    w2_scale = _empty(2, 4, 4, dtype=torch.float32)
    payload = SglLoraFp8QuantInfo(
        w13_weight=_empty(2, 256, 128, dtype=torch.float8_e4m3fn),
        w2_weight=_empty(2, 128, 128, dtype=torch.float8_e4m3fn),
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=(128, 128),
        num_local_experts=2,
        intermediate_size=128,
        hidden_size=128,
    )
    assert payload.w13_scale is w13_scale
    assert payload.w2_scale is w2_scale
    assert payload.block_shape == (128, 128)


def test_attach_time_marlin_selection_uses_resident_scheme_backend():
    class _Backend:
        def __init__(self, is_marlin):
            self._is_marlin = is_marlin

        def is_marlin(self):
            return self._is_marlin

    stock_info = SimpleNamespace(
        w13_qweight=_empty(2, 16, 32, dtype=torch.int32),
        w2_qweight=_empty(2, 16, 32, dtype=torch.int32),
        w13_scales=_empty(2, 4, 64),
        w2_scales=_empty(2, 4, 64),
        weight_bits=4,
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        w13_g_idx=None,
        w2_g_idx=None,
        w13_qzeros=None,
        w2_qzeros=None,
        w13_global_scale=None,
        w2_global_scale=None,
        w13_bias=None,
        w2_bias=None,
        expert_map=None,
        global_num_experts=-1,
        is_k_full=True,
    )
    scheme = SimpleNamespace(
        runner=SimpleNamespace(runner_backend=_Backend(True)),
        get_marlin_quant_info=lambda _layer: stock_info,
    )
    layer = SimpleNamespace(
        quant_method=object(),
        scheme=scheme,
        num_local_experts=2,
        intermediate_size_per_partition=768,
        hidden_size=1024,
    )

    payload = build_sgl_lora_quant_info(layer)
    assert isinstance(payload, SglLoraMarlinQuantInfo)
    assert payload.w13_weight is stock_info.w13_qweight

    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    layer.dispatcher = SimpleNamespace(local_expert_mapping=expert_map)
    layer.moe_runner_config = SimpleNamespace(num_experts=4)
    payload = build_sgl_lora_quant_info(layer)
    assert payload.expert_map is expert_map
    assert payload.global_num_experts == 4
    with pytest.raises(NotImplementedError, match="global-ID expert_map"):
        resolve_base_gemm(payload, SimpleNamespace(top_k=2))

    # A forwarding getter on a non-Marlin resident scheme is not evidence that
    # its tensors use the Marlin packing contract.
    scheme.runner.runner_backend = _Backend(False)
    with pytest.raises(NotImplementedError, match="no injectable SGL LoRA provider"):
        build_sgl_lora_quant_info(layer)
