import pytest
import torch

from sglang.srt.lora.sgl_lora.workspace import (
    MoeLoraWorkspacePlanner,
    estimate_bf16_moe_lora_workspace,
)


def _estimate(num_tokens: int = 8192):
    return estimate_bf16_moe_lora_workspace(
        num_tokens=num_tokens,
        top_k=8,
        hidden_size=7168,
        intermediate_size=2048,
        rank=64,
        num_local_experts=32,
        max_loras=8,
    )


def test_workspace_estimate_tracks_masked_rows_and_capture_lifetime():
    estimate = _estimate()

    assert estimate.m_max == 8448
    assert estimate.capture_peak_bytes > estimate.eager_peak_bytes
    assert estimate.eager_peak_bytes > estimate.routing_bytes > 0
    assert _estimate(16384).eager_peak_bytes > 1.9 * estimate.eager_peak_bytes


def test_workspace_estimate_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="must all be positive"):
        estimate_bf16_moe_lora_workspace(
            num_tokens=0,
            top_k=8,
            hidden_size=64,
            intermediate_size=32,
            rank=16,
            num_local_experts=4,
            max_loras=2,
        )


def test_workspace_planner_fails_early_with_chunking_remedy():
    estimate = _estimate()
    planner = MoeLoraWorkspacePlanner(
        memory_info=lambda _device: (estimate.eager_peak_bytes, 80 << 30),
        reserve_fraction=0,
        minimum_reserve_bytes=1,
    )

    with pytest.raises(MemoryError, match="--chunked-prefill-size"):
        planner.admit(
            estimate=estimate,
            device=torch.device("cuda", 0),
            capture=False,
            geometry_key=(32, 7168, 2048, 64, 8, 8),
        )


def test_workspace_planner_caches_successful_shape_across_layers():
    estimate = _estimate(256)
    calls = 0

    def memory_info(_device):
        nonlocal calls
        calls += 1
        return 40 << 30, 80 << 30

    planner = MoeLoraWorkspacePlanner(memory_info=memory_info)
    kwargs = dict(
        estimate=estimate,
        device=torch.device("cuda", 0),
        capture=False,
        geometry_key=(32, 7168, 2048, 64, 8, 8),
    )
    planner.admit(**kwargs)
    planner.admit(**kwargs)

    assert calls == 1


def test_workspace_planner_uses_preflight_snapshot_during_capture():
    estimate = _estimate(256)
    calls = 0

    def memory_info(_device):
        nonlocal calls
        calls += 1
        return 40 << 30, 80 << 30

    planner = MoeLoraWorkspacePlanner(memory_info=memory_info)
    device = torch.device("cuda", 0)
    planner.admit(
        estimate=estimate,
        device=device,
        capture=False,
        geometry_key=("same-provider",),
    )
    planner.admit(
        estimate=estimate,
        device=device,
        capture=True,
        geometry_key=("same-provider",),
        memory_query_safe=False,
    )

    assert calls == 1


def test_workspace_planner_requires_preflight_before_stream_capture():
    estimate = _estimate(256)
    planner = MoeLoraWorkspacePlanner(
        memory_info=lambda _device: pytest.fail("capture must not query CUDA memory")
    )

    with pytest.raises(RuntimeError, match="preflighted"):
        planner.admit(
            estimate=estimate,
            device=torch.device("cuda", 0),
            capture=True,
            geometry_key=("provider",),
            memory_query_safe=False,
        )
