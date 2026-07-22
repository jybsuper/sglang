import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.lora.sgl_lora.moe_lora_runner import run_sgl_lora_moe
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class _FakeBaseGemm:
    def __init__(
        self,
        *,
        num_tokens: int,
        top_k: int,
        hidden: int,
        inter: int,
        routed_scaling_factor=None,
    ):
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.hidden = hidden
        self.inter = inter
        self.routed_scaling_factor = routed_scaling_factor
        self.calls = []

    def prepare(self, hidden_states, topk_ids, top_k):
        self.calls.append("prepare")
        assert top_k == self.top_k
        return SimpleNamespace(hidden_permuted=hidden_states.clone())

    def gateup_out_shape(self, _ws):
        return (self.num_tokens, self.top_k, 2 * self.inter)

    def gateup(self, _ws, output):
        self.calls.append("gateup")
        output.fill_(1.0)

    def act_out_shape(self, _ws):
        return (self.num_tokens, self.top_k, self.inter)

    def act_with_delta(
        self,
        _ws,
        gateup_out,
        gate_up_delta,
        _topk_ids,
        act_out,
        activation_lora_input,
    ):
        self.calls.append("act")
        assert torch.all(gateup_out == 1.0)
        assert torch.all(gate_up_delta == 0.25)
        act_out.fill_(0.5)
        activation_lora_input.fill_(0.5)

    def down_out_shape(self, _ws):
        return (self.num_tokens, self.top_k, self.hidden)

    def down(self, _ws, act_out, output):
        self.calls.append("down")
        assert torch.all(act_out == 0.5)
        output.fill_(1.5)

    def finalize(
        self,
        _ws,
        down_out,
        _topk_ids,
        _topk_weights,
        routed_scaling_factor,
        output,
    ):
        self.calls.append("finalize")
        assert torch.all(down_out == 1.5)
        assert routed_scaling_factor == self.routed_scaling_factor
        output.fill_(2.0)


def test_runner_wires_gate_up_and_production_down_options(monkeypatch):
    """Cover the complete serial runner and its two virtual-expert call sites."""
    import sglang.srt.distributed as distributed
    import sglang.srt.layers.dp_attention as dp_attention
    import sglang.srt.lora.sgl_lora.triton_ops.virtual_experts as virtual_experts
    import sglang.srt.utils as srt_utils

    device = "cuda"
    num_tokens, top_k = 2, 2
    num_experts, hidden, inter, rank = 3, 32, 48, 16
    calls = []

    routed_scaling_factor = 1.75

    def fake_merged_experts_fused_moe_lora_add(**kwargs):
        calls.append(kwargs)
        if kwargs["num_output_slices"] == 2:
            assert not kwargs["mul_routed_weight"]
            assert not kwargs.get("fuse_sum_all_reduce", False)
            assert kwargs["stage"] == "all"
            assert kwargs["intermediate_buffer"].shape == (
                num_tokens,
                top_k,
                2 * rank,
            )
            kwargs["output"].fill_(0.25)
        else:
            assert kwargs["mul_routed_weight"]
            assert kwargs["fuse_sum_all_reduce"]
            assert not kwargs["fuse_add_to_output"]
            assert kwargs["hidden_states"].shape == (
                num_tokens * top_k,
                inter,
            )
            assert torch.all(kwargs["hidden_states"] == 0.5)
            torch.testing.assert_close(
                kwargs["topk_weights"],
                topk_weights * routed_scaling_factor,
                rtol=0,
                atol=0,
            )
            kwargs["output"].add_(1.0)

    monkeypatch.setattr(
        virtual_experts,
        "merged_experts_fused_moe_lora_add",
        fake_merged_experts_fused_moe_lora_add,
    )
    monkeypatch.setattr(
        distributed, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(dp_attention, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(srt_utils, "dispose_tensor", lambda _tensor: None)
    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    topk_ids = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32, device=device
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=torch.empty(0, device=device),
        ),
    )
    token_lora_mapping = torch.tensor([0, -1], dtype=torch.int32, device=device)
    lora_info = SimpleNamespace(
        token_lora_mapping=token_lora_mapping,
        gate_up_lora_a_weights=torch.empty(
            1, num_experts, 2 * rank, hidden, dtype=torch.bfloat16, device=device
        ),
        gate_up_lora_b_weights=torch.empty(
            1,
            num_experts,
            2 * inter,
            rank,
            dtype=torch.bfloat16,
            device=device,
        ),
        down_lora_a_weights=torch.empty(
            1, num_experts, rank, inter, dtype=torch.bfloat16, device=device
        ),
        down_lora_b_weights=torch.empty(
            1, num_experts, hidden, rank, dtype=torch.bfloat16, device=device
        ),
        max_lora_rank=rank,
        experts_shared_outer_loras=False,
    )
    quant_info = SimpleNamespace(
        num_local_experts=num_experts,
        intermediate_size=inter,
        hidden_size=hidden,
    )
    runner_config = SimpleNamespace(
        top_k=top_k, routed_scaling_factor=routed_scaling_factor
    )
    base = _FakeBaseGemm(
        num_tokens=num_tokens,
        top_k=top_k,
        hidden=hidden,
        inter=inter,
        routed_scaling_factor=routed_scaling_factor,
    )

    result = run_sgl_lora_moe(
        dispatch_output,
        quant_info,
        runner_config,
        lora_info,
        base,
        two_stream_enabled=False,
    )

    assert base.calls == ["prepare", "gateup", "act", "down", "finalize"]
    assert len(calls) == 2
    assert calls[0]["token_lora_mapping"].data_ptr() == token_lora_mapping.data_ptr()
    assert calls[1]["token_lora_mapping"].data_ptr() == token_lora_mapping.data_ptr()
    torch.testing.assert_close(
        result.hidden_states,
        torch.full_like(result.hidden_states, 3.0),
        rtol=0,
        atol=0,
    )

    lora_info.token_lora_mapping = token_lora_mapping[:1]
    with pytest.raises(RuntimeError, match="token/adapter assignment"):
        run_sgl_lora_moe(
            dispatch_output,
            quant_info,
            runner_config,
            lora_info,
            base,
            two_stream_enabled=False,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
