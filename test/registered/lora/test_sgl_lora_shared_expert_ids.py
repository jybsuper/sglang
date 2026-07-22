import unittest

import torch

from sglang.srt.lora.sgl_lora.shared_experts import (
    MoeLoraExpertTopology,
    build_routed_expert_id_map,
)
from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
    _fused_virtual_topk_ids,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSglLoraSharedExpertVirtualIds(unittest.TestCase):
    def test_interleaved_two_shared_slots_never_alias_routed_factors(self):
        topology = MoeLoraExpertTopology(
            num_routed_experts=8,
            num_fused_shared_experts=2,
            ep_size=2,
            ep_rank=1,
            id_layout="global_per_rank_shared",
            factor_domain="global",
        )
        expert_id_map = build_routed_expert_id_map(topology, device="cuda")
        physical_ids = torch.tensor(
            [[0, 4, 5, 6, 9, 10, 11, -1]], dtype=torch.int32, device="cuda"
        )
        adapter = torch.tensor([1], dtype=torch.int32, device="cuda")
        got, mask, virtual_count = _fused_virtual_topk_ids(
            physical_ids,
            adapter,
            topology.num_factor_experts,
            shared_outer=False,
            max_loras=2,
            expert_id_map=expert_id_map,
        )
        expected = torch.tensor(
            [[8, -1, -1, 12, 15, -1, -1, -1]],
            dtype=torch.int32,
            device="cuda",
        )
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        self.assertTrue(bool(mask.item()))
        self.assertEqual(virtual_count, 16)

    def test_local_factor_map_drops_non_home_rank(self):
        topology = MoeLoraExpertTopology(
            num_routed_experts=8,
            num_fused_shared_experts=1,
            ep_size=2,
            ep_rank=1,
            id_layout="global_per_rank_shared",
            factor_domain="local",
        )
        expert_id_map = build_routed_expert_id_map(topology, device="cuda")
        physical_ids = torch.tensor([[0, 4, 5, 8, 9]], dtype=torch.int32, device="cuda")
        adapter = torch.tensor([0], dtype=torch.int32, device="cuda")
        got, _, _ = _fused_virtual_topk_ids(
            physical_ids,
            adapter,
            topology.num_factor_experts,
            False,
            1,
            expert_id_map=expert_id_map,
        )
        expected = torch.tensor([[-1, -1, 0, 3, -1]], dtype=torch.int32, device="cuda")
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
