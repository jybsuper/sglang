import unittest

import torch

from sglang.srt.lora.sgl_lora.shared_experts import (
    MoeLoraExpertTopology,
    build_routed_expert_id_map,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMoeLoraSharedExpertIdMaps(unittest.TestCase):
    def test_global_shared_slots_are_excluded(self):
        topology = MoeLoraExpertTopology(
            num_routed_experts=8,
            num_fused_shared_experts=2,
        )
        self.assertTrue(
            torch.equal(
                build_routed_expert_id_map(topology),
                torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, -1, -1], dtype=torch.int32),
            )
        )

    def test_two_per_rank_shared_slots_global_factors(self):
        topology = MoeLoraExpertTopology(
            num_routed_experts=8,
            num_fused_shared_experts=2,
            ep_size=2,
            ep_rank=1,
            id_layout="global_per_rank_shared",
            factor_domain="global",
        )
        expected = torch.tensor(
            [0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1], dtype=torch.int32
        )
        self.assertTrue(torch.equal(build_routed_expert_id_map(topology), expected))

    def test_per_rank_shared_slots_local_factors_drop_other_rank(self):
        topology = MoeLoraExpertTopology(
            num_routed_experts=8,
            num_fused_shared_experts=1,
            ep_size=2,
            ep_rank=1,
            id_layout="global_per_rank_shared",
            factor_domain="local",
        )
        expected = torch.tensor([-1, -1, -1, -1, -1, 0, 1, 2, 3, -1], dtype=torch.int32)
        self.assertTrue(torch.equal(build_routed_expert_id_map(topology), expected))

    def test_local_contiguous_shared_slots_are_excluded(self):
        topology = MoeLoraExpertTopology(
            num_routed_experts=8,
            num_fused_shared_experts=2,
            ep_size=2,
            ep_rank=1,
            id_layout="local_contiguous",
            factor_domain="local",
        )
        expected = torch.tensor([0, 1, 2, 3, -1, -1], dtype=torch.int32)
        self.assertTrue(torch.equal(build_routed_expert_id_map(topology), expected))

    def test_global_contiguous_ids_with_local_factors_drop_other_rank(self):
        topology = MoeLoraExpertTopology(
            num_routed_experts=8,
            num_fused_shared_experts=2,
            ep_size=2,
            ep_rank=1,
            id_layout="global_contiguous",
            factor_domain="local",
        )
        expected = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3, -1, -1], dtype=torch.int32)
        self.assertTrue(torch.equal(build_routed_expert_id_map(topology), expected))


if __name__ == "__main__":
    unittest.main()
