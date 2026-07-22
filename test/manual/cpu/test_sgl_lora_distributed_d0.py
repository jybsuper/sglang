from benchmark.kernels.lora_moe.bench_distributed import (
    TOPOLOGIES,
    adapter_ids,
    expected_group_ranks,
    global_expert_for_route,
    logical_coordinates,
)


def test_tp8_ep2_dp2_rank_coordinates_and_groups():
    spec = TOPOLOGIES["tp8_ep2_dp2"]
    expected = {
        0: ((0, 2), (0, 1), (0, 4)),
        1: ((1, 3), (0, 1), (1, 5)),
        2: ((0, 2), (2, 3), (2, 6)),
        3: ((1, 3), (2, 3), (3, 7)),
        4: ((4, 6), (4, 5), (0, 4)),
        5: ((5, 7), (4, 5), (1, 5)),
        6: ((4, 6), (6, 7), (2, 6)),
        7: ((5, 7), (6, 7), (3, 7)),
    }
    for rank, (ep, moe_tp, dp) in expected.items():
        groups = expected_group_ranks(rank, spec)
        assert groups["moe_ep"] == ep
        assert groups["moe_tp"] == moe_tp
        assert groups["moe_dp"] == dp
        coords = logical_coordinates(rank, spec)
        assert rank == (
            coords.dp_rank * spec.ep * spec.moe_tp
            + coords.ep_rank * spec.moe_tp
            + coords.moe_tp_rank
        )


def test_tp4_ep4_route_modes_have_requested_locality():
    spec = TOPOLOGIES["tp4_ep4_dp1"]
    num_experts = 8
    local_experts = num_experts // spec.ep
    for origin in range(spec.ep):
        for token in range(8):
            all_local = {
                global_expert_for_route(
                    token=token,
                    slot=slot,
                    origin_ep_rank=origin,
                    route_mode="all_local",
                    num_experts=num_experts,
                    ep_size=spec.ep,
                )
                // local_experts
                for slot in range(2)
            }
            no_local = {
                global_expert_for_route(
                    token=token,
                    slot=slot,
                    origin_ep_rank=origin,
                    route_mode="no_local",
                    num_experts=num_experts,
                    ep_size=spec.ep,
                )
                // local_experts
                for slot in range(2)
            }
            mixed = [
                global_expert_for_route(
                    token=token,
                    slot=slot,
                    origin_ep_rank=origin,
                    route_mode="mixed",
                    num_experts=num_experts,
                    ep_size=spec.ep,
                )
                // local_experts
                for slot in range(2)
            ]
            assert all_local == {origin}
            assert origin not in no_local
            assert mixed[0] == origin
            assert mixed[1] != origin


def test_adapter_modes_and_uneven_dp_counts_are_explicit():
    assert adapter_ids(6, "active", 2) == [0, 1, 0, 1, 0, 1]
    assert adapter_ids(6, "mixed", 2) == [-1, 1, 0, -1, 0, 1]
    assert adapter_ids(6, "base", 2) == [-1] * 6
    assert TOPOLOGIES["tp4_ep1_dp4"].token_counts == (1, 3, 7, 13)
    assert TOPOLOGIES["tp8_ep2_dp2"].token_counts == (5, 11)
