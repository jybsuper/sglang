"""Matched MoE-LoRA boundaries built around the optimized CuTe grouped GEMM.

The CuTe kernel itself requires contiguous rows per problem.  MoE routing does
not naturally provide that layout, so this benchmark-owned module makes every
conversion explicit and timed:

* compact nonempty virtual-group route metadata is built once for K0;
* Triton gather/unpack kernels move only valid routed rows (no block padding);
* one CuTe grouped Tensor Core launch computes all nonempty groups;
* C2 uses two grouped launches around a compact activation kernel;
* down-B keeps the grouped result packed and finalizes through a pair-to-packed
  inverse map, avoiding a full delta scatter.

The hybrid label is intentional: a grouped GEMM cannot satisfy the raw routed
pair ABI without gather/finalize work.  Raw compute-only timings are useful as
an upper bound, but only ``invoke_boundary`` is eligible for dispatch decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from benchmark.kernels.lora_moe.cutedsl_grouped_tensorcore import (
    BlackwellGroupedGemmPlan,
    GroupedTactic,
)


@dataclass(slots=True)
class CompactGroupedRoute:
    """Nonempty virtual groups and both directions of the packed row map."""

    sorted_pair_ids: torch.Tensor
    sorted_groups: torch.Tensor
    active_groups: torch.Tensor
    pair_to_packed: torch.Tensor
    group_offsets: tuple[int, ...]
    group_counts: tuple[int, ...]
    num_pairs: int
    num_experts: int
    num_adapters: int
    local_expert_offset: int
    build_ms: float

    @property
    def valid_pairs(self) -> int:
        return int(self.sorted_pair_ids.numel())

    @property
    def active_group_count(self) -> int:
        return len(self.group_counts)

    def metadata(self) -> dict[str, object]:
        return {
            "num_pairs": self.num_pairs,
            "valid_pairs": self.valid_pairs,
            "active_groups": self.active_group_count,
            "group_counts": list(self.group_counts),
            "group_offsets": list(self.group_offsets),
            "num_experts": self.num_experts,
            "num_adapters": self.num_adapters,
            "local_expert_offset": self.local_expert_offset,
            "route_build_ms": self.build_ms,
        }


def build_compact_grouped_route(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_experts: int,
    num_adapters: int,
    local_expert_offset: int = 0,
) -> CompactGroupedRoute:
    """Sort only valid LoRA pairs and materialize a graph-stable inverse map."""

    if topk_ids.ndim != 2 or token_lora_mapping.ndim != 1:
        raise ValueError("topk_ids must be [T,top_k] and mapping must be [T]")
    if topk_ids.shape[0] != token_lora_mapping.shape[0]:
        raise ValueError("topk_ids and mapping token dimensions differ")
    started = time.perf_counter()
    top_k = topk_ids.shape[1]
    num_pairs = topk_ids.numel()
    experts = topk_ids.reshape(-1).to(torch.int64) - local_expert_offset
    adapters = token_lora_mapping[:, None].expand(-1, top_k).reshape(-1).to(torch.int64)
    valid = (
        (experts >= 0)
        & (experts < num_experts)
        & (adapters >= 0)
        & (adapters < num_adapters)
    )
    pair_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if not pair_ids.numel():
        raise ValueError("route contains no valid LoRA pairs")
    groups = adapters[pair_ids] * num_experts + experts[pair_ids]
    sorted_groups, order = torch.sort(groups, stable=True)
    sorted_pair_ids = pair_ids[order].contiguous()
    active_groups, counts = torch.unique_consecutive(sorted_groups, return_counts=True)
    counts_host = tuple(int(value) for value in counts.cpu().tolist())
    offsets = [0]
    for count in counts_host:
        offsets.append(offsets[-1] + count)
    pair_to_packed = torch.full(
        (num_pairs,), -1, dtype=torch.int32, device=topk_ids.device
    )
    pair_to_packed[sorted_pair_ids] = torch.arange(
        sorted_pair_ids.numel(), dtype=torch.int32, device=topk_ids.device
    )
    torch.cuda.synchronize(topk_ids.device)
    return CompactGroupedRoute(
        sorted_pair_ids=sorted_pair_ids,
        sorted_groups=sorted_groups.contiguous(),
        active_groups=active_groups.contiguous(),
        pair_to_packed=pair_to_packed,
        group_offsets=tuple(offsets),
        group_counts=counts_host,
        num_pairs=num_pairs,
        num_experts=num_experts,
        num_adapters=num_adapters,
        local_expert_offset=local_expert_offset,
        build_ms=(time.perf_counter() - started) * 1e3,
    )


@triton.jit
def _gather_rows_kernel(
    source_ptr,
    row_ids_ptr,
    packed_ptr,
    num_rows,
    width,
    stride_sm,
    stride_sn,
    stride_pm,
    stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < num_rows
    col_mask = cols < width
    source_rows = tl.load(row_ids_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    values = tl.load(
        source_ptr + source_rows[:, None] * stride_sm + cols[None, :] * stride_sn,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    tl.store(
        packed_ptr + rows[:, None] * stride_pm + cols[None, :] * stride_pn,
        values,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def gather_rows(
    source: torch.Tensor,
    row_ids: torch.Tensor,
    packed: torch.Tensor,
    *,
    block_m: int = 8,
    block_n: int = 128,
) -> None:
    if source.ndim != 2 or packed.ndim != 2:
        raise ValueError("source and packed must be 2D")
    if packed.shape != (row_ids.numel(), source.shape[1]):
        raise ValueError("packed shape does not match row ids/source width")
    _gather_rows_kernel[
        (triton.cdiv(row_ids.numel(), block_m), triton.cdiv(source.shape[1], block_n))
    ](
        source,
        row_ids,
        packed,
        row_ids.numel(),
        source.shape[1],
        source.stride(0),
        source.stride(1),
        packed.stride(0),
        packed.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _unpack_or_zero_kernel(
    packed_ptr,
    pair_to_packed_ptr,
    output_ptr,
    num_rows,
    width,
    stride_pm,
    stride_pn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < num_rows
    col_mask = cols < width
    packed_rows = tl.load(pair_to_packed_ptr + rows, mask=row_mask, other=-1).to(
        tl.int64
    )
    valid = packed_rows >= 0
    values = tl.load(
        packed_ptr
        + tl.maximum(packed_rows, 0)[:, None] * stride_pm
        + cols[None, :] * stride_pn,
        mask=row_mask[:, None] & valid[:, None] & col_mask[None, :],
        other=0.0,
    )
    tl.store(
        output_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,
        values,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def unpack_or_zero(
    packed: torch.Tensor,
    pair_to_packed: torch.Tensor,
    output: torch.Tensor,
    *,
    block_m: int = 8,
    block_n: int = 128,
) -> None:
    if output.shape != (pair_to_packed.numel(), packed.shape[1]):
        raise ValueError("output shape does not match inverse map/packed width")
    _unpack_or_zero_kernel[
        (
            triton.cdiv(output.shape[0], block_m),
            triton.cdiv(output.shape[1], block_n),
        )
    ](
        packed,
        pair_to_packed,
        output,
        output.shape[0],
        output.shape[1],
        packed.stride(0),
        packed.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _scatter_rows_kernel(
    packed_ptr,
    destination_rows_ptr,
    output_ptr,
    num_rows,
    width,
    stride_pm,
    stride_pn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < num_rows
    col_mask = cols < width
    destinations = tl.load(destination_rows_ptr + rows, mask=row_mask, other=0).to(
        tl.int64
    )
    values = tl.load(
        packed_ptr + rows[:, None] * stride_pm + cols[None, :] * stride_pn,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    tl.store(
        output_ptr + destinations[:, None] * stride_om + cols[None, :] * stride_on,
        values,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def scatter_rows(
    packed: torch.Tensor,
    destination_rows: torch.Tensor,
    output: torch.Tensor,
    *,
    width: int | None = None,
    block_m: int = 8,
    block_n: int = 128,
) -> None:
    width = packed.shape[1] if width is None else width
    _scatter_rows_kernel[
        (
            triton.cdiv(destination_rows.numel(), block_m),
            triton.cdiv(width, block_n),
        )
    ](
        packed,
        destination_rows,
        output,
        destination_rows.numel(),
        width,
        packed.stride(0),
        packed.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _packed_activation_kernel(
    base_ptr,
    delta_ptr,
    output_ptr,
    rows,
    logical_i,
    physical_i,
    stride_bm,
    stride_bn,
    stride_dm,
    stride_dn,
    stride_om,
    stride_on,
    ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    physical_mask = (row < rows) & (cols < physical_i)
    logical_mask = physical_mask & (cols < logical_i)
    value = tl.load(
        base_ptr + row * stride_bm + cols * stride_bn,
        mask=logical_mask,
        other=0.0,
    ).to(tl.float32)
    value += tl.load(
        delta_ptr + row * stride_dm + cols * stride_dn,
        mask=logical_mask,
        other=0.0,
    ).to(tl.float32)
    if ACTIVATION == 0:
        up = tl.load(
            base_ptr + row * stride_bm + (physical_i + cols) * stride_bn,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        up += tl.load(
            delta_ptr + row * stride_dm + (physical_i + cols) * stride_dn,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        activated = value * tl.sigmoid(value) * up
    else:
        activated = tl.maximum(value, 0.0)
        activated *= activated
    tl.store(
        output_ptr + row * stride_om + cols * stride_on,
        activated,
        mask=physical_mask,
    )


@triton.jit
def _base_only_activation_kernel(
    base_ptr,
    output_ptr,
    source_rows_ptr,
    topk_ids_ptr,
    mapping_ptr,
    num_pairs,
    top_k: tl.constexpr,
    local_expert_offset,
    num_experts,
    logical_i,
    physical_i,
    stride_bm,
    stride_bn,
    stride_om,
    stride_on,
    ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pair = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    expert = tl.load(topk_ids_ptr + pair).to(tl.int64) - local_expert_offset
    adapter = tl.load(mapping_ptr + pair // top_k).to(tl.int64)
    destination = tl.load(source_rows_ptr + pair).to(tl.int64)
    valid_pair = (
        (pair < num_pairs) & (expert >= 0) & (expert < num_experts) & (adapter < 0)
    )
    physical_mask = valid_pair & (cols < physical_i)
    logical_mask = physical_mask & (cols < logical_i)
    value = tl.load(
        base_ptr + destination * stride_bm + cols * stride_bn,
        mask=logical_mask,
        other=0.0,
    ).to(tl.float32)
    if ACTIVATION == 0:
        up = tl.load(
            base_ptr + destination * stride_bm + (physical_i + cols) * stride_bn,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        activated = value * tl.sigmoid(value) * up
    else:
        activated = tl.maximum(value, 0.0)
        activated *= activated
    tl.store(
        output_ptr + destination * stride_om + cols * stride_on,
        activated,
        mask=physical_mask,
    )


@triton.jit
def _packed_down_finalize_kernel(
    base_ptr,
    packed_delta_ptr,
    pair_to_packed_ptr,
    topk_weights_ptr,
    output_ptr,
    tokens,
    hidden,
    stride_bm,
    stride_bn,
    stride_dm,
    stride_dn,
    stride_tw_t,
    stride_tw_k,
    stride_om,
    stride_on,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    col_mask = (token < tokens) & (cols < hidden)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for slot in range(TOP_K):
        pair = token * TOP_K + slot
        packed = tl.load(pair_to_packed_ptr + pair).to(tl.int64)
        base = tl.load(
            base_ptr + pair * stride_bm + cols * stride_bn,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        delta = tl.load(
            packed_delta_ptr + tl.maximum(packed, 0) * stride_dm + cols * stride_dn,
            mask=col_mask & (packed >= 0),
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            topk_weights_ptr + token * stride_tw_t + slot * stride_tw_k,
            mask=token < tokens,
            other=0.0,
        ).to(tl.float32)
        acc += (base + delta) * weight
    tl.store(
        output_ptr + token * stride_om + cols * stride_on,
        acc,
        mask=col_mask,
    )


def _group_views(
    route: CompactGroupedRoute,
    packed_input: torch.Tensor,
    weight: torch.Tensor,
    packed_output: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    a_tensors: list[torch.Tensor] = []
    b_tensors: list[torch.Tensor] = []
    c_tensors: list[torch.Tensor] = []
    groups_host = route.active_groups.cpu().tolist()
    for index, group in enumerate(groups_host):
        start, end = route.group_offsets[index : index + 2]
        a_tensors.append(packed_input[start:end])
        b_tensors.append(weight[int(group)])
        c_tensors.append(packed_output[start:end])
    return a_tensors, b_tensors, c_tensors


class GroupedGemmBoundary:
    """One routed A/B site with compact gather and zero-producing unpack."""

    def __init__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        route: CompactGroupedRoute,
        *,
        input_pair_major: bool,
        top_k: int,
        tactic: GroupedTactic,
    ) -> None:
        self.x = x
        self.weight = weight
        self.output = output
        self.route = route
        self.input_pair_major = input_pair_major
        self.top_k = top_k
        self.input_rows = (
            route.sorted_pair_ids
            if input_pair_major
            else torch.div(route.sorted_pair_ids, top_k, rounding_mode="floor")
        ).contiguous()
        self.packed_input = torch.empty(
            (route.valid_pairs, x.shape[1]), dtype=x.dtype, device=x.device
        )
        self.packed_output = torch.empty(
            (route.valid_pairs, output.shape[1]),
            dtype=output.dtype,
            device=output.device,
        )
        views = _group_views(route, self.packed_input, weight, self.packed_output)
        self.plan = BlackwellGroupedGemmPlan(*views, tactic=tactic)
        self.prepack()

    def prepack(self) -> None:
        gather_rows(self.x, self.input_rows, self.packed_input)

    def invoke_compute_only(self) -> None:
        self.plan()

    def invoke_boundary(self) -> None:
        gather_rows(self.x, self.input_rows, self.packed_input)
        self.plan()
        unpack_or_zero(self.packed_output, self.route.pair_to_packed, self.output)

    def metadata(self) -> dict[str, object]:
        return {
            "boundary": "gather_grouped_gemm_unpack_or_zero",
            "input_pair_major": self.input_pair_major,
            "route": self.route.metadata(),
            "grouped_gemm": self.plan.metadata(),
        }


class GroupedC2Boundary:
    """Gate/value-B + activation + down-A using two grouped CuTe launches."""

    def __init__(
        self,
        *,
        value: torch.Tensor,
        value_a: torch.Tensor,
        value_b: torch.Tensor,
        down_a: torch.Tensor,
        act_out: torch.Tensor,
        down_rank: torch.Tensor,
        src2dst: torch.Tensor,
        topk_ids: torch.Tensor,
        mapping: torch.Tensor,
        route: CompactGroupedRoute,
        logical_i: int,
        physical_i: int,
        activation: str,
        gate_tactic: GroupedTactic,
        down_tactic: GroupedTactic,
    ) -> None:
        if activation not in ("swiglu", "relu2"):
            raise ValueError("activation must be swiglu or relu2")
        self.value = value
        self.value_a = value_a.reshape(route.num_pairs, -1)
        self.value_b = value_b.reshape(
            route.num_adapters * route.num_experts, value_b.shape[-2], value_b.shape[-1]
        )
        self.down_a = down_a.reshape(
            route.num_adapters * route.num_experts,
            down_a.shape[-2],
            down_a.shape[-1],
        )
        self.act_out = act_out
        self.down_rank = down_rank.reshape(route.num_pairs, -1)
        self.src2dst = src2dst.reshape(-1)
        self.topk_ids = topk_ids
        self.mapping = mapping
        self.route = route
        self.logical_i = logical_i
        self.physical_i = physical_i
        self.activation = activation
        self.slices = 2 if activation == "swiglu" else 1
        self.rank = self.down_rank.shape[1]
        self.top_k = topk_ids.shape[1]
        valid = route.valid_pairs
        self.destination_rows = self.src2dst[route.sorted_pair_ids].contiguous()
        self.packed_rank = torch.empty(
            (valid, self.slices * self.rank),
            dtype=value_a.dtype,
            device=value_a.device,
        )
        self.packed_base = torch.empty(
            (valid, self.slices * physical_i),
            dtype=value.dtype,
            device=value.device,
        )
        self.packed_delta = torch.empty_like(self.packed_base)
        self.packed_act = torch.empty(
            (valid, physical_i), dtype=value.dtype, device=value.device
        )
        self.packed_down = torch.empty(
            (valid, self.rank), dtype=down_rank.dtype, device=down_rank.device
        )

        gate_as: list[torch.Tensor] = []
        gate_bs: list[torch.Tensor] = []
        gate_cs: list[torch.Tensor] = []
        down_as: list[torch.Tensor] = []
        down_bs: list[torch.Tensor] = []
        down_cs: list[torch.Tensor] = []
        for index, group_value in enumerate(route.active_groups.cpu().tolist()):
            group = int(group_value)
            start, end = route.group_offsets[index : index + 2]
            for slice_index in range(self.slices):
                rank_start = slice_index * self.rank
                inter_start = slice_index * physical_i
                gate_as.append(
                    self.packed_rank[start:end, rank_start : rank_start + self.rank]
                )
                gate_bs.append(
                    self.value_b[
                        group, inter_start : inter_start + physical_i, : self.rank
                    ]
                )
                gate_cs.append(
                    self.packed_delta[start:end, inter_start : inter_start + physical_i]
                )
            down_as.append(self.packed_act[start:end])
            down_bs.append(self.down_a[group, : self.rank, :physical_i])
            down_cs.append(self.packed_down[start:end])
        self.gate_plan = BlackwellGroupedGemmPlan(
            gate_as, gate_bs, gate_cs, tactic=gate_tactic
        )
        self.down_plan = BlackwellGroupedGemmPlan(
            down_as, down_bs, down_cs, tactic=down_tactic
        )
        self.prepack()

    def prepack(self) -> None:
        gather_rows(self.value_a, self.route.sorted_pair_ids, self.packed_rank)
        gather_rows(self.value, self.destination_rows, self.packed_base)

    def _activate(self) -> None:
        _packed_activation_kernel[
            (self.route.valid_pairs, triton.cdiv(self.physical_i, 128))
        ](
            self.packed_base,
            self.packed_delta,
            self.packed_act,
            self.route.valid_pairs,
            self.logical_i,
            self.physical_i,
            self.packed_base.stride(0),
            self.packed_base.stride(1),
            self.packed_delta.stride(0),
            self.packed_delta.stride(1),
            self.packed_act.stride(0),
            self.packed_act.stride(1),
            ACTIVATION=0 if self.activation == "swiglu" else 1,
            BLOCK_N=128,
            num_warps=4,
            num_stages=2,
        )

    def _base_only_activation(self) -> None:
        _base_only_activation_kernel[
            (self.route.num_pairs, triton.cdiv(self.physical_i, 128))
        ](
            self.value,
            self.act_out,
            self.src2dst,
            self.topk_ids,
            self.mapping,
            self.route.num_pairs,
            top_k=self.top_k,
            local_expert_offset=self.route.local_expert_offset,
            num_experts=self.route.num_experts,
            logical_i=self.logical_i,
            physical_i=self.physical_i,
            stride_bm=self.value.stride(0),
            stride_bn=self.value.stride(1),
            stride_om=self.act_out.stride(0),
            stride_on=self.act_out.stride(1),
            ACTIVATION=0 if self.activation == "swiglu" else 1,
            BLOCK_N=128,
            num_warps=4,
            num_stages=2,
        )

    def invoke_compute_only(self) -> None:
        self.gate_plan()
        self._activate()
        self.down_plan()

    def invoke_boundary(self) -> None:
        gather_rows(self.value_a, self.route.sorted_pair_ids, self.packed_rank)
        gather_rows(self.value, self.destination_rows, self.packed_base)
        self.gate_plan()
        self._activate()
        self.down_plan()
        self._base_only_activation()
        scatter_rows(
            self.packed_act,
            self.destination_rows,
            self.act_out,
            width=self.physical_i,
        )
        unpack_or_zero(self.packed_down, self.route.pair_to_packed, self.down_rank)

    def metadata(self) -> dict[str, object]:
        return {
            "boundary": (
                "gather__grouped_gate_B__activation__grouped_down_A__"
                "base_fill_scatter_unpack"
            ),
            "activation": self.activation,
            "logical_i": self.logical_i,
            "physical_i": self.physical_i,
            "rank": self.rank,
            "slices": self.slices,
            "route": self.route.metadata(),
            "gate_grouped_gemm": self.gate_plan.metadata(),
            "down_grouped_gemm": self.down_plan.metadata(),
        }


class GroupedDownFinalizeBoundary:
    """Down-B grouped GEMM followed by packed token-owned finalize."""

    def __init__(
        self,
        *,
        rank_input: torch.Tensor,
        down_b: torch.Tensor,
        base_pairs: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
        route: CompactGroupedRoute,
        tactic: GroupedTactic,
    ) -> None:
        self.rank_input = rank_input
        self.down_b = down_b
        self.base_pairs = base_pairs
        self.topk_weights = topk_weights
        self.output = output
        self.route = route
        self.top_k = topk_weights.shape[1]
        self.packed_rank = torch.empty(
            (route.valid_pairs, rank_input.shape[1]),
            dtype=rank_input.dtype,
            device=rank_input.device,
        )
        self.packed_delta = torch.empty(
            (route.valid_pairs, base_pairs.shape[1]),
            dtype=torch.bfloat16,
            device=base_pairs.device,
        )
        views = _group_views(route, self.packed_rank, down_b, self.packed_delta)
        self.plan = BlackwellGroupedGemmPlan(*views, tactic=tactic)
        self.prepack()

    def prepack(self) -> None:
        gather_rows(self.rank_input, self.route.sorted_pair_ids, self.packed_rank)

    def invoke_compute_only(self) -> None:
        self.plan()

    def invoke_boundary(self) -> None:
        gather_rows(self.rank_input, self.route.sorted_pair_ids, self.packed_rank)
        self.plan()
        _packed_down_finalize_kernel[
            (
                self.output.shape[0],
                triton.cdiv(self.output.shape[1], 128),
            )
        ](
            self.base_pairs,
            self.packed_delta,
            self.route.pair_to_packed,
            self.topk_weights,
            self.output,
            self.output.shape[0],
            self.output.shape[1],
            self.base_pairs.stride(0),
            self.base_pairs.stride(1),
            self.packed_delta.stride(0),
            self.packed_delta.stride(1),
            self.topk_weights.stride(0),
            self.topk_weights.stride(1),
            self.output.stride(0),
            self.output.stride(1),
            TOP_K=self.top_k,
            BLOCK_H=128,
            num_warps=4,
            num_stages=2,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "boundary": "gather__grouped_down_B__packed_token_finalize",
            "route": self.route.metadata(),
            "grouped_gemm": self.plan.metadata(),
        }


__all__ = [
    "CompactGroupedRoute",
    "GroupedC2Boundary",
    "GroupedDownFinalizeBoundary",
    "GroupedGemmBoundary",
    "build_compact_grouped_route",
    "gather_rows",
    "scatter_rows",
    "unpack_or_zero",
]
