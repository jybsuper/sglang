"""Per-forward packed tile schedules for the direct-schedule CuTeDSL GEMMs.

The direct-schedule kernel consumes a packed int32 per tile —
``expert | token_cluster << 10 | output_cluster << 20`` — plus a device tile
count. Both GEMM stages' schedules derive from the SAME ``masked_m`` in ONE
launch, which is the section 45 dual-ownership rule: a schedule built from any
other row-count source than the masked_m the GEMMs will read is silent
corruption, so this module never accepts row counts through a second path.

Per-expert tile ordering follows the study's heuristic: when an expert has no
more token clusters than output clusters, tiles iterate token-fastest
(consecutive tiles share the output cluster); otherwise output-fastest. The
loop bound is a runtime scalar, so no per-m_max recompilation.

ABI bounds are enforced HERE. Packing and the device decoder import the same
field shifts and masks from ``schedule_abi`` so a width change cannot silently
change only one side. Expert counts up to 1024, token clusters <= 1024 per
expert, output clusters <= 2048 (a non-negative-word policy), and the
worst-case capacity must fit int32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.schedule_abi import (
    MAX_EXPERTS,
    MAX_OUTPUT_CLUSTERS,
    MAX_TOKEN_CLUSTERS,
    OUTPUT_CLUSTER_SHIFT,
    TOKEN_CLUSTER_SHIFT,
)


@triton.jit
def _dual_schedule_kernel(
    masked_m_ptr,
    schedule1_ptr,
    tiles1_ptr,
    schedule2_ptr,
    tiles2_ptr,
    token_width,
    out_clusters1,
    out_clusters2,
    EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    ENTRY_BLOCK: tl.constexpr,
    TOKEN_SHIFT: tl.constexpr,
    OUTPUT_SHIFT: tl.constexpr,
):
    """One program per (expert, stage); entries written as vector blocks.

    The first version ran ONE program with a serial per-entry loop across all
    experts — 17 us at prefill scale, larger than the GEMM saving it enabled
    (the measured cause of the pipeline's 0.98x prefill). Each program now
    recomputes its own exclusive prefix from masked_m (O(E) vectorized loads,
    E <= 1024) and fills its entries ENTRY_BLOCK at a time.
    """
    expert = tl.program_id(0)
    stage = tl.program_id(1)
    out_clusters = tl.where(stage == 0, out_clusters1, out_clusters2)

    offs = tl.arange(0, BLOCK_EXPERTS)
    all_rows = tl.load(masked_m_ptr + offs, mask=offs < EXPERTS, other=0)
    all_entries = tl.cdiv(all_rows, token_width) * out_clusters
    begin = tl.sum(tl.where(offs < expert, all_entries, 0))
    if expert == 0:
        total = tl.sum(tl.where(offs < EXPERTS, all_entries, 0))
        if stage == 0:
            tl.store(tiles1_ptr, total)
        else:
            tl.store(tiles2_ptr, total)

    rows = tl.load(masked_m_ptr + expert)
    token_clusters = tl.cdiv(rows, token_width)
    safe_tc = tl.maximum(token_clusters, 1)
    entries = token_clusters * out_clusters
    token_major = token_clusters > out_clusters
    for base in range(0, entries, ENTRY_BLOCK):
        local = base + tl.arange(0, ENTRY_BLOCK)
        valid = local < entries
        oc_a = local // safe_tc
        tc_a = local - oc_a * safe_tc
        tc_b = local // out_clusters
        oc_b = local - tc_b * out_clusters
        tc_i = tl.where(token_major, tc_b, tc_a)
        oc_i = tl.where(token_major, oc_b, oc_a)
        packed = expert | (tc_i << TOKEN_SHIFT) | (oc_i << OUTPUT_SHIFT)
        if stage == 0:
            tl.store(schedule1_ptr + begin + local, packed, mask=valid)
        else:
            tl.store(schedule2_ptr + begin + local, packed, mask=valid)


def dual_stage_schedule_capacities(
    *,
    num_experts: int,
    m_max: int,
    token_width: int,
    n_gemm1: int,
    n_gemm2: int,
    output_width: int,
    cluster_shape_mn: tuple[int, int] = (1, 1),
    use_2cta_instrs: bool = False,
) -> tuple[int, int]:
    """Validate the packed ABI and return both worst-case entry capacities.

    The provider uses the same result to reserve address-stable graph buffers
    that :func:`build_dual_stage_schedules` later fills.  Keeping geometry
    validation here prevents the allocator and builder from drifting.

    ``cluster_shape_mn`` / ``use_2cta_instrs`` are the CALLER'S kernel config,
    taken only to be REJECTED unless trivial. This builder emits indices at
    CTA-tile granularity (derived from ``token_width`` / ``output_width``),
    while the device scheduler treats them as CLUSTER indices and expands them
    by ``cluster_shape_mn``. Those agree only at a 1x1 cluster with 1-CTA MMA;
    with a real cluster the schedule would over-enumerate and address tiles
    out of range, silently — no device assertion exists to catch it. Supporting
    clusters means dividing the counts here by the cluster extents, which is a
    deliberate change with its own boundary cases, not a default.
    """
    if tuple(cluster_shape_mn) != (1, 1) or use_2cta_instrs:
        raise ValueError(
            "the packed direct schedule is emitted at CTA-tile granularity and "
            "the device expands it by cluster_shape_mn, so only a (1, 1) "
            "cluster with 1-CTA MMA is representable; got "
            f"cluster_shape_mn={tuple(cluster_shape_mn)}, "
            f"use_2cta_instrs={use_2cta_instrs}"
        )
    out_clusters1 = (n_gemm1 + output_width - 1) // output_width
    out_clusters2 = (n_gemm2 + output_width - 1) // output_width
    max_token_clusters = (m_max + token_width - 1) // token_width
    if num_experts > MAX_EXPERTS:
        raise ValueError(
            f"direct schedule packs expert INDICES in 10 bits (counts up to "
            f"{MAX_EXPERTS}); got {num_experts}"
        )
    if max_token_clusters > MAX_TOKEN_CLUSTERS:
        raise ValueError(
            f"m_max={m_max} at token width {token_width} needs "
            f"{max_token_clusters} token clusters; the packing holds "
            f"{MAX_TOKEN_CLUSTERS}. Admission must pick a wider tile."
        )
    if max(out_clusters1, out_clusters2) > MAX_OUTPUT_CLUSTERS:
        raise ValueError(
            f"{max(out_clusters1, out_clusters2)} output clusters exceed the "
            f"{MAX_OUTPUT_CLUSTERS} the non-negative-word policy allows"
        )
    # The kernel's begin/total arithmetic is int32 over the worst-case
    # capacity; per-field width caps do not imply this bound, so check it
    # directly (reviewer finding).
    capacity = num_experts * max_token_clusters * max(out_clusters1, out_clusters2)
    if capacity > 2**31 - 1:
        raise ValueError(
            f"worst-case schedule capacity {capacity} overflows the builder's "
            "int32 prefix arithmetic"
        )
    return (
        num_experts * max_token_clusters * out_clusters1,
        num_experts * max_token_clusters * out_clusters2,
    )


def _schedule_output(
    output: torch.Tensor | None,
    *,
    elements: int,
    device: torch.device,
    label: str,
) -> torch.Tensor:
    if output is None:
        return torch.empty(elements, dtype=torch.int32, device=device)
    if (
        output.shape != (elements,)
        or output.dtype != torch.int32
        or output.device != device
        or not output.is_contiguous()
    ):
        raise ValueError(f"{label} must be contiguous int32 [{elements}] on {device}")
    return output


def _tile_count_output(
    output: torch.Tensor | None,
    *,
    device: torch.device,
    label: str,
) -> torch.Tensor:
    return _schedule_output(output, elements=1, device=device, label=label)


def build_dual_stage_schedules(
    masked_m: torch.Tensor,
    *,
    m_max: int,
    token_width: int,
    n_gemm1: int,
    n_gemm2: int,
    output_width: int,
    cluster_shape_mn: tuple[int, int] = (1, 1),
    use_2cta_instrs: bool = False,
    schedule1_out: torch.Tensor | None = None,
    tiles1_out: torch.Tensor | None = None,
    schedule2_out: torch.Tensor | None = None,
    tiles2_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(schedule1, tiles1, schedule2, tiles2)`` for one forward.

    Capacity is the worst case over this forward's ``m_max`` (every expert
    full), so the kernel's device-side counts can never overflow the buffers.
    Optional outputs let a runner retain exact-shape buffers across CUDA-graph
    replay; omitted outputs preserve the standalone eager API.
    """
    num_experts = masked_m.numel()
    capacity1, capacity2 = dual_stage_schedule_capacities(
        num_experts=num_experts,
        m_max=m_max,
        token_width=token_width,
        n_gemm1=n_gemm1,
        n_gemm2=n_gemm2,
        output_width=output_width,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    out_clusters1 = (n_gemm1 + output_width - 1) // output_width
    out_clusters2 = (n_gemm2 + output_width - 1) // output_width

    device = masked_m.device
    schedule1 = _schedule_output(
        schedule1_out,
        elements=capacity1,
        device=device,
        label="schedule1_out",
    )
    schedule2 = _schedule_output(
        schedule2_out,
        elements=capacity2,
        device=device,
        label="schedule2_out",
    )
    tiles1 = _tile_count_output(
        tiles1_out,
        device=device,
        label="tiles1_out",
    )
    tiles2 = _tile_count_output(
        tiles2_out,
        device=device,
        label="tiles2_out",
    )
    _dual_schedule_kernel[(num_experts, 2)](
        masked_m,
        schedule1,
        tiles1,
        schedule2,
        tiles2,
        token_width,
        out_clusters1,
        out_clusters2,
        EXPERTS=num_experts,
        BLOCK_EXPERTS=max(triton.next_power_of_2(num_experts), 2),
        ENTRY_BLOCK=128,
        TOKEN_SHIFT=TOKEN_CLUSTER_SHIFT,
        OUTPUT_SHIFT=OUTPUT_CLUSTER_SHIFT,
    )
    return schedule1, tiles1, schedule2, tiles2
