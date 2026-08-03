"""MoE-LoRA runner for the SGL LoRA execution engine.

``SglMoeLoraRunner`` is the single object the LoRA layer wrapper holds for one
MoE layer. Construction admits the resident provider contract and binds the
base provider; ``run`` executes the pipeline. No stock ``MoeRunner`` is
involved — the per-quant base stages live behind :class:`MoeBaseProvider`,
and this class owns the LoRA route views, the LoRA kernels, and every pipeline
buffer.

The default remains the serial correctness pipeline.  A typed
``MoeLoraExecutionPlan`` may instead force one of the retained fusion and
overlap candidates; every consumed stage then has exactly one owner and every
required route representation is built once:

    gate/up LoRA A  (grouped_lora_a: token-major hidden -> pair-major rank)
    gate/up LoRA B  (one-launch sliced B -> canonical [gate | up] delta)
    S1 prepare      (provider permute to its physical row domain)
    S2 gateup       (provider grouped GEMM)
    S3 act          (base + delta -> activation; writes provider rows and,
                     when required, a canonical pair-major down-A source)
    down LoRA A     (grouped_lora_a, canonical pairs or provider-mapped rows)
    down LoRA B     (one-launch sliced B -> unweighted pair delta [T, K, H])
    S4 down         (provider grouped GEMM)
    S5 finalize     (provider fixed-order top-k reduction; router coefficient
                     and routed scaling applied EXACTLY ONCE over
                     base + pair delta, at the provider-declared coefficient
                     precision)

Every batch runs this one LoRA-capable topology — base-only, mixed, and active
alike — so they share a single graph shape. Inactive assignments ride sentinel
routes and contribute exact zeros rather than being diverted to another path.

Base rows: serving gives the base model a REAL resident slot whose factors are
zero-filled and whose ``adapter_enabled`` entry is 0. Batch preparation
canonicalizes such assignments to the ``-1`` execution sentinel before any
layer runs — otherwise base rows build routed work against zero weights, which
is numerically harmless but inflates route padding, group counts, and every
LoRA GEMM's row count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.sgl_lora.base_gemm_provider.base import (
    MappedLoraAInput,
    MoeBaseProvider,
)
from sglang.srt.lora.sgl_lora.execution_plan import (
    ActivationFamily,
    EarlyOverlap,
    FactorLayout,
    FactorOwnership,
    FactorSite,
    FinalizeFamily,
    LateOverlap,
    LoraAFamily,
    LoraASpec,
    LoraBSpec,
    MiddleFamily,
    MoeLoraExecutionPlan,
    MoeLoraFactorLayout,
)
from sglang.srt.lora.sgl_lora.launch_config import (
    PROVISIONAL_LAUNCH_CONFIG,
    MoeLoraLaunchConfig,
)
from sglang.srt.lora.sgl_lora.lora_a import run_lora_a
from sglang.srt.lora.sgl_lora.lora_b import run_lora_b
from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo
from sglang.srt.lora.sgl_lora.route_factory import (
    MoeLoraRoutes,
    build_routes,
)
from sglang.srt.lora.sgl_lora.routing import RouteView
from sglang.srt.lora.sgl_lora.workspace import MoeLoraWorkspace, run_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass(slots=True)
class _GateLoraState:
    rank: torch.Tensor | None = None
    delta: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _DownAInput:
    """Standalone down-A source without exposing provider workspace details."""

    rows: torch.Tensor
    pair_to_row: torch.Tensor | None = None


_EARLY_PARALLEL_REGION = {
    EarlyOverlap.GATE_A: "early_gate_a",
    EarlyOverlap.GATE_A_B: "early_gate_a_b",
}
_LATE_PARALLEL_REGION = {
    LateOverlap.DOWN_A: "late_down_a",
    LateOverlap.DOWN_B: "late_down_b",
    LateOverlap.DOWN_A_B: "late_down_a_b",
    LateOverlap.SHARED_FINALIZE: "late_shared_finalize",
}


class SglMoeLoraBatch(msgspec.Struct, kw_only=True):
    """The per-batch state the MoE-LoRA runner actually consumes.

    Narrow by design: the legacy ``LoRAInfo`` carries ~18 fields for the old
    kernels, and passing it wholesale would make it impossible to see what this
    runner depends on. ``token_slots`` holds canonical active physical slot
    IDs, with every inactive assignment represented by the ``-1`` sentinel.
    """

    gate_up_lora_a: torch.Tensor  # [L_cap, E_f, slices*R_phys, H]
    gate_up_lora_b: torch.Tensor  # [L_cap, E_local, slices*I, R_phys]
    down_lora_a: torch.Tensor  # [L_cap, E_local, R_phys, I]
    down_lora_b: torch.Tensor  # [L_cap, E_f_down, H, R_phys]
    token_slots: torch.Tensor  # [T] int, physical slot per token (-1 = base)
    adapter_enabled: torch.Tensor | None  # [L_cap], 0 marks an inactive slot
    physical_rank: int
    factor_layout: MoeLoraFactorLayout
    use_cuda_graph: bool = False
    is_prefill: bool = False
    has_active_lora: bool = True

    @property
    def slot_capacity(self) -> int:
        return self.gate_up_lora_a.shape[0]


class SglMoeLoraRunner:
    """One MoE layer's SGL LoRA execution state and pipeline."""

    def __init__(
        self,
        *,
        provider: MoeBaseProvider,
        top_k: int,
        routed_scaling_factor: float | None,
        execution_plan: MoeLoraExecutionPlan,
        activation: ActivationFamily = ActivationFamily.SWIGLU,
        launch_config: MoeLoraLaunchConfig = PROVISIONAL_LAUNCH_CONFIG,
        workspace: MoeLoraWorkspace | None = None,
        side_stream_priority: int = 0,
    ) -> None:
        self.provider = provider
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.activation = activation
        self.execution_plan = execution_plan
        self.launch_config = launch_config
        self.workspace = workspace if workspace is not None else MoeLoraWorkspace()
        self.side_stream_priority = int(side_stream_priority)
        self._bound_factor_layout: MoeLoraFactorLayout | None = None
        self._bound_physical_rank: int | None = None
        self._bound_slot_capacity: int | None = None

    @classmethod
    def from_layer(
        cls,
        base_layer: FusedMoE,
        *,
        provider_name: str,
        execution_plan: MoeLoraExecutionPlan,
        launch_config: MoeLoraLaunchConfig = PROVISIONAL_LAUNCH_CONFIG,
        workspace: MoeLoraWorkspace | None = None,
    ) -> SglMoeLoraRunner:
        """Admit the layer's resident state and bind a base provider to it."""
        cls._admit(base_layer)
        config = base_layer.moe_runner_config
        return cls(
            provider=cls._build_provider(base_layer, provider_name=provider_name),
            # Layer-static routing scalars, read once rather than per forward.
            top_k=int(config.top_k),
            routed_scaling_factor=config.routed_scaling_factor,
            activation=(
                ActivationFamily.SWIGLU
                if config.activation == "silu"
                else ActivationFamily.RELU2
            ),
            execution_plan=execution_plan,
            launch_config=launch_config,
            workspace=workspace,
        )

    # ---- attach-time admission and validation ---------------------------

    @staticmethod
    def _admit(base_layer: FusedMoE) -> None:
        """Reject any resident state this engine does not actually consume.

        The base layer picks its runner backend, reformats resident weights,
        configures dispatch, and decides routed-scaling ownership BEFORE the
        LoRA layer attaches, so all of that is validated together here rather
        than assumed.
        """
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
        from sglang.srt.layers.moe.utils import get_moe_runner_backend
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        if not isinstance(base_layer.quant_method, UnquantizedFusedMoEMethod):
            raise NotImplementedError(
                "sgl_lora currently supports unquantized BF16 MoE only"
            )
        if (
            not isinstance(base_layer.dispatcher, StandardDispatcher)
            or base_layer.w13_weight.dtype != torch.bfloat16
            or base_layer.w2_weight.dtype != torch.bfloat16
        ):
            raise NotImplementedError(
                "sgl_lora BF16 currently requires Standard dispatch and a "
                "resident BF16 provider"
            )

        resident_backend = (
            base_layer.quant_method.runner.runner_backend
            if base_layer.quant_method.runner is not None
            else get_moe_runner_backend()
        )
        # Both shipped providers (DeepGEMM and CuTeDSL) consume the DeepGEMM
        # backend's canonical resident weight layout ([E, 2I, H] gate-first
        # BF16 with EP-local expert IDs), so admission is gated on that
        # backend AND on DeepGEMM being usable regardless of which provider
        # the env selects; a Triton-resident provider is separate later work.
        if not resident_backend.is_deep_gemm():
            raise NotImplementedError(
                "sgl_lora BF16 currently requires --moe-runner-backend "
                "deep_gemm (canonical gate-first [E, 2I, H] BF16 weights and "
                f"EP-local expert IDs); this layer resolved to "
                f"{resident_backend}"
            )
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            raise NotImplementedError(
                "sgl_lora BF16 requires a usable JIT DeepGEMM build: every "
                "base provider consumes the DeepGEMM-resident weight layout"
            )
        if base_layer.dispatcher.skip_local_expert_mapping:
            raise NotImplementedError(
                "sgl_lora BF16 requires EP-local expert IDs at the runner "
                "boundary, but this dispatcher keeps global IDs"
            )
        if base_layer.should_fuse_routed_scaling_factor_in_topk:
            raise NotImplementedError(
                "sgl_lora BF16 applies routed scaling exactly once in its own "
                "finalize; this layer already folds it into the top-k weights"
            )

        config = base_layer.moe_runner_config
        supported_activation = (config.activation == "silu" and config.is_gated) or (
            config.activation == "relu2" and not config.is_gated
        )
        if (
            not supported_activation
            or config.gemm1_alpha is not None
            or config.gemm1_clamp_limit is not None
            or config.swiglu_limit is not None
            or config.apply_router_weight_on_input
            or config.no_combine
            or config.num_fused_shared_experts
        ):
            raise NotImplementedError(
                "sgl_lora BF16 supports canonical gated SiLU or non-gated "
                "ReLU2 without fused shared experts, with route weighting "
                "owned by finalize"
            )

    @staticmethod
    def select_provider_cls(
        provider_name: str,
    ) -> type[MoeBaseProvider]:
        """Resolve the provider explicitly selected by the serving policy."""
        from sglang.srt.lora.sgl_lora.base_gemm_provider.deep_gemm_bf16 import (
            DeepGemmBf16Provider,
        )

        if provider_name == "deepgemm":
            return DeepGemmBf16Provider
        if provider_name == "cutedsl":
            from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_bf16 import (
                CuteDslBf16Provider,
            )

            return CuteDslBf16Provider
        raise ValueError(
            f"unknown SGL LoRA MoE provider {provider_name!r}; expected "
            "'deepgemm' or 'cutedsl'"
        )

    @classmethod
    def _build_provider(
        cls,
        base_layer: FusedMoE,
        *,
        provider_name: str,
    ) -> MoeBaseProvider:
        return cls.select_provider_cls(provider_name)(
            SglLoraBf16QuantInfo(
                w13_weight=base_layer.w13_weight,
                w2_weight=base_layer.w2_weight,
                num_local_experts=int(base_layer.num_local_experts),
                intermediate_size=int(base_layer.w2_weight.shape[2]),
                hidden_size=int(base_layer.w2_weight.shape[1]),
            )
        )

    def validate_factors(
        self,
        *,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        down_lora_a: torch.Tensor,
        down_lora_b: torch.Tensor,
        factor_layout: MoeLoraFactorLayout,
    ) -> None:
        """Check the immutable LoRA-weight contract once, when weights bind.

        Dtype and expert domain cannot change between forwards, so validating
        per forward would only add launch overhead.
        """
        factor_layout.validate()
        expert_count = self.provider.num_local_experts
        factors_by_name = {
            "gate_up_lora_a": gate_up_lora_a,
            "gate_up_lora_b": gate_up_lora_b,
            "down_lora_a": down_lora_a,
            "down_lora_b": down_lora_b,
        }
        for name, weight in factors_by_name.items():
            if weight.ndim != 4:
                raise ValueError(
                    f"{name} must be [slots, experts, rows, columns], got "
                    f"{tuple(weight.shape)}"
                )
            if not weight.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous; flattening a strided resident "
                    "factor would allocate inside each forward"
                )

        slot_capacities = {weight.shape[0] for weight in factors_by_name.values()}
        if len(slot_capacities) != 1 or next(iter(slot_capacities)) < 1:
            raise ValueError(
                "all SGL MoE-LoRA factors must share one positive slot capacity"
            )
        factor_devices = {weight.device for weight in factors_by_name.values()}
        if len(factor_devices) != 1:
            raise ValueError(
                f"SGL MoE-LoRA factors span devices {sorted(map(str, factor_devices))}"
            )

        expected = (
            (
                1
                if factor_layout.gate_up_a is FactorOwnership.SHARED_OUTER
                else expert_count
            ),
            expert_count,
            expert_count,
            1 if factor_layout.down_b is FactorOwnership.SHARED_OUTER else expert_count,
        )
        factors = tuple(factors_by_name.values())
        actual = tuple(weight.shape[1] for weight in factors)
        if actual != expected:
            raise ValueError(
                "SGL LoRA factor domains do not match the resident provider: "
                f"expected {expected}, got {actual}"
            )
        expected_dtype = self.provider.contract.lora_delta_dtype
        for name, weight in factors_by_name.items():
            if weight.dtype != expected_dtype:
                raise TypeError(
                    f"sgl_lora requires {expected_dtype} {name}, got {weight.dtype}"
                )

        physical_rank = int(down_lora_a.shape[2])
        if physical_rank < 1:
            raise ValueError("the resident physical LoRA rank must be positive")
        slices = self.provider.gate_up_slices
        hidden = self.provider.hidden_size
        intermediate = self.provider.intermediate_size
        expected_shapes = {
            "gate_up_lora_a": (slices * physical_rank, hidden),
            "gate_up_lora_b": (slices * intermediate, physical_rank),
            "down_lora_a": (physical_rank, intermediate),
            "down_lora_b": (hidden, physical_rank),
        }
        for name, weight in factors_by_name.items():
            if tuple(weight.shape[2:]) != expected_shapes[name]:
                raise ValueError(
                    f"{name} trailing shape must be {expected_shapes[name]}, "
                    f"got {tuple(weight.shape[2:])}"
                )

        self.execution_plan.validate_factor_layout(factor_layout)
        self._validate_plan_provider(self.execution_plan)
        self._warm_parallel_regions(
            self.execution_plan,
            next(iter(factor_devices)),
        )
        self._bound_factor_layout = factor_layout
        self._bound_physical_rank = physical_rank
        self._bound_slot_capacity = next(iter(slot_capacities))

    def _warm_parallel_regions(
        self,
        plan: MoeLoraExecutionPlan,
        device: torch.device,
    ) -> None:
        """Create the exact overlap resources before any CUDA graph capture.

        Execution plans are immutable after factor binding. A future dynamic
        selector must warm the union of every region it may select here rather
        than lazily creating streams or events during capture.
        """

        if device.type != "cuda":
            return
        for region in (
            _EARLY_PARALLEL_REGION.get(plan.early_overlap),
            _LATE_PARALLEL_REGION.get(plan.late_overlap),
        ):
            if region is not None:
                self.workspace.warm_parallel_region(
                    device,
                    region,
                    priority=self.side_stream_priority,
                )

    def _validate_plan_provider(self, plan: MoeLoraExecutionPlan) -> None:
        """Reject unsupported provider/plan pairs before forward CUDA work."""
        plan.validate()
        if plan.middle.activation is not self.activation:
            raise ValueError(
                f"plan activation {plan.middle.activation.value} does not match "
                f"resident layer activation {self.activation.value}"
            )
        expected_slices = 2 if self.activation is ActivationFamily.SWIGLU else 1
        if self.provider.gate_up_slices != expected_slices:
            raise ValueError(
                f"provider exposes {self.provider.gate_up_slices} gate/up "
                f"slices but {self.activation.value} needs {expected_slices}"
            )

        if plan.middle.family is not MiddleFamily.MATERIALIZED:
            family, implementation = self._middle_implementation(plan)
            if not self.provider.supports_fused_middle(
                family,
                activation=self._activation_name(),
                implementation=implementation,
            ):
                raise NotImplementedError(
                    f"{self.provider.contract.key} does not implement "
                    f"{family}/{implementation}"
                )
        if plan.finalize.family is not FinalizeFamily.MATERIALIZED:
            family, implementation = self._finalize_implementation(plan)
            ownership = plan.finalize.consumed_down_b
            assert ownership is not None
            ownership_name = (
                "shared"
                if ownership.ownership is FactorOwnership.SHARED_OUTER
                else "per_expert"
            )
            if not self.provider.supports_fused_finalize(
                family,
                ownership_name,
                implementation=implementation,
            ):
                raise NotImplementedError(
                    f"{self.provider.contract.key} does not implement "
                    f"{family}/{ownership_name}/{implementation}"
                )

    @staticmethod
    def _middle_implementation(
        plan: MoeLoraExecutionPlan,
    ) -> tuple[str, str]:
        return plan.middle.family.value, "triton"

    @staticmethod
    def _finalize_implementation(
        plan: MoeLoraExecutionPlan,
    ) -> tuple[str, str]:
        return plan.finalize.family.value, "triton"

    # ---- forward --------------------------------------------------------

    def run(
        self,
        dispatch_output: StandardDispatchOutput,
        batch: SglMoeLoraBatch,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> StandardCombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        provider = self.provider
        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        assert TopKOutputChecker.format_is_standard(topk_output)
        topk_ids = topk_output.topk_ids

        output_dtype = hidden_states.dtype if output_dtype is None else output_dtype
        provider.validate_runtime_inputs(hidden_states, output_dtype=output_dtype)
        num_tokens = self._checked_token_count(hidden_states, batch)
        self.workspace.begin_forward(graph_mode=batch.use_cuda_graph)
        plan = self._plan_for(batch)
        self.launch_config.validate_for_plan(plan)
        routes = build_routes(
            plan,
            topk_ids=topk_ids,
            token_slots=batch.token_slots,
            num_local_experts=provider.num_local_experts,
            max_loras=batch.slot_capacity,
            block_size=self.launch_config.routing_block_size,
            gate_a_block_size=self.launch_config.gate_a_routing_block_size,
            workspace=self.workspace,
        )

        gate, ws, gateup_out = self._run_early(
            plan,
            routes,
            hidden_states,
            topk_ids,
            batch,
            num_tokens,
        )
        act_out, down_a_input, down_rank = self._run_middle(
            plan,
            routes,
            ws,
            gateup_out,
            gate,
            topk_ids,
            batch,
            num_tokens,
        )
        output = self._allocate_output(
            num_tokens=num_tokens,
            dtype=output_dtype,
            device=act_out.device,
        )
        down_out, down_rank, down_delta, token_rank = self._run_late(
            plan,
            routes,
            ws,
            act_out,
            down_a_input,
            down_rank,
            topk_output,
            batch,
            num_tokens,
        )
        output = self._run_finalize(
            plan,
            routes,
            ws,
            output,
            down_out,
            down_rank,
            down_delta,
            token_rank,
            topk_output,
            batch,
            num_tokens,
        )
        return StandardCombineInput(hidden_states=output)

    def _checked_token_count(
        self, hidden_states: torch.Tensor, batch: SglMoeLoraBatch
    ) -> int:
        num_tokens = hidden_states.shape[0]
        if batch.token_slots.ndim != 1 or batch.token_slots.shape[0] != num_tokens:
            raise RuntimeError(
                "sgl_lora token/adapter assignment does not match the MoE "
                f"token domain: mapping has {batch.token_slots.shape[0]} rows "
                f"but the runner received {num_tokens}. Gather/remap "
                "assignments before MoE-DP execution."
            )
        if batch.token_slots.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "sgl_lora token_slots must be int32 or int64, got "
                f"{batch.token_slots.dtype}"
            )
        if batch.token_slots.device != hidden_states.device:
            raise ValueError(
                "sgl_lora token_slots and hidden states must share a device"
            )
        if batch.adapter_enabled is not None:
            if (
                batch.adapter_enabled.ndim != 1
                or batch.adapter_enabled.shape[0] != batch.slot_capacity
            ):
                raise ValueError(
                    "adapter_enabled must have one entry per resident LoRA slot"
                )
            if batch.adapter_enabled.device != hidden_states.device:
                raise ValueError(
                    "adapter_enabled and hidden states must share a device"
                )
        return num_tokens

    def _plan_for(self, batch: SglMoeLoraBatch) -> MoeLoraExecutionPlan:
        if self._bound_factor_layout is None:
            self.validate_factors(
                gate_up_lora_a=batch.gate_up_lora_a,
                gate_up_lora_b=batch.gate_up_lora_b,
                down_lora_a=batch.down_lora_a,
                down_lora_b=batch.down_lora_b,
                factor_layout=batch.factor_layout,
            )
        elif self._bound_factor_layout != batch.factor_layout:
            raise ValueError(
                "resident MoE-LoRA factor layout changed after layer binding"
            )
        if batch.physical_rank != self._bound_physical_rank:
            raise ValueError(
                f"batch physical_rank {batch.physical_rank} does not match "
                f"resident rank {self._bound_physical_rank}"
            )
        if batch.slot_capacity != self._bound_slot_capacity:
            raise ValueError(
                f"batch slot capacity {batch.slot_capacity} does not match "
                f"resident capacity {self._bound_slot_capacity}"
            )
        return self.execution_plan

    def _route_for_a(self, spec: LoraASpec, routes: MoeLoraRoutes) -> RouteView:
        if spec.family is LoraAFamily.TOKEN_DEDUP_GROUPED:
            if routes.shared_token is None:
                raise ValueError("shared token route was not constructed")
            return routes.shared_token
        if spec.family is LoraAFamily.INDEXED:
            return routes.raw(spec.ownership)
        if (
            spec.site is FactorSite.GATE_UP
            and spec.ownership is FactorOwnership.PER_EXPERT
            and routes.gate_a_aligned_per_expert is not None
        ):
            return routes.gate_a_aligned_per_expert
        return routes.aligned(spec.ownership)

    @staticmethod
    def _route_for_b(spec: LoraBSpec, routes: MoeLoraRoutes) -> RouteView:
        return routes.aligned(spec.ownership)

    def _run_a(
        self,
        spec: LoraASpec,
        input: torch.Tensor,
        weight: torch.Tensor,
        routes: MoeLoraRoutes,
        name: str,
        *,
        input_row_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        route = self._route_for_a(spec, routes)
        if input_row_map is not None:
            if not (
                spec.site is FactorSite.DOWN and spec.family is LoraAFamily.GROUPED
            ):
                raise ValueError(
                    "mapped provider rows are supported only by standalone "
                    "grouped down-A"
                )
            if (
                input_row_map.ndim != 1
                or input_row_map.dtype != torch.int32
                or input_row_map.device != input.device
                or not input_row_map.is_contiguous()
                or input_row_map.numel() != route.topk_ids.numel()
            ):
                raise ValueError(
                    "mapped down-A pair_to_row must be one contiguous int32 "
                    "entry per canonical routed pair on the input device"
                )
        num_tokens = (
            input.shape[0]
            if spec.site is FactorSite.GATE_UP
            else (
                route.topk_ids.shape[0]
                if input_row_map is not None
                else input.shape[0] // self.top_k
            )
        )
        rows = (
            num_tokens
            if spec.output_layout is FactorLayout.TOKEN_MAJOR
            else num_tokens * self.top_k
        )
        output = self.workspace.tensor(
            f"{name}:output",
            (rows, weight.shape[1]),
            dtype=self.provider.contract.lora_delta_dtype,
            device=input.device,
        )
        config = self.launch_config.for_a(spec.site)
        return run_lora_a(
            spec,
            input=input,
            weight=weight,
            output=output,
            routing=route,
            config=config,
            input_row_map=input_row_map,
        )

    def _run_b(
        self,
        spec: LoraBSpec,
        bridge: torch.Tensor,
        weight: torch.Tensor,
        destination: torch.Tensor,
        routes: MoeLoraRoutes,
    ) -> torch.Tensor:
        route = self._route_for_b(spec, routes)
        config = self.launch_config.for_b(spec.site)
        if "BLOCK_SIZE_M" in config:
            configured_block = int(config["BLOCK_SIZE_M"])
            if configured_block != route.block_size:
                raise ValueError(
                    "LoRA-B consumes the aligned route's exact BLOCK_SIZE_M: "
                    "config declares "
                    f"{configured_block}, route uses {route.block_size}"
                )
        if spec.site is FactorSite.GATE_UP:
            width = weight.shape[1] // self.provider.gate_up_slices
            offsets = tuple(
                slice_id * width for slice_id in range(self.provider.gate_up_slices)
            )
        else:
            offsets = (0,)
        run_lora_b(
            bridge=bridge,
            weight=weight,
            destination=destination,
            routing=route,
            destination_offsets=offsets,
            config=config,
            intermediate_top_k=(
                self.top_k if spec.input_layout is FactorLayout.TOKEN_MAJOR else 1
            ),
        )

    def _run_gate_a(
        self,
        plan: MoeLoraExecutionPlan,
        routes: MoeLoraRoutes,
        hidden_states: torch.Tensor,
        batch: SglMoeLoraBatch,
    ) -> torch.Tensor:
        return self._run_a(
            plan.gate_a,
            hidden_states,
            batch.gate_up_lora_a.flatten(0, 1),
            routes,
            "gate_a",
        )

    def _run_gate_b(
        self,
        plan: MoeLoraExecutionPlan,
        routes: MoeLoraRoutes,
        rank: torch.Tensor,
        batch: SglMoeLoraBatch,
        num_tokens: int,
    ) -> torch.Tensor:
        if plan.gate_b is None:
            raise ValueError("the selected middle owns gate B")
        delta = self.workspace.tensor(
            "gate_b:delta",
            (
                num_tokens * self.top_k,
                self.provider.gate_up_slices * self.provider.intermediate_size,
            ),
            dtype=self.provider.contract.lora_delta_dtype,
            device=rank.device,
        )
        self._run_b(
            plan.gate_b,
            rank,
            batch.gate_up_lora_b.flatten(0, 1),
            delta,
            routes,
        )
        return delta

    def _run_base_gateup(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[object, torch.Tensor]:
        provider = self.provider
        ws = provider.prepare(
            hidden_states,
            topk_ids,
            self.top_k,
            self.workspace,
        )
        gateup_out = self.workspace.tensor(
            "base:gateup",
            provider.gateup_out_shape(ws),
            dtype=provider.contract.gate_up_output_dtype,
            device=hidden_states.device,
        )
        provider.gateup(ws, gateup_out)
        provider.release_prepared_inputs(ws)
        return ws, gateup_out

    def _run_early(
        self,
        plan: MoeLoraExecutionPlan,
        routes: MoeLoraRoutes,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        batch: SglMoeLoraBatch,
        num_tokens: int,
    ) -> tuple[_GateLoraState, object, torch.Tensor]:
        state = _GateLoraState()

        def gate_a() -> None:
            state.rank = self._run_gate_a(
                plan,
                routes,
                hidden_states,
                batch,
            )

        def gate_b() -> None:
            if state.rank is None:
                raise RuntimeError("gate B ran before gate A")
            state.delta = self._run_gate_b(
                plan,
                routes,
                state.rank,
                batch,
                num_tokens,
            )

        def base() -> tuple[object, torch.Tensor]:
            return self._run_base_gateup(hidden_states, topk_ids)

        if plan.early_overlap is EarlyOverlap.NONE:
            gate_a()
            if plan.gate_b is not None:
                gate_b()
            ws, gateup = base()
        elif plan.early_overlap is EarlyOverlap.GATE_A:
            ws, gateup = run_parallel(
                self.workspace,
                name=_EARLY_PARALLEL_REGION[EarlyOverlap.GATE_A],
                device=hidden_states.device,
                compute=base,
                side=gate_a,
                side_priority=self.side_stream_priority,
            )
            if plan.gate_b is not None:
                gate_b()
        else:

            def gate_a_b() -> None:
                gate_a()
                gate_b()

            ws, gateup = run_parallel(
                self.workspace,
                name=_EARLY_PARALLEL_REGION[EarlyOverlap.GATE_A_B],
                device=hidden_states.device,
                compute=base,
                side=gate_a_b,
                side_priority=self.side_stream_priority,
            )
        return state, ws, gateup

    def _activation_name(self) -> str:
        return "silu_mul" if self.activation is ActivationFamily.SWIGLU else "relu2"

    def _run_middle(
        self,
        plan: MoeLoraExecutionPlan,
        routes: MoeLoraRoutes,
        ws,
        gateup_out: torch.Tensor,
        gate: _GateLoraState,
        topk_ids: torch.Tensor,
        batch: SglMoeLoraBatch,
        num_tokens: int,
    ) -> tuple[torch.Tensor, _DownAInput | None, torch.Tensor | None]:
        provider = self.provider
        act_out = self.workspace.tensor(
            "middle:act_masked",
            provider.act_out_shape(ws),
            dtype=provider.contract.lora_activation_dtype,
            device=gateup_out.device,
        )
        exposes_pair_activation = True
        mapped_down_a: MappedLoraAInput | None = None
        if (
            plan.middle.family is MiddleFamily.B_ACTIVATION
            and plan.down_a is not None
            and plan.down_a.family is LoraAFamily.GROUPED
        ):
            mapped_down_a = provider.mapped_down_lora_a_input(ws, act_out)
            if mapped_down_a is not None:
                exposes_pair_activation = False
        act_pairs = (
            self.workspace.tensor(
                "middle:act_pairs",
                (num_tokens, self.top_k, provider.intermediate_size),
                dtype=provider.contract.lora_activation_dtype,
                device=gateup_out.device,
            )
            if exposes_pair_activation
            else None
        )
        if plan.middle.family is MiddleFamily.MATERIALIZED:
            assert act_pairs is not None
            if gate.delta is None:
                raise RuntimeError("materialized middle requires gate/up delta")
            provider.act_with_delta(
                ws,
                gateup_out,
                gate.delta.view(
                    num_tokens,
                    self.top_k,
                    provider.gate_up_slices * provider.intermediate_size,
                ),
                topk_ids,
                act_out,
                act_pairs,
                activation=self._activation_name(),
            )
            return act_out, _DownAInput(act_pairs), None

        consumed_route = plan.middle.consumed_gate_b
        assert consumed_route is not None
        route = routes.aligned(consumed_route.ownership)
        family, implementation = self._middle_implementation(plan)
        provider.run_fused_middle(
            ws,
            family,
            implementation=implementation,
            activation=self._activation_name(),
            base_gateup=gateup_out,
            act_masked=act_out,
            act_pairs=act_pairs,
            routing=route,
            config=self.launch_config.for_middle(plan.middle.family),
            bridge_gateup=gate.rank,
            b_gate_up=batch.gate_up_lora_b.flatten(0, 1),
            bridge_top_k=(
                self.top_k
                if plan.gate_a.output_layout is FactorLayout.TOKEN_MAJOR
                else 1
            ),
        )
        if mapped_down_a is not None:
            down_a_input = _DownAInput(
                mapped_down_a.rows,
                mapped_down_a.pair_to_row,
            )
        elif act_pairs is not None:
            down_a_input = _DownAInput(act_pairs)
        else:
            down_a_input = None
        return act_out, down_a_input, None

    def _run_down_a(
        self,
        plan: MoeLoraExecutionPlan,
        routes: MoeLoraRoutes,
        down_a_input: _DownAInput,
        batch: SglMoeLoraBatch,
    ) -> torch.Tensor:
        if plan.down_a is None:
            raise ValueError("the selected middle owns down A")
        return self._run_a(
            plan.down_a,
            down_a_input.rows.view(-1, self.provider.intermediate_size),
            batch.down_lora_a.flatten(0, 1),
            routes,
            "down_a",
            input_row_map=down_a_input.pair_to_row,
        )

    def _run_down_b(
        self,
        plan: MoeLoraExecutionPlan,
        routes: MoeLoraRoutes,
        rank: torch.Tensor,
        batch: SglMoeLoraBatch,
    ) -> torch.Tensor:
        if plan.down_b is None:
            raise ValueError("the selected finalizer owns down B")
        delta = self.workspace.tensor(
            "down_b:delta",
            (rank.shape[0], self.provider.hidden_size),
            dtype=self.provider.contract.lora_delta_dtype,
            device=rank.device,
        )
        self._run_b(
            plan.down_b,
            rank,
            batch.down_lora_b.flatten(0, 1),
            delta,
            routes,
        )
        return delta

    def _run_base_down(
        self,
        ws,
        act_out: torch.Tensor,
    ) -> torch.Tensor:
        down_out = self.workspace.tensor(
            "base:down",
            self.provider.down_out_shape(ws),
            dtype=torch.bfloat16,
            device=act_out.device,
        )
        self.provider.down(ws, act_out, down_out)
        return down_out

    def _shared_finalize_route(
        self, plan: MoeLoraExecutionPlan, routes: MoeLoraRoutes
    ) -> RouteView:
        consumed = plan.finalize.consumed_down_b
        if consumed is None:
            raise ValueError("shared finalizer does not own down B")
        return routes.raw(consumed.ownership)

    def _run_late(
        self,
        plan: MoeLoraExecutionPlan,
        routes: MoeLoraRoutes,
        ws,
        act_out: torch.Tensor,
        down_a_input: _DownAInput | None,
        down_rank: torch.Tensor | None,
        topk_output,
        batch: SglMoeLoraBatch,
        num_tokens: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        rank_holder = {"value": down_rank}
        delta_holder: dict[str, torch.Tensor | None] = {"value": None}
        token_rank: torch.Tensor | None = None

        def down_a() -> None:
            if down_a_input is None:
                raise RuntimeError("standalone down A requires pair activation")
            rank_holder["value"] = self._run_down_a(
                plan,
                routes,
                down_a_input,
                batch,
            )

        def down_b() -> None:
            rank = rank_holder["value"]
            if rank is None:
                raise RuntimeError("down B ran before down A")
            delta_holder["value"] = self._run_down_b(
                plan,
                routes,
                rank,
                batch,
            )

        def base() -> torch.Tensor:
            return self._run_base_down(ws, act_out)

        if plan.late_overlap is LateOverlap.NONE:
            if rank_holder["value"] is None:
                down_a()
            if plan.down_b is not None:
                down_b()
            down_out = base()
        elif plan.late_overlap is LateOverlap.DOWN_A:
            down_out = run_parallel(
                self.workspace,
                name=_LATE_PARALLEL_REGION[LateOverlap.DOWN_A],
                device=act_out.device,
                compute=base,
                side=down_a,
                side_priority=self.side_stream_priority,
            )
            if plan.down_b is not None:
                down_b()
        elif plan.late_overlap is LateOverlap.DOWN_B:
            if rank_holder["value"] is None:
                down_a()
            down_out = run_parallel(
                self.workspace,
                name=_LATE_PARALLEL_REGION[LateOverlap.DOWN_B],
                device=act_out.device,
                compute=base,
                side=down_b,
                side_priority=self.side_stream_priority,
            )
        elif plan.late_overlap is LateOverlap.DOWN_A_B:

            def down_a_b() -> None:
                down_a()
                down_b()

            down_out = run_parallel(
                self.workspace,
                name=_LATE_PARALLEL_REGION[LateOverlap.DOWN_A_B],
                device=act_out.device,
                compute=base,
                side=down_a_b,
                side_priority=self.side_stream_priority,
            )
        else:
            if rank_holder["value"] is None:
                down_a()
            rank = rank_holder["value"]
            assert rank is not None
            route = self._shared_finalize_route(plan, routes)
            token_rank = self.workspace.tensor(
                "finalize:shared_token_rank",
                (num_tokens, rank.shape[1]),
                dtype=rank.dtype,
                device=rank.device,
            )
            _, implementation = self._finalize_implementation(plan)

            def rank_reduce() -> None:
                self.provider.run_shared_rank_reduce(
                    ws,
                    implementation=implementation,
                    bridge=rank,
                    routing=route,
                    topk_weights=topk_output.topk_weights,
                    routed_scaling_factor=self.routed_scaling_factor,
                    token_rank=token_rank,
                    config=self.launch_config.shared_finalize["reduce"],
                )

            down_out = run_parallel(
                self.workspace,
                name=_LATE_PARALLEL_REGION[LateOverlap.SHARED_FINALIZE],
                device=act_out.device,
                compute=base,
                side=rank_reduce,
                side_priority=self.side_stream_priority,
            )

        rank = rank_holder["value"]
        if rank is None:
            raise RuntimeError("execution plan did not produce down-A output")
        return down_out, rank, delta_holder["value"], token_rank

    def _allocate_output(
        self,
        *,
        num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            use_symmetric_memory,
        )
        from sglang.srt.layers.dp_attention import is_allocation_symmetric

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            return torch.empty(
                (num_tokens, self.provider.hidden_size),
                dtype=dtype,
                device=device,
            )

    def _run_finalize(
        self,
        plan: MoeLoraExecutionPlan,
        routes: MoeLoraRoutes,
        ws,
        output: torch.Tensor,
        down_out: torch.Tensor,
        down_rank: torch.Tensor,
        down_delta: torch.Tensor | None,
        token_rank: torch.Tensor | None,
        topk_output,
        batch: SglMoeLoraBatch,
        num_tokens: int,
    ) -> torch.Tensor:
        if plan.finalize.family is FinalizeFamily.MATERIALIZED:
            if down_delta is None:
                raise RuntimeError(
                    "materialized finalize requires a pair-major down delta"
                )
            self.provider.finalize(
                ws,
                down_out,
                topk_output.topk_ids,
                topk_output.topk_weights,
                self.routed_scaling_factor,
                output,
                pair_delta=down_delta.view(
                    num_tokens, self.top_k, self.provider.hidden_size
                ),
            )
            return output

        consumed = plan.finalize.consumed_down_b
        assert consumed is not None
        route = routes.raw(consumed.ownership)
        b_down = batch.down_lora_b.flatten(0, 1)
        family, implementation = self._finalize_implementation(plan)
        if token_rank is None:
            self.provider.run_shared_rank_finalize(
                ws,
                implementation=implementation,
                down_masked=down_out,
                bridge=down_rank,
                b_down=b_down,
                routing=route,
                topk_weights=topk_output.topk_weights,
                routed_scaling_factor=self.routed_scaling_factor,
                output=output,
                token_rank=self.workspace.tensor(
                    "finalize:shared_token_rank",
                    (num_tokens, down_rank.shape[1]),
                    dtype=down_rank.dtype,
                    device=down_rank.device,
                ),
                config=self.launch_config.shared_finalize,
            )
        else:
            self.provider.finish_shared_rank_finalize(
                ws,
                implementation=implementation,
                down_masked=down_out,
                b_down=b_down,
                routing=route,
                topk_weights=topk_output.topk_weights,
                routed_scaling_factor=self.routed_scaling_factor,
                output=output,
                token_rank=token_rank,
                config=self.launch_config.shared_finalize["tail"],
            )
        return output
