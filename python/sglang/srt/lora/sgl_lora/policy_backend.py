"""Production policy boundary for the BF16 SGL MoE-LoRA engine.

The selector is intentionally pure, while :class:`SglMoeLoraPolicyBackend`
owns all stateful execution objects.  Every runner that the selector may
return is constructed and factor-validated when the resident LoRA buffers are
bound.  A forward, including CUDA-graph capture, therefore performs only a
pure policy lookup followed by a dictionary lookup; it can never initialize a
provider, stream, event, or runner lazily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.lora.sgl_lora.execution_plan import (
    ActivationFamily,
    MoeLoraFactorLayout,
)
from sglang.srt.lora.sgl_lora.moe_lora_runner import (
    SglMoeLoraBatch,
    SglMoeLoraRunner,
)
from sglang.srt.lora.sgl_lora.selector import (
    DeviceArchitecture,
    PolicyChoice,
    PolicyInput,
    PolicyMode,
    ProviderKey,
    architecture_for_device,
    choices_for,
    select_policy,
)
from sglang.srt.lora.sgl_lora.workspace import MoeLoraWorkspace

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


class SglMoeLoraPolicyBackend:
    """One MoE layer's prevalidated policy choices and shared workspace.

    The layer's activation and GPU architecture are static.  Factor layout is
    learned when the LoRA memory pool binds its resident buffers; that is the
    last point at which this object may construct runners or providers.
    """

    def __init__(
        self,
        base_layer: FusedMoE,
        *,
        architecture: DeviceArchitecture,
        base_gemm_provider: ProviderKey,
        activation: ActivationFamily,
        workspace: MoeLoraWorkspace | None = None,
    ) -> None:
        self._base_layer = base_layer
        self.architecture = architecture
        self.base_gemm_provider = base_gemm_provider
        self.activation = activation
        self.workspace = workspace if workspace is not None else MoeLoraWorkspace()
        self._factor_layout: MoeLoraFactorLayout | None = None
        self._runners: dict[str, SglMoeLoraRunner] = {}
        self._choices: tuple[PolicyChoice, ...] = ()

    @classmethod
    def from_layer(
        cls,
        base_layer: FusedMoE,
        *,
        base_gemm_provider: ProviderKey,
        workspace: MoeLoraWorkspace | None = None,
    ) -> SglMoeLoraPolicyBackend:
        """Read and validate layer-static inputs before provider construction."""
        weight_device = base_layer.w2_weight.device
        if weight_device.type != "cuda":
            raise NotImplementedError("SGL MoE-LoRA policy requires a CUDA layer")
        capability = torch.cuda.get_device_capability(weight_device)
        device_name = torch.cuda.get_device_name(weight_device)
        architecture = architecture_for_device(device_name, capability)
        if base_gemm_provider not in ("deepgemm", "cutedsl"):
            raise ValueError(
                "base_gemm_provider must be resolved to 'deepgemm' or 'cutedsl'"
            )
        config = base_layer.moe_runner_config
        if config.activation == "silu" and config.is_gated:
            activation = ActivationFamily.SWIGLU
        elif config.activation == "relu2" and not config.is_gated:
            activation = ActivationFamily.RELU2
        else:
            raise NotImplementedError(
                "SGL MoE-LoRA policy supports gated SiLU or non-gated ReLU2"
            )
        return cls(
            base_layer,
            architecture=architecture,
            base_gemm_provider=base_gemm_provider,
            activation=activation,
            workspace=workspace,
        )

    @property
    def is_bound(self) -> bool:
        return self._factor_layout is not None

    @property
    def choices(self) -> tuple[PolicyChoice, ...]:
        """All choices available after factor binding."""
        return self._choices

    def bind_factors(
        self,
        *,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        down_lora_a: torch.Tensor,
        down_lora_b: torch.Tensor,
        factor_layout: MoeLoraFactorLayout,
    ) -> None:
        """Construct and validate the complete runner set before capture.

        Initial binding is transactional: an unsupported choice leaves this
        backend unbound rather than exposing a partially populated policy.
        Rebinding a compatible resident buffer contract revalidates the
        existing runners and never recreates providers.
        """
        factor_layout.validate()
        selected_choices = choices_for(
            self.architecture,
            factor_layout,
            self.activation,
            self.base_gemm_provider,
        )
        providers = {choice.provider for choice in selected_choices}
        if providers != {self.base_gemm_provider}:
            raise RuntimeError(
                "MoE-LoRA policy changed the server's fixed base-GEMM provider: "
                f"expected {self.base_gemm_provider!r}, got {sorted(providers)!r}"
            )
        factor_kwargs = {
            "gate_up_lora_a": gate_up_lora_a,
            "gate_up_lora_b": gate_up_lora_b,
            "down_lora_a": down_lora_a,
            "down_lora_b": down_lora_b,
            "factor_layout": factor_layout,
        }
        if self.is_bound:
            if factor_layout != self._factor_layout:
                raise ValueError(
                    "resident MoE-LoRA factor layout changed after policy binding"
                )
            if tuple(choice.key for choice in selected_choices) != tuple(
                choice.key for choice in self._choices
            ):
                raise RuntimeError("the bound MoE-LoRA policy choice set changed")
            for runner in self._runners.values():
                runner.validate_factors(**factor_kwargs)
            return

        runners: dict[str, SglMoeLoraRunner] = {}
        for choice in selected_choices:
            if choice.key in runners:
                raise RuntimeError(f"duplicate MoE-LoRA policy key {choice.key!r}")
            runner = SglMoeLoraRunner.from_layer(
                self._base_layer,
                provider_name=choice.provider,
                execution_plan=choice.plan,
                launch_config=choice.launch_config,
                workspace=self.workspace,
            )
            runner.validate_factors(**factor_kwargs)
            runners[choice.key] = runner

        if not runners:
            raise RuntimeError("the MoE-LoRA policy produced no executable choices")
        self._runners = runners
        self._choices = selected_choices
        self._factor_layout = factor_layout

    def select(
        self,
        batch: SglMoeLoraBatch,
        *,
        num_tokens: int,
    ) -> PolicyChoice:
        """Select an already-created runner for every batch.

        Policy uses the resident physical rank in eager and graph mode.  The
        current kernels contract over the padded resident factor tensors, so
        the logical adapter rank does not reduce their GEMM K dimension.
        Selecting a logical-rank-tuned launch here would therefore describe a
        different workload than the one actually executed.

        Base-only eager batches intentionally keep the same runner topology as
        active and graph batches.  Besides keeping one ownership model, this
        avoids calling a resident base MoE runner that may have been created
        with in-place output before the LoRA wrapper attached; such a call can
        mutate the shared-expert input tensor needed after routed experts.
        """
        self._check_bound_batch(batch)
        resolved_mode = PolicyMode.PREFILL if batch.is_prefill else PolicyMode.DECODE
        active_rank = batch.physical_rank
        choice = select_policy(
            PolicyInput(
                architecture=self.architecture,
                base_gemm_provider=self.base_gemm_provider,
                factor_layout=batch.factor_layout,
                activation=self.activation,
                mode=resolved_mode,
                num_tokens=int(num_tokens),
                active_rank=int(active_rank),
            )
        )
        if choice.key not in self._runners:
            raise RuntimeError(
                f"policy selected unbound MoE-LoRA runner {choice.key!r}"
            )
        return choice

    def run_selected(
        self,
        choice: PolicyChoice,
        dispatch_output: StandardDispatchOutput,
        batch: SglMoeLoraBatch,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> StandardCombineInput:
        """Execute a choice returned by :meth:`select` after dispatch."""
        self._check_bound_batch(batch)
        runner = self._runners.get(choice.key)
        if runner is None:
            raise ValueError(f"choice {choice.key!r} is not bound to this layer")
        bound_choice = next(
            (candidate for candidate in self._choices if candidate.key == choice.key),
            None,
        )
        if bound_choice != choice:
            raise ValueError(
                f"choice {choice.key!r} does not match this layer's bound policy"
            )
        return runner.run(
            dispatch_output,
            batch,
            output_dtype=output_dtype,
        )

    def _check_bound_batch(self, batch: SglMoeLoraBatch) -> None:
        if not self.is_bound:
            raise RuntimeError(
                "SGL MoE-LoRA factors must be bound before policy selection"
            )
        if batch.factor_layout != self._factor_layout:
            raise ValueError(
                "batch MoE-LoRA factor layout does not match the bound policy"
            )
