"""SGLang-native LoRA execution engine.

Phase 1a provides ``moe_lora_runner.run_sgl_lora_moe`` over a pluggable
``base_gemm.MoeLoraBaseGemm``. The current provider envelope is deliberately
narrow: unquantized BF16 MoE on Hopper or Blackwell through DeepGEMM.

Standard SGLang weight layouts are used throughout. The engine requires
``--enable-lora`` and enables virtual-expert semantics internally. Dense and
special LoRA layers remain on their existing ``--lora-backend`` during Phase 1a.

LoRA-batch contract: ``LoRAInfo`` built from the backend-agnostic
``MoELoRABatchInfo`` — works identically under the triton and csgmv dense-LoRA
backends. Expert-id convention in Phase 1a: local IDs from standard dispatch.
"""
