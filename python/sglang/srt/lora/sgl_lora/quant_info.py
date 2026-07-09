"""Weight bundle for the Phase-1a SGL LoRA BF16 provider.

Standard sglang weight layouts only — no trtllm shuffle / reorder / BlockMajorK.
Field contract: ``w13_weight`` and ``w2_weight`` must stay 3-D ``[E, N, K]`` under
these exact names (``BaseLoRABackend.init_cuda_graph_moe_buffers`` unpacks
``E, N, _ = qinfo.w13_weight.shape`` and ``hidden = qinfo.w2_weight.shape[1]``).

Deliberately NOT a subclass of the ``MoeQuantInfo`` dataclass ABC — no isinstance
sites exist, and new containers are msgspec.Struct per repo convention.
"""

from __future__ import annotations

import msgspec
import torch


class SglLoraBf16QuantInfo(msgspec.Struct, kw_only=True):
    """Unquantized (bf16) MoE weights, standard layout.

    w13_weight: [E_local, 2 * intermediate_size, hidden]  (gate-first stacking)
    w2_weight:  [E_local, hidden, intermediate_size]
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    num_local_experts: int
    intermediate_size: int
    hidden_size: int


SglLoraQuantInfo = SglLoraBf16QuantInfo
