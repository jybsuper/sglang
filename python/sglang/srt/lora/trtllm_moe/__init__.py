"""Two-stream LoRA overlap (O1 + O7 + O8) — installed as a monkey-patch.

Activates when env ``SGLANG_LORA_TWO_STREAM=1``. Triggered exactly once via
:func:`install_two_stream_overrides` (called at end of ``sglang/srt/lora/layers.py``).

When enabled, three call sites are redirected to side-stream-overlapped versions
defined entirely in this package:

  * ``QKVParallelLinearWithLoRA.forward``  → :mod:`.attention.qkv_proj_lora_forward`
  * ``RowParallelLinearWithLoRA.forward``  → :mod:`.attention.row_parallel_lora_forward`
  * ``fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora`` →
    :mod:`.moe_overlap.fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora_two_stream`

When disabled (env unset), ``install_two_stream_overrides`` is a no-op and all
the original functions / methods in ``sglang/srt/lora/layers.py`` and
``sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`` run unchanged.

Per-batch gating still happens inside the patched callables — they fall back
to the saved-original implementation for non-decode batches (token count above
``SGLANG_TWO_STREAM_MAX_TOKENS`` default 256), so prefill stays on the serial
path even with the patch installed.
"""
import os
from typing import Callable, Optional

import torch

_ENV_KEY = "SGLANG_LORA_TWO_STREAM"
_MAX_TOKENS_KEY = "SGLANG_TWO_STREAM_MAX_TOKENS"
_MAX_TOKENS_DEFAULT = 256


def is_two_stream_active(x: torch.Tensor) -> bool:
    """Per-batch gate. True iff env is on AND batch is decode-shaped."""
    if os.environ.get(_ENV_KEY) != "1":
        return False
    try:
        max_tok = int(os.environ.get(_MAX_TOKENS_KEY, str(_MAX_TOKENS_DEFAULT)))
    except ValueError:
        max_tok = _MAX_TOKENS_DEFAULT
    return x.shape[0] <= max_tok


_LORA_SIDE_STREAM: Optional[torch.cuda.Stream] = None


def get_lora_side_stream() -> torch.cuda.Stream:
    """Lazily allocate a single shared LoRA side stream.

    Within one decode layer the three sites (qkv → attn → o_proj → moe_gate_up)
    run sequentially, so one stream suffices and avoids extra graph-capture
    nodes from per-site streams.
    """
    global _LORA_SIDE_STREAM
    if _LORA_SIDE_STREAM is None:
        _LORA_SIDE_STREAM = torch.cuda.Stream()
    return _LORA_SIDE_STREAM


# References to the original implementations, captured at install time so the
# patched callables can defer to them for non-decode batches.
_ORIGINAL_QKV_FORWARD: Optional[Callable] = None
_ORIGINAL_ROW_FORWARD: Optional[Callable] = None
_ORIGINAL_MOE_LORA_FUNC: Optional[Callable] = None
_INSTALLED: bool = False


def get_original_qkv_forward() -> Callable:
    return _ORIGINAL_QKV_FORWARD


def get_original_row_forward() -> Callable:
    return _ORIGINAL_ROW_FORWARD


def get_original_moe_lora_func() -> Callable:
    return _ORIGINAL_MOE_LORA_FUNC


def install_two_stream_overrides() -> None:
    """Install the side-stream overlapped overrides if ``SGLANG_LORA_TWO_STREAM=1``.

    Idempotent: subsequent calls are a no-op. Patches:

      1. ``QKVParallelLinearWithLoRA.forward`` (O7 — qkv LoRA shrink overlap)
      2. ``RowParallelLinearWithLoRA.forward`` (O8 — o_proj LoRA shrink overlap)
      3. ``flashinfer_trtllm.fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora``
         (O1 — MoE gate_up LoRA overlap)

    The saved originals are exposed via :func:`get_original_qkv_forward`,
    :func:`get_original_row_forward`, :func:`get_original_moe_lora_func` so the
    new versions can fall back when their per-batch gate says single-stream.
    """
    global _INSTALLED, _ORIGINAL_QKV_FORWARD, _ORIGINAL_ROW_FORWARD, _ORIGINAL_MOE_LORA_FUNC

    if _INSTALLED:
        return
    if os.environ.get(_ENV_KEY) != "1":
        return

    from sglang.srt.lora.layers import (
        QKVParallelLinearWithLoRA,
        RowParallelLinearWithLoRA,
    )
    from sglang.srt.lora.trtllm_moe.attention import (
        qkv_proj_lora_forward,
        row_parallel_lora_forward,
    )

    _ORIGINAL_QKV_FORWARD = QKVParallelLinearWithLoRA.forward
    _ORIGINAL_ROW_FORWARD = RowParallelLinearWithLoRA.forward
    QKVParallelLinearWithLoRA.forward = qkv_proj_lora_forward
    RowParallelLinearWithLoRA.forward = row_parallel_lora_forward

    import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm as ft
    from sglang.srt.lora.trtllm_moe.moe_overlap import (
        fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora_two_stream,
    )

    _ORIGINAL_MOE_LORA_FUNC = ft.fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora
    ft.fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora = (
        fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora_two_stream
    )

    _INSTALLED = True


__all__ = [
    "is_two_stream_active",
    "get_lora_side_stream",
    "get_original_qkv_forward",
    "get_original_row_forward",
    "get_original_moe_lora_func",
    "install_two_stream_overrides",
]
