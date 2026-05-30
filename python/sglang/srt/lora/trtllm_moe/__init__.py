"""Two-stream LoRA overlap (O7 + O8 + O9) — installed as a monkey-patch.

Activates when env ``SGLANG_OPT_LORA_TWO_STREAM=1``. Triggered exactly once via
:func:`install_two_stream_overrides` (called at the end of ``sglang/srt/lora/layers.py``).

When enabled, these LoRA call sites are redirected to side-stream-overlapped versions
that compute the LoRA-A shrink on a side CUDA stream concurrent with the base GEMM:

  * ``QKVParallelLinearWithLoRA.forward``         → :func:`.attention.qkv_proj_lora_forward`
  * ``RowParallelLinearWithLoRA.forward``         → :func:`.attention.row_parallel_lora_forward`
  * ``MergedColumnParallelLinearWithLoRA.forward``→ :func:`.merged_column.merged_column_lora_forward`

Per-batch gating happens inside the patched callables — they fall back to the saved-original
implementation for non-decode batches (token count above ``SGLANG_OPT_LORA_TWO_STREAM_MAX_TOKENS``,
default 256), so prefill stays on the serial path even with the patch installed.

NOTE: the package keeps the ``trtllm_moe`` name for parity with the lora-opti source these
overrides were ported from, but O7/O8/O9 are backend-agnostic (they patch the LoRA *linear
layers*, independent of the MoE runner backend). The trtllm-specific MoE overlap (O1) is
intentionally NOT included here — the cutlass MoE path has its own runner.
"""
from typing import Callable, Optional

import torch

from sglang.srt.environ import envs


def is_two_stream_active(x: torch.Tensor) -> bool:
    """Per-batch gate. True iff env is on AND the batch is decode-shaped."""
    if not envs.SGLANG_OPT_LORA_TWO_STREAM.get():
        return False
    return x.shape[0] <= envs.SGLANG_OPT_LORA_TWO_STREAM_MAX_TOKENS.get()


_LORA_SIDE_STREAM: Optional[torch.cuda.Stream] = None


def get_lora_side_stream() -> torch.cuda.Stream:
    """Lazily allocate a single shared LoRA side stream.

    Within one decode layer the sites (qkv → attn → o_proj) run sequentially, so one
    stream suffices and avoids extra graph-capture nodes from per-site streams.
    """
    global _LORA_SIDE_STREAM
    if _LORA_SIDE_STREAM is None:
        _LORA_SIDE_STREAM = torch.cuda.Stream()
    return _LORA_SIDE_STREAM


def init_lora_two_stream_resources(device: Optional[torch.device] = None) -> None:
    """Eagerly create the side stream before cuda-graph capture begins.

    ``torch.cuda.Stream()`` is a driver call that must not run inside a cuda-graph capture
    region. Since :func:`get_lora_side_stream` is otherwise lazy, the first eligible decode
    forward would create it — which can fall inside capture if warmup didn't happen to
    exercise a two-stream batch. Pin creation to init on the correct device. No-op unless
    ``SGLANG_OPT_LORA_TWO_STREAM=1``.
    """
    if not envs.SGLANG_OPT_LORA_TWO_STREAM.get():
        return
    if device is not None:
        with torch.cuda.device(device):
            get_lora_side_stream()
    else:
        get_lora_side_stream()


# Originals captured at install time so the patched callables can defer to them for
# non-decode batches.
_ORIGINAL_QKV_FORWARD: Optional[Callable] = None
_ORIGINAL_ROW_FORWARD: Optional[Callable] = None
_ORIGINAL_MERGED_FORWARD: Optional[Callable] = None
_INSTALLED: bool = False


def get_original_qkv_forward() -> Callable:
    return _ORIGINAL_QKV_FORWARD


def get_original_row_forward() -> Callable:
    return _ORIGINAL_ROW_FORWARD


def get_original_merged_column_forward() -> Callable:
    return _ORIGINAL_MERGED_FORWARD


def install_two_stream_overrides() -> None:
    """Install the side-stream overlapped LoRA forwards if ``SGLANG_OPT_LORA_TWO_STREAM=1``.

    Idempotent (subsequent calls are a no-op). Patches:
      1. ``QKVParallelLinearWithLoRA.forward``          (O7 — qkv LoRA-A shrink overlap)
      2. ``RowParallelLinearWithLoRA.forward``          (O8 — o_proj LoRA-A shrink overlap)
      3. ``MergedColumnParallelLinearWithLoRA.forward`` (O9 — dense merged-column shrink overlap)
    """
    global _INSTALLED, _ORIGINAL_QKV_FORWARD, _ORIGINAL_ROW_FORWARD, _ORIGINAL_MERGED_FORWARD

    if _INSTALLED:
        return
    if not envs.SGLANG_OPT_LORA_TWO_STREAM.get():
        return

    from sglang.srt.lora.layers import (
        MergedColumnParallelLinearWithLoRA,
        QKVParallelLinearWithLoRA,
        RowParallelLinearWithLoRA,
    )
    from sglang.srt.lora.trtllm_moe.attention import (
        qkv_proj_lora_forward,
        row_parallel_lora_forward,
    )
    from sglang.srt.lora.trtllm_moe.merged_column import merged_column_lora_forward

    _ORIGINAL_QKV_FORWARD = QKVParallelLinearWithLoRA.forward
    _ORIGINAL_ROW_FORWARD = RowParallelLinearWithLoRA.forward
    _ORIGINAL_MERGED_FORWARD = MergedColumnParallelLinearWithLoRA.forward
    QKVParallelLinearWithLoRA.forward = qkv_proj_lora_forward
    RowParallelLinearWithLoRA.forward = row_parallel_lora_forward
    MergedColumnParallelLinearWithLoRA.forward = merged_column_lora_forward

    _INSTALLED = True


__all__ = [
    "is_two_stream_active",
    "get_lora_side_stream",
    "init_lora_two_stream_resources",
    "get_original_qkv_forward",
    "get_original_row_forward",
    "get_original_merged_column_forward",
    "install_two_stream_overrides",
]
