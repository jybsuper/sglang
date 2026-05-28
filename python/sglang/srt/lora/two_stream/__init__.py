"""Two-stream LoRA overlap helpers (O1, O7, O8).

Side-stream overlap optimizations for SGLang LoRA serving on the
``sgl_flashinfer_trtllm`` MoE backend. The optimizations run the LoRA
shrink/expand on a dedicated CUDA stream concurrently with the base
attention or MoE GEMMs on the main stream. Decode-only — prefill kernels
already saturate the GPU, so the extra stream sync/merge would only add
overhead.

This package is the centralized landing for all two-stream code so the
existing files (``lora/layers.py`` and
``layers/moe/moe_runner/flashinfer_trtllm.py``) need only minimal injection
points (env gate + one delegate per call site).

Master switch: env ``SGLANG_LORA_TWO_STREAM=1``. Token-count threshold:
``SGLANG_TWO_STREAM_MAX_TOKENS`` (default 256, matches typical decode bs).

Currently active overlaps:

  - **O1**: gate_up MoE LoRA shrink+expand on side stream, concurrent with
    main-stream per-token-group FP8 quant (``flashinfer_trtllm.py``).
  - **O7**: QKV-attention LoRA shrink on side stream, concurrent with the
    base qkv_proj GEMM on main; rejoin before LoRA expand atomic-add.
  - **O8**: o_proj (row-parallel) LoRA shrink on side stream, concurrent
    with the base o_proj GEMM on main; rejoin before all-reduce/expand.

A single global side stream serves all three sites — within one layer they
run sequentially (qkv → attn → o_proj → moe_gate_up), so reuse is safe.
"""
import os
from typing import Optional

import torch

_ENV_KEY = "SGLANG_LORA_TWO_STREAM"
_MAX_TOKENS_KEY = "SGLANG_TWO_STREAM_MAX_TOKENS"
_MAX_TOKENS_DEFAULT = 256


def is_two_stream_active(x: torch.Tensor) -> bool:
    """Whether side-stream LoRA overlap should fire for this batch.

    Returns ``False`` unless ``SGLANG_LORA_TWO_STREAM=1`` AND the leading
    dim of ``x`` (= token count for typical decode input shapes) is at or
    below ``SGLANG_TWO_STREAM_MAX_TOKENS``. Prefill batches with thousands
    of chunked tokens stay on the serial path.
    """
    if os.environ.get(_ENV_KEY) != "1":
        return False
    try:
        max_tok = int(os.environ.get(_MAX_TOKENS_KEY, str(_MAX_TOKENS_DEFAULT)))
    except ValueError:
        max_tok = _MAX_TOKENS_DEFAULT
    return x.shape[0] <= max_tok


_LORA_SIDE_STREAM: Optional[torch.cuda.Stream] = None


def get_lora_side_stream() -> torch.cuda.Stream:
    """Lazily allocate and return the shared LoRA side stream.

    Reused across O1/O7/O8 sites within a layer — their overlap windows
    are sequential (qkv → attn → o_proj → moe_gate_up), so one stream is
    enough and avoids the per-site allocation + capture-graph node overhead
    of separate streams.
    """
    global _LORA_SIDE_STREAM
    if _LORA_SIDE_STREAM is None:
        _LORA_SIDE_STREAM = torch.cuda.Stream()
    return _LORA_SIDE_STREAM


__all__ = ["is_two_stream_active", "get_lora_side_stream"]
