"""Side-stream LoRA overlap helpers for the trtllm MoE path (O1).

Used by ``layers/moe/moe_runner/flashinfer_trtllm.py`` to fork the
gate_up LoRA shrink+expand onto a side stream while the main stream
runs the base FP8 quant, then rejoin before the trtllm op (whose
activation step consumes gate_up_lora_delta).
"""
from typing import Callable, Optional

import torch

from sglang.srt.lora.two_stream import get_lora_side_stream, is_two_stream_active


def maybe_fork_lora_overlap(
    run_fn: Callable[[], None], gate_tensor: torch.Tensor
) -> Optional[torch.cuda.Stream]:
    """Fork the LoRA side stream and run ``run_fn`` on it if two-stream is
    active for this batch; otherwise return ``None`` and run nothing.

    The caller is responsible for joining the returned stream before
    reading any tensor that ``run_fn`` writes — use :func:`maybe_join`.
    ``gate_tensor`` is the tensor whose leading dim drives the
    decode-only token-count check (typically ``hidden_states``).
    """
    if not is_two_stream_active(gate_tensor):
        return None
    side_stream = get_lora_side_stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        run_fn()
    return side_stream


def maybe_join(side_stream: Optional[torch.cuda.Stream]) -> None:
    """Rejoin a side stream returned by :func:`maybe_fork_lora_overlap`.

    No-op if ``side_stream`` is ``None`` (= two-stream wasn't active).
    """
    if side_stream is not None:
        torch.cuda.current_stream().wait_stream(side_stream)
