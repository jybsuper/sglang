"""Runtime resources owned by the ``sgl_lora`` execution engine."""

from typing import Optional

import torch


def get_lora_side_stream() -> torch.cuda.Stream:
    from sglang.srt.runtime_context import get_stream

    return get_stream("lora_side")


def init_lora_two_stream_resources(device: Optional[torch.device] = None) -> None:
    """Create the side stream before CUDA graph capture starts."""
    if device is None:
        get_lora_side_stream()
        return
    with torch.cuda.device(device):
        get_lora_side_stream()


__all__ = ["get_lora_side_stream", "init_lora_two_stream_resources"]
