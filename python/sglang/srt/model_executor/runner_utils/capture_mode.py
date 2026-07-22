# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Process-global capture-mode flags shared by the decode runner and the
speculative-draft runners. Read by model code that needs to take a
capture-time branch (e.g. lora dual-graph capture decides per-batch
which variant to use).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)

# Detect whether the current forward pass is in capture mode.
is_capture_mode = False

# When capturing dual MoE backends, tracks which variant is being captured.
# None = not dual, "lora" = capturing lora variant, "nolora" = capturing nolora variant.
_capture_lora_variant: Optional[str] = None


def get_is_capture_mode() -> bool:
    return is_capture_mode or is_in_breakable_cuda_graph()


def compile_in_capture_mode(func):
    """Decorator: wrap func with torch.compile only when defined
    inside model capture mode; passthrough otherwise.

    Used by model code (e.g. DeepSeek-V4) to opt nested helpers into
    torch.compile during cuda-graph capture without paying the
    compilation cost in the eager forward path.
    """
    if is_capture_mode:
        return torch.compile(func)
    return func


def get_capture_lora_variant() -> Optional[str]:
    """Return the lora variant being captured, or None if not in dual capture."""
    return _capture_lora_variant


def _set_capture_lora_variant(variant: Optional[str]) -> None:
    global _capture_lora_variant
    _capture_lora_variant = variant


@contextmanager
def capture_lora_variant(variant: Optional[str]):
    """Select one LoRA graph topology and restore the previous selection."""
    previous = get_capture_lora_variant()
    _set_capture_lora_variant(variant)
    try:
        yield
    finally:
        _set_capture_lora_variant(previous)


def should_record_lora_graph_variants(server_args, spec_algorithm) -> bool:
    """Whether decode should capture distinct LoRA-on and LoRA-off graphs.

    The first implementation is intentionally scoped to the opt-in ``sgl_lora``
    execution engine and ordinary decode.  Speculative draft/verify runners own
    different assignment domains and continue to force this feature off.
    """
    return bool(
        getattr(server_args, "enable_lora", False)
        and getattr(server_args, "lora_execution_engine", "legacy") == "sgl_lora"
        and spec_algorithm.is_none()
    )


@contextmanager
def model_capture_mode():
    global is_capture_mode
    from sglang.srt.runtime_context import get_flags

    # Disable dispose_tensor() during capture: freeing mid-capture records data_ptr()==0 into the graph.
    is_capture_mode = True
    get_flags().capture.disable_dispose_tensor = True
    try:
        yield
    finally:
        is_capture_mode = False
        get_flags().capture.disable_dispose_tensor = False
