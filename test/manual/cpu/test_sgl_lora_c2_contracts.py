import importlib.util
from pathlib import Path

import pytest
import torch


def _load_experimental_c2_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/lora/sgl_lora/experimental_c2.py"
    )
    spec = importlib.util.spec_from_file_location("_test_experimental_c2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_C2 = _load_experimental_c2_module()


def test_c2_token_lora_mapping_must_match_token_domain():
    _C2._validate_token_lora_mapping(torch.tensor([0, 1, -1]), 3)

    with pytest.raises(RuntimeError, match="mapping has 2 rows"):
        _C2._validate_token_lora_mapping(torch.tensor([0, 1]), 3)


def test_c2_down_lora_routed_scale_is_applied_exactly_once():
    weights = torch.tensor([[0.25, 0.75]])

    assert _C2._scaled_down_lora_topk_weights(weights, None) is weights
    assert _C2._scaled_down_lora_topk_weights(weights, 1.0) is weights
    assert torch.allclose(
        _C2._scaled_down_lora_topk_weights(weights, 0.4),
        torch.tensor([[0.1, 0.3]]),
    )
