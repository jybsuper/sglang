import importlib.util
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "kernels"
    / "lora_moe"
    / "bench_lifecycle_e2e.py"
)
_SPEC = importlib.util.spec_from_file_location("bench_lifecycle_e2e", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_extract_output_ids_normalizes_single_response():
    assert _MODULE._extract_output_ids({"output_ids": [1, 2, 3]}) == [[1, 2, 3]]


def test_extract_output_ids_normalizes_batch_response():
    response = {
        "response": [
            {"output_ids": [1, 2]},
            {"output_ids": [3, 4]},
        ]
    }
    assert _MODULE._extract_output_ids(response) == [[1, 2], [3, 4]]


def test_transition_is_serializable():
    transition = _MODULE.Transition("base_before", None)
    assert transition.name == "base_before"
    assert transition.lora_name is None
