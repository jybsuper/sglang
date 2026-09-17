"""CPU contracts for graph metadata and scratch using exact production method bodies.
Narrow dependencies avoid importing the scheduler/CUDA stack.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_BACKEND_PATH = REPO_ROOT / "python/sglang/srt/lora/backend/base_backend.py"
V2_BACKEND_PATH = REPO_ROOT / "python/sglang/srt/lora/backend/triton_v2_backend.py"
LAYERS_PATH = REPO_ROOT / "python/sglang/srt/lora/layers.py"
MOE_RUNNER_PATH = REPO_ROOT / "python/sglang/srt/lora/moe/runner.py"
SHRINK_PATH = REPO_ROOT / "python/sglang/kernels/ops/lora/common/lora_a.py"
WORKSPACE_PATH = REPO_ROOT / "python/sglang/srt/lora/workspace.py"


def _load_class_with_members(
    path: Path,
    source_class: str,
    *member_names: str,
    include_fields: bool = False,
    namespace: dict[str, object] | None = None,
):
    tree = ast.parse(path.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == source_class
    )
    members = [
        node
        for node in class_node.body
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in member_names
        )
        or (include_fields and isinstance(node, ast.AnnAssign))
    ]
    assert {
        node.name
        for node in members
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    } == set(member_names)
    test_class = ast.ClassDef(
        name=f"_{source_class}UnderTest",
        bases=[],
        keywords=[],
        body=members,
        decorator_list=class_node.decorator_list if include_fields else [],
    )
    scope = {"__name__": __name__, **(namespace or {})}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope[test_class.name]


def _load_function(
    path: Path,
    function_name: str,
    *,
    namespace: dict[str, object] | None = None,
):
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    scope = dict(namespace or {})
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope[function_name]


@pytest.mark.parametrize("has_w13_weight", (False, True))
def test_legacy_moe_buffers_use_prepared_quant_info_device(has_w13_weight):
    backend_type = _load_class_with_members(
        BASE_BACKEND_PATH,
        "BaseLoRABackend",
        "init_cuda_graph_moe_buffers",
        namespace={"torch": torch},
    )
    quant_info = SimpleNamespace(w13_qweight=torch.empty(1, device="meta"))
    if has_w13_weight:
        quant_info.w13_weight = torch.empty(1)
    expected_device = "cpu" if has_w13_weight else "meta"
    layer = SimpleNamespace(
        _quant_info=quant_info,
        base_layer=SimpleNamespace(
            top_k=2,
            num_experts=4,
            w13_weight=torch.empty(1, device="meta" if has_w13_weight else "cpu"),
        ),
    )
    backend = backend_type()
    backend.init_cuda_graph_moe_buffers(8, 2, torch.bfloat16, layer)
    assert backend.moe_cg_buffers
    assert all(
        buffer.device.type == expected_device
        for buffer in backend.moe_cg_buffers.values()
    )


def test_v2_bypasses_legacy_moe_preallocation():
    reserve = _load_function(
        REPO_ROOT / "python/sglang/srt/lora/lora_manager.py",
        "init_lora_cuda_graph_moe_buffers",
        namespace={
            "torch": torch,
            "LoRAManager": object,
            "logger": SimpleNamespace(debug=lambda *_: None),
        },
    )
    allocations = []
    lora_manager = SimpleNamespace(
        lora_backend=SimpleNamespace(name="triton_v2"),
        init_cuda_graph_moe_buffers=lambda *args: allocations.append(args),
    )

    def modules():
        raise AssertionError("V2 must not scan the model for legacy MoE layers")

    model = SimpleNamespace(modules=modules)
    assert reserve(model=model, lora_manager=lora_manager, dtype=torch.bfloat16) is None
    assert allocations == []


@pytest.fixture
def grouped_a_runner():
    # Keep allocation/reuse and grid arithmetic real; only record the GPU call.
    triton = SimpleNamespace(cdiv=lambda a, b: -(-a // b))
    shrink_tiles = _load_function(
        SHRINK_PATH, "shrink_column_tiles", namespace={"triton": triton}
    )
    workspace_type = _load_class_with_members(
        WORKSPACE_PATH,
        "LoraWorkspace",
        "__init__",
        "tensor",
        "_capturing",
        namespace={"torch": torch},
    )
    workspace = workspace_type()
    allocations, launches = [], []

    def tensor(name, shape, **kwargs):
        value = workspace.tensor(name, shape, **kwargs)
        allocations.append((name, tuple(shape), kwargs, value))
        return value

    site = SimpleNamespace(GATE_UP=object(), DOWN=object())
    runner_type = _load_class_with_members(
        MOE_RUNNER_PATH,
        "MoeLoraRunner",
        "_run_a",
        namespace={
            "torch": torch,
            "triton": triton,
            "shrink_column_tiles": shrink_tiles,
            "AFamily": SimpleNamespace(
                TOKEN_GROUPED=object(), PER_ROW=object(), TOKEN_DENSE=object()
            ),
            "BridgeLayout": SimpleNamespace(TOKEN_MAJOR=object()),
            "Site": site,
            "grouped_lora_a": lambda *args, **kwargs: launches.append((args, kwargs)),
        },
    )
    runner = runner_type()
    runner.workspace = SimpleNamespace(tensor=tensor)
    runner.lora_delta_dtype = torch.bfloat16
    return runner, workspace, site, allocations, launches


@pytest.mark.parametrize(
    (
        "split_k",
        "down",
        "tokens",
        "rows",
        "capacity",
        "block_m",
        "columns",
        "block_n",
        "tiles",
    ),
    [
        (1, False, 7, 21, 84, 16, 35, 32, 12),
        (4, False, 7, 21, 84, 16, 35, 32, 12),
        (8, True, 3, 9, 40, 16, 16, 64, 3),
    ],
)
def test_grouped_a_splitk_scratch_and_forwarded_operands(
    grouped_a_runner,
    split_k,
    down,
    tokens,
    rows,
    capacity,
    block_m,
    columns,
    block_n,
    tiles,
):
    runner, _, site, allocations, launches = grouped_a_runner
    spec = SimpleNamespace(
        site=site.DOWN if down else site.GATE_UP,
        family=SimpleNamespace(value="grouped"),
        is_shared_outer=False,
        output_layout=object(),
    )
    route = SimpleNamespace(
        num_tokens=tokens,
        num_rows=rows,
        sorted_pair_ids=torch.empty(capacity, dtype=torch.int32),
        block_size=block_m,
    )
    config = {"SPLIT_K": split_k, "BLOCK_SIZE_N": block_n}
    requested_sites = []

    def for_a(selected_site):
        requested_sites.append(selected_site)
        return config

    input = torch.empty((rows if down else tokens, 128), dtype=torch.bfloat16)
    weight = torch.empty((4, columns, 128), dtype=torch.bfloat16)
    pair_to_row = torch.arange(rows, dtype=torch.int32) if down else None
    output = runner._run_a(
        SimpleNamespace(for_a=for_a),
        spec,
        input,
        weight,
        SimpleNamespace(aligned=lambda shared: route),
        "down_a" if down else "gate_up_a",
        pair_to_row=pair_to_row,
    )
    name = "down_a" if down else "gate_up_a"
    assert requested_sites == [spec.site]
    assert output.shape == (rows, columns)  # Neither token count nor padded capacity.
    assert output.dtype == torch.bfloat16
    assert allocations[0][:2] == (f"{name}:output", (rows, columns))
    assert all(kwargs["device"] == input.device for _, _, kwargs, _ in allocations)
    assert len(launches) == 1
    args, kwargs = launches[0]
    assert len(args) == 4
    assert all(
        actual is expected
        for actual, expected in zip(args, (input, weight, output, route))
    )
    assert kwargs["config"] is config
    assert kwargs["pair_input"] is down
    assert kwargs["pair_to_row"] is pair_to_row
    if split_k == 1:
        assert len(allocations) == 1
        assert kwargs["partial"] is kwargs["counters"] is None
    else:
        assert len(allocations) == 3
        partial, counters = kwargs["partial"], kwargs["counters"]
        assert partial is allocations[1][3]
        assert counters is allocations[2][3]
        assert allocations[1][:2] == (f"{name}:partial", (split_k, rows, columns))
        assert allocations[2][:2] == (f"{name}:splitk_counters", (tiles,))
        assert partial.shape == (split_k, rows, columns)
        assert counters.shape == (tiles,)
        assert partial.dtype == torch.float32
        assert counters.dtype == torch.int32
        assert allocations[2][2]["zero_on_first_allocation"] is True
        assert counters.count_nonzero().item() == 0


@pytest.mark.parametrize("graph_mode", [False, True])
def test_grouped_a_splitk_reuses_storage_and_separates_sites(
    grouped_a_runner, graph_mode
):
    runner, workspace, site, allocations, launches = grouped_a_runner
    workspace._graph_mode = graph_mode
    workspace._is_prefill_graph = graph_mode
    route = SimpleNamespace(
        num_tokens=7,
        num_rows=21,
        sorted_pair_ids=torch.empty(84, dtype=torch.int32),
        block_size=16,
    )
    config = {"SPLIT_K": 4, "BLOCK_SIZE_N": 32}
    launch_config = SimpleNamespace(for_a=lambda selected_site: config)
    input = torch.empty((21, 128), dtype=torch.bfloat16)
    weight = torch.empty((4, 35, 128), dtype=torch.bfloat16)
    for name, selected_site in (
        ("gate_up_a", site.GATE_UP),
        ("gate_up_a", site.GATE_UP),
        ("down_a", site.DOWN),
    ):
        spec = SimpleNamespace(
            site=selected_site,
            family=SimpleNamespace(value="grouped"),
            is_shared_outer=False,
            output_layout=object(),
        )
        runner._run_a(
            launch_config,
            spec,
            input,
            weight,
            SimpleNamespace(aligned=lambda shared: route),
            name,
        )
        # Poison only after the first call: the runner must not re-zero reused
        # counters. The real GPU kernel, not this host test, resets completed tiles.
        if len(launches) == 1:
            launches[0][1]["counters"].fill_(7)
    first, repeat, other = (allocations[start : start + 3] for start in (0, 3, 6))
    for initial, reused, separate in zip(first, repeat, other):
        assert initial[0] == reused[0]
        assert initial[3].data_ptr() == reused[3].data_ptr()
        assert separate[0].startswith("down_a:")
        assert initial[3].data_ptr() != separate[3].data_ptr()
    assert launches[1][1]["counters"].tolist() == [7] * 12
    assert launches[2][1]["counters"].count_nonzero().item() == 0


@pytest.fixture
def mapping_functions():
    return _load_function(
        BASE_BACKEND_PATH,
        "_compute_token_lora_mapping",
        namespace={
            "torch": torch,
            "triton": SimpleNamespace(cdiv=lambda a, b: -(-a // b)),
        },
    )


@pytest.fixture
def v2_backend(mapping_functions):
    def unpinned_empty(*args, **kwargs):
        # CPU-only builds have no pinned allocator; tensor operations stay real.
        kwargs.pop("pin_memory", None)
        return torch.empty(*args, **kwargs)

    cpu_torch = SimpleNamespace(
        empty=unpinned_empty,
        zeros=torch.zeros,
        arange=torch.arange,
        as_tensor=torch.as_tensor,
        int32=torch.int32,
        float32=torch.float32,
    )
    batch_type = _load_class_with_members(
        V2_BACKEND_PATH,
        "_BatchInfo",
        include_fields=True,
        namespace={"torch": torch, "dataclass": dataclass},
    )
    get_counts = _load_function(
        REPO_ROOT / "python/sglang/srt/lora/utils.py", "get_batch_token_counts"
    )
    backend_type = _load_class_with_members(
        V2_BACKEND_PATH,
        "TritonV2LoRABackend",
        "_new_batch",
        "_fill_batch",
        "prepare_lora_batch",
        "init_decode_cuda_graph_batch_info",
        "init_prefill_cuda_graph_batch_info",
        "init_cuda_graph_moe_buffers",
        namespace={
            "torch": cpu_torch,
            "_BatchInfo": batch_type,
            "_compute_token_lora_mapping": mapping_functions,
            "get_batch_token_counts": get_counts,
            "Phase": SimpleNamespace(PREFILL="prefill", DECODE="decode"),
        },
    )
    backend = backend_type()
    backend.name = "triton_v2"
    backend.max_loras_per_batch = 3
    backend.device = torch.device("cpu")
    backend.get_batch_info = MethodType(
        _load_class_with_members(
            BASE_BACKEND_PATH, "BaseLoRABackend", "get_batch_info"
        ).get_batch_info,
        backend,
    )
    backend.lm_head_runner = None
    bindings = []
    backend.runner = SimpleNamespace(
        begin_batch=lambda **kwargs: bindings.append(kwargs)
    )
    backend.bindings = bindings
    return backend


@pytest.fixture
def moe_payload_builder(monkeypatch):
    module_name = "sglang.srt.lora.moe.runner"
    module = ModuleType(module_name)
    module.MoeLoraBatch = SimpleNamespace
    monkeypatch.setitem(sys.modules, module_name, module)
    layer_type = _load_class_with_members(
        LAYERS_PATH,
        "FusedMoEWithLoRA",
        "_run_moe_core_with_lora",
        namespace={"LoRABatchLayout": SimpleNamespace(TP_GLOBAL=object())},
    )
    layer = layer_type()
    layer.gate_up_lora_a_weights = torch.empty(1)
    layer.gate_up_lora_b_weights = torch.empty(1)
    layer.down_lora_a_weights = torch.empty(1)
    layer.down_lora_b_weights = torch.empty(1)

    def build(backend):
        layer.lora_backend = backend
        layer._moe_lora_runner = SimpleNamespace(run=lambda dispatch, batch: batch)
        payload = layer._run_moe_core_with_lora(object())
        assert payload.gate_up_lora_a is layer.gate_up_lora_a_weights
        assert payload.gate_up_lora_b is layer.gate_up_lora_b_weights
        assert payload.down_lora_a is layer.down_lora_a_weights
        assert payload.down_lora_b is layer.down_lora_b_weights
        return payload

    return build


@pytest.mark.parametrize("graph", [False, True])
def test_v2_packed_metadata_has_no_legacy_mask_or_nested_payload(v2_backend, graph):
    batch = v2_backend._new_batch(5, 12, graph=graph)
    assert batch.packed.dtype == torch.int32
    assert batch.packed.numel() == 2 * 3 + 2 * 5 + 1
    assert not hasattr(batch, "adapter_enabled")
    assert not hasattr(batch, "moe_lora_info")
    assert batch.num_tokens == 0
    assert batch.has_active_lora is False
    assert batch.use_cuda_graph is graph
    for tensor, offset in (
        (batch.lora_ranks, 0),
        (batch.scalings, 3),
        (batch.weight_indices, 6),
        (batch.seg_indptr, 11),
    ):
        assert tensor.data_ptr() == batch.packed.data_ptr() + 4 * offset
    if graph:
        assert batch.packed.count_nonzero().item() == 0
        assert batch.token_slots.tolist() == [-1] * 12

    v2_backend._fill_batch(batch, [2, 1], [0, 8, 16], [0.0, 0.25, 0.5], [3, 2])
    assert batch.packed[:3].tolist() == [0, 8, 16]
    assert batch.packed[3:6].view(torch.float32).tolist() == [0.0, 0.25, 0.5]
    assert batch.packed[6:8].tolist() == [2, 1]
    assert batch.seg_indptr[:3].tolist() == [0, 3, 5]
    with pytest.raises(ValueError, match="capacity"):
        v2_backend._fill_batch(batch, [0] * 6, [0] * 3, [0.0] * 3, [1] * 6)


@pytest.mark.parametrize(("max_tokens", "max_requests"), [(12, None), (12, 4), (3, 5)])
def test_v2_prefill_graph_request_capacity_matches_admission(
    v2_backend, max_tokens, max_requests
):
    if max_requests is None:
        v2_backend.init_prefill_cuda_graph_batch_info(max_tokens)
    else:
        v2_backend.init_prefill_cuda_graph_batch_info(max_tokens, max_requests)
    requests = max_tokens if max_requests is None else max_requests
    batch = v2_backend.prefill_cuda_graph_batch_info
    assert batch.weight_indices.numel() == requests
    assert batch.seg_indptr.numel() == requests + 1
    assert batch.token_slots.numel() == max_tokens
    assert v2_backend.prefill_cuda_graph_max_bs == requests
    assert v2_backend.prefill_cuda_graph_max_tokens == max_tokens
    assert batch.use_cuda_graph
    with pytest.raises(ValueError, match="capacity"):
        v2_backend._fill_batch(
            batch, [0] * (requests + 1), [0] * 3, [0.0] * 3, [0] * (requests + 1)
        )


@pytest.mark.parametrize(
    ("mode", "max_requests"),
    [
        ("eager", None),
        ("prefill_graph", None),
        ("prefill_graph", 4),
        ("prefill_graph", 16),
        ("decode_graph", None),
    ],
)
def test_v2_live_moe_views_preserve_dense_graph_capacity(
    v2_backend, moe_payload_builder, mode, max_requests
):
    decode = mode == "decode_graph"
    graph = mode != "eager"
    if decode:
        v2_backend.init_decode_cuda_graph_batch_info(6, 1)
        cases = [([1, 1, 1], [1, 2, 0]), ([1], [2]), ([], [])]
    else:
        if graph:
            v2_backend.init_prefill_cuda_graph_batch_info(12, max_requests)
        cases = [([2, 0, 3, 1], [1, 2, 0, 1]), ([1, 2], [2, 0]), ([0, 0], [1, 2])]
        if max_requests == 16:
            # Full capture's fixed request axis can exceed a small token bucket.
            cases[0] = (cases[0][0] + [0] * 12, cases[0][1] + [0] * 12)
    addresses = None
    for lengths, slots in cases:
        forward_batch = SimpleNamespace(
            batch_size=len(lengths),
            extend_num_tokens=sum(lengths),
            extend_seq_lens_cpu=lengths,
            forward_mode=SimpleNamespace(
                is_decode=lambda: decode,
                is_target_verify=lambda: False,
                is_idle=lambda: False,
                is_extend=lambda: not decode,
                is_extend_without_speculative=lambda: not decode,
            ),
        )
        ranks = [0, 8, 0]
        v2_backend.prepare_lora_batch(
            forward_batch,
            slots,
            ranks,
            [0.0, 0.25, 0.0],
            use_decode_cuda_graph=decode,
            use_prefill_cuda_graph=mode == "prefill_graph",
        )
        batch = v2_backend.batch_info
        # The manager selects the MoE phase after backend preparation.
        batch.is_prefill = not decode
        expected = [
            slot if ranks[slot] else -1
            for length, slot in zip(lengths, slots)
            for _ in range(length)
        ]
        assert batch.num_tokens == len(expected)
        capacity = (6 if decode else 12) if graph else len(expected)
        request_capacity = (
            (6 if decode else (12 if max_requests is None else max_requests))
            if graph
            else len(lengths)
        )
        assert batch.token_slots.tolist() == expected + [-1] * (
            capacity - len(expected)
        )
        assert batch.weight_indices[: len(slots)].tolist() == slots
        assert not hasattr(batch, "adapter_enabled")
        assert not hasattr(batch, "moe_lora_info")
        bound = v2_backend.bindings[-1]
        assert bound["token_slots"] is batch.token_slots
        assert bound["token_slots"].shape == (capacity,)
        assert bound["num_tokens"] == len(expected)
        assert bound["graph_mode"] is graph
        assert bound["is_prefill_graph"] is (mode == "prefill_graph")

        payload = moe_payload_builder(v2_backend)
        assert payload.token_lora_mapping.tolist() == expected
        assert payload.token_lora_mapping.shape == (batch.num_tokens,)
        assert payload.token_lora_mapping.untyped_storage().data_ptr() == (
            batch.token_slots.untyped_storage().data_ptr()
        )
        assert payload.use_cuda_graph is graph
        assert payload.is_prefill is (not decode)
        if graph:
            current = batch.packed.data_ptr(), batch.token_slots.data_ptr()
            if addresses is not None:
                assert current == addresses
            addresses = current


def test_v2_moe_graph_initialization_allocates_no_legacy_metadata(
    v2_backend, monkeypatch
):
    def no_allocation(*args, **kwargs):
        raise AssertionError("V2 must not allocate unused legacy MoE graph metadata")

    scope = v2_backend.init_cuda_graph_moe_buffers.__func__.__globals__
    for name in ("empty", "zeros", "full"):
        monkeypatch.setattr(scope["torch"], name, no_allocation, raising=False)
    before = vars(v2_backend).copy()
    for prefill in (False, True):
        v2_backend.init_cuda_graph_moe_buffers(
            max_bs=12,
            max_loras=3,
            compute_dtype=torch.float32,
            moe_layer=None,
            prefill=prefill,
        )
    assert vars(v2_backend) == before


def test_lm_head_pruning_metadata_binds_current_batch_adapters(mapping_functions):
    info_type = _load_class_with_members(
        V2_BACKEND_PATH,
        "_LmHeadBatchInfo",
        include_fields=True,
        namespace={"torch": torch, "dataclass": dataclass, "ClassVar": ClassVar},
    )
    field_names = {"seg_indptr", "weight_indices", "expected_tokens", "max_len"}
    assert {field.name for field in fields(info_type)} == field_names
    assert info_type.use_cuda_graph is False
    infos = [
        info_type(torch.tensor(indptr), torch.tensor(slots), 3, 2)
        for indptr, slots in (([0, 2, 3], [1, 0]), ([0, 1, 3], [2, 1]))
    ]
    for info in infos:
        assert set(vars(info)) == field_names
        assert info.use_cuda_graph is False
    backend_type = _load_class_with_members(
        V2_BACKEND_PATH,
        "TritonV2LoRABackend",
        "_runner_for",
        namespace={"_compute_token_lora_mapping": mapping_functions},
    )
    backend = backend_type()
    bindings = []
    backend.runner = SimpleNamespace(phase="prefill")
    backend.lm_head_runner = SimpleNamespace(
        begin_batch=lambda **kwargs: bindings.append(kwargs)
    )
    assert backend._runner_for(None) is backend.runner
    for ranks, scales, expected in (
        ([0, 8, 16], [0.0, 0.25, 0.5], [[1, 1, -1], [2, 1, 1]]),
        ([0, 0, 4], [0.0, 0.0, 0.75], [[-1, -1, -1], [2, -1, -1]]),
    ):
        current = SimpleNamespace(
            lora_ranks=torch.tensor(ranks, dtype=torch.int32),
            scalings=torch.tensor(scales, dtype=torch.float32),
        )
        backend.batch_info = current
        for info, mapping in zip(infos, expected):
            assert backend._runner_for(info) is backend.lm_head_runner
            bound = bindings[-1]
            assert bound["lora_ranks"] is current.lora_ranks
            assert bound["scalings"] is current.scalings
            assert bound["token_slots"].tolist() == mapping
            assert bound["num_tokens"] == info.expected_tokens
            assert bound["phase"] == backend.runner.phase
            assert bound["graph_mode"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
