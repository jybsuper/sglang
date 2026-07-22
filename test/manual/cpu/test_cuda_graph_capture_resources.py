import pytest

from sglang.srt.model_executor.runner_utils.capture_resources import (
    cuda_graph_capture_resource_scope,
    keep_cuda_graph_capture_resource,
)


def test_capture_resource_scope_owns_objects_and_resets():
    resource = object()

    with cuda_graph_capture_resource_scope() as resources:
        keep_cuda_graph_capture_resource(resource)
        assert resources == [resource]

    with pytest.raises(RuntimeError, match="no active owner"):
        keep_cuda_graph_capture_resource(object())


def test_nested_capture_resource_scopes_have_independent_owners():
    outer_resource = object()
    inner_resource = object()

    with cuda_graph_capture_resource_scope() as outer:
        keep_cuda_graph_capture_resource(outer_resource)
        with cuda_graph_capture_resource_scope() as inner:
            keep_cuda_graph_capture_resource(inner_resource)
        keep_cuda_graph_capture_resource(outer_resource)

    assert outer == [outer_resource, outer_resource]
    assert inner == [inner_resource]
