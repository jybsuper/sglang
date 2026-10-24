"""TRTLLM MLA graph variants must retain the buffers replay updates."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import trtllm_mla_backend as backend_module
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _backend():
    backend = backend_module.TRTLLMMLABackend.__new__(backend_module.TRTLLMMLABackend)
    backend.decode_cuda_graph_metadata = {}
    backend.decode_cuda_graph_kv_indices = torch.zeros((4, 4), dtype=torch.int32)
    backend.max_context_len = 32
    backend.num_draft_tokens = 3
    backend._calc_padded_blocks = lambda _: 4
    backend.kv_index_translator = SimpleNamespace(
        is_translating=True,
        copy_page_table=lambda plan, out: out.copy_(plan[: out.shape[0]]),
    )
    return backend


class TestTRTLLMMLAGraphMetadata(unittest.TestCase):
    @patch.object(
        backend_module, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
    )
    def test_repeated_capture_keeps_the_buffers_refreshed_by_replay(self, _parallel):
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            with self.subTest(mode=mode):
                backend = _backend()
                lengths = torch.tensor([1, 1], dtype=torch.int64)
                num_tokens = 2 if mode.is_decode_or_idle() else 6
                backend._init_cuda_graph_metadata(
                    2, num_tokens, mode, lengths, lengths.device
                )
                captured = backend.forward_decode_metadata
                backend._init_cuda_graph_metadata(
                    2, num_tokens, mode, lengths, lengths.device
                )
                page_table = torch.arange(8, dtype=torch.int32).reshape(2, 4)
                lengths.copy_(torch.tensor([7, 11]))
                backend._apply_cuda_graph_metadata(
                    page_table, 2, torch.tensor([0, 1]), lengths, mode
                )

                expected = lengths.to(torch.int32)
                if mode.is_target_verify():
                    expected += backend.num_draft_tokens
                torch.testing.assert_close(captured.seq_lens_k, expected)
                torch.testing.assert_close(captured.block_kv_indices, page_table)
                self.assertIs(backend.forward_decode_metadata, captured)

    @patch.object(
        backend_module, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
    )
    def test_modes_and_batch_sizes_are_isolated_but_idle_reuses_decode(self, _parallel):
        backend = _backend()
        lengths = torch.ones(4, dtype=torch.int64)
        captured = []
        for mode, bs in (
            (ForwardMode.DECODE, 2),
            (ForwardMode.TARGET_VERIFY, 2),
            (ForwardMode.DRAFT_EXTEND_V2, 2),
            (ForwardMode.DECODE, 4),
        ):
            backend._init_cuda_graph_metadata(
                bs, bs * 3, mode, lengths[:bs], lengths.device
            )
            captured.append(backend.forward_decode_metadata)
        self.assertEqual(len({m.seq_lens_k.data_ptr() for m in captured}), 4)
        backend._init_cuda_graph_metadata(
            2, 2, ForwardMode.IDLE, lengths[:2], lengths.device
        )
        self.assertIs(backend.forward_decode_metadata, captured[0])


if __name__ == "__main__":
    unittest.main()
