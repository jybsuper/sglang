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
"""Check captured-pointer lifetimes, pool ownership and audit lifecycle."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_executor.runner_backend_utils import cuda_graph_pointer_audit
from sglang.srt.model_executor.runner_backend_utils.cuda_graph_pointer_audit import (
    CudaGraphPointerAudit,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

_AUDIT_ON = {"SGLANG_DEBUG_CUDA_GRAPH_POINTERS": "1"}


class TestCudaGraphPointerAudit(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.audit = CudaGraphPointerAudit(self.device)
        self.audit.begin_session()
        self.stream = torch.cuda.Stream()

    def tearDown(self):
        self.audit.close()
        cuda_graph_pointer_audit.shutdown()

    def _graph(self, fn):
        self.audit.begin_graph()
        graph = self.audit.new_graph()
        with torch.cuda.graph(graph, stream=self.stream):
            out = fn()
        self.audit.add_graph(graph)
        return graph, out

    def _capture(self, fn, label):
        self.audit.begin_capture(self, label)
        graph, out = self._graph(fn)
        self.audit.end_capture()
        return graph, out

    def test_tensor_freed_after_capture_is_dangling(self):
        static = torch.ones(1024, device=self.device)
        victims = [torch.full((1024,), 2.0, device=self.device)]
        victim_ptr = victims[0].data_ptr()
        self._capture(lambda: (static + victims[0]) * 3.0, "victim-graph")
        victims.clear()

        with self.assertRaises(RuntimeError) as raised:
            self.audit.end_session()
        message = str(raised.exception)
        self.assertIn("audit dangling", message)
        self.assertIn("victim-graph", message)
        self.assertIn(f"{victim_ptr:#x}", message)
        self.assertIn("allocated before capture", message)
        self.assertIn("freed after capture", message)
        self.assertIn("test_cuda_graph_pointer_audit.py", message)

    def test_tensor_freed_during_capture_is_dangling(self):
        static = torch.ones(1024, device=self.device)
        holder = [torch.full((1024,), 2.0, device=self.device)]
        # The only reference dies inside the capture, after its kernel is recorded.
        self._capture(lambda: holder.pop() + static, "mid-capture")

        with self.assertRaises(RuntimeError) as raised:
            self.audit.end_session()
        self.assertIn("freed during capture", str(raised.exception))

    def test_static_inputs_and_temporaries_pass(self):
        static = torch.ones(1024, device=self.device)
        other = torch.full((1024,), 2.0, device=self.device)

        graph, out = self._capture(lambda: (static + other) * 3.0, "clean-graph")
        # A graph may read a tensor an earlier capture produced while it is alive.
        self._capture(lambda: out + static, "reader-graph")

        findings = self.audit.end_session()
        self.assertEqual(findings.status, "clean")
        self.assertEqual(findings.graphs, 2)
        self.assertGreater(findings.nodes, 0)
        self.assertGreater(findings.static, 0)
        self.assertGreater(findings.temporaries, 0)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, torch.full((1024,), 9.0, device=self.device)))

    def test_eager_bridge_between_segments_must_stay_alive(self):
        static = torch.ones(1024, device=self.device)
        self.audit.begin_capture(self, "segmented")
        _, first = self._graph(lambda: static * 2.0)
        # Eager work between two segments, as an eager_on_graph break: its
        # output is allocated outside the graph pool and read by the next segment.
        bridges = [first + 1.0]
        self._graph(lambda: bridges[0] * 2.0 + first)
        self.audit.end_capture()
        self.assertEqual(self.audit.analyze().status, "clean")

        bridges.clear()
        with self.assertRaises(RuntimeError) as raised:
            self.audit.end_session()
        message = str(raised.exception)
        self.assertIn("eager bridge allocated during the captured forward", message)
        self.assertIn("segmented segment=1", message)

    def test_bridge_from_an_unrelated_custom_pool_must_stay_alive(self):
        static = torch.ones(1024, device=self.device)
        pool = torch.cuda.MemPool()
        self.audit.begin_capture(self, "custom-pool")
        _, first = self._graph(lambda: static * 2.0)
        # Not the default pool and not the graphs' pool: still an eager bridge.
        with torch.cuda.use_mem_pool(pool):
            bridges = [first + 1.0]
        self._graph(lambda: bridges[0] * 2.0)
        self.audit.end_capture()
        self.assertEqual(self.audit.analyze().status, "clean")

        bridges.clear()
        with self.assertRaises(RuntimeError) as raised:
            self.audit.end_session()
        self.assertIn(
            "eager bridge allocated during the captured forward", str(raised.exception)
        )

    def test_missing_pool_information_is_incomplete(self):
        static = torch.ones(1024, device=self.device)
        self._capture(lambda: (static * 2.0) + 1.0, "no-pool-info")
        real_snapshot = torch.cuda.memory._snapshot

        def without_pool_ids(device=None):
            snapshot = real_snapshot(device)
            for entry in snapshot["device_traces"][self.device.index]:
                entry.pop("pool_id", None)
            return snapshot

        with patch.object(torch.cuda.memory, "_snapshot", side_effect=without_pool_ids):
            with self.assertRaises(RuntimeError) as raised:
                self.audit.end_session()
        message = str(raised.exception)
        self.assertIn("audit incomplete", message)
        self.assertIn("has no allocator pool information", message)

    def test_incomplete_trace_fails_instead_of_passing(self):
        static = torch.ones(1024, device=self.device)
        self.audit.close()
        begin = self.audit.mark()  # allocated before the trace starts
        self.audit.begin_session()
        graph = self.audit.new_graph()
        with torch.cuda.graph(graph, stream=self.stream):
            static * 2.0
        end = self.audit.mark()
        graph.instantiate()
        self.audit.records.append(
            cuda_graph_pointer_audit._GraphRecord(
                "evicted", id(self), graph, begin, begin, end, frozenset([graph.pool()])
            )
        )

        with self.assertRaises(RuntimeError) as raised:
            self.audit.end_session()
        message = str(raised.exception)
        self.assertIn("audit incomplete", message)
        self.assertIn("capture markers are no longer in the allocator trace", message)

    def test_checkpoint_sees_a_free_after_the_session_until_retired(self):
        inputs = [torch.ones(1024, device=self.device)]
        self._capture(lambda: inputs[0] * 2.0, "serving-graph")
        self.assertEqual(self.audit.end_session().status, "clean")

        inputs.clear()
        cuda_graph_pointer_audit._audits[self.device.index] = self.audit
        with self.assertRaisesRegex(RuntimeError, "after growth: .*audit dangling"):
            cuda_graph_pointer_audit.checkpoint("after growth")
        cuda_graph_pointer_audit.retire(self)
        self.assertIsNone(cuda_graph_pointer_audit.checkpoint("after retire"))

    def test_metadata_glue_reset_retires_only_its_records(self):
        from sglang.srt.model_executor.runner.metadata_glue_graph import (
            MetadataGlueGraph,
        )

        glue = MetadataGlueGraph(self.device)
        static = torch.ones(1024, device=self.device)
        self.audit.begin_capture(glue, "glue")
        self._graph(lambda: static * 2.0)
        self.audit.end_capture()
        self._capture(lambda: static * 3.0, "backend")
        cuda_graph_pointer_audit._audits[self.device.index] = self.audit

        glue.reset()

        self.assertEqual([record.label for record in self.audit.records], ["backend"])
        self.assertEqual(self.audit.end_session().status, "clean")

    def test_failed_session_keeps_earlier_records_and_stackless_tracing(self):
        inputs = [torch.ones(1024, device=self.device)]
        self._capture(lambda: inputs[0] * 2.0, "first-session")
        self.assertEqual(self.audit.end_session().status, "clean")
        cuda_graph_pointer_audit._audits[self.device.index] = self.audit

        with patch.dict(os.environ, _AUDIT_ON):
            with self.assertRaises(ValueError):
                with cuda_graph_pointer_audit.session() as audit:
                    self.assertIs(audit, self.audit)
                    audit.begin_capture(self, "doomed")
                    raise ValueError("capture failed")

        self.assertEqual(
            [record.label for record in self.audit.records], ["first-session"]
        )
        self.assertFalse(self.audit.capturing)
        probe = torch.empty(512, dtype=torch.uint8, device=self.device)
        traces = torch.cuda.memory._snapshot(self.device)["device_traces"]
        entry = [
            t
            for t in traces[self.device.index]
            if t["action"] == "alloc" and t["addr"] == probe.data_ptr()
        ][-1]
        self.assertEqual(entry.get("frames") or [], [])  # stackless between sessions
        self.assertEqual(
            cuda_graph_pointer_audit.checkpoint("after the failed session").status,
            "clean",
        )

    def test_failed_capture_loop_cleans_up(self):
        self.audit.close()
        with patch.dict(os.environ, _AUDIT_ON):
            with self.assertRaises(ValueError):
                with cuda_graph_pointer_audit.session() as audit:
                    self.assertIsNotNone(audit)
                    audit.begin_capture(self, "doomed")
                    raise ValueError("capture failed")
        self.assertIsNone(cuda_graph_pointer_audit.active())
        self.assertFalse(audit.capturing)
        self.assertEqual(audit.records, [])
        traces = torch.cuda.memory._snapshot(self.device)["device_traces"]
        self.assertEqual(traces[self.device.index], [])


if __name__ == "__main__":
    unittest.main()
