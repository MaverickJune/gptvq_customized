import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark
from gptvq_eval.benchmarks import DEFAULT_TASKS, save_json


class BenchmarkTests(unittest.TestCase):
    def test_default_lm_eval_tasks_match_sbvr(self):
        args = benchmark.parser().parse_args(["lm_eval"])
        self.assertEqual(args.tasks, DEFAULT_TASKS)
        self.assertEqual(args.batch_size, 1)

    def test_latency_defaults_to_explicit_sbvr_graph(self):
        args = benchmark.parser().parse_args(["latency"])
        self.assertEqual(args.cudagraph, "sbvr")
        self.assertTrue(args.disable_internal_cudagraphs)

    def test_save_json_creates_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            save_json({"score": 1.25}, output)
            self.assertEqual(json.loads(output.read_text()), {"score": 1.25})

    @mock.patch("benchmark.torch.cuda.is_available", return_value=False)
    def test_preflight_reports_missing_gpu(self, _cuda_available):
        with self.assertRaisesRegex(RuntimeError, "does not expose an NVIDIA driver"):
            benchmark.require_cuda("cuda:0")


if __name__ == "__main__":
    unittest.main()
