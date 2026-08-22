import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    @mock.patch("benchmark.AutoTokenizer.from_pretrained")
    @mock.patch("benchmark.AutoModelForCausalLM.from_pretrained")
    @mock.patch("benchmark.AutoConfig.from_pretrained")
    @mock.patch("benchmark.require_cuda")
    def test_dense_gptvq_checkpoint_is_loaded_for_accuracy_evaluation(
        self, _require_cuda, config_loader, model_loader, tokenizer_loader
    ):
        config = SimpleNamespace()
        config_loader.return_value = config
        model = mock.Mock()
        model_loader.return_value = model
        tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=2)
        tokenizer_loader.return_value = tokenizer

        loaded_model, loaded_tokenizer = benchmark.load_model("dense-gptvq")

        model_loader.assert_called_once_with(
            "dense-gptvq",
            config=config,
            device_map={"": "cuda:0"},
            torch_dtype=benchmark.torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval.assert_called_once_with()
        self.assertIs(loaded_model, model)
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)

    @mock.patch("benchmark.AutoConfig.from_pretrained")
    @mock.patch("benchmark.require_cuda")
    def test_latency_rejects_dense_gptvq_checkpoint(self, _require_cuda, config_loader):
        config_loader.return_value = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "unsuitable for latency tests"):
            benchmark.load_model("dense-gptvq", require_packed=True)


if __name__ == "__main__":
    unittest.main()
