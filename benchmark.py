#!/usr/bin/env python3
"""Load a packed 4-bit checkpoint and run SBVR-style experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gptvq_eval.benchmarks import (
    DEFAULT_TASKS,
    evaluate_latency,
    evaluate_lm_eval,
    evaluate_ppl,
    save_json,
)
from gptvq_eval.cudagraph import attach_sbvr_cudagraph_generate


DEFAULT_MODEL = "models/Llama-2-7B-GPTQ"


def require_cuda(device):
    if torch.device(device).type != "cuda":
        raise RuntimeError("Packed GPTQ and CUDA-graph benchmarks require a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot access CUDA. Run this command in a GPU-enabled session; "
            "the current host/container does not expose an NVIDIA driver."
        )


def load_model(
    model_path,
    device="cuda:0",
    disable_internal_cudagraphs=True,
    require_packed=False,
):
    """Load a dense GPTVQ accuracy checkpoint or packed GPTQ checkpoint."""
    require_cuda(device)
    config = AutoConfig.from_pretrained(model_path)
    quant = getattr(config, "quantization_config", None)
    if quant is None and require_packed:
        raise ValueError(
            f"{model_path!r} is not a packed quantized checkpoint. The original "
            "GPTVQ output stores dequantized dense weights and is unsuitable for latency tests."
        )
    if quant is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            device_map={"": device},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    elif disable_internal_cudagraphs:
        # AutoGPTQ's legacy CUDA op captures but does not replay correctly.
        # Triton is warmed up by our helper before raw CUDA graph capture.
        from auto_gptq import AutoGPTQForCausalLM

        wrapped = AutoGPTQForCausalLM.from_quantized(
            model_path,
            device_map={"": device},
            low_cpu_mem_usage=True,
            use_triton=True,
            warmup_triton=False,
            disable_exllama=True,
            disable_exllamav2=True,
            inject_fused_attention=False,
            inject_fused_mlp=False,
        )
        model = wrapped.model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            device_map={"": device},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL)
    common.add_argument("--device", default="cuda:0")
    common.add_argument(
        "--disable-internal-cudagraphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable ExLlama/loader-specific optimized kernels (default; use "
            "--no-disable-internal-cudagraphs to opt in)."
        ),
    )
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    ppl = commands.add_parser("ppl", parents=[common])
    ppl.add_argument("--max-length", type=int, default=2048)
    ppl.add_argument("--stride", type=int, default=512)
    ppl.add_argument("--max-chunks", type=int)
    ppl.add_argument("--output", default="results/ppl.json")

    lm_eval = commands.add_parser("lm_eval", parents=[common])
    lm_eval.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    lm_eval.add_argument("--batch-size", type=int, default=1)
    lm_eval.add_argument("--limit", type=float)
    lm_eval.add_argument("--output", default="results/lm_eval.json")

    latency = commands.add_parser("latency", parents=[common])
    latency.add_argument("--cudagraph", choices=["none", "sbvr"], default="sbvr")
    latency.add_argument("--prompt", default="Hello my name is")
    latency.add_argument("--new-tokens", type=int, default=20)
    latency.add_argument("--warmup-runs", type=int, default=5)
    latency.add_argument("--measured-runs", type=int, default=10)
    latency.add_argument("--graph-warmups", type=int, default=3)
    latency.add_argument("--output", default="results/latency.json")

    commands.add_parser("preflight", parents=[common])
    return root


def main():
    args = parser().parse_args()
    model_path = Path(args.model)
    if not model_path.exists() and args.model == DEFAULT_MODEL:
        raise FileNotFoundError(
            f"Default model not found at {model_path}. Run scripts/download_model.sh first."
        )
    model, tokenizer = load_model(
        args.model,
        device=args.device,
        disable_internal_cudagraphs=args.disable_internal_cudagraphs,
        require_packed=args.command == "latency",
    )
    if args.command == "preflight":
        print(
            f"Loaded {args.model} on {args.device}; "
            f"quantization={getattr(model.config, 'quantization_config', None)}"
        )
        return
    if args.command == "ppl":
        result = {
            "wikitext": evaluate_ppl(
                model, tokenizer, args.max_length, args.stride, args.max_chunks
            )
        }
    elif args.command == "lm_eval":
        result = evaluate_lm_eval(
            model,
            tokenizer,
            args.tasks,
            args.batch_size,
            args.device,
            args.limit,
        )
    else:
        if args.cudagraph == "sbvr":
            attach_sbvr_cudagraph_generate(model, args.device, torch.float16, args.graph_warmups)
        result = evaluate_latency(
            model, tokenizer, args.prompt, args.new_tokens, args.warmup_runs, args.measured_runs
        )
        result["cudagraph"] = args.cudagraph
    save_json(result, args.output)
    print(result)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
