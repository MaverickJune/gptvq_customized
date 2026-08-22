"""SBVR-compatible PPL, lm-eval, and generation latency benchmarks."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch


DEFAULT_TASKS = [
    "commonsense_qa",
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "piqa",
    "winogrande",
]


def save_json(value, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


@torch.no_grad()
def evaluate_ppl(model, tokenizer, max_length=2048, stride=512, max_chunks=None):
    """Run the same overlapping WikiText-2 validation evaluation as SBVR."""
    from datasets import load_dataset
    from tqdm import tqdm

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    device = next(model.parameters()).device
    max_length = min(max_length, model.config.max_position_embeddings)
    starts = list(range(0, input_ids.numel() - max_length + 1, stride))
    if max_chunks is not None:
        starts = starts[:max_chunks]
    if not starts:
        raise ValueError(f"WikiText-2 contains fewer than {max_length} tokens")

    nll = torch.zeros((), device=device, dtype=torch.float32)
    for start in tqdm(starts, desc="WikiText-2 PPL chunks"):
        chunk = input_ids[start:start + max_length].unsqueeze(0).to(device)
        nll += model(input_ids=chunk, labels=chunk).loss.float() * max_length
    return math.exp((nll / (len(starts) * max_length)).item())


@torch.no_grad()
def evaluate_lm_eval(
    model, tokenizer, tasks=None, batch_size=1, device="cuda:0", limit=None
):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tasks = tasks or DEFAULT_TASKS
    wrapped = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    result = lm_eval.simple_evaluate(model=wrapped, tasks=tasks, limit=limit)
    return {task: result["results"][task] for task in tasks}


@torch.no_grad()
def evaluate_latency(
    model,
    tokenizer,
    prompt="Hello my name is",
    new_tokens=20,
    warmup_runs=5,
    measured_runs=10,
):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def timed(tokens):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=tokens, do_sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - start, output

    # Match SBVR's full-generation warmup, then prime the TTFT path twice.
    timed(new_tokens)
    for _ in range(warmup_runs):
        timed(new_tokens)
    for _ in range(2):
        timed(1)

    ttft = [timed(1)[0] for _ in range(measured_runs)]
    full = []
    output = None
    for _ in range(measured_runs):
        elapsed, output = timed(new_tokens)
        full.append(elapsed)
    ttft_s = sum(ttft) / len(ttft)
    full_s = sum(full) / len(full)
    tbt_ms = 0.0 if new_tokens == 1 else (full_s - ttft_s) * 1000 / (new_tokens - 1)
    return {
        "ttft_ms": ttft_s * 1000,
        "tbt_ms": tbt_ms,
        "full_latency_ms": full_s * 1000,
        "new_tokens": new_tokens,
        "prompt": prompt,
        "response": tokenizer.decode(output[0], skip_special_tokens=True),
    }
