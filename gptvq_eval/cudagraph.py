"""SBVR-style explicit CUDA graph decoding for Hugging Face causal LMs."""

from __future__ import annotations

import types

import torch
from transformers.cache_utils import StaticCache


def attach_sbvr_cudagraph_generate(model, device="cuda:0", dtype=torch.float16, warmups=3):
    """Replace ``generate`` with eager prefill plus graphed one-token decode.

    The graph is built lazily for each prompt length/new-token count.  This is
    intentionally limited to batch size one and greedy decoding, matching the
    SBVR latency experiment.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SBVR CUDA-graph generation requires an accessible CUDA GPU")
    if warmups < 1:
        raise ValueError("warmups must be at least 1")

    model._gptvq_eager_generate = model.generate
    model._gptvq_cudagraphs = {}

    def build_graph(max_cache_len):
        cache = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        positions = torch.zeros_like(ids)
        cache_position = torch.zeros((1,), dtype=torch.long, device=device)
        mask = torch.zeros((1, max_cache_len), dtype=torch.bool, device=device)
        mask[:, 0] = True

        def decode_one():
            return model(
                input_ids=ids,
                position_ids=positions,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            ).logits

        # Quantized kernels often initialize handles or autotune on first use.
        with torch.inference_mode():
            for _ in range(warmups):
                cache.reset()
                decode_one()
        torch.cuda.synchronize(device)

        cache.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            logits = decode_one()
        return {
            "graph": graph,
            "logits": logits,
            "cache": cache,
            "ids": ids,
            "positions": positions,
            "cache_position": cache_position,
            "mask": mask,
        }

    def graph_generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=20,
        do_sample=False,
        **_kwargs,
    ):
        if do_sample:
            raise ValueError("SBVR CUDA-graph generation only supports greedy decoding")
        if input_ids.shape[0] != 1:
            raise ValueError("SBVR CUDA-graph generation only supports batch_size=1")
        if max_new_tokens <= 0:
            return input_ids

        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[1]
        max_cache_len = prompt_len + max_new_tokens
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)
        if not torch.all(attention_mask == 1):
            raise ValueError("The CUDA-graph latency path requires an unpadded prompt")

        state = self._gptvq_cudagraphs.get(max_cache_len)
        if state is None:
            state = build_graph(max_cache_len)
            self._gptvq_cudagraphs[max_cache_len] = state

        cache = state["cache"]
        mask = state["mask"]
        cache.reset()
        mask.zero_()
        mask[:, :prompt_len] = True
        with torch.inference_mode():
            output = self(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(prompt_len, device=device),
            )
            next_token = output.logits[:, -1].argmax(-1, keepdim=True)

            generated = [input_ids, next_token]
            for step in range(max_new_tokens - 1):
                position = prompt_len + step
                state["ids"].copy_(next_token)
                state["positions"].fill_(position)
                state["cache_position"].fill_(position)
                mask[:, position] = True
                state["graph"].replay()
                next_token = state["logits"][:, -1].argmax(-1, keepdim=True)
                generated.append(next_token)
        return torch.cat(generated, dim=1)

    model.generate = types.MethodType(graph_generate, model)
    return model
