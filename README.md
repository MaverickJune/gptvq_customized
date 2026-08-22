# GPTVQ: quantization and SBVR-style evaluation

This repository contains the code for the paper [GPTVQ: The Blessing of Dimensionality in LLM Quantization](https://arxiv.org/abs/2402.15319) and a reproducible evaluation entry point for perplexity, `lm-eval`, and CUDA-graph latency experiments.
This codebase is based upon the codebase for for the ICLR 2023 paper [GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323),
downloaded [The GPTQ GitHub page](https://github.com/IST-DASLab/gptq/).


## Abstract
In this work we show that the accuracy and efficiency of neural network quantization can be significantly improved by increasing the quantization dimensionality. We propose the GPTVQ~method, a new fast method for post-training vector quantization (VQ) that scales well to Large Language Models (LLMs). 
Our method interleaves quantization of one or more columns with updates to the remaining unquantized weights, using information from the Hessian of the per-layer output reconstruction MSE.
Quantization codebooks are initialized using an efficient data-aware version of the EM algorithm. The codebooks are then updated, and further compressed by using integer quantization and SVD-based compression. 
GPTVQ establishes a new state-of-the art in the size vs accuracy trade-offs on a wide range of LLMs such as Llama-v2 and Mistral. 
Furthermore, our method is efficient: on a single H100 it takes between 3 and 11 hours to process a Llamav2-70B model, depending on quantization setting.
Lastly, with on-device timings for VQ decompression on a mobile CPU we show that VQ leads to improved latency compared to using a 4-bit integer format.


## Environment

The tested environment name is `gptvq_env`:

```bash
conda create -n gptvq_env python=3.10 pip
conda activate gptvq_env
pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

CUDA 12.x and an NVIDIA GPU are required for packed GPTQ inference and CUDA graph capture. Check access before loading the 7B model:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

### Llama 2 7B, packed 4-bit checkpoint

Download the ungated 4-bit GPTQ checkpoint used by the evaluation commands (about 4 GB):

```bash
bash scripts/download_model.sh
```

This writes `models/Llama-2-7B-GPTQ`. A different Hugging Face repository and destination can be supplied as the first and second arguments. Validate the complete load on a GPU with:

```bash
python benchmark.py preflight
```

There are two different meanings of "4-bit" in this repository. `llama.py --wbits 4` runs the original GPTVQ/GPTQ research algorithm and replaces weights with quantized values, but saves those values as dense floating-point Hugging Face tensors. That output is appropriate for accuracy experiments, not compressed-memory or kernel-latency measurements. `benchmark.py` deliberately requires a packed checkpoint with `quantization_config` for latency tests.

## SBVR-style evaluation

All commands below load the same packed model and save machine-readable JSON under `results/`. The PPL implementation follows the referenced SBVR experiment: WikiText-2 validation text, 2048-token overlapping windows, and stride 512.

### 1. WikiText-2 perplexity

```bash
conda activate gptvq_env
python benchmark.py ppl \
  --model models/Llama-2-7B-GPTQ \
  --max-length 2048 \
  --stride 512 \
  --output results/llama2_7b_gptq4_ppl.json
```

For a quick smoke test, add `--max-chunks 2`. Omit it for the reported score.

### 2. lm-evaluation-harness

The default task set matches the SBVR package: `commonsense_qa`, `arc_challenge`, `arc_easy`, `hellaswag`, `piqa`, and `winogrande`.

```bash
python benchmark.py lm_eval \
  --model models/Llama-2-7B-GPTQ \
  --batch-size 1 \
  --output results/llama2_7b_gptq4_lm_eval.json
```

Use a subset by passing, for example, `--tasks piqa arc_easy`. For a pipeline-only smoke test, add `--limit 2`; do not use `--limit` for reported scores.

Datasets are downloaded by Hugging Face on first use. Set `HF_HOME` before running if the cache should live on a different disk.

## CUDA-graph latency

The latency test mirrors the SBVR measurement: eager prompt prefill, one-token CUDA graph replay during decode, five warmup calls, and ten measured calls. It reports batched-call TTFT, average time between subsequent tokens (TBT), and 20-token end-to-end latency.

```bash
python benchmark.py latency \
  --model models/Llama-2-7B-GPTQ \
  --cudagraph sbvr \
  --new-tokens 20 \
  --output results/llama2_7b_gptq4_sbvr_cudagraph_latency.json
```

For an eager comparison using the same prompt and timing loop:

```bash
python benchmark.py latency \
  --model models/Llama-2-7B-GPTQ \
  --cudagraph none \
  --output results/llama2_7b_gptq4_eager_latency.json
```

By default, and explicitly with `--disable-internal-cudagraphs`, the loader disables ExLlama and selects AutoGPTQ's Triton linear kernel. The Triton kernels are warmed eagerly before the SBVR-style raw graph is captured. `--no-disable-internal-cudagraphs` opts back into loader-specific optimized kernels; do not use that option when measuring the explicit SBVR graph. The explicit graph path supports batch size 1, greedy decoding, and an unpadded prompt.

## Original GPTVQ experiments

The original shell scripts remain available. To create a dense 4-bit GPTVQ accuracy checkpoint from an authorized Llama 2 model, a representative invocation is:

```bash
python llama.py \
  --use-vq --wbits 4 --vq-dim 2 \
  --columns-per-group 256 --groupsize 32768 \
  --kmeans-iters 100 --kmeans-init-method mahalanobis \
  --hessian-weighted-lookups --include-m-step \
  --codebook-bitwidth 8 --quantize-per-codebook \
  --output-dir outputs/llama2_7b_gptvq4_dense \
  meta-llama/Llama-2-7b-hf wikitext2
```

Access to `meta-llama/Llama-2-7b-hf` is gated. Authenticate with `huggingface-cli login` or set `HF_TOKEN`; do not put a token in source code. Quantization is compute-intensive and the paper used a single 80 GB H100. Reduce `--assignment-chunk-size` if memory is constrained.

## Original notes

See `requirements.txt` for pinned versions.

All experiments were run on a single 80GB NVIDIA H100. However, most experiments will work on a GPU with a lot less memory as well.
In case experiments run out of memory, the `--assignment-chunk-size` argument can be used to reduce memory requirements.
A lower value for this argument will reduce memory requirements at the expense of longer run times.


## Reproducibility

### Installation
Requirements are listed in `requirements.txt`. Install these in your environment using
```
pip install -r requirements.txt
```

Modify your `PYTHONPATH` to include the root directory of this repository.

The original experiments used Python 3.9; the maintained environment above uses Python 3.10.


### Experiments
Scripts to reproduce results in the paper are included as shell scripts. 

### Models
To run these scripts, the following environment variables need to be set to point to the relevant models:

```
LLAMA1_13B_PATH
LLAMA1_30B_PATH
LLAMA1_65B_PATH
LLAMA1_7B_PATH
LLAMA2_13B_PATH
LLAMA2_70B_PATH
LLAMA2_7B_PATH
MISTRAL_7B_PATH
MIXTRAL_PATH
```

These can point to either models on the HuggingFace model hub, or to local checkpoints for the corresponding architectures.

NB1: For gated models, authenticate with `huggingface-cli login` or set `HF_TOKEN` in the environment.
NB2: The HuggingFace checkpoints in these scripts were not necessarily the same as the model checkpoints used for the experiments in the paper. As a result, minor differences might occur. 


### Datasets
For calibration the `wikitext2` training set from the HuggingFace datasets hub is used. Perplexity results are generated using the `wikitext2` test set is used.

To generate zero-shot results, add the `--output-dir` argument to the command for an experiment. The VQ quantized model will be saved in this directory.
This command is by default *NOT* included in the experiment scripts in this repository, to avoid excessive file storage requirements.
Afterwards, run the [`llm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) on the checkpoint stored in this directory.

```
lm_eval --model hf \
    --model_args pretrained=/PATH/TO/CHECKPOINT \
    --tasks piqa,boolq,winogrande,hellaswag,arc_easy,arc_challenge \
    --device cuda:0
    --batch_size auto
```

## Cite

If you found this work useful, please consider citing:

```
@article{vanbaalen-gptvq,
  title={GPTVQ: The Blessing of Dimensionality in LLM Quantization}, 
  author={Mart van Baalen and Andrey Kuzmin and Markus Nagel and Peter Couperus and Cedric Bastoul and Eric Mahurin and Tijmen Blankevoort and Paul Whatmough},
  year={2024},
  journal={arXiv preprint arXiv:2402.15319}
}
```
