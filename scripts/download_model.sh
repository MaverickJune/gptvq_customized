#!/usr/bin/env bash
set -euo pipefail

repo_id="${1:-TheBloke/Llama-2-7B-GPTQ}"
local_dir="${2:-models/Llama-2-7B-GPTQ}"

huggingface-cli download "$repo_id" \
  --exclude '*.safetensors' '*.bin' \
  --local-dir "$local_dir"
huggingface-cli download "$repo_id" \
  --include '*.safetensors' \
  --local-dir "$local_dir"
