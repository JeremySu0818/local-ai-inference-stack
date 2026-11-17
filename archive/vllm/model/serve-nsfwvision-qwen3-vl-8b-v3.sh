#!/usr/bin/env bash
export MAX_JOBS=6
export NVCC_THREADS=6
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_BIN="$SCRIPT_DIR/../../../.venv/bin/vllm"

exec "$VLLM_BIN" serve GitMylo/nsfwvision-qwen3-vl-8b-v3-safetensors \
  --host 127.0.0.1 \
  --served-model-name GitMylo/nsfwvision-qwen3-vl-8b-v3-safetensors \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype nvfp4 \
  --attention-backend FLASHINFER \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
