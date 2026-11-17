#!/usr/bin/env bash
export MAX_JOBS=6
export NVCC_THREADS=6
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_BIN="$SCRIPT_DIR/../../../.venv/bin/vllm"

exec "$VLLM_BIN" serve google/gemma-4-E4B-it \
  --host 127.0.0.1 \
  --served-model-name google/gemma-4-E4B-it \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.97 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --performance-mode throughput