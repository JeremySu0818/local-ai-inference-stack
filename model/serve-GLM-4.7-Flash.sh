#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_BIN="$SCRIPT_DIR/../.venv/bin/vllm"

exec "$VLLM_BIN" serve unsloth/GLM-4.7-Flash-NVFP4 \
  --host 127.0.0.1 \
  --served-model-name unsloth/GLM-4.7-Flash-NVFP4 \
  --tensor-parallel-size 1 \
  --max-model-len 202752 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_MLA \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
