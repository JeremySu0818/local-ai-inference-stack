#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required. Install uv, then run this script again.\n' >&2
  exit 127
fi

cd "$ROOT_DIR"
uv venv --python 3.12 --allow-existing "$VENV_DIR"
uv pip install --torch-backend=auto -r "$ROOT_DIR/requirements.txt"

printf 'Environment ready: %s\n' "$VENV_DIR"
printf 'Router: %s/bin/uvicorn\n' "$VENV_DIR"
printf 'vLLM: %s/bin/vllm\n' "$VENV_DIR"
