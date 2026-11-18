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
uv pip sync --torch-backend=cu132 "$ROOT_DIR/requirements/locked.txt"

PYTHON="$VENV_DIR/bin/python"
LLAMA_DIR=$("$PYTHON" -m backends.llama_cpp source-dir)
bash "$ROOT_DIR/backends/llama_cpp/build.sh" "$LLAMA_DIR"
"$PYTHON" -m scripts.prepare_models "${AI_ROUTER_CONFIG:-$ROOT_DIR/config/models.yaml}"

printf 'Environment ready: %s\n' "$VENV_DIR"
printf 'Router: %s/bin/uvicorn\n' "$VENV_DIR"
printf 'llama.cpp: %s/build/bin/llama-server\n' "$LLAMA_DIR"
