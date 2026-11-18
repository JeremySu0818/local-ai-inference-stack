#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
TOOLS_DIR="$ROOT_DIR/.tools"

if ! command -v curl >/dev/null 2>&1; then
  printf 'Missing curl, required by setup.sh.\n' >&2
  exit 127
fi
if ! command -v uv >/dev/null 2>&1; then
  printf 'Installing uv...\n'
  curl -fsSL --retry 5 https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is still missing. Install uv, then run this script again.\n' >&2
    exit 127
  fi
fi

cd "$ROOT_DIR"
uv venv --python 3.12 --allow-existing "$VENV_DIR"
uv pip sync --torch-backend=cu132 "$ROOT_DIR/requirements/locked.txt"
uv pip install -e "$ROOT_DIR"

if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  printf 'Installing cmake and ninja...\n'
  uv venv --python 3.12 --allow-existing "$TOOLS_DIR"
  uv pip install --python "$TOOLS_DIR/bin/python" -r "$ROOT_DIR/requirements/build.txt"
  export PATH="$TOOLS_DIR/bin:$PATH"
fi
if ! command -v nvcc >/dev/null 2>&1; then
  printf 'Missing nvcc, required to build the CUDA backends. Install CUDA 13.\n' >&2
  exit 127
fi

PYTHON="$VENV_DIR/bin/python"
LLAMA_DIR=$("$PYTHON" -m local_ai.backends.llama_cpp source-dir)
bash "$ROOT_DIR/src/local_ai/backends/llama_cpp/build.sh" "$LLAMA_DIR"
EDGE0_DIR=$("$PYTHON" -m local_ai.backends.edge0 source-dir)
bash "$ROOT_DIR/src/local_ai/backends/edge0/build.sh" "$EDGE0_DIR"
"$PYTHON" -m scripts.prepare_models "${AI_ROUTER_CONFIG:-$ROOT_DIR/config/models.yaml}"

printf 'Environment ready: %s\n' "$VENV_DIR"
printf 'Router: %s/bin/uvicorn\n' "$VENV_DIR"
printf 'llama.cpp: %s/build/bin/llama-server\n' "$LLAMA_DIR"
printf 'Edge0: %s/edge0-infer/build/cpu/bin/edge0-infer\n' "$EDGE0_DIR"
