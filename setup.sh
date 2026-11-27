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

PYTHON="$VENV_DIR/bin/python"
CONFIG_PATH="${AI_ROUTER_CONFIG:-$ROOT_DIR/config/models.yaml}"

BACKENDS="$(cd "$ROOT_DIR" && "$PYTHON" -c '
import sys
from local_ai.router.config import load_config
print(" ".join(sorted({m.backend for m in load_config(sys.argv[1]).models.values()})))
' "$CONFIG_PATH")"

need_llama=0
if [[ "$BACKENDS" == *llama_cpp* ]]; then need_llama=1; fi
need_edge0=0
if [[ "$BACKENDS" == *edge0* ]]; then need_edge0=1; fi

LLAMA_DIR=""
if [[ "$need_llama" == 1 ]]; then
  LLAMA_DIR=$("$PYTHON" -m local_ai.backends.llama_cpp source-dir)
fi
EDGE0_DIR=""
if [[ "$need_edge0" == 1 ]]; then
  EDGE0_DIR=$("$PYTHON" -m local_ai.backends.edge0 source-dir)
fi

build_llama=0
if [[ "$need_llama" == 1 && ! -x "$LLAMA_DIR/build/bin/llama-server" ]]; then build_llama=1; fi
build_edge0=0
if [[ "$need_edge0" == 1 && ! ( -f "$EDGE0_DIR/.router-access-log-v1" \
    && -f "$EDGE0_DIR/.router-vision-v1" \
    && -x "$EDGE0_DIR/edge0-infer/build/cpu/bin/edge0-infer" \
    && -x "$EDGE0_DIR/edge0-strata/build/strata" \
    && -x "$EDGE0_DIR/edge0-strata/.venv/bin/python" \
    && -x "$EDGE0_DIR/edge0-strata/build-vision/bin/strata-vision" ) ]]; then build_edge0=1; fi

need_cmake=0
need_ninja=0
need_nvcc=0
if [[ "$build_llama" == 1 || "$build_edge0" == 1 ]]; then need_cmake=1; need_nvcc=1; fi
if [[ "$build_edge0" == 1 ]]; then need_ninja=1; fi
if [[ "$build_llama" == 1 && "${CMAKE_GENERATOR:-}" == *[Nn]inja* ]]; then need_ninja=1; fi

install_build_tool() {
  if [[ ! -f "$TOOLS_DIR/bin/python" ]]; then
    uv venv --python 3.12 --allow-existing "$TOOLS_DIR"
  fi
  local pin
  pin="$(grep -E "^${1}==" "$ROOT_DIR/requirements/build.txt" || true)"
  uv pip install --python "$TOOLS_DIR/bin/python" "${pin:-$1}"
  export PATH="$TOOLS_DIR/bin:$PATH"
}

if [[ "$need_cmake" == 1 ]] && ! command -v cmake >/dev/null 2>&1; then
  printf 'Installing cmake...\n'
  install_build_tool cmake
fi
if [[ "$need_ninja" == 1 ]] && ! command -v ninja >/dev/null 2>&1; then
  printf 'Installing ninja...\n'
  install_build_tool ninja
fi
if [[ "$need_nvcc" == 1 ]] && ! command -v nvcc >/dev/null 2>&1; then
  printf 'Missing nvcc, required to build the CUDA backends. Install CUDA 13.\n' >&2
  exit 127
fi

if [[ "$build_llama" == 1 ]]; then
  bash "$ROOT_DIR/src/local_ai/backends/llama_cpp/build.sh" "$LLAMA_DIR"
fi
if [[ "$build_edge0" == 1 ]]; then
  bash "$ROOT_DIR/src/local_ai/backends/edge0/build.sh" "$EDGE0_DIR"
fi

"$PYTHON" -m scripts.prepare_models "$CONFIG_PATH"

printf 'Environment ready: %s\n' "$VENV_DIR"
printf 'Router: %s/bin/uvicorn\n' "$VENV_DIR"
if [[ "$need_llama" == 1 ]]; then
  printf 'llama.cpp: %s/build/bin/llama-server\n' "$LLAMA_DIR"
fi
if [[ "$need_edge0" == 1 ]]; then
  printf 'Edge0: %s/edge0-infer/build/cpu/bin/edge0-infer\n' "$EDGE0_DIR"
  printf 'Edge0 images: %s/edge0-strata/build-vision/bin/strata-vision\n' "$EDGE0_DIR"
fi
