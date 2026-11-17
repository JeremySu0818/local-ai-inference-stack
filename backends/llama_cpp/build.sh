#!/usr/bin/env bash
set -euo pipefail

LLAMA_DIR="$1"
if [[ ! -x "$LLAMA_DIR/build/bin/llama-server" ]]; then
  for tool in curl tar sha256sum cmake nvcc; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      printf 'Missing %s, required to build llama.cpp.\n' "$tool" >&2
      exit 127
    fi
  done
  if [[ ! -f "$LLAMA_DIR/CMakeLists.txt" ]]; then
    mkdir -p "$LLAMA_DIR"
    archive=$(mktemp)
    trap 'rm -f "$archive"' EXIT
    curl -fL --retry 5 --retry-delay 2 --connect-timeout 15 \
      https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/v1.0.0 -o "$archive"
    printf '%s  %s\n' '3f5f750e36326aea4d65c7623e7d597a33e395d65f4d76e5e0532171f252532b' "$archive" | sha256sum -c -
    tar -xzf "$archive" --strip-components=1 -C "$LLAMA_DIR"
    rm -f "$archive"
    trap - EXIT
  fi
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build" --config Release --target llama-server -j "${MAX_JOBS:-16}"
fi
