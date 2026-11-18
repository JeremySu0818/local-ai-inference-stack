#!/usr/bin/env bash
set -euo pipefail

REVISION=4a7964de4b337532aad74b03714125deb7506e70
EDGE0_URL=https://github.com/Edge0-AI/Edge0/archive/${REVISION}.tar.gz
EDGE0_SHA=0e1080a9762aea455d64d40e917db6f04eb3a5515be70b63ab78b4ff8b67b3c9
GGML_REVISION=3cf03257f219afbe7334045ff7c6a06ac68c627d
GGML_URL=https://github.com/ggml-org/llama.cpp/archive/${GGML_REVISION}.tar.gz
GGML_SHA=c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280

EDGE0_DIR="$1"
INFER="$EDGE0_DIR/edge0-infer/build/cpu/bin/edge0-infer"
STRATA="$EDGE0_DIR/edge0-strata/build/strata"
STRATA_PYTHON="$EDGE0_DIR/edge0-strata/.venv/bin/python"
if [[ -x "$INFER" && -x "$STRATA" && -x "$STRATA_PYTHON" ]]; then
  exit 0
fi
jobs="${MAX_JOBS:-$(nproc)}"
for tool in curl tar sha256sum cmake ninja nvcc uv; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'Missing %s, required to build Edge0.\n' "$tool" >&2
    exit 127
  fi
done

fetch() {
  local url=$1 sha=$2 dir=$3
  local archive
  archive=$(mktemp)
  trap 'rm -f "$archive"' EXIT
  curl -fL --retry 5 --retry-delay 2 --connect-timeout 15 "$url" -o "$archive"
  printf '%s  %s\n' "$sha" "$archive" | sha256sum -c -
  mkdir -p "$dir"
  tar -xzf "$archive" --strip-components=1 -C "$dir"
  rm -f "$archive"
  trap - EXIT
}

if [[ ! -f "$EDGE0_DIR/edge0-strata/edge0/CONTRACT" ]]; then
  printf 'Fetching Edge0 %s...\n' "${REVISION:0:7}"
  fetch "$EDGE0_URL" "$EDGE0_SHA" "$EDGE0_DIR"
fi
ggml="$EDGE0_DIR/edge0-strata/third_party/llama.cpp"
if [[ ! -f "$ggml/ggml/CMakeLists.txt" ]]; then
  printf 'Fetching ggml...\n'
  fetch "$GGML_URL" "$GGML_SHA" "$ggml"
fi

if [[ ! -x "$STRATA" ]]; then
  printf 'Building edge0-strata...\n'
  cmake -S "$EDGE0_DIR/edge0-strata" -B "$EDGE0_DIR/edge0-strata/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
    -DCMAKE_CUDA_ARCHITECTURES="${STRATA_CUDA_ARCHITECTURES:-native}" \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_GGML_DIR="$ggml"
  cmake --build "$EDGE0_DIR/edge0-strata/build" --target strata -j "$jobs"
fi

if [[ ! -x "$STRATA_PYTHON" ]]; then
  printf 'Preparing the engine environment...\n'
  uv venv --python 3.12 --allow-existing "$EDGE0_DIR/edge0-strata/.venv"
  uv pip install --python "$EDGE0_DIR/edge0-strata/.venv/bin/python" \
    -r "$EDGE0_DIR/edge0-strata/requirements.txt"
fi

if [[ ! -x "$INFER" ]]; then
  printf 'Building edge0-infer...\n'
  (
    cd "$EDGE0_DIR/edge0-infer"
    cmake --preset cpu
    cmake --build --preset cpu --target edge0-infer -j "$jobs"
  )
fi
