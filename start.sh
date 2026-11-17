#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${AI_ROUTER_CONFIG:-$ROOT_DIR/config/models.yaml}"
UVICORN="$ROOT_DIR/.venv/bin/uvicorn"

if [[ ! -x "$UVICORN" ]]; then
  printf 'Uvicorn is missing. Run %s/setup.sh first.\n' "$ROOT_DIR" >&2
  exit 127
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  printf 'Router config not found: %s\n' "$CONFIG_PATH" >&2
  exit 1
fi

cd "$ROOT_DIR"
export AI_ROUTER_CONFIG="$CONFIG_PATH"

exec "$UVICORN" router.app:app_factory \
  --factory \
  --host "${AI_ROUTER_HOST:-0.0.0.0}" \
  --port "${AI_ROUTER_PORT:-9000}" \
  --workers 1 \
  --timeout-graceful-shutdown "${AI_ROUTER_GRACEFUL_SHUTDOWN_SECONDS:-3700}"
