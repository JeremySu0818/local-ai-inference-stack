"""llama.cpp backend: GGUF models served by llama-server."""
from __future__ import annotations

import os
from pathlib import Path

from ..base import Backend, download, repository

VERSION = "v1.0.0"
CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
SOURCE_DIR = CACHE_ROOT / "local-ai-inference-stack" / f"llama.cpp-{VERSION}"
SERVER = SOURCE_DIR / "build/bin/llama-server"


def configure(item: dict) -> dict:
    """Validate one llama.cpp entry and the fields llama-server needs."""
    mid = item["id"]
    gguf = item.get("gguf")
    if not isinstance(gguf, str) or not gguf.endswith(".gguf"):
        raise ValueError(f"{mid}: gguf filename is required")
    mmproj = item.get("mmproj")
    if mmproj is not None and (not isinstance(mmproj, str) or not mmproj.endswith(".gguf")):
        raise ValueError(f"{mid}: invalid mmproj filename")
    if type(item.get("context_size")) is not int or item["context_size"] <= 0:
        raise ValueError(f"{mid}: context_size must be a positive integer")
    parallel = item.get("parallel", 1)
    if type(parallel) is not int or not 1 <= parallel <= 256:
        raise ValueError(f"{mid}: parallel must be an integer from 1 to 256")
    return {
        "cwd": None,
        "hf_repo": repository(item.get("hf_repo", mid), mid),
        "gguf": gguf,
        "mmproj": mmproj,
        "context_size": item["context_size"],
        "parallel": parallel,
    }


def prepare(model) -> None:
    """Put the model's weights in the shared cache so llama-server can open them."""
    download(model.hf_repo, tuple(f for f in (model.gguf, model.mmproj) if f))


BACKEND = Backend(
    name="llama_cpp",
    server="local_ai.backends.llama_cpp",
    configure=configure,
    prepare=prepare,
)
