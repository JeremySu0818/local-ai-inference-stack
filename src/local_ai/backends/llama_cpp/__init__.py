"""llama.cpp backend: GGUF models served by llama-server."""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import struct

from ..base import Backend, download, repository

VERSION = "b12653"

# llama.cpp's default logical batch size, which the physical batch is capped at.
DEFAULT_BATCH_SIZE = 2048

# The flags server.py sets itself and their aliases in llama.cpp's common/arg.cpp.
RESERVED_ARGS = (
    "-m", "--model", "-mm", "--mmproj", "-a", "--alias", "--host", "--port",
    "-c", "--ctx-size", "-np", "--parallel",
    "-kvu", "--kv-unified", "-no-kvu", "--no-kv-unified",
    "-ngl", "--gpu-layers", "--n-gpu-layers", "-fit", "--fit",
    "-fa", "--flash-attn", "-ctk", "--cache-type-k", "-ctv", "--cache-type-v",
    "--jinja", "--no-jinja", "--embedding", "--embeddings",
    "-b", "--batch-size", "-ub", "--ubatch-size",
)


def normalize_engine_arg(word: str) -> str:
    return word.replace("_", "-") if word.startswith("--") else word


def cache_root() -> Path:
    """The shared cache the build lives in, resolved only when a path is needed.

    The registry imports this package into the router, so ``Path.home()`` must
    not run at import time: a process with no ``HOME`` and no passwd entry for
    its UID would otherwise fail to start even an HTTP-only router.
    """
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def source_dir() -> Path:
    """The llama.cpp build this revision expects."""
    return cache_root() / "local-ai-inference-stack" / f"llama.cpp-{VERSION}"


def server_path() -> Path:
    """The llama-server binary the backend launches."""
    return source_dir() / "build/bin/llama-server"


def configure(item: dict) -> dict:
    """Validate one llama.cpp entry and the fields llama-server needs."""
    mid = item["id"]
    gguf = item.get("gguf")
    if not isinstance(gguf, str) or not gguf.endswith(".gguf"):
        raise ValueError(f"{mid}: gguf filename is required")
    hf_repo = repository(item.get("hf_repo", mid), mid)
    mmproj = item.get("mmproj")
    if mmproj is not None and (not isinstance(mmproj, str) or not mmproj.endswith(".gguf")):
        raise ValueError(f"{mid}: invalid mmproj filename")
    # Repositories that ship only the weights point elsewhere for the projector, and the
    # cache keys every file by its repository, so llama-server needs to know which one.
    mmproj_repo = item.get("mmproj_repo")
    if mmproj_repo is not None:
        if mmproj is None:
            raise ValueError(f"{mid}: mmproj_repo requires mmproj")
        mmproj_repo = repository(mmproj_repo, mid)
    if type(item.get("context_size")) is not int or item["context_size"] <= 0:
        raise ValueError(f"{mid}: context_size must be a positive integer")
    parallel = item.get("parallel", 1)
    if type(parallel) is not int or not 1 <= parallel <= 256:
        raise ValueError(f"{mid}: parallel must be an integer from 1 to 256")
    # llama-server sizes part of its scratch buffer as context x ubatch, so the engine's
    # 512-token ubatch needs gigabytes at long context. It also caps the physical batch
    # at the logical one, which it leaves at its own default.
    ubatch = item.get("ubatch_size")
    if ubatch is not None and (type(ubatch) is not int or not 1 <= ubatch <= DEFAULT_BATCH_SIZE):
        raise ValueError(
            f"{mid}: ubatch_size must be an integer from 1 to {DEFAULT_BATCH_SIZE} "
            f"(llama.cpp's default batch size)"
        )
    if ubatch is not None and ubatch > item["context_size"]:
        raise ValueError(f"{mid}: ubatch_size cannot exceed context_size")
    # server.py raises the ubatch to the whole context for embeddings, whose bidirectional
    # attention fails outright on a prompt longer than one ubatch.
    if ubatch is not None and "embeddings" in item.get("endpoints", []):
        raise ValueError(f"{mid}: ubatch_size is fixed to context_size for embeddings")
    dimensions = item.get("embedding_dimensions", [])
    if "embedding_dimensions" in item:
        if "embeddings" not in item.get("endpoints", []):
            raise ValueError(f"{mid}: embedding_dimensions requires the embeddings endpoint")
        if (
            not isinstance(dimensions, list)
            or not dimensions
            or any(type(size) is not int or size <= 0 for size in dimensions)
            or len(set(dimensions)) != len(dimensions)
            or dimensions[0] != max(dimensions)
        ):
            raise ValueError(
                f"{mid}: embedding_dimensions must list the native dimension first, "
                "followed by any supported smaller dimensions"
            )
    return {
        "cwd": None,
        "hf_repo": hf_repo,
        "gguf": gguf,
        "mmproj": mmproj,
        "mmproj_repo": mmproj_repo or hf_repo,
        "context_size": item["context_size"],
        "parallel": parallel,
        "ubatch_size": ubatch,
        "embedding_dimensions": tuple(dimensions),
    }


def prepare(model) -> None:
    """Put the model's weights in the shared cache so llama-server can open them."""
    for repo, filename in ((model.hf_repo, model.gguf), (model.mmproj_repo, model.mmproj)):
        if filename is not None:
            download(repo, (filename,))


async def parse(model, endpoint: str, body: bytes, content_type: str) -> tuple[int, int] | None:
    """Validate an embeddings request and return native and target sizes for truncation."""
    if endpoint != "embeddings":
        return None
    if not content_type.lower().startswith("application/json"):
        raise ValueError("embeddings requests require application/json")
    request = json.loads(body)
    if not isinstance(request, dict):
        raise ValueError("embeddings requests require a JSON object")
    dimensions = request.get("dimensions")
    if dimensions is not None and request.get("input") is None:
        raise ValueError("dimensions requires a non-null input field")
    if request.get("input", request.get("content")) is None:
        raise ValueError("embeddings require a non-null input or content field")
    if dimensions is None:
        return None
    supported = model.embedding_dimensions
    if not supported:
        raise ValueError("dimensions is not configured for this model")
    if type(dimensions) is not int or dimensions not in supported:
        raise ValueError(
            f"dimensions must be one of {', '.join(map(str, supported))}"
        )
    native = supported[0]
    return None if dimensions == native else (native, dimensions)


def transform(body: bytes, sizes: tuple[int, int]) -> bytes:
    """Truncate Matryoshka embeddings and renormalize each resulting vector."""
    native, dimensions = sizes
    response = json.loads(body)
    for item in response["data"]:
        vector = item["embedding"]
        encoded = isinstance(vector, str)
        if encoded:
            raw = base64.b64decode(vector, validate=True)
            vector = [value for (value,) in struct.iter_unpack("<f", raw)]
        if not isinstance(vector, list) or len(vector) != native:
            raise ValueError(f"upstream embedding must contain {native} dimensions")
        shortened = _unit(vector[:dimensions])
        item["embedding"] = (
            base64.b64encode(struct.pack(f"<{dimensions}f", *shortened)).decode("ascii")
            if encoded else shortened
        )
    return json.dumps(response).encode()


def _unit(values):
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else list(values)


BACKEND = Backend(
    name="llama_cpp",
    server="local_ai.backends.llama_cpp",
    reserved_args=RESERVED_ARGS,
    normalize_engine_arg=normalize_engine_arg,
    configure=configure,
    parse=parse,
    transform=transform,
    prepare=prepare,
)
