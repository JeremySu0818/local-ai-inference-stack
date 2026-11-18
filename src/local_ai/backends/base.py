"""The contract between the router and an inference backend, and the helpers it shares.

A backend is declared once, in its own package, as a :class:`Backend`.  The router
reads that declaration and never names an engine, so an engine is added by adding
a package and one entry to ``BACKENDS``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Any, Callable


ENDPOINTS = (
    "chat/completions",
    "completions",
    "embeddings",
    "images/generations",
    "images/edits",
    "audio/transcriptions",
    "audio/translations",
    "audio/speech",
    "responses",
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


@dataclass(frozen=True)
class Backend:
    """How the router runs one backend, whatever inference engine is behind it.

    A *server* backend names, in ``server``, the package whose ``serve`` entry
    point starts an OpenAI-compatible HTTP server on ``base_url``: the router
    starts it as a process, waits for ``health_path`` and proxies requests.

    A backend without a server is *native*: ``process`` builds an object with
    ``load``, ``generate`` and ``unload`` that runs in a worker of its own,
    ``parse`` turns a request body into the arguments of ``generate``, and
    ``load_hint`` picks a hint to pass to a cold ``load``.

    ``configure`` validates the keys that belong to this backend and returns the
    ``Model`` fields they produce; ``prepare`` fetches and checks its assets.
    """

    name: str
    server: str = ""
    base_url: str = DEFAULT_BASE_URL
    endpoints: tuple[str, ...] = ENDPOINTS
    local_model: bool = False
    configure: Callable[[dict], dict[str, Any]] | None = None
    process: Callable[[Any], Any] | None = None
    parse: Callable[[str, bytes, str], Any] | None = None
    load_hint: Callable[[Any], str | None] | None = None
    prepare: Callable[[Any], None] | None = None

    @property
    def native(self) -> bool:
        return self.process is not None


def configure_http(item: dict) -> dict[str, Any]:
    """A backend the configuration starts itself, on a URL it declares."""
    mid = item["id"]
    return {
        "command": command(item.get("command", []), mid),
        "base_url": item.get("base_url", "").rstrip("/"),
        "cwd": item.get("cwd"),
    }


def command(args: Any, mid: str) -> tuple[str, ...]:
    if not isinstance(args, list) or not args or not all(isinstance(x, str) and x for x in args):
        raise ValueError(f"{mid}: command must be a non-empty argument list")
    return tuple(args)


def repository(value: Any, mid: str) -> str:
    if not isinstance(value, str) or value.count("/") != 1 or not all(value.split("/")):
        raise ValueError(f"{mid}: hf_repo must be a Hugging Face owner/repository ID")
    return value


def cached_file(repo: str, filename: str) -> str | None:
    """The cached path of a repository file, or None when it is not there."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo, filename)
    return path if isinstance(path, str) and Path(path).is_file() else None


def download(repo: str, filenames: tuple[str, ...] = (), revision: str | None = None) -> None:
    """Fetch a repository into the shared cache, or say how to do it by hand."""
    try:
        if filenames:
            from huggingface_hub import hf_hub_download

            for filename in filenames:
                if cached_file(repo, filename) is None:
                    hf_hub_download(repo, filename)
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(repo, revision=revision)
    except Exception as error:
        args = [".venv/bin/hf", "download", repo, *filenames]
        if revision:
            args.extend(("--revision", revision))
        raise SystemExit(
            f"Download failed for {repo}: {error}\n"
            f"To download manually, run {shlex.join(args)} and retry ./setup.sh."
        ) from error
