"""Edge0 backend: Edge0 145B and 120B, served by edge0-infer on the edge0-strata engine.

edge0-infer is a portal: it starts edge0-strata, waits for the weights, and serves
the OpenAI endpoints on a loopback port, so the router treats it like any other
server backend.  Only this package knows the two binaries, the pack layout, and
the engine options the portal passes through.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

from ..base import Backend, download, repository

REVISION = "4a7964de4b337532aad74b03714125deb7506e70"


def cache_root() -> Path:
    """The shared cache the build lives in, resolved only when a path is needed.

    The registry imports this package into the router, so ``Path.home()`` must
    not run at import time: a process with no ``HOME`` and no passwd entry for
    its UID would otherwise fail to start even an HTTP-only router.
    """
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def source_dir() -> Path:
    """The Edge0 build this revision expects."""
    return cache_root() / "local-ai-inference-stack" / f"edge0-{REVISION[:7]}"


def infer() -> Path:
    """The edge0-infer portal the backend launches."""
    return source_dir() / "edge0-infer/build/cpu/bin/edge0-infer"


def strata_dir() -> Path:
    """The edge0-strata directory the portal starts."""
    return source_dir() / "edge0-strata"


def strata() -> Path:
    """The strata engine binary inside the strata directory."""
    return strata_dir() / "build/strata"


def strata_python() -> Path:
    """The interpreter the portal uses to start strata."""
    return strata_dir() / ".venv/bin/python"

DEFAULT_DEVICE = "cuda:0"
READY_TIMEOUT = 1800.0
RESERVED_ARGS = (
    "--pack", "--native", "--native-dense-gguf", "--ple-gguf", "--ple-io",
    "--expert-profile", "--expert-cache", "--prefill", "--spec", "--spec-min-p",
    "--mtp", "--max-context", "--kv",
)


def configure(item: dict) -> dict:
    """Validate one Edge0 entry and the engine arguments it is started with."""
    mid = item["id"]
    repo = item.get("hf_repo")
    if repo is None and not item.get("model_path"):
        raise ValueError(f"{mid}: hf_repo or model_path is required")
    if repo is not None:
        repository(repo, mid)
    device = item.get("device", DEFAULT_DEVICE)
    if not isinstance(device, str) or not re.fullmatch(r"cuda(:\d+)?", device):
        raise ValueError(f"{mid}: device must be cuda or cuda:<n>")
    context = item.get("context_size")
    if context is not None and (type(context) is not int or context <= 0):
        raise ValueError(f"{mid}: context_size must be a positive integer")
    return {
        "cwd": None,
        "hf_repo": repo,
        "device": device,
        "context_size": context,
    }


def model_dir(model) -> str:
    """The pack directory the engine reads: `model_path`, or the cached repository."""
    if model.model_path:
        return model.model_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model.hf_repo, local_files_only=True)
    except Exception as error:
        raise SystemExit(
            f"Edge0 pack {model.hf_repo} is not in the cache.\n"
            f"To download it, run .venv/bin/hf download {model.hf_repo} and retry ./setup.sh."
        ) from error


def engine_args(model, path: str, command: str) -> list[str]:
    """The edge0-infer command line for a model directory."""
    u = urlsplit(model.base_url)
    args = [str(infer()), command, "--model", path]
    for option in model.model_options:
        args.extend(("--model-option", option))
    if command == "inspect":
        return args + ["--json"]
    args.extend(("--alias", model.id, "--host", u.hostname, "--port", str(u.port)))
    if model.device:
        args.extend(("--device", model.device))
    if model.context_size:
        args.extend(("--context", str(model.context_size)))
    args.extend(("--strata-dir", str(strata_dir()), "--strata-python", str(strata_python())))
    for word in model.engine_args:
        args.extend(("--strata-arg", word))
    return args


def check(model, path: str | None = None) -> None:
    """Ask edge0-infer whether the pack has the files this entry's options need."""
    path = path or model_dir(model)
    report = subprocess.run(engine_args(model, path, "inspect"), capture_output=True, text=True)
    try:
        ok = report.returncode == 0 and json.loads(report.stdout).get("ok") is True
    except ValueError:
        ok = False
    if not ok:
        detail = (report.stderr or report.stdout).strip()
        raise SystemExit(f"Edge0 model {path} is not complete:\n{detail}")


def prepare(model) -> None:
    """Fetch the pack, then check it against the model options this entry uses."""
    if not model.model_path:
        download(model.hf_repo)
    check(model)


BACKEND = Backend(
    name="edge0",
    server="local_ai.backends.edge0",
    endpoints=("chat/completions", "responses"),
    options={"du": ("q8", "q4"), "ple-io": ("direct", "ram")},
    reserved_args=RESERVED_ARGS,
    local_model=True,
    ready_timeout=READY_TIMEOUT,
    configure=configure,
    prepare=prepare,
)
