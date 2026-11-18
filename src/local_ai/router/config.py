from __future__ import annotations
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import sys
from typing import Literal
from urllib.parse import urlsplit
import yaml

from local_ai.backends import get_backend


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROUTER_CONFIG = PROJECT_ROOT / "config/router.yaml"


@dataclass(frozen=True)
class Model:
    id: str
    command: tuple[str, ...]
    base_url: str
    health_path: str
    endpoints: tuple[str, ...]
    startup_timeout: float
    stop_timeout: float
    cwd: str | None
    backend: str = "http"
    model_path: str | None = None
    cpu_offload: bool | Literal["auto"] = False
    model_revision: str | None = None
    hf_repo: str | None = None
    gguf: str | None = None
    mmproj: str | None = None
    context_size: int | None = None
    parallel: int = 1


@dataclass(frozen=True)
class Config:
    models: dict[str, Model]
    queue_limit: int
    request_timeout: float
    state_dir: Path


def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise ValueError("configuration must contain a models list")
    models = {}
    for item in raw["models"]:
        mid = item["id"]
        if not isinstance(mid, str) or not mid or mid in models:
            raise ValueError("empty or duplicate model id")
        try:
            spec = get_backend(item.get("backend", "http"))
        except ValueError as exc:
            raise ValueError(f"{mid}: {exc}") from exc

        configured = spec.configure(item) if spec.configure is not None else {}

        cpu_offload = item.get("cpu_offload", False)
        if not spec.native:
            cpu_offload = bool(cpu_offload)
        elif type(cpu_offload) is not bool and cpu_offload != "auto":
            raise ValueError(f"{mid}: cpu_offload must be true, false, or auto")

        model_path = item.get("model_path") if spec.local_model else None
        if model_path and (not item.get("model_revision") or model_path.startswith(("/", ".", "~"))):
            model_path = str(Path(model_path).expanduser().resolve())

        ep = tuple(item.get("endpoints", ["chat/completions"]))
        if not ep or any(x not in spec.endpoints for x in ep):
            raise ValueError(f"{mid}: unsupported endpoint")
        health = item.get("health_path", "/health")
        if not health.startswith("/") or health.startswith("//") or "?" in health:
            raise ValueError("invalid health_path")
        start, stop = (
            float(item.get("startup_timeout_seconds", 600)),
            float(item.get("stop_timeout_seconds", 30)),
        )
        if min(start, stop) <= 0:
            raise ValueError("timeouts must be positive")

        fields = {
            "id": mid,
            "backend": spec.name,
            "command": (),
            "base_url": "",
            "cwd": None,
            "health_path": health,
            "endpoints": ep,
            "startup_timeout": start,
            "stop_timeout": stop,
            "model_path": model_path,
            "model_revision": item.get("model_revision"),
            "cpu_offload": cpu_offload,
            **configured,
        }
        if spec.server:
            fields["command"] = (
                sys.executable, "-m", spec.server, "serve", str(Path(path).resolve()), mid,
            )
            fields["base_url"] = spec.base_url
        if not spec.native:
            loopback(fields["base_url"], mid)
        models[mid] = Model(**fields)
    if not models:
        raise ValueError("at least one model required")
    runtime_config = ROUTER_CONFIG
    state_root = PROJECT_ROOT
    if not runtime_config.is_file():
        runtime_config = files("local_ai.router").joinpath("router.yaml")
        state_root = Path.cwd()
    with runtime_config.open(encoding="utf-8") as f:
        runtime = yaml.safe_load(f)
    if not isinstance(runtime, dict):
        raise ValueError("router configuration must be a mapping")

    q = int(raw.get("queue_limit", runtime["queue_limit"]))
    t = float(raw.get("request_timeout_seconds", runtime["request_timeout_seconds"]))
    if min(q, t) <= 0:
        raise ValueError("limits must be positive")
    state_dir = (
        Path(raw["state_dir"]).expanduser() if "state_dir" in raw
        else state_root / Path(runtime["state_dir"]).expanduser()
    )
    return Config(models, q, t, state_dir.resolve())


def loopback(base_url: str, mid: str) -> None:
    """The router only proxies to a server on this machine."""
    u = urlsplit(base_url)
    if (
        u.scheme != "http"
        or u.hostname not in ("127.0.0.1", "localhost", "::1")
        or not u.port
        or u.username
        or u.password
        or u.query
        or u.fragment
    ):
        raise ValueError(f"{mid}: base_url must be an HTTP loopback URL with port")
