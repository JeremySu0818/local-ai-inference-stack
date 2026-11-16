from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
import yaml


@dataclass(frozen=True)
class Model:
    id: str
    command: tuple[str, ...]
    base_url: str
    health_path: str
    endpoints: tuple[str, ...]
    startup_timeout: float
    stop_timeout: float
    ttl: float
    cwd: str | None
    backend: str = "http"
    model_path: str | None = None
    cpu_offload: bool = False


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
    allowed = {
        "chat/completions",
        "completions",
        "embeddings",
        "images/generations",
        "images/edits",
        "audio/transcriptions",
        "audio/translations",
        "audio/speech",
        "responses",
    }
    for item in raw["models"]:
        mid = item["id"]
        backend = item.get("backend", "http")
        if backend not in ("http", "qwen_image"):
            raise ValueError(f"{mid}: unsupported backend")
        native = backend == "qwen_image"
        if native and not item.get("model_path"):
            raise ValueError(f"{mid}: model_path is required")
        cmd = item.get("command", []) if native else item["command"]
        u = urlsplit(item.get("base_url", ""))
        if not isinstance(mid, str) or not mid or mid in models:
            raise ValueError("empty or duplicate model id")
        if not native and (
            not isinstance(cmd, list)
            or not cmd
            or not all(isinstance(x, str) and x for x in cmd)
        ):
            raise ValueError(f"{mid}: command must be a non-empty argument list")
        if not native and (
            u.scheme != "http"
            or u.hostname not in ("127.0.0.1", "localhost", "::1")
            or not u.port
            or u.username
            or u.password
            or u.query
            or u.fragment
        ):
            raise ValueError(f"{mid}: base_url must be an HTTP loopback URL with port")
        ep = tuple(item.get("endpoints", ["chat/completions"]))
        if native and any(x not in ("images/generations", "images/edits") for x in ep):
            raise ValueError(f"{mid}: native image backend only supports images endpoints")
        if not ep or any(x not in allowed for x in ep):
            raise ValueError(f"{mid}: unsupported endpoint")
        health = item.get("health_path", "/health")
        if not health.startswith("/") or health.startswith("//") or "?" in health:
            raise ValueError("invalid health_path")
        start, stop, ttl = (
            float(item.get("startup_timeout_seconds", 600)),
            float(item.get("stop_timeout_seconds", 30)),
            float(item.get("idle_ttl_seconds", 10800)),
        )
        if min(start, stop, ttl) <= 0:
            raise ValueError("timeouts must be positive")
        models[mid] = Model(
            mid,
            tuple(cmd),
            item.get("base_url", "").rstrip("/"),
            health,
            ep,
            start,
            stop,
            ttl,
            item.get("cwd"),
            backend,
            str(Path(item["model_path"]).expanduser().resolve()) if native else None,
            bool(item.get("cpu_offload", False)),
        )
    if not models:
        raise ValueError("at least one model required")
    q, t = (
        int(raw.get("queue_limit", 1024)),
        float(raw.get("request_timeout_seconds", 3600)),
    )
    if min(q, t) <= 0:
        raise ValueError("limits must be positive")
    return Config(
        models, q, t, Path(raw.get("state_dir", "./state")).expanduser().resolve()
    )
