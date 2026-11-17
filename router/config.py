from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
from urllib.parse import urlsplit
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    cpu_offload: bool = False
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
        if backend not in ("http", "qwen_image", "llama_cpp"):
            raise ValueError(f"{mid}: unsupported backend")
        native = backend == "qwen_image"
        if native and not item.get("model_path"):
            raise ValueError(f"{mid}: model_path is required")
        if not isinstance(mid, str) or not mid or mid in models:
            raise ValueError("empty or duplicate model id")
        llama = backend == "llama_cpp"
        model_path = item.get("model_path") if native else None
        if model_path and (not item.get("model_revision") or model_path.startswith(("/", ".", "~"))):
            model_path = str(Path(model_path).expanduser().resolve())
        base_url = "http://127.0.0.1:8000/v1" if llama else item.get("base_url", "")
        u = urlsplit(base_url)
        parallel = item.get("parallel", 1) if llama else 1
        hf_repo = item.get("hf_repo", mid) if llama else None
        if llama:
            if not isinstance(hf_repo, str) or hf_repo.count("/") != 1 or not all(hf_repo.split("/")):
                raise ValueError(f"{mid}: hf_repo must be a Hugging Face owner/repository ID")
            if type(parallel) is not int or not 1 <= parallel <= 256:
                raise ValueError(f"{mid}: parallel must be an integer from 1 to 256")
            if not isinstance(item.get("gguf"), str) or not item["gguf"].endswith(".gguf"):
                raise ValueError(f"{mid}: gguf filename is required")
            if item.get("mmproj") is not None and (
                not isinstance(item["mmproj"], str) or not item["mmproj"].endswith(".gguf")
            ):
                raise ValueError(f"{mid}: invalid mmproj filename")
            if type(item.get("context_size")) is not int or item["context_size"] <= 0:
                raise ValueError(f"{mid}: context_size must be a positive integer")
            cmd = [sys.executable, "-m", "backends.llama_cpp", "serve", str(Path(path).resolve()), mid]
        else:
            cmd = item.get("command", [])
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
        start, stop = (
            float(item.get("startup_timeout_seconds", 600)),
            float(item.get("stop_timeout_seconds", 30)),
        )
        if min(start, stop) <= 0:
            raise ValueError("timeouts must be positive")
        models[mid] = Model(
            id=mid,
            command=tuple(cmd),
            base_url=base_url.rstrip("/"),
            health_path=health,
            endpoints=ep,
            startup_timeout=start,
            stop_timeout=stop,
            cwd=None if llama else item.get("cwd"),
            backend=backend,
            model_path=model_path,
            cpu_offload=bool(item.get("cpu_offload", False)),
            model_revision=item.get("model_revision"),
            hf_repo=hf_repo,
            gguf=item.get("gguf"),
            mmproj=item.get("mmproj"),
            context_size=item.get("context_size"),
            parallel=parallel,
        )
    if not models:
        raise ValueError("at least one model required")
    with ROUTER_CONFIG.open(encoding="utf-8") as f:
        runtime = yaml.safe_load(f)
    if not isinstance(runtime, dict):
        raise ValueError("router configuration must be a mapping")

    q = int(raw.get("queue_limit", runtime["queue_limit"]))
    t = float(raw.get("request_timeout_seconds", runtime["request_timeout_seconds"]))
    if min(q, t) <= 0:
        raise ValueError("limits must be positive")
    state_dir = (
        Path(raw["state_dir"]).expanduser() if "state_dir" in raw
        else PROJECT_ROOT / Path(runtime["state_dir"]).expanduser()
    )
    return Config(models, q, t, state_dir.resolve())
