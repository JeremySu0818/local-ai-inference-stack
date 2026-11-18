"""Launch a configured llama.cpp model from the shared Hugging Face cache."""
import os
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

from local_ai.router.config import load_config

VERSION = "v1.0.0"
CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
SOURCE_DIR = CACHE_ROOT / "local-ai-inference-stack" / f"llama.cpp-{VERSION}"
SERVER = SOURCE_DIR / "build/bin/llama-server"


def cached_file(repo, filename):
    path = try_to_load_from_cache(repo, filename)
    return path if isinstance(path, str) and Path(path).is_file() else None


def serve(config_path, model_id):
    model = load_config(config_path).models[model_id]
    if model.backend != "llama_cpp":
        raise SystemExit(f"{model_id} is not a llama.cpp model")
    if not os.access(SERVER, os.X_OK):
        raise SystemExit("llama-server is missing. Run ./setup.sh")

    args = [str(SERVER)]
    for flag, filename in (("-m", model.gguf), ("--mmproj", model.mmproj)):
        if filename is not None:
            path = cached_file(model.hf_repo, filename)
            if path is None:
                raise SystemExit(f"Missing cached file {model.hf_repo}/{filename}. Run ./setup.sh")
            args.extend((flag, path))

    args.extend((
        "--alias", model.id, "--host", "127.0.0.1", "--port", "8000",
        "--ctx-size", str(model.context_size), "--parallel", str(model.parallel),
        "--kv-unified",
        "--n-gpu-layers", "all", "--fit", "off", "--flash-attn", "on",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja",
    ))
    os.execv(SERVER, args)
