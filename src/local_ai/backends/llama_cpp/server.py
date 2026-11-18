"""Launch a configured llama.cpp model from the shared Hugging Face cache."""
import os

from local_ai.router.config import load_config

from ..base import cached_file
from . import server_path


def server_args(model) -> list[str]:
    """The llama-server command line for a model entry.

    Embedding models need more than the ``--embeddings`` flag: their attention
    is bidirectional, so a prompt longer than the ubatch fails outright.
    With unified KV, one slot can use the full configured context.
    """
    args = [str(server_path())]
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
    if "embeddings" in model.endpoints:
        context = str(model.context_size)
        args.extend(("--embeddings", "--batch-size", context, "--ubatch-size", context))
    args.extend(model.engine_args)
    return args


def serve(config_path, model_id):
    model = load_config(config_path).models[model_id]
    if model.backend != "llama_cpp":
        raise SystemExit(f"{model_id} is not a llama.cpp model")
    server = server_path()
    if not os.access(server, os.X_OK):
        raise SystemExit("llama-server is missing. Run ./setup.sh")
    os.execv(server, server_args(model))
