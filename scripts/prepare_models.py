"""Populate the shared Hugging Face cache for the configured backends."""
from pathlib import Path
import shlex
import sys

from huggingface_hub import hf_hub_download, snapshot_download

from backends.llama_cpp.server import cached_file
from backends.qwen_image.size import MODEL_ID, REVISION
from router.config import load_config


def download(repo, filenames=(), revision=None):
    try:
        if filenames:
            for filename in filenames:
                if cached_file(repo, filename) is None:
                    hf_hub_download(repo, filename)
        else:
            snapshot_download(repo, revision=revision)
    except Exception as error:
        args = [".venv/bin/hf", "download", repo, *filenames]
        if revision:
            args.extend(("--revision", revision))
        raise SystemExit(
            f"Download failed for {repo}: {error}\n"
            f"To download manually, run {shlex.join(args)} and retry ./setup.sh."
        ) from error


def prepare(config_path):
    for model in load_config(config_path).models.values():
        if model.backend == "llama_cpp":
            download(model.hf_repo, tuple(file for file in (model.gguf, model.mmproj) if file))
        elif model.backend == "qwen_image":
            if model.model_revision and not Path(model.model_path).is_absolute():
                download(model.model_path, revision=model.model_revision)
            download(MODEL_ID, revision=REVISION)


if __name__ == "__main__":
    prepare(sys.argv[1])
