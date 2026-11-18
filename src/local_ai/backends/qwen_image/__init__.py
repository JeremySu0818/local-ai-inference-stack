"""Qwen Image backend."""
from __future__ import annotations

from pathlib import Path

from ..base import Backend, download


def configure(item: dict) -> dict:
    """Validate one Qwen Image entry; its other keys are the router's own."""
    if not item.get("model_path"):
        raise ValueError(f"{item['id']}: model_path is required")
    return {}


def process(model):
    """The worker that holds the model, one per loaded model."""
    from .runtime.process import ImageProcess

    return ImageProcess(model)


async def parse(_model, endpoint: str, body: bytes, content_type: str):
    """Turn an images request into the arguments of ``generate``."""
    from .api.request import parse_request

    return await parse_request(endpoint, body, content_type)


def load_hint(parsed) -> str | None:
    """The prompt to choose a size from, when the request left the size to the model."""
    options, _ = parsed
    return options.prompt if options.width is None else None


def prepare(model) -> None:
    """Download the checkpoint and the small model that picks image sizes."""
    from .inference.size import MODEL_ID, REVISION

    if model.model_revision and not Path(model.model_path).is_absolute():
        download(model.model_path, revision=model.model_revision)
    download(MODEL_ID, revision=REVISION)


BACKEND = Backend(
    name="qwen_image",
    base_url="",
    endpoints=("images/generations", "images/edits"),
    local_model=True,
    configure=configure,
    process=process,
    parse=parse,
    load_hint=load_hint,
    prepare=prepare,
)
