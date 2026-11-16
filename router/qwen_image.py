"""Native, offline Qwen Image 2.1 inference. Heavy imports stay lazy."""
from __future__ import annotations

import base64
import binascii
import gc
import io
import json
import math
import secrets
import threading
import time

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str = Field(min_length=1, max_length=32000)
    negative_prompt: str | None = None
    width: int | None = Field(default=None, ge=256, le=4096)
    height: int | None = Field(default=None, ge=256, le=4096)
    size: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    seed: int = Field(default_factory=lambda: secrets.randbits(32), ge=0, le=2**63-1)
    num_inference_steps: int = Field(default=40, ge=1, le=100)
    true_cfg_scale: float = Field(default=1.0, ge=1, le=20, allow_inf_nan=False)
    use_kv_cache: bool = True
    output_resolution: int = Field(default=1024, ge=256, le=2048)
    response_format: str = "b64_json"
    images: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_options(self):
        if not self.prompt.strip():
            raise ValueError("prompt cannot be blank")
        if self.size and self.size != "auto":
            try:
                self.width, self.height = map(int, self.size.lower().split("x"))
            except ValueError:
                raise ValueError("size must be WIDTHxHEIGHT")
        if (self.width is None) != (self.height is None):
            raise ValueError("provide both width and height, or omit both for automatic sizing")
        if self.width is None:
            return self
        if any(v < 256 or v > 4096 or v % 32 for v in (self.width, self.height)):
            raise ValueError("dimensions must be multiples of 32 between 256 and 4096")
        if self.width * self.height > 4096 * 2048:
            raise ValueError("maximum output area is 8388608 pixels")
        if self.response_format != "b64_json":
            raise ValueError("only response_format=b64_json is supported")
        return self


async def parse_request(endpoint, body, content_type):
    """Decode bounded uploads before acquiring GPU resources; never fetch remote URLs."""
    from PIL import Image, UnidentifiedImageError
    from starlette.requests import Request
    from starlette.datastructures import UploadFile

    uploads = []
    if content_type.lower().startswith("application/json"):
        data = json.loads(body)
        options = ImageOptions.model_validate(data)
        for value in options.images:
            if value.startswith("data:"):
                prefix, value = value.split(",", 1)
                if not prefix.startswith("data:image/") or not prefix.endswith(";base64"):
                    raise ValueError("image data URLs must be base64 encoded images")
            try:
                uploads.append(base64.b64decode(value, validate=True))
            except (ValueError, binascii.Error):
                raise ValueError("images must contain base64 image data, not URLs or paths")
    else:
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        req = Request({"type": "http", "method": "POST", "path": "/",
                       "headers": [(b"content-type", content_type.encode())]}, receive)
        async with req.form(max_files=10, max_fields=32) as form:
            data = {}
            for key, value in form.multi_items():
                if isinstance(value, UploadFile):
                    if key not in ("image", "image[]", "images"):
                        raise ValueError(f"unsupported upload field: {key}")
                    uploads.append(await value.read())
                else:
                    if key in data:
                        raise ValueError(f"duplicate field: {key}")
                    data[key] = value
            options = ImageOptions.model_validate(data)
            if options.images:
                raise ValueError("multipart images must be uploaded files")
    if len(uploads) > 10:
        raise ValueError("at most 10 reference images are supported")
    if endpoint == "images/edits" and not uploads:
        raise ValueError("image editing requires at least one image")
    if endpoint == "images/generations" and uploads:
        raise ValueError("use images/edits for reference images")
    images = []
    for raw in uploads:
        try:
            with Image.open(io.BytesIO(raw)) as img:
                if img.width * img.height > 16777216:
                    raise ValueError("reference image exceeds 16 megapixels")
                img.load()
                images.append(img.convert("RGBA" if "A" in img.getbands() else "RGB"))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("invalid reference image") from exc
    if options.width is None and images:
        # Match the pipeline's automatic sizing, while retaining the API size limits.
        ratio = images[-1].width / images[-1].height if images else 1.0
        options.width = round(options.output_resolution * math.sqrt(ratio) / 32) * 32
        options.height = round(options.output_resolution / math.sqrt(ratio) / 32) * 32
        options.validate_options()
    return options, images


class QwenImageBackend:
    def __init__(self, model):
        self.model = model
        self.pipe = None
        self.lock = threading.Lock()
        from .image_size import SizeSelector
        self.size_selector = SizeSelector()

    def load(self):
        import torch
        from pathlib import Path
        from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen Image requires an available CUDA GPU/driver")
        path = Path(self.model.model_path)
        if not (path / "model_index.json").exists():
            raise RuntimeError("Native model is not prepared; run scripts/prepare_qwen_image.py")
        checkpoint = path.parent / "qwen_image_2.1_bf16.safetensors"
        kwargs = {}
        if checkpoint.exists():
            kwargs["transformer"] = QwenImage21Transformer2DModel.from_single_file(
                str(checkpoint), config=str(path), subfolder="transformer",
                dtype=torch.bfloat16, local_files_only=True,
            )
        self.pipe = QwenImage21Pipeline.from_pretrained(
            str(path), dtype=torch.bfloat16, local_files_only=True, **kwargs,
        )
        self.pipe.vae.enable_tiling()
        if self.model.cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")

    def generate(self, options, images):
        import torch

        # The scheduler may admit multiple same-model requests; pipelines are mutable.
        with self.lock, torch.inference_mode():
            if options.width is None:
                ratio = self.size_selector.choose(options.prompt)
                options.width = round(options.output_resolution * math.sqrt(ratio) / 32) * 32
                options.height = round(options.output_resolution / math.sqrt(ratio) / 32) * 32
                options.validate_options()
            result = self.pipe(
                prompt=options.prompt, image=images or None,
                negative_prompt=options.negative_prompt,
                true_cfg_scale=options.true_cfg_scale,
                width=options.width, height=options.height,
                num_inference_steps=options.num_inference_steps,
                num_images_per_prompt=options.n,
                output_resolution=options.output_resolution,
                use_kv_cache=options.use_kv_cache,
                generator=torch.Generator("cpu").manual_seed(options.seed),
            )
            data = []
            for img in result.images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                data.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})
            return {"created": int(time.time()), "seed": options.seed,
                    "size": f"{options.width}x{options.height}", "data": data}

    def unload(self):
        import torch

        self.pipe = None
        self.size_selector.model = None
        self.size_selector.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
