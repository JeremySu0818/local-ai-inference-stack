from __future__ import annotations

import base64
import binascii
import io
import json

from .options import ImageOptions


async def parse_request(endpoint, body, content_type):
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
        options.set_aspect_ratio(images[-1].width / images[-1].height)
    return options, images
