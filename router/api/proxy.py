"""Validate inference requests and dispatch them to the selected backend."""
import json

from fastapi import Request

from .responses import error
from .streaming import stream_inference


async def proxy(endpoint: str, request: Request):
    config = request.app.state.config
    body = bytearray()
    try:
        async for chunk in request.stream():
            body.extend(chunk)
    except Exception:
        return error("Request body interrupted", 400)
    ctype = request.headers.get("content-type", "")
    try:
        if ctype.lower().startswith("application/json"):
            data = json.loads(body)
            mid = data.get("model") if isinstance(data, dict) else None
            background = isinstance(data, dict) and (
                data.get("background") == "pending" or data.get("async") is True
            )
        elif ctype.lower().startswith("multipart/form-data"):
            from starlette.requests import Request as SRequest

            async def receive():
                return {
                    "type": "http.request",
                    "body": bytes(body),
                    "more_body": False,
                }

            clone = SRequest(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/",
                    "headers": [(b"content-type", ctype.encode())],
                },
                receive,
            )
            form = await clone.form(max_files=32, max_fields=128)
            mid = form.get("model")
            background = form.get("background") == "pending" or str(
                form.get("async", "")
            ).lower() in ("true", "1")
            await form.close()
        else:
            return error(
                "Content-Type must be application/json or multipart/form-data", 415
            )
    except Exception:
        return error("Malformed request body", 400)
    if background:
        return error("Background tasks are not supported by the router", 400)
    if not isinstance(mid, str) or mid not in config.models:
        return error("Unknown or missing model", 404)
    m = config.models[mid]
    if endpoint not in m.endpoints:
        return error("Endpoint not enabled for this model", 404)
    native_input = None
    if m.backend == "qwen_image":
        from backends.qwen_image.request import parse_request
        try:
            native_input = await parse_request(endpoint, bytes(body), ctype)
        except (ValueError, TypeError) as exc:
            return error(str(exc), 400)
    return await stream_inference(request, m, endpoint, body, native_input)
