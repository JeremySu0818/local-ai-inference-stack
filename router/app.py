from __future__ import annotations
import asyncio, logging, os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from .config import load_config
from .scheduler import Scheduler, Busy, BackendError, StartupBlocked

log = logging.getLogger("router")
HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "authorization",
    "accept-encoding",
}
END = object()


def error(message, status, kind="invalid_request_error"):
    return JSONResponse(
        {"error": {"message": str(message), "type": kind, "param": None, "code": None}},
        status_code=status,
    )


def app_factory():
    config = load_config(os.environ.get("AI_ROUTER_CONFIG", "config.yaml"))

    @asynccontextmanager
    async def lifespan(app):
        scheduler = app.state.scheduler
        app.state.scheduler = scheduler
        app.state.jobs = set()
        app.state.upstream = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout, connect=10),
            trust_env=False,
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=256),
        )
        yield
        # Stop admitting requests, then wait for detached producers to finish.
        async with scheduler.cv:
            scheduler.closing = True
            scheduler.cv.notify_all()
        if app.state.jobs:
            await asyncio.gather(*list(app.state.jobs), return_exceptions=True)
        await scheduler.close()
        await app.state.upstream.aclose()

    class RouterApp(FastAPI):
        async def __call__(self, scope, receive, send):
            if scope["type"] == "lifespan":
                scheduler = Scheduler(config)
                try:
                    await scheduler.open()
                except StartupBlocked as exc:
                    await scheduler.http.aclose()
                    await receive()  # lifespan.startup
                    await send({"type": "lifespan.startup.failed", "message": str(exc)})
                    return
                self.state.scheduler = scheduler
            await super().__call__(scope, receive, send)

    app = RouterApp(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health(request: Request):
        s = request.app.state.scheduler
        return {
            "state": s.state,
            "model": s.current,
            "active": s.active,
            "queued": len(s.queue),
            "accepting": not s.closing,
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {"id": m.id, "object": "model", "created": 0, "owned_by": "local"}
                for m in config.models.values()
            ],
        }

    @app.get("/v1/models/{model_id:path}")
    async def model_detail(model_id: str):
        if model_id not in config.models:
            return error("Model not found", 404)
        return {"id": model_id, "object": "model", "created": 0, "owned_by": "local"}

    @app.post("/v1/{endpoint:path}")
    async def proxy(endpoint: str, request: Request):
        import json

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
                # Parse a synthetic request so the exact original multipart bytes are forwarded.
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
            from .qwen_image import parse_request
            try:
                native_input = await parse_request(endpoint, bytes(body), ctype)
            except (ValueError, TypeError) as exc:
                return error(str(exc), 400)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        # Preserve the query string; endpoint itself is restricted to the allowlist.
        url = m.base_url + "/" + endpoint
        if request.url.query:
            url += "?" + request.url.query
        ready = asyncio.get_running_loop().create_future()
        q = asyncio.Queue(maxsize=8)
        abandoned = asyncio.Event()
        consumed = asyncio.Event()
        admitted = asyncio.Event()
        ready.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)

        async def put(item):
            if abandoned.is_set():
                return
            a = asyncio.create_task(q.put(item))
            b = asyncio.create_task(abandoned.wait())
            done, _ = await asyncio.wait((a, b), return_when=asyncio.FIRST_COMPLETED)
            for t in (a, b):
                if t not in done:
                    t.cancel()
            await asyncio.gather(a, b, return_exceptions=True)

        async def producer():
            lease = None
            failed = False
            try:
                lease = await request.app.state.scheduler.acquire(mid)
                admitted.set()
                if native_input is not None:
                    result = await asyncio.to_thread(
                        request.app.state.scheduler.native.generate, *native_input
                    )
                    ready.set_result((200, {"content-type": "application/json"}))
                    await put(json.dumps(result).encode())
                    return
                async with request.app.state.upstream.stream(
                    "POST", url, content=bytes(body), headers=headers
                ) as resp:
                    response_headers = {
                        k: v
                        for k, v in resp.headers.items()
                        if k.lower() not in HOP
                        and k.lower() not in ("content-encoding",)
                    }
                    if not ready.done():
                        ready.set_result((resp.status_code, response_headers))
                    async for chunk in resp.aiter_bytes():
                        await put(chunk)
            except asyncio.CancelledError:
                failed = lease is not None
                if not ready.done():
                    ready.cancel()
                raise
            except Exception as e:
                failed = lease is not None
                log.warning("upstream %s failed: %s", mid, e)
                if not ready.done():
                    ready.set_exception(e)
            finally:
                await put(END)
                # Keep the lease through downstream delivery, or until disconnect.
                if (
                    lease
                    and ready.done()
                    and not ready.cancelled()
                    and ready.exception() is None
                ):
                    a = asyncio.create_task(consumed.wait())
                    b = asyncio.create_task(abandoned.wait())
                    done, _ = await asyncio.wait(
                        (a, b), return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in (a, b):
                        if t not in done:
                            t.cancel()
                    await asyncio.gather(a, b, return_exceptions=True)
                if lease:
                    await asyncio.shield(lease.finish(failed))

        task = asyncio.create_task(producer())
        request.app.state.jobs.add(task)
        task.add_done_callback(request.app.state.jobs.discard)

        # Before response headers, notice a disconnected client while waiting in the queue.
        async def watch_disconnect():
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    return

        watcher = asyncio.create_task(watch_disconnect())
        try:
            done, _ = await asyncio.wait(
                (ready, watcher), return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done and not ready.done():
                abandoned.set()
                # Queued requests can be removed; admitted requests must drain.
                if not admitted.is_set():
                    task.cancel()
                return error("Client disconnected", 499)
            status, rheaders = await ready
        except asyncio.CancelledError:
            abandoned.set()
            if not admitted.is_set():
                task.cancel()
            raise
        except Busy as e:
            return error(e, 503, "server_error")
        except BackendError as e:
            return error(e, 503, "server_error")
        except Exception as e:
            return error("Backend request failed: " + str(e), 502, "server_error")
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

        async def stream():
            try:
                while True:
                    item = await q.get()
                    if item is END:
                        break
                    yield item
            finally:
                consumed.set()
                abandoned.set()

        return StreamingResponse(stream(), status_code=status, headers=rheaders)

    return app
