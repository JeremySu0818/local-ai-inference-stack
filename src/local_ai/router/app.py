from __future__ import annotations
import asyncio
import logging
import os
import time
from uuid import uuid4
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request

from .config import load_config
from .runtime.errors import StartupBlocked
from .api.proxy import proxy
from .api.responses import error
from .runtime.scheduler import Scheduler

def app_factory():
    config = load_config(os.environ.get("AI_ROUTER_CONFIG", "config/models.yaml"))

    access_log = logging.Logger("router.access", logging.INFO)

    @asynccontextmanager
    async def lifespan(app):
        scheduler = app.state.scheduler
        app.state.jobs = set()
        app.state.upstream = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout, connect=10),
            trust_env=False,
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=256),
        )
        access_path = config.state_dir / "access.log"
        for suffix in (".1", ".2", ".3"):
            access_path.with_name(access_path.name + suffix).unlink(missing_ok=True)
        handler = logging.FileHandler(access_path, mode="w", encoding="utf-8")
        console = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(message)s")
        for output in (handler, console):
            output.setFormatter(formatter)
            access_log.addHandler(output)
        try:
            yield
        finally:
            try:
                async with scheduler.cv:
                    scheduler.closing = True
                    scheduler.cv.notify_all()
                if app.state.jobs:
                    await asyncio.gather(*list(app.state.jobs), return_exceptions=True)
                await scheduler.close()
                await app.state.upstream.aclose()
            finally:
                for output in (handler, console):
                    access_log.removeHandler(output)
                    output.close()

    class RouterApp(FastAPI):
        async def __call__(self, scope, receive, send):
            if scope["type"] == "lifespan":
                scheduler = Scheduler(config)
                try:
                    await scheduler.open()
                except StartupBlocked as exc:
                    await scheduler.http.aclose()
                    await receive()
                    await send({"type": "lifespan.startup.failed", "message": str(exc)})
                    return
                self.state.scheduler = scheduler
            if scope["type"] != "http":
                await super().__call__(scope, receive, send)
                return

            started = time.perf_counter()
            status = 500
            request_state = scope.setdefault("state", {})
            request_id = request_state["request_id"] = uuid4().hex

            async def record_status(message):
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]
                    headers = [
                        (key, value) for key, value in message.get("headers", [])
                        if key.lower() != b"x-request-id"
                    ]
                    message = {**message, "headers": [
                        *headers, (b"x-request-id", request_id.encode("ascii")),
                    ]}
                await send(message)

            try:
                await super().__call__(scope, receive, record_status)
            except asyncio.CancelledError:
                request_state.setdefault("outcome", "cancelled")
                raise
            except Exception:
                request_state.setdefault("outcome", "router_error")
                logging.getLogger("uvicorn.error").error("Request failed request_id=%s", request_id)
                raise
            finally:
                client = scope.get("client")
                access_log.info(
                    "source=%r method=%r path=%r status=%d duration_ms=%.1f request_id=%s model=%r outcome=%s",
                    client[0] if client else "unknown",
                    scope["method"],
                    scope["path"],
                    status,
                    (time.perf_counter() - started) * 1000,
                    request_id,
                    request_state.get("model"),
                    request_state.get("outcome") or (
                        "server_error" if status >= 500 else
                        "client_error" if status >= 400 else "completed"
                    ),
                )

    app = RouterApp(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.config = config

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

    app.add_api_route("/v1/{endpoint:path}", proxy, methods=["POST"])
    return app
