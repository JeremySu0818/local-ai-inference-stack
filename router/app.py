from __future__ import annotations
import asyncio
import os
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

    @asynccontextmanager
    async def lifespan(app):
        scheduler = app.state.scheduler
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
