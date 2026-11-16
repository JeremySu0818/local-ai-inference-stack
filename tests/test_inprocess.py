import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import yaml

from router.app import app_factory


class Lease:
    async def finish(self, failed=False):
        pass


class Scheduler:
    def __init__(self, config):
        import asyncio

        self.cv = asyncio.Condition()
        self.closing = False
        self.state, self.current, self.active, self.queue = "STOPPED", None, 0, []

    async def open(self):
        pass

    async def close(self):
        pass

    async def acquire(self, model):
        return Lease()


class RouterTest(unittest.IsolatedAsyncioTestCase):
    async def test_no_key_and_forwarding_for_both_models(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            models = [
                dict(
                    id=name,
                    command=["fake"],
                    base_url="http://127.0.0.1:8000/v1",
                    endpoints=[
                        "chat/completions",
                        "images/generations",
                        "images/edits",
                    ],
                )
                for name in ("text", "image")
            ]
            path.write_text(yaml.safe_dump({"models": models, "state_dir": temp}))
            with (
                patch.dict(os.environ, {"AI_ROUTER_CONFIG": str(path)}, clear=True),
                patch("router.app.Scheduler", Scheduler),
            ):
                app = app_factory()
                async with app.router.lifespan_context(app):
                    await app.state.upstream.aclose()

                    def upstream(request):
                        self.assertNotIn("authorization", request.headers)
                        return httpx.Response(200, json={"path": request.url.path})

                    app.state.upstream = httpx.AsyncClient(
                        transport=httpx.MockTransport(upstream)
                    )
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app), base_url="http://router"
                    ) as client:
                        self.assertEqual(
                            (await client.get("/v1/models")).status_code, 200
                        )
                        for name, endpoint in [
                            ("text", "chat/completions"),
                            ("image", "images/generations"),
                        ]:
                            result = await client.post(
                                "/v1/" + endpoint, json={"model": name}
                            )
                            self.assertEqual(result.status_code, 200)
                            self.assertEqual(result.json()["path"], "/v1/" + endpoint)
                        result = await client.post(
                            "/v1/images/edits",
                            data={"model": "image"},
                            files={"image": ("x.png", b"abc", "image/png")},
                        )
                        self.assertEqual(result.status_code, 200)
                        result = await client.post(
                            "/v1/images/generations",
                            json={"model": "image", "background": "pending"},
                        )
                        self.assertEqual(result.status_code, 400)
