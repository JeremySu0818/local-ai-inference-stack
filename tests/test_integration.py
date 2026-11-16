import asyncio, json, os, signal, socket, subprocess, sys, tempfile, time, unittest
from pathlib import Path
import httpx, yaml

ROOT = Path(__file__).resolve().parents[1]


def freeport():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Integration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.port = freeport()
        self.log = self.path / "events"
        models = []
        for n in ("A", "B"):
            port = freeport()
            models.append(
                {
                    "id": n,
                    "command": [
                        sys.executable,
                        str(ROOT / "tests/fake_backend.py"),
                        n,
                        str(port),
                        str(self.log),
                    ],
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "health_path": "/health",
                    "endpoints": [
                        "chat/completions",
                        "images/generations",
                        "images/edits",
                    ],
                    "startup_timeout_seconds": 5,
                    "stop_timeout_seconds": 1,
                    "idle_ttl_seconds": 0.5,
                }
            )
        self.models = models
        cfg = {
            "models": models,
            "state_dir": str(self.path / "state"),
            "request_timeout_seconds": 5,
            "queue_limit": 2000,
        }
        (self.path / "config.yaml").write_text(yaml.safe_dump(cfg))
        env = os.environ.copy()
        env.pop("AI_ROUTER_API_KEY", None)
        env.update(AI_ROUTER_CONFIG=str(self.path / "config.yaml"))
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "router.app:app_factory",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{self.port}",
            timeout=30,
            trust_env=False,
            limits=httpx.Limits(max_connections=400, max_keepalive_connections=200),
        )
        for _ in range(150):
            try:
                if (await self.client.get("/health")).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None:
                self.fail(self.proc.stderr.read().decode())
            await asyncio.sleep(0.05)
        else:
            self.fail("router did not start")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.proc.send_signal(signal.SIGTERM)
        try:
            await asyncio.to_thread(self.proc.wait, 10)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            await asyncio.to_thread(self.proc.wait)
        self.proc.stderr.close()
        self.tmp.cleanup()

    def events(self):
        return (
            [json.loads(x) for x in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )

    async def wait(self, predicate, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return
            await asyncio.sleep(0.02)
        self.fail("event deadline exceeded: " + repr(self.events()[-15:]))

    async def send(self, model, **data):
        return await self.client.post(
            "/v1/chat/completions", json={"model": model, **data}
        )

    async def test_fair_switch_stress_and_ttl(self):
        a = [
            asyncio.create_task(self.send("A", delay=0.08, tag="first"))
            for _ in range(100)
        ]
        await self.wait(lambda: any(x["event"] == "begin" for x in self.events()))
        b = [
            asyncio.create_task(self.send("B", delay=0.05, tag="second"))
            for _ in range(100)
        ]
        await asyncio.sleep(0.03)
        c = [
            asyncio.create_task(self.send("A", delay=0.03, tag="third"))
            for _ in range(100)
        ]
        results = await asyncio.gather(*a, *b, *c)
        self.assertTrue(
            all(x.status_code == 200 for x in results),
            [(x.status_code, x.text) for x in results if x.status_code != 200][:4],
        )
        self.assertEqual(
            [x.json()["model"] for x in results],
            ["A"] * 100 + ["B"] * 100 + ["A"] * 100,
        )
        await self.wait(
            lambda: len([x for x in self.events() if x["event"] == "stop"]) >= 3, 10
        )
        events = self.events()
        running = set()
        active = {}
        for e in events:
            m = e["model"]
            ev = e["event"]
            if ev == "start":
                self.assertFalse(running, (running, e))
                running.add(m)
            if ev == "begin":
                self.assertIn(m, running)
                active[m] = active.get(m, 0) + 1
            if ev == "end":
                active[m] -= 1
            if ev == "stop":
                self.assertEqual(active.get(m, 0), 0)
                running.remove(m)
        self.assertEqual(sum(e["event"] == "begin" for e in events), 300)

    async def test_disconnected_image_and_stream_drain(self):
        for ep, payload, tag in [
            ("images/generations", {"model": "A", "delay": 0.6, "tag": "img"}, "img"),
            (
                "chat/completions",
                {"model": "A", "delay": 0.6, "stream": True, "tag": "sse"},
                "sse",
            ),
        ]:
            task = asyncio.create_task(self.client.post("/v1/" + ep, json=payload))
            await self.wait(
                lambda: any(
                    e["event"] == "begin" and e.get("tag") == tag for e in self.events()
                )
            )
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            b = asyncio.create_task(self.send("B", delay=0.05))
            await asyncio.sleep(0.15)
            self.assertFalse(b.done())
            self.assertEqual((await b).status_code, 200)
            ev = self.events()
            end = next(
                x["time"] for x in ev if x["event"] == "end" and x.get("tag") == tag
            )
            self.assertTrue(
                any(
                    x["event"] == "start" and x["model"] == "B" and x["time"] > end
                    for x in ev
                ),
                ev,
            )

    async def test_multipart_and_errors(self):
        r = await self.client.post(
            "/v1/images/edits",
            data={"model": "A", "delay": ".02"},
            files={"image": ("x.png", b"abc", "image/png")},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(
            (await self.client.get("/v1/models")).json()["data"][0]["id"], "A"
        )
        self.assertEqual(
            (
                await self.client.post(
                    "/v1/chat/completions", json={"model": "unknown"}
                )
            ).status_code,
            404,
        )
        self.assertEqual(
            (
                await self.client.get(
                    "/v1/models", headers={"Authorization": "Bearer ignored"}
                )
            ).status_code,
            200,
        )

    async def test_backend_crash_recovers(self):
        self.assertEqual((await self.send("A")).status_code, 200)
        try:
            await self.client.post(
                self.models[0]["base_url"].removesuffix("/v1") + "/crash"
            )
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.2)
        self.assertEqual((await self.send("B")).status_code, 200)

    async def test_stream_disconnect_after_first_event(self):
        async with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "A", "delay": 1, "stream": True, "tag": "live"},
        ) as response:
            self.assertEqual(response.status_code, 200)
            lines = response.aiter_lines()
            self.assertTrue((await anext(lines)).startswith("data:"))
            next_model = asyncio.create_task(self.send("B", delay=0.03))
            await asyncio.sleep(0.15)
            self.assertFalse(next_model.done())
        self.assertEqual((await next_model).status_code, 200)
        events = self.events()
        finish = next(
            e["time"] for e in events if e["event"] == "end" and e.get("tag") == "live"
        )
        self.assertTrue(
            any(
                e["event"] == "start" and e["model"] == "B" and e["time"] > finish
                for e in events
            )
        )

    async def test_port_in_use_fails_closed(self):
        # Occupying a configured backend port must never be mistaken for readiness.
        host = self.models[0]["base_url"].split(":")[2].split("/")[0]
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", int(host)))
            listener.listen()
            result = await self.send("A")
            self.assertEqual(result.status_code, 503, result.text)
        self.assertEqual((await self.send("B")).status_code, 200)

    async def test_cancel_queued_switch_resumes_current_model(self):
        first = asyncio.create_task(self.send("A", delay=1.5, tag="slow"))
        await self.wait(
            lambda: any(
                e["event"] == "begin" and e.get("tag") == "slow" for e in self.events()
            )
        )
        switch = asyncio.create_task(self.send("B", delay=0.05))
        for _ in range(100):
            if (await self.client.get("/health")).json()["queued"] >= 1:
                break
            await asyncio.sleep(0.02)
        else:
            self.fail("switch was not queued")
        switch.cancel()
        await asyncio.gather(switch, return_exceptions=True)
        for _ in range(100):
            if (await self.client.get("/health")).json()["queued"] == 0:
                break
            await asyncio.sleep(0.02)
        else:
            self.fail("cancelled switch was not removed")
        other = asyncio.create_task(self.send("A", delay=0.05, tag="resumed"))
        self.assertEqual((await asyncio.wait_for(other, 1)).status_code, 200)
        self.assertFalse(first.done())
        self.assertEqual((await first).status_code, 200)
        self.assertFalse(
            any(e["event"] == "start" and e["model"] == "B" for e in self.events())
        )
