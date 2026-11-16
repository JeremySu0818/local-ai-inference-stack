import asyncio, json, os, sys, time
from aiohttp import web

name, port, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]


def record(event, **kw):
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    os.write(
        fd,
        (
            json.dumps({"model": name, "event": event, "time": time.monotonic(), **kw})
            + "\n"
        ).encode(),
    )
    os.close(fd)


async def health(req):
    return web.Response(text="ok")


async def crash(req):
    os._exit(7)


async def run(req):
    data = (
        await req.post()
        if req.content_type.startswith("multipart/")
        else await req.json()
    )
    delay = float(data.get("delay", 0.03))
    tag = data.get("tag", "")
    record("begin", tag=tag)
    try:
        if data.get("stream"):
            r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await r.prepare(req)
            for _ in range(5):
                await asyncio.sleep(delay / 5)
                await r.write(b'data: {"ok":true}\n\n')
            await r.write(b"data: [DONE]\n\n")
            await r.write_eof()
            return r
        await asyncio.sleep(delay)
        return web.json_response({"model": name, "data": [{"b64_json": "aW1hZ2U="}]})
    finally:
        record("end", tag=tag)


app = web.Application()
app.router.add_get("/health", health)
app.router.add_post("/crash", crash)
app.router.add_post("/v1/{tail:.*}", run)
record("start")
try:
    web.run_app(app, host="127.0.0.1", port=port, print=None, shutdown_timeout=1)
finally:
    record("stop")
