"""Stream upstream responses while holding a scheduler lease."""
import asyncio
import json
import logging

from fastapi.responses import StreamingResponse

from ..runtime.errors import BackendError, Busy
from .responses import error

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


async def stream_inference(request, model, endpoint, body, native_input):
    mid = model.id
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    url = model.base_url + "/" + endpoint
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

