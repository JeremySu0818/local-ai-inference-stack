"""Stream upstream responses while holding a scheduler lease."""
import asyncio
import json
import logging
import threading

from fastapi.responses import StreamingResponse

from local_ai.backends import get_backend
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


class InferenceResponse(StreamingResponse):
    def __init__(self, *args, watcher, abandon, **kwargs):
        super().__init__(*args, **kwargs)
        self.watcher = watcher
        self.abandon = abandon

    async def __call__(self, scope, receive, send):
        delivery = asyncio.create_task(self.stream_response(send))
        try:
            done, _ = await asyncio.wait(
                (delivery, self.watcher), return_when=asyncio.FIRST_COMPLETED
            )
            if delivery in done:
                await delivery
        except OSError:
            pass
        finally:
            if not delivery.done() or delivery.cancelled() or delivery.exception():
                self.abandon()
            delivery.cancel()
            self.watcher.cancel()
            await asyncio.gather(delivery, self.watcher, return_exceptions=True)


async def stream_inference(request, model, endpoint, body, parsed):
    mid = model.id
    spec = get_backend(model.backend)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    url = model.base_url + "/" + endpoint
    if request.url.query:
        url += "?" + request.url.query
    ready = asyncio.get_running_loop().create_future()
    q = asyncio.Queue(maxsize=8)
    abandoned = asyncio.Event()
    consumed = asyncio.Event()
    admitted = asyncio.Event()
    engine_cancel = threading.Event()
    producing = True
    transform = (
        spec.transform if spec.transform is not None and parsed is not None else None
    )
    ready.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)

    async def put(item):
        if abandoned.is_set():
            return
        a = asyncio.create_task(q.put(item))
        b = asyncio.create_task(abandoned.wait())
        try:
            await asyncio.wait((a, b), return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (a, b):
                t.cancel()
            await asyncio.gather(a, b, return_exceptions=True)

    async def producer():
        nonlocal producing
        lease = None
        failed = False
        try:
            load_hint = spec.load_hint(parsed) if spec.load_hint is not None else None
            lease = await request.app.state.scheduler.acquire(mid, load_hint)
            admitted.set()
            if spec.native:
                result = await asyncio.to_thread(
                    request.app.state.scheduler.engine.generate, *parsed, engine_cancel
                )
                if abandoned.is_set():
                    return
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
                if transform is None or resp.status_code >= 400:
                    if not ready.done():
                        ready.set_result((resp.status_code, response_headers))
                    async for chunk in resp.aiter_bytes():
                        await put(chunk)
                    return
                payload = bytearray()
                async for chunk in resp.aiter_bytes():
                    payload.extend(chunk)
                rewritten = await asyncio.to_thread(transform, bytes(payload), parsed)
                if not ready.done():
                    ready.set_result((resp.status_code, response_headers))
                await put(rewritten)
        except asyncio.CancelledError:
            failed = lease is not None and not abandoned.is_set()
            if not ready.done():
                ready.cancel()
            raise
        except Exception as e:
            failed = lease is not None
            log.warning("upstream %s failed: %s", mid, e)
            if not ready.done():
                ready.set_exception(e)
        finally:
            producing = False
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

    def abandon():
        if abandoned.is_set():
            return
        abandoned.set()
        engine_cancel.set()
        if producing and (not spec.native or not admitted.is_set()):
            task.cancel()

    async def watch_disconnect():
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                abandon()
                return

    watcher = asyncio.create_task(watch_disconnect())
    handoff = False
    try:
        done, _ = await asyncio.wait(
            (ready, watcher), return_when=asyncio.FIRST_COMPLETED
        )
        if watcher in done:
            abandon()
            return error("Client disconnected", 499)
        status, rheaders = await ready
        handoff = True
    except asyncio.CancelledError:
        abandon()
        raise
    except Busy as e:
        return error(e, 503, "server_error")
    except BackendError as e:
        return error(e, 503, "server_error")
    except Exception as e:
        return error("Backend request failed: " + str(e), 502, "server_error")
    finally:
        if not handoff:
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

    return InferenceResponse(
        stream(), status_code=status, headers=rheaders, watcher=watcher, abandon=abandon
    )

