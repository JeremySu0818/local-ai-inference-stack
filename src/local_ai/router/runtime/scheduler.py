from __future__ import annotations
import asyncio
import collections
import enum
import logging
from dataclasses import dataclass
import httpx
from local_ai.backends import get_backend
from .backend_process import start_process, stop_process, wait_until_ready
from ..config import Config, Model
from .errors import BackendError, Busy
from .state_lock import acquire_state_lock, release_state_lock

log = logging.getLogger("router.scheduler")


class State(str, enum.Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    READY = "READY"
    DRAINING = "DRAINING"
    STOPPING = "STOPPING"
    FAILED = "FAILED"


@dataclass(eq=False)
class Waiter:
    model: str
    future: asyncio.Future
    load_hint: str | None = None


class Scheduler:
    def __init__(self, config: Config):
        self.config = config
        self.cv = asyncio.Condition()
        self.queue = collections.deque()
        self.state = State.STOPPED
        self.current = None
        self.proc = None
        self.engine = None
        self.active = 0
        self.dirty = False
        self.closing = False
        self.worker = None
        self.lockfile = None
        self.monitor = None
        self.http = httpx.AsyncClient(timeout=2, trust_env=False)

    async def open(self):
        self.lockfile = acquire_state_lock(self.config.state_dir)
        self.worker = asyncio.create_task(self._run(), name="scheduler")

    async def acquire(self, mid: str, load_hint: str | None = None):
        fut = asyncio.get_running_loop().create_future()
        w = Waiter(mid, fut, load_hint)
        async with self.cv:
            if self.closing:
                raise Busy("router is shutting down")
            if len(self.queue) >= self.config.queue_limit:
                raise Busy("request queue is full")
            self.queue.append(w)
            self.cv.notify_all()
        try:
            await asyncio.shield(fut)
        except asyncio.CancelledError:
            async with self.cv:
                if w in self.queue:
                    self.queue.remove(w)
                    fut.cancel()
                elif fut.done() and not fut.cancelled() and fut.exception() is None:
                    self.active -= 1
                self.cv.notify_all()
            raise
        return Lease(self, mid)

    async def release(self, failed=False):
        async with self.cv:
            assert self.active > 0
            self.active -= 1
            if failed:
                self.dirty = True
            self.cv.notify_all()

    async def _run(self):
        try:
            while True:
                action = None
                model = None
                load_hint = None
                async with self.cv:
                    while action is None:
                        if self.closing:
                            while self.queue:
                                w = self.queue.popleft()
                                if not w.future.done():
                                    w.future.set_exception(
                                        Busy("router is shutting down")
                                    )
                            if self.active:
                                await self.cv.wait()
                                continue
                            if self.current:
                                self.state = State.STOPPING
                                action = "stop"
                                break
                            return
                        if self.current and (
                            self.dirty
                            or (self.proc and self.proc.returncode is not None)
                        ):
                            self.state = State.DRAINING
                            if self.active:
                                await self.cv.wait()
                                continue
                            self.state = State.STOPPING
                            action = "stop"
                            break
                        if self.state == State.DRAINING and (
                            not self.queue or self.queue[0].model == self.current
                        ):
                            self.state = State.READY
                        if self.queue:
                            first = self.queue[0].model
                            if self.current != first:
                                if self.current:
                                    self.state = State.DRAINING
                                    if self.active:
                                        await self.cv.wait()
                                        continue
                                    self.state = State.STOPPING
                                    action = "stop"
                                    break
                                model = self.config.models[first]
                                load_hint = self.queue[0].load_hint
                                self.current = first
                                self.state = State.STARTING
                                action = "start"
                                break
                            if self.state == State.READY:
                                while (
                                    self.queue and self.queue[0].model == self.current
                                ):
                                    w = self.queue.popleft()
                                    if not w.future.done():
                                        self.active += 1
                                        w.future.set_result(None)
                                continue
                        await self.cv.wait()
                if action == "start":
                    try:
                        await self._start(model, load_hint)
                        async with self.cv:
                            if self.proc and self.proc.returncode is not None:
                                raise BackendError("backend exited during startup")
                            self.state = State.READY
                            self.cv.notify_all()
                    except Exception as e:
                        log.exception("backend startup failed: %s", model.id)
                        await self._stop()
                        async with self.cv:
                            self.current = None
                            self.state = State.FAILED
                            self.dirty = False
                            while self.queue and self.queue[0].model == model.id:
                                w = self.queue.popleft()
                                if not w.future.done():
                                    w.future.set_exception(BackendError(str(e)))
                            self.cv.notify_all()
                else:
                    await self._stop()
                    async with self.cv:
                        self.current = None
                        self.state = State.STOPPED
                        self.dirty = False
                        self.cv.notify_all()
        except BaseException:
            log.exception("scheduler stopped unexpectedly")
            async with self.cv:
                self.closing = True
                self.state = State.FAILED
                while self.queue:
                    w = self.queue.popleft()
                    if not w.future.done():
                        w.future.set_exception(BackendError("scheduler failed"))
                self.cv.notify_all()
            raise

    async def _start(self, m: Model, load_hint=None):
        spec = get_backend(m.backend)
        if spec.process is None:
            self.proc = await start_process(m)
            self.monitor = asyncio.create_task(self._watch(self.proc))
            await wait_until_ready(self.proc, m, self.http)
            return
        log.info("loading %s model: %s", spec.name, m.id)
        self.engine = spec.process(m)
        await asyncio.to_thread(self.engine.load, load_hint)

    async def _watch(self, proc):
        await proc.wait()
        async with self.cv:
            if self.proc is proc and self.state in (
                State.STARTING,
                State.READY,
                State.DRAINING,
            ):
                self.dirty = True
                self.cv.notify_all()

    async def _stop(self):
        if self.engine is not None:
            log.info("unloading %s model: %s", self.config.models[self.current].backend, self.current)
            await asyncio.to_thread(self.engine.unload)
            self.engine = None
        proc = self.proc
        if proc is None:
            return
        await stop_process(proc, self.config.models[self.current].stop_timeout)
        self.proc = None
        if self.monitor:
            await self.monitor
            self.monitor = None

    async def close(self):
        async with self.cv:
            self.closing = True
            self.cv.notify_all()
        await self.worker
        await self.http.aclose()
        release_state_lock(self.config.state_dir, self.lockfile)


class Lease:
    def __init__(self, scheduler, model):
        self.scheduler = scheduler
        self.model = model
        self.released = False

    async def finish(self, failed=False):
        if self.released:
            return
        self.released = True
        await self.scheduler.release(failed)
