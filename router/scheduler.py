from __future__ import annotations
import asyncio, collections, enum, fcntl, logging, os, signal, socket, time
from dataclasses import dataclass
from urllib.parse import urlsplit
import httpx
from .config import Config, Model

log = logging.getLogger("router.scheduler")


class State(str, enum.Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    READY = "READY"
    DRAINING = "DRAINING"
    STOPPING = "STOPPING"
    FAILED = "FAILED"


class RouterError(Exception):
    pass


class StartupBlocked(RouterError):
    pass


class Busy(RouterError):
    pass


class BackendError(RouterError):
    pass


@dataclass(eq=False)
class Waiter:
    model: str
    future: asyncio.Future


class Scheduler:
    def __init__(self, config: Config):
        self.config = config
        self.cv = asyncio.Condition()
        self.queue = collections.deque()
        self.state = State.STOPPED
        self.current = None
        self.proc = None
        self.native = None
        self.active = 0
        self.dirty = False
        self.closing = False
        self.worker = None
        self.lockfile = None
        self.monitor = None
        self.http = httpx.AsyncClient(timeout=2, trust_env=False)

    async def open(self):
        self.config.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lockfile = open(self.config.state_dir / "router.lock", "a+")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lockfile.close()
            raise StartupBlocked("已有 router 使用此狀態目錄，略過啟動")
        marker = self.config.state_dir / "unclean"
        if marker.exists():
            try:
                owner = int(marker.read_text().strip())
                if owner <= 0:
                    raise ValueError("invalid PID")
            except ValueError:
                owner = None
            if owner is not None:
                try:
                    os.kill(owner, 0)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    self.lockfile.close()
                    raise StartupBlocked(f"unclean 記錄的程序 {owner} 仍存活，略過啟動")
                else:
                    self.lockfile.close()
                    raise StartupBlocked(f"unclean 記錄的程序 {owner} 仍存活，略過啟動")
            log.warning("removing stale shutdown marker: %s (PID %s)", marker, owner)
            marker.unlink()
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"{os.getpid()}\n")
            f.flush()
            os.fsync(f.fileno())
        self.worker = asyncio.create_task(self._run(), name="scheduler")

    async def acquire(self, mid: str):
        fut = asyncio.get_running_loop().create_future()
        w = Waiter(mid, fut)
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
                        await self._start(model)
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

    async def _start(self, m: Model):
        if m.backend == "qwen_image":
            from .image_process import ImageProcess
            log.info("loading native image model: %s", m.id)
            self.native = ImageProcess(m)
            await asyncio.to_thread(self.native.load)
            return
        u = urlsplit(m.base_url)
        # Refuse to adopt an unrelated service already listening on the target port.
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(u.hostname, u.port), 1
            )
        except (OSError, asyncio.TimeoutError):
            pass
        else:
            writer.close()
            await writer.wait_closed()
            raise BackendError("backend port already in use")
        self.proc = await asyncio.create_subprocess_exec(
            *m.command, cwd=m.cwd, start_new_session=True
        )
        self.monitor = asyncio.create_task(self._watch(self.proc))
        deadline = time.monotonic() + m.startup_timeout
        while time.monotonic() < deadline:
            if self.proc.returncode is not None:
                raise BackendError(f"backend exited with status {self.proc.returncode}")
            try:
                r = await self.http.get(f"{u.scheme}://{u.netloc}{m.health_path}")
                if r.is_success:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)
        raise BackendError("backend readiness timed out")

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
        if self.native is not None:
            log.info("unloading native image model: %s", self.current)
            await asyncio.to_thread(self.native.unload)
            self.native = None
        proc = self.proc
        if proc is None:
            return
        pgid = proc.pid

        def send(sig):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass

        send(signal.SIGTERM)
        # Always kill the whole group after the grace period, including descendants
        # whose group leader has already exited.
        end = time.monotonic() + self.config.models[self.current].stop_timeout
        while time.monotonic() < end:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.05)
        send(signal.SIGKILL)
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except asyncio.TimeoutError:
            raise BackendError("failed to reap backend")
        # A surviving process group means unloading was not established; fail closed.
        deadline = time.monotonic() + 5
        while True:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() >= deadline:
                raise BackendError("backend process group survived SIGKILL")
            await asyncio.sleep(0.05)
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
        (self.config.state_dir / "unclean").unlink()
        fcntl.flock(self.lockfile, fcntl.LOCK_UN)
        self.lockfile.close()


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
