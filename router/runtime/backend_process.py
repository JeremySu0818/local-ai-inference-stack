"""Check, start and stop HTTP backend processes."""
import asyncio
import os
import signal
import time
from urllib.parse import urlsplit

import httpx

from .errors import BackendError


async def start_process(model):
    u = urlsplit(model.base_url)
    # Refuse to adopt an unrelated service already listening on the target port.
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(u.hostname, u.port), 1
        )
    except (OSError, asyncio.TimeoutError):
        pass
    else:
        writer.close()
        await writer.wait_closed()
        raise BackendError("backend port already in use")
    return await asyncio.create_subprocess_exec(
        *model.command, cwd=model.cwd, start_new_session=True
    )


async def wait_until_ready(proc, model, http):
    u = urlsplit(model.base_url)
    deadline = time.monotonic() + model.startup_timeout
    while time.monotonic() < deadline:
        if proc.returncode is not None:
            raise BackendError(f"backend exited with status {proc.returncode}")
        try:
            r = await http.get(f"{u.scheme}://{u.netloc}{model.health_path}")
            if r.is_success:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.25)
    raise BackendError("backend readiness timed out")


async def stop_process(proc, timeout):
    pgid = proc.pid

    def send(sig):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    # Always kill the whole group after the grace period, including descendants
    # whose group leader has already exited.
    end = time.monotonic() + timeout
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
