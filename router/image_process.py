"""Keep image inference's CUDA context out of the long-lived router."""
from __future__ import annotations

import multiprocessing
import threading
import traceback


def _worker(connection, model):
    # Only the spawned interpreter imports torch and creates CUDA resources.
    from .qwen_image import QwenImageBackend

    try:
        backend = QwenImageBackend(model)
        backend.load()
        connection.send((True, None))
        while True:
            message = connection.recv()
            if message is None:
                return
            try:
                result = backend.generate(*message)
                connection.send((True, result))
            except Exception:
                connection.send((False, traceback.format_exc()))
    except EOFError:
        pass
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


class ImageProcess:
    def __init__(self, model):
        self.model = model
        self.process = None
        self.connection = None
        self.lock = threading.Lock()

    def _receive(self, timeout=None):
        if timeout is not None and not self.connection.poll(timeout):
            raise RuntimeError("image worker startup timed out")
        try:
            success, result = self.connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError("image worker exited unexpectedly") from exc
        if not success:
            raise RuntimeError(result)
        return result

    def load(self):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, self.model), daemon=True)
        try:
            self.process.start()
        finally:
            child.close()
        self._receive(self.model.startup_timeout)

    def generate(self, options, images):
        with self.lock:
            self.connection.send((options, images))
            return self._receive()

    def unload(self):
        # A cancelled asyncio.to_thread call may still be receiving a generation.
        # Terminate the process independently of that thread's lock or pipe state.
        process = self.process
        if process is None:
            return
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(self.model.stop_timeout)
            if process.is_alive():
                process.kill()
                process.join(5)
            if process.is_alive():
                raise RuntimeError("image worker survived SIGKILL")
        with self.lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
        process.close()
        self.process = None
