from __future__ import annotations

import multiprocessing
import threading
import traceback


def _worker(connection, model, cancelled, first_prompt):
    from ..inference.generation import QwenImageBackend

    try:
        backend = QwenImageBackend(model)
        backend.load(first_prompt)
        connection.send((True, None))
        while True:
            message = connection.recv()
            if message is None:
                return
            try:
                result = backend.generate(*message, cancelled)
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
        self.cancelled = None
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

    def load(self, first_prompt=None):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.cancelled = context.Event()
        self.process = context.Process(
            target=_worker, args=(child, self.model, self.cancelled, first_prompt), daemon=True
        )
        try:
            self.process.start()
        finally:
            child.close()
        self._receive(self.model.startup_timeout)

    def generate(self, options, images, cancel_request):
        with self.lock:
            if cancel_request.is_set():
                return None
            self.cancelled.clear()
            self.connection.send((options, images))
            while not self.connection.poll(0.05):
                if cancel_request.is_set():
                    self.cancelled.set()
                    break
            return self._receive()

    def unload(self):
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
