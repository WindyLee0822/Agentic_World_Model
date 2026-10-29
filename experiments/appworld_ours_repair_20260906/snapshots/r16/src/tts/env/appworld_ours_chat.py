"""Ours-only model transport outside AppWorld's process-wide safety guard.

The child receives JSON messages and completion parameters, never benchmark
objects or generated executable code. AppWorld and the restricted computation
worker keep their existing guards. Authentication and HTTP retries use real time.
"""

from __future__ import annotations

import atexit
from dataclasses import fields
import json
import multiprocessing
import os
import threading
import time

_monotonic = time.monotonic


def _send(connection, value):
    connection.send_bytes(json.dumps(value, ensure_ascii=True).encode("utf-8"))


def _receive(connection):
    return json.loads(connection.recv_bytes(16_000_000))


def _model_process(connection, config):
    from ..llm.chat_client import ChatClient

    try:
        client = ChatClient(**config)
        _send(connection, {"ready": True})
    except Exception as exc:
        _send(connection, {"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        connection.close()
        return
    try:
        while True:
            request = _receive(connection)
            if request.get("close"):
                return
            request_id = request.get("id")
            try:
                kwargs = request.get("kwargs", {})
                if set(kwargs) - {"temperature", "max_tokens", "extra_body"}:
                    raise ValueError("Unsupported completion parameters")
                result = client.complete(request["messages"], **kwargs)
                response = {"id": request_id, "result": result, "usage": client.last_usage}
            except Exception as exc:
                response = {"id": request_id, "error": f"{type(exc).__name__}: {exc}"}
            _send(connection, response)
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


class IsolatedAppWorldChat:
    def __init__(self, config, timeout_s=180.0):
        self.owner_pid = os.getpid()
        self.timeout_s = timeout_s
        self.last_usage = 0
        self._lock = threading.Lock()
        self._request_id = 0
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_model_process, args=(child, config), daemon=True)
        self._process.start()
        child.close()
        if not self._connection.poll(60.0):
            self.close()
            raise RuntimeError("AppWorld model transport did not start")
        ready = _receive(self._connection)
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(ready.get("error", "AppWorld model transport startup failed"))

    @property
    def alive(self):
        return (self.owner_pid == os.getpid() and self._connection is not None
                and self._process is not None and self._process.is_alive())

    def complete(self, messages, **kwargs):
        with self._lock:
            if not self.alive:
                raise RuntimeError("AppWorld model transport is unavailable")
            self._request_id += 1
            request_id = self._request_id
            try:
                _send(self._connection, {"id": request_id, "messages": messages, "kwargs": kwargs})
                deadline = _monotonic() + self.timeout_s
                while True:
                    remaining = deadline - _monotonic()
                    if remaining <= 0 or not self._connection.poll(remaining):
                        raise TimeoutError("AppWorld model transport exceeded its response deadline")
                    response = _receive(self._connection)
                    response_id = response.get("id")
                    if isinstance(response_id, int) and response_id < request_id:
                        # AppWorld's custom cell-timeout exception may interrupt
                        # a previous wait. Its late completion is never evidence
                        # for the current prediction.
                        continue
                    if response_id != request_id:
                        self._connection.close()
                        self._connection = None
                        raise RuntimeError("AppWorld model transport returned an unexpected response")
                    break
            except (EOFError, BrokenPipeError, OSError, TimeoutError):
                # A late response must never be mistaken for the next prediction.
                self._connection.close()
                self._connection = None
                raise
            if "error" in response:
                raise RuntimeError(response["error"])
            self.last_usage = int(response.get("usage", 0))
            return response.get("result", "")

    def close(self):
        if self.owner_pid != os.getpid():
            return
        if self._connection is not None:
            try:
                _send(self._connection, {"close": True})
            except (BrokenPipeError, OSError):
                pass
            self._connection.close()
            self._connection = None
        if self._process is not None:
            self._process.join(timeout=0.2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None


_WORKERS = {}


def isolated_chat(client):
    """Reuse one transport per model configuration in each benchmark worker."""
    from ..llm.chat_client import ChatClient

    if not isinstance(client, ChatClient):
        return client  # In-process test doubles and explicit injected clients.
    config = {field.name: getattr(client, field.name) for field in fields(client)
              if field.init and not field.name.startswith("_") and field.name != "last_usage"}
    key = (os.getpid(), json.dumps(config, sort_keys=True))
    worker = _WORKERS.get(key)
    if worker is None or not worker.alive:
        if worker is not None:
            worker.close()
        timeout = float(os.environ.get("TTS_APPWORLD_OURS_LLM_TIMEOUT_S", "180"))
        worker = IsolatedAppWorldChat(config, timeout_s=max(1.0, timeout))
        _WORKERS[key] = worker
    return worker


def _close_workers():
    for worker in _WORKERS.values():
        worker.close()


atexit.register(_close_workers)
