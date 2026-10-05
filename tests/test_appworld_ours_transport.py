"""Reproduce OBO authentication under the parent process's URL-opening guard."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import urllib.request

import pytest

from tts.env.appworld_ours_chat import IsolatedAppWorldChat


def test_model_authentication_stays_outside_parent_url_guard(monkeypatch):
    pytest.importorskip("openai")
    pytest.importorskip("azure.identity")
    from tts.llm.chat_client import ChatClient

    token_requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            token_requests.append(self.path)
            self.reply({"access_token": "local-test-token", "expires_on": int(time.time()) + 3600})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.reply({"id": "local", "object": "chat.completion", "created": 1,
                        "model": "local-test-model", "usage": {"total_tokens": 7},
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "PYTHON_FINAL:\nprint_json(7)"}}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("OBO_ENDPOINT", url + "/token")
    config = {"model": "local-test-model", "backend": "azure", "azure_endpoint": url,
              "azure_auth_mode": "aml", "max_retries": 1, "max_rate_limit_retries": 1}
    worker = None
    try:
        worker = IsolatedAppWorldChat(config, timeout_s=20)

        def forbidden(*args, **kwargs):
            raise RuntimeError("Usage of urllib.request.urlopen is not allowed")

        monkeypatch.setattr(urllib.request, "urlopen", forbidden)
        messages = [{"role": "user", "content": "Predict a value"}]
        assert worker.complete(messages, max_tokens=100, temperature=0) == "PYTHON_FINAL:\nprint_json(7)"
        assert worker.last_usage == 7
        assert token_requests
        # AppWorld uses a custom TimeoutError, independent of the builtin.
        # Interrupt a pending wait and leave its completion queued; the next
        # request must discard that completion and use its own response ID.
        class CellInterrupted(Exception):
            pass

        connection = worker._connection

        class InterruptOnce:
            interrupted = False

            def __getattr__(self, name):
                return getattr(connection, name)

            def poll(self, timeout):
                if not self.interrupted:
                    self.interrupted = True
                    raise CellInterrupted("cell deadline")
                return connection.poll(timeout)

        worker._connection = InterruptOnce()
        with pytest.raises(CellInterrupted):
            worker.complete(messages, max_tokens=100, temperature=0)
        assert worker.complete(messages, max_tokens=100, temperature=0) == "PYTHON_FINAL:\nprint_json(7)"
        assert worker.alive
        # The original in-process transport still sees the guard: no shared
        # authentication function or sandbox restriction has been relaxed.
        with pytest.raises(RuntimeError, match="urlopen is not allowed"):
            ChatClient(**config).complete(messages, max_tokens=100, temperature=0)
    finally:
        if worker is not None:
            worker.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
