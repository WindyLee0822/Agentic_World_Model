"""Local (self-hosted) OpenAI-compatible backend wiring (``--local``).

These verify pure-local routing and the mixed tau2 mode where local ``openai/*`` agent calls coexist
with an explicitly qualified ``azure/*`` user simulator.
"""

from __future__ import annotations

import os

import pytest

import tts.llm.chat_client as _cc
from tts.llm.azure import ensure_litellm_azure_env
from tts.llm.local import (
    DEFAULT_LOCAL_BASE_URL,
    configure_local_llm,
    litellm_model_for,
    local_base_url,
    local_llm_configured,
    local_model,
    probe_local_server,
)


@pytest.fixture(autouse=True)
def _isolate_env():
    """Snapshot and restore ``os.environ`` — ``configure_local_llm`` mutates it in place."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_not_configured_by_default():
    # A fresh (Azure-only) shell is not in local mode.
    for k in ("TTS_LOCAL_LLM", "TTS_LLM_BACKEND"):
        os.environ.pop(k, None)
    assert local_llm_configured() is False
    assert local_model() is None
    assert local_base_url() is None


def test_configure_sets_all_transport_env_vars():
    # Simulate the project's Azure shell being present before --local.
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://example.openai.azure.com/"
    returned = configure_local_llm(
        base_url="http://localhost:9001/v1", model="Qwen-AgentWorld-35B-A3B", api_key="EMPTY"
    )
    assert returned == "Qwen-AgentWorld-35B-A3B"
    assert local_llm_configured() is True
    assert local_model() == "Qwen-AgentWorld-35B-A3B"
    assert local_base_url() == "http://localhost:9001/v1"

    # ChatClient transport (AppWorld / OfficeBench)
    assert os.environ["TTS_LLM_BACKEND"] == "openai"
    assert os.environ["OPENAI_BASE_URL"] == "http://localhost:9001/v1"
    assert os.environ["OPENAI_API_KEY"] == "EMPTY"
    assert os.environ["WM_MODEL"] == "Qwen-AgentWorld-35B-A3B"

    # litellm transport (tau2)
    assert os.environ["OPENAI_API_BASE"] == "http://localhost:9001/v1"
    assert os.environ["TTS_LITELLM_MODEL"] == "openai/Qwen-AgentWorld-35B-A3B"

    # a local server never wants the stale Azure endpoint
    assert "AZURE_OPENAI_ENDPOINT" not in os.environ


def test_configure_defaults():
    configure_local_llm(model="my-model")
    assert local_base_url() == DEFAULT_LOCAL_BASE_URL
    assert os.environ["OPENAI_API_KEY"] == "EMPTY"


def test_configure_requires_model():
    with pytest.raises(ValueError):
        configure_local_llm(model="")


def test_detect_backend_override_beats_azure_endpoint():
    # Even with an Azure endpoint present, TTS_LLM_BACKEND=openai forces the OpenAI backend so
    # ChatClient targets the local OPENAI_BASE_URL.
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://example.openai.azure.com/"
    os.environ["TTS_LLM_BACKEND"] = "openai"
    assert _cc.detect_backend() == "openai"


def test_detect_backend_without_override():
    os.environ.pop("TTS_LLM_BACKEND", None)
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://example.openai.azure.com/"
    assert _cc.detect_backend() == "azure"
    os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
    assert _cc.detect_backend() == "openai"


def test_default_model_is_local_model():
    configure_local_llm(model="Qwen-AgentWorld-35B-A3B")
    assert _cc.detect_backend() == "openai"
    assert _cc.default_model() == "Qwen-AgentWorld-35B-A3B"


def test_litellm_model_for_preserves_configured_provider():
    # Outside local mode: bare models inherit a qualified default's provider.
    os.environ.pop("TTS_LOCAL_LLM", None)
    assert litellm_model_for("gpt-4.1", "gpt-4.1") == "gpt-4.1"
    assert litellm_model_for("gpt-4.1", "azure/gpt-4.1") == "azure/gpt-4.1"
    assert litellm_model_for(None, "azure/gpt-4.1") == "azure/gpt-4.1"

    # In local mode: bare id -> openai/<id>, already-qualified strings untouched.
    configure_local_llm(model="Qwen-AgentWorld-35B-A3B")
    assert litellm_model_for("Qwen-AgentWorld-35B-A3B", "x") == "openai/Qwen-AgentWorld-35B-A3B"
    assert litellm_model_for(None, "Qwen-AgentWorld-35B-A3B") == "openai/Qwen-AgentWorld-35B-A3B"
    assert litellm_model_for("openai/foo", "x") == "openai/foo"
    assert litellm_model_for("hosted_vllm/foo", "x") == "hosted_vllm/foo"


def test_ensure_litellm_env_returns_local_string_without_azure():
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://example.openai.azure.com/"
    configure_local_llm(base_url="http://localhost:8123/v1", model="Qwen-AgentWorld-35B-A3B")
    # ensure_litellm_azure_env must NOT reach for Azure; it returns the openai/ litellm string and
    # leaves the litellm OpenAI env pointing at the local server.
    model = ensure_litellm_azure_env()
    assert model == "openai/Qwen-AgentWorld-35B-A3B"
    assert os.environ["OPENAI_API_BASE"] == "http://localhost:8123/v1"
    assert os.environ.get("AZURE_OPENAI_AD_TOKEN") is None


def test_mixed_local_and_azure_litellm_env_preserves_both_providers():
    os.environ["TTS_LOCAL_LLM_ALLOW_AZURE"] = "1"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://example.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"
    os.environ["AZURE_OPENAI_API_KEY"] = "azure-key"
    os.environ["AZURE_API_KEY"] = "azure-key"
    os.environ["AZURE_AUTH_MODE"] = "api_key"

    configure_local_llm(
        base_url="http://localhost:8123/v1",
        model="Qwen3.5-35B-A3B",
        api_key="EMPTY",
    )
    default_model = ensure_litellm_azure_env()

    assert default_model == "openai/Qwen3.5-35B-A3B"
    assert litellm_model_for("Qwen3.5-35B-A3B", default_model) == "openai/Qwen3.5-35B-A3B"
    assert litellm_model_for("azure/gpt-5.4", default_model) == "azure/gpt-5.4"
    assert os.environ["OPENAI_API_BASE"] == "http://localhost:8123/v1"
    assert os.environ["OPENAI_API_KEY"] == "EMPTY"
    assert os.environ["AZURE_API_BASE"] == "https://example.openai.azure.com/"
    assert os.environ["AZURE_API_VERSION"] == "2024-12-01-preview"
    assert os.environ["AZURE_API_KEY"] == "azure-key"


def test_probe_local_server_unreachable():
    # A closed port -> not reachable, with an explanatory detail (no exception).
    reachable, detail = probe_local_server("http://127.0.0.1:1/v1", timeout=1.0)
    assert reachable is False
    assert "cannot reach" in detail


def test_probe_local_server_reachable_and_model_check():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def do_GET(self):
            body = json.dumps({"object": "list", "data": [{"id": "served-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), _H)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}/v1"
        ok, _ = probe_local_server(base, model="served-model", timeout=3.0)
        assert ok is True
        # a model the server does not advertise -> still reachable, but the detail warns about it
        ok2, detail2 = probe_local_server(base, model="other-model", timeout=3.0)
        assert ok2 is True
        assert "does not advertise" in detail2
    finally:
        srv.shutdown()
