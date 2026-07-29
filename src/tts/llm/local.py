"""Route every LLM call to a **locally-served, OpenAI-compatible** model.

The real agents normally talk to a hosted API (Azure OpenAI / OpenAI).  To run experiments against
a self-hosted model instead (e.g. ``Qwen-AgentWorld-35B-A3B`` served by vLLM or SGLang, which both
expose an OpenAI-compatible ``/v1`` endpoint), call :func:`configure_local_llm` once in the parent
process.  It sets a small set of environment variables that the two transports already understand,
so **no** code path needs an Azure endpoint or an ``az login`` token:

* :class:`~tts.llm.chat_client.ChatClient` (AppWorld / OfficeBench) — ``TTS_LLM_BACKEND=openai``
  forces the OpenAI backend even when ``AZURE_OPENAI_ENDPOINT`` is still set in the shell, and
  ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` point it at the local server.
* **litellm** (τ²-bench) — ``OPENAI_API_BASE`` / ``OPENAI_API_KEY`` point litellm at the local
  server, and model strings are prefixed with the ``openai/`` provider (see
  :func:`litellm_model_for`) so litellm routes them to that OpenAI-compatible endpoint.

Because the runner sets this up in the *parent* (see ``experiments/run.py --local``), the env vars
are inherited by every forked/spawned worker, so it works with the AppWorld process pool and the τ²
thread pool alike.
"""

from __future__ import annotations

import os
from typing import Optional

#: Sentinel env var flipped on by :func:`configure_local_llm`; makes :func:`local_llm_configured`
#: unambiguous (rather than sniffing at ``OPENAI_BASE_URL``, which a user may set for other reasons).
_LOCAL_FLAG = "TTS_LOCAL_LLM"

#: Default endpoint of a vLLM / SGLang OpenAI-compatible server (``--host 0.0.0.0 --port 8000``).
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000/v1"


def configure_local_llm(
    *,
    base_url: str = DEFAULT_LOCAL_BASE_URL,
    model: str,
    api_key: str = "EMPTY",
) -> str:
    """Point every LLM transport at a local OpenAI-compatible server; return the served model id.

    ``model`` is the id the local server advertises (vLLM ``--served-model-name`` / SGLang
    ``--served-model-name``), e.g. ``"Qwen-AgentWorld-35B-A3B"``.  Idempotent: safe to call again
    with the same values.  The written env vars are inherited by child workers.
    """
    if not model:
        raise ValueError("configure_local_llm requires a non-empty served model id.")
    base_url = base_url or DEFAULT_LOCAL_BASE_URL
    api_key = api_key or "EMPTY"

    os.environ[_LOCAL_FLAG] = "1"
    os.environ["TTS_LOCAL_MODEL"] = model

    # ChatClient (AppWorld / OfficeBench): force the openai backend + local endpoint.
    os.environ["TTS_LLM_BACKEND"] = "openai"
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key
    # Default model resolution (ChatClient.default_model / azure_deployment) when --llm-agent is None.
    os.environ["WM_MODEL"] = model
    os.environ["TTS_AGENT_MODEL"] = model

    # litellm (τ²-bench): route the openai/ provider to the local server.
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["TTS_LITELLM_MODEL"] = litellm_model_for(model, model)

    # A local server never wants Azure credentials; drop any stale Azure endpoint so nothing
    # accidentally falls back to it (ChatClient already honours TTS_LLM_BACKEND, but litellm's
    # azure alignment sniffs AZURE_OPENAI_ENDPOINT directly).
    for _k in ("AZURE_OPENAI_ENDPOINT", "AZURE_API_BASE"):
        os.environ.pop(_k, None)
    return model


def local_llm_configured() -> bool:
    """True iff :func:`configure_local_llm` has pointed the transports at a local server."""
    return os.environ.get(_LOCAL_FLAG) == "1"


def local_model() -> Optional[str]:
    """The served model id of the local server, or ``None`` when local mode is off."""
    if not local_llm_configured():
        return None
    return os.environ.get("TTS_LOCAL_MODEL") or None


def local_base_url() -> Optional[str]:
    """The base URL of the local OpenAI-compatible server, or ``None`` when local mode is off."""
    if not local_llm_configured():
        return None
    return os.environ.get("OPENAI_BASE_URL") or DEFAULT_LOCAL_BASE_URL


def litellm_model_for(name: Optional[str], default: str) -> str:
    """The litellm model string for ``name`` (falling back to ``default``).

    In local mode a bare served-model id (no ``provider/`` prefix) is routed through litellm's
    OpenAI-compatible provider, so it is prefixed with ``openai/``; an already-qualified string
    (``openai/…``, ``hosted_vllm/…``, ``azure/…``) is returned unchanged. Outside local mode, a
    bare explicit deployment inherits the provider of a qualified default, so
    ``--llm-agent gpt-4.1`` remains ``azure/gpt-4.1`` when Azure is configured.
    """
    model = name or default
    if not model or "/" in model:
        return model
    if local_llm_configured() and model and "/" not in model:
        return f"openai/{model}"
    if "/" in default:
        provider = default.split("/", 1)[0]
        return f"{provider}/{model}"
    return model


def probe_local_server(base_url: str, *, model: Optional[str] = None, timeout: float = 5.0):
    """Best-effort health check of a local OpenAI-compatible server.

    Returns ``(reachable, detail)``.  ``reachable`` is ``True`` as soon as the server answers the
    ``GET {base_url}/models`` probe with *any* HTTP status (even 401/404) — that proves something is
    listening — and, when ``model`` is given and the response is a readable model list, ``detail``
    notes whether that served id is advertised.  ``reachable`` is ``False`` only when the connection
    is refused / times out (no server), with ``detail`` explaining the failure.  Uses the stdlib
    (``urllib``) so the preflight check needs no extra deps.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET", headers={"Authorization": "Bearer EMPTY"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local, trusted URL)
            raw = resp.read(65536)
    except urllib.error.HTTPError as err:  # server answered (auth/route error) -> it IS up
        return True, f"server responded (HTTP {err.code}) at {url}"
    except Exception as err:  # noqa: BLE001 - connection refused / timeout / DNS -> not up
        return False, f"cannot reach {url}: {type(err).__name__}: {err}"

    if model:
        try:
            ids = [m.get("id") for m in (_json.loads(raw).get("data") or [])]
        except Exception:  # noqa: BLE001
            ids = []
        if ids and model not in ids:
            return True, (f"server is up at {url}, but it does not advertise model {model!r} "
                          f"(served: {', '.join(str(i) for i in ids) or 'none'}). Pass --llm-agent "
                          f"matching the server's --served-model-name.")
    return True, f"server is up at {url}"


__all__ = [
    "DEFAULT_LOCAL_BASE_URL",
    "configure_local_llm",
    "local_llm_configured",
    "local_model",
    "local_base_url",
    "litellm_model_for",
    "probe_local_server",
]
