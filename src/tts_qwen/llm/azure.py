"""Azure OpenAI auth helpers shared by the real, official-prompt agents.

Both real agents talk to the *same* model the official benchmarks use (``gpt-4.1`` by default).
The two benchmarks live in different interpreters, so the transport differs:

* τ²-bench drives its official orchestrator through **litellm** (model string ``azure/<deploy>``);
  litellm reads ``AZURE_API_BASE`` / ``AZURE_API_VERSION`` / ``AZURE_OPENAI_AD_TOKEN`` from the
  environment.  :func:`ensure_litellm_azure_env` populates those from the project's Azure config
  (endpoint + ``az login`` bearer token on the ``cognitiveservices`` scope), refreshing the token.
* AppWorld uses the lightweight :class:`~tts_qwen.llm.chat_client.ChatClient` (openai + azure-identity),
  which already manages AAD tokens itself.

Nothing here is imported at module load of the env wrappers; the agents call it lazily so that
offline/unit-test paths never require Azure credentials.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


def azure_endpoint() -> Optional[str]:
    return os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_API_BASE")


def azure_api_version() -> str:
    return (
        os.environ.get("AZURE_OPENAI_API_VERSION")
        or os.environ.get("AZURE_API_VERSION")
        or "2024-12-01-preview"
    )


def azure_deployment() -> str:
    """The Azure deployment / model name the official benchmarks are scored with."""
    return (
        os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("WM_MODEL")
        or os.environ.get("TTS_AGENT_MODEL")
        or "gpt-4.1"
    )


_TOKEN_LOCK = threading.Lock()
_TOKEN_CACHE = {"token": None, "exp": 0.0, "cred": None}


def fetch_aad_token(scope: str = COGNITIVE_SERVICES_SCOPE, *, force_refresh: bool = False,
                    min_remaining: float = 300.0) -> str:
    """A valid Azure AD bearer token (``az login`` identity), cached process-wide and thread-safe.

    A single :class:`AzureCliCredential` is reused and the token is cached until ``min_remaining``
    seconds before expiry, so high ``--workers`` runs don't spawn many concurrent ``az`` calls (a
    common source of spurious auth failures).  ``force_refresh`` bypasses the cache (used by the
    rollout retry when Azure rejects a token)."""
    now = time.time()
    with _TOKEN_LOCK:
        tok = _TOKEN_CACHE["token"]
        if tok and not force_refresh and (_TOKEN_CACHE["exp"] - now) > min_remaining:
            return tok
        try:
            from azure.identity import AzureCliCredential  # type: ignore

            if _TOKEN_CACHE["cred"] is None:
                _TOKEN_CACHE["cred"] = AzureCliCredential()
            access = _TOKEN_CACHE["cred"].get_token(scope)
            _TOKEN_CACHE["token"] = access.token
            _TOKEN_CACHE["exp"] = float(getattr(access, "expires_on", now + 3000))
            return access.token
        except Exception:  # pragma: no cover - fall back to the az CLI
            import subprocess

            resource = scope.rsplit("/", 1)[0]
            out = subprocess.run(
                ["az", "account", "get-access-token", "--resource", resource,
                 "--query", "accessToken", "-o", "tsv"],
                capture_output=True, text=True, check=True,
            )
            _TOKEN_CACHE["token"] = out.stdout.strip()
            _TOKEN_CACHE["exp"] = now + 3000
            return _TOKEN_CACHE["token"]


def litellm_model() -> str:
    """The litellm model string for the official benchmark model (``azure/<deployment>``)."""
    explicit = os.environ.get("TTS_LITELLM_MODEL")
    if explicit:
        return explicit
    if azure_endpoint():
        return f"azure/{azure_deployment()}"
    return azure_deployment()


_PROVIDER_INSTALLED = False


def _install_litellm_token_provider() -> bool:
    """Make litellm authenticate Azure via a self-refreshing token *provider* (a callable) backed by
    our process-wide cached :func:`fetch_aad_token`, rather than a static bearer token.

    Why: litellm caches the ``AzureOpenAI`` client keyed by ``(api_base, api_version, …)`` but **not**
    by the token (see ``litellm/llms/openai/common_utils.py::get_openai_client_cache_key``).  So a
    static ``AZURE_AD_TOKEN`` env var is baked into the first client and reused forever; once it
    expires (~1 h into a long, high ``--workers`` run) every later call 401s with
    "invalid subscription key or wrong API endpoint", and refreshing the env var changes nothing
    because the cached client is reused.  A provider callable is instead invoked *per request*, so the
    cached client always sends a fresh token.  We point litellm's provider at our cached token getter
    (refreshed only near expiry, lock-guarded) so high concurrency doesn't trigger an ``az`` shell-out
    per request.  Idempotent.
    """
    global _PROVIDER_INSTALLED
    if _PROVIDER_INSTALLED:
        return True
    try:
        import litellm
        from litellm.llms.azure import common_utils as _azure_cu
    except Exception:
        return False

    litellm.enable_azure_ad_token_refresh = True

    def _our_provider(*_args, **_kwargs):  # signature-tolerant replacement
        return lambda: fetch_aad_token()  # AzureOpenAI calls this per request

    _azure_cu.get_azure_ad_token_provider = _our_provider
    _install_completion_auth_retry()
    _PROVIDER_INSTALLED = True
    return True


_COMPLETION_WRAPPED = False


def _install_completion_auth_retry() -> None:
    """Wrap ``litellm.completion`` (and tau2's imported reference) so *any* Azure auth failure — from
    any call in the official tau2 stack (agent, user-simulator, NL-assertion judge) — self-heals:
    clear stale api-keys, force-refresh the AAD token, flush litellm's client cache, and retry.  This
    is defence-in-depth on top of the provider + key-clearing, so a single transient 401 deep in a
    long run can't kill a task."""
    global _COMPLETION_WRAPPED
    if _COMPLETION_WRAPPED:
        return
    try:
        import functools

        import litellm
    except Exception:
        return

    def _wrap(orig):
        @functools.wraps(orig)
        def _retrying(*args, **kwargs):
            for attempt in range(4):
                try:
                    return orig(*args, **kwargs)
                except Exception as err:  # noqa: BLE001
                    msg = str(err).lower()
                    is_auth = (
                        "authenticationerror" in type(err).__name__.lower()
                        or "invalid subscription key" in msg
                        or "access denied" in msg
                        or "wrong api endpoint" in msg
                        or "401" in msg
                    )
                    if not is_auth or attempt == 3:
                        raise
                    _clear_azure_api_keys()
                    fetch_aad_token(force_refresh=True)
                    _flush_litellm_client_cache()
                    time.sleep(0.5 * (attempt + 1))
            raise RuntimeError("unreachable")  # pragma: no cover

        return _retrying

    litellm.completion = _wrap(litellm.completion)
    try:  # tau2 does `from litellm import completion`, holding its own reference
        from tau2.utils import llm_utils as _tau2_llm

        _tau2_llm.completion = _wrap(_tau2_llm.completion)
    except Exception:
        pass
    _COMPLETION_WRAPPED = True


def _flush_litellm_client_cache() -> None:
    """Drop litellm's in-memory Azure client cache so the next call rebuilds the client (used on a
    retry, in case a client was cached with bad/stale auth)."""
    try:
        import litellm

        litellm.in_memory_llm_clients_cache.flush_cache()
    except Exception:
        pass


def _azure_auth_mode() -> str:
    """Resolve the intended Azure auth mode, mirroring ``chat_client``'s detection.

    Explicit ``AZURE_AUTH_MODE`` wins (``gcr``/``managedidentity``/… ⇒ AAD; ``api_key`` ⇒ key auth).
    Otherwise: a key env var implies ``api_key``, else default to ``gcr`` (AAD via ``az login``).
    """
    explicit = os.environ.get("AZURE_AUTH_MODE")
    if explicit:
        return explicit.strip().lower()
    if os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY"):
        return "api_key"
    return "gcr"


def _clear_azure_api_keys() -> None:
    """Remove every api-key source litellm might prefer over the AAD provider.

    litellm only builds/uses the ``azure_ad_token_provider`` when **no api_key is present**
    (``litellm/llms/azure/common_utils.py`` requires ``not api_key``).  A stale
    ``AZURE_OPENAI_API_KEY``/``AZURE_API_KEY`` (left in the shell from another tool) therefore makes
    litellm silently ignore AAD and send that bad key → "Access denied due to invalid subscription
    key or wrong API endpoint".  In AAD mode those keys are never wanted, so we strip them."""
    for _k in ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"):
        os.environ.pop(_k, None)
    try:
        import litellm

        litellm.api_key = None  # also clear the litellm global, which outranks the env vars
    except Exception:
        pass


def _configure_litellm_azure_transport(force_refresh: bool = False) -> bool:
    """Configure Azure routing/auth without changing the default model."""
    endpoint = azure_endpoint()
    if not endpoint:
        return False
    os.environ["AZURE_API_BASE"] = endpoint
    os.environ["AZURE_API_VERSION"] = azure_api_version()

    if _azure_auth_mode() == "api_key":
        return True

    if _install_litellm_token_provider():
        for _k in ("AZURE_OPENAI_AD_TOKEN", "AZURE_AD_TOKEN"):
            os.environ.pop(_k, None)
        _clear_azure_api_keys()
    else:
        token = fetch_aad_token(force_refresh=force_refresh)
        os.environ["AZURE_OPENAI_AD_TOKEN"] = token
        os.environ["AZURE_AD_TOKEN"] = token
        _clear_azure_api_keys()
    if force_refresh:
        fetch_aad_token(force_refresh=True)
        _flush_litellm_client_cache()
    return True


def ensure_litellm_azure_env(force_refresh: bool = False) -> str:
    """Configure litellm for Azure auth and return the model string.

    In AAD mode (``AZURE_AUTH_MODE`` ≠ ``api_key`` — the project default ``gcr``) we authenticate with
    a self-refreshing token *provider* (see :func:`_install_litellm_token_provider`) and **clear any
    api-key** so litellm can't fall back to a stale key.  Two distinct failure modes are handled:

    * a static bearer token gets cached into litellm's client and goes stale ~1 h into a long run →
      fixed by the per-request provider;
    * a leftover ``AZURE_OPENAI_API_KEY`` makes litellm ignore AAD entirely and 401 with
      "invalid subscription key" → fixed by clearing the key here.

    ``force_refresh=True`` (rollout retry) refreshes the token cache and flushes litellm's client
    cache.  Honours ``AZURE_AUTH_MODE=api_key`` by leaving the key in place.
    """
    from .local import local_llm_configured, local_model, litellm_model_for

    if local_llm_configured():
        # Local, self-hosted server: litellm routes the openai/ provider via OPENAI_API_BASE /
        # OPENAI_API_KEY. Mixed mode additionally configures azure/* models without changing the
        # local model used by the agent/world model.
        base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        if base:
            os.environ["OPENAI_API_BASE"] = base
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        if os.environ.get("TTS_LOCAL_LLM_ALLOW_AZURE") == "1":
            _configure_litellm_azure_transport(force_refresh=force_refresh)
        return litellm_model_for(local_model(), litellm_model())

    if not _configure_litellm_azure_transport(force_refresh=force_refresh):
        return litellm_model()
    return litellm_model()


__all__ = [
    "COGNITIVE_SERVICES_SCOPE",
    "azure_endpoint",
    "azure_api_version",
    "azure_deployment",
    "fetch_aad_token",
    "litellm_model",
    "ensure_litellm_azure_env",
]
