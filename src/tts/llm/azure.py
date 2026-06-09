"""Azure OpenAI auth helpers shared by the real, official-prompt agents.

Both real agents talk to the *same* model the official benchmarks use (``gpt-4.1`` by default).
The two benchmarks live in different interpreters, so the transport differs:

* τ²-bench drives its official orchestrator through **litellm** (model string ``azure/<deploy>``);
  litellm reads ``AZURE_API_BASE`` / ``AZURE_API_VERSION`` / ``AZURE_OPENAI_AD_TOKEN`` from the
  environment.  :func:`ensure_litellm_azure_env` populates those from the project's Azure config
  (endpoint + ``az login`` bearer token on the ``cognitiveservices`` scope), refreshing the token.
* AppWorld uses the lightweight :class:`~tts.llm.chat_client.ChatClient` (openai + azure-identity),
  which already manages AAD tokens itself.

Nothing here is imported at module load of the env wrappers; the agents call it lazily so that
offline/unit-test paths never require Azure credentials.
"""

from __future__ import annotations

import os
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


def fetch_aad_token(scope: str = COGNITIVE_SERVICES_SCOPE) -> str:
    """A fresh Azure AD bearer token (``az login`` identity), via azure-identity then ``az`` CLI."""
    try:
        from azure.identity import AzureCliCredential  # type: ignore

        return AzureCliCredential().get_token(scope).token
    except Exception:  # pragma: no cover - fall back to the CLI
        import subprocess

        resource = scope.rsplit("/", 1)[0]
        out = subprocess.run(
            ["az", "account", "get-access-token", "--resource", resource,
             "--query", "accessToken", "-o", "tsv"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()


def litellm_model() -> str:
    """The litellm model string for the official benchmark model (``azure/<deployment>``)."""
    explicit = os.environ.get("TTS_LITELLM_MODEL")
    if explicit:
        return explicit
    if azure_endpoint():
        return f"azure/{azure_deployment()}"
    return azure_deployment()


def ensure_litellm_azure_env() -> str:
    """Populate the env vars litellm needs for Azure-AAD auth and return the model string.

    Idempotent; refreshes the (short-lived) bearer token on each call so longer runs keep working.
    """
    endpoint = azure_endpoint()
    if endpoint:
        os.environ["AZURE_API_BASE"] = endpoint
        os.environ["AZURE_API_VERSION"] = azure_api_version()
        if not os.environ.get("AZURE_OPENAI_API_KEY") and not os.environ.get("AZURE_API_KEY"):
            os.environ["AZURE_OPENAI_AD_TOKEN"] = fetch_aad_token()
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
