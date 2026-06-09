"""Optional real chat-model utility (Azure / OpenAI).

Only the low-level :class:`~tts.llm.chat_client.ChatClient` remains — the synthetic LLM roles
were removed with the analog environments.  The real best-of-N runner uses deterministic,
read-grounded predictors and needs no model; this client is here for building a future
LLM-driven solver / world-model on top of the read-only backing-DB access.  ``openai`` is
imported lazily, so importing this package never requires it.
"""

from __future__ import annotations

from typing import Any


def make_chat_client(**kwargs: Any):
    """Construct a :class:`ChatClient` (auto-detects Azure from ``AZURE_OPENAI_ENDPOINT``;
    honours ``AZURE_AUTH_MODE=gcr`` for ``az login`` bearer tokens)."""
    from .chat_client import ChatClient

    return ChatClient(**kwargs)


def ensure_litellm_azure_env() -> str:
    """Populate litellm's Azure-AAD env vars and return the official model string."""
    from .azure import ensure_litellm_azure_env as _ensure

    return _ensure()


__all__ = ["make_chat_client", "ensure_litellm_azure_env"]
