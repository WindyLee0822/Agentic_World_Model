"""Low-level chat client for OpenAI-compatible and Azure OpenAI endpoints.

This mirrors the auth/retry behaviour of the project's existing ``world_model/llm.py``
(see ``/home/t-wendili/env-tts``): the backend is auto-detected from the environment, and
Azure AAD auth (``AZURE_AUTH_MODE=gcr`` → ``az login`` via :class:`AzureCliCredential`) is
supported with automatic recovery from expired bearer tokens.

It is intentionally decoupled from the TTS abstractions — :class:`tts.llm.openai_llm.OpenAIChatLLM`
adapts it to the :class:`tts.abcs.LLM` interface.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def detect_backend() -> str:
    """``"azure"`` if an Azure endpoint is configured, else ``"openai"``."""
    return "azure" if os.environ.get("AZURE_OPENAI_ENDPOINT") else "openai"


def default_model() -> str:
    """Worker model id (OpenAI) or deployment name (Azure)."""
    if detect_backend() == "azure":
        return (
            os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or os.environ.get("WM_MODEL")
            or "gpt-4o-mini"
        )
    return os.environ.get("WM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"


def _default_azure_auth_mode() -> str:
    explicit = os.environ.get("AZURE_AUTH_MODE")
    if explicit:
        return explicit.lower()
    if os.environ.get("AZURE_OPENAI_USE_AAD", "0") in ("1", "true", "True"):
        return "default"
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        return "api_key"
    return "gcr"  # no key, no explicit mode → assume `az login` (GCR-style)


def _build_token_provider(auth_mode: str) -> Callable[[], str]:
    try:
        from azure.identity import (  # type: ignore
            AzureCliCredential,
            DefaultAzureCredential,
            get_bearer_token_provider,
        )
    except ImportError as err:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            f"azure_auth_mode='{auth_mode}' requires `pip install azure-identity`."
        ) from err

    mode = auth_mode.lower()
    if mode == "gcr":
        credential = AzureCliCredential()
    elif mode in ("aml", "default", "aad"):
        credential = DefaultAzureCredential()
    else:
        raise ValueError(
            f"Unsupported azure_auth_mode {auth_mode!r}; use api_key|gcr|aml|default."
        )
    return get_bearer_token_provider(credential, COGNITIVE_SERVICES_SCOPE)


@dataclass
class ChatClient:
    """Minimal chat wrapper with Azure-AAD and reasoning-model auto-recovery.

    Token usage from the last call is exposed via :attr:`last_usage` so callers can do
    real cost/token accounting (the recursive simulator charges per node).
    """

    model: str = field(default_factory=default_model)
    backend: str = field(default_factory=detect_backend)

    base_url: Optional[str] = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL") or None)
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "EMPTY"))

    azure_endpoint: Optional[str] = field(default_factory=lambda: os.environ.get("AZURE_OPENAI_ENDPOINT") or None)
    azure_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("AZURE_OPENAI_API_KEY") or None)
    azure_api_version: str = field(default_factory=lambda: os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))
    azure_auth_mode: str = field(default_factory=_default_azure_auth_mode)

    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 4

    # Rate-limit (HTTP 429 "Too Many Requests") handling.  These get their own,
    # larger retry budget so transient throttling — common when many AppWorld
    # workers hit the same Azure deployment in parallel — doesn't exhaust the
    # normal retry allowance.  Defaults are tunable via env vars.
    max_rate_limit_retries: int = field(
        default_factory=lambda: _env_int("TTS_LLM_RATE_LIMIT_RETRIES", 8)
    )
    rate_limit_base_delay: float = field(
        default_factory=lambda: _env_float("TTS_LLM_RATE_LIMIT_BASE_DELAY", 2.0)
    )
    rate_limit_max_delay: float = field(
        default_factory=lambda: _env_float("TTS_LLM_RATE_LIMIT_MAX_DELAY", 60.0)
    )
    rate_limit_jitter: float = field(
        default_factory=lambda: _env_float("TTS_LLM_RATE_LIMIT_JITTER", 1.0)
    )

    _use_completion_tokens: bool = False
    _supports_temperature: bool = True
    last_usage: int = 0

    def __post_init__(self) -> None:
        self._build_client()

    def _build_client(self) -> None:
        from openai import AzureOpenAI, OpenAI  # imported lazily

        if self.backend == "azure":
            if not self.azure_endpoint:
                raise ValueError("backend='azure' but AZURE_OPENAI_ENDPOINT is not set.")
            kwargs: dict[str, Any] = {
                "azure_endpoint": self.azure_endpoint,
                "api_version": self.azure_api_version,
            }
            mode = (self.azure_auth_mode or "api_key").lower()
            if mode == "api_key":
                if not self.azure_api_key:
                    raise ValueError(
                        "azure_auth_mode='api_key' but AZURE_OPENAI_API_KEY is not set."
                    )
                kwargs["api_key"] = self.azure_api_key
            else:
                kwargs["azure_ad_token_provider"] = _build_token_provider(mode)
            kwargs["max_retries"] = self.max_rate_limit_retries
            self._client = AzureOpenAI(**kwargs)
        else:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=self.max_rate_limit_retries,
            )

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.complete(messages, temperature=temperature, max_tokens=max_tokens)

    def complete(
        self,
        messages: list,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a raw ``[{role, content}, ...]`` message list (multi-turn) and return the text."""
        delay = 1.0
        rate_limit_attempts = 0
        attempt = 0
        last_err: Optional[Exception] = None
        while attempt < self.max_retries:
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            tok = self.max_tokens if max_tokens is None else max_tokens
            kwargs["max_completion_tokens" if self._use_completion_tokens else "max_tokens"] = tok
            if self._supports_temperature:
                kwargs["temperature"] = self.temperature if temperature is None else temperature
            try:
                resp = self._client.chat.completions.create(**kwargs)
                usage = getattr(resp, "usage", None)
                self.last_usage = int(getattr(usage, "total_tokens", 0) or 0)
                return (resp.choices[0].message.content or "").strip()
            except Exception as err:  # noqa: BLE001
                last_err = err
                msg = str(err).lower()
                # Rate limit (HTTP 429): back off and retry using a dedicated budget so a
                # throttled endpoint doesn't burn the normal retry allowance.  Honour the
                # server's Retry-After hint when present; otherwise exponential backoff with
                # jitter to avoid a thundering herd of parallel workers retrying in lockstep.
                if self._is_rate_limit_error(err, msg):
                    rate_limit_attempts += 1
                    if rate_limit_attempts >= self.max_rate_limit_retries:
                        break
                    wait = self._retry_after_seconds(err)
                    if wait is None:
                        wait = min(
                            self.rate_limit_base_delay * (2 ** (rate_limit_attempts - 1)),
                            self.rate_limit_max_delay,
                        )
                    wait = min(wait, self.rate_limit_max_delay)
                    if self.rate_limit_jitter:
                        wait += random.uniform(0.0, self.rate_limit_jitter)
                    time.sleep(wait)
                    continue
                if (
                    self.backend == "azure"
                    and (self.azure_auth_mode or "").lower() != "api_key"
                    and ("401" in msg or "unauthorized" in msg or "expired" in msg or "invalid_issuer" in msg)
                ):
                    try:
                        self._build_client()
                    except Exception:  # noqa: BLE001
                        pass
                    time.sleep(delay)
                    delay = min(delay * 2, 16.0)
                    attempt += 1
                    continue
                if not self._use_completion_tokens and "max_tokens" in msg and "max_completion_tokens" in msg:
                    self._use_completion_tokens = True
                    continue
                if self._supports_temperature and "temperature" in msg and ("unsupported" in msg or "does not support" in msg):
                    self._supports_temperature = False
                    continue
                time.sleep(delay)
                delay = min(delay * 2, 16.0)
                attempt += 1
        if rate_limit_attempts and self._is_rate_limit_error(last_err, str(last_err).lower()):
            raise RuntimeError(
                f"LLM call failed after {rate_limit_attempts} rate-limit retries "
                f"(HTTP 429 'Too Many Requests'): {last_err}"
            )
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    @staticmethod
    def _is_rate_limit_error(err: Optional[Exception], msg: str) -> bool:
        """True if ``err`` is an HTTP 429 / rate-limit ("too many requests") error."""
        if err is None:
            return False
        if getattr(err, "status_code", None) == 429:
            return True
        if getattr(getattr(err, "response", None), "status_code", None) == 429:
            return True
        code = getattr(err, "code", None)
        if code in (429, "429", "rate_limit_exceeded", "TooManyRequests"):
            return True
        return (
            "error code: 429" in msg
            or "status 429" in msg
            or " 429 " in f" {msg} "
            or "too many requests" in msg
            or "rate limit" in msg
            or "ratelimit" in msg
        )

    @staticmethod
    def _retry_after_seconds(err: Exception) -> Optional[float]:
        """Seconds to wait per the server's ``Retry-After`` hint, or ``None`` if absent."""
        headers = getattr(getattr(err, "response", None), "headers", None)
        get = getattr(headers, "get", None)
        if get is None:
            return None
        ra_ms = get("retry-after-ms") or get("x-ratelimit-reset-after-ms")
        if ra_ms:
            try:
                return max(0.0, float(ra_ms) / 1000.0)
            except (TypeError, ValueError):
                pass
        ra = get("retry-after")
        if not ra:
            return None
        try:
            return max(0.0, float(ra))
        except (TypeError, ValueError):
            pass
        try:  # HTTP-date form
            import datetime as _dt
            from email.utils import parsedate_to_datetime

            when = parsedate_to_datetime(ra)
            if when is not None:
                now = _dt.datetime.now(when.tzinfo)
                return max(0.0, (when - now).total_seconds())
        except Exception:  # noqa: BLE001
            pass
        return None


__all__ = ["ChatClient", "detect_backend", "default_model"]
