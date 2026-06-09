"""Rate-limit (HTTP 429 "Too Many Requests") retry behaviour of :class:`ChatClient`.

These cover the AppWorld pain point where many parallel workers throttle the shared Azure
deployment: the client must back off and retry 429s on a dedicated budget (honouring the
server's ``Retry-After`` hint) instead of failing the rollout on the first throttle.

No network or credentials are needed — a fake completions object is injected into the client.
"""

from __future__ import annotations

import time

import pytest

from tts.llm.chat_client import ChatClient


class _Headers(dict):
    def get(self, key, default=None):  # mimic httpx.Headers.get
        return super().get(key, default)


class _Response:
    def __init__(self, status_code: int, headers: dict):
        self.status_code = status_code
        self.headers = _Headers(headers)


class _RateLimitError(Exception):
    """Stand-in for ``openai.RateLimitError`` (carries status_code + response.headers)."""

    def __init__(self, message="Error code: 429 - Too Many Requests", headers=None, status=429):
        super().__init__(message)
        self.status_code = status
        self.response = _Response(status, headers or {})


class _FakeCompletions:
    def __init__(self, fail_n: int, headers=None):
        self.calls = 0
        self.fail_n = fail_n
        self.headers = headers or {"retry-after": "0"}

    def create(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.fail_n:
            raise _RateLimitError(headers=self.headers)

        class _Msg:
            content = "hello"

        class _Choice:
            message = _Msg()

        class _Usage:
            total_tokens = 7

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        return _Resp()


class _FakeClient:
    def __init__(self, fail_n: int, headers=None):
        self.chat = type("C", (), {})()
        self.chat.completions = _FakeCompletions(fail_n, headers)


def _client(**overrides) -> ChatClient:
    defaults = dict(
        backend="openai",
        api_key="x",
        max_rate_limit_retries=5,
        rate_limit_base_delay=0.0,
        rate_limit_max_delay=0.0,
        rate_limit_jitter=0.0,
        max_retries=2,
    )
    defaults.update(overrides)
    c = ChatClient(**defaults)
    return c


def test_is_rate_limit_error_detection():
    assert ChatClient._is_rate_limit_error(_RateLimitError(), "error code: 429")
    assert ChatClient._is_rate_limit_error(Exception("Rate limit reached"), "rate limit reached")
    assert ChatClient._is_rate_limit_error(Exception("429 Too Many Requests"), "429 too many requests")
    assert not ChatClient._is_rate_limit_error(Exception("boom"), "boom")
    assert not ChatClient._is_rate_limit_error(None, "")


def test_retry_after_seconds_parsing():
    assert ChatClient._retry_after_seconds(_RateLimitError(headers={"retry-after": "3"})) == 3.0
    assert ChatClient._retry_after_seconds(_RateLimitError(headers={"retry-after-ms": "1500"})) == 1.5
    assert ChatClient._retry_after_seconds(_RateLimitError(headers={})) is None
    assert ChatClient._retry_after_seconds(Exception("no response")) is None


def test_recovers_after_transient_429():
    c = _client(max_rate_limit_retries=5)
    c._client = _FakeClient(fail_n=3)
    assert c.complete([{"role": "user", "content": "hi"}]) == "hello"
    assert c._client.chat.completions.calls == 4  # 3 throttled + 1 success
    assert c.last_usage == 7


def test_exhausting_rate_limit_budget_raises_clear_error():
    c = _client(max_rate_limit_retries=3)
    c._client = _FakeClient(fail_n=99)
    with pytest.raises(RuntimeError) as excinfo:
        c.complete([{"role": "user", "content": "hi"}])
    assert "429" in str(excinfo.value)
    assert "rate-limit" in str(excinfo.value)
    assert c._client.chat.completions.calls == 3  # budget == total attempts under throttling


def test_honours_retry_after_header():
    c = _client(max_rate_limit_retries=5, rate_limit_jitter=0.0, rate_limit_max_delay=5.0)
    c._client = _FakeClient(fail_n=1, headers={"retry-after": "0.2"})
    start = time.time()
    assert c.complete([{"role": "user", "content": "hi"}]) == "hello"
    elapsed = time.time() - start
    assert 0.18 <= elapsed < 1.0  # waited ~the Retry-After hint, not the default backoff
