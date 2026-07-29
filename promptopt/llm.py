"""LLM backends for prompt optimization.

:class:`RealLLM` wraps the project's :class:`tts.llm.chat_client.ChatClient` (Azure/OpenAI).
:class:`MockLLM` is a deterministic, offline stand-in so the whole optimization loop is testable
without network access — it classifies a rendered prompt with a tiny rule and can "improve" a
template by echoing a caller-supplied edit, which is enough to exercise the control flow.

Both expose the same minimal surface used by this package:

* ``chat(system, user, *, temperature=0.0, max_tokens=...) -> str``
"""

from __future__ import annotations

import os
import re
from typing import Optional, Protocol


class LLM(Protocol):
    def chat(self, system: str, user: str, *, temperature: float = 0.0,
             max_tokens: Optional[int] = None) -> str: ...


class RealLLM:
    """Thin adapter over ``tts.llm.chat_client.ChatClient``."""

    def __init__(self, model: Optional[str] = None, *, seed: Optional[int] = None) -> None:
        from tts.llm import make_chat_client

        self.model = model or os.environ.get("WM_MODEL") or os.environ.get("OPENAI_MODEL")
        self._client = make_chat_client(model=self.model, seed=seed)
        self.n_calls = 0

    def chat(self, system: str, user: str, *, temperature: float = 0.0,
             max_tokens: Optional[int] = None) -> str:
        self.n_calls += 1
        if system:
            return self._client.chat(system, user, temperature=temperature, max_tokens=max_tokens)
        # τ² verifier uses a single user message (no system role).
        return self._client.complete([{"role": "user", "content": user}],
                                     temperature=temperature, max_tokens=max_tokens)


class MockLLM:
    """Deterministic offline LLM.

    * As a **verifier**: emits ``SUCCESS``/``VERDICT: SUCCESS`` when the rendered prompt contains a
      positive cue, else a failure verdict.  A cue can be injected per prompt so tests can control
      the accuracy of a given template (used to prove the optimizer prefers a better template).
    * As an **optimizer**: returns whatever JSON is pre-loaded in ``scripted_edits`` (FIFO), so a
      test can drive the reflection loop deterministically.
    """

    #: substring that, when present in the *system or user* text, flips the mock to predict SUCCESS.
    POSITIVE_CUE = "GROUND-TRUTH"

    def __init__(self, scripted_edits: Optional[list] = None,
                 success_if: Optional[str] = None) -> None:
        self.scripted_edits = list(scripted_edits or [])
        self.success_if = success_if
        self.n_calls = 0

    def chat(self, system: str, user: str, *, temperature: float = 0.0,
             max_tokens: Optional[int] = None) -> str:
        self.n_calls += 1
        text = f"{system}\n{user}"
        # Optimizer role: the reflection prompt asks for JSON; return the next scripted edit.
        if "propose an improved" in text.lower() or "return only a json" in text.lower():
            return self.scripted_edits.pop(0) if self.scripted_edits else "{}"
        # Verifier role: predict from an injected/true cue embedded in the example data.
        cue = self.success_if or self.POSITIVE_CUE
        positive = ("__TRUE_SUCCESS__" in text) if "__TRUE_" in text else (cue in text)
        if re.search(r"VERDICT:\s*SUCCESS", text) or "answer exactly" in text.lower() and "VERDICT" in text:
            return "VERDICT: SUCCESS" if positive else "VERDICT: FAIL"
        return "SUCCESS" if positive else "FAILURE"


def make_llm(kind: str = "real", **kw) -> LLM:
    if kind == "mock":
        return MockLLM(**kw)
    return RealLLM(**kw)


__all__ = ["LLM", "RealLLM", "MockLLM", "make_llm"]
