"""Runtime monkeypatching — used for (a) capturing verifier inputs during collection and (b) opting
in to an optimized prompt at inference.  No source file is modified; patches are applied inside a
``with`` block and reverted on exit.

Capture
-------
``capture_tau2`` / ``capture_officebench`` replace the source verifier entry points so that, while a
normal rollout runs, each verifier call records its inputs (as template data slots) into a sink list.

Apply (opt-in inference)
------------------------
``apply_best`` replaces the source verifier so it renders the stored *best* template instead of the
built-in prompt.  If no optimized prompt is stored (or you never enter the context), the current
source prompt is used unchanged — so the baseline remains the default.
"""

from __future__ import annotations

import contextlib
from typing import List, Optional

from .slots import officebench_slots, tau2_slots
from .templates import PromptTemplate, parse_label


# ---------------------------------------------------------------------------
# Capture (collection)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def capture_tau2(sink: List[dict]):
    """Record the one-shot τ² verifier's inputs (facts / candidate / gold / tool_docs) as slots.

    The patched ``predict_success`` records and returns ``False`` (no LLM call) — collection only
    needs the data, not a verdict.  The agentic verifier is left untouched."""
    from tts.env import real_tau2 as R

    P = R._LLMWorldModelPredictor
    orig = P.predict_success

    def patched(self, facts, candidate_actions, gold_actions, tool_docs=""):
        sink.append({"domain": getattr(self, "_domain", "?"),
                     "slots": tau2_slots(getattr(self, "_domain", "?"), facts,
                                         list(candidate_actions), list(gold_actions), tool_docs)})
        return False

    P.predict_success = patched
    try:
        yield sink
    finally:
        P.predict_success = orig


@contextlib.contextmanager
def capture_officebench(sink: List[dict]):
    """Record the OfficeBench verifier's inputs as slots; returns a neutral 0.5 confidence."""
    from tts.env import officebench_wm as W

    V = W.OfficeBenchVerifier
    orig = V.predict_success

    def patched(self, instruction, actions, observations, *, config=None, final_state=None,
                final_answer=None, gold_files=None):
        sink.append({"slots": officebench_slots(instruction, list(actions), list(observations),
                                                 config=config, final_state=final_state,
                                                 final_answer=final_answer, gold_files=gold_files)})
        return 0.5

    V.predict_success = patched
    try:
        yield sink
    finally:
        V.predict_success = orig


# ---------------------------------------------------------------------------
# Apply (opt-in inference with an optimized prompt)
# ---------------------------------------------------------------------------

def _render_and_judge(template: PromptTemplate, data: dict, chat, *, k: int, temperature: float,
                      max_tokens: int) -> float:
    system, user = template.render(data)
    votes = 0
    for _ in range(max(1, k)):
        temp = temperature if k > 1 else 0.0
        out = chat.chat(system, user, temperature=temp, max_tokens=max_tokens) if system else \
            chat.complete([{"role": "user", "content": user}], temperature=temp, max_tokens=max_tokens)
        if parse_label(template.env, out):
            votes += 1
    return votes / float(max(1, k))


@contextlib.contextmanager
def apply_best(env: str, store_dir: Optional[str] = None, *, k: int = 1, temperature: float = 0.7,
               max_tokens: int = 512):
    """Patch the source verifier of ``env`` to use the stored best template (opt-in)."""
    from .optimize import load_best

    template = load_best(env, store_dir)
    if template is None:
        yield False  # nothing stored — leave the source prompt in place
        return

    if env == "tau2":
        from tts.env import real_tau2 as R
        P = R._LLMWorldModelPredictor
        orig = P.predict_success

        def patched(self, facts, candidate_actions, gold_actions, tool_docs=""):
            data = tau2_slots(getattr(self, "_domain", "?"), facts, list(candidate_actions),
                              list(gold_actions), tool_docs)
            conf = _render_and_judge(template, data, self._chat(), k=k, temperature=temperature,
                                     max_tokens=max_tokens)
            return conf >= 0.5

        P.predict_success = patched
        try:
            yield True
        finally:
            P.predict_success = orig

    elif env == "officebench":
        from tts.env import officebench_wm as W
        V = W.OfficeBenchVerifier
        orig = V.predict_success

        def patched(self, instruction, actions, observations, *, config=None, final_state=None,
                    final_answer=None, gold_files=None):
            data = officebench_slots(instruction, list(actions), list(observations), config=config,
                                     final_state=final_state, final_answer=final_answer,
                                     gold_files=gold_files)
            return _render_and_judge(template, data, self._chat, k=k, temperature=temperature,
                                     max_tokens=max_tokens)

        V.predict_success = patched
        try:
            yield True
        finally:
            V.predict_success = orig
    else:
        raise KeyError(f"unknown env {env!r}")


__all__ = ["capture_tau2", "capture_officebench", "apply_best"]
