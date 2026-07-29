"""Prompt templates for the WM success verifiers, plus faithful baselines.

A template is a ``system`` string + an ordered list of ``sections``.  Each section is either:

* ``text`` — a literal instruction string (this is what the optimizer may reword), or
* ``data`` — a slot filled verbatim from a :class:`~promptopt.dataset.LabeledExample`'s ``data``.

Only wording (``system`` + ``text`` sections) is mutable; the ``data`` sections and their order are
fixed, so the optimizer can never break the data plumbing.  The baselines reproduce the *current*
source prompts (``officebench_wm._SYSTEM`` / ``_build_prompt`` and ``real_tau2.predict_success``) so
the baseline score is a faithful reference the optimizer must beat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


class _Safe(dict):
    def __missing__(self, key: str) -> str:  # leave unknown {placeholders} untouched
        return "{" + key + "}"


def _text(value: str, requires: Optional[str] = None) -> Dict:
    s = {"kind": "text", "value": value}
    if requires:
        s["requires"] = requires
    return s


def _data(key: str, requires: Optional[str] = None) -> Dict:
    s = {"kind": "data", "key": key}
    s["requires"] = requires or key
    return s


@dataclass
class PromptTemplate:
    env: str
    name: str
    system: str
    sections: List[Dict]
    version: int = 0
    notes: str = ""

    # -- rendering ------------------------------------------------------- #
    def render(self, data: Dict[str, str]) -> "tuple[str, str]":
        """Return ``(system, user)`` for the given example data."""
        parts: List[str] = []
        for sec in self.sections:
            req = sec.get("requires")
            if req is not None and not str(data.get(req, "")).strip():
                continue
            if sec["kind"] == "text":
                parts.append(str(sec["value"]).format_map(_Safe(data)))
            else:
                val = data.get(sec["key"], "")
                if str(val).strip():
                    parts.append(str(val))
        system = self.system.format_map(_Safe(data)) if self.system else ""
        return system, "\n".join(p for p in parts if p is not None)

    # -- optimizer surface (text sections only) -------------------------- #
    def text_values(self) -> List[str]:
        return [s["value"] for s in self.sections if s["kind"] == "text"]

    def with_text(self, system: str, texts: List[str]) -> "PromptTemplate":
        """Return a copy with ``system`` and the ordered text sections replaced.

        ``texts`` must match the number of text sections (data sections/order are preserved)."""
        want = sum(1 for s in self.sections if s["kind"] == "text")
        if len(texts) != want:
            raise ValueError(f"expected {want} text sections, got {len(texts)}")
        it = iter(texts)
        new_sections: List[Dict] = []
        for s in self.sections:
            if s["kind"] == "text":
                ns = dict(s)
                ns["value"] = next(it)
                new_sections.append(ns)
            else:
                new_sections.append(dict(s))
        return PromptTemplate(env=self.env, name=self.name, system=system,
                              sections=new_sections, version=self.version + 1, notes=self.notes)

    # -- persistence ----------------------------------------------------- #
    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @staticmethod
    def from_dict(d: Dict) -> "PromptTemplate":
        return PromptTemplate(env=d["env"], name=d["name"], system=d["system"],
                              sections=list(d["sections"]), version=int(d.get("version", 0)),
                              notes=d.get("notes", ""))


# ---------------------------------------------------------------------------
# Parsing (faithful to the source verifiers)
# ---------------------------------------------------------------------------

def parse_label(env: str, text: str) -> bool:
    if env == "officebench":
        from tts.env.officebench_wm import _is_success
        return _is_success(text)
    if env == "tau2":
        from tts.env.real_tau2 import _parse_verdict
        return _parse_verdict(text)
    raise KeyError(f"unknown env {env!r}")


def predict(template: PromptTemplate, data: Dict[str, str], llm, *,
            k: int = 1, temperature: float = 0.0, max_tokens: int = 512) -> float:
    """Return the success fraction over ``k`` verifier samples (a confidence in ``[0, 1]``)."""
    system, user = template.render(data)
    votes = 0
    for i in range(max(1, k)):
        temp = temperature if k > 1 else 0.0
        out = llm.chat(system, user, temperature=temp, max_tokens=max_tokens)
        if parse_label(template.env, out):
            votes += 1
    return votes / float(max(1, k))


# ---------------------------------------------------------------------------
# Baselines — reproduce the current source prompts
# ---------------------------------------------------------------------------

def baseline_officebench() -> PromptTemplate:
    """Mirror of ``officebench_wm._SYSTEM`` + ``OfficeBenchVerifier._build_prompt``."""
    from tts.env.officebench_wm import _SYSTEM

    sections = [
        _text("TASK INSTRUCTION:"),
        _data("instruction"),
        _text("\nREALIZED AGENT TRAJECTORY (action -> observation):"),
        _data("trajectory"),
        _text("\nPREDICTED RESULTING FILE STATE (world-model prediction; writes were NOT "
              "executed — reason about whether the writes achieve the task):", requires="final_state"),
        _data("final_state"),
        _data("giveup_note"),
        _text("\nGROUND-TRUTH SUCCESS CHECKS (the task is graded on ALL of these passing):",
              requires="requirements"),
        _data("requirements"),
        _text("\nGOLD EXPECTED OUTPUT (applying the write actions above to the initial "
              "content must yield a file matching this exactly):", requires="gold_files"),
        _data("gold_files"),
        _text("\nUsing the PREDICTED resulting file state (not the agent's claims), did the agent "
              "correctly and completely accomplish the task? Answer with exactly one word: "
              "SUCCESS or FAILURE."),
    ]
    return PromptTemplate(env="officebench", name="baseline", system=_SYSTEM, sections=sections)


def baseline_tau2() -> PromptTemplate:
    """Mirror of ``real_tau2._LLMWorldModelPredictor.predict_success`` (one-shot verifier)."""
    sections = [
        _text("You are a world model of the τ²-bench {domain} database. Decide whether the "
              "CANDIDATE tool-call sequence leaves the database in the SAME final state as the GOLD "
              "(known-correct) sequence, starting from the initial records below. Reason about each "
              "sequence's net effect on the records."),
        _data("ref"),
        _text("Initial records (read-grounded):"),
        _data("facts"),
        _text("GOLD actions:"),
        _data("gold"),
        _text("CANDIDATE actions:"),
        _data("cand"),
        _text("On the LAST line answer exactly 'VERDICT: SUCCESS' or 'VERDICT: FAIL'."),
    ]
    return PromptTemplate(env="tau2", name="baseline", system="", sections=sections)


def baseline(env: str) -> PromptTemplate:
    if env == "officebench":
        return baseline_officebench()
    if env == "tau2":
        return baseline_tau2()
    raise KeyError(f"unknown env {env!r}")


__all__ = ["PromptTemplate", "parse_label", "predict", "baseline",
           "baseline_officebench", "baseline_tau2"]
