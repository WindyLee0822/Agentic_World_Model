"""Reflection-based prompt optimization loop.

Round structure (Automatic-Prompt-Optimization style):

1. Evaluate the current best template on the held-in subset (spanning all OfficeBench app-levels and
   ≥2 τ² domains) → collect the misclassified examples (false positives / false negatives).
2. An **advanced model** reads a balanced batch of those errors, summarizes the failure patterns, and
   rewrites the mutable instruction wording (``system`` + ``text`` sections) — the *textual gradient*.
3. Evaluate the candidate; adopt it only if it beats the current best by a margin.  The **baseline is
   always retained**, so if nothing beats it you keep the current production prompt.

Data plumbing is immutable: the model may only replace the ordered ``text`` sections and ``system``;
``with_text`` reinserts them around the fixed ``data`` slots.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import LabeledExample, slice_counts, label_balance
from .evaluate import Metrics, evaluate_template
from .templates import PromptTemplate, baseline


@dataclass
class OptimizerConfig:
    rounds: int = 4
    k: int = 1                       # verifier self-consistency samples during eval
    verifier_temperature: float = 0.7
    objective: str = "balanced_accuracy"
    min_improve: float = 1e-9        # margin a candidate must beat best by to be adopted
    n_error_examples: int = 8        # error cases shown to the optimizer per round
    error_prompt_chars: int = 1600   # truncation of each error's rendered prompt
    max_workers: int = 4
    optimizer_max_tokens: int = 4000
    seed: int = 0


_OPT_SYSTEM = (
    "You are an expert prompt engineer improving the *verifier* prompt of a world model. The verifier "
    "reads a description of an agent's trajectory and predicted resulting state, and must output a "
    "single success/failure judgement that should match the TRUE outcome graded by the real "
    "environment. You are shown the current prompt and a batch of cases it got WRONG. Diagnose the "
    "systematic failure patterns and rewrite ONLY the instruction wording to fix them."
)


def _describe_env(env: str) -> str:
    if env == "officebench":
        return ("OfficeBench: an office-automation agent edits files/emails/calendars/spreadsheets. "
                "The verifier must answer exactly 'SUCCESS' or 'FAILURE'. Its writes were PREDICTED "
                "(not executed), so it must reason over the predicted file state, not the agent's "
                "self-report.")
    if env == "tau2":
        return ("τ²-bench: an agent issues tool calls against a customer-service database. The verifier "
                "must decide whether the CANDIDATE tool-call sequence reaches the same final DB state "
                "as a known-correct GOLD sequence, and end with 'VERDICT: SUCCESS' or 'VERDICT: FAIL'. "
                "NOTE the real reward also checks information communicated to the user and treats "
                "auto-generated ids/timestamps structurally.")
    return env


def _render_error(ex: LabeledExample, template: PromptTemplate, conf: float, kind: str,
                  limit: int) -> str:
    _, user = template.render(ex.data)
    truth = "SUCCESS" if ex.true_label else "FAILURE"
    predicted = "SUCCESS" if conf >= 0.5 else "FAILURE"
    body = user if len(user) <= limit else user[:limit] + "\n…[truncated]"
    return (f"--- CASE ({kind}); slice={ex.slice_key} predicted={predicted} TRUE={truth} ---\n"
            f"{body}")


def _balanced_errors(errors: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    fp = [e for e in errors if e["kind"] == "false_positive"]
    fn = [e for e in errors if e["kind"] == "false_negative"]
    rng.shuffle(fp); rng.shuffle(fn)
    out: List[Dict] = []
    while len(out) < n and (fp or fn):
        if fp and (len(out) % 2 == 0 or not fn):
            out.append(fp.pop())
        elif fn:
            out.append(fn.pop())
    return out[:n]


def _extract_json(text: str) -> Optional[Dict]:
    """Best-effort JSON object extraction from an LLM reply (handles ```json fences / prose)."""
    if not text:
        return None
    t = text.strip()
    if "```" in t:
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
        if m:
            t = m.group(1)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    for cand in (t[start:end + 1], t):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def propose_improved_template(current: PromptTemplate, errors: List[Dict], llm, env: str, *,
                              limit: int = 1600, max_tokens: int = 4000
                              ) -> Tuple[Optional[PromptTemplate], str]:
    """Ask the advanced model to reword ``current`` given a batch of misclassified cases."""
    texts = current.text_values()
    numbered = "\n".join(f"[text_section {i}]\n{v}" for i, v in enumerate(texts))
    cases = "\n\n".join(_render_error(e["ex"], current, e["conf"], e["kind"], limit) for e in errors)
    user = (
        f"ENVIRONMENT: {_describe_env(env)}\n\n"
        f"CURRENT SYSTEM PROMPT:\n{current.system or '(none)'}\n\n"
        f"CURRENT INSTRUCTION SECTIONS (there are exactly {len(texts)}; keep this count and order, and "
        f"keep every {{placeholder}} intact — data is inserted between them):\n{numbered}\n\n"
        f"CASES THE CURRENT PROMPT GOT WRONG (predicted vs TRUE):\n{cases}\n\n"
        "First, briefly summarize the systematic reasons these were misjudged. Then rewrite the SYSTEM "
        "prompt and the instruction sections to fix those failure modes — be specific and add decision "
        "rules, but do NOT reference any data that is not already provided, and do NOT change the "
        "required output format. Return ONLY a JSON object:\n"
        '{"lessons": "<short diagnosis>", "system": "<new system prompt>", '
        f'"text_sections": [<exactly {len(texts)} strings, same order>]}}'
    )
    try:
        reply = llm.chat(_OPT_SYSTEM, user, temperature=0.4, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001
        return None, f"optimizer call failed: {type(exc).__name__}: {exc}"
    obj = _extract_json(reply)
    if not obj:
        return None, "optimizer returned no parseable JSON"
    new_sections = obj.get("text_sections")
    new_system = obj.get("system", current.system)
    lessons = str(obj.get("lessons", ""))[:1000]
    if not isinstance(new_sections, list) or len(new_sections) != len(texts):
        return None, f"optimizer returned {len(new_sections) if isinstance(new_sections, list) else 'no'} sections (need {len(texts)})"
    try:
        candidate = current.with_text(str(new_system), [str(s) for s in new_sections])
    except ValueError as exc:
        return None, f"rebuild failed: {exc}"
    return candidate, lessons


@dataclass
class RoundLog:
    round: int
    score: float
    accepted: bool
    lessons: str
    metrics_summary: str


@dataclass
class OptimizeResult:
    env: str
    objective: str
    baseline: PromptTemplate
    baseline_metrics: Metrics
    best: PromptTemplate
    best_metrics: Metrics
    history: List[RoundLog] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.best.version != self.baseline.version or \
            self.best_metrics.score(self.objective) > self.baseline_metrics.score(self.objective)

    def report(self) -> Dict:
        return {
            "env": self.env,
            "objective": self.objective,
            "baseline_score": self.baseline_metrics.score(self.objective),
            "best_score": self.best_metrics.score(self.objective),
            "improved": self.improved,
            "baseline_metrics": self.baseline_metrics.summary(),
            "best_metrics": self.best_metrics.summary(),
            "history": [rl.__dict__ for rl in self.history],
        }


def optimize(env: str, examples: Sequence[LabeledExample], llm_verifier, llm_optimizer,
             cfg: Optional[OptimizerConfig] = None, *, log=print) -> OptimizeResult:
    """Run the reflection loop and return the best (≥ baseline) template."""
    cfg = cfg or OptimizerConfig()
    rng = random.Random(cfg.seed)
    obj = cfg.objective

    def _eval(tmpl: PromptTemplate) -> Metrics:
        return evaluate_template(tmpl, examples, llm_verifier, k=cfg.k,
                                 temperature=cfg.verifier_temperature, max_workers=cfg.max_workers)

    base = baseline(env)
    base_m = _eval(base)
    log(f"[{env}] baseline: {base_m.summary()}")
    log(f"[{env}] data: slices={slice_counts(examples)} labels={label_balance(examples)}")

    best, best_m = base, base_m
    best_score = best_m.score(obj)
    history = [RoundLog(0, best_score, True, "baseline", base_m.summary())]

    for r in range(1, cfg.rounds + 1):
        if not best_m.errors:
            log(f"[{env}] round {r}: no misclassifications on best — stopping.")
            break
        batch = _balanced_errors(best_m.errors, cfg.n_error_examples, rng)
        candidate, lessons = propose_improved_template(
            best, batch, llm_optimizer, env, limit=cfg.error_prompt_chars,
            max_tokens=cfg.optimizer_max_tokens)
        if candidate is None:
            log(f"[{env}] round {r}: no candidate ({lessons})")
            history.append(RoundLog(r, best_score, False, lessons, "n/a"))
            continue
        cand_m = _eval(candidate)
        cand_score = cand_m.score(obj)
        accept = cand_score > best_score + cfg.min_improve
        log(f"[{env}] round {r}: candidate {obj}={cand_score:.3f} vs best {best_score:.3f} "
            f"-> {'ACCEPT' if accept else 'reject'} | {cand_m.summary()}")
        history.append(RoundLog(r, cand_score, accept, lessons, cand_m.summary()))
        if accept:
            best, best_m, best_score = candidate, cand_m, cand_score

    log(f"[{env}] DONE: best {obj}={best_score:.3f} (baseline {base_m.score(obj):.3f}); "
        f"improved={best.version != base.version}")
    return OptimizeResult(env, obj, base, base_m, best, best_m, history)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def default_store() -> str:
    return os.path.join(os.path.dirname(__file__), "store")


def save_result(result: OptimizeResult, store_dir: Optional[str] = None) -> str:
    store_dir = store_dir or default_store()
    os.makedirs(store_dir, exist_ok=True)
    env = result.env
    with open(os.path.join(store_dir, f"{env}_baseline.json"), "w", encoding="utf-8") as fh:
        fh.write(result.baseline.to_json())
    with open(os.path.join(store_dir, f"{env}_best.json"), "w", encoding="utf-8") as fh:
        fh.write(result.best.to_json())
    with open(os.path.join(store_dir, f"{env}_report.json"), "w", encoding="utf-8") as fh:
        json.dump(result.report(), fh, indent=2, ensure_ascii=False)
    return store_dir


def load_best(env: str, store_dir: Optional[str] = None) -> Optional[PromptTemplate]:
    store_dir = store_dir or default_store()
    path = os.path.join(store_dir, f"{env}_best.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return PromptTemplate.from_dict(json.load(fh))


__all__ = ["OptimizerConfig", "OptimizeResult", "RoundLog", "optimize",
           "propose_improved_template", "save_result", "load_best", "default_store"]
