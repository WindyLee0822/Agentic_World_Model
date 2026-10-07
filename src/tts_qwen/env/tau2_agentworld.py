"""Task-grounded Qwen-AgentWorld support for tau2 airline and retail.

Only the simulated environment boundary lives here.  The official tau2 agent,
user simulator, orchestration, and committed evaluator remain unchanged.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from typing import Any, Dict, Iterable, Optional

from ..llm.agentworld import AgentWorldSession, build_agentworld_system_prompt


_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "your", "you", "want", "wish",
    "have", "has", "but", "not", "into", "same", "one", "all", "are", "was", "were",
    "agent", "customer", "order", "reservation", "received", "exchange", "cancel", "flight",
}
_CONFIDENCE_RE = re.compile(
    r"(?:success_probability|confidence|probability|score)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def tau2_agentworld_tool_docs(env) -> Dict[str, str]:
    """Return full native tool documentation without changing shared tau2 docs."""
    docs: Dict[str, str] = {}
    for toolkit in (getattr(env, "tools", None), getattr(env, "user_tools", None)):
        if toolkit is None:
            continue
        try:
            names = list(toolkit.get_tools())
        except Exception:
            names = []
        for name in names:
            fn = getattr(toolkit, name, None)
            doc = " ".join((getattr(fn, "__doc__", None) or "").split())
            if doc:
                docs[str(name)] = doc[:4000]
    docs["__agentworld__.verify_task"] = (
        "Judge whether the interaction is correctly resolved under the supplied domain policy in the "
        "CURRENT SIMULATED tau2 state and conversation. A clear policy-required refusal is successful "
        "even though the requested mutation is not performed; an unnecessary refusal or transfer is "
        "not. Do not treat the agent stopping as proof of success. Check every permitted requested state "
        "change, policy prerequisite, and required communication. Return exactly one JSON object "
        'with "success" (boolean), "success_probability" (0 to 1), and "reason" (string).'
    )
    return docs


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return str(value)


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[#A-Za-z0-9_@.-]+", text or "")
        if len(token.strip("#._-")) >= 3 and token.casefold() not in _STOPWORDS
    }


def _scalars(value: Any, depth: int = 0):
    if depth > 7:
        return
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        yield str(value)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _scalars(child, depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _scalars(child, depth + 1)


def build_tau2_initial_state(
    model,
    instruction: str,
    *,
    max_seed_records: int = 8,
    max_records: int = 30,
    max_record_chars: int = 7000,
    max_total_chars: int = 48000,
) -> Dict[str, Any]:
    """Retrieve task-relevant pristine records and close over referenced entity IDs."""
    terms = _terms(instruction)
    candidates = []
    for bucket, ids in model.id_sets.items():
        for entity_id in ids:
            row = model.read(bucket, entity_id)
            data = _dump(row)
            text = json.dumps(data, ensure_ascii=False, default=str).casefold()
            exact_id = str(entity_id).casefold() in terms or str(entity_id).casefold() in instruction.casefold()
            matches = len(terms.intersection(_terms(text)))
            if exact_id or matches >= 2:
                candidates.append((exact_id, matches, bucket, str(entity_id), data))
    candidates.sort(key=lambda item: (-int(item[0]), -item[1], item[2], item[3]))

    selected: Dict[tuple[str, str], Any] = {}
    buckets_with_exact = {bucket for exact, _, bucket, _, _ in candidates if exact}
    per_bucket: Dict[str, int] = {}
    seeds = []
    for exact, _, bucket, entity_id, data in candidates:
        if not exact and bucket in buckets_with_exact:
            continue
        if not exact and per_bucket.get(bucket, 0) >= 2:
            continue
        seeds.append((bucket, entity_id, data))
        per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
        if len(seeds) >= max_seed_records:
            break
    for bucket, entity_id, data in seeds:
        selected[(bucket, entity_id)] = data

    # Orders/reservations contain the exact user, product, and flight identifiers needed by later
    # reads. Follow those references without scanning based on candidate actions or gold actions.
    frontier = list(selected.items())
    for _ in range(2):
        new_frontier = []
        for _, data in frontier:
            for value in _scalars(data):
                for bucket, entity_id in model.matches(value):
                    key = (bucket, entity_id)
                    if key in selected or len(selected) >= max_records:
                        continue
                    child = _dump(model.read(bucket, entity_id))
                    selected[key] = child
                    new_frontier.append((key, child))
        frontier = new_frontier
        if not frontier or len(selected) >= max_records:
            break

    records: Dict[str, Any] = {}
    used = 0
    for (bucket, entity_id), data in selected.items():
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(raw) > max_record_chars:
            raw = raw[:max_record_chars] + "…"
        if used + len(raw) > max_total_chars:
            break
        try:
            records[f"{bucket}:{entity_id}"] = json.loads(raw)
        except json.JSONDecodeError:
            records[f"{bucket}:{entity_id}"] = raw
        used += len(raw)

    return {
        "records": records,
        "record_count": len(records),
        "task_conditioned": True,
        "complete_database": False,
    }


def build_tau2_agentworld_prompt(
    env,
    domain: str,
    instruction: str,
    initial_state: Dict[str, Any],
    *,
    template_file: str,
) -> str:
    prompt = build_agentworld_system_prompt(
        tau2_agentworld_tool_docs(env), template_file=template_file)
    context = {
        "domain": domain,
        "customer_request": instruction,
        # This is the same public domain policy supplied to the official agent. It is required for
        # causal transition validation and for recognizing policy-correct refusals/transfers.
        "domain_policy": str(env.get_policy()),
        "initial_state": initial_state,
    }
    return (
        prompt
        + "\n\n# tau2 Task and Initial Environment State\n"
        + "The following is a task-conditioned subset of the pristine airline/retail database. "
          "It is environment-only context, not a gold trajectory. Maintain all successful changes "
          "cumulatively, preserve native schemas, and do not assume an unlisted record is absent.\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
    )


def _parse_object(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start = text.find("{")
    if start >= 0:
        text = text[start:]
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {}


def parse_tau2_agentworld_confidence(raw: str) -> tuple[float, Dict[str, Any]]:
    parsed = _parse_object(raw)
    value: Any = next(
        (parsed[key] for key in ("success_probability", "confidence", "probability", "score")
         if key in parsed),
        None,
    )
    if value is None and isinstance(parsed.get("success"), bool):
        value = 1.0 if parsed["success"] else 0.0
    if value is None:
        match = _CONFIDENCE_RE.search(str(raw or ""))
        value = match.group(1) if match else 0.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence)), parsed


class Tau2AgentWorldSimulator:
    def __init__(
        self,
        session: AgentWorldSession,
        valid_tools: Iterable[str],
    ) -> None:
        self.session = session
        self.valid_tools = set(valid_tools)
        self.n_transition_calls = 0
        self.n_transition_failures = 0

    def predict(self, name: str, arguments: Dict[str, Any]) -> tuple[str, bool]:
        self.n_transition_calls += 1
        if name not in self.valid_tools:
            content = f"Error: unknown tau2 tool {name}"
            self.session.observe_tool(name, arguments, content)
            return content, True
        try:
            content = self.session.predict_tool(name, arguments)
        except Exception as exc:
            self.n_transition_failures += 1
            return f"Error: AgentWorld {type(exc).__name__}: {exc}", True
        upper = content.lstrip().upper()
        is_error = upper.startswith(("ERROR", "EXCEPTION", "TRACEBACK"))
        return content, is_error

    def verify(
        self,
        instruction: str,
        conversation: str,
        termination_reason: str,
    ) -> tuple[float, Dict[str, Any], str]:
        raw = self.session.predict_tool(
            "__agentworld__.verify_task",
            {
                "customer_request": instruction,
                "conversation": conversation,
                "termination_reason": termination_reason,
            },
            reference_context=(
                "Inspect the supplied domain policy, current simulated state, and conversation. "
                "Success means the agent resolved the request correctly under policy. If policy "
                "forbids the requested action, a clear accurate refusal is success and the absence "
                "of that state change is correct. Score every permitted subrequest and all required "
                "user-facing communication. Do not use agent termination alone as evidence."
            ),
            max_tokens=min(self.session.max_tokens, 16384),
        )
        confidence, parsed = parse_tau2_agentworld_confidence(raw)
        return confidence, parsed, raw


__all__ = [
    "Tau2AgentWorldSimulator",
    "build_tau2_agentworld_prompt",
    "build_tau2_initial_state",
    "parse_tau2_agentworld_confidence",
    "tau2_agentworld_tool_docs",
]
