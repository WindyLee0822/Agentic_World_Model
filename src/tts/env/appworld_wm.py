"""World-model interception for AppWorld with exact-read and predict-all modes.

The official ReAct code agent solves a task by emitting Python that calls ``apis.<app>.<endpoint>(...)``;
every such call funnels through the single chokepoint ``requester.request(_app_name, _api_name, **data)``
(injected into the execution shell).  This module patches that chokepoint so the agent interacts with a
**world-model simulator of the backing database** instead of mutating it:

* documented **read** endpoints (non-mutating HTTP methods such as ``GET``/``HEAD``/``OPTIONS``)
  execute exactly by default; under predict-all, recursive Ours grounds through the whitelisted
  read-only simulator tools and has the world model generate the result, while flat B1 predicts from
  fixed context;
* **write** endpoints (``POST`` / ``PATCH`` / ``PUT`` / ``DELETE``) are **predicted** by an LLM from the
  call args + the endpoint's ``response_schemas`` + the running call history — never executed, so the real
  backing DB is never written.
* the one rollout-control write, ``supervisor.complete_task``, is applied only to predictor-local logical
  state so the agent loop can stop without writing the real supervisor database.

This is the AppWorld analog of the τ² ``environment.get_response`` interception.  Because writes are
predicted (not executed), ``N`` candidate trajectories can be sampled without mutating the env; the
selected one is committed by re-executing its realized code on a fresh real world and scoring with
``AppWorld.evaluate()``.

Read-after-write (τ² logic): once an app has been written, a subsequent ``GET`` to it is **predicted**
by combining the exact *pre-write* real result with the write history (``predict_read``) — so reads
reflect the agent's prior writes instead of returning stale state.

AppWorld predictions use a bounded Python-only subagent protocol by default. ``PYTHON_READ:`` code emits
structured read requests; the host validates each endpoint as read-only, executes it, and copies results
back into the next turn. ``PYTHON_FINAL:`` code computes the final state prediction or verifier verdict.
Direct final output remains available for simple cases. Generated code receives copied JSON-like inputs
only and cannot access AppWorld, requester/database handles, files, network, credentials, or write APIs.
The older textual ``READ``/``LIST``/``CALL``/``TRACE`` loop remains available only as an explicit legacy
ablation via ``TTS_APPWORLD_WM_TOOL_MODE=text``.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..simulator import TransitionLog

_APPWORLD_SIMULATOR_SYSTEM = """You are a causal simulator for AppWorld APIs, not a task-solving
assistant. Reproduce the response that the real API would return for the current call.

Simulation rules, in priority order:
1. Follow API documentation, current arguments, grounded database evidence, frozen time, and prior
   applied transitions. Treat predicted success messages as observations, not proof of hidden state.
2. For an untouched read, first use the identical whitelisted read API when available, then generate
   the final response by copying its JSON value exactly. Do not infer a cleaner or more helpful answer.
3. For a write, validate required fields, authorization, target existence, current status, ownership,
   duplicate-state constraints, and financial limits. Rejected writes do not mutate logical state.
4. Preserve the documented response type and complete schema, including empty lists, nulls, ordering,
   error bodies, and unchanged fields. Never invent IDs, timestamps, records, or successful effects.
5. Task goals, evaluator checks, gold values, and reference data are audit context only. Never use
   them to coach the benchmark agent or fabricate the response that would make the task succeed.
Return only the requested API response or tool-protocol program, with no explanatory prose."""

#: control-plane reads that stay exact even after a predictor-local control transition.
_PASSTHROUGH_APPS = {"supervisor", "api_docs"}

#: login must execute for real even though it is a POST: the returned token authorizes subsequent exact
#: reads. AppWorld login validates existing credentials and creates a token but does not save task-app
#: records. Logout is deliberately not included because it blacklists a token (a real state mutation).
_PASSTHROUGH_APIS = {"login"}

#: HTTP methods that mutate the backing DB. Documented non-mutating methods execute exactly; missing
#: metadata is blocked fail-closed and never reaches the real requester.
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

#: AppWorld control-plane apps are present in the agent shell but do not count toward task app coverage.
_CONTROL_APPS = {"supervisor", "api_docs"}

# These writes can reject on live financial/precondition state even when their success response is
# only a status message. They must inspect grounded state instead of taking the cached optimistic path.
_REJECTION_SENSITIVE_WRITES = {
    ("amazon", "place_order"),
    ("venmo", "add_to_venmo_balance"),
    ("venmo", "approve_payment_request"),
    ("venmo", "create_payment_request"),
    ("venmo", "create_transaction"),
    ("venmo", "withdraw_from_venmo_balance"),
}

# Private AppWorld endpoints whose HTTP method is POST only to hide their data from the task agent.
# They are side-effect free and may be used by the WM subagent to simulate financial preconditions.
_WM_READ_ONLY_POST_APIS = {
    ("admin", "show_card_balance"),
    ("admin", "validate_card"),
}


def _available_app_names(world) -> Tuple[str, ...]:
    """Return the substantive AppWorld app catalog, excluding control-plane apps."""
    task = getattr(world, "task", None)
    descriptions = getattr(task, "app_descriptions", {}) or {}
    names = descriptions.keys() if hasattr(descriptions, "keys") else ()
    return tuple(
        str(name).strip()
        for name in names
        if str(name).strip() and str(name).strip() not in _CONTROL_APPS
    )


def _wall_clock() -> float:
    """Real elapsed-seconds clock that is **immune to AppWorld's freezegun time-freeze**.

    AppWorld runs the agent (and the WM predictions) inside ``world.execute`` under ``freezegun``, which
    patches ``time.monotonic``/``time.perf_counter`` to the frozen task wall-clock (~1.68e9).  The WM's
    rollout budget/hard-timeout are computed from the *real* clock when the predictor is created (outside
    the freeze), so using ``time.monotonic`` for the deadline checks inside the sandbox compares ~1.68e9
    against ``real+600`` and fires the timeout INSTANTLY — aborting every prediction.  ``clock_gettime``
    is NOT patched by freezegun, so it yields true elapsed time inside and outside the freeze alike.
    """
    try:
        return time.clock_gettime(time.CLOCK_MONOTONIC)
    except (AttributeError, OSError):  # non-POSIX fallback (freezegun not in play there anyway)
        return time.monotonic()


def _is_write(app_name: str, api_name: str) -> bool:
    """True unless metadata proves the endpoint is safe to execute as an exact read."""
    if api_name in _PASSTHROUGH_APIS:
        return False
    doc = _api_doc(app_name, api_name)
    if not doc:
        return True
    return str(doc.get("method") or "").upper() in _MUTATING_METHODS


def _is_strict_read(app_name: str, api_name: str) -> bool:
    """Whether the WM subagent may execute this API as a genuinely read-only grounding tool.

    Unlike agent routing, this deliberately does not exempt control-plane or auth endpoints: a POST
    such as ``supervisor.complete_task`` or ``login`` must never be callable by the predictive subagent.
    """
    doc = _api_doc(app_name, api_name)
    method = doc.get("method") if doc else None
    if not method:
        return False
    if (app_name, api_name) in _WM_READ_ONLY_POST_APIS:
        return True
    return str(method).upper() not in _MUTATING_METHODS


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("0", "false", "False")


def _wm_tool_mode() -> str:
    """Subagent tool protocol: Python-only by default; legacy text tools are opt-in."""
    value = str(os.environ.get("TTS_APPWORLD_WM_TOOL_MODE", "python")).strip().lower()
    return "text" if value in ("text", "legacy", "read") else "python"


def _deterministic_dirty_reads_enabled() -> bool:
    return _env_enabled("TTS_APPWORLD_DETERMINISTIC_DIRTY_READS", False)


class WMHardTimeout(RuntimeError):
    """Raised inside a WM prediction once the rollout's hard wall-clock deadline passes.

    It deliberately propagates out of the agent's (possibly non-terminating) code cell so the cell
    ends and the rollout can finish, instead of spinning forever on a ``while True`` API loop that
    AppWorld's own per-cell SIGALRM timeout fails to abort.
    """


def _api_docs(app_name: str) -> Dict[str, Dict[str, Any]]:
    from appworld.requester import _get_api_name_to_doc  # type: ignore

    try:
        return _get_api_name_to_doc(app_name, include_private_apis=True) or {}
    except Exception:
        return {}


def _api_doc(app_name: str, api_name: str) -> Dict[str, Any]:
    return _api_docs(app_name).get(api_name, {}) or {}


def _api_method(app_name: str, api_name: str) -> str:
    return str(_api_doc(app_name, api_name).get("method") or "GET").upper()


def _success_schema(app_name: str, api_name: str) -> Any:
    schemas = _api_doc(app_name, api_name).get("response_schemas") or {}
    return schemas.get("success", schemas)


def _failure_schema(app_name: str, api_name: str) -> Any:
    """The endpoint's documented FAILURE response shape (every AppWorld write ships one), or None."""
    schemas = _api_doc(app_name, api_name).get("response_schemas") or {}
    return schemas.get("failure")


class WMPredictedHTTPError(Exception):
    """A write the world model predicts the *real* API would REJECT.

    The real ``requester`` raises ``Exception(f"Response status code is {code}:\\n{text}")`` for a
    non-200 response (requester.py ``raise_if_failure``); the agent's API client surfaces that as a
    traceback it must react to.  When the WM infers (from the call history) that a write would be
    rejected — e.g. approving an already-approved request, following an already-followed artist — it
    raises this so the agent sees the SAME failure it would on the real env, instead of a misleading
    fake success.  ``install_wm_requester`` translates it into the real-format plain ``Exception``
    (or, when ``raise_on_failure=False``, returns ``body``) to match the real requester exactly.
    """

    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = int(status_code)
        self.body = body
        payload = json.dumps(body, default=str, separators=(",", ":"))
        super().__init__(f"Response status code is {self.status_code}:\n{payload}")


#: response fields that carry no readable state (status/message-only writes).  Used only by
#: ``_synth_success`` (the LLM-failure fallback); writes are otherwise always LLM-predicted.
_TRIVIAL_FIELDS = {"message", "success", "status", "detail", "ok", "code"}


def _is_trivial_schema(schema: Any) -> bool:
    return isinstance(schema, dict) and bool(schema) and set(schema.keys()).issubset(_TRIVIAL_FIELDS)


def _synth_success(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return {"message": "Success."}
    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if k == "message":
            out[k] = "Success."
        elif isinstance(v, str):
            out[k] = "ok" if k == "status" else v
        elif k in ("success", "ok"):
            out[k] = True
        else:
            out[k] = v
    return out


def _is_success_verdict(resp: Optional[str]) -> bool:
    """True iff the judge's reply ends on ``VERDICT: SUCCESS`` (the last verdict token wins)."""
    up = (resp or "").upper()
    return up.rfind("SUCCESS") > up.rfind("FAIL")


def _python_context_inventory(context: Dict[str, Any]) -> List[Dict[str, str]]:
    """Describe copied variables for repair without exposing their values."""
    inventory: List[Dict[str, str]] = []
    for key, value in sorted(context.items()):
        inventory.append({
            "variable": f"data[{key!r}]",
            "value_type": type(value).__name__,
        })
        if key == "reads" and isinstance(value, list):
            for index, record in enumerate(value):
                if not isinstance(record, dict) or "result" not in record:
                    continue
                inventory.append({
                    "variable": f"data['reads'][{index}]['result']",
                    "value_type": type(record["result"]).__name__,
                })
    return inventory


def _parse_json(text: str) -> Optional[Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None


_PYTHON_PREFIX = "PYTHON:"
_PYTHON_READ_PREFIX = "PYTHON_READ:"
_PYTHON_FINAL_PREFIX = "PYTHON_FINAL:"


def _extract_prefixed_python(text: str, prefix: str) -> Optional[str]:
    """Return code when a response starts with one explicit Python mode prefix."""
    stripped = (text or "").lstrip()
    if not stripped.startswith(prefix):
        return None
    code = stripped[len(prefix):].strip()
    # A model occasionally emits both modes in one response. Execute only the first
    # program so a valid read request is not made syntactically invalid by trailing code.
    code = re.split(
        rf"(?m)^\s*(?:{re.escape(_PYTHON_READ_PREFIX)}|"
        rf"{re.escape(_PYTHON_FINAL_PREFIX)}|{re.escape(_PYTHON_PREFIX)})\s*",
        code,
        maxsplit=1,
    )[0].rstrip()
    if code.startswith("```"):
        first_line, _, remainder = code.partition("\n")
        if first_line.strip().lower() in ("```", "```python", "```py"):
            code = remainder
        if code.rstrip().endswith("```"):
            code = code.rstrip()[:-3]
    return code.strip()


def _extract_python_program(text: str) -> Optional[str]:
    """Return final-prediction code; ``PYTHON:`` remains a backward-compatible alias."""
    code = _extract_prefixed_python(text, _PYTHON_FINAL_PREFIX)
    if code is not None:
        return code
    return _extract_prefixed_python(text, _PYTHON_PREFIX)


def _extract_python_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Classify a response as a Python read request, Python final, or non-Python output."""
    code = _extract_prefixed_python(text, _PYTHON_READ_PREFIX)
    if code is not None:
        return "read", code
    code = _extract_python_program(text)
    if code is not None:
        return "final", code
    return None, None


def _contains_legacy_text_tool(text: str) -> bool:
    """Whether a response tries to invoke the opt-in textual tool protocol."""
    return any(
        line.strip().upper().startswith(("READ ", "LIST ", "CALL ", "TRACE "))
        for line in (text or "").splitlines()
    )


def _redact_python_value(value: Any) -> Any:
    """Copy prediction inputs while removing credentials that computation never needs."""
    if isinstance(value, dict):
        return {
            str(key): ("<redacted>" if str(key).lower() in {
                "access_token", "api_key", "password", "secret", "token",
            } else _redact_python_value(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_python_value(item) for item in value]
    return value


def _bounded_json_text(value: Any, limit: int) -> str:
    """Serialize only the bounded prefix needed for a repair prompt."""
    chunks = []
    length = 0
    encoder = json.JSONEncoder(default=str, sort_keys=True)
    for chunk in encoder.iterencode(value):
        remaining = limit - length
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            length = limit
            break
        chunks.append(chunk)
        length += len(chunk)
    text = "".join(chunks)
    return text + ("\n... bounded context truncated ..." if length >= limit else "")


def _json_size_hint(value: Any, stop_after: int) -> int:
    """Return serialized size, stopping once it is known to exceed ``stop_after``."""
    length = 0
    encoder = json.JSONEncoder(default=str, sort_keys=True)
    for chunk in encoder.iterencode(value):
        length += len(chunk)
        if length > stop_after:
            return length
    return length


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUERY_RESULT_CHARS = 5000


def _parse_api_command(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Parse ``app.api {json}`` or ``app api {json}`` from a subagent command."""
    text = str(text or "").strip()
    if not text:
        return "", None
    first, _, tail = text.partition(" ")
    if "." in first:
        name = first
        payload = tail.strip()
    else:
        api, sep, payload = tail.strip().partition(" ")
        if not api:
            return "", None
        name = f"{first}.{api}"
        payload = payload.strip() if sep else ""
    args: Any = {}
    if payload:
        try:
            args = json.loads(payload)
        except Exception:
            return name, None
    return name, args if isinstance(args, dict) else None


class AppWorldWMQuery:
    """Read-only grounding interface for the AppWorld world-model subagent.

    Clean reads return exact backing-DB results.  A read of an app changed by predicted writes returns
    the exact pristine result plus the ordered transition trace, rather than recursively invoking
    ``predict_read`` from inside another prediction.
    """

    def __init__(self, real_request, predictor: "AppWorldWMPredictor") -> None:
        self._request = real_request
        self._predictor = predictor
        self.reads = 0
        self.lists = 0
        self.calls = 0
        self.traces = 0

    @staticmethod
    def _split_name(name: str) -> Optional[Tuple[str, str]]:
        if "." not in str(name or ""):
            return None
        app, api = str(name).split(".", 1)
        if not _NAME_RE.fullmatch(app) or not _NAME_RE.fullmatch(api):
            return None
        return app, api

    @staticmethod
    def _render(value: Any) -> str:
        return json.dumps(
            _redact_python_value(value), default=str, sort_keys=True
        )[:_QUERY_RESULT_CHARS]

    def _read_api(self, name: str, args: Optional[Dict[str, Any]]) -> str:
        split = self._split_name(name)
        if split is None:
            return "ERROR: expected an API name in the form app.api"
        if args is None:
            return "ERROR: API arguments must be one JSON object"
        app, api = split
        if not _is_strict_read(app, api):
            return f"ERROR: {app}.{api} is a write API; the world-model subagent may only read"
        if any(key in args for key in ("client", "raise_on_failure", "show", "track")):
            return "ERROR: transport-only requester arguments are not allowed"
        request_args = self._predictor.hydrate_read_args(app, api, args)
        try:
            pristine = self._request(app, api, raise_on_failure=False, **request_args)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        if (app in self._predictor.dirty_apps
                and app not in _PASSTHROUGH_APPS and api not in _PASSTHROUGH_APIS):
            return self._render({
                "logical_state": "pristine result plus predicted transitions",
                "pre_write_result": pristine,
                "predicted_transitions": self._predictor.trace_text(app),
            })
        return self._render(pristine)

    def read(self, name: str, args: Optional[Dict[str, Any]]) -> str:
        self.reads += 1
        return self._read_api(name, args)

    def read_raw(self, name: str, args: Optional[Dict[str, Any]]) -> Any:
        """Return untouched pristine evidence while preserving read-only validation errors."""
        self.reads += 1
        split = self._split_name(name)
        if split is None:
            return "ERROR: expected an API name in the form app.api"
        if args is None:
            return "ERROR: API arguments must be one JSON object"
        app, api = split
        if not _is_strict_read(app, api):
            return f"ERROR: {app}.{api} is a write API; the world-model subagent may only read"
        if any(key in args for key in ("client", "raise_on_failure", "show", "track")):
            return "ERROR: transport-only requester arguments are not allowed"
        request_args = self._predictor.hydrate_read_args(app, api, args)
        try:
            return self._request(app, api, raise_on_failure=False, **request_args)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    def read_value(self, name: str, args: Optional[Dict[str, Any]]) -> Any:
        """Return one exact pristine read requested by the simulator, never to generated code."""
        self.reads += 1
        split = self._split_name(name)
        if split is None or args is None:
            raise ValueError("expected a read API name and one argument object")
        app, api = split
        if not _is_strict_read(app, api):
            raise ValueError(f"{app}.{api} is not an allowed read")
        request_args = self._predictor.hydrate_read_args(app, api, args)
        return self._request(app, api, raise_on_failure=False, **request_args)

    def list(self, target: str) -> str:
        """List apps, or list the read-only APIs exposed by one app."""
        self.lists += 1
        target = str(target or "").strip()
        if target.lower() in ("apps", "*"):
            try:
                result = self._request(
                    "api_docs", "show_app_descriptions", raise_on_failure=False)
                return self._render(result)
            except Exception as exc:
                return f"ERROR: {type(exc).__name__}: {exc}"
        if not _NAME_RE.fullmatch(target):
            return "ERROR: LIST expects an app name or 'apps'"
        rows = []
        for api, doc in sorted(_api_docs(target).items()):
            if not _is_strict_read(target, api):
                continue
            method = str(doc.get("method") or "GET").upper()
            desc = str(doc.get("description") or "").replace("\n", " ")[:240]
            rows.append(f"{target}.{api} [{method}] - {desc}".rstrip(" -"))
        return "\n".join(rows) if rows else f"(no read-only APIs found for {target})"

    def call(self, name: str, args: Optional[Dict[str, Any]]) -> str:
        self.calls += 1
        return self._read_api(name, args)

    def trace(self, resource: str) -> str:
        self.traces += 1
        resource = str(resource or "").strip()
        if not resource:
            return "ERROR: TRACE expects an app or app.api"
        return self._predictor.trace_text(resource)


def _parse_agentic(text: str):
    """Parse one AppWorld subagent turn into grounding commands or a terminal answer."""
    reads: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    lists: List[str] = []
    calls: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    traces: List[str] = []
    terminal = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        command = stripped.upper()
        if command.startswith("PREDICT:"):
            terminal = ("predict", stripped[len("PREDICT:"):].strip())
        elif command.startswith("VERDICT:"):
            answer = command[len("VERDICT:"):].strip()
            terminal = ("verdict", "SUCCESS" if answer.startswith("SUCCESS") else "FAILURE")
        elif command.startswith("READ "):
            reads.append(_parse_api_command(stripped[5:]))
        elif command.startswith("LIST "):
            lists.append(stripped[5:].strip().strip("\"'"))
        elif command.startswith("CALL "):
            calls.append(_parse_api_command(stripped[5:]))
        elif command.startswith("TRACE "):
            traces.append(stripped[6:].strip().strip("\"'"))
    if terminal is not None:
        return [], [], [], [], terminal
    return reads, lists, calls, traces, None


def run_agentic_read_loop(
        complete: Callable[[str, int], str],
        query: AppWorldWMQuery,
        header: str,
        final_instr: str,
        *,
        max_turns: int = 6,
        max_tokens: int = 1500,
        known_out: Optional[Dict[str, str]] = None,
):
    """Run the bounded AppWorld ``READ``/``LIST``/``CALL``/``TRACE`` grounding loop."""
    known: Dict[str, str] = {}

    def finish(result):
        if known_out is not None:
            known_out.clear()
            known_out.update(known)
        return result

    protocol = (
        "You may inspect the CURRENT LOGICAL AppWorld state before answering. The real backing DB "
        "remains pristine; reads of a modified app return its exact pre-write result plus the ordered "
        "predicted transitions that define the logical post-write state. Each turn output one or more:\n"
        "  READ <app.api> {json args}  (read logical state through a read API)\n"
        "  LIST <app>                  (list read-only APIs; LIST apps lists available apps)\n"
        "  CALL <app.api> {json args}  (invoke a read-only API)\n"
        "  TRACE <app-or-app.api>      (inspect predicted transitions)\n"
        f"OR, when enough evidence is gathered, output:\n  {final_instr}\n"
        f"Use only the minimum evidence needed; you have at most {max_turns} read turns."
    )
    for _turn in range(max_turns):
        facts = "\n".join(f"{key}: {value}" for key, value in known.items()) or "(nothing read yet)"
        response = complete(
            f"{header}\n\n{protocol}\n\nFacts gathered so far:\n{facts}\n\nYour move:",
            max_tokens,
        )
        reads, lists, calls, traces, terminal = _parse_agentic(response)
        if terminal is not None:
            return finish(terminal)
        acted = False
        for target in lists[:3]:
            known[f"list {target}"] = query.list(target)
            acted = True
        for name, args in reads[:4]:
            known[f"read {name}({args})"] = query.read(name, args)
            acted = True
        for name, args in calls[:4]:
            known[f"call {name}({args})"] = query.call(name, args)
            acted = True
        for resource in traces[:3]:
            known[f"trace {resource}"] = query.trace(resource)
            acted = True
        if not acted:
            return finish(("predict", (response or "").strip()))

    facts = "\n".join(f"{key}: {value}" for key, value in known.items()) or "(nothing read)"
    response = complete(
        f"{header}\n\nFacts gathered:\n{facts}\n\nYou are out of read turns. {final_instr}",
        max_tokens,
    )
    _, _, _, _, terminal = _parse_agentic(response)
    return finish(terminal if terminal is not None else ("predict", (response or "").strip()))


class AppWorldWMPredictor:
    """LLM predictor for AppWorld API responses and a success verdict.

    * ``predict_write`` returns a structurally-valid response dict (guided by the endpoint's success
      schema) so the agent's downstream code can consume it, and marks the written app *dirty*.
    * ``predict_read`` predicts clean reads from fixed/tool-gathered evidence and dirty reads from the
      pre-write result plus write history.
    * ``note_read`` records exact (clean) reads so predictions stay consistent with what the agent saw.
    """

    MAX_TURNS = 6

    def __init__(self, chat_client, time_budget_s: Optional[float] = None,
                 max_llm_calls: Optional[int] = None, hard_timeout_s: Optional[float] = None) -> None:
        self._chat = chat_client
        self._query: Optional[AppWorldWMQuery] = None
        self._flat_system_prompt: Optional[str] = None
        self._flat_context_data: Optional[Dict[str, Any]] = None
        self.available_apps: Tuple[str, ...] = ()
        self.transition_log = TransitionLog()
        # The real env freezes "now" to a fixed per-task datetime (freezegun on ``task.datetime``),
        # so every server-generated timestamp (created_at/added_at/updated_at/...) is DETERMINISTIC and
        # identical between the WM rollout and the real commit world. Surfacing it to the predictor lets
        # a predicted write/read reproduce those timestamps EXACTLY instead of the LLM guessing a date
        # (the τ²-style fixed-value approach). Set by ``install_wm_requester`` from ``world.task``.
        self.frozen_datetime: Optional[str] = None
        self.history: List[Tuple[str, str, dict, Any]] = []
        self.write_history: List[Tuple[str, str, dict, Any]] = []
        self.dirty_apps: set = set()
        self.simulated_task_completion: Optional[Dict[str, Any]] = None
        self.last_percheck_breakdown: Optional[Dict[str, Any]] = None
        # Set of (app, api, frozen-arg-key) for writes that SUCCEEDED in this rollout. A repeat of the
        # same state-changing write to the same target is the most common, history-inferable rejection
        # ("already approved/followed/liked/...") — detecting it routes the prediction through the
        # rejection-aware path so the WM can reproduce the real API's failure instead of a fake success.
        self.applied_writes: set = set()
        self.unresolved_write_failures: set = set()
        self.n_completion_deferrals = 0
        self.n_writes = 0
        self.first_write_seen = False
        self.n_reads = 0
        self.n_predicted_reads = 0
        self.n_dirty_reads = 0
        self.n_llm_calls = 0
        self.n_python_calls = 0
        self.n_python_failures = 0
        self.n_python_forced = 0
        self.n_python_mode_misses = 0
        self._python_logged_errors = 0
        self.python_disabled = False
        self.program_only = False
        # Per-rollout budget. A single agent code block can issue thousands of write API calls in a
        # loop, and in WM mode every write/dirty-read is a (slow) LLM prediction — so one pathological
        # rollout can run for HOURS and stall the whole sweep. Once the SOFT budget (wall-clock or
        # call-count) is spent, further writes/dirty-reads skip the LLM and use the instant fallback
        # (structurally-valid synth for writes, exact pre-write state for reads) so the rollout still
        # makes progress. A real agent can still emit an UNBOUNDED loop (e.g. ``while True:
        # apis.spotify.previous_song(...)``) that never terminates even with instant fallbacks — a
        # pure-CPU spin that AppWorld's own per-cell SIGALRM timeout fails to abort in the
        # IPython+freezegun+process-pool context. The HARD timeout backstops that: once it passes,
        # predict_write/predict_read RAISE, which propagates out of the agent's loop, ends the runaway
        # code cell, and lets the rollout finish. Tune via TTS_APPWORLD_WM_TIME_BUDGET_S (soft, secs),
        # TTS_APPWORLD_WM_MAX_LLM_CALLS (soft, count) and TTS_APPWORLD_WM_HARD_TIMEOUT_S (hard, secs);
        # each <=0 disables. Normal rollouts finish well under these and never hit them.
        if time_budget_s is None:
            time_budget_s = _env_float("TTS_APPWORLD_WM_TIME_BUDGET_S", 240.0)
        if max_llm_calls is None:
            max_llm_calls = _env_int("TTS_APPWORLD_WM_MAX_LLM_CALLS", 0)
        if hard_timeout_s is None:
            hard_timeout_s = _env_float("TTS_APPWORLD_WM_HARD_TIMEOUT_S", 600.0)
        self.time_budget_s = time_budget_s
        self.max_llm_calls = max_llm_calls
        self.hard_timeout_s = hard_timeout_s
        self._deadline = (_wall_clock() + time_budget_s) if time_budget_s and time_budget_s > 0 else None
        self._hard_deadline = (_wall_clock() + hard_timeout_s) if hard_timeout_s and hard_timeout_s > 0 else None
        self.budget_exhausted = False
        self.hard_timed_out = False
        self.warmed_up = False
        # Backstop for runaway in-cell loops (e.g. ``while True: apis.x.write(...)``): once the SOFT
        # budget is spent each write/read returns the instant fallback, but an UNBOUNDED loop can still
        # call it tens of millions of times before the wall-clock HARD deadline fires (observed ~19M
        # fallback calls / ~360s wasted). Cap the number of post-budget fallback predictions so such a
        # loop is broken in milliseconds instead. Tune via TTS_APPWORLD_WM_MAX_FALLBACK_CALLS (<=0 off).
        self._fallback_calls = 0
        self.max_fallback_calls = _env_int("TTS_APPWORLD_WM_MAX_FALLBACK_CALLS", 50000)

    def bind_query_interface(self, query: AppWorldWMQuery) -> None:
        """Give observation and success predictors read-only access to logical AppWorld state."""
        self._query = query

    @property
    def uses_fixed_context(self) -> bool:
        """Whether this predictor is using B1's pre-action fixed context."""
        return self._flat_system_prompt is not None

    @property
    def flat_system_prompt(self) -> Optional[str]:
        return self._flat_system_prompt

    def configure_flat_context(
        self,
        *,
        task_text: str,
        tool_docs: Dict[str, Any],
        reference_checklist: Optional[str] = None,
        reference_checks: Optional[Tuple[List[str], str]] = None,
        fixed_state: Optional[Dict[str, Any]] = None,
        predict_all_actions: bool = True,
    ) -> str:
        """Prepare AppWorld B1's action-independent simulator prompt."""
        checks, gold = reference_checks or ([], "")
        context = {
            "task": task_text,
            "tool_documentation": tool_docs,
            "evaluation_checklist": reference_checklist or "",
            "evaluation_checks": list(checks),
            "gold_values": gold,
            "fixed_initial_state": fixed_state or {},
        }
        self._flat_context_data = copy.deepcopy(context)
        rendered = _bounded_json_text(
            _redact_python_value(context),
            max(8_000, _env_int("TTS_APPWORLD_B1_CONTEXT_CHARS", 60_000)),
        )
        read_contract = (
            "every task-app read and write observation is predicted during sampling; clean reads "
            "use only this fixed context; dirty reads additionally apply ordered predicted writes; "
            if predict_all_actions
            else
            "clean reads are executed exactly outside this prompt; writes are never executed during "
            "sampling; dirty reads are inferred from the fixed initial context and ordered predicted "
            "writes; "
        )
        self._flat_system_prompt = (
            f"{_APPWORLD_SIMULATOR_SYSTEM}\n\n"
            "You are the fixed-context B1 world model for AppWorld. The complete simulator context "
            "below was selected before any candidate agent action was observed. Use only this fixed "
            "context plus the candidate action/history supplied later in the user message. Do not "
            "request or perform READ, LIST, CALL, TRACE, or PYTHON_READ operations, and do not assume "
            "access to any additional environment state.\n\n"
            f"Simulate AppWorld faithfully: {read_contract}"
            "and success means every native evaluator criterion is satisfied with no missing or extra "
            "mutation. Match documented response schemas and failure behavior exactly.\n\n"
            f"FIXED PRE-ACTION CONTEXT:\n{rendered}"
        )
        return self._flat_system_prompt

    def bind_app_catalog(self, available_apps) -> None:
        """Record the substantive AppWorld catalog for rollout activity diagnostics."""
        self.available_apps = tuple(dict.fromkeys(
            str(app) for app in (available_apps or ())
            if str(app) and str(app) not in _CONTROL_APPS
        ))

    @property
    def query_interface(self) -> Optional[AppWorldWMQuery]:
        return self._query

    def active_task_apps(self) -> Tuple[str, ...]:
        """Catalog apps with at least one substantive read or write in this rollout."""
        active = {
            app
            for app, api, _args, _result in self.history
            if app in self.available_apps and api not in _PASSTHROUGH_APIS
        }
        return tuple(app for app in self.available_apps if app in active)

    def _task_app_activity_text(self) -> str:
        active = self.active_task_apps()
        return (
            f"- active apps ({len(active)}): {', '.join(active) or '(none)'}"
        )

    def prepare_python_tool(self) -> bool:
        """Start the shared pure-computation worker before entering AppWorld's execution sandbox."""
        if not _env_enabled("TTS_APPWORLD_WM_PYTHON", True):
            return False
        try:
            from ..llm.python_tool import shared_python_tool

            prepared = shared_python_tool().prepare()
            self.python_disabled = not prepared
            return prepared
        except Exception:
            self.python_disabled = True
            return False

    def _check_runaway(self) -> None:
        """Raise once too many post-budget fallback predictions happen (unbounded agent loop)."""
        if self.max_fallback_calls and self._fallback_calls >= self.max_fallback_calls:
            self.hard_timed_out = True
            raise WMHardTimeout(
                f"AppWorld WM rollout made {self._fallback_calls} post-budget fallback predictions "
                f"(likely a non-terminating agent loop); aborting the code cell."
            )

    def warm_up(self) -> bool:
        """Pre-fetch the LLM auth token OUTSIDE the AppWorld sandbox so in-sandbox predictions work.

        AppWorld runs the agent's code (and therefore every WM ``predict_write``/``predict_read``) inside
        ``world.execute`` under a ``SafetyGuard`` that **no-ops ``subprocess.run``** (and friends).  The
        Azure ``gcr`` auth mode fetches its bearer token via ``AzureCliCredential`` (which shells out to
        ``az`` through ``subprocess``); blocked inside the sandbox, that returns ``None`` and the LLM call
        fails — so every prediction would silently fall back to the ``_synth_success`` template instead of
        a faithful LLM response.  ``azure-identity`` caches the token, so making one warm-up call *here*
        (outside the sandbox) populates the cache and lets the in-sandbox requests reuse it with no
        ``subprocess`` call.  Call this once before the agent loop.  Best-effort: never raises.
        """
        try:
            self._chat.complete([{"role": "user", "content": "ok"}], temperature=0.0, max_tokens=1)
            self.warmed_up = True
        except Exception:
            self.warmed_up = False
        return self.warmed_up

    def _check_hard_deadline(self) -> None:
        """Raise once the rollout's HARD wall-clock deadline passes, to break a runaway agent loop."""
        if self._hard_deadline is not None and _wall_clock() > self._hard_deadline:
            self.hard_timed_out = True
            raise WMHardTimeout(
                f"AppWorld WM rollout exceeded hard timeout {self.hard_timeout_s:.0f}s "
                f"(likely a non-terminating agent loop); aborting the code cell."
            )

    def _over_budget(self) -> bool:
        """True once this rollout's SOFT WM-prediction budget (wall-clock or call count) is spent."""
        if self.budget_exhausted:
            return True
        if self._deadline is not None and _wall_clock() > self._deadline:
            self.budget_exhausted = True
        elif self.max_llm_calls and self.n_llm_calls >= self.max_llm_calls:
            self.budget_exhausted = True
        return self.budget_exhausted

    def _history_text(self, k: int = 8) -> str:
        out = []
        for app, api, args, res in self.history[-k:]:
            a = json.dumps(args, default=str)[:160]
            r = json.dumps(res, default=str)[:200]
            out.append(f"{app}.{api}({a}) -> {r}")
        return "\n".join(out) or "(no prior calls)"

    def _writes_text(self, app: str) -> str:
        out = []
        for w_app, api, args, res in self.write_history:
            if w_app != app:
                continue
            a = json.dumps(args, default=str)[:200]
            r = json.dumps(res, default=str)[:200]
            out.append(f"{w_app}.{api}({a}) -> {r}")
        return "\n".join(out) or "(no writes to this app)"

    def _frozen_time_text(self) -> str:
        """A prompt line pinning server-generated timestamps to the env's frozen 'now', or ''."""
        if not self.frozen_datetime:
            return ""
        return (
            f"The server clock is FROZEN at '{self.frozen_datetime}'. Use EXACTLY this value for any "
            f"server-generated timestamp field (e.g. created_at/added_at/updated_at/timestamp).\n"
        )

    def _complete(self, prompt: str, *, max_tokens: int, temperature: float = 0.0,
                  budgeted: bool = True) -> str:
        if budgeted:
            self._check_hard_deadline()
            if self._over_budget():
                return ""
        try:
            self.n_llm_calls += 1
            messages = [{"role": "user", "content": prompt}]
            if self._flat_system_prompt is not None:
                messages.insert(
                    0, {"role": "system", "content": self._flat_system_prompt})
            else:
                messages.insert(
                    0, {"role": "system", "content": _APPWORLD_SIMULATOR_SYSTEM})
            return (self._chat.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ) or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _python_protocol(
        context: Dict[str, Any],
        *,
        required: bool,
        context_chars: int,
        output_kind: str = "json",
        allow_reads: bool = False,
    ) -> str:
        keys = ", ".join(sorted(map(str, context))) or "(none)"
        routing = (
            f"The copied structured input is larger than the direct-output reliability threshold "
            f"(at least {context_chars} characters). You MUST use PYTHON_FINAL for the final "
            "prediction; do not "
            "start with JSON or prose."
            if required else
            "If the prediction is straightforward and you are confident, answer directly in the "
            "requested format. Use PYTHON when exact computation is safer."
        )
        if output_kind == "verdict":
            output_instruction = (
                'End by calling `print_json({"verdict": "SUCCESS"})` or '
                '`print_json({"verdict": "FAIL"})`; alternatively, print exactly '
                '`VERDICT: SUCCESS` or `VERDICT: FAIL` as the final line.'
            )
        elif output_kind == "status":
            output_instruction = (
                'End by calling `print_json({"status": "MET"})`, '
                '`print_json({"status": "UNMET"})`, or '
                '`print_json({"status": "UNKNOWN"})`; alternatively, print exactly '
                '`STATUS: MET`, `STATUS: UNMET`, or `STATUS: UNKNOWN` as the final line.'
            )
        else:
            output_instruction = (
                "End with `print(value)` or `print_json(value)` for the complete JSON prediction; "
                "container values printed with `print` are serialized as bounded JSON."
            )
        structured_sources = [
            f"`data[{key!r}]`" for key in ("writes", "history") if key in context
        ]
        structured_hint = ""
        if structured_sources:
            structured_hint = (
                f" Entries in {' and '.join(structured_sources)} use "
                "`{'app': <app>, 'api': <endpoint-only>, 'name': '<app>.<endpoint>', 'args': {...}, "
                "'result': ...}`; compare `api` to names such as `delete_from_cart`, or `name` to "
                "`amazon.delete_from_cart`. Copied `result` values are intentionally untouched and "
                "may be JSON strings; before indexing one, use "
                "`value = json_value(record['result'])`."
            )
        gold_hint = (
            " Gold blobs are under `data['gold']['sections']` (for example, "
            "`json_value(data['gold']['sections']['private_data'])`); a section can itself be an "
            "untouched JSON string, and the original text is in `data['gold']['raw']`."
            if "gold" in context else
            ""
        )
        read_instruction = ""
        if allow_reads:
            read_instruction = (
                "\nTo gather environment evidence, start a turn with `PYTHON_READ:` followed by a "
                "program. It receives the current copied `data` and must print exactly "
                '`{"queries":[{"app":"<app>","api":"<read_api>","args":{...}}, ...]}`. '
                "The host validates every endpoint as documented read-only, executes at most a bounded "
                "number of reads, copies their results into `data['reads']`/`data['facts']`, and asks "
                "you for another turn. Generated code never receives a requester or database handle. "
                "For protected reads, omit `access_token` or use `\"<redacted>\"`; the host injects a "
                "previously obtained real login token without exposing it to the program. After a read, "
                "if its untouched JSON envelope is uncertain, use another `PYTHON_READ:` program to "
                "inspect it and print `{\"queries\":[],\"facts\":...}`. Those facts must be computed "
                "inside that generated program and are shown on the next turn; the host never parses or "
                "summarizes the read value for you.\n"
            )
        exact_read_instruction = ""
        exact_call = context.get("exact_read_call")
        if context.get("grounded_clean_read_required"):
            exact_read_instruction = (
                "\nEXACT GROUNDED CLEAN-READ PROTOCOL (overrides every other routing option):\n"
                "- The host already executed the proven read-only API and copied its exact result into "
                "`data['grounded_clean_read']`.\n"
                "- Return a PYTHON_FINAL program with exactly this shape (no indexing or parsing): "
                "`PYTHON_FINAL:\\nvalue = data['grounded_clean_read']\\nprint(value)`.\n"
                "- Do not request another read or replace the variable with a value transcribed into "
                "the program. Only generated Python may process the copied result.\n"
            )
        elif (
            allow_reads
            and context.get("pristine_read_required")
            and isinstance(exact_call, dict)
        ):
            exact_read_instruction = (
                "\nREQUIRED DIRTY-READ GROUNDING PROTOCOL:\n"
                "- Your first response MUST be PYTHON_READ and request exactly the pristine API in "
                f"`data['exact_read_call']`: "
                f"{json.dumps(exact_call, ensure_ascii=True, sort_keys=True)}.\n"
                "- On the next turn, decode `data['reads'][-1]['result']` with `json_value`, then use "
                "generated Python to apply every relevant ordered entry in `data['writes']` before "
                "printing the logical post-write result. Do NOT copy the pristine result unchanged.\n"
                "- For Spotify player state: `next_song` advances one queue position with wraparound, "
                "`previous_song` moves back one position with wraparound, and `play_music` selects its "
                "`queue_position` when supplied and marks playback active. Queue removals shift later "
                "positions; derive current-song and `is_current` fields from the resulting cursor.\n"
                "- Do not transcribe evidence into source code. All inspection and transition "
                "application must happen inside the generated program.\n"
            )
        elif (
            allow_reads
            and context.get("exact_read_required")
            and isinstance(exact_call, dict)
        ):
            exact_read_instruction = (
                "\nEXACT CLEAN-READ PROTOCOL (overrides direct-answer routing):\n"
                "- Your first response MUST be PYTHON_READ and request exactly the API in "
                f"`data['exact_read_call']`: "
                f"{json.dumps(exact_call, ensure_ascii=True, sort_keys=True)}.\n"
                "- Do not answer or request any other evidence before that read.\n"
                "- On the next turn, return a PYTHON_FINAL program that processes "
                "`data['reads'][-1]['result']` directly. For an exact copy, use exactly: "
                "`value = data['reads'][-1]['result']; print(value)`.\n"
                "- Do not transcribe the read result into source code or return it directly.\n"
            )
        return (
            "\n\nPURE COMPUTATION OPTION:\n"
            f"{routing} For exact counting, filtering, date arithmetic, set comparison, or a long "
            f"structured transformation, use Python.{read_instruction}{exact_read_instruction}"
            "To compute the final answer, make "
            "the FIRST non-whitespace characters of your response exactly `PYTHON_FINAL:` followed by a Python "
            "program. The program receives a JSON-like dictionary named `data` with the prediction "
            f"inputs under these keys: {keys}; `data['facts']` contains any host-mediated read "
            f"evidence gathered by the subagent.{structured_hint}{gold_hint} It may use ordinary "
            "list/dict/set operations plus "
            "`parse_date`, `parse_datetime`, `date_add`, `month_start`, `month_end`, `timedelta`, "
            "`json_value`, `json_records`, `json_loads`, `json_dumps`, `deepcopy`, `Counter`, and "
            "`defaultdict`. `json_value(value)` leaves non-strings "
            "untouched and decodes bounded JSON-string layers, so call it before indexing an "
            "untouched read result or gold section. `json_records(value)` extracts the record list "
            "from common nested JSON response envelopes. Keep programs concise (normally under 25 "
            "lines), avoid comments, and print on every branch. The safe `json`, `datetime`, `copy`, "
            "and `collections` modules may be imported; "
            "all other imports, files, network, environment access, and benchmark APIs are unavailable. "
            "`data` is already a local variable; use it directly rather than calling `globals()` or "
            "`locals()` (their safe compatibility forms reveal only `data`). The listed helpers are "
            f"already defined, so do not reimplement them unless necessary. {output_instruction} "
            "Print ONLY the complete requested prediction; never abbreviate, summarize, or omit records. "
            "The legacy `PYTHON:` prefix is accepted as an alias for `PYTHON_FINAL:`."
        )

    @staticmethod
    def _normalize_python_output(output: str, output_kind: str) -> Optional[str]:
        """Validate verifier stdout and discard diagnostics outside the final contract value."""
        if output_kind == "json":
            payload = _parse_json(output)
            return output if payload is not None else None
        for line in reversed((output or "").splitlines()):
            stripped = line.strip().upper()
            if output_kind == "verdict":
                match = re.fullmatch(r"VERDICT\s*[:=]\s*(SUCCESS|FAIL|FAILURE)", stripped)
                if match:
                    return (
                        "VERDICT: SUCCESS"
                        if match.group(1) == "SUCCESS"
                        else "VERDICT: FAIL"
                    )
            elif output_kind == "status":
                match = re.fullmatch(r"STATUS\s*[:=]\s*(MET|UNMET|UNKNOWN)", stripped)
                if match:
                    return f"STATUS: {match.group(1)}"
        payload = None
        # Verifier programs sometimes print a diagnostic worksheet before the final contract object.
        # Accept only the last standalone JSON object, then discard every other byte of stdout.
        for line in reversed((output or "").splitlines()):
            try:
                candidate = json.loads(line.strip())
            except Exception:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            payload = _parse_json(output)
        if payload is None:
            return None
        if not isinstance(payload, dict):
            return None
        if output_kind == "verdict":
            verdict = str(payload.get("verdict", "")).upper()
            if verdict == "SUCCESS":
                return "VERDICT: SUCCESS"
            if verdict in ("FAIL", "FAILURE"):
                return "VERDICT: FAIL"
            success = payload.get("success")
            if isinstance(success, bool):
                return "VERDICT: SUCCESS" if success else "VERDICT: FAIL"
            return None
        if output_kind == "status":
            status = str(payload.get("status", "")).upper()
            if status in ("MET", "UNMET", "UNKNOWN"):
                return f"STATUS: {status}"
            return None
        return None

    def _run_python_prediction(
        self,
        code: str,
        context: Dict[str, Any],
        *,
        allow_bare_container_output: bool = True,
    ):
        from ..llm.python_tool import PythonToolResult, shared_python_tool

        if self.python_disabled:
            return PythonToolResult(error="Python worker is unavailable", fatal=True)
        self.n_python_calls += 1
        result = shared_python_tool().run(
            code,
            _redact_python_value(context),
            timeout_s=_env_float("TTS_APPWORLD_WM_PYTHON_TIMEOUT_S", 1.0),
            max_input_chars=_env_int("TTS_APPWORLD_WM_PYTHON_MAX_INPUT_CHARS", 200_000),
            max_output_chars=_env_int("TTS_APPWORLD_WM_PYTHON_MAX_OUTPUT_CHARS", 20_000),
            max_code_chars=_env_int("TTS_APPWORLD_WM_PYTHON_MAX_CODE_CHARS", 12_000),
            max_nodes=_env_int("TTS_APPWORLD_WM_PYTHON_MAX_AST_NODES", 2_000),
            max_steps=_env_int("TTS_APPWORLD_WM_PYTHON_MAX_STEPS", 100_000),
            allow_bare_container_output=allow_bare_container_output,
        )
        if result.error:
            self.n_python_failures += 1
            if (
                _env_enabled("TTS_APPWORLD_WM_PYTHON_LOG_ERRORS", False)
                and self._python_logged_errors < 20
            ):
                print(
                    f"[appworld-python-error] {result.error}\n"
                    f"[appworld-python-code] {code[:800]}",
                    flush=True,
                )
                self._python_logged_errors += 1
        if result.fatal:
            # A replacement process cannot be safely started from inside AppWorld's ``world.execute``
            # sandbox. Keep the current rollout on direct LLM predictions after helper loss; the next
            # rollout prepares a fresh shared worker before entering the sandbox.
            self.python_disabled = True
        return result

    def _resolve_python_prediction(
        self,
        response: str,
        *,
        prompt: str,
        context: Dict[str, Any],
        max_tokens: int,
        temperature: float,
        budgeted: bool,
        python_required: bool,
        output_kind: str = "json",
    ) -> str:
        code = _extract_python_program(response)
        if code is None:
            if _extract_prefixed_python(response, _PYTHON_READ_PREFIX) is not None:
                return ""
            if not python_required:
                return self._normalize_python_output(response, output_kind) or response
            self.n_python_mode_misses += 1
            response = self._complete(
                f"{prompt}\n\nYour previous response incorrectly used direct output. The structured "
                "input is too large for reliable direct regeneration. Return ONLY a PYTHON_FINAL: program "
                "that computes and prints the complete prediction.",
                max_tokens=max_tokens,
                temperature=temperature,
                budgeted=budgeted,
            )
            code = _extract_python_program(response)
            if code is None:
                return ""
        result = self._run_python_prediction(code, context)
        result_error = result.error
        normalized = None
        if result_error is None:
            normalized = self._normalize_python_output(result.output, output_kind)
        if result_error is None and normalized is None:
            output_label = "JSON" if output_kind == "json" else output_kind
            result_error = f"program output is not a valid {output_label} prediction"
            self.n_python_failures += 1
        if result_error is None:
            return normalized or ""
        if not _env_enabled("TTS_APPWORLD_WM_PYTHON_REPAIR", True):
            return ""

        if self.python_disabled:
            repair_instruction = (
                "The Python helper is unavailable for the rest of this rollout. Return the prediction "
                "DIRECTLY in the requested format; do not return PYTHON again."
            )
        else:
            repair_instruction = (
                "Return one corrected PYTHON_FINAL program; direct output is not allowed for this long input."
                if python_required else
                "Return either the direct prediction or one corrected PYTHON_FINAL program. Do not use "
                "PYTHON_READ during final-output repair. Do not use unavailable imports, files, network, "
                "or benchmark APIs."
            )
        repair_context_cap = max(
            2_000, _env_int("TTS_APPWORLD_WM_PYTHON_REPAIR_CONTEXT_CHARS", 12_000))
        repair_inventory = _bounded_json_text(
            _python_context_inventory(context), repair_context_cap)
        failed_code = code
        repair_error = result_error
        attempts = _env_int("TTS_APPWORLD_WM_PYTHON_REPAIR_ATTEMPTS", 2)
        for attempt in range(attempts):
            repair_prompt = (
                f"{prompt}\n\nYour previous PYTHON program failed safely with:\n{repair_error}\n\n"
                f"FAILED PROGRAM:\n{failed_code[:4_000]}\n\n"
                "AVAILABLE COPIED VARIABLES (names and types only; inspect values in generated "
                f"code):\n{repair_inventory}\n\n{repair_instruction} This is repair attempt "
                f"{attempt + 1}/{attempts}. Prefer the shortest valid program, call `json_value` "
                "before indexing an untouched string variable, and do not explain the failure."
            )
            repaired = self._complete(
                repair_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                budgeted=budgeted,
            )
            repaired_code = _extract_python_program(repaired)
            if repaired_code is None:
                if _extract_prefixed_python(repaired, _PYTHON_READ_PREFIX) is not None:
                    repair_error = "PYTHON_READ is unavailable during final-output repair"
                    failed_code = repaired
                    continue
                if python_required:
                    repair_error = "the required PYTHON_FINAL repair response was invalid"
                    failed_code = repaired
                    continue
                return self._normalize_python_output(repaired, output_kind) or repaired
            if self.python_disabled:
                return ""
            repaired_result = self._run_python_prediction(repaired_code, context)
            if repaired_result.error:
                repair_error = repaired_result.error
                failed_code = repaired_code
                continue
            normalized = self._normalize_python_output(
                repaired_result.output, output_kind)
            if normalized is not None:
                return normalized
            self.n_python_failures += 1
            repair_error = (
                f"repaired program output is not a valid "
                f"{'JSON' if output_kind == 'json' else output_kind} prediction"
            )
            failed_code = repaired_code
        return ""

    @staticmethod
    def _parse_python_read_payload(
        output: str,
    ) -> Optional[Tuple[List[Dict[str, Any]], Any]]:
        """Parse read requests and optional facts computed by generated Python."""
        payload = _parse_json(output)
        if not isinstance(payload, dict):
            return None
        queries = payload.get("queries")
        if not isinstance(queries, list):
            return None
        computed_facts = payload.get("facts")
        parsed = []
        for query in queries:
            if not isinstance(query, dict):
                continue
            app = str(query.get("app", ""))
            api = str(query.get("api", ""))
            args = query.get("args") or {}
            if not _NAME_RE.fullmatch(app) or not _NAME_RE.fullmatch(api):
                continue
            if not isinstance(args, dict):
                continue
            parsed.append({"app": app, "api": api, "args": args})
        if not parsed and computed_facts is None:
            return None
        return parsed, computed_facts

    def _python_read_loop(
        self,
        prompt: str,
        final_instr: str,
        *,
        context: Dict[str, Any],
        read_fn: Callable[[str, str, dict], Any],
        max_tokens: int,
        temperature: float,
        budgeted: bool,
        python_required: bool,
        output_kind: str,
    ) -> str:
        """Run bounded Python-emitted read rounds, followed by a direct or Python final answer."""
        reads = list(context.get("reads") or [])
        base_facts = context.get("facts")
        facts = dict(base_facts) if isinstance(base_facts, dict) else {}
        max_turns = max(1, _env_int("TTS_APPWORLD_WM_PYTHON_READ_TURNS", self.MAX_TURNS))
        max_reads = max(1, _env_int("TTS_APPWORLD_WM_PYTHON_READS_PER_TURN", 4))
        protocol = self._python_protocol(
            context,
            required=python_required,
            context_chars=_json_size_hint(_redact_python_value(context), 2_000),
            output_kind=output_kind,
            allow_reads=True,
        )
        model_prompt = f"{prompt}{protocol}\n\nFINAL OUTPUT FORMAT:\n{final_instr}"

        def current_context() -> Dict[str, Any]:
            return {**context, "reads": list(reads), "facts": dict(facts)}

        for turn in range(max_turns):
            read_variables = [
                {
                    **{key: value for key, value in record.items() if key != "result"},
                    "variable": f"data['reads'][{index}]['result']",
                    "value_type": type(record.get("result")).__name__,
                }
                for index, record in enumerate(reads)
            ]
            visible_reads = _bounded_json_text(
                _redact_python_value(read_variables),
                max(2_000, _env_int("TTS_APPWORLD_WM_PYTHON_READ_CONTEXT_CHARS", 12_000)),
            )
            visible_facts = _bounded_json_text(
                _redact_python_value(facts),
                max(2_000, _env_int("TTS_APPWORLD_WM_PYTHON_READ_CONTEXT_CHARS", 12_000)),
            )
            response = self._complete(
                f"{model_prompt}\n\nCOPIED READ RESULTS SO FAR:\n{visible_reads or '[]'}\n\n"
                f"TOOL FACTS AND ERRORS:\n{visible_facts or '{}'}\n\n"
                f"Python tool turn {turn + 1}/{max_turns}:",
                max_tokens=max_tokens,
                temperature=temperature,
                budgeted=budgeted,
            )
            mode, code = _extract_python_action(response)
            if mode == "read" and code is not None:
                result = self._run_python_prediction(
                    code, current_context(), allow_bare_container_output=True)
                if result.error:
                    facts[f"python_read_error_{turn + 1}"] = result.error
                    if result.fatal:
                        return ""
                    continue
                payload = self._parse_python_read_payload(result.output)
                if payload is None:
                    self.n_python_failures += 1
                    facts[f"python_read_error_{turn + 1}"] = (
                        "PYTHON_READ output must be one JSON object with a queries list and either "
                        "at least one valid query or a generated facts value."
                    )
                    continue
                requests, computed_facts = payload
                if computed_facts is not None:
                    facts[f"python_computed_facts_{turn + 1}"] = computed_facts
                for request in requests[:max_reads]:
                    app = request["app"]
                    api = request["api"]
                    args = request["args"]
                    try:
                        read_result = read_fn(app, api, args)
                    except Exception as exc:
                        read_result = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                    record = {
                        "app": app,
                        "api": api,
                        "name": f"{app}.{api}",
                        "args": args,
                        "result": read_result,
                    }
                    reads.append(record)
                continue
            if mode == "final":
                return self._resolve_python_prediction(
                    response,
                    prompt=model_prompt,
                    context=current_context(),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    budgeted=budgeted,
                    python_required=python_required,
                    output_kind=output_kind,
                )
            if _contains_legacy_text_tool(response):
                facts[f"python_tool_error_{turn + 1}"] = (
                    "READ/LIST/CALL/TRACE are disabled. Use PYTHON_READ to request evidence."
                )
                continue
            return self._resolve_python_prediction(
                response,
                prompt=model_prompt,
                context=current_context(),
                max_tokens=max_tokens,
                temperature=temperature,
                budgeted=budgeted,
                python_required=python_required,
                output_kind=output_kind,
            )

        response = self._complete(
            f"{model_prompt}\n\nYou are out of read turns. Return the final answer directly or with "
            "`PYTHON_FINAL:`; do not request more reads.",
            max_tokens=max_tokens,
            temperature=temperature,
            budgeted=budgeted,
        )
        mode, _code = _extract_python_action(response)
        if mode == "read" or _contains_legacy_text_tool(response):
            return ""
        return self._resolve_python_prediction(
            response,
            prompt=model_prompt,
            context=current_context(),
            max_tokens=max_tokens,
            temperature=temperature,
            budgeted=budgeted,
            python_required=python_required,
            output_kind=output_kind,
        )

    def _predict(self, prompt: str, final_instr: str, *, max_tokens: int,
                 temperature: float = 0.0, budgeted: bool = True,
                 python_context: Optional[Dict[str, Any]] = None,
                 python_output_kind: str = "json",
                 use_query: bool = True,
                 python_read_fn: Optional[Callable[[str, str, dict], Any]] = None,
                 python_force: bool = False) -> str:
        """Predict directly or through the bounded tool-using subagent when it is bound."""
        if self.uses_fixed_context:
            use_query = False
            python_read_fn = None
        tool_mode = "python" if self.program_only else _wm_tool_mode()
        use_python = (
            python_context is not None
            and _env_enabled("TTS_APPWORLD_WM_PYTHON", True)
            and not self.python_disabled
        )
        if self.program_only and not use_python:
            raise RuntimeError(
                "program-only AppWorld prediction requires the restricted Python worker")
        python_required = False
        context_chars = 0
        if use_python:
            # Direct dirty-read prompts expose at most ~2500 characters of the pristine result, plus
            # individually capped args/history. Force code below that truncation boundary so direct
            # mode is used only when the model actually sees the complete structured input.
            threshold = max(
                500, _env_int("TTS_APPWORLD_WM_PYTHON_FORCE_CHARS", 2_000))
            context_chars = _json_size_hint(
                _redact_python_value(python_context), threshold)
            # Long state regeneration is fail-closed because truncation corrupts the prediction.
            # Verifier code remains optional: mapping natural-language requirements to evidence is
            # semantic reasoning, so forcing code there can create a systematic failure bias.
            python_required = bool(
                python_force
                or (
                    python_output_kind == "json"
                    and context_chars > threshold
                )
            )
            if python_required:
                self.n_python_forced += 1

        read_executor = python_read_fn
        if read_executor is None and self._query is not None and use_query:
            raw_reader = getattr(self._query, "read_raw", None)
            if not callable(raw_reader):
                raw_reader = getattr(self._query, "read_value", None)
            if callable(raw_reader):
                read_executor = lambda app, api, args: raw_reader(
                    f"{app}.{api}", args)
            else:
                read_executor = lambda app, api, args: self._query.read(
                    f"{app}.{api}", args)
        if use_python and tool_mode == "python" and read_executor is not None:
            output = self._python_read_loop(
                prompt,
                final_instr,
                context=python_context or {},
                read_fn=read_executor,
                max_tokens=max_tokens,
                temperature=temperature,
                budgeted=budgeted,
                python_required=python_required,
                output_kind=python_output_kind,
            )
            if self.program_only and not output:
                raise RuntimeError(
                    "program-only AppWorld prediction did not produce a valid Python result")
            return output

        model_prompt = prompt
        if use_python:
            model_prompt += self._python_protocol(
                python_context,
                required=python_required,
                context_chars=context_chars,
                output_kind=python_output_kind,
            )
        if self._query is None or not use_query or tool_mode != "text":
            response = self._complete(
                model_prompt, max_tokens=max_tokens, temperature=temperature, budgeted=budgeted)
            if not use_python:
                return response
            output = self._resolve_python_prediction(
                response,
                prompt=model_prompt,
                context={**python_context, "facts": {}},
                max_tokens=max_tokens,
                temperature=temperature,
                budgeted=budgeted,
                python_required=python_required,
                output_kind=python_output_kind,
            )
            if self.program_only and not output:
                raise RuntimeError(
                    "program-only AppWorld prediction did not produce a valid Python result")
            return output
        known: Dict[str, str] = {}
        terminal = run_agentic_read_loop(
            lambda text, budget: self._complete(
                text, max_tokens=budget, temperature=temperature, budgeted=budgeted),
            self._query,
            model_prompt,
            final_instr,
            max_turns=self.MAX_TURNS,
            max_tokens=max_tokens,
            known_out=known,
        )
        response = (terminal[1] if terminal else "").strip()
        if not use_python:
            return response
        return self._resolve_python_prediction(
            response,
            prompt=model_prompt,
            context={**python_context, "facts": known},
            max_tokens=max_tokens,
            temperature=temperature,
            budgeted=budgeted,
            python_required=python_required,
            output_kind=python_output_kind,
        )

    def note_read(self, app: str, api: str, args: dict, result: Any) -> None:
        self.n_reads += 1
        self.history.append((app, api, dict(args), result))

    def hydrate_read_args(self, app: str, api: str, args: dict) -> dict:
        """Inject a prior real login token host-side without exposing it to generated Python."""
        hydrated = dict(args or {})
        doc = _api_doc(app, api)
        needs_token = any(
            isinstance(parameter, dict) and parameter.get("name") == "access_token"
            for parameter in (doc.get("parameters") or [])
        )
        if not needs_token:
            return hydrated
        supplied = hydrated.get("access_token")
        if supplied not in (None, "", "<redacted>"):
            return hydrated
        for h_app, h_api, _h_args, result in reversed(self.history):
            if h_app != app or h_api != "login" or not isinstance(result, dict):
                continue
            token = result.get("access_token")
            if token:
                hydrated["access_token"] = token
                break
        return hydrated

    def trace_text(self, resource: str) -> str:
        """Render predicted AppWorld transitions for an app or one app API."""
        resource = str(resource or "").strip()
        entries = self.transition_log.entries(resource)
        if entries:
            return self.transition_log.render(resource)
        app = resource.split(".", 1)[0]
        return self._writes_text(app)

    def _record_applied_write(self, app: str, api: str, args: dict, result: Any,
                              key: tuple) -> Any:
        self.history.append((app, api, dict(args), result))
        self.write_history.append((app, api, dict(args), result))
        self.dirty_apps.add(app)
        self.applied_writes.add(key)
        self.unresolved_write_failures.discard(
            self._write_intent_key(app, api, args))
        self.transition_log.record(
            {"app": app, "api": api, "args": dict(args)},
            json.dumps(result, default=str),
            [app, f"{app}.{api}"],
            applied=True,
        )
        return result

    def _record_rejected_write(self, app: str, api: str, args: dict, body: Any) -> None:
        self.unresolved_write_failures.add(self._write_intent_key(app, api, args))
        self.transition_log.record(
            {"app": app, "api": api, "args": dict(args)},
            json.dumps(body, default=str),
            [app, f"{app}.{api}"],
            applied=False,
        )

    @staticmethod
    def _find_id_record(value: Any, id_field: str, target_id: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, dict):
            if str(value.get(id_field)) == str(target_id):
                return value
            for item in value.values():
                found = AppWorldWMPredictor._find_id_record(item, id_field, target_id)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for item in value:
                found = AppWorldWMPredictor._find_id_record(item, id_field, target_id)
                if found is not None:
                    return found
        return None

    def _venmo_payment_request_amount(self, payment_request_id: Any) -> Optional[float]:
        for app, api, _args, result in reversed(self.history):
            if app != "venmo" or "payment_request" not in api:
                continue
            record = self._find_id_record(
                result, "payment_request_id", payment_request_id)
            if record is not None and isinstance(record.get("amount"), (int, float)):
                return float(record["amount"])
        return None

    def _venmo_payment_card(self, payment_card_id: Any) -> Optional[Dict[str, Any]]:
        for app, api, _args, result in reversed(self.history):
            if app != "venmo" or "payment_card" not in api:
                continue
            record = self._find_id_record(result, "payment_card_id", payment_card_id)
            if record is not None:
                return record
        if self._query is None or self.uses_fixed_context:
            return None
        try:
            cards = self._query.read_value("venmo.show_payment_cards", {})
        except Exception:
            return None
        return self._find_id_record(cards, "payment_card_id", payment_card_id)

    def _venmo_balance(self, pristine_result: Any = None) -> Optional[float]:
        balance = (
            float(pristine_result["venmo_balance"])
            if isinstance(pristine_result, dict)
            and isinstance(pristine_result.get("venmo_balance"), (int, float))
            else None
        )
        for app, api, _args, result in self.history:
            if app == "venmo" and api == "show_venmo_balance" and isinstance(result, dict):
                value = result.get("venmo_balance")
                if isinstance(value, (int, float)):
                    balance = float(value)
                    break
        if balance is None and self._query is not None and not self.uses_fixed_context:
            try:
                result = self._query.read_value("venmo.show_venmo_balance", {})
            except Exception:
                result = None
            if isinstance(result, dict) and isinstance(
                    result.get("venmo_balance"), (int, float)):
                balance = float(result["venmo_balance"])
        if balance is None:
            return None
        for app, api, args, _result in self.write_history:
            if app != "venmo":
                continue
            amount = args.get("amount")
            if not isinstance(amount, (int, float)):
                if api == "approve_payment_request":
                    amount = self._venmo_payment_request_amount(
                        args.get("payment_request_id"))
                if not isinstance(amount, (int, float)):
                    continue
            if api == "add_to_venmo_balance":
                balance += float(amount)
            elif api == "withdraw_from_venmo_balance":
                balance -= float(amount)
            elif api == "create_transaction" and args.get("payment_card_id") is None:
                balance -= float(amount)
            elif api == "approve_payment_request" and args.get("payment_card_id") is None:
                balance -= float(amount)
        return balance

    def _predict_venmo_dirty_read(
        self, api: str, args: dict, pre_write_result: Any
    ) -> Optional[Any]:
        """Apply known Venmo transitions exactly for high-value post-write reads."""
        if api == "show_venmo_balance":
            balance = self._venmo_balance(pre_write_result)
            if balance is not None:
                return {"venmo_balance": round(balance, 2)}

        payment_card_writes = {
            "add_payment_card",
            "delete_payment_card",
            "update_payment_card",
        }
        if api in {"show_payment_card", "show_payment_cards"} and not any(
            app == "venmo" and write_api in payment_card_writes
            for app, write_api, _args, _result in self.write_history
        ):
            return copy.deepcopy(pre_write_result)

        if api not in {"show_received_payment_requests", "show_sent_payment_requests"}:
            return None
        if not isinstance(pre_write_result, list):
            return None
        if any(
            app == "venmo" and write_api in {
                "create_payment_request",
                "update_payment_request",
            }
            for app, write_api, _args, _result in self.write_history
        ):
            return None

        rows = copy.deepcopy(pre_write_result)
        approved = {
            str(write_args.get("payment_request_id"))
            for app, write_api, write_args, _result in self.write_history
            if app == "venmo" and write_api == "approve_payment_request"
        }
        denied = {
            str(write_args.get("payment_request_id"))
            for app, write_api, write_args, _result in self.write_history
            if app == "venmo" and write_api == "deny_payment_request"
        }
        deleted = {
            str(write_args.get("payment_request_id"))
            for app, write_api, write_args, _result in self.write_history
            if app == "venmo" and write_api == "delete_payment_request"
        }
        status = args.get("status")
        result = []
        for row in rows:
            request_id = str(row.get("payment_request_id"))
            if request_id in deleted:
                continue
            if request_id in approved:
                row["approved_at"] = self.frozen_datetime or row.get("approved_at") or "now"
                if status in {"pending", "denied"}:
                    continue
            if request_id in denied:
                row["denied_at"] = self.frozen_datetime or row.get("denied_at") or "now"
                if status in {"pending", "approved"}:
                    continue
            if status == "approved" and request_id not in approved and not row.get("approved_at"):
                continue
            if status == "denied" and request_id not in denied and not row.get("denied_at"):
                continue
            if status == "pending" and (
                request_id in approved
                or request_id in denied
                or row.get("approved_at")
                or row.get("denied_at")
            ):
                continue
            result.append(row)
        return result

    def _venmo_card_balance(self, payment_card_id: Any) -> Optional[float]:
        card = self._venmo_payment_card(payment_card_id)
        if card is None or self._query is None or self.uses_fixed_context:
            return None
        required = ("owner_name", "card_number", "expiry_year", "expiry_month", "cvv_number")
        if any(card.get(key) is None for key in required):
            return None
        try:
            result = self._query.read_value(
                "admin.show_card_balance",
                {key: card[key] for key in required},
            )
        except Exception:
            return None
        if not isinstance(result, dict) or not isinstance(result.get("balance"), (int, float)):
            return None
        balance = float(result["balance"])
        for app, api, args, _result in self.write_history:
            if app != "venmo" or str(args.get("payment_card_id")) != str(payment_card_id):
                continue
            amount = args.get("amount")
            if not isinstance(amount, (int, float)):
                if api == "approve_payment_request":
                    amount = self._venmo_payment_request_amount(
                        args.get("payment_request_id"))
                if not isinstance(amount, (int, float)):
                    continue
            if api in {
                "add_to_venmo_balance",
                "create_transaction",
                "approve_payment_request",
            }:
                balance -= float(amount)
            elif api == "withdraw_from_venmo_balance":
                balance += float(amount)
        return balance

    def _predict_grounded_venmo_funding(
        self, api: str, args: dict, schema: Any
    ) -> Optional[Tuple[Any, Optional[Tuple[int, Any]]]]:
        amount = args.get("amount")
        if api == "approve_payment_request":
            amount = self._venmo_payment_request_amount(
                args.get("payment_request_id"))
        if not isinstance(amount, (int, float)):
            return None
        amount = round(float(amount), 2)
        payment_card_id = args.get("payment_card_id")
        if api == "add_to_venmo_balance":
            available = self._venmo_card_balance(payment_card_id)
            failure_message = (
                f"Your payment card does not have ${amount:.2f} to add to your Venmo balance."
            )
        elif api == "withdraw_from_venmo_balance":
            available = self._venmo_balance()
            failure_message = (
                f"Your venmo balance is not enough to withdraw ${amount:.2f} from it."
            )
        elif api in {"create_transaction", "approve_payment_request"}:
            available = (
                self._venmo_card_balance(payment_card_id)
                if payment_card_id is not None
                else self._venmo_balance()
            )
            if api == "approve_payment_request":
                failure_message = (
                    "Insufficient money in the payment card."
                    if payment_card_id is not None
                    else "Insufficient money in the Venmo balance."
                )
            else:
                funding_source = "payment card" if payment_card_id is not None else "Venmo balance"
                failure_message = (
                    f"Your {funding_source} does not have ${amount:.2f} to make this transaction."
                )
        else:
            return None
        if available is None:
            return None
        if amount > available + 1e-9:
            body = {"message": failure_message}
            return body, (422, body)
        synthetic_id = 1_000_000 + len(self.write_history) + 1
        if api == "add_to_venmo_balance":
            result = {
                "message": "Money added to Venmo balance.",
                "bank_transfer_id": synthetic_id,
            }
        elif api == "withdraw_from_venmo_balance":
            result = {
                "message": "Money withdrawn from Venmo balance.",
                "bank_transfer_id": synthetic_id,
            }
        elif api == "create_transaction":
            result = {"message": "Sent money.", "transaction_id": synthetic_id}
        else:
            result = {"message": "Payment request approved."}
        if isinstance(schema, dict):
            result = {key: result.get(key, value) for key, value in schema.items()}
        return result, None

    def _find_product_snapshot(self, product_id: Any) -> Dict[str, Any]:
        """Find the richest exact pre-write product record previously returned to the agent."""
        target = str(product_id)
        best: Dict[str, Any] = {}

        def visit(value: Any) -> None:
            nonlocal best
            if isinstance(value, dict):
                if str(value.get("product_id")) == target and len(value) > len(best):
                    best = copy.deepcopy(value)
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        for h_app, _h_api, _h_args, result in self.history:
            if h_app == "amazon":
                visit(result)
        return best

    def _predict_amazon_cart(self, pre_write_result: Any) -> Optional[Dict[str, Any]]:
        """Apply known Amazon cart transitions exactly instead of asking the LLM to reconstruct them."""
        if not isinstance(pre_write_result, dict):
            return None
        item_key = "cart_items" if "cart_items" in pre_write_result else (
            "cart_entries" if "cart_entries" in pre_write_result else None
        )
        if item_key is None or not isinstance(pre_write_result.get(item_key), list):
            return None

        cart_writes = [
            (api, args)
            for app, api, args, _result in self.write_history
            if app == "amazon" and api in {
                "clear_cart",
                "add_product_to_cart",
                "move_product_from_wish_list_to_cart",
                "delete_product_from_cart",
                "update_product_quantity_in_cart",
                "add_gift_wrapping_to_product",
                "remove_gift_wrapping_from_product",
                "place_order",
            }
        ]
        if not cart_writes:
            return None

        items = [copy.deepcopy(item) for item in pre_write_result[item_key]]

        def index_of(product_id: Any) -> Optional[int]:
            target = str(product_id)
            return next(
                (index for index, item in enumerate(items)
                 if str(item.get("product_id")) == target),
                None,
            )

        for api, args in cart_writes:
            product_id = args.get("product_id")
            index = index_of(product_id) if product_id is not None else None
            if api in ("clear_cart", "place_order"):
                items = []
            elif api == "delete_product_from_cart":
                if index is not None:
                    items.pop(index)
            elif api in ("add_product_to_cart", "move_product_from_wish_list_to_cart"):
                if args.get("clear_cart_first"):
                    items = []
                    index = None
                quantity = args.get("quantity", 1)
                if index is not None:
                    items[index]["quantity"] = items[index].get("quantity", 0) + quantity
                else:
                    product = self._find_product_snapshot(product_id)
                    items.append({
                        "product_id": product_id,
                        "product_name": product.get("product_name") or product.get("name"),
                        "delivery_days": product.get("delivery_days"),
                        "quantity": quantity,
                        "price": product.get("price"),
                        "gift_wrap_quantity": 0,
                    })
            elif api == "update_product_quantity_in_cart" and index is not None:
                quantity = args.get("quantity", 0)
                if quantity == 0:
                    items.pop(index)
                else:
                    items[index]["quantity"] = quantity
            elif api == "add_gift_wrapping_to_product" and index is not None:
                items[index]["gift_wrap_quantity"] = args.get("quantity", 1)
            elif api == "remove_gift_wrapping_from_product" and index is not None:
                items[index]["gift_wrap_quantity"] = 0

        result = copy.deepcopy(pre_write_result)
        result[item_key] = items
        if item_key == "cart_items":
            prices_known = all(isinstance(item.get("price"), (int, float)) for item in items)
            if not items:
                result.update({
                    "base_cost": 0,
                    "delivery_fee": 0.0,
                    "gift_wrap_fee": 0.0,
                    "tax": 0.0,
                    "total_cost": 0.0,
                })
            elif prices_known:
                base_cost = round(sum(
                    float(item["price"]) * float(item.get("quantity", 0))
                    for item in items
                ), 2)
                gift_wrap_fee = round(5.0 * sum(
                    float(item.get("gift_wrap_quantity", 0))
                    for item in items
                ), 2)
                had_items = bool(pre_write_result.get(item_key))
                delivery_fee = (
                    float(pre_write_result.get("delivery_fee") or 0.0)
                    if had_items else 5.0
                )
                tax = round((base_cost + gift_wrap_fee) * 0.10, 2)
                discount = float(result.get("discount") or 0.0)
                result.update({
                    "base_cost": base_cost,
                    "delivery_fee": delivery_fee,
                    "gift_wrap_fee": gift_wrap_fee,
                    "tax": tax,
                    "total_cost": round(
                        base_cost + delivery_fee + gift_wrap_fee + tax - discount, 2),
                })
        return result

    def simulate_task_completion(self, args: dict) -> Dict[str, str]:
        """Record ``supervisor.complete_task`` without writing the real supervisor database."""
        self.n_writes += 1
        self.first_write_seen = True
        copied_args = dict(args or {})
        status = copied_args.get("status", "success")
        completion_guard = _env_enabled("TTS_APPWORLD_COMPLETION_GUARD", True)
        if completion_guard and status != "fail" and self.unresolved_write_failures:
            self.n_completion_deferrals += 1
            failed_apis = sorted({
                f"{key[0]}.{key[1]}" for key in self.unresolved_write_failures
            })
            result = {
                "message": (
                    "Task completion deferred because prior write attempts are unresolved: "
                    + ", ".join(failed_apis)
                    + ". Repair or retry them before calling complete_task again."
                )
            }
            self.transition_log.record(
                {"app": "supervisor", "api": "complete_task", "args": copied_args},
                json.dumps(result),
                ["supervisor", "supervisor.complete_task"],
                applied=False,
            )
            return result
        self.simulated_task_completion = {
            "status": status,
            "answer": copied_args.get("answer"),
        }
        result = {"message": "Marked the active task complete."}
        return self._record_applied_write(
            "supervisor",
            "complete_task",
            copied_args,
            result,
            self._write_key("supervisor", "complete_task", copied_args),
        )

    @staticmethod
    def _write_key(app: str, api: str, args: dict) -> tuple:
        """Identity of a write's *target* (api + non-auth args) for duplicate/idempotency detection."""
        ids = tuple(sorted(
            (k, json.dumps(v, default=str, sort_keys=True))
            for k, v in (args or {}).items()
            if k not in ("access_token", "password", "token")
        ))
        return (app, api, ids)

    @staticmethod
    def _write_intent_key(app: str, api: str, args: dict) -> tuple:
        """Stable task intent across funding-source retries of the same write."""
        if app == "venmo" and api == "approve_payment_request":
            return (app, api, str(args.get("payment_request_id")))
        if app == "venmo" and api == "create_transaction":
            return (
                app,
                api,
                str(args.get("receiver_email")),
                args.get("amount"),
                args.get("description", ""),
                bool(args.get("private", False)),
            )
        if app == "venmo" and api in {
            "add_to_venmo_balance",
            "withdraw_from_venmo_balance",
        }:
            return (app, api, args.get("amount"))
        return AppWorldWMPredictor._write_key(app, api, args)

    def _predict_with_rejection(self, app: str, api: str, args: dict, schema: Any,
                                is_repeat: bool) -> Tuple[Any, Optional[Tuple[int, Any]]]:
        """LLM-predict a write that MAY be rejected, returning ``(response, failure)``.

        Unlike the optimistic success-only path, this gives the LLM both the success **and** failure
        response schemas plus the call history, and asks it to decide — from the history — whether the
        real API would SUCCEED or be REJECTED (target already in the requested end-state, not found/not
        present, unauthorized, own-resource, or a documented limit like insufficient balance / expired
        card).  ``failure`` is ``None`` on success, else ``(status_code, body)``.
        """
        doc = _api_doc(app, api)
        desc = (doc.get("description") or "").strip()[:300]
        fail_schema = _failure_schema(app, api) or {"message": "string"}
        repeat_note = (
            "NOTE: the history shows this exact write was ALREADY applied to this target in this "
            "rollout. Many state-changing writes are NOT idempotent and reject a repeat (e.g. "
            "'already approved/completed', 'already following', 'already liked'); but some are "
            "legitimately repeatable (sending a message/email, creating a new record). Decide which.\n"
            if is_repeat else ""
        )
        financial_note = (
            "This is a rejection-sensitive financial write. Do not guess or optimistically assume "
            "success. Use PYTHON_READ unless the recent history already contains the exact required "
            "funding facts. Venmo balance is available through venmo.show_venmo_balance; card details "
            "through venmo.show_payment_cards; and the side-effect-free private "
            "admin.show_card_balance read can verify a card's live balance from those details.\n"
            if (app, api) in _REJECTION_SENSITIVE_WRITES else ""
        )
        prompt = (
            f"You are a world model simulating the AppWorld '{app}' app's backing database. Predict the "
            f"EXACT JSON response the real API returns for this WRITE call, deciding from the recent call "
            f"history whether it SUCCEEDS or is REJECTED.\n\n"
            f"It is REJECTED (return the FAILURE response) when the history shows a precondition is "
            f"violated, e.g.: the target is already in the requested end-state (a repeated state-change), "
            f"the target is not found / not present, the action is on the user's own resource where "
            f"disallowed, the user is unauthorized, or a documented limit is violated (insufficient "
            f"balance, expired/invalid card, missing required field). Otherwise it SUCCEEDS.\n\n"
            f"{repeat_note}"
            f"{financial_note}"
            f"API: {app}.{api}\nDescription: {desc}\n"
            f"SUCCESS response schema: {json.dumps(schema, default=str)[:400]}\n"
            f"FAILURE response schema: {json.dumps(fail_schema, default=str)[:200]}\n"
            f"{self._frozen_time_text()}"
            "Recent prior API calls are available only through `data['history']`; generated Python "
            "must inspect or transform their result variables.\n"
            f"This call: {app}.{api}({json.dumps(args, default=str)[:400]})\n\n"
            f"Return ONLY a JSON object of the form "
            f'{{"outcome": "success"|"failure", "status_code": 200 or 422, "response": <the success or '
            f'failure response JSON>}}. Prefer SUCCESS unless the history CLEARLY shows a violation. No prose.'
        )
        resp = self._predict(
            prompt,
            'PREDICT: {"outcome":"success"|"failure","status_code":200|422,'
            '"response":<JSON response>}',
            max_tokens=800,
            python_context={
                "mode": "write_with_rejection",
                "app": app,
                "api": api,
                "args": args,
                "success_schema": schema,
                "failure_schema": fail_schema,
                "is_repeat": is_repeat,
                "frozen_datetime": self.frozen_datetime,
                "history": [
                    {
                        "app": h_app,
                        "api": h_api,
                        "name": f"{h_app}.{h_api}",
                        "args": h_args,
                        "result": result,
                    }
                    for h_app, h_api, h_args, result in self.history
                ],
            },
            python_force=self.program_only,
        )
        parsed = _parse_json(resp)
        if not isinstance(parsed, dict):
            return _synth_success(schema), None
        if "outcome" in parsed:
            body = parsed.get("response")
            if not isinstance(body, (dict, list)):
                body = {"message": "Rejected."} if str(parsed.get("outcome")).lower() == "failure" else _synth_success(schema)
            if str(parsed.get("outcome")).lower() == "failure":
                code = parsed.get("status_code") or 422
                try:
                    code = int(code)
                except (TypeError, ValueError):
                    code = 422
                if code == 200:
                    code = 422
                return body, (code, body)
            return body, None
        # LLM returned a bare response dict (ignored the wrapper) -> treat as success.
        return parsed, None

    def predict_write(self, app: str, api: str, args: dict, schema: Any) -> Any:
        """Predict a write API's response with the LLM (the faithful path — like the τ² WM).

        Every write is LLM-generated (no synthesised template), so the response matches what the real
        API returns (e.g. ``follow_artist`` → ``{"message": "Artist followed."}``).

        **Rejection awareness:** the real API rejects some writes (HTTP 4xx, raised as an exception the
        agent must react to) — e.g. approving an already-approved request, following an already-followed
        artist, acting on a missing/unauthorized resource.  A first-time status/message-only write uses a
        *stable* prompt (no per-call args/history) so the response cache folds write-heavy bursts of
        DISTINCT targets into one LLM call (the τ²-bench 429 lever).  But a **repeat** of the same write
        to the same target, or any **substantive** (non-trivial-schema) write, is routed through the
        rejection-aware path (:meth:`_predict_with_rejection`), which is given both the success and
        failure schemas plus the history and decides success-vs-rejection — raising
        :class:`WMPredictedHTTPError` when the real API would reject.  ``_synth_success`` is only a
        fallback if the LLM call fails.
        """
        self.n_writes += 1
        self.first_write_seen = True
        self._check_hard_deadline()
        key = self._write_key(app, api, args)
        is_repeat = key in self.applied_writes
        if self._over_budget():
            # rollout budget spent -> skip the LLM (and even the api-doc lookup), return the instant
            # structurally-valid fallback so a runaway write loop finishes instead of issuing
            # thousands of sequential predictions. A call-count backstop breaks unbounded loops fast.
            self._fallback_calls += 1
            self._check_runaway()
            result = _synth_success(schema)
            return self._record_applied_write(app, api, args, result, key)
        doc = _api_doc(app, api)
        missing = [
            str(parameter.get("name"))
            for parameter in (doc.get("parameters") or [])
            if isinstance(parameter, dict)
            and parameter.get("required")
            and args.get(parameter.get("name")) is None
        ]
        if missing:
            body = {
                "message": (
                    "Validation error. Missing required field(s): "
                    + ", ".join(missing)
                )
            }
            self._record_rejected_write(app, api, args, body)
            raise WMPredictedHTTPError(422, body)
        grounded_funding = (
            self._predict_grounded_venmo_funding(api, args, schema)
            if app == "venmo" else None
        )
        if grounded_funding is not None:
            result, failure = grounded_funding
            if failure is not None:
                status_code, body = failure
                self._record_rejected_write(app, api, args, body)
                raise WMPredictedHTTPError(status_code, body)
            return self._record_applied_write(app, api, args, result, key)
        desc = (doc.get("description") or "").strip()[:300]
        rejection_sensitive = (app, api) in _REJECTION_SENSITIVE_WRITES
        if _is_trivial_schema(schema) and not is_repeat and not rejection_sensitive:
            # First-time status/message-only write: args-independent success -> stable cached prompt so
            # a burst of distinct successful writes folds into one LLM call.
            prompt = (
                f"You are a world model of the AppWorld '{app}' app's backing database. The write API "
                f"'{app}.{api}' just executed SUCCESSFULLY. Return ONLY the exact JSON success response "
                f"the real API returns, matching its message wording exactly (e.g. a 'follow artist' "
                f"API returns {{\"message\": \"Artist followed.\"}}).\n"
                f"Description: {desc}\n"
                f"Success response schema: {json.dumps(schema, default=str)[:500]}\n"
                f"Return ONLY the JSON object. No prose."
            )
            resp = self._predict(
                prompt,
                "PREDICT: <the exact JSON success response>",
                max_tokens=800,
                python_context={
                    "mode": "write_response",
                    "app": app,
                    "api": api,
                    "args": args,
                    "success_schema": schema,
                    "frozen_datetime": self.frozen_datetime,
                },
                python_force=self.program_only,
            )
            result = _parse_json(resp)
            if result is None:
                result = _synth_success(schema)  # LLM unavailable -> structurally-valid fallback
        else:
            # Repeat of a prior write, or a substantive write: predict success OR rejection.
            result, failure = self._predict_with_rejection(app, api, args, schema, is_repeat)
            if failure is not None:
                # Real API would REJECT this write: it does NOT mutate state, so do not record it as a
                # write/applied/dirty change — surface the failure to the agent like the real env.
                status_code, body = failure
                self._record_rejected_write(app, api, args, body)
                raise WMPredictedHTTPError(status_code, body)
        return self._record_applied_write(app, api, args, result, key)

    def predict_read(
        self,
        app: str,
        api: str,
        args: dict,
        pre_write_result: Any,
        schema: Any,
        *,
        dirty: bool = True,
    ) -> Any:
        """Predict one read response from fixed or tool-gathered evidence.

        Dirty reads additionally apply the ordered predicted writes to an optional exact pre-write
        result. Clean reads receive no direct action result; recursive Ours may gather it through the
        existing bounded read-only tool loop, while flat B1 must use its fixed pre-action context.
        """
        self.n_predicted_reads += 1
        if dirty:
            self.n_dirty_reads += 1
        self._check_hard_deadline()
        if (
            not dirty
            and pre_write_result is None
            and _env_enabled("TTS_APPWORLD_PREGROUND_CLEAN_READS", False)
            and self._query is not None
        ):
            pre_write_result = self._query.read_value(f"{app}.{api}", args)
        if (
            not dirty
            and pre_write_result is not None
            and _env_enabled(
                "TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS", False
            )
        ):
            # The simulator itself requested this read through its restricted query
            # interface. With no logical transitions to apply, its state transition
            # is the deterministic identity function.
            result = copy.deepcopy(pre_write_result)
            self.history.append((app, api, dict(args), result))
            return result
        if not self.program_only and dirty and app == "venmo":
            venmo_result = self._predict_venmo_dirty_read(api, args, pre_write_result)
            if venmo_result is not None:
                self.history.append((app, api, dict(args), venmo_result))
                return venmo_result
        if (
            not self.program_only
            and dirty
            and _deterministic_dirty_reads_enabled()
            and app == "amazon"
            and api == "show_cart"
        ):
            cart = self._predict_amazon_cart(pre_write_result)
            if cart is not None:
                return cart
        if self._over_budget():
            # Rollout budget spent: recursive mode returns the exact pre-write state; flat B1 has no
            # action-conditioned baseline, so return an empty structured result.
            self._fallback_calls += 1
            self._check_runaway()
            fallback = pre_write_result if pre_write_result is not None else {}
            self.history.append((app, api, dict(args), fallback))
            return fallback
        doc = _api_doc(app, api)
        desc = (doc.get("description") or "").strip()[:200]
        if self.uses_fixed_context:
            baseline = (
                "Fixed pre-action evidence is available only through `data['fixed_initial_state']`."
            )
        else:
            baseline = (
                "Read evidence is available only through variables added to `data['reads']` by bounded "
                "read-only grounding tools."
            )
        state_instruction = (
            "Some earlier writes changed this app. Apply the ordered predicted writes to the pristine "
            "evidence before returning the logical post-write result."
            if dirty else
            (
                "This clean read is not executed directly for the agent. The host has copied the exact "
                "read-only API result into the grounding evidence below; generate the agent-visible "
                "response by reproducing that value exactly."
                if pre_write_result is not None else
                "This clean read is not executed directly for the agent. Predict its exact result from "
                "the available fixed context or read-only grounding evidence."
            )
        )
        exact_read_contract = ""
        grounded_clean_read = bool(not dirty and pre_write_result is not None)
        if grounded_clean_read:
            exact_read_contract = (
                "\nHIGHEST-PRIORITY CLEAN-READ RULE:\n"
                "- The PRISTINE / FIXED evidence below is the exact result of this native read API.\n"
                "- Generate the final response by copying that JSON value VERBATIM. Preserve every field, "
                "list element, null, empty value, ordering, and error body.\n"
                "- Do not make another read, infer a different value, summarize, repair, merge, or wrap it.\n"
            )
        elif not dirty and not self.uses_fixed_context and self._query is not None:
            if self.program_only or _wm_tool_mode() == "python":
                first_read = (
                    "On the FIRST tool turn, issue PYTHON_READ for exactly this native read API"
                )
            else:
                first_read = (
                    "On the FIRST tool turn, issue one READ command for exactly this native read API"
                )
            exact_read_contract = (
                "\nHIGHEST-PRIORITY CLEAN-READ RULE:\n"
                f"- {first_read}: "
                f"{app}.{api} with args {json.dumps(args, default=str)}.\n"
                "- Do not answer from memory, schema, prior history, or nearby records before making "
                "that read.\n"
                "- When its result arrives, return that JSON value VERBATIM: preserve every field, "
                "list element, null, empty value, and error body. Do not summarize, repair, merge, or "
                "wrap it.\n"
            )
        elif dirty and not self.uses_fixed_context and self._query is not None:
            exact_read_contract = (
                "\nHIGHEST-PRIORITY DIRTY-READ RULE:\n"
                f"- On the FIRST tool turn, issue PYTHON_READ for the pristine native API "
                f"{app}.{api} with args {json.dumps(args, default=str)}.\n"
                "- Treat that result only as the pre-write baseline. Generated Python must apply all "
                "relevant ordered entries from `data['writes']` and print the resulting logical "
                "post-write response.\n"
                "- Never return the pristine value unchanged when a relevant write affects it.\n"
            )
        prompt = (
            f"You are a world model of the AppWorld '{app}' app's backing database. "
            f"{state_instruction}{exact_read_contract}\n"
            f"READ API: {app}.{api}\nDescription: {desc}\n"
            f"This call: {app}.{api}({json.dumps(args, default=str)[:300]})\n\n"
            f"{self._frozen_time_text()}"
            f"EVIDENCE VARIABLES:\n{baseline}\n"
            "Applied writes are available only through `data['writes']`. Do not transcribe, truncate, "
            "summarize, or otherwise process read/write result text outside generated Python.\n\n"
            "Return a PYTHON_FINAL program whose printed value is the exact JSON result in the "
            "documented schema."
        )
        resp = self._predict(
            prompt,
            "PREDICT: <the exact resulting JSON read response>",
            max_tokens=2000,
            python_context={
                "mode": "dirty_read" if dirty else "clean_read",
                "app": app,
                "api": api,
                "args": args,
                "response_schema": schema,
                "frozen_datetime": self.frozen_datetime,
                "pre_write_result": pre_write_result,
                "grounded_clean_read": pre_write_result if grounded_clean_read else None,
                "grounded_clean_read_required": grounded_clean_read,
                "pristine_read_required": bool(
                    dirty and not self.uses_fixed_context and self._query is not None
                ),
                "fixed_initial_state": copy.deepcopy(
                    (self._flat_context_data or {}).get("fixed_initial_state", {})
                ) if self.uses_fixed_context else {},
                "exact_read_required": bool(exact_read_contract) and not dirty,
                "exact_read_call": (
                    {"app": app, "api": api, "args": dict(args)}
                    if exact_read_contract else None
                ),
                "writes": [
                    {
                        "app": w_app,
                        "api": w_api,
                        "name": f"{w_app}.{w_api}",
                        "args": w_args,
                        "result": result,
                    }
                    for w_app, w_api, w_args, result in self.write_history
                    if w_app == app
                ],
            },
            python_force=self.program_only,
        )
        result = _parse_json(resp)
        if result is None:
            result = pre_write_result if pre_write_result is not None else {}
        self.history.append((app, api, dict(args), result))
        return result

    def _writes_summary(self, limit: int = 40) -> str:
        """All predicted writes (the agent's intended mutations), for the verifier to reason about."""
        if not self.write_history:
            return "(the agent performed no write/mutation calls)"
        out = []
        for app, api, args, res in self.write_history[-limit:]:
            a = json.dumps(args, default=str)[:200]
            out.append(f"{app}.{api}({a})")
        return "\n".join(out)

    def _write_log(self, cap: int = 60) -> str:
        """The agent's write calls for reference-grounded auditing.

        Preserve both the beginning and end of long loops, and give each call enough argument space
        for exact IDs, email bodies, and multi-entity values. The previous first-60/160-char rendering
        often hid the final writes or truncated the value that a gold check compared.
        """
        if not self.write_history:
            return "(the agent performed no write/mutation calls)"
        wh = self.write_history
        arg_chars = max(160, _env_int("TTS_APPWORLD_VERIFIER_WRITE_ARG_CHARS", 400))
        if len(wh) <= cap:
            shown = list(enumerate(wh))
        else:
            head = max(1, cap // 2)
            tail = max(1, cap - head)
            shown = list(enumerate(wh[:head])) + list(enumerate(wh[-tail:], start=len(wh) - tail))
        rows = []
        previous_index = -1
        for index, (app, api, args, _res) in shown:
            if previous_index >= 0 and index > previous_index + 1:
                rows.append(f"... ({index - previous_index - 1} intermediate writes omitted) ...")
            rows.append(f"{index + 1}. {app}.{api}({json.dumps(args, default=str)[:arg_chars]})")
            previous_index = index
        if len(wh) > cap:
            rows.append(f"(total writes: {len(wh)})")
        return "\n".join(rows)

    def _structured_writes(self) -> List[Dict[str, Any]]:
        """Copied write records for exact verifier-side computation."""
        return [
            {
                "app": app,
                "api": api,
                "name": f"{app}.{api}",
                "args": args,
                "result": result,
            }
            for app, api, args, result in self.write_history
        ]

    def _structured_transitions(self) -> List[Dict[str, Any]]:
        """Copied transition records; no environment or requester handles are exposed."""
        return [
            {
                "action": entry.action,
                "observation": entry.observation,
                "resources": list(entry.resources),
                "applied": entry.applied,
            }
            for entry in self.transition_log.entries()
        ]

    @staticmethod
    def _structured_gold(gold: str) -> Dict[str, Any]:
        """Retain the raw gold text and parse any complete labelled JSON sections."""
        sections: Dict[str, Any] = {}
        for line in str(gold or "").splitlines():
            label, separator, payload = line.partition(":")
            if not separator:
                continue
            label = label.strip()
            match = re.fullmatch(r"GOLD VALUES \(([^)]+)\)", label, re.IGNORECASE)
            if match:
                label = match.group(1)
            try:
                sections[label] = json.loads(payload.strip())
            except Exception:
                continue
        return {"raw": str(gold or ""), "sections": sections}

    def _verifier_python_context(
        self,
        instruction: str,
        *,
        checklist: str = "",
        check: str = "",
        gold: str = "",
        steps: Optional[List[Dict[str, Any]]] = None,
        reads: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build the copied, execution-free evidence available to verifier Python."""
        if not _env_enabled("TTS_APPWORLD_VERIFIER_PYTHON", True):
            return None
        copied_reads = list(reads or [])
        copied_steps = [
            {
                **item,
                "code": str(item.get("code", ""))[:4_000],
                "output": str(item.get("output", ""))[-1_000:],
            }
            for item in (steps or [])
        ]
        return {
            "task": instruction,
            "active_task_apps": list(self.active_task_apps()),
            "checklist": checklist,
            "check": check,
            "gold": self._structured_gold(gold or checklist),
            "writes": self._structured_writes(),
            "transitions": self._structured_transitions(),
            "steps": copied_steps,
            "reads": copied_reads,
            "facts": {
                f"read {item.get('name', '')}({item.get('args', {})})": item.get("result")
                for item in copied_reads
            },
            "frozen_datetime": self.frozen_datetime,
        }

    def _reference_verdict(self, instruction: str, checklist: str, temperature: float) -> Optional[bool]:
        """Reference-grounded audit: FAIL only on a violation provable from the write log vs the gold spec.

        Given the task's ground-truth checks + gold values and the agent's complete write log, the model
        fills a per-check WORKSHEET — for each check it must extract the REQUIRED gold set/value, the set
        the agent's writes actually cover, and the MISSING/EXTRA difference, then mark MET / UNMET /
        UNKNOWN.  This explicit set-arithmetic catches the dominant false-positive source (a required
        change the agent never made, e.g. an id that should be removed but has no matching write), which
        the old holistic 'find a violation' prompt missed.  A check that genuinely cannot be decided from
        the write log alone (shuffle order, internal bookkeeping, cursor/playing state) is UNKNOWN and
        does not by itself fail the rollout; only a positively UNMET check fails it.
        """
        prompt = (
            "You audit an AppWorld agent against the task's GROUND-TRUTH checks. Its WRITE calls were "
            "SIMULATED rather than officially executed. Logical READ/TRACE evidence consists of the exact "
            "pre-write state plus predicted transitions, so judge whether those writes produce the exact "
            "required changes rather than trusting a simulated success response.\n"
            "For EACH ground-truth check, work it out explicitly:\n"
            "  REQUIRED = the exact gold ids/values the check demands;\n"
            "  AGENT = the matching ids/values the agent's write calls actually cover (from the write log);\n"
            "  MISSING = REQUIRED not covered by AGENT; EXTRA = AGENT covered but not REQUIRED;\n"
            "  STATUS = MET (MISSING and EXTRA both empty) / UNMET (any MISSING, wrong, or EXTRA change) / "
            "UNKNOWN (cannot be decided from the write log alone — e.g. shuffle order, internal record "
            "bookkeeping, cursor/playing state).\n"
            "VERDICT RULE: FAIL if ANY check is UNMET; otherwise SUCCESS (UNKNOWN checks do NOT fail it).\n\n"
            f"TASK:\n{instruction}\n\n"
            f"SUBSTANTIVE APP ACTIVITY:\n{self._task_app_activity_text()}\n\n"
            f"{checklist}\n\n"
            f"AGENT'S COMPLETE WRITE CALLS (full args; loops noted):\n{self._write_log()}\n\n"
            f"PREDICTED TRANSITION TRACE (APPLIED vs REJECTED effects):\n"
            f"{self.transition_log.render()}\n\n"
            "Fill every check with its REQUIRED / AGENT / MISSING / EXTRA / STATUS, then on the LAST line "
            "answer exactly 'VERDICT: SUCCESS' or 'VERDICT: FAIL'."
        )
        resp = self._predict(
            prompt,
            "VERDICT: SUCCESS  or  VERDICT: FAIL",
            temperature=temperature,
            max_tokens=700,
            budgeted=False,
            python_context=self._verifier_python_context(
                instruction, checklist=checklist),
            python_output_kind="verdict",
        )
        if not resp:
            return None
        return _is_success_verdict(resp)

    def _reference_score_percheck(self, instruction: str, checks: List[str], gold: str,
                                  temperature: float, read_fn=None, max_queries: int = 3) -> Optional[float]:
        """PER-CHECK reference audit: a focused sub-pass classifies EACH ground-truth check, then aggregate.

        Each check is judged in isolation (it sees only that one check + the gold values + the FULL write
        log) as MET / UNMET / UNKNOWN.  Focused single-check reasoning catches decidable violations the
        all-at-once worksheet rubber-stamps (better precision), and the aggregate is naturally GRADED —
        ``score = MET / (MET + UNMET)`` over the DECIDABLE checks (UNKNOWN excluded) — which gives the
        score-spread the binary worksheet lacks (better coverage / ranking).  In legacy mode UNKNOWN
        checks are excluded.  With ``TTS_APPWORLD_VERIFIER_COVERAGE_AWARE=1``, every requested check
        stays in the denominator and UNKNOWN/failed checks receive
        ``TTS_APPWORLD_VERIFIER_UNKNOWN_CREDIT`` (default 0.25).  This prevents one proven check plus
        many unknown checks from tying a fully proven candidate.  Returns ``None`` only if every
        sub-pass failed.

        If ``read_fn`` is given, each check runs as a small MULTI-TURN loop in which the sub-pass may issue
        **read-only** queries against the REAL pre-write DB (exact, faithful — NOT predict_read) to ground
        its judgment: resolve entities, fetch the actual starting set, or compute a required delta (e.g.
        'befriend = phone_friends − current_venmo_friends').  Reads return the state BEFORE the agent's
        (simulated) writes, so the sub-pass reasons about the write log's net effect relative to it.
        """
        if not checks:
            return None
        wl = self._write_log()
        met = unmet = unknown = valid = 0
        for check in checks:
            status = (self._percheck_with_reads(instruction, check, gold, wl, read_fn, temperature, max_queries)
                      if read_fn is not None
                      else self._percheck_oneshot(instruction, check, gold, wl, temperature))
            if status is None:
                continue
            valid += 1
            if status == "UNMET":
                unmet += 1
            elif status == "MET":
                met += 1
            else:
                unknown += 1
        if valid == 0:
            self.last_percheck_breakdown = {
                "met": 0,
                "unmet": 0,
                "unknown": 0,
                "failed": len(checks),
                "total": len(checks),
                "coverage_aware": _env_enabled(
                    "TTS_APPWORLD_VERIFIER_COVERAGE_AWARE", False),
            }
            return None
        failed = len(checks) - valid
        coverage_aware = _env_enabled(
            "TTS_APPWORLD_VERIFIER_COVERAGE_AWARE", False)
        unknown_credit = min(
            1.0,
            max(0.0, _env_float(
                "TTS_APPWORLD_VERIFIER_UNKNOWN_CREDIT", 0.25)),
        )
        self.last_percheck_breakdown = {
            "met": met,
            "unmet": unmet,
            "unknown": unknown,
            "failed": failed,
            "total": len(checks),
            "coverage_aware": coverage_aware,
            "unknown_credit": unknown_credit if coverage_aware else None,
        }
        if coverage_aware:
            return (
                met + unknown_credit * (unknown + failed)
            ) / max(1, len(checks))
        decidable = met + unmet
        if decidable == 0:
            return 0.5  # nothing verifiable from the write log -> neutral
        return met / decidable

    @staticmethod
    def _parse_status(resp: str) -> Optional[str]:
        """Parse a per-check sub-pass response into MET / UNMET / UNKNOWN (last STATUS line wins)."""
        if not resp:
            return None
        up = resp.upper()
        tail = up.rsplit("STATUS:", 1)[-1] if "STATUS:" in up else up[-40:]
        if "UNMET" in tail:
            return "UNMET"
        if "UNKNOWN" in tail:
            return "UNKNOWN"
        if "MET" in tail:
            return "MET"
        return "UNKNOWN"

    def _percheck_oneshot(self, instruction: str, check: str, gold: str, wl: str,
                          temperature: float) -> Optional[str]:
        """Classify ONE check from the write log alone (no DB reads)."""
        prompt = (
            "Audit ONE ground-truth success check of an AppWorld task against the agent's write log. "
            "The agent's WRITE calls were SIMULATED (reported success regardless of correctness), so "
            "judge ONLY from whether the write calls perform the exact change THIS check requires.\n"
            "Classify the check:\n"
            "  MET = the write log clearly performs the exact required change (right gold ids/values, "
            "nothing required missing, no wrong/extra change for this check);\n"
            "  UNMET = a required id/value is missing from the writes, or a wrong/extra change is present;\n"
            "  UNKNOWN = cannot be decided from the write log alone because it depends on the true final "
            "state (e.g. shuffle order, cursor position, playing state, internal record bookkeeping).\n\n"
            f"TASK:\n{instruction}\n\nTHE ONE CHECK TO AUDIT:\n{check}\n\nGOLD VALUES:\n{gold}\n\n"
            f"AGENT'S COMPLETE WRITE CALLS (full args; loops noted):\n{wl}\n\n"
            "Do the explicit id/value comparison, then on the LAST line answer exactly "
            "'STATUS: MET' or 'STATUS: UNMET' or 'STATUS: UNKNOWN'."
        )
        resp = self._predict(
            prompt,
            "STATUS: MET  or  STATUS: UNMET  or  STATUS: UNKNOWN",
            temperature=temperature,
            max_tokens=500,
            budgeted=False,
            python_context=self._verifier_python_context(
                instruction, check=check, gold=gold),
            python_output_kind="status",
            use_query=False,
        )
        return self._parse_status(resp)

    def _percheck_with_reads(self, instruction: str, check: str, gold: str, wl: str,
                             read_fn, temperature: float, max_queries: int) -> Optional[str]:
        """Classify ONE check as a small MULTI-TURN loop that may READ the real pre-write DB to ground it.

        ``read_fn(app, api, **args)`` performs an EXACT, read-only query of the real backing DB (the true
        state BEFORE the agent's simulated writes).  The sub-pass uses it to resolve entities / fetch the
        actual starting set / compute a required delta, then judges whether the write log performs the
        exact change THIS check requires relative to that real state + the gold values.
        """
        if _wm_tool_mode() == "python":
            prompt = (
                "Verify ONE ground-truth success check of an AppWorld task. Database evidence is the "
                "exact REAL state BEFORE the agent's simulated writes. Use Python read rounds when "
                "needed, then decide whether the write log produces the exact required change relative "
                "to that starting state and the gold values.\n\n"
                f"TASK:\n{instruction}\n\nTHE ONE CHECK TO AUDIT:\n{check}\n\n"
                f"GOLD VALUES:\n{gold}\n\nAGENT'S COMPLETE WRITE CALLS:\n{wl}\n\n"
                "MET means exact required change with nothing missing/extra; UNMET means a missing or "
                "wrong/extra change; UNKNOWN means the true final state remains undecidable."
            )
            response = self._predict(
                prompt,
                "STATUS: MET  or  STATUS: UNMET  or  STATUS: UNKNOWN",
                max_tokens=500,
                temperature=temperature,
                budgeted=False,
                python_context=self._verifier_python_context(
                    instruction, check=check, gold=gold),
                python_output_kind="status",
                use_query=False,
                python_read_fn=lambda app, api, args: read_fn(app, api, **args),
            )
            return self._parse_status(response)

        read_facts: List[Dict[str, Any]] = []
        python_context = self._verifier_python_context(
            instruction, check=check, gold=gold, reads=read_facts)
        python_available = (
            python_context is not None
            and _env_enabled("TTS_APPWORLD_WM_PYTHON", True)
            and not self.python_disabled
        )
        python_protocol = ""
        if python_available:
            python_protocol = self._python_protocol(
                python_context,
                required=False,
                context_chars=_json_size_hint(python_context, 2_000),
                output_kind="status",
            )
        sys = (
            "You verify ONE ground-truth success check of an AppWorld task against the agent's write log. "
            "The agent's WRITE calls were SIMULATED (NOT applied), so a database READ returns the REAL "
            "state BEFORE those writes. You may issue read-only queries to gather facts (the current "
            "entities, the actual starting set, anything needed to compute the required change), then "
            "decide whether the agent's write calls perform the exact change THIS check requires relative "
            "to that starting state + the gold values.\n"
            "Reply with ONE JSON action object per turn, or use the Python option below to finish:\n"
            '  {"action":"query","app":"<app>","api":"<read_api>","args":{...}}  to read the DB, or\n'
            '  {"action":"verdict","status":"MET"|"UNMET"|"UNKNOWN"}  to finish.\n'
            "MET = writes perform the exact required change (right ids/values, none missing, none wrong/"
            "extra). UNMET = a required id/value is missing or a wrong/extra change is present. UNKNOWN = "
            "depends on the true FINAL state not decidable even with reads (shuffle order, cursor/playing "
            "state, internal record bookkeeping). Use 'api_docs' to find read APIs; query only READ apis."
            f"{python_protocol}"
        )
        user = (
            f"TASK:\n{instruction}\n\nTHE ONE CHECK TO AUDIT:\n{check}\n\nGOLD VALUES:\n{gold}\n\n"
            f"AGENT'S COMPLETE WRITE CALLS (full args; loops noted):\n{wl}\n\n"
            "Investigate with read-only queries if helpful, then give your verdict."
        )
        messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        for _ in range(max_queries):
            try:
                self.n_llm_calls += 1
                resp = self._chat.complete(messages, temperature=temperature, max_tokens=400)
            except Exception:
                return None
            if python_available and _extract_python_program(resp) is not None:
                current_context = self._verifier_python_context(
                    instruction, check=check, gold=gold, reads=read_facts)
                resolved = self._resolve_python_prediction(
                    resp,
                    prompt=f"{sys}\n\n{user}",
                    context=current_context or {},
                    max_tokens=400,
                    temperature=temperature,
                    budgeted=False,
                    python_required=False,
                    output_kind="status",
                )
                return self._parse_status(resolved)
            action = _parse_json(resp)
            if not isinstance(action, dict):
                messages.append({"role": "user", "content": 'Reply with ONE JSON action object.'})
                continue
            kind = str(action.get("action", "")).lower()
            if kind == "verdict":
                st = str(action.get("status", "")).upper()
                return "UNMET" if "UNMET" in st else ("UNKNOWN" if "UNKNOWN" in st else ("MET" if "MET" in st else "UNKNOWN"))
            if kind == "query":
                app = str(action.get("app", "")); api = str(action.get("api", ""))
                qargs = action.get("args") or {}
                if not isinstance(qargs, dict):
                    qargs = {}
                try:
                    result = read_fn(app, api, **qargs)
                except Exception as exc:
                    result = {"error": str(exc)[:200]}
                read_facts.append({
                    "app": app,
                    "api": api,
                    "name": f"{app}.{api}",
                    "args": qargs,
                    "result": result,
                })
                messages.append({"role": "assistant", "content": resp[:400]})
                messages.append({"role": "user",
                                 "content": f"READ RESULT ({app}.{api}):\n{json.dumps(result, default=str)[:1500]}"})
                continue
            messages.append({"role": "user", "content": 'Use {"action":"query"|"verdict",...}.'})
        # ran out of query turns -> force a verdict.
        messages.append({"role": "user",
                         "content": 'Decide now. Reply ONLY {"action":"verdict","status":"MET"|"UNMET"|"UNKNOWN"}.'})
        try:
            self.n_llm_calls += 1
            resp = self._chat.complete(messages, temperature=temperature, max_tokens=120)
        except Exception:
            return None
        if python_available and _extract_python_program(resp) is not None:
            current_context = self._verifier_python_context(
                instruction, check=check, gold=gold, reads=read_facts)
            resolved = self._resolve_python_prediction(
                resp,
                prompt=f"{sys}\n\n{user}",
                context=current_context or {},
                max_tokens=400,
                temperature=temperature,
                budgeted=False,
                python_required=False,
                output_kind="status",
            )
            return self._parse_status(resolved)
        action = _parse_json(resp)
        if isinstance(action, dict) and "status" in action:
            st = str(action.get("status", "")).upper()
            return "UNMET" if "UNMET" in st else ("UNKNOWN" if "UNKNOWN" in st else ("MET" if "MET" in st else "UNKNOWN"))
        return self._parse_status(resp)

    def _grounded_verdict(self, instruction: str, steps_text: str, query_fn,
                          temperature: float, max_queries: int) -> Optional[bool]:
        """One read-grounded verifier pass: investigate the real DB, then judge.

        The verifier is told the task, the agent's executed code, and the agent's INTENDED writes (the
        WM-predicted mutations — which were NOT applied to the real DB).  It may issue **read-only**
        queries (executed by ``query_fn`` against the real backing DB, so it sees the true pre-write
        state) to ground its judgment, then decides whether applying the agent's writes to that state
        satisfies the task.  Returns the SUCCESS bool, or ``None`` if the pass produced no verdict.
        """
        if _wm_tool_mode() == "python":
            prompt = (
                "You are a strict AppWorld verifier. The agent acted inside a world model. Use Python "
                "read rounds to inspect the resulting logical state when needed, then decide whether "
                "EVERY task requirement is satisfied with the right targets/values and nothing missing "
                "or extra. Never trust simulated write success without checking relevant evidence.\n\n"
                f"TASK:\n{instruction}\n\nAGENT'S EXECUTED CODE:\n{steps_text}\n\n"
                f"AGENT'S WRITE CALLS:\n{self._writes_summary()}"
            )
            response = self._predict(
                prompt,
                "VERDICT: SUCCESS  or  VERDICT: FAIL",
                max_tokens=500,
                temperature=temperature,
                budgeted=False,
                python_context=self._verifier_python_context(
                    instruction, steps=[{"rendered": steps_text}]),
                python_output_kind="verdict",
                use_query=False,
                python_read_fn=lambda app, api, args: query_fn(app, api, **args),
            )
            return _is_success_verdict(response) if response else None

        writes = self._writes_summary()
        read_facts: List[Dict[str, Any]] = []
        python_context = self._verifier_python_context(
            instruction, steps=[{"rendered": steps_text}], reads=read_facts)
        python_available = (
            python_context is not None
            and _env_enabled("TTS_APPWORLD_WM_PYTHON", True)
            and not self.python_disabled
        )
        python_protocol = ""
        if python_available:
            python_protocol = self._python_protocol(
                python_context,
                required=False,
                context_chars=_json_size_hint(python_context, 2_000),
                output_kind="verdict",
            )
        sys = (
            "You are a strict verifier deciding whether an AppWorld agent COMPLETED its task. The agent "
            "acted inside a world model. You may QUERY the database (read-only); a query returns the "
            "RESULTING state AFTER the agent's actions (the world model applies the agent's writes to a "
            "query of an app it modified). Use queries to check the actual outcome, then decide whether "
            "EVERY requirement of the task is satisfied (right targets, right values, nothing missing, "
            "nothing extra/duplicated).\n\n"
            "IMPORTANT: the agent's write results were simulated, so do NOT trust a write just because it "
            "ran — VERIFY the resulting state by querying it. But also remember a query already reflects "
            "the agent's writes, so the change SHOULD be visible if the agent did it correctly.\n\n"
            "Respond with ONE JSON action object per turn, or use the Python option below to finish:\n"
            '  {"action":"query","app":"<app>","api":"<read_api>","args":{...}}  to inspect state, or\n'
            '  {"action":"verdict","success":true|false,"reason":"<short>"}  to finish.\n'
            "Use 'api_docs' (e.g. show_api_descriptions/show_api_doc) if you need to find a read API. "
            "Query only READ apis. Investigate the key facts before you decide; be strict but fair."
            f"{python_protocol}"
        )
        user = (
            f"TASK:\n{instruction}\n\n"
            f"AGENT'S EXECUTED CODE (with simulated outputs):\n{steps_text}\n\n"
            f"AGENT'S WRITE/MUTATION CALLS (their effects are reflected when you query the DB):\n{writes}\n\n"
            f"Investigate the resulting state with read-only queries, then give your verdict."
        )
        messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        for _ in range(max_queries):
            try:
                resp = self._chat.complete(messages, temperature=temperature, max_tokens=400)
            except Exception:
                return None
            if python_available and _extract_python_program(resp) is not None:
                current_context = self._verifier_python_context(
                    instruction, steps=[{"rendered": steps_text}], reads=read_facts)
                resolved = self._resolve_python_prediction(
                    resp,
                    prompt=f"{sys}\n\n{user}",
                    context=current_context or {},
                    max_tokens=400,
                    temperature=temperature,
                    budgeted=False,
                    python_required=False,
                    output_kind="verdict",
                )
                return _is_success_verdict(resolved) if resolved else None
            action = _parse_json(resp)
            if not isinstance(action, dict):
                messages.append({"role": "user", "content": "Reply with ONE valid JSON action object."})
                continue
            kind = str(action.get("action", "")).lower()
            if kind == "verdict":
                return bool(action.get("success"))
            if kind == "query":
                app = str(action.get("app", "")); api = str(action.get("api", ""))
                qargs = action.get("args") or {}
                if not isinstance(qargs, dict):
                    qargs = {}
                try:
                    result = query_fn(app, api, **qargs)
                except Exception as exc:  # surface the error so the verifier can adjust
                    result = {"error": str(exc)[:200]}
                read_facts.append({
                    "app": app,
                    "api": api,
                    "name": f"{app}.{api}",
                    "args": qargs,
                    "result": result,
                })
                messages.append({"role": "assistant", "content": resp[:500]})
                messages.append({"role": "user",
                                 "content": f"QUERY RESULT ({app}.{api}):\n{json.dumps(result, default=str)[:1500]}"})
                continue
            messages.append({"role": "user", "content": 'Use {"action":"query"|"verdict",...}.'})
        # ran out of query turns -> force a verdict.
        messages.append({"role": "user",
                         "content": 'Decide now. Reply ONLY {"action":"verdict","success":true|false}.'})
        try:
            resp = self._chat.complete(messages, temperature=temperature, max_tokens=200)
        except Exception:
            return None
        if python_available and _extract_python_program(resp) is not None:
            current_context = self._verifier_python_context(
                instruction, steps=[{"rendered": steps_text}], reads=read_facts)
            resolved = self._resolve_python_prediction(
                resp,
                prompt=f"{sys}\n\n{user}",
                context=current_context or {},
                max_tokens=200,
                temperature=temperature,
                budgeted=False,
                python_required=False,
                output_kind="verdict",
            )
            return _is_success_verdict(resolved) if resolved else None
        action = _parse_json(resp)
        if isinstance(action, dict) and "success" in action:
            return bool(action.get("success"))
        return _is_success_verdict(resp) if resp else None

    def predict_success(self, instruction: str, code_blocks: List[str], outputs: List[str],
                        query_fn=None, reference_checklist: Optional[str] = None,
                        reference_checks: Optional[Tuple[List[str], str]] = None,
                        real_read_fn=None) -> float:
        """WM-only selection signal: confidence in [0,1] that the agent COMPLETED the task.

        The candidate's writes must NOT be executed on the real env. By default, the judge receives
        AppWorld's native requirement strings and gold values (as OfficeBench does), but it never sees
        an officially evaluated final state; blind WM-only verification remains available as an ablation.

        * **Per-check reference auditor** (when ``reference_checks`` is given and
          ``TTS_APPWORLD_VERIFIER_PERCHECK`` != ``0``): a focused sub-pass audits EACH ground-truth check
          in isolation (MET/UNMET/UNKNOWN) and the score is the graded ``MET / (MET + UNMET)`` over the
          decidable checks — sharper per-check precision and a naturally graded score for ranking.
        * **Reference-grounded auditor** (default when ``reference_checklist`` is available; disable with
          ``TTS_APPWORLD_VERIFIER_REFERENCE=0``): one worksheet pass over all checks; FAIL only on a
          write-log/state-trace-provable violation; ``TTS_APPWORLD_VERIFIER_SAMPLES`` consistency samples.
          Both use the task's ground-truth spec → REFERENCE-GROUNDED (τ²-style), never executing the
          candidate or observing its final state.
        * **Read-grounded agentic verifier** (``query_fn`` + ``TTS_APPWORLD_VERIFIER_GROUNDED``): may
          issue read-only DB queries; bounded by predict_read fidelity, off by default.
        * **Uniform WM subagent** (default when the requester interception has bound a query interface):
          uses ``PYTHON_READ`` rounds to inspect logical state before a direct or ``PYTHON_FINAL`` verdict.
        * **Prompt-only judge**: fallback when no query interface is bound or agentic mode is disabled.
          A single binary verdict or ``K>=2`` rubric self-consistency is supported.
        * **Restricted Python computation** (enabled by ``TTS_APPWORLD_VERIFIER_PYTHON``): every path may
          compute exact set/count/date comparisons over copied gold, writes, transitions, steps, and
          read facts. It receives no environment handle and cannot execute candidate writes.
        """
        if self.uses_fixed_context:
            fixed = self._flat_context_data or {}
            instruction = str(fixed.get("task") or "")
            reference_checklist = str(fixed.get("evaluation_checklist") or "") or None
            fixed_checks = list(fixed.get("evaluation_checks") or [])
            fixed_gold = str(fixed.get("gold_values") or "")
            reference_checks = (
                (fixed_checks, fixed_gold) if fixed_checks else None
            )
            query_fn = None
            real_read_fn = None

        step_records = [
            {"index": i, "code": code, "output": str(output)}
            for i, (code, output) in enumerate(zip(code_blocks, outputs))
        ]
        steps = "\n\n".join(
            f"# step {item['index']}\n{item['code']}\nOutput: {item['output'][-300:]}"
            for item in step_records
        )[:6000]
        samples = _env_int("TTS_APPWORLD_VERIFIER_SAMPLES", 1)

        # Per-check reference auditor: a focused sub-pass per ground-truth check -> graded met/decidable.
        # When TTS_APPWORLD_VERIFIER_PERCHECK_READ != 0 and a real read fn is available, each check may
        # query the REAL pre-write DB (exact) to ground its judgment (resolve entities / required deltas).
        if reference_checks and os.environ.get("TTS_APPWORLD_VERIFIER_PERCHECK", "0") not in ("0", "false", "False"):
            checks, gold = reference_checks
            temp = _env_float("TTS_APPWORLD_VERIFIER_TEMP", 0.0) if samples > 1 else 0.0
            use_reads = (not self.uses_fixed_context and real_read_fn is not None
                         and os.environ.get("TTS_APPWORLD_VERIFIER_PERCHECK_READ", "1") not in ("0", "false", "False"))
            max_q = _env_int("TTS_APPWORLD_VERIFIER_PERCHECK_QUERIES", 3)
            score = self._reference_score_percheck(
                instruction, checks, gold, temp,
                read_fn=(real_read_fn if use_reads else None), max_queries=max_q)
            if score is not None:
                return score
            # fall through if every sub-pass failed.

        # Reference-grounded auditor: check the write log against the task's ground-truth success spec.
        # Match OfficeBench: native evaluator requirements/gold are the default verifier grounding.
        # Set TTS_APPWORLD_VERIFIER_REFERENCE=0 for the blind WM-only ablation.
        if reference_checklist and _env_enabled("TTS_APPWORLD_VERIFIER_REFERENCE", True):
            k = max(1, samples)
            temp = _env_float("TTS_APPWORLD_VERIFIER_TEMP", 0.7) if k > 1 else 0.0
            votes = 0
            valid = 0
            for _ in range(k):
                verdict = self._reference_verdict(instruction, reference_checklist, temp)
                if verdict is None:
                    continue
                valid += 1
                if verdict:
                    votes += 1
            if valid:
                return votes / valid
            # fall through to the prompt-only judge if every reference pass failed.

        # Read-grounded agentic verifier (opt-in via TTS_APPWORLD_VERIFIER_GROUNDED=1): investigate the
        # DB before judging. NOTE: queries to apps the agent wrote go through predict_read, so this
        # verifier's accuracy is currently bounded by dirty-read fidelity — when predict_read hallucinates
        # post-write state it can wrongly reject real successes. Off by default until predict_read improves.
        if (not self.uses_fixed_context and query_fn is not None
                and os.environ.get("TTS_APPWORLD_VERIFIER_GROUNDED", "0") not in ("0", "false", "False")):
            k = max(1, samples)
            temp = _env_float("TTS_APPWORLD_VERIFIER_TEMP", 0.7) if k > 1 else 0.0
            max_queries = _env_int("TTS_APPWORLD_VERIFIER_MAX_QUERIES", 6)
            votes = 0
            valid = 0
            for _ in range(k):
                verdict = self._grounded_verdict(instruction, steps, query_fn, temp, max_queries)
                if verdict is None:
                    continue
                valid += 1
                if verdict:
                    votes += 1
            if valid:
                return votes / valid
            # fall through to the prompt-only judge if every grounded pass failed.

        # Default AppWorld WM subagent: the same Python-emitted read protocol used for observation
        # prediction also gathers evidence for the terminal success verdict.
        if (not self.uses_fixed_context and self._query is not None
                and os.environ.get("TTS_APPWORLD_WM_AGENTIC", "1") not in ("0", "false", "False")):
            k = max(1, samples)
            temp = _env_float("TTS_APPWORLD_VERIFIER_TEMP", 0.7) if k > 1 else 0.0
            header = (
                "You are a strict AppWorld world-model verifier. Decide whether every task requirement "
                "is satisfied by the logical state produced by the predicted writes. Do not trust a "
                "write merely because its simulated response reported success; inspect the relevant "
                "logical state and transition trace.\n\n"
                f"TASK:\n{instruction}\n\nAGENT STEPS:\n{steps}\n\n"
                f"PREDICTED WRITE TRANSITIONS:\n{self.transition_log.render()}"
            )
            votes = 0
            valid = 0
            for _ in range(k):
                verdict = self._predict(
                    header,
                    "VERDICT: SUCCESS  or  VERDICT: FAILURE",
                    max_tokens=600,
                    temperature=temp,
                    budgeted=False,
                    python_context=self._verifier_python_context(
                        instruction, steps=step_records),
                    python_output_kind="verdict",
                )
                if not verdict:
                    continue
                valid += 1
                if _is_success_verdict(verdict):
                    votes += 1
            if valid:
                return votes / valid

        if samples <= 1:
            prompt = (
                "You judge whether an AppWorld agent COMPLETED its task, given the task and the code it ran "
                "(with outputs). Note tool results for write actions were simulated, so judge from the plan's "
                "coherence and whether the required steps were performed.\n\n"
                f"TASK:\n{instruction}\n\nAGENT STEPS:\n{steps}\n\n"
                "On the LAST line answer exactly 'VERDICT: SUCCESS' or 'VERDICT: FAIL'."
            )
            resp = self._predict(
                prompt,
                "VERDICT: SUCCESS  or  VERDICT: FAIL",
                max_tokens=400,
                temperature=0.0,
                budgeted=False,
                python_context=self._verifier_python_context(
                    instruction, steps=step_records),
                python_output_kind="verdict",
                use_query=False,
            )
            return 1.0 if _is_success_verdict(resp) else 0.0

        # K>=2: rubric-grounded, self-consistency-ensembled judge (WM-only, no real execution).
        prompt = (
            "You judge whether an AppWorld agent COMPLETED its task. Tool results for WRITE actions were "
            "SIMULATED, so do NOT assume a step worked just because it ran — judge whether every required "
            "step was actually performed with the right arguments.\n"
            "1) List the DISTINCT sub-tasks the instruction requires.\n"
            "2) For EACH sub-task, decide whether the agent's code performed it, citing the step number. "
            "Be strict: a missing, wrong-argument, skipped, or unverified sub-task means NOT complete.\n"
            "3) Answer SUCCESS only if ALL sub-tasks are satisfied.\n\n"
            f"TASK:\n{instruction}\n\nAGENT STEPS:\n{steps}\n\n"
            "On the LAST line answer exactly 'VERDICT: SUCCESS' or 'VERDICT: FAIL'."
        )
        temp = _env_float("TTS_APPWORLD_VERIFIER_TEMP", 0.7)
        votes = 0
        valid = 0
        for _ in range(samples):
            resp = self._predict(
                prompt,
                "VERDICT: SUCCESS  or  VERDICT: FAIL",
                max_tokens=500,
                temperature=temp,
                budgeted=False,
                python_context=self._verifier_python_context(
                    instruction, steps=step_records),
                python_output_kind="verdict",
                use_query=False,
            )
            if not resp:
                continue
            valid += 1
            if _is_success_verdict(resp):
                votes += 1
        if valid == 0:
            return 0.0
        return votes / valid


def install_wm_requester(
    world,
    predictor: AppWorldWMPredictor,
    *,
    predict_all_actions: bool = True,
):
    """Patch ``world.requester.request`` with exact-read or predict-all WM routing.

    Returns the original bound ``request`` (so the caller can restore it).  Routing:

    * Predict-all routes substantive documented **reads** through the WM by default. Unknown metadata
      is blocked rather than assumed safe.
    * ``login`` executes exactly to obtain a token for subsequent protected reads. ``logout`` and all
      mutating admin/control endpoints are simulated, never sent to the real requester.
    * ``supervisor.complete_task`` is recorded in predictor-local logical state; a logical
      ``show_active_task`` overlay stops the loop without changing the real supervisor database.
    * **Writes** (``POST``/``PUT``/``PATCH``/``DELETE``) are predicted (no real mutation), marking
      their app *dirty*.
    * A read of a *dirty* app (one a prior write touched) is predicted from its exact pre-write result
      + the write history (``predict_read``) instead of returning stale state.

    By default, the read-grounded prefix stays exact. With ``predict_all_actions=True``, flat B1
    predicts substantive clean reads from fixed context; recursive Ours routes them through the
    prompted world-model grounding loop, which may call the identical whitelisted read-only simulator
    tool before generating the result. Dirty reads remain predicted from pristine evidence plus
    ordered writes. Required auth/control passthroughs stay exact.
    """
    orig_request = world.requester.request
    predictor.program_only = bool(
        predict_all_actions
        and _env_enabled("TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY", True)
    )

    # Pin the predictor's clock to the env's frozen per-task datetime so predicted server timestamps
    # match the real (also-frozen) commit world exactly. Best-effort: never block WM install on it.
    if not predictor.frozen_datetime:
        try:
            predictor.frozen_datetime = world.task.datetime.isoformat()
        except Exception:
            pass

    def wm_request(_app_name, _api_name, client=None, raise_on_failure=None,
                   show=False, track=True, **data):
        doc = _api_doc(_app_name, _api_name)
        if not doc:
            body = {
                "message": (
                    f"{_app_name}.{_api_name} has no API metadata; blocked to preserve "
                    "read-only world-model sampling."
                )
            }
            if raise_on_failure is False:
                return body
            raise Exception(f"Response status code is 404:\n{json.dumps(body)}")

        if _app_name == "supervisor" and _api_name == "complete_task":
            return predictor.simulate_task_completion(data)

        if not _is_write(_app_name, _api_name):
            passthrough = (
                _app_name in _PASSTHROUGH_APPS
                or _api_name in _PASSTHROUGH_APIS
            )
            if predict_all_actions and not passthrough:
                dirty = _app_name in predictor.dirty_apps
                return predictor.predict_read(
                    _app_name,
                    _api_name,
                    data,
                    None,
                    _success_schema(_app_name, _api_name),
                    dirty=dirty,
                )
            if (predictor.uses_fixed_context
                    and predictor.first_write_seen
                    and _app_name in predictor.dirty_apps
                    and _app_name not in _PASSTHROUGH_APPS
                    and _api_name not in _PASSTHROUGH_APIS):
                return predictor.predict_read(
                    _app_name, _api_name, data, None,
                    _success_schema(_app_name, _api_name))
            real = orig_request(_app_name, _api_name, client=client,
                                raise_on_failure=raise_on_failure, show=show, track=track, **data)
            if _app_name == "supervisor" and _api_name == "show_active_task":
                completion = predictor.simulated_task_completion
                if completion and isinstance(real, dict):
                    real = {
                        **real,
                        "status": completion["status"],
                        "answer": completion["answer"],
                    }
            if passthrough:
                try:
                    predictor.note_read(_app_name, _api_name, data, real)
                except Exception:
                    pass
                return real
            # Read-grounded prefix: before any write, and for any app no write has touched, the read
            # is exact.  Only a read of a dirty app *after* a write is predicted (post-write state).
            if not predictor.first_write_seen or _app_name not in predictor.dirty_apps:
                try:
                    predictor.note_read(_app_name, _api_name, data, real)
                except Exception:
                    pass
                return real
            # read after a write to this app -> predict the post-write result from pre-write + history
            return predictor.predict_read(_app_name, _api_name, data, real,
                                          _success_schema(_app_name, _api_name))
        # write -> predicted (no real mutation), marks the app dirty. A predicted REJECTION is surfaced
        # to the agent exactly like the real requester: raised as the plain real-format Exception when
        # raise_on_failure is not explicitly False, else returned as the failure body dict.
        try:
            return predictor.predict_write(_app_name, _api_name, data,
                                           _success_schema(_app_name, _api_name))
        except WMPredictedHTTPError as exc:
            if raise_on_failure is False:
                return exc.body
            raise Exception(str(exc)) from None

    world.requester.request = wm_request
    predictor.bind_app_catalog(_available_app_names(world))
    predictor.bind_query_interface(AppWorldWMQuery(orig_request, predictor))
    return orig_request


def appworld_flat_tool_docs(available_apps) -> Dict[str, Dict[str, Any]]:
    """Return fixed API documentation for the task's substantive apps."""
    out: Dict[str, Dict[str, Any]] = {}
    for app in available_apps or ():
        endpoints = {}
        for api, doc in sorted(_api_docs(str(app)).items()):
            endpoints[api] = {
                "method": doc.get("method"),
                "description": doc.get("description"),
                "parameters": doc.get("parameters") or [],
                "response_schemas": doc.get("response_schemas") or {},
            }
        out[str(app)] = endpoints
    return out


__all__ = [
    "AppWorldWMPredictor",
    "AppWorldWMQuery",
    "install_wm_requester",
    "run_agentic_read_loop",
    "_api_method",
    "_is_write",
    "_success_schema",
    "_failure_schema",
    "_available_app_names",
    "appworld_flat_tool_docs",
    "WMPredictedHTTPError",
]
