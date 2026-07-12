"""World-model interception for AppWorld: read APIs exact, write APIs LLM-predicted.

The official ReAct code agent solves a task by emitting Python that calls ``apis.<app>.<endpoint>(...)``;
every such call funnels through the single chokepoint ``requester.request(_app_name, _api_name, **data)``
(injected into the execution shell).  This module patches that chokepoint so the agent interacts with a
**world-model simulator of the backing database** instead of mutating it:

* **read** endpoints (any non-mutating HTTP method — ``GET``/``HEAD``/``OPTIONS``) and the
  control-plane apps (``supervisor`` / ``api_docs`` / ``admin``) execute exactly against the real
  backing SQLite DBs (read-only, no mutation);
* **write** endpoints (``POST`` / ``PATCH`` / ``PUT`` / ``DELETE``) are **predicted** by an LLM from the
  call args + the endpoint's ``response_schemas`` + the running call history — never executed, so the real
  backing DB is never written.

This is the AppWorld analog of the τ² ``environment.get_response`` interception.  Because writes are
predicted (not executed), ``N`` candidate trajectories can be sampled without mutating the env; the
selected one is committed by re-executing its realized code on a fresh real world and scoring with
``AppWorld.evaluate()``.

Read-after-write (τ² logic): once an app has been written, a subsequent ``GET`` to it is **predicted**
by combining the exact *pre-write* real result with the write history (``predict_read``) — so reads
reflect the agent's prior writes instead of returning stale state.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

#: control-plane apps whose calls always pass through to the real env (task signalling / docs / admin).
_PASSTHROUGH_APPS = {"supervisor", "api_docs", "admin"}

#: auth/session endpoints that must execute for REAL even though they are POSTs: the agent uses the
#: returned access token for every subsequent (real) read, so a *predicted* (fake) token would make
#: all those reads fail auth.  These authenticate an existing account; they don't mutate task state.
_PASSTHROUGH_APIS = {"login", "logout"}

#: HTTP methods that MUTATE the backing DB.  Only these are world-model predicted; every other method
#: (``GET``/``HEAD``/``OPTIONS`` and any unknown/missing method, which defaults to ``GET``) is a read
#: and executes exactly against the real DB — so the whole read-only prefix before the first write is
#: fully read-grounded.
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


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
    """True iff the endpoint mutates state (so the world model must predict it instead of executing)."""
    if app_name in _PASSTHROUGH_APPS or api_name in _PASSTHROUGH_APIS:
        return False
    return _api_method(app_name, api_name) in _MUTATING_METHODS


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


class WMHardTimeout(RuntimeError):
    """Raised inside a WM prediction once the rollout's hard wall-clock deadline passes.

    It deliberately propagates out of the agent's (possibly non-terminating) code cell so the cell
    ends and the rollout can finish, instead of spinning forever on a ``while True`` API loop that
    AppWorld's own per-cell SIGALRM timeout fails to abort.
    """


def _api_doc(app_name: str, api_name: str) -> Dict[str, Any]:
    from appworld.requester import _get_api_name_to_doc  # type: ignore

    try:
        return _get_api_name_to_doc(app_name, include_private_apis=True).get(api_name, {}) or {}
    except Exception:
        return {}


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
        super().__init__(f"Response status code is {self.status_code}:\n{json.dumps(body, default=str)}")


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


class AppWorldWMPredictor:
    """LLM predictor for AppWorld **write** API responses, **dirty reads**, and a success verdict.

    * ``predict_write`` returns a structurally-valid response dict (guided by the endpoint's success
      schema) so the agent's downstream code can consume it, and marks the written app *dirty*.
    * ``predict_read`` handles a **read after a write** (a GET to a dirty app): it combines the exact
      *pre-write* real result with the write history to predict the *post-write* read — the τ²
      dirty-read logic — instead of returning stale state.
    * ``note_read`` records exact (clean) reads so predictions stay consistent with what the agent saw.
    """

    def __init__(self, chat_client, time_budget_s: Optional[float] = None,
                 max_llm_calls: Optional[int] = None, hard_timeout_s: Optional[float] = None) -> None:
        self._chat = chat_client
        # The real env freezes "now" to a fixed per-task datetime (freezegun on ``task.datetime``),
        # so every server-generated timestamp (created_at/added_at/updated_at/...) is DETERMINISTIC and
        # identical between the WM rollout and the real commit world. Surfacing it to the predictor lets
        # a predicted write/read reproduce those timestamps EXACTLY instead of the LLM guessing a date
        # (the τ²-style fixed-value approach). Set by ``install_wm_requester`` from ``world.task``.
        self.frozen_datetime: Optional[str] = None
        self.history: List[Tuple[str, str, dict, Any]] = []
        self.write_history: List[Tuple[str, str, dict, Any]] = []
        self.dirty_apps: set = set()
        # Set of (app, api, frozen-arg-key) for writes that SUCCEEDED in this rollout. A repeat of the
        # same state-changing write to the same target is the most common, history-inferable rejection
        # ("already approved/followed/liked/...") — detecting it routes the prediction through the
        # rejection-aware path so the WM can reproduce the real API's failure instead of a fake success.
        self.applied_writes: set = set()
        self.n_writes = 0
        self.first_write_seen = False
        self.n_reads = 0
        self.n_dirty_reads = 0
        self.n_llm_calls = 0
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

    def note_read(self, app: str, api: str, args: dict, result: Any) -> None:
        self.n_reads += 1
        self.history.append((app, api, dict(args), result))

    @staticmethod
    def _write_key(app: str, api: str, args: dict) -> tuple:
        """Identity of a write's *target* (api + non-auth args) for duplicate/idempotency detection."""
        ids = tuple(sorted(
            (k, json.dumps(v, default=str, sort_keys=True))
            for k, v in (args or {}).items()
            if k not in ("access_token", "password", "token")
        ))
        return (app, api, ids)

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
            f"API: {app}.{api}\nDescription: {desc}\n"
            f"SUCCESS response schema: {json.dumps(schema, default=str)[:400]}\n"
            f"FAILURE response schema: {json.dumps(fail_schema, default=str)[:200]}\n"
            f"{self._frozen_time_text()}"
            f"Recent prior API calls (writes change state; reads show state):\n{self._history_text()}\n\n"
            f"This call: {app}.{api}({json.dumps(args, default=str)[:400]})\n\n"
            f"Return ONLY a JSON object of the form "
            f'{{"outcome": "success"|"failure", "status_code": 200 or 422, "response": <the success or '
            f'failure response JSON>}}. Prefer SUCCESS unless the history CLEARLY shows a violation. No prose.'
        )
        try:
            self.n_llm_calls += 1
            resp = self._chat.complete([{"role": "user", "content": prompt}],
                                       temperature=0.0, max_tokens=800)
        except Exception:
            resp = ""
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
            self.history.append((app, api, dict(args), result))
            self.write_history.append((app, api, dict(args), result))
            self.dirty_apps.add(app)
            self.applied_writes.add(key)
            return result
        doc = _api_doc(app, api)
        desc = (doc.get("description") or "").strip()[:300]
        if _is_trivial_schema(schema) and not is_repeat:
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
            try:
                self.n_llm_calls += 1
                resp = self._chat.complete([{"role": "user", "content": prompt}],
                                           temperature=0.0, max_tokens=800)
            except Exception:  # never let a prediction failure crash the rollout
                resp = ""
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
                raise WMPredictedHTTPError(status_code, body)
        self.history.append((app, api, dict(args), result))
        self.write_history.append((app, api, dict(args), result))
        self.dirty_apps.add(app)
        self.applied_writes.add(key)
        return result

    def predict_read(self, app: str, api: str, args: dict, pre_write_result: Any, schema: Any) -> Any:
        """A read AFTER a write (GET to a dirty app): combine the pre-write state + write history.

        ``pre_write_result`` is the EXACT result the real (unmutated) backing DB returns; the write
        history holds the predicted writes since.  The model applies those writes to the pre-write
        result and returns the post-write read — so reads reflect the agent's prior writes.
        """
        self.n_dirty_reads += 1
        self._check_hard_deadline()
        if self._over_budget():
            # rollout budget spent -> return the exact pre-write state without an LLM call.
            self._fallback_calls += 1
            self._check_runaway()
            self.history.append((app, api, dict(args), pre_write_result))
            return pre_write_result
        doc = _api_doc(app, api)
        desc = (doc.get("description") or "").strip()[:200]
        prompt = (
            f"You are a world model of the AppWorld '{app}' app's backing database. A READ API was called "
            f"AFTER some WRITE calls changed the state. Predict the EXACT JSON result the read returns NOW, "
            f"by applying those writes to the pre-write result.\n\n"
            f"READ API: {app}.{api}\nDescription: {desc}\n"
            f"This call: {app}.{api}({json.dumps(args, default=str)[:300]})\n\n"
            f"{self._frozen_time_text()}"
            f"PRE-WRITE result (what the real DB returns, BEFORE the writes below were applied):\n"
            f"{json.dumps(pre_write_result, default=str)[:2500]}\n\n"
            f"WRITE calls applied since (in order):\n{self._writes_text(app)}\n\n"
            f"Return ONLY the JSON result reflecting those writes (e.g. an added/updated/removed record). "
            f"If none of the writes affect this read, return the pre-write result UNCHANGED. No prose."
        )
        try:
            self.n_llm_calls += 1
            resp = self._chat.complete([{"role": "user", "content": prompt}],
                                       temperature=0.0, max_tokens=2000)
        except Exception:
            return pre_write_result
        result = _parse_json(resp)
        if result is None:
            result = pre_write_result  # fall back to the exact pre-write state
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
        """The agent's COMPLETE write calls (full args), for reference-grounded auditing."""
        if not self.write_history:
            return "(the agent performed no write/mutation calls)"
        wh = self.write_history
        rows = [f"{app}.{api}({json.dumps(args, default=str)[:160]})" for app, api, args, _ in wh[:cap]]
        if len(wh) > cap:
            rows.append(f"... ({len(wh) - cap} more looped/identical writes)")
        return "\n".join(rows)

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
            "SIMULATED (reported success regardless of correctness), so you CANNOT see final state — judge "
            "ONLY from whether the agent's write calls produce the exact required changes.\n"
            "For EACH ground-truth check, work it out explicitly:\n"
            "  REQUIRED = the exact gold ids/values the check demands;\n"
            "  AGENT = the matching ids/values the agent's write calls actually cover (from the write log);\n"
            "  MISSING = REQUIRED not covered by AGENT; EXTRA = AGENT covered but not REQUIRED;\n"
            "  STATUS = MET (MISSING and EXTRA both empty) / UNMET (any MISSING, wrong, or EXTRA change) / "
            "UNKNOWN (cannot be decided from the write log alone — e.g. shuffle order, internal record "
            "bookkeeping, cursor/playing state).\n"
            "VERDICT RULE: FAIL if ANY check is UNMET; otherwise SUCCESS (UNKNOWN checks do NOT fail it).\n\n"
            f"TASK:\n{instruction}\n\n{checklist}\n\n"
            f"AGENT'S COMPLETE WRITE CALLS (full args; loops noted):\n{self._write_log()}\n\n"
            "Fill every check with its REQUIRED / AGENT / MISSING / EXTRA / STATUS, then on the LAST line "
            "answer exactly 'VERDICT: SUCCESS' or 'VERDICT: FAIL'."
        )
        try:
            resp = self._chat.complete([{"role": "user", "content": prompt}],
                                       temperature=temperature, max_tokens=700)
        except Exception:
            return None
        return _is_success_verdict(resp)

    def _reference_score_percheck(self, instruction: str, checks: List[str], gold: str,
                                  temperature: float, read_fn=None, max_queries: int = 3) -> Optional[float]:
        """PER-CHECK reference audit: a focused sub-pass classifies EACH ground-truth check, then aggregate.

        Each check is judged in isolation (it sees only that one check + the gold values + the FULL write
        log) as MET / UNMET / UNKNOWN.  Focused single-check reasoning catches decidable violations the
        all-at-once worksheet rubber-stamps (better precision), and the aggregate is naturally GRADED —
        ``score = MET / (MET + UNMET)`` over the DECIDABLE checks (UNKNOWN excluded) — which gives the
        score-spread the binary worksheet lacks (better coverage / ranking).  Checks that depend on the
        true final state (cursor/playing state, shuffle order, internal record bookkeeping) are UNKNOWN
        and excluded — the simulated-write ceiling, which no decomposition can remove.  All-UNKNOWN →
        0.5 (neutral).  Returns ``None`` only if every sub-pass failed.

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
            return None
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
        try:
            self.n_llm_calls += 1
            resp = self._chat.complete([{"role": "user", "content": prompt}],
                                       temperature=temperature, max_tokens=500)
        except Exception:
            return None
        return self._parse_status(resp)

    def _percheck_with_reads(self, instruction: str, check: str, gold: str, wl: str,
                             read_fn, temperature: float, max_queries: int) -> Optional[str]:
        """Classify ONE check as a small MULTI-TURN loop that may READ the real pre-write DB to ground it.

        ``read_fn(app, api, **args)`` performs an EXACT, read-only query of the real backing DB (the true
        state BEFORE the agent's simulated writes).  The sub-pass uses it to resolve entities / fetch the
        actual starting set / compute a required delta, then judges whether the write log performs the
        exact change THIS check requires relative to that real state + the gold values.
        """
        sys = (
            "You verify ONE ground-truth success check of an AppWorld task against the agent's write log. "
            "The agent's WRITE calls were SIMULATED (NOT applied), so a database READ returns the REAL "
            "state BEFORE those writes. You may issue read-only queries to gather facts (the current "
            "entities, the actual starting set, anything needed to compute the required change), then "
            "decide whether the agent's write calls perform the exact change THIS check requires relative "
            "to that starting state + the gold values.\n"
            "Reply with ONE JSON object per turn:\n"
            '  {"action":"query","app":"<app>","api":"<read_api>","args":{...}}  to read the DB, or\n'
            '  {"action":"verdict","status":"MET"|"UNMET"|"UNKNOWN"}  to finish.\n'
            "MET = writes perform the exact required change (right ids/values, none missing, none wrong/"
            "extra). UNMET = a required id/value is missing or a wrong/extra change is present. UNKNOWN = "
            "depends on the true FINAL state not decidable even with reads (shuffle order, cursor/playing "
            "state, internal record bookkeeping). Use 'api_docs' to find read APIs; query only READ apis."
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
        writes = self._writes_summary()
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
            "Respond with ONE JSON object per turn:\n"
            '  {"action":"query","app":"<app>","api":"<read_api>","args":{...}}  to inspect state, or\n'
            '  {"action":"verdict","success":true|false,"reason":"<short>"}  to finish.\n'
            "Use 'api_docs' (e.g. show_api_descriptions/show_api_doc) if you need to find a read API. "
            "Query only READ apis. Investigate the key facts before you decide; be strict but fair."
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
        action = _parse_json(resp)
        if isinstance(action, dict) and "success" in action:
            return bool(action.get("success"))
        return _is_success_verdict(resp) if resp else None

    def predict_success(self, instruction: str, code_blocks: List[str], outputs: List[str],
                        query_fn=None, reference_checklist: Optional[str] = None,
                        reference_checks: Optional[Tuple[List[str], str]] = None,
                        real_read_fn=None) -> float:
        """WM-only selection signal: confidence in [0,1] that the agent COMPLETED the task.

        No gold/test checkers exist at selection time and the candidate's writes must NOT be executed
        on the real env, so this is purely the world model's own judgement of its realized plan.

        * **Per-check reference auditor** (when ``reference_checks`` is given and
          ``TTS_APPWORLD_VERIFIER_PERCHECK`` != ``0``): a focused sub-pass audits EACH ground-truth check
          in isolation (MET/UNMET/UNKNOWN) and the score is the graded ``MET / (MET + UNMET)`` over the
          decidable checks — sharper per-check precision and a naturally graded score for ranking.
        * **Reference-grounded auditor** (when ``reference_checklist`` is given and
          ``TTS_APPWORLD_VERIFIER_REFERENCE`` != ``0``): one worksheet pass over all checks; FAIL only on
          a write-log-provable violation; ``TTS_APPWORLD_VERIFIER_SAMPLES`` self-consistency samples.
          Both use the task's ground-truth spec → REFERENCE-GROUNDED (τ²-style), never executing the
          candidate or observing its final state.
        * **Read-grounded agentic verifier** (``query_fn`` + ``TTS_APPWORLD_VERIFIER_GROUNDED``): may
          issue read-only DB queries; bounded by predict_read fidelity, off by default.
        * **Prompt-only judge** (default): single binary verdict, or ``K>=2`` rubric self-consistency.
          Pair with ``--success-score simulator_confidence`` to rank on the continuous confidence.
        """
        steps = "\n\n".join(
            f"# step {i}\n{c}\nOutput: {str(o)[-300:]}" for i, (c, o) in enumerate(zip(code_blocks, outputs))
        )[:6000]
        samples = _env_int("TTS_APPWORLD_VERIFIER_SAMPLES", 1)

        # Per-check reference auditor: a focused sub-pass per ground-truth check -> graded met/decidable.
        # When TTS_APPWORLD_VERIFIER_PERCHECK_READ != 0 and a real read fn is available, each check may
        # query the REAL pre-write DB (exact) to ground its judgment (resolve entities / required deltas).
        if reference_checks and os.environ.get("TTS_APPWORLD_VERIFIER_PERCHECK", "0") not in ("0", "false", "False"):
            checks, gold = reference_checks
            temp = _env_float("TTS_APPWORLD_VERIFIER_TEMP", 0.0) if samples > 1 else 0.0
            use_reads = (real_read_fn is not None
                         and os.environ.get("TTS_APPWORLD_VERIFIER_PERCHECK_READ", "1") not in ("0", "false", "False"))
            max_q = _env_int("TTS_APPWORLD_VERIFIER_PERCHECK_QUERIES", 3)
            score = self._reference_score_percheck(
                instruction, checks, gold, temp,
                read_fn=(real_read_fn if use_reads else None), max_queries=max_q)
            if score is not None:
                return score
            # fall through if every sub-pass failed.

        # Reference-grounded auditor: check the write log against the task's ground-truth success spec.
        if reference_checklist and os.environ.get("TTS_APPWORLD_VERIFIER_REFERENCE", "0") not in ("0", "false", "False"):
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
        if query_fn is not None and os.environ.get("TTS_APPWORLD_VERIFIER_GROUNDED", "0") not in ("0", "false", "False"):
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

        if samples <= 1:
            prompt = (
                "You judge whether an AppWorld agent COMPLETED its task, given the task and the code it ran "
                "(with outputs). Note tool results for write actions were simulated, so judge from the plan's "
                "coherence and whether the required steps were performed.\n\n"
                f"TASK:\n{instruction}\n\nAGENT STEPS:\n{steps}\n\n"
                "On the LAST line answer exactly 'VERDICT: SUCCESS' or 'VERDICT: FAIL'."
            )
            try:
                resp = self._chat.complete([{"role": "user", "content": prompt}],
                                           temperature=0.0, max_tokens=400)
            except Exception:
                return 0.0
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
            try:
                resp = self._chat.complete([{"role": "user", "content": prompt}],
                                           temperature=temp, max_tokens=500)
            except Exception:
                continue
            valid += 1
            if _is_success_verdict(resp):
                votes += 1
        if valid == 0:
            return 0.0
        return votes / valid


def install_wm_requester(world, predictor: AppWorldWMPredictor):
    """Patch ``world.requester.request`` so reads are exact, **dirty** reads + writes are WM-predicted.

    Returns the original bound ``request`` (so the caller can restore it).  Routing:

    * **Reads** (any non-mutating method — ``GET``/``HEAD``/``OPTIONS``/unknown — plus the
      control-plane apps ``supervisor``/``api_docs``/``admin`` and the ``login``/``logout`` auth
      endpoints) execute *exactly* against the real backing DBs (read-only, no mutation).
    * **Writes** (``POST``/``PUT``/``PATCH``/``DELETE``) are predicted (no real mutation), marking
      their app *dirty*.
    * A read of a *dirty* app (one a prior write touched) is predicted from its exact pre-write result
      + the write history (``predict_read``) instead of returning stale state.

    **Invariant — read-grounded before the first write:** until the first write happens
    (``predictor.first_write_seen``), no read is ever predicted — the entire read-only prefix is
    served by the real environment.  Dirty-read prediction can only fire *after* a write has run, so
    the agent gathers information from true state before any prediction enters the trajectory.
    """
    orig_request = world.requester.request

    # Pin the predictor's clock to the env's frozen per-task datetime so predicted server timestamps
    # match the real (also-frozen) commit world exactly. Best-effort: never block WM install on it.
    if not predictor.frozen_datetime:
        try:
            predictor.frozen_datetime = world.task.datetime.isoformat()
        except Exception:
            pass

    def wm_request(_app_name, _api_name, client=None, raise_on_failure=None,
                   show=False, track=True, **data):
        if not _is_write(_app_name, _api_name):
            real = orig_request(_app_name, _api_name, client=client,
                                raise_on_failure=raise_on_failure, show=show, track=track, **data)
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
    return orig_request


__all__ = ["AppWorldWMPredictor", "install_wm_requester", "_api_method", "_is_write",
           "_success_schema", "_failure_schema", "WMPredictedHTTPError"]
