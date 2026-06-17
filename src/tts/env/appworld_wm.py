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
        self.history: List[Tuple[str, str, dict, Any]] = []
        self.write_history: List[Tuple[str, str, dict, Any]] = []
        self.dirty_apps: set = set()
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
        self._deadline = (time.monotonic() + time_budget_s) if time_budget_s and time_budget_s > 0 else None
        self._hard_deadline = (time.monotonic() + hard_timeout_s) if hard_timeout_s and hard_timeout_s > 0 else None
        self.budget_exhausted = False
        self.hard_timed_out = False

    def _check_hard_deadline(self) -> None:
        """Raise once the rollout's HARD wall-clock deadline passes, to break a runaway agent loop."""
        if self._hard_deadline is not None and time.monotonic() > self._hard_deadline:
            self.hard_timed_out = True
            raise WMHardTimeout(
                f"AppWorld WM rollout exceeded hard timeout {self.hard_timeout_s:.0f}s "
                f"(likely a non-terminating agent loop); aborting the code cell."
            )

    def _over_budget(self) -> bool:
        """True once this rollout's SOFT WM-prediction budget (wall-clock or call count) is spent."""
        if self.budget_exhausted:
            return True
        if self._deadline is not None and time.monotonic() > self._deadline:
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

    def note_read(self, app: str, api: str, args: dict, result: Any) -> None:
        self.n_reads += 1
        self.history.append((app, api, dict(args), result))

    def predict_write(self, app: str, api: str, args: dict, schema: Any) -> Any:
        """Predict a write API's response with the LLM (the faithful path — like the τ² WM).

        Every write is LLM-generated (no synthesised template), so the response matches what the real
        API returns (e.g. ``follow_artist`` → ``{"message": "Artist followed."}``).  To bound the
        rate-limit (429) pressure of write-heavy bursts (e.g. a loop that follows 19 artists), a
        **status/message-only** write — whose success response is a fixed template independent of the
        args — is predicted with a *stable* prompt (no per-call args/history), so the chat client's
        deterministic response cache (the τ²-bench rate-limit lever) folds the whole burst into **one**
        LLM call; the message is still LLM-generated.  Substantive writes (that echo args / return an
        id) use the full per-call prompt.  ``_synth_success`` is only a fallback if the LLM call fails.
        """
        self.n_writes += 1
        self.first_write_seen = True
        self._check_hard_deadline()
        if self._over_budget():
            # rollout budget spent -> skip the LLM (and even the api-doc lookup), return the instant
            # structurally-valid fallback so a runaway write loop finishes instead of issuing
            # thousands of sequential predictions.
            result = _synth_success(schema)
            self.history.append((app, api, dict(args), result))
            self.write_history.append((app, api, dict(args), result))
            self.dirty_apps.add(app)
            return result
        doc = _api_doc(app, api)
        desc = (doc.get("description") or "").strip()[:300]
        if _is_trivial_schema(schema):
            # args-independent success response -> stable prompt so identical calls hit the cache.
            prompt = (
                f"You are a world model of the AppWorld '{app}' app's backing database. The write API "
                f"'{app}.{api}' just executed SUCCESSFULLY. Return ONLY the exact JSON success response "
                f"the real API returns, matching its message wording exactly (e.g. a 'follow artist' "
                f"API returns {{\"message\": \"Artist followed.\"}}).\n"
                f"Description: {desc}\n"
                f"Success response schema: {json.dumps(schema, default=str)[:500]}\n"
                f"Return ONLY the JSON object. No prose."
            )
        else:
            prompt = (
                f"You are a world model simulating the AppWorld '{app}' app's backing database. Predict "
                f"the EXACT JSON response that this WRITE API call would return, as if it executed on the "
                f"current state — matching what the real API returns.\n\n"
                f"API: {app}.{api}\nDescription: {desc}\n"
                f"Success response schema (field name -> example value/type): {json.dumps(schema, default=str)[:500]}\n\n"
                f"Recent prior API calls (these may have changed state):\n{self._history_text()}\n\n"
                f"This call: {app}.{api}({json.dumps(args, default=str)[:400]})\n\n"
                f"Return ONLY a single JSON object: the realistic SUCCESS response, with concrete values "
                f"consistent with the arguments and the schema (e.g. a created record's id as an integer, "
                f"echoing relevant argument values). No prose."
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
        self.history.append((app, api, dict(args), result))
        self.write_history.append((app, api, dict(args), result))
        self.dirty_apps.add(app)
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

    def predict_success(self, instruction: str, code_blocks: List[str], outputs: List[str]) -> float:
        """WM-only selection signal: confidence in [0,1] that the agent COMPLETED the task.

        No gold/test checkers exist at selection time and the candidate's writes must NOT be executed
        on the real env, so this is purely the world model's own judgement of its realized plan.

        * ``TTS_APPWORLD_VERIFIER_SAMPLES <= 1`` (default): the original single binary verdict, i.e.
          ``0.0``/``1.0`` — behaviour-preserving.
        * ``K >= 2``: **self-consistency** — K rubric-grounded verdicts are sampled at
          ``TTS_APPWORLD_VERIFIER_TEMP`` (default 0.7) and the confidence is the fraction voting
          SUCCESS.  Forcing the judge to check every sub-task and averaging independent samples damps
          the optimistic false positives a single verdict produces (the cause of the verifier's
          precision collapse as N grows).  Pair with ``--success-score simulator_confidence`` so
          best-of-N ranks on this calibrated probability instead of the binary flag.
        """
        steps = "\n\n".join(
            f"# step {i}\n{c}\nOutput: {str(o)[-300:]}" for i, (c, o) in enumerate(zip(code_blocks, outputs))
        )[:6000]
        samples = _env_int("TTS_APPWORLD_VERIFIER_SAMPLES", 1)
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
        # write -> predicted (no real mutation), marks the app dirty
        return predictor.predict_write(_app_name, _api_name, data, _success_schema(_app_name, _api_name))

    world.requester.request = wm_request
    return orig_request


__all__ = ["AppWorldWMPredictor", "install_wm_requester", "_api_method", "_is_write", "_success_schema"]
