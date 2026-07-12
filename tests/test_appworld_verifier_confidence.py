"""WM-only verifier: self-consistency confidence for the AppWorld success judge.

The selection signal must come from the world model alone (candidate trajectories contain write
actions and may never be executed on the real env).  ``predict_success`` therefore stays an LLM
judge, but with ``TTS_APPWORLD_VERIFIER_SAMPLES >= 2`` it self-consistency-ensembles K rubric
verdicts and returns the fraction voting SUCCESS — a calibrated confidence that damps the optimistic
false positives a single verdict produces.  These tests use a fake chat client (no API access).
"""

from __future__ import annotations

import pytest

from tts.env.appworld_wm import AppWorldWMPredictor
from tts.selection import Selector
from tts.trajectory import RewardInfo, RolloutResult, Trajectory


class _ScriptChat:
    """Returns successive canned verdict strings (cycling) and counts calls."""

    def __init__(self, verdicts):
        self._verdicts = verdicts
        self.calls = 0

    def complete(self, messages, **kwargs):
        v = self._verdicts[self.calls % len(self._verdicts)]
        self.calls += 1
        if isinstance(v, Exception):
            raise v
        return f"...\nVERDICT: {v}"


def _pred(chat):
    return AppWorldWMPredictor(chat)


def test_default_single_verdict_is_binary(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    s = _pred(_ScriptChat(["SUCCESS"])).predict_success("task", ["code"], ["out"])
    f = _pred(_ScriptChat(["FAIL"])).predict_success("task", ["code"], ["out"])
    assert s == 1.0 and f == 0.0


def test_default_makes_one_call(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    chat = _ScriptChat(["SUCCESS"])
    _pred(chat).predict_success("task", ["code"], ["out"])
    assert chat.calls == 1


def test_self_consistency_returns_vote_fraction(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_SAMPLES", "4")
    chat = _ScriptChat(["SUCCESS", "FAIL", "SUCCESS", "FAIL"])
    conf = _pred(chat).predict_success("task", ["code"], ["out"])
    assert chat.calls == 4
    assert conf == 0.5


def test_self_consistency_strong_majority(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_SAMPLES", "5")
    chat = _ScriptChat(["SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS", "FAIL"])
    conf = _pred(chat).predict_success("task", ["code"], ["out"])
    assert conf == pytest.approx(0.8)


def test_exceptions_do_not_count_and_zero_when_all_fail(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_SAMPLES", "3")
    # two valid (1 success) + one raising -> confidence = 1/2 over the valid samples
    chat = _ScriptChat(["SUCCESS", RuntimeError("boom"), "FAIL"])
    conf = _pred(chat).predict_success("task", ["code"], ["out"])
    assert conf == 0.5

    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_SAMPLES", "2")
    allbad = _pred(_ScriptChat([RuntimeError("x")])).predict_success("t", ["c"], ["o"])
    assert allbad == 0.0


def _roll(index, reward, success, cost=0.0):
    return RolloutResult(index=index, trajectory=Trajectory(steps=[], final_message=str(index)),
                         predicted=RewardInfo(reward=reward, success=success, grounded=False),
                         cost=cost)


def test_confidence_scorer_ranks_by_reward():
    # All three pass the binary flag (tie under simulator_success); confidence breaks the tie.
    rollouts = [_roll(0, 0.6, True), _roll(1, 0.8, True), _roll(2, 0.4, False)]
    idx, _ = Selector("simulator_confidence").select(rollouts)
    assert idx == 1  # highest confidence


def test_simulator_success_ties_break_arbitrarily():
    # Under the binary scorer the two 'True' rollouts tie -> lowest cost/index wins (idx 0), even
    # though rollout 1 has higher confidence -> motivates simulator_confidence.
    rollouts = [_roll(0, 0.6, True), _roll(1, 0.95, True)]
    idx, _ = Selector("simulator_success").select(rollouts)
    assert idx == 0


class _GroundedChat:
    """Drives the agentic verifier: emits a JSON query, then a JSON verdict (cycling per pass)."""

    def __init__(self, script):
        self._script = script  # list of raw strings returned per call
        self.calls = 0
        self.queries = []

    def complete(self, messages, **kwargs):
        out = self._script[self.calls % len(self._script)]
        self.calls += 1
        return out


def test_grounded_verifier_queries_then_judges(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_GROUNDED", "1")
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    seen = []

    def query_fn(app, api, **args):
        seen.append((app, api, args))
        return {"payment_request_id": 6098, "status": "approved"}

    # turn 1: issue a read query; turn 2: verdict SUCCESS
    chat = _GroundedChat([
        '{"action":"query","app":"venmo","api":"show_payment_request","args":{"payment_request_id":6098}}',
        '{"action":"verdict","success":true,"reason":"approved"}',
    ])
    pred = AppWorldWMPredictor(chat)
    pred.write_history = [("venmo", "approve_payment_request", {"payment_request_id": 6098}, {"message": "ok"})]
    conf = pred.predict_success("Approve request 6098", ["code"], ["out"], query_fn=query_fn)
    assert conf == 1.0
    assert seen == [("venmo", "show_payment_request", {"payment_request_id": 6098})]


def test_grounded_verifier_can_return_fail(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_GROUNDED", "1")
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    chat = _GroundedChat(['{"action":"verdict","success":false,"reason":"not done"}'])
    conf = AppWorldWMPredictor(chat).predict_success("t", ["c"], ["o"], query_fn=lambda *a, **k: {})
    assert conf == 0.0


def test_grounded_verifier_self_consistency_fraction(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_GROUNDED", "1")
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_SAMPLES", "4")
    # each pass immediately verdicts; alternate success/fail -> 0.5 confidence
    chat = _GroundedChat([
        '{"action":"verdict","success":true}',
        '{"action":"verdict","success":false}',
    ])
    conf = AppWorldWMPredictor(chat).predict_success("t", ["c"], ["o"], query_fn=lambda *a, **k: {})
    assert conf == 0.5


def test_grounded_disabled_falls_back_to_prompt_judge(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_GROUNDED", "0")
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    # query_fn provided but grounding disabled -> uses the binary prompt judge (VERDICT: ...)
    conf = AppWorldWMPredictor(_ScriptChat(["SUCCESS"])).predict_success(
        "t", ["c"], ["o"], query_fn=lambda *a, **k: {})
    assert conf == 1.0


def test_reference_verifier_audits_write_log(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_REFERENCE", "1")
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    chat = _ScriptChat(["FAIL"])  # auditor finds a violation
    pred = AppWorldWMPredictor(chat)
    pred.write_history = [("venmo", "add_friend", {"user_email": "a@b.com"}, {"message": "ok"})]
    conf = pred.predict_success("reset friends", ["c"], ["o"], reference_checklist="CHECKS:\n- unfriend X")
    assert conf == 0.0
    assert chat.calls == 1  # one reference-audit call


def test_reference_verifier_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_REFERENCE", raising=False)
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    # checklist provided but flag off -> falls back to the prompt-only judge
    conf = AppWorldWMPredictor(_ScriptChat(["SUCCESS"])).predict_success(
        "t", ["c"], ["o"], reference_checklist="CHECKS")
    assert conf == 1.0


def test_reference_verifier_self_consistency(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_REFERENCE", "1")
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_SAMPLES", "4")
    chat = _ScriptChat(["SUCCESS", "FAIL", "SUCCESS", "SUCCESS"])
    conf = AppWorldWMPredictor(chat).predict_success(
        "t", ["c"], ["o"], reference_checklist="CHECKS")
    assert conf == 0.75


class _StatusChat:
    """Returns a STATUS:<x> line per call, cycling through a script (for the per-check auditor)."""

    def __init__(self, statuses):
        self._statuses = statuses
        self.calls = 0

    def complete(self, messages, **kwargs):
        s = self._statuses[self.calls % len(self._statuses)]
        self.calls += 1
        return f"...analysis...\nSTATUS: {s}"


def test_percheck_graded_met_over_decidable(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_PERCHECK", "1")
    # 4 checks: MET, MET, UNMET, UNKNOWN -> decidable=3, met=2 -> score 2/3
    chat = _StatusChat(["MET", "MET", "UNMET", "UNKNOWN"])
    pred = AppWorldWMPredictor(chat)
    pred.write_history = [("venmo", "add_friend", {"x": 1}, {"message": "ok"})]
    conf = pred.predict_success("t", ["c"], ["o"],
                                reference_checks=(["c1", "c2", "c3", "c4"], "gold"))
    assert chat.calls == 4
    assert abs(conf - 2/3) < 1e-9


def test_percheck_all_unknown_is_neutral(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_PERCHECK", "1")
    chat = _StatusChat(["UNKNOWN", "UNKNOWN"])
    conf = AppWorldWMPredictor(chat).predict_success(
        "t", ["c"], ["o"], reference_checks=(["c1", "c2"], "gold"))
    assert conf == 0.5


def test_percheck_disabled_falls_back(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_PERCHECK", raising=False)
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_REFERENCE", raising=False)
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    # per-check off and reference off -> prompt-only judge
    conf = AppWorldWMPredictor(_ScriptChat(["SUCCESS"])).predict_success(
        "t", ["c"], ["o"], reference_checks=(["c1"], "gold"))
    assert conf == 1.0


class _ReadThenVerdictChat:
    """First turn issues a query JSON, second turn returns a verdict JSON (per check)."""

    def __init__(self, status):
        self.status = status
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.calls % 2 == 1:
            return '{"action":"query","app":"venmo","api":"show_friends","args":{}}'
        return f'{{"action":"verdict","status":"{self.status}"}}'


def test_percheck_with_reads_queries_then_verdicts(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_PERCHECK", "1")
    seen = []

    def read_fn(app, api, **a):
        seen.append((app, api))
        return {"friends": [1, 2, 3]}

    chat = _ReadThenVerdictChat("MET")
    pred = AppWorldWMPredictor(chat)
    pred.write_history = [("venmo", "remove_friend", {"user_email": "x"}, {})]
    conf = pred.predict_success("reset friends", ["c"], ["o"],
                                reference_checks=(["deleted match to_unfriend"], "gold"),
                                real_read_fn=read_fn)
    assert seen == [("venmo", "show_friends")]   # the per-check sub-pass actually read the DB
    assert conf == 1.0                            # 1 check, MET -> met/decidable = 1/1

