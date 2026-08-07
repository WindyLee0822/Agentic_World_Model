"""WM-only verifier: self-consistency confidence for the AppWorld success judge.

The selection signal must come from the world model alone (candidate trajectories contain write
actions and may never be executed on the real env).  ``predict_success`` therefore stays an LLM
judge, but with ``TTS_APPWORLD_VERIFIER_SAMPLES >= 2`` it self-consistency-ensembles K rubric
verdicts and returns the fraction voting SUCCESS — a calibrated confidence that damps the optimistic
false positives a single verdict produces.  These tests use a fake chat client (no API access).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from tts.env.appworld_wm import AppWorldWMPredictor, AppWorldWMQuery
from tts.selection import Selector
from tts.trajectory import RewardInfo, RolloutResult, Trajectory


def _load_experiment_runner():
    path = Path(__file__).parents[1] / "experiments" / "run.py"
    spec = importlib.util.spec_from_file_location("test_experiments_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_app_activity_uses_catalog_without_treating_every_app_as_required():
    chat = _ScriptChat(["SUCCESS"])
    pred = AppWorldWMPredictor(chat)
    pred.bind_app_catalog(("gmail", "phone", "spotify"))
    pred.note_read("gmail", "login", {}, {"access_token": "secret"})
    pred.note_read("gmail", "search_threads", {}, {"threads": []})
    pred.note_read("phone", "show_events", {}, {"events": []})

    assert pred.active_task_apps() == ("gmail", "phone")


def test_self_consistency_returns_vote_fraction(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_SAMPLES", "4")
    chat = _ScriptChat(["SUCCESS", "FAIL", "SUCCESS", "FAIL"])
    conf = _pred(chat).predict_success("task", ["code"], ["out"])
    assert chat.calls == 4
    assert conf == 0.5


def test_grounded_verifier_python_can_compute_over_query_results(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_GROUNDED", "1")
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_PYTHON", "1")
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    seen = []

    def query_fn(app, api, **args):
        seen.append((app, api, args))
        return {"cart_entries": [{"product_id": 101}]}

    code = """PYTHON_FINAL:
remaining = {
    item["product_id"]
    for item in data["reads"][0]["result"]["cart_entries"]
}
print_json({"success": remaining == {101, 102}})
"""
    chat = _GroundedChat([
        'PYTHON_READ:\nprint_json({"queries":[{"app":"amazon","api":"show_cart","args":{}}]})',
        code,
    ])
    pred = AppWorldWMPredictor(chat)

    confidence = pred.predict_success(
        "Keep exactly products 101 and 102 in the cart",
        ["code"],
        ["out"],
        query_fn=query_fn,
    )

    assert seen == [("amazon", "show_cart", {})]
    assert confidence == 0.0
    assert pred.n_python_calls == 2
    assert pred.n_python_failures == 0


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
    monkeypatch.setenv("TTS_APPWORLD_WM_TOOL_MODE", "text")
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


def test_reference_verifier_normalizes_direct_json_false(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_REFERENCE", "1")
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)

    confidence = AppWorldWMPredictor(
        _GroundedChat(['{"success":false}'])
    ).predict_success("t", ["c"], ["o"], reference_checklist="CHECKS")

    assert confidence == 0.0


def test_reference_verifier_enabled_by_default_with_explicit_opt_out(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_REFERENCE", raising=False)
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    # Match OfficeBench: a supplied native checklist is audited by default.
    conf = AppWorldWMPredictor(_ScriptChat(["FAIL"])).predict_success(
        "t", ["c"], ["o"], reference_checklist="CHECKS")
    assert conf == 0.0

    # Explicit opt-out restores the blind prompt-only ablation.
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_REFERENCE", "0")
    blind = AppWorldWMPredictor(_ScriptChat(["SUCCESS"])).predict_success(
        "t", ["c"], ["o"], reference_checklist="CHECKS")
    assert blind == 1.0


def test_reference_verifier_uses_uniform_read_trace_loop(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_REFERENCE", raising=False)
    monkeypatch.setenv("TTS_APPWORLD_WM_TOOL_MODE", "text")
    chat = _GroundedChat(["TRACE venmo", "VERDICT: FAILURE"])
    pred = AppWorldWMPredictor(chat)
    pred.transition_log.record(
        {"app": "venmo", "api": "approve_payment_request", "args": {"id": 7}},
        '{"message":"approved"}',
        ["venmo", "venmo.approve_payment_request"],
        applied=True,
    )
    query = AppWorldWMQuery(lambda *args, **kwargs: {}, pred)
    pred.bind_query_interface(query)

    conf = pred.predict_success(
        "Approve the correct request", ["code"], ["out"],
        reference_checklist="CHECKS:\n- approved_request_id == 8")

    assert conf == 0.0
    assert query.traces == 1
    assert chat.calls == 2


def test_reference_write_log_keeps_tail_and_long_arguments(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_WRITE_ARG_CHARS", "500")
    pred = AppWorldWMPredictor(_ScriptChat(["FAIL"]))
    pred.write_history = [
        ("gmail", "send_email", {"subject": f"message-{i}", "body": "x" * 250}, {"message": "ok"})
        for i in range(65)
    ]

    log = pred._write_log(cap=10)

    assert "intermediate writes omitted" in log
    assert "message-0" in log and "message-64" in log
    assert "x" * 200 in log


def test_reference_verifier_self_consistency(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_REFERENCE", "1")
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_SAMPLES", "4")
    chat = _ScriptChat(["SUCCESS", "FAIL", "SUCCESS", "SUCCESS"])
    conf = AppWorldWMPredictor(chat).predict_success(
        "t", ["c"], ["o"], reference_checklist="CHECKS")
    assert conf == 0.75


def test_reference_verifier_can_compute_exact_write_set_with_python(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_REFERENCE", "1")
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_PYTHON", "1")
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    code = """PYTHON:
required = set(data["gold"]["sections"]["private_data"]["product_ids"])
written = {
    write["args"]["product_id"]
    for write in data["writes"]
    if write["name"] == "amazon.delete_from_cart"
}
print(f"MISSING = {sorted(required - written)}")
print("VERDICT: SUCCESS" if written == required else "VERDICT: FAIL")
"""
    chat = _GroundedChat([code])
    pred = AppWorldWMPredictor(chat)
    pred.write_history = [
        ("amazon", "delete_from_cart", {"product_id": 101}, {"message": "ok"}),
    ]

    confidence = pred.predict_success(
        "Remove products 101 and 102",
        ["code"],
        ["out"],
        reference_checklist=(
            "CHECKS:\n- deleted_product_ids == [101, 102]\n"
            'GOLD VALUES (private_data): {"product_ids": [101, 102]}'
        ),
    )

    assert confidence == 0.0
    assert pred.n_python_calls == 1
    assert pred.n_python_failures == 0
    assert pred.n_python_forced == 0
    assert chat.calls == 1


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


def test_percheck_coverage_aware_penalizes_unknown_checks(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_PERCHECK", "1")
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_COVERAGE_AWARE", "1")
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_UNKNOWN_CREDIT", "0.25")
    chat = _StatusChat(["MET", "UNKNOWN", "UNKNOWN", "UNKNOWN"])
    pred = AppWorldWMPredictor(chat)

    conf = pred.predict_success(
        "t", ["c"], ["o"],
        reference_checks=(["c1", "c2", "c3", "c4"], "gold"),
    )

    assert conf == pytest.approx(0.4375)
    assert pred.last_percheck_breakdown == {
        "met": 1,
        "unmet": 0,
        "unknown": 3,
        "failed": 0,
        "total": 4,
        "coverage_aware": True,
        "unknown_credit": 0.25,
    }


def test_appworld_gpt54_bon_defaults_are_coverage_aware_and_agent_fair(monkeypatch):
    from argparse import Namespace

    _configure_appworld_bon = _load_experiment_runner()._configure_appworld_bon

    for name in (
        "TTS_APPWORLD_VERIFIER_PERCHECK",
        "TTS_APPWORLD_VERIFIER_REFERENCE",
        "TTS_APPWORLD_VERIFIER_COVERAGE_AWARE",
        "TTS_APPWORLD_VERIFIER_UNKNOWN_CREDIT",
        "TTS_APPWORLD_INSTRUCTION_PREFLIGHTS",
        "TTS_SELECTION_CONFIDENCE_MARGIN",
        "TTS_APPWORLD_PREGROUND_CLEAN_READS",
        "TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS",
        "TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY",
    ):
        monkeypatch.delenv(name, raising=False)
    args = Namespace(
        agent="llm_wm",
        llm_agent="gpt-5.4",
        no_verifier_reference=False,
        verifier_percheck=False,
        appworld_instruction_preflights=None,
        success_score="simulator_success",
        _success_score_explicit=False,
        predict_all_actions=True,
        _predict_all_actions_explicit=False,
        simulation_only=False,
    )

    _configure_appworld_bon(args)

    assert args.verifier_percheck is True
    assert args.appworld_instruction_preflights is False
    assert args.success_score == "simulator_confidence_guarded"
    assert args.predict_all_actions is True
    assert os.environ["TTS_APPWORLD_VERIFIER_COVERAGE_AWARE"] == "1"
    assert os.environ["TTS_APPWORLD_VERIFIER_UNKNOWN_CREDIT"] == "0.25"
    assert os.environ["TTS_APPWORLD_INSTRUCTION_PREFLIGHTS"] == "0"
    assert os.environ["TTS_SELECTION_CONFIDENCE_MARGIN"] == "0.1"
    assert os.environ["TTS_APPWORLD_PREGROUND_CLEAN_READS"] == "1"
    assert os.environ["TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS"] == "1"
    assert os.environ["TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY"] == "0"


def test_appworld_gpt54_bon_allows_explicit_noncomparable_preflight_ablation(monkeypatch):
    from argparse import Namespace

    _configure_appworld_bon = _load_experiment_runner()._configure_appworld_bon

    monkeypatch.delenv("TTS_APPWORLD_INSTRUCTION_PREFLIGHTS", raising=False)
    args = Namespace(
        agent="llm_wm",
        llm_agent="gpt-5.4",
        no_verifier_reference=False,
        verifier_percheck=False,
        appworld_instruction_preflights=True,
        success_score="simulator_success",
        _success_score_explicit=False,
        predict_all_actions=True,
        _predict_all_actions_explicit=False,
        simulation_only=False,
    )

    _configure_appworld_bon(args)

    assert args.appworld_instruction_preflights is True
    assert os.environ["TTS_APPWORLD_INSTRUCTION_PREFLIGHTS"] == "1"


def test_percheck_verifier_python_receives_structured_gold_and_writes(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_PERCHECK", "1")
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_PYTHON", "1")
    code = """PYTHON:
required = set(data["gold"]["sections"]["private_data"]["product_ids"])
written = {
    write["args"]["product_id"]
    for write in data["writes"]
    if write["api"] == "delete_from_cart"
}
print_json({"status": "MET" if written == required else "UNMET"})
"""
    pred = AppWorldWMPredictor(_GroundedChat([code]))
    pred.write_history = [
        ("amazon", "delete_from_cart", {"product_id": 101}, {"message": "ok"}),
    ]

    confidence = pred.predict_success(
        "Remove products 101 and 102",
        ["code"],
        ["out"],
        reference_checks=(
            ["deleted_product_ids match private_data.product_ids"],
            'private_data: {"product_ids": [101, 102]}',
        ),
    )

    assert confidence == 0.0
    assert pred.n_python_calls == 1
    assert pred.n_python_failures == 0


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
    monkeypatch.setenv("TTS_APPWORLD_WM_TOOL_MODE", "text")
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


def test_percheck_python_can_compute_over_exact_read_facts(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_PERCHECK", "1")
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_PYTHON", "1")
    seen = []

    def read_fn(app, api, **args):
        seen.append((app, api, args))
        return {"friends": ["a@example.com", "b@example.com"]}

    code = """PYTHON_FINAL:
existing = set(data["reads"][0]["result"]["friends"])
removed = {
    write["args"]["user_email"]
    for write in data["writes"]
    if write["api"] == "remove_friend"
}
print_json({"status": "MET" if existing == removed else "UNMET"})
"""
    chat = _GroundedChat([
        'PYTHON_READ:\nprint_json({"queries":[{"app":"venmo","api":"show_friends","args":{}}]})',
        code,
    ])
    pred = AppWorldWMPredictor(chat)
    pred.write_history = [
        ("venmo", "remove_friend", {"user_email": "a@example.com"}, {"message": "ok"}),
    ]

    confidence = pred.predict_success(
        "Remove all friends",
        ["code"],
        ["out"],
        reference_checks=(["friends == []"], "(none)"),
        real_read_fn=read_fn,
    )

    assert seen == [("venmo", "show_friends", {})]
    assert confidence == 0.0
    assert pred.n_python_calls == 2
    assert pred.n_python_failures == 0


def test_bound_query_uses_uniform_agentic_success_loop_by_default(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_WM_AGENTIC", raising=False)
    monkeypatch.setenv("TTS_APPWORLD_WM_TOOL_MODE", "text")
    monkeypatch.delenv("TTS_APPWORLD_VERIFIER_SAMPLES", raising=False)
    chat = _GroundedChat([
        "TRACE venmo",
        "VERDICT: SUCCESS",
    ])
    pred = AppWorldWMPredictor(chat)
    pred.transition_log.record(
        {"app": "venmo", "api": "approve_payment_request", "args": {"id": 6098}},
        '{"message":"approved"}',
        ["venmo", "venmo.approve_payment_request"],
        applied=True,
    )
    query = AppWorldWMQuery(lambda *args, **kwargs: {}, pred)
    pred.bind_query_interface(query)

    confidence = pred.predict_success("Approve request 6098", ["code"], ["out"])

    assert confidence == 1.0
    assert query.traces == 1
    assert chat.calls == 2
