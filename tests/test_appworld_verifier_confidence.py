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
