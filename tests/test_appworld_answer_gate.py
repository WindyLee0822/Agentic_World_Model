from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tts.env.real_appworld import (
    _answer_gate_failed,
    _answers_match,
    _rollout_hard_invalid,
    load_task_answer,
    load_task_checklist,
    load_task_checks,
)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (" YES ", "yes"),
        ('"yes"', "yes"),
        ("335", 335),
        (12855.779, 12855.78),
        (["b", "a"], ["a", "b"]),
        (False, False),
        (0, 0),
    ],
)
def test_answer_matching_follows_appworld_normalization(actual, expected):
    assert _answers_match(actual, expected)


def test_answer_loader_distinguishes_missing_from_json_null(tmp_path, monkeypatch):
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    ground_truth = tmp_path / "data" / "tasks" / "task-1" / "ground_truth"
    ground_truth.mkdir(parents=True)

    assert load_task_answer("task-1") == (False, None)

    (ground_truth / "answer.json").write_text("null", encoding="utf-8")
    assert load_task_answer("task-1") == (True, None)

    (ground_truth / "answer.json").write_text("0", encoding="utf-8")
    assert load_task_answer("task-1") == (True, 0)


def test_checklist_and_structured_gold_include_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    ground_truth = tmp_path / "data" / "tasks" / "task-1" / "ground_truth"
    ground_truth.mkdir(parents=True)
    (ground_truth / "test_data.json").write_text(
        json.dumps([{"requirement": "assert answers match."}]),
        encoding="utf-8",
    )
    (ground_truth / "private_data.json").write_text("{}", encoding="utf-8")
    (ground_truth / "public_data.json").write_text("{}", encoding="utf-8")
    (ground_truth / "answer.json").write_text(json.dumps(335), encoding="utf-8")

    assert "GOLD VALUES (answer): 335" in load_task_checklist("task-1")
    checks, gold = load_task_checks("task-1")
    assert checks == ["assert answers match."]
    assert "answer: 335" in gold


def test_answer_gate_rejects_missing_or_wrong_required_answer():
    missing = SimpleNamespace(simulated_task_completion=None)
    wrong = SimpleNamespace(simulated_task_completion={"status": "success", "answer": 336})
    matching = SimpleNamespace(simulated_task_completion={"status": "fail", "answer": "335"})

    assert _answer_gate_failed(missing, (True, 335))
    assert _answer_gate_failed(wrong, (True, 335))
    assert not _answer_gate_failed(matching, (True, 335))


def test_answer_gate_ignores_missing_key_or_json_null():
    predictor = SimpleNamespace(simulated_task_completion=None)

    assert not _answer_gate_failed(predictor, (False, None))
    assert not _answer_gate_failed(predictor, (True, None))


def test_rollout_gate_rejects_missing_completion_without_answer_key():
    missing = SimpleNamespace(simulated_task_completion=None)
    completed = SimpleNamespace(
        simulated_task_completion={"status": "success", "answer": None})

    assert _rollout_hard_invalid(missing, (False, None))
    assert not _rollout_hard_invalid(completed, (False, None))
