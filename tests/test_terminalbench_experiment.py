"""Tests for the isolated ``experiments/run.py --env terminalbench`` path."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from tts.env import terminalbench_experiment as tbexp

_RUN_PATH = Path(__file__).resolve().parents[1] / "experiments" / "run.py"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _decision(path: Path) -> None:
    _write_json(path, {
        "method": "ours",
        "selected_index": 1,
        "committed": True,
        "rollouts": [
            {
                "index": 0,
                "confidence": 0.2,
                "n_reads": 2,
                "n_spawns": 1,
                "max_depth": 1,
                "run": {"commands": ["cat input"], "total_tokens": 5},
            },
            {
                "index": 1,
                "confidence": 0.9,
                "n_reads": 3,
                "n_spawns": 2,
                "max_depth": 1,
                "run": {"commands": ["printf ok > out"], "total_tokens": 7},
            },
        ],
    })


def test_parse_tb1_normalizes_native_result(tmp_path):
    cell = tmp_path / "cell"
    run = cell / "run"
    trial = run / "hello-world" / "trial-1"
    _write_json(run / "results.json", {
        "results": [{
            "task_id": "hello-world",
            "trial_name": "trial-1",
            "is_resolved": True,
            "trial_started_at": "2026-01-01T00:00:00+00:00",
            "trial_ended_at": "2026-01-01T00:00:10+00:00",
        }],
    })
    _decision(trial / "agent-logs" / "tts-terminalbench.json")

    record = tbexp._parse_tb1(cell, "ours", 2, "uncertainty")[0]

    assert record["env"] == "terminalbench"
    assert record["subset"] == "v1"
    assert record["task_id"] == "hello-world"
    assert record["success"] is True and record["reward"] == 1.0
    assert record["n_rollouts"] == 2 and record["selected_index"] == 1
    assert record["n_reads"] == 5 and record["n_spawns"] == 3
    assert record["tokens"] == 12 and record["wall_clock_s"] == 10.0


def test_parse_tb2_normalizes_native_result(tmp_path):
    cell = tmp_path / "cell"
    trial = cell / "job" / "cancel-async-tasks__abc"
    _write_json(trial / "result.json", {
        "task_name": "cancel-async-tasks",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:20Z",
        "verifier_result": {"rewards": {"reward": 1.0}},
    })
    _decision(trial / "agent" / "tts-terminalbench.json")

    record = tbexp._parse_tb2(cell, "ours", 2, "uncertainty")[0]

    assert record["subset"] == "v2"
    assert record["task_id"] == "cancel-async-tasks"
    assert record["success"] is True and record["reward"] == 1.0
    assert record["wall_clock_s"] == 20.0


def test_b2_selects_best_official_isolated_trial():
    base = {
        "env": "terminalbench",
        "subset": "v1",
        "task_id": "task",
        "n_reads": 0,
        "n_spawns": 0,
        "max_depth": 0,
        "tokens": 4,
        "cost": 0.0,
        "wall_clock_s": 3.0,
        "native_result_path": "result",
    }
    attempts = [
        [{**base, "reward": 0.0, "success": False, "committed_action": "[\"bad\"]"}],
        [{**base, "reward": 1.0, "success": True, "committed_action": "[\"good\"]"}],
        [{**base, "reward": 0.5, "success": False, "committed_action": "[\"partial\"]"}],
    ]

    record = tbexp._b2_records(attempts, 3)[0]

    assert record["method"] == "b2"
    assert record["n_rollouts"] == 3
    assert record["selected_index"] == 1
    assert record["reward"] == 1.0 and record["success"] is True
    assert record["committed_action"] == "[\"good\"]"
    assert record["tokens"] == 12 and record["wall_clock_s"] == 9.0


def test_b2_treats_a_missing_native_trial_as_zero_reward():
    base = {
        "env": "terminalbench",
        "subset": "v2",
        "task_id": "task",
        "reward": 1.0,
        "success": True,
        "committed_action": "[\"good\"]",
        "n_reads": 0,
        "n_spawns": 0,
        "max_depth": 0,
        "tokens": 4,
        "cost": 0.0,
        "wall_clock_s": 3.0,
        "native_result_path": "result",
    }

    record = tbexp._b2_records([[base], []], 2)[0]

    assert record["reward"] == 1.0
    assert record["selected_index"] == 0
    assert record["native_result_paths"] == ["result", None]


def test_harbor_reward_requires_an_unambiguous_aggregate():
    assert tbexp._reward_from_harbor({"verifier_result": None}) == 0.0
    assert tbexp._reward_from_harbor({
        "verifier_result": {"rewards": {"score": 0.5}},
    }) == 0.5

    try:
        tbexp._reward_from_harbor({
            "verifier_result": {"rewards": {"a": 0.5, "b": 1.0}},
        })
    except ValueError as error:
        assert "multiple reward components" in str(error)
    else:
        raise AssertionError("ambiguous Harbor rewards must fail explicitly")


def test_terminalbench_task_filters_can_target_one_generation():
    values = ["shared-*", "1:hello-world", "2:cancel-async-tasks"]

    assert tbexp._tasks_for_version(values, 1) == ["shared-*", "hello-world"]
    assert tbexp._tasks_for_version(values, 2) == ["shared-*", "cancel-async-tasks"]


def test_terminalbench_timeout_scales_with_world_model_rollouts():
    assert tbexp._agent_timeout_args(1, "ours", 8) == [
        "--global-timeout-multiplier", "16",
    ]
    assert tbexp._agent_timeout_args(2, "b1", 4) == [
        "--agent-timeout-multiplier", "8",
    ]
    assert tbexp._agent_timeout_args(2, "b0", 16) == []


def test_terminalbench_resume_requires_a_complete_native_cell(tmp_path):
    tb1 = tmp_path / "v1"
    _write_json(tb1 / "run" / "run_metadata.json", {
        "dataset_size": 2,
        "end_time": None,
    })
    _write_json(tb1 / "run" / "results.json", {"results": [{"task_id": "one"}]})
    assert tbexp._cell_result_exists(tb1, 1) is False

    _write_json(tb1 / "run" / "run_metadata.json", {
        "dataset_size": 2,
        "end_time": "2026-01-01T00:01:00+00:00",
    })
    _write_json(tb1 / "run" / "results.json", {
        "results": [{"task_id": "one"}, {"task_id": "two"}],
    })
    assert tbexp._cell_result_exists(tb1, 1) is True

    tb2 = tmp_path / "v2"
    _write_json(tb2 / "job" / "result.json", {
        "finished_at": None,
        "n_total_trials": 2,
        "stats": {
            "n_completed_trials": 1,
            "n_running_trials": 1,
            "n_pending_trials": 0,
        },
    })
    assert tbexp._cell_result_exists(tb2, 2) is False

    _write_json(tb2 / "job" / "result.json", {
        "finished_at": "2026-01-01T00:01:00",
        "n_total_trials": 2,
        "stats": {
            "n_completed_trials": 2,
            "n_running_trials": 0,
            "n_pending_trials": 0,
        },
    })
    assert tbexp._cell_result_exists(tb2, 2) is True


def test_sweep_reuses_one_b0_pool_for_all_b2_n(monkeypatch, tmp_path):
    calls = []
    completed = []

    def fake_cell(args, *, version, method, n, seed, cell_name):
        calls.append((version, method, n, seed, cell_name))
        reward = 1.0 if seed % 4 == 3 else 0.0
        return [{
            "method": method,
            "env": "terminalbench",
            "subset": f"v{version}",
            "task_id": "task",
            "n_rollouts": n,
            "reward": reward,
            "success": reward == 1.0,
            "committed_action": json.dumps([f"seed-{seed}"]),
            "n_reads": 0,
            "n_spawns": 0,
            "max_depth": 0,
            "tokens": 1,
            "cost": 0.0,
            "wall_clock_s": 1.0,
            "native_result_path": "result",
        }]

    monkeypatch.setattr(tbexp, "_run_native_cell", fake_cell)
    args = SimpleNamespace(
        terminalbench_versions="1,2",
        methods="ours,b2",
        N="1,2,4",
        depth_sweep=False,
        seed_base=0,
        out=str(tmp_path),
    )

    records = tbexp.run_terminalbench_sweep(args, on_cell_done=completed.append)

    b0_calls = [call for call in calls if call[1] == "b0"]
    ours_calls = [call for call in calls if call[1] == "ours"]
    assert len(b0_calls) == 8  # max(N)=4 attempts for each benchmark version
    assert len(ours_calls) == 6  # three N cells for each benchmark version
    assert len(completed) == 14
    assert tbexp.terminalbench_native_cell_count(args) == 14
    assert len(records) == 12  # three B2 + three Ours records per version
    assert {
        (record["subset"], record["method"], record["n_rollouts"])
        for record in records
    } == {
        (f"v{version}", method, n)
        for version in (1, 2)
        for method in ("b2", "ours")
        for n in (1, 2, 4)
    }


def test_terminalbench_cli_returns_before_existing_env_registry(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("experiments_run_terminalbench", _RUN_PATH)
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    called = []

    monkeypatch.setattr(run, "_run_terminalbench_cli", lambda args: called.append(args.env))
    monkeypatch.setattr(
        run,
        "available_real_envs",
        lambda: (_ for _ in ()).throw(AssertionError("existing registry path was entered")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--env", "terminalbench", "--out", str(tmp_path)],
    )

    run.main()

    assert called == ["terminalbench"]
