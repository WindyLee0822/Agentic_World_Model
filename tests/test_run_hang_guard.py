"""Regression: experiments/run.py `_run_tasks` must not hang on a single wedged task.

A real AppWorld task can wedge a worker forever (non-terminating agent API loop). The sweep's
hang-guard waits in slices and, if NO task completes within ``TTS_RUN_STALL_TIMEOUT_S``, abandons
the remaining futures and returns the completed records instead of blocking forever.
"""
from __future__ import annotations

import importlib.util
import json
import os
import threading
import types
from pathlib import Path

import pytest

_RUN_PATH = Path(__file__).resolve().parents[1] / "experiments" / "run.py"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("experiments_run", _RUN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spec(idx):
    return types.SimpleNamespace(index=idx, subset="test/L2")


def test_run_tasks_hang_guard_abandons_wedged_task(monkeypatch):
    run = _load_run_module()

    release = threading.Event()
    monkeypatch.setenv("TTS_RUN_STALL_TIMEOUT_S", "2")  # abandon if nothing finishes in 2s

    def fake_run_task_cells(payload):
        env_name, spec, cells, args, usable_only = payload
        if spec.index == 2:  # the "wedged" task: blocks past the stall timeout
            release.wait(timeout=30)
            return []
        return [{"task_id": f"t{spec.index}", "success": True}]

    monkeypatch.setattr(run, "_run_task_cells", fake_run_task_cells)

    specs = [_spec(0), _spec(1), _spec(2), _spec(3)]
    args = types.SimpleNamespace(workers=4, max_tasks_per_child=0)

    try:
        # env_name != "appworld" -> ThreadPoolExecutor (no process spawn in the test)
        records = run._run_tasks("dummy", specs, ["cell"], args, False, "test")
    finally:
        release.set()  # let the leaked worker thread exit promptly

    got = {r["task_id"] for r in records}
    assert got == {"t0", "t1", "t3"}  # the 3 healthy tasks returned, wedged t2 abandoned


def test_run_tasks_all_complete(monkeypatch):
    run = _load_run_module()
    monkeypatch.setenv("TTS_RUN_STALL_TIMEOUT_S", "10")

    def fake_run_task_cells(payload):
        _, spec, _, _, _ = payload
        return [{"task_id": f"t{spec.index}", "success": True}]

    monkeypatch.setattr(run, "_run_task_cells", fake_run_task_cells)
    specs = [_spec(i) for i in range(5)]
    args = types.SimpleNamespace(workers=4, max_tasks_per_child=0)
    records = run._run_tasks("dummy", specs, ["cell"], args, False, "test")
    assert {r["task_id"] for r in records} == {f"t{i}" for i in range(5)}


def test_cell_sidecars_survive_whole_task_hang(monkeypatch, tmp_path):
    run = _load_run_module()

    release = threading.Event()
    monkeypatch.setenv("TTS_RUN_STALL_TIMEOUT_S", "1")
    args = types.SimpleNamespace(
        workers=2,
        max_tasks_per_child=0,
        _cell_checkpoint_dir=str(tmp_path / "cells"),
    )

    def fake_run_task_cells(payload):
        env_name, spec, _, worker_args, _ = payload
        run._write_cell_checkpoint(worker_args, {
            "env": env_name,
            "subset": spec.subset,
            "spec_subset": spec.subset,
            "spec_index": spec.index,
            "method": "ours",
            "n_rollouts": 1,
            "task_id": f"t{spec.index}",
            "success": True,
        })
        release.wait(timeout=30)
        return []

    monkeypatch.setattr(run, "_run_task_cells", fake_run_task_cells)
    try:
        records = run._run_tasks(
            "dummy", [_spec(0), _spec(1)], ["cell"], args, False, "test")
    finally:
        release.set()

    assert {record["task_id"] for record in records} == {"t0", "t1"}
    assert len(args._last_run_status["abandoned_tasks"]) == 2


def test_task_cell_failure_preserves_other_cells(monkeypatch, tmp_path):
    run = _load_run_module()
    args = types.SimpleNamespace(_cell_checkpoint_dir=str(tmp_path / "cells"))
    spec = _spec(7)

    monkeypatch.setattr(run, "make_env_from_spec", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        run, "_cfg_for",
        lambda env_name, spec, method, N, args, recursion_mode, L: types.SimpleNamespace(N=N))

    def fake_run_decision(env, config, rollout_cache):
        if config.N == 2:
            raise RuntimeError("cell exploded")
        record = {
            "env": "dummy",
            "subset": spec.subset,
            "method": "ours",
            "n_rollouts": config.N,
            "task_id": f"t{spec.index}",
            "success": True,
        }
        return types.SimpleNamespace(
            record=types.SimpleNamespace(to_dict=lambda record=record: dict(record)))

    monkeypatch.setattr(run, "run_decision", fake_run_decision)
    records = run._run_task_cells((
        "dummy",
        spec,
        [("ours", 1, None, None), ("ours", 2, None, None), ("ours", 4, None, None)],
        args,
        False,
    ))

    assert [record["n_rollouts"] for record in records] == [1, 4]
    assert len(run._read_cell_sidecars(args._cell_checkpoint_dir, "record")) == 2
    errors = run._read_cell_sidecars(args._cell_checkpoint_dir, "error")
    assert len(errors) == 1
    assert errors[0]["n_rollouts"] == 2
    assert errors[0]["error_type"] == "RuntimeError"


def test_checkpoint_io_failure_does_not_mark_successful_cell_failed(monkeypatch, tmp_path):
    run = _load_run_module()
    args = types.SimpleNamespace(_cell_checkpoint_dir=str(tmp_path / "cells"))
    spec = _spec(3)

    monkeypatch.setattr(run, "make_env_from_spec", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        run, "_cfg_for",
        lambda env_name, spec, method, N, args, recursion_mode, L: types.SimpleNamespace(N=N))
    monkeypatch.setattr(
        run, "run_decision",
        lambda *args, **kwargs: types.SimpleNamespace(record=types.SimpleNamespace(
            to_dict=lambda: {
                "env": "dummy", "subset": spec.subset, "method": "ours",
                "n_rollouts": 1, "task_id": "t3", "success": True,
            })))
    monkeypatch.setattr(
        run, "_write_cell_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk unavailable")))

    records = run._run_task_cells((
        "dummy", spec, [("ours", 1, None, None)], args, False))

    assert len(records) == 1 and records[0]["success"] is True
    assert run._read_cell_sidecars(args._cell_checkpoint_dir, "error") == []


def test_task_cell_retries_transient_failure_without_missing_n(monkeypatch, tmp_path):
    run = _load_run_module()
    args = types.SimpleNamespace(
        _cell_checkpoint_dir=str(tmp_path / "cells"),
        cell_retries=1,
    )
    spec = _spec(9)
    attempts = {}

    monkeypatch.setattr(run, "make_env_from_spec", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        run, "_cfg_for",
        lambda env_name, spec, method, N, args, recursion_mode, L: types.SimpleNamespace(N=N))

    def fake_run_decision(env, config, rollout_cache):
        attempts[config.N] = attempts.get(config.N, 0) + 1
        if config.N == 2 and attempts[config.N] == 1:
            raise IndexError("transient evaluation indexing failure")
        record = {
            "env": "dummy", "subset": spec.subset, "method": "ours",
            "n_rollouts": config.N, "task_id": "t9", "success": True,
        }
        return types.SimpleNamespace(
            record=types.SimpleNamespace(to_dict=lambda record=record: dict(record)))

    monkeypatch.setattr(run, "run_decision", fake_run_decision)
    records = run._run_task_cells((
        "dummy",
        spec,
        [("ours", 1, None, None), ("ours", 2, None, None), ("ours", 4, None, None)],
        args,
        False,
    ))

    assert [record["n_rollouts"] for record in records] == [1, 2, 4]
    assert attempts[2] == 2
    assert run._read_cell_sidecars(args._cell_checkpoint_dir, "error") == []


def test_tau2_domain_records_resume_under_tau2_run(tmp_path):
    run = _load_run_module()
    records_path = tmp_path / "records.jsonl"
    expected_cells = {("b2", n) for n in (1, 2, 4, 8, 16)}
    records = [
        {
            "env": "tau2_airline",
            "subset": "airline",
            "spec_subset": "airline",
            "spec_index": 46,
            "method": method,
            "n_rollouts": n,
        }
        for method, n in sorted(expected_cells)
    ]
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    loaded, done = run._load_checkpoint(str(records_path))
    assert done[("tau2", "airline", 46)] == expected_cells

    manifest = run._coverage_manifest(
        "tau2",
        [types.SimpleNamespace(subset="airline", index=46)],
        loaded,
        expected_cells,
        {},
        [],
    )
    assert manifest["complete_tasks"] == 1
    assert manifest["missing_tasks"] == 0
    assert manifest["expected_decisions"] == 5
    assert manifest["recorded_decisions"] == 5
    assert manifest["missing_decisions"] == 0
    assert {record["env"] for record in loaded} == {"tau2_airline"}


def test_missing_cells_are_metric_failures_without_polluting_checkpoint():
    run = _load_run_module()
    specs = [_spec(0), _spec(1)]
    expected_cells = {("ours", 1), ("ours", 4)}
    records = [
        {
            "env": "dummy", "run_env": "dummy", "subset": "test/L2",
            "spec_subset": "test/L2", "spec_index": 0, "method": "ours",
            "n_rollouts": 1, "success": True, "reward": 1.0,
            "n_spawns": 1, "max_depth": 1, "n_reads": 1, "n_commits": 1,
            "simulator_writes": 0, "prediction_correct": True, "tokens": 10,
            "cost": 0.1, "cache_hits": 0, "wall_clock_s": 1.0,
        },
        {
            "env": "dummy", "run_env": "dummy", "subset": "test/L2",
            "spec_subset": "test/L2", "spec_index": 1, "method": "ours",
            "n_rollouts": 1, "success": False, "reward": 0.0,
            "n_spawns": 1, "max_depth": 1, "n_reads": 1, "n_commits": 1,
            "simulator_writes": 0, "prediction_correct": False, "tokens": 10,
            "cost": 0.1, "cache_hits": 0, "wall_clock_s": 1.0,
        },
        {
            "env": "dummy", "run_env": "dummy", "subset": "test/L2",
            "spec_subset": "test/L2", "spec_index": 1, "method": "ours",
            "n_rollouts": 4, "success": True, "reward": 1.0,
            "n_spawns": 4, "max_depth": 1, "n_reads": 1, "n_commits": 1,
            "simulator_writes": 0, "prediction_correct": True, "tokens": 40,
            "cost": 0.4, "cache_hits": 0, "wall_clock_s": 4.0,
        },
    ]

    metric_records = run._records_with_missing_failures(
        "dummy", specs, records, expected_cells)
    manifest = run._coverage_manifest(
        "dummy", specs, records, expected_cells, {}, [])
    rows = {
        row["n_rollouts"]: row
        for row in run.aggregate(metric_records, ["method", "n_rollouts"])
    }

    assert len(records) == 3
    assert len(metric_records) == 4
    assert manifest["expected_decisions"] == 4
    assert manifest["recorded_decisions"] == 3
    assert manifest["missing_decisions"] == 1
    assert manifest["expected_cells"] == [
        {"method": "ours", "n_rollouts": 1},
        {"method": "ours", "n_rollouts": 4},
    ]
    assert rows[1]["success_rate"] == 0.5
    assert rows[1]["imputed_failures"] == 0
    assert rows[4]["success_rate"] == 0.5
    assert rows[4]["recorded_decisions"] == 1
    assert rows[4]["imputed_failures"] == 1
