"""Regression: experiments/run.py `_run_tasks` must not hang on a single wedged task.

A real AppWorld task can wedge a worker forever (non-terminating agent API loop). The sweep's
hang-guard waits in slices and, if NO task completes within ``TTS_RUN_STALL_TIMEOUT_S``, abandons
the remaining futures and returns the completed records instead of blocking forever.
"""
from __future__ import annotations

import importlib.util
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
