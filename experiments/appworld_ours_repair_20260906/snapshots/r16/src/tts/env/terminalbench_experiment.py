"""Unified ``experiments/run.py`` adapter for the official Terminal-Bench harnesses.

The existing AppWorld, OfficeBench, and tau2 paths use in-process ``TaskEnvironment``
implementations. Terminal-Bench owns its Docker lifecycle and verifier inside two native CLIs, so
this adapter deliberately stays out-of-process: it launches ``tb``/Harbor through the repository
script, then normalizes their per-task results into the shared ``records.jsonl`` schema.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUN_SCRIPT = _REPO_ROOT / "scripts" / "run_terminalbench.sh"
_SUPPORTED_METHODS = {"b0", "b1", "b2", "ours"}


def _csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("Terminal-Bench N values must be positive integers")
    return list(dict.fromkeys(values))


def _versions(value: str) -> list[int]:
    versions = _csv_ints(value)
    if any(version not in {1, 2} for version in versions):
        raise ValueError("Terminal-Bench versions must be 1 and/or 2")
    return versions


def _tasks_for_version(values: Optional[Iterable[str]], version: int) -> list[str]:
    tasks = []
    for value in values or []:
        prefix, separator, pattern = value.partition(":")
        if separator and prefix in {"1", "2"}:
            if int(prefix) == version and pattern:
                tasks.append(pattern)
        else:
            tasks.append(value)
    return tasks


def _duration_seconds(start: Optional[str], end: Optional[str]) -> float:
    if not start or not end:
        return 0.0
    try:
        return max(
            0.0,
            (datetime.fromisoformat(end.replace("Z", "+00:00"))
             - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds(),
        )
    except (TypeError, ValueError):
        return 0.0


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _decision_metrics(decision_path: Path) -> dict:
    if not decision_path.exists():
        return {
            "selected_index": 0,
            "committed_action": None,
            "predicted_confidence": None,
            "n_reads": 0,
            "n_spawns": 0,
            "max_depth": 0,
            "tokens": 0,
        }
    decision = _load_json(decision_path)
    rollouts = decision.get("rollouts") or []
    selected_index = int(decision.get("selected_index", 0) or 0)
    selected = (
        rollouts[selected_index]
        if 0 <= selected_index < len(rollouts)
        else {}
    )
    run = selected.get("run") or {}
    return {
        "selected_index": selected_index,
        "committed_action": json.dumps(run.get("commands") or [], ensure_ascii=True),
        "predicted_confidence": (
            float(selected["confidence"]) if selected.get("confidence") is not None else None
        ),
        "n_reads": sum(int(rollout.get("n_reads", 0) or 0) for rollout in rollouts),
        "n_spawns": sum(int(rollout.get("n_spawns", 0) or 0) for rollout in rollouts),
        "max_depth": max(
            (int(rollout.get("max_depth", 0) or 0) for rollout in rollouts),
            default=0,
        ),
        "tokens": sum(
            int((rollout.get("run") or {}).get("total_tokens", 0) or 0)
            for rollout in rollouts
        ),
    }


def _record(
    *,
    version: int,
    method: str,
    n: int,
    task_id: str,
    reward: float,
    success: bool,
    wall_clock_s: float,
    decision_path: Path,
    native_result_path: Path,
    recursion_mode: Optional[str],
) -> dict:
    metrics = _decision_metrics(decision_path)
    confidence = metrics["predicted_confidence"]
    predicted_success = None if method == "b0" or confidence is None else confidence >= 0.5
    return {
        "method": method,
        "env": "terminalbench",
        "subset": f"v{version}",
        "task_id": task_id,
        "n_rollouts": 1 if method == "b0" else n,
        "recursion_mode": recursion_mode if method == "ours" else None,
        "selected_index": metrics["selected_index"],
        "committed_action": metrics["committed_action"],
        "success": bool(success),
        "reward": float(reward),
        "n_spawns": metrics["n_spawns"],
        "max_depth": metrics["max_depth"],
        "depth_histogram": {},
        "n_reads": metrics["n_reads"],
        "n_commits": 1,
        "cache_hits": 0,
        "cache_misses": 0,
        "tokens": metrics["tokens"],
        "cost": 0.0,
        "wall_clock_s": float(wall_clock_s),
        "prediction_correct": (
            None if predicted_success is None else predicted_success == bool(success)
        ),
        "predicted_success": predicted_success,
        "predicted_confidence": confidence,
        "simulator_writes": 0,
        "split": None,
        "level": version,
        "domain": None,
        "spec_subset": f"v{version}",
        "spec_index": task_id,
        "terminalbench_version": version,
        "native_result_path": str(native_result_path),
    }


def _parse_tb1(
    cell_root: Path,
    method: str,
    n: int,
    recursion_mode: Optional[str],
) -> list[dict]:
    run_dir = cell_root / "run"
    result_path = run_dir / "results.json"
    payload = _load_json(result_path)
    records = []
    for result in payload.get("results") or []:
        task_id = str(result["task_id"])
        trial_name = str(result["trial_name"])
        trial_dir = run_dir / task_id / trial_name
        records.append(_record(
            version=1,
            method=method,
            n=n,
            task_id=task_id,
            reward=1.0 if result.get("is_resolved") else 0.0,
            success=bool(result.get("is_resolved")),
            wall_clock_s=_duration_seconds(
                result.get("trial_started_at"),
                result.get("trial_ended_at"),
            ),
            decision_path=trial_dir / "agent-logs" / "tts-terminalbench.json",
            native_result_path=trial_dir / "results.json",
            recursion_mode=recursion_mode,
        ))
    return records


def _reward_from_harbor(result: dict) -> float:
    rewards = ((result.get("verifier_result") or {}).get("rewards") or {})
    value = rewards.get("reward")
    if value is None and not rewards:
        return 0.0
    if value is None:
        numeric = [item for item in rewards.values() if isinstance(item, (int, float))]
        if len(numeric) != 1:
            raise ValueError(
                "Harbor verifier returned multiple reward components without an aggregate "
                f"'reward': {sorted(rewards)}"
            )
        value = numeric[0]
    return float(value or 0.0)


def _parse_tb2(
    cell_root: Path,
    method: str,
    n: int,
    recursion_mode: Optional[str],
) -> list[dict]:
    job_dir = cell_root / "job"
    records = []
    for trial_dir in sorted(path for path in job_dir.iterdir() if path.is_dir()):
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            continue
        result = _load_json(result_path)
        if not result.get("task_name"):
            continue
        reward = _reward_from_harbor(result)
        records.append(_record(
            version=2,
            method=method,
            n=n,
            task_id=str(result["task_name"]),
            reward=reward,
            success=reward >= 1.0,
            wall_clock_s=_duration_seconds(
                result.get("started_at"),
                result.get("finished_at"),
            ),
            decision_path=trial_dir / "agent" / "tts-terminalbench.json",
            native_result_path=result_path,
            recursion_mode=recursion_mode,
        ))
    return records


def _cell_result_exists(cell_root: Path, version: int) -> bool:
    try:
        if version == 1:
            run_root = cell_root / "run"
            metadata = _load_json(run_root / "run_metadata.json")
            results = _load_json(run_root / "results.json")
            expected = int(metadata.get("dataset_size") or 0)
            return (
                bool(metadata.get("end_time"))
                and expected > 0
                and len(results.get("results") or []) == expected
            )

        result = _load_json(cell_root / "job" / "result.json")
        stats = result.get("stats") or {}
        expected = int(result.get("n_total_trials") or 0)
        return (
            bool(result.get("finished_at"))
            and expected > 0
            and int(stats.get("n_completed_trials") or 0) == expected
            and int(stats.get("n_running_trials") or 0) == 0
            and int(stats.get("n_pending_trials") or 0) == 0
        )
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _agent_timeout_args(version: int, method: str, n: int) -> list[str]:
    if method not in {"b1", "ours"}:
        return []
    flag = (
        "--global-timeout-multiplier"
        if version == 1
        else "--agent-timeout-multiplier"
    )
    return [flag, str(max(2, 2 * n))]


def _run_native_cell(
    args,
    *,
    version: int,
    method: str,
    n: int,
    seed: int,
    cell_name: str,
) -> list[dict]:
    cell_root = Path(args.out).resolve() / "terminalbench-native" / f"v{version}" / cell_name
    if not (args.resume and _cell_result_exists(cell_root, version)):
        if cell_root.exists():
            shutil.rmtree(cell_root)
        cell_root.mkdir(parents=True, exist_ok=True)
        command = [
            str(_RUN_SCRIPT),
            "--version", str(version),
            "--agent", "tts",
            "--all",
            "--env-name", args.terminalbench_conda_env,
            "--root", args.terminalbench_root,
        ]
        if args.llm_agent:
            command.extend(["--model", args.llm_agent])
        command.append("--")
        if version == 1:
            command.extend(["--output-path", str(cell_root), "--run-id", "run"])
        else:
            command.extend(["--jobs-dir", str(cell_root), "--job-name", "job"])
        command.extend(["--n-concurrent", str(max(1, int(args.workers)))])
        command.extend(_agent_timeout_args(version, method, n))
        if args.tasks is not None:
            command.extend(["--n-tasks", str(args.tasks)])
        for task in _tasks_for_version(args.terminalbench_task, version):
            command.extend([
                "--task-id" if version == 1 else "--include-task-name",
                task,
            ])

        environment = os.environ.copy()
        environment.update({
            "TTS_TERMINALBENCH_METHOD": method,
            "TTS_TERMINALBENCH_N": str(n),
            "TTS_TERMINALBENCH_TEMPERATURE": str(args.temperature),
            "TTS_TERMINALBENCH_SEED": str(seed),
            "TTS_TERMINALBENCH_MAX_STEPS": str(args.max_steps),
        })
        if args.verifier_samples is not None:
            environment["TTS_TERMINALBENCH_VERIFIER_SAMPLES"] = str(args.verifier_samples)
        subprocess.run(command, cwd=_REPO_ROOT, env=environment, check=True)

    parser = _parse_tb1 if version == 1 else _parse_tb2
    return parser(
        cell_root,
        method,
        n,
        args.recursion_mode if method == "ours" else None,
    )


def _by_task(records: Iterable[dict]) -> dict[str, dict]:
    return {str(record["task_id"]): record for record in records}


def _b2_records(attempts: list[list[dict]], n: int) -> list[dict]:
    selected_attempts = attempts[:n]
    task_maps = [_by_task(records) for records in selected_attempts]
    expected = set().union(*(set(task_map) for task_map in task_maps))

    records = []
    for task_id in sorted(expected):
        template = next(task_map[task_id] for task_map in task_maps if task_id in task_map)
        candidates = []
        for task_map in task_maps:
            if task_id in task_map:
                candidates.append(task_map[task_id])
                continue
            missing = copy.deepcopy(template)
            missing.update({
                "reward": 0.0,
                "success": False,
                "committed_action": None,
                "n_reads": 0,
                "n_spawns": 0,
                "max_depth": 0,
                "tokens": 0,
                "cost": 0.0,
                "wall_clock_s": 0.0,
                "native_result_path": None,
                "native_trial_missing": True,
            })
            candidates.append(missing)
        selected_index = max(
            range(len(candidates)),
            key=lambda index: (
                candidates[index]["reward"],
                not candidates[index].get("native_trial_missing", False),
                -index,
            ),
        )
        selected = copy.deepcopy(candidates[selected_index])
        selected.update({
            "method": "b2",
            "n_rollouts": n,
            "recursion_mode": None,
            "selected_index": selected_index,
            "n_reads": sum(candidate.get("n_reads", 0) for candidate in candidates),
            "n_spawns": sum(candidate.get("n_spawns", 0) for candidate in candidates),
            "max_depth": max(
                (candidate.get("max_depth", 0) for candidate in candidates),
                default=0,
            ),
            "tokens": sum(candidate.get("tokens", 0) for candidate in candidates),
            "cost": sum(candidate.get("cost", 0.0) for candidate in candidates),
            "wall_clock_s": sum(
                candidate.get("wall_clock_s", 0.0) for candidate in candidates
            ),
            "prediction_correct": True,
            "predicted_success": bool(selected["success"]),
            "predicted_confidence": float(selected["reward"]),
            "native_result_path": None,
            "native_result_paths": [
                candidate.get("native_result_path") for candidate in candidates
            ],
        })
        records.append(selected)
    return records


def run_terminalbench_sweep(
    args,
    *,
    on_records: Optional[Callable[[list[dict]], None]] = None,
    on_cell_done: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Run Terminal-Bench cells and return shared-schema records."""
    versions = _versions(args.terminalbench_versions)
    methods = [method.strip().lower() for method in args.methods.split(",") if method.strip()]
    unknown = set(methods) - _SUPPORTED_METHODS
    if unknown:
        raise ValueError(
            f"unsupported Terminal-Bench methods: {sorted(unknown)}; "
            f"use {sorted(_SUPPORTED_METHODS)}"
        )
    ns = _csv_ints(args.N)
    if args.depth_sweep:
        raise ValueError("--depth-sweep is not supported for Terminal-Bench")
    if not _RUN_SCRIPT.exists():
        raise FileNotFoundError(f"Terminal-Bench launcher not found: {_RUN_SCRIPT}")

    all_records: list[dict] = []

    def emit(records: list[dict]) -> None:
        all_records.extend(records)
        if on_records is not None and records:
            on_records(records)

    for version in versions:
        b2_attempts: list[list[dict]] = []
        if "b0" in methods or "b2" in methods:
            attempt_count = max(ns) if "b2" in methods else 1
            for attempt in range(attempt_count):
                attempt_records = _run_native_cell(
                    args,
                    version=version,
                    method="b0",
                    n=1,
                    seed=args.seed_base + attempt,
                    cell_name=f"b2-pool-attempt-{attempt:03d}",
                )
                b2_attempts.append(attempt_records)
                if on_cell_done is not None:
                    label = (
                        f"v{version} b2 pool {attempt + 1}/{attempt_count}"
                        if "b2" in methods else f"v{version} b0"
                    )
                    on_cell_done(label)
                if attempt == 0 and "b0" in methods:
                    emit(attempt_records)
            if "b2" in methods:
                for n in ns:
                    emit(_b2_records(b2_attempts, n))

        for method in methods:
            if method in {"b0", "b2"}:
                continue
            for n in ns:
                records = _run_native_cell(
                    args,
                    version=version,
                    method=method,
                    n=n,
                    seed=args.seed_base,
                    cell_name=f"{method}-n{n}",
                )
                if on_cell_done is not None:
                    on_cell_done(f"v{version} {method} N={n}")
                emit(records)
    return all_records


def terminalbench_native_cell_count(args) -> int:
    """Number of native harness jobs in one Terminal-Bench sweep."""
    versions = _versions(args.terminalbench_versions)
    methods = {
        method.strip().lower() for method in args.methods.split(",") if method.strip()
    }
    ns = _csv_ints(args.N)
    per_version = 0
    if "b2" in methods:
        per_version += max(ns)
    elif "b0" in methods:
        per_version += 1
    per_version += len(ns) * len(methods & {"b1", "ours"})
    return len(versions) * per_version


__all__ = ["run_terminalbench_sweep", "terminalbench_native_cell_count"]
