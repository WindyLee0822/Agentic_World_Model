#!/usr/bin/env python3
"""Run trajectory-level TTS experiments over the REAL environments and log the full metric set.

For each real task we sample ``N`` trajectories against one shared, non-forkable task
environment (executed + scored through the official AppWorld / OfficeBench / tau2 / Terminal-Bench
runners). Each
trajectory is scored by its read-grounded world model (or, for B2, an oracle on a forked
snapshot); the single most-promising trajectory is selected and its **whole action sequence**
is committed.  Per decision we log: success/reward, #spawns, depth distribution, #reads,
#commits, prediction accuracy (predicted vs official committed reward), wall-clock, tokens/cost,
and the simulator-writes counter (must be 0).

Produces the **TTS curve** (success vs ``N``, per method) and a **depth curve** (success & cost
vs recursion depth ``L``, fixed mode), reported overall and per subset; writes JSON/CSV + a
spawn-tree dump.  The two real benchmarks live in different interpreters (AppWorld → Python 3.11
``env``; τ²-bench → Python 3.12 ``dw``); the runner auto-detects the available one or takes ``--env``.

Examples
--------
    python experiments/run.py --env tau2 --tasks 12 --N 1,2,4,8
    APPWORLD_ROOT=/path PYTHONPATH=src python experiments/run.py --env appworld --splits test_normal --levels 1 --tasks 8
    OFFICEBENCH_ROOT=/path PYTHONPATH=src python experiments/run.py --env officebench --levels 1 --tasks 8
    python experiments/run.py --env terminalbench --terminalbench-versions 1,2 --methods ours,b2 --N 1,2,4,8
    python experiments/run.py --depth-sweep --env tau2 --tasks 12
    python experiments/run.py --report results/records.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from typing import Dict, List

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from tts.config import Config, SimulatorConfig, SolverConfig  # noqa: E402
from tts.env import build_dataset, dataset_size, subset_size  # noqa: E402
from tts.env import subsets as env_subsets  # noqa: E402
from tts.env.registry import available_real_envs, make_env, make_env_from_spec  # noqa: E402
from tts.harness import run_decision  # noqa: E402

_METHOD_ORDER = {"b0": 0, "b1": 1, "b2": 2, "ours": 3}


def _quiet_logging() -> None:
    """Silence the noisy tau2 (loguru) / litellm / http DEBUG logs so only the progress bar and
    the result tables are shown.  Must run before tau2 is first imported (done at the top of main).
    """
    import logging

    os.environ.setdefault("LITELLM_LOG", "ERROR")
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("LiteLLM", "litellm", "httpx", "httpcore", "openai", "urllib3", "tau2"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:  # tau2 logs via loguru — drop its handlers entirely (registry dump + per-call spam)
        from loguru import logger as _loguru

        _loguru.remove()
    except Exception:
        pass
    try:
        import litellm

        litellm.suppress_debug_info = True
    except Exception:
        pass


class _Progress:
    """A minimal progress bar over processed tasks (uses ``tqdm`` if present, else a plain fallback)."""

    def __init__(self, total: int, desc: str = "tasks") -> None:
        self.total = max(0, int(total))
        self.n = 0
        self._bar = None
        try:
            from tqdm import tqdm  # type: ignore

            self._bar = tqdm(total=self.total, desc=desc, unit="task")
        except Exception:  # pragma: no cover - fallback when tqdm is unavailable
            self._desc = desc
            self._render()

    def update(self, k: int = 1, postfix: str = "") -> None:
        self.n += k
        if self._bar is not None:
            if postfix:
                self._bar.set_postfix_str(postfix)
            self._bar.update(k)
        else:  # pragma: no cover
            self._render(postfix)

    def _render(self, postfix: str = "") -> None:  # pragma: no cover - fallback only
        pct = (100.0 * self.n / self.total) if self.total else 100.0
        tail = f" {postfix}" if postfix else ""
        sys.stderr.write(f"\r{self._desc}: {self.n}/{self.total} ({pct:5.1f}%){tail}")
        sys.stderr.flush()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        else:  # pragma: no cover
            sys.stderr.write("\n")
            sys.stderr.flush()


def sim_config(args, recursion_mode=None, L=None) -> SimulatorConfig:
    return SimulatorConfig(
        predict_all_actions=bool(getattr(args, "predict_all_actions", True)),
        recursion_mode=recursion_mode or args.recursion_mode, tau=args.tau,
        L=args.L if L is None else L, hard_cap=args.hard_cap, branching=args.branching,
        node_read_limit=args.node_read_limit, caching=not args.no_cache,
        selective_invalidation=not args.no_selective_invalidation,
    )


def _specs_for(env_name, args):
    levels = [int(x) for x in args.levels.split(",")] if args.levels else None
    splits = args.splits.split(",") if args.splits else None
    domains = args.domains.split(",") if args.domains else (["airline"] if env_name == "tau2" else None)
    sub_filter = args.subsets.split(",") if args.subsets else None
    specs = build_dataset(env_name, tasks_per_subset=args.tasks, splits=splits, levels=levels,
                          domains=domains, subsets_filter=sub_filter)
    task_indices = getattr(args, "task_indices", None)
    if task_indices:
        selected = {int(value.strip()) for value in task_indices.split(",") if value.strip()}
        specs = [spec for spec in specs if spec.index in selected]
    return specs


def _cfg_for(env_name, spec, method, N, args, recursion_mode=None, L=None) -> Config:
    return Config(method=method, env=env_name, N=N, seed=args.seed_base + spec.index * 1000,
                  success_score=args.success_score,
                  solver=SolverConfig(agent=args.agent, llm_agent=args.llm_agent,
                                      llm_user=args.llm_user or args.llm_agent,
                                      temperature=args.temperature, max_steps=args.max_steps,
                                      rollout_retries=getattr(args, "rollout_retries", 2),
                                      p_correct=args.p_correct,
                                      self_aggregate=getattr(args, "agent_self_aggregate", False),
                                      aggregation_max_tokens=getattr(
                                          args, "aggregation_max_tokens", 2048),
                                      aggregation_strategy=getattr(
                                          args, "aggregation_strategy", "flat"),
                                      aggregation_group_size=getattr(
                                          args, "aggregation_group_size", 4)),
                  simulator=sim_config(args, recursion_mode, L))


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _nonnegative_int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("0", "false", "False")


def _is_gpt_54(model: str | None) -> bool:
    return "gpt-5.4" in str(model or "").strip().lower().replace("_", "-")


def _configure_appworld_bon(args) -> None:
    """Use robust generation and coverage-aware selection for AppWorld GPT-5.4 sweeps."""
    if getattr(args, "agent", None) != "llm_wm":
        return

    model = (
        getattr(args, "llm_agent", None)
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("WM_MODEL")
        or os.environ.get("TTS_AGENT_MODEL")
    )
    if not _is_gpt_54(model):
        return

    if (
        not getattr(args, "no_verifier_reference", False)
        and "TTS_APPWORLD_VERIFIER_PERCHECK" not in os.environ
    ):
        os.environ["TTS_APPWORLD_VERIFIER_PERCHECK"] = "1"
        os.environ["TTS_APPWORLD_VERIFIER_REFERENCE"] = "1"
        args.verifier_percheck = True

    os.environ.setdefault("TTS_APPWORLD_VERIFIER_COVERAGE_AWARE", "1")
    os.environ.setdefault("TTS_APPWORLD_VERIFIER_UNKNOWN_CREDIT", "0.25")
    os.environ.setdefault("TTS_SELECTION_CONFIDENCE_MARGIN", "0.1")
    os.environ.setdefault("TTS_APPWORLD_PREGROUND_CLEAN_READS", "1")
    os.environ.setdefault("TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS", "1")
    os.environ.setdefault("TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY", "0")

    preflights = getattr(args, "appworld_instruction_preflights", None)
    if preflights is None:
        configured = os.environ.get("TTS_APPWORLD_INSTRUCTION_PREFLIGHTS")
        preflights = (
            configured not in ("0", "false", "False")
            if configured is not None
            else False
        )
        args.appworld_instruction_preflights = preflights
    os.environ["TTS_APPWORLD_INSTRUCTION_PREFLIGHTS"] = "1" if preflights else "0"

    if not getattr(args, "_success_score_explicit", False):
        args.success_score = "simulator_confidence_guarded"


def _configure_officebench_bon(args) -> None:
    """Use atomic-check confidence and guarded nested selection for OfficeBench GPT-5.4."""
    if getattr(args, "agent", None) != "llm_wm":
        return

    # OfficeBench tasks depend heavily on discovering exact filenames and copying source content
    # across apps. Execute pristine read-only actions against the shadow by default; writes and reads
    # after predicted writes remain world-model predictions. Keep predict-all as an explicit ablation
    # (and as the required mode for simulation-only runs).
    if (
        not getattr(args, "_predict_all_actions_explicit", False)
        and not getattr(args, "simulation_only", False)
    ):
        args.predict_all_actions = False

    model = (
        getattr(args, "llm_agent", None)
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("WM_MODEL")
        or os.environ.get("TTS_AGENT_MODEL")
    )
    is_gpt54 = _is_gpt_54(model)
    if (
        getattr(args, "verifier_samples", None) is None
        and "TTS_OFFICEBENCH_VERIFIER_SAMPLES" not in os.environ
        and is_gpt54
    ):
        os.environ["TTS_OFFICEBENCH_VERIFIER_SAMPLES"] = "5"

    if is_gpt54:
        os.environ.setdefault("TTS_OFFICEBENCH_VERIFIER_PERCHECK", "1")
        os.environ.setdefault("TTS_OFFICEBENCH_VERIFIER_UNKNOWN_CREDIT", "0.25")
        os.environ.setdefault("TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT", "0.25")
        os.environ.setdefault("TTS_OFFICEBENCH_COMPARATIVE_RERANK", "1")
        os.environ.setdefault("TTS_OFFICEBENCH_COMPARATIVE_CONFIDENCE_MARGIN", "0.25")
        office_margin = os.environ.get(
            "TTS_OFFICEBENCH_SELECTION_CONFIDENCE_MARGIN", "0.05")
        os.environ["TTS_SELECTION_CONFIDENCE_MARGIN"] = office_margin

    if not getattr(args, "_success_score_explicit", False):
        args.success_score = (
            "simulator_confidence_guarded" if is_gpt54
            else "simulator_confidence"
        )


def _record_run_env(record: dict):
    """Return the CLI environment identity used for resume and coverage."""
    run_env = record.get("run_env")
    if run_env:
        return run_env
    env = record.get("env")
    if isinstance(env, str) and env.startswith("tau2_"):
        return "tau2"
    return env


def _cell_checkpoint_key(record: dict) -> tuple:
    spec_index = record.get("spec_index")
    return (
        _record_run_env(record),
        record.get("spec_subset", record.get("subset")),
        spec_index if spec_index is not None else record.get("task_id"),
        record.get("method"),
        record.get("n_rollouts"),
    )


def _cell_checkpoint_path(directory: str, record: dict, prefix: str = "record") -> str:
    raw = "|".join(str(part) for part in _cell_checkpoint_key(record))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return os.path.join(directory, f"{prefix}-{digest}.json")


def _write_cell_checkpoint(args, record: dict) -> None:
    """Atomically persist one completed cell from inside a worker process.

    A task runs several increasingly expensive N cells in one worker to share its rollout cache.
    Without this sidecar, an exception or hard-kill in a later cell discards every earlier cell because
    the parent receives results only when the whole task future returns.
    """
    directory = getattr(args, "_cell_checkpoint_dir", None)
    if not directory:
        return
    os.makedirs(directory, exist_ok=True)
    path = _cell_checkpoint_path(directory, record)
    tmp = f"{path}.tmp-p{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_cell_error(args, env_name, spec, method, N, error: Exception) -> None:
    """Persist a failed cell's identity and exception so missing records are explainable."""
    directory = getattr(args, "_cell_checkpoint_dir", None)
    if not directory:
        return
    payload = {
        "env": env_name,
        "spec_subset": spec.subset,
        "spec_index": spec.index,
        "method": method,
        "n_rollouts": N,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    os.makedirs(directory, exist_ok=True)
    path = _cell_checkpoint_path(directory, payload, prefix="error")
    tmp = f"{path}.tmp-p{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _report_cell_error(args, env_name, spec, method, N, error: Exception) -> None:
    try:
        _write_cell_error(args, env_name, spec, method, N, error)
    except OSError as checkpoint_err:
        print(f"\n[cell-checkpoint {spec.subset} idx={spec.index} {method} N={N}] failed: "
              f"{type(checkpoint_err).__name__}: {checkpoint_err}", flush=True)
    print(f"\n[cell {spec.subset} idx={spec.index} {method} N={N}] failed: "
          f"{type(error).__name__}: {error}", flush=True)


def _read_cell_sidecars(directory: str, prefix: str) -> List[dict]:
    if not directory or not os.path.isdir(directory):
        return []
    records = []
    for name in sorted(os.listdir(directory)):
        if not name.startswith(prefix + "-") or not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), "r", encoding="utf-8") as fh:
                records.append(json.load(fh))
        except (OSError, ValueError, TypeError):
            continue
    return records


def _run_task_cells(payload) -> List[dict]:
    """Run EVERY (method, N) cell for ONE task, sharing a task-local agent-rollout pool.

    This is the unit of parallel work.  Grouping all cells of a task in one worker keeps the
    fair-comparison guarantee (B0/B1/B2/Ours over this task select from the same rollout pool —
    the cache key already includes the task id, so cross-task sharing never happened anyway) and
    makes each job fully independent, so workers never touch shared mutable state.

    Must be module-level + picklable so it can run under a ``ProcessPoolExecutor``.
    """
    env_name, spec, cells, args, usable_only = payload
    cache: dict = {}
    records: List[dict] = []
    cell_retries = max(0, int(getattr(args, "cell_retries", 2) or 0))
    max_n = max((N for _method, N, _recursion_mode, _L in cells), default=1)
    for (method, N, recursion_mode, L) in cells:
        rec = None
        last_error = None
        for attempt in range(cell_retries + 1):
            try:
                env = make_env_from_spec(spec, usable_only=usable_only)
            except (NotImplementedError, IndexError) as err:
                last_error = err
                break  # a structurally unavailable task will not become available on retry
            except Exception as err:  # noqa: BLE001 - retry transient environment setup failures
                last_error = err
                if attempt < cell_retries:
                    print(f"\n[cell-retry {spec.subset} idx={spec.index} {method} N={N}] "
                          f"attempt {attempt + 1}/{cell_retries + 1} setup failed: "
                          f"{type(err).__name__}: {err}", flush=True)
                continue
            try:
                cfg = _cfg_for(env_name, spec, method, N, args, recursion_mode, L)
                if getattr(args, "simulation_only", False):
                    outcome = run_decision(
                        env, cfg, rollout_cache=cache, commit=False)
                else:
                    outcome = run_decision(env, cfg, rollout_cache=cache)
                rec = outcome.record.to_dict()
                rec["candidate_confidences"] = [
                    float(rollout.predicted.reward)
                    if rollout.predicted is not None else None
                    for rollout in outcome.rollouts
                ]
                rec["candidate_scores"] = [
                    float(rollout.score) for rollout in outcome.rollouts
                ]
                rec["candidate_actions"] = [
                    rollout.trajectory.signature() for rollout in outcome.rollouts
                ]
                rec["candidate_done"] = [
                    bool(rollout.trajectory.done) for rollout in outcome.rollouts
                ]
                rec["candidate_errors"] = [
                    (
                        rollout.trajectory.final_message
                        if not rollout.trajectory.done else None
                    )
                    for rollout in outcome.rollouts
                ]
                rec["candidate_grounded_success"] = [
                    (
                        bool(rollout.grounded.success)
                        if rollout.grounded is not None
                        else None
                    )
                    for rollout in outcome.rollouts
                ]
                if env_name == "appworld" and method == "ours":
                    rec["candidate_setup_prediction"] = [
                        {"enabled": (rollout.predicted.metadata or {}).get("setup_predictions_enabled", False),
                         "count": (rollout.predicted.metadata or {}).get("setup_predictions", 0)}
                        if rollout.predicted is not None else {"enabled": False, "count": 0}
                        for rollout in outcome.rollouts
                    ]
                rec["candidate_verifier_percheck"] = [
                    (rollout.predicted.metadata or {}).get("verifier_percheck")
                    if rollout.predicted is not None else None
                    for rollout in outcome.rollouts
                ]
                rec["candidate_comparative_rerank"] = [
                    (rollout.predicted.metadata or {}).get("comparative_rerank")
                    if rollout.predicted is not None else None
                    for rollout in outcome.rollouts
                ]
                if (
                    env_name == "officebench"
                    and N == max_n
                    and not getattr(args, "simulation_only", False)
                    and _env_enabled(
                        "TTS_OFFICEBENCH_DIAGNOSTIC_ALL_CANDIDATES", False)
                ):
                    # Offline calibration only: selection and the selected commit have already
                    # completed. Never feed these official evaluator outcomes back into ranking.
                    candidate_actual = [
                        env.evaluate(rollout.trajectory)
                        if rollout.trajectory.done
                        else None
                        for rollout in outcome.rollouts
                    ]
                    rec["candidate_actual_success"] = [
                        bool(actual.success) if actual is not None else False
                        for actual in candidate_actual
                    ]
                    rec["candidate_actual_reward"] = [
                        float(actual.reward) if actual is not None else 0.0
                        for actual in candidate_actual
                    ]
                last_error = None
                break
            except Exception as err:  # noqa: BLE001 - retry transient env/commit/evaluation failures
                last_error = err
                if attempt < cell_retries:
                    print(f"\n[cell-retry {spec.subset} idx={spec.index} {method} N={N}] "
                          f"attempt {attempt + 1}/{cell_retries + 1} failed: "
                          f"{type(err).__name__}: {err}", flush=True)
        if rec is None:
            _report_cell_error(
                args, env_name, spec, method, N,
                last_error or RuntimeError("cell produced no record"))
            continue
        # Stamp the spec identity so a resumed run can tell which tasks are already done
        # (the record stores the resolved ``task_id``, not the enumeration ``index``).
        rec["run_env"] = env_name
        rec["spec_subset"] = spec.subset
        rec["spec_index"] = spec.index
        rec["solver_agent"] = getattr(args, "agent", None)
        rec["llm_agent"] = (
            getattr(args, "llm_agent", None)
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or os.environ.get("WM_MODEL")
            or os.environ.get("TTS_AGENT_MODEL")
        )
        rec["solver_temperature"] = getattr(args, "temperature", None)
        rec["solver_max_steps"] = getattr(args, "max_steps", None)
        rec["agent_self_aggregate"] = bool(
            getattr(args, "agent_self_aggregate", False))
        rec["aggregation_strategy"] = getattr(args, "aggregation_strategy", "flat")
        rec["seed_base"] = getattr(args, "seed_base", None)
        rec["success_score"] = getattr(args, "success_score", None)
        rec["predict_all_actions"] = bool(
            getattr(args, "predict_all_actions", True))
        rec["simulation_only"] = bool(
            getattr(args, "simulation_only", False))
        if env_name == "appworld":
            rec["appworld_wm_tool_mode"] = getattr(
                args, "appworld_wm_tool_mode", "python")
            rec["appworld_precision_prompt"] = bool(
                getattr(args, "appworld_precision_prompt", False))
            rec["appworld_agent_reference"] = bool(
                getattr(args, "appworld_agent_reference", False))
            rec["appworld_deterministic_dirty_reads"] = bool(
                getattr(args, "appworld_deterministic_dirty_reads", False))
            rec["appworld_preground_clean_reads"] = _env_enabled(
                "TTS_APPWORLD_PREGROUND_CLEAN_READS", False)
            rec["appworld_deterministic_grounded_clean_reads"] = _env_enabled(
                "TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS", False)
            rec["appworld_predict_all_program_only"] = _env_enabled(
                "TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY", True)
            rec["appworld_instruction_preflights"] = _env_enabled(
                "TTS_APPWORLD_INSTRUCTION_PREFLIGHTS", False)
            rec["appworld_verifier_percheck"] = _env_enabled(
                "TTS_APPWORLD_VERIFIER_PERCHECK", False)
            rec["appworld_verifier_samples"] = _positive_int_env(
                "TTS_APPWORLD_VERIFIER_SAMPLES", 1)
            rec["appworld_verifier_coverage_aware"] = _env_enabled(
                "TTS_APPWORLD_VERIFIER_COVERAGE_AWARE", False)
            rec["appworld_verifier_unknown_credit"] = _float_env(
                "TTS_APPWORLD_VERIFIER_UNKNOWN_CREDIT", 0.25)
            rec["selection_confidence_margin"] = _float_env(
                "TTS_SELECTION_CONFIDENCE_MARGIN", 0.1)
        if env_name == "officebench":
            rec["officebench_wm_tool_mode"] = getattr(
                args, "officebench_wm_tool_mode", "python")
            rec["officebench_verifier_samples"] = _positive_int_env(
                "TTS_OFFICEBENCH_VERIFIER_SAMPLES", 1)
            rec["officebench_verifier_reference"] = _env_enabled(
                "TTS_OFFICEBENCH_VERIFIER_REFERENCE", True)
            rec["officebench_verifier_state"] = _env_enabled(
                "TTS_OFFICEBENCH_VERIFIER_STATE", True)
            rec["officebench_verifier_percheck"] = _env_enabled(
                "TTS_OFFICEBENCH_VERIFIER_PERCHECK", False)
            rec["officebench_verifier_atomic_cap"] = _positive_int_env(
                "TTS_OFFICEBENCH_VERIFIER_ATOMIC_CAP", 4)
            rec["officebench_verifier_unknown_credit"] = _float_env(
                "TTS_OFFICEBENCH_VERIFIER_UNKNOWN_CREDIT", 0.25)
            rec["officebench_verifier_whole_weight"] = _float_env(
                "TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT", 0.25)
            rec["officebench_reconstruct_gold"] = _env_enabled(
                "TTS_OFFICEBENCH_WM_RECONSTRUCT_GOLD", False)
            rec["officebench_comparative_rerank"] = _env_enabled(
                "TTS_OFFICEBENCH_COMPARATIVE_RERANK", False)
            rec["officebench_comparative_confidence_margin"] = _float_env(
                "TTS_OFFICEBENCH_COMPARATIVE_CONFIDENCE_MARGIN", 0.25)
            rec["officebench_selection_confidence_margin"] = _float_env(
                "TTS_OFFICEBENCH_SELECTION_CONFIDENCE_MARGIN", 0.05)
        records.append(rec)
        try:
            _write_cell_checkpoint(args, rec)
        except OSError as checkpoint_err:
            # The cell itself succeeded and remains in ``records`` for the parent; report only the
            # durability degradation, not a false execution failure in run_manifest.json.
            print(f"\n[cell-checkpoint {spec.subset} idx={spec.index} {method} N={N}] failed: "
                  f"{type(checkpoint_err).__name__}: {checkpoint_err}", flush=True)
    return records


def _run_tasks(env_name, specs, cells, args, usable_only, desc, on_task_done=None) -> List[dict]:
    """Run ``cells`` for every task in ``specs``, optionally across parallel workers.

    The dominant cost is the agent's LLM rollouts (each task = many sequential network calls), so
    these are I/O-bound and parallelising across tasks gives near-linear wall-clock speed-up.
    AppWorld is **not** thread-safe (thread-affine sqlite + module-global ground-truth loader), so
    it uses a PROCESS pool; τ² is thread-safe per task, so it uses the lighter THREAD pool (which
    also shares the already-loaded tau2 registry).

    ``on_task_done`` (if given) is called in THIS process with each finished task's record list as
    soon as it completes, so the caller can checkpoint progress to disk incrementally.  This makes
    a long sweep resumable: if it is interrupted (crash, OOM, terminal hang-up, pool deadlock), the
    records written so far survive and a re-run skips the tasks already present.
    """
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    progress = _Progress(total=len(specs), desc=desc)
    records: List[dict] = []
    emitted_cells = set()

    def _emit(recs):
        fresh = []
        for rec in recs:
            key = _cell_checkpoint_key(rec)
            if key in emitted_cells:
                continue
            emitted_cells.add(key)
            fresh.append(rec)
        records.extend(fresh)
        if on_task_done is not None and fresh:
            on_task_done(fresh)

    if workers == 1 or len(specs) <= 1:
        for spec in specs:
            _emit(_run_task_cells((env_name, spec, cells, args, usable_only)))
            progress.update(1, postfix=getattr(spec, "subset", ""))
        progress.close()
        return records

    from concurrent.futures import (
        FIRST_COMPLETED,
        ProcessPoolExecutor,
        ThreadPoolExecutor,
        wait,
    )

    # AppWorld uses a process pool. Recycle workers after ``--max-tasks-per-child`` tasks so a long
    # run cannot accumulate unbounded per-worker memory (each AppWorld worker carries the ground-
    # truth loader + caches); unbounded growth risks an OOM that kills the parent and orphans the
    # pool in a result-queue deadlock.  Worker recycling is incompatible with the default ``fork``
    # start method, so it is opt-in and switches to a ``forkserver`` context (which also forks
    # workers from a clean server with no inherited locks, avoiding result-queue deadlocks).
    pool_kwargs = {"max_workers": workers}
    if env_name in ("appworld", "officebench"):
        Executor = ProcessPoolExecutor
        mtpc = int(getattr(args, "max_tasks_per_child", 0) or 0)
        if mtpc > 0:
            import multiprocessing as _mp

            pool_kwargs["mp_context"] = _mp.get_context("forkserver")
            pool_kwargs["max_tasks_per_child"] = mtpc
    else:
        Executor = ThreadPoolExecutor
    payloads = [(env_name, spec, cells, args, usable_only) for spec in specs]

    # Hang-guard: a single AppWorld task can wedge a worker forever (e.g. an agent that emits a
    # non-terminating ``while True:`` API loop, which AppWorld's own per-cell SIGALRM does not abort
    # in the process-pool context). With a plain ``as_completed`` + ``with Executor`` that one task
    # hangs the whole sweep ("stuck at the last sample") and then blocks ``shutdown(wait=True)``.
    # Instead we wait in slices and observe worker-side CELL checkpoints too: if neither a task nor a
    # cell completes for ``stall_s`` seconds we treat the remaining futures as wedged, abandon them,
    # and force-terminate the workers. A slow task that is still completing successive N cells counts
    # as progress and is not incorrectly abandoned.
    stall_s = float(os.environ.get("TTS_RUN_STALL_TIMEOUT_S", "1200") or "1200")
    poll_s = min(stall_s, 30.0)
    checkpoint_dir = getattr(args, "_cell_checkpoint_dir", None)
    seen_sidecars = set()

    def _drain_sidecars() -> int:
        if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
            return 0
        fresh = []
        for name in sorted(os.listdir(checkpoint_dir)):
            if not name.startswith("record-") or not name.endswith(".json") or name in seen_sidecars:
                continue
            try:
                with open(os.path.join(checkpoint_dir, name), "r", encoding="utf-8") as fh:
                    fresh.append(json.load(fh))
                seen_sidecars.add(name)
            except (OSError, ValueError, TypeError):
                continue
        _emit(fresh)
        return len(fresh)

    ex = Executor(**pool_kwargs)
    last_progress = time.monotonic()
    abandoned = []
    task_errors = []
    try:
        futures = {ex.submit(_run_task_cells, p): p[1] for p in payloads}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=poll_s, return_when=FIRST_COMPLETED)
            drained = _drain_sidecars()
            if done or drained:
                last_progress = time.monotonic()
            if not done:
                if time.monotonic() - last_progress < stall_s:
                    continue
                abandoned = [
                    {"subset": futures[fut].subset, "index": futures[fut].index}
                    for fut in pending
                ]
                print(f"\n[hang-guard] no task or cell finished in {stall_s:.0f}s; abandoning "
                      f"{len(pending)} wedged task(s) (records checkpointed). "
                      f"Re-run with --resume to retry them.")
                for fut in pending:
                    fut.cancel()
                break
            for fut in done:
                try:
                    _emit(fut.result())
                except Exception as err:  # noqa: BLE001 - report and continue the sweep
                    spec = futures[fut]
                    task_errors.append({
                        "subset": getattr(spec, "subset", "?"),
                        "index": getattr(spec, "index", None),
                        "error_type": type(err).__name__,
                        "error": str(err),
                    })
                    print(f"\n[task {getattr(spec, 'subset', '?')} "
                          f"idx={getattr(spec, 'index', '?')}] failed: {type(err).__name__}: {err}")
                progress.update(1, postfix=getattr(futures[fut], "subset", ""))
    finally:
        _drain_sidecars()
        args._last_run_status = {
            "abandoned_tasks": abandoned,
            "task_errors": task_errors,
        }
        # Force the pool down without joining a wedged worker: cancel queued futures, then SIGTERM any
        # still-alive worker processes (a ProcessPool exposes them via ``_processes``).  Threads can't
        # be killed, but ``shutdown(wait=False)`` at least won't actively block on them.
        procs = list(getattr(ex, "_processes", {}).values() or [])
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # pragma: no cover - Python < 3.9 has no cancel_futures
            ex.shutdown(wait=False)
        for proc in procs:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
    progress.close()
    return records


def run_cell(env_name, method, N, args, recursion_mode=None, L=None, rollout_cache=None,
             progress=None) -> List[dict]:
    """Single-cell runner (one method, one N) — retained for ad-hoc/programmatic use."""
    records = []
    usable_only = False  # Run on all tasks, not just WM-eligible mutating tasks
    for spec in _specs_for(env_name, args):
        try:
            env = make_env_from_spec(spec, usable_only=usable_only)
        except (NotImplementedError, IndexError):
            if progress is not None:
                progress.update(1)
            continue
        cfg = _cfg_for(env_name, spec, method, N, args, recursion_mode, L)
        records.append(run_decision(env, cfg, rollout_cache=rollout_cache).record.to_dict())
        if progress is not None:
            progress.update(1, postfix=f"{method} N={N} {spec.subset}")
    return records


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def aggregate(records: List[dict], by: List[str]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in records:
        groups[tuple(r.get(k) for k in by)].append(r)
    rows = []
    for key, rs in groups.items():
        row = dict(zip(by, key))
        preds = [r["prediction_correct"] for r in rs if r.get("prediction_correct") is not None]
        imputed_failures = sum(bool(r.get("coverage_imputed_failure")) for r in rs)
        row.update({
            "decisions": len(rs),
            "recorded_decisions": len(rs) - imputed_failures,
            "imputed_failures": imputed_failures,
            "success_rate": _mean([r["success"] for r in rs]),
            "reward": _mean([r["reward"] for r in rs]),
            "mean_spawns": _mean([r["n_spawns"] for r in rs]),
            "mean_max_depth": _mean([r["max_depth"] for r in rs]),
            "mean_reads": _mean([r["n_reads"] for r in rs]),
            "single_commit_rate": _mean([r["n_commits"] == 1 for r in rs]),
            "simulator_writes": sum(r["simulator_writes"] for r in rs),
            "prediction_accuracy": (statistics.fmean(preds) if preds else None),
            "mean_tokens": _mean([r["tokens"] for r in rs]),
            "mean_cost": _mean([r["cost"] for r in rs]),
            "mean_cache_hits": _mean([r["cache_hits"] for r in rs]),
            "mean_wall_clock_s": _mean([r["wall_clock_s"] for r in rs]),
            "mean_rollout_errors": _mean([r.get("n_rollout_errors", 0) for r in rs]),
        })
        rows.append(row)
    return rows


def _ordered_methods(records):
    return sorted({r["method"] for r in records}, key=lambda m: _METHOD_ORDER.get(m, 99))


def tts_table(records: List[dict], title: str) -> str:
    Ns = sorted({r["n_rollouts"] for r in records})
    lut = {(r["method"], r["n_rollouts"]): r["success_rate"] for r in aggregate(records, ["method", "n_rollouts"])}
    out = [f"\n{title}", "-" * len(title), "  method | " + " ".join(f"N={n:<4}" for n in Ns)]
    for m in _ordered_methods(records):
        if m == "b0":  # N-independent: broadcast its single measured value
            base = next((lut[k] for k in lut if k[0] == "b0"), float("nan"))
            vals = [base] * len(Ns)
        else:
            vals = [lut.get((m, n), float("nan")) for n in Ns]
        out.append(f"  {m:>6} | " + " ".join(f"{v:<6.2f}" for v in vals))
    return "\n".join(out)


def print_report(records: List[dict]) -> None:
    if not records:
        print("(no decisions)")
        return
    for env_name in sorted({r["env"] for r in records}):
        env_recs = [r for r in records if r["env"] == env_name]
        print(tts_table(env_recs, f"TTS curve — success vs N  [{env_name} | ALL subsets]"))
        for sub in sorted({r["subset"] for r in env_recs}):
            print(tts_table([r for r in env_recs if r["subset"] == sub], f"TTS — success vs N  [{env_name} | {sub}]"))


def write_csv(path, rows):
    if not rows:
        return
    import csv

    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_records(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _expected_cells(methods, Ns):
    """The set of ``(method, n_rollouts)`` cells one task produces for this sweep."""
    return {(m, (1 if m == "b0" else N)) for m in methods for N in ([1] if m == "b0" else Ns)}


def _load_checkpoint(path, cell_checkpoint_dir=None):
    """Read an existing ``records.jsonl`` for resume.

    Returns ``(records, done)`` where ``records`` is the de-duplicated record list already on disk
    (last writer wins per ``(run_env, spec_subset, spec_index, method, n_rollouts)``) and ``done``
    maps each task key ``(run_env, spec_subset, spec_index)`` to the set of
    ``(method, n_rollouts)`` cells it already has. Worker-side cell checkpoints are also recovered
    when supplied, so a killed task does not lose the lower-N cells it completed before the kill.
    Malformed entries are skipped.
    """
    by_cell: Dict[tuple, dict] = {}
    done: Dict[tuple, set] = defaultdict(set)

    def _add(r):
        if not isinstance(r, dict):
            return
        si = r.get("spec_index")
        sub = r.get("spec_subset", r.get("subset"))
        run_env = _record_run_env(r)
        task_key = (run_env, sub, si)
        cell = (r.get("method"), r.get("n_rollouts"))
        by_cell[(run_env, sub, si, r.get("method"), r.get("n_rollouts"))] = r
        if si is not None:
            done[task_key].add(cell)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    _add(json.loads(line))
                except Exception:
                    continue
    for r in _read_cell_sidecars(cell_checkpoint_dir, "record"):
        _add(r)
    return list(by_cell.values()), done


def _rewrite_checkpoint(path: str, records: List[dict]) -> None:
    """Rewrite the canonical checkpoint after recovering/de-duplicating worker sidecars."""
    tmp = f"{path}.tmp-p{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    if hasattr(os, "O_DIRECTORY"):
        directory_fd = os.open(os.path.dirname(path) or ".", os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _records_with_missing_failures(env_name, specs, records, expected_cells):
    """Return metric-only records with every absent expected cell counted as failure."""
    metric_records = list(records)
    existing = set()
    metric_env_by_subset = {}
    for record in records:
        subset = record.get("spec_subset", record.get("subset"))
        if subset is not None and record.get("env"):
            metric_env_by_subset.setdefault(subset, record["env"])
        if _record_run_env(record) != env_name or record.get("spec_index") is None:
            continue
        existing.add((
            subset,
            record.get("spec_index"),
            record.get("method"),
            record.get("n_rollouts"),
        ))

    for spec in specs:
        metric_env = metric_env_by_subset.get(spec.subset)
        if metric_env is None:
            metric_env = f"tau2_{spec.subset}" if env_name == "tau2" else env_name
        for method, n_rollouts in expected_cells:
            key = (spec.subset, spec.index, method, n_rollouts)
            if key in existing:
                continue
            metric_records.append({
                "env": metric_env,
                "run_env": env_name,
                "subset": spec.subset,
                "spec_subset": spec.subset,
                "spec_index": spec.index,
                "task_id": f"missing:{spec.subset}:{spec.index}",
                "method": method,
                "n_rollouts": n_rollouts,
                "success": False,
                "reward": 0.0,
                "n_spawns": None,
                "max_depth": None,
                "n_reads": None,
                "n_commits": 0,
                "simulator_writes": 0,
                "prediction_correct": None,
                "tokens": None,
                "cost": None,
                "cache_hits": None,
                "wall_clock_s": None,
                "n_rollout_errors": None,
                "coverage_imputed_failure": True,
            })
    return metric_records


def _coverage_manifest(env_name, specs, records, expected_cells, run_status, cell_errors):
    """Summarize expected, complete, partial, failed, and abandoned tasks for this run."""
    records_by_task = defaultdict(set)
    for r in records:
        if _record_run_env(r) != env_name or r.get("spec_index") is None:
            continue
        key = (r.get("spec_subset", r.get("subset")), r.get("spec_index"))
        records_by_task[key].add((r.get("method"), r.get("n_rollouts")))

    by_subset = {}
    complete = partial = missing = 0
    recorded_decisions = missing_decisions = 0
    for spec in specs:
        key = (spec.subset, spec.index)
        cells = records_by_task.get(key, set())
        state = "complete" if expected_cells.issubset(cells) else ("partial" if cells else "missing")
        if state == "complete":
            complete += 1
        elif state == "partial":
            partial += 1
        else:
            missing += 1
        row = by_subset.setdefault(spec.subset, {
            "expected_tasks": 0, "complete_tasks": 0, "partial_tasks": 0, "missing_tasks": 0,
            "expected_decisions": 0, "recorded_decisions": 0, "missing_decisions": 0,
        })
        recorded_cells = len(cells.intersection(expected_cells))
        absent_cells = len(expected_cells) - recorded_cells
        recorded_decisions += recorded_cells
        missing_decisions += absent_cells
        row["expected_tasks"] += 1
        row[f"{state}_tasks"] += 1
        row["expected_decisions"] += len(expected_cells)
        row["recorded_decisions"] += recorded_cells
        row["missing_decisions"] += absent_cells

    return {
        "env": env_name,
        "expected_tasks": len(specs),
        "complete_tasks": complete,
        "partial_tasks": partial,
        "missing_tasks": missing,
        "expected_cells_per_task": len(expected_cells),
        "expected_cells": [
            {"method": method, "n_rollouts": n_rollouts}
            for method, n_rollouts in sorted(
                expected_cells,
                key=lambda cell: (_METHOD_ORDER.get(cell[0], 99), cell[0], cell[1]),
            )
        ],
        "expected_decisions": len(specs) * len(expected_cells),
        "recorded_decisions": recorded_decisions,
        "missing_decisions": missing_decisions,
        "cell_errors": cell_errors,
        "abandoned_tasks": run_status.get("abandoned_tasks", []),
        "task_errors": run_status.get("task_errors", []),
        "by_subset": by_subset,
    }


def dump_spawn_tree(env_name, args) -> None:
    if env_name != "tau2" or "ours" not in args.methods.split(","):
        return  # the read-grounded spawn tree is an "ours" (entity-store) construct
    try:
        env = make_env(env_name, index=args.seed_base)
        out = run_decision(env, Config(method="ours", env=env_name, N=4, seed=args.seed_base,
                                       solver=SolverConfig(agent=args.agent, p_correct=args.p_correct,
                                                           llm_agent=args.llm_agent, max_steps=args.max_steps),
                                       simulator=sim_config(args)))
        sel = out.rollouts[out.selected_index]
        root = sel.spawn_root
        if root is not None:
            print("\nExample spawn tree (selected trajectory):")
            print(root.render_ascii())
            with open(os.path.join(args.out, "spawn_tree.json"), "w") as fh:
                json.dump(root.to_dict(), fh, indent=2)
    except Exception as e:  # never let the cosmetic spawn-tree dump break the run
        print(f"(spawn-tree dump skipped: {type(e).__name__}: {e})")


def _run_terminalbench_cli(args) -> None:
    """Run the native Terminal-Bench harnesses and write the shared experiment artifacts.

    This path returns before dataset/registry/environment construction, so adding Terminal-Bench
    cannot alter AppWorld, OfficeBench, or tau2 task selection, rollouts, or aggregation.
    """
    from tts.env.terminalbench_experiment import (
        run_terminalbench_sweep,
        terminalbench_native_cell_count,
    )

    if args.list_subsets:
        versions = [item.strip() for item in args.terminalbench_versions.split(",") if item.strip()]
        for version in versions:
            print(f"  {'v' + version:>20}  official native dataset")
        return

    os.makedirs(args.out, exist_ok=True)
    records_path = os.path.join(args.out, "records.jsonl")
    prior_records: List[dict] = []
    if args.resume:
        prior_records, _ = _load_checkpoint(records_path)
        print(f"(resume: {len(prior_records)} Terminal-Bench records on disk; "
              "reusing completed native jobs)")
    else:
        open(records_path, "w").close()

    seen = {
        (record.get("env"), record.get("spec_subset"), record.get("spec_index"),
         record.get("method"), record.get("n_rollouts"))
        for record in prior_records
    }
    checkpoint = open(records_path, "a", buffering=1)
    progress = _Progress(
        terminalbench_native_cell_count(args),
        desc="terminalbench jobs",
    )

    def _checkpoint(records):
        for record in records:
            key = (
                record.get("env"),
                record.get("spec_subset"),
                record.get("spec_index"),
                record.get("method"),
                record.get("n_rollouts"),
            )
            if key in seen:
                continue
            checkpoint.write(json.dumps(record, default=str) + "\n")
            seen.add(key)
        checkpoint.flush()
        os.fsync(checkpoint.fileno())

    try:
        new_records = run_terminalbench_sweep(
            args,
            on_records=_checkpoint,
            on_cell_done=lambda label: progress.update(1, postfix=label),
        )
    finally:
        progress.close()
        checkpoint.close()

    all_records, _ = _load_checkpoint(records_path)
    if not all_records:
        all_records = prior_records + new_records
    print_report(all_records)
    aggregates = aggregate(all_records, ["env", "subset", "method", "n_rollouts"])
    with open(os.path.join(args.out, "aggregates.json"), "w") as handle:
        json.dump(aggregates, handle, indent=2)
    write_csv(os.path.join(args.out, "aggregates.csv"), aggregates)
    print(f"\nWrote {len(all_records)} decisions and {len(aggregates)} aggregate rows "
          f"to {args.out}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default=None, choices=["appworld", "tau2", "officebench", "terminalbench"])
    p.add_argument("--methods", default="b0,b1,b2,ours")
    p.add_argument("--N", default="1,2,4,8")
    p.add_argument("--tasks", type=int, default=None, help="tasks PER SUBSET (default: full usable size)")
    p.add_argument("--task-indices", default=None,
                   help="Exact task indices to run after split/level/subset filtering, comma-separated "
                        "(e.g. 20,27,28,32,45). Useful for targeted diagnostic reruns.")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--success-score", default=None,
                   help="trajectory ranker (default: simulator_confidence for OfficeBench llm_wm; "
                        "simulator_success otherwise)")
    p.add_argument("--verifier-reference", action="store_true",
                   help="AppWorld llm_wm: explicitly enable the REFERENCE-GROUNDED verifier (the default, "
                        "matching OfficeBench; audits the agent's write log against the task's "
                        "ground-truth checklist + gold values). Sets "
                        "TTS_APPWORLD_VERIFIER_REFERENCE=1 and, unless --success-score is given explicitly, "
                        "switches ranking to simulator_confidence. NOTE: uses the task answer key -> this is "
                        "reference-grounded (tau2-style), not blind WM-only.")
    p.add_argument("--no-verifier-reference", action="store_true",
                   help="AppWorld llm_wm: disable the default reference grounding and use the blind "
                        "WM-only verifier ablation (sets TTS_APPWORLD_VERIFIER_REFERENCE=0).")
    p.add_argument("--verifier-samples", type=int, default=None,
                   help="K self-consistency samples for the AppWorld/OfficeBench WM verifier. "
                        "K>=2 gives a graded confidence in {0,1/K,..,1}. OfficeBench GPT-5.4 "
                        "llm_wm defaults to 5; other paths default to 1.")
    p.add_argument("--verifier-percheck", action="store_true",
                   help="AppWorld llm_wm: PER-CHECK reference auditor — a focused sub-pass audits each "
                        "ground-truth check in isolation (MET/UNMET/UNKNOWN). GPT-5.4 uses coverage-aware "
                        "graded confidence so UNKNOWN checks remain in the denominator. "
                        "Implies --verifier-reference; sets TTS_APPWORLD_VERIFIER_PERCHECK=1.")
    p.add_argument("--sandbox-prompt", action="store_true",
                   help="AppWorld: prepend a truthful sandbox preamble to the agent prompt so safety-aligned "
                        "models (e.g. gpt-5.2) don't REFUSE consequential actions (orders/payments). Sets "
                        "TTS_APPWORLD_SANDBOX_PROMPT=1. NOTE: deviates from the official prompt — apply "
                        "uniformly across compared models for a fair A/B.")
    p.add_argument("--appworld-precision-prompt", action="store_true",
                   help="AppWorld: prepend an exact-state reasoning skill covering pagination, set/map "
                        "reconciliation, Amazon cart isolation, calendar boundaries, and final-state "
                        "verification. Sets TTS_APPWORLD_PRECISION_PROMPT=1 and intentionally augments "
                        "the official agent prompt.")
    p.add_argument("--appworld-agent-reference", action="store_true",
                   help="AppWorld llm_wm: prepend the native evaluator checklist and gold values to "
                        "candidate generation. Sets TTS_APPWORLD_AGENT_REFERENCE=1. NOTE: this uses the "
                        "task answer key and is reference-conditioned, not blind benchmark generation.")
    p.add_argument("--appworld-deterministic-dirty-reads", action="store_true",
                   help="AppWorld llm_wm: apply supported post-write state transitions (currently Amazon "
                        "cart operations) deterministically before falling back to the Python WM. Sets "
                        "TTS_APPWORLD_DETERMINISTIC_DIRTY_READS=1.")
    appworld_preflights = p.add_mutually_exclusive_group()
    appworld_preflights.add_argument(
        "--appworld-instruction-preflights",
        dest="appworld_instruction_preflights",
        action="store_true",
        help="AppWorld llm_wm AGENT-SIDE ABLATION: replace the stochastic ReAct loop for recognized "
             "tasks with instruction-only plans. Disabled by default and not comparable to B2.",
    )
    appworld_preflights.add_argument(
        "--no-appworld-instruction-preflights",
        dest="appworld_instruction_preflights",
        action="store_false",
        help="Use the unchanged stochastic ReAct agent for every AppWorld task (default).",
    )
    p.set_defaults(appworld_instruction_preflights=None)
    appworld_tool_default = (
        "python"
        if str(os.environ.get("TTS_APPWORLD_WM_TOOL_MODE", "python")).strip().lower()
        in ("python", "code", "coding")
        else "text"
    )
    p.add_argument(
        "--appworld-wm-tool-mode",
        choices=["text", "python"],
        default=appworld_tool_default,
        help="AppWorld llm_wm subagent protocol. 'python' uses the restricted "
             "PYTHON_READ/PYTHON_FINAL loop (default); 'text' restores the legacy "
             "READ/LIST/CALL/TRACE ablation.",
    )
    officebench_tool_default = (
        "python"
        if str(os.environ.get("TTS_OFFICEBENCH_WM_TOOL_MODE", "python")).strip().lower()
        in ("python", "code", "coding")
        else "text"
    )
    p.add_argument(
        "--officebench-wm-tool-mode",
        choices=["text", "python"],
        default=officebench_tool_default,
        help="OfficeBench llm_wm subagent protocol. 'python' uses AppWorld-style "
             "PYTHON_READ/PYTHON_FINAL restricted computation (default); 'text' restores the "
             "legacy READ/LIST/CALL/TRACE behavior.",
    )
    p.add_argument("--agent", default="llm", choices=["llm", "llm_wm", "scripted"],
                   help="real official-prompt LLM agent on the real env (default, reproduces the "
                        "benchmark) | llm_wm: LLM solver that interacts with the world model | "
                        "scripted: offline stub")
    p.add_argument(
        "--agent-self-aggregate",
        action="store_true",
        help="Opt-in: after sampling N trajectories, ask the solver model to aggregate them into "
             "the one trajectory to commit instead of using verifier/grounded-score selection. "
             "Disabled by default, so legacy experiments are unchanged.",
    )
    p.add_argument(
        "--aggregation-max-tokens",
        type=int,
        default=2048,
        help="Maximum output tokens for the deterministic agent self-aggregation call.",
    )
    p.add_argument("--llm-agent", default=None, help="agent model (default: Azure deployment / WM_MODEL, e.g. gpt-4.1)")
    p.add_argument("--llm-user", default=None, help="τ² user-simulator model (default: --llm-agent)")
    p.add_argument(
        "--tau2-qwen-agent-profile",
        default=os.environ.get("TTS_TAU2_QWEN_AGENT_PROFILE", "off"),
        choices=["auto", "off", "prompt", "policy"],
        help="Opt-in Qwen-only tau2 interface: prompt adds structured tool/subgoal guidance "
             "without thinking; policy also enables per-request thinking; off/auto preserve the "
             "stock tau2 agent.",
    )
    p.add_argument("--local", action="store_true",
                   help="Route the agent/world model to a LOCAL, OpenAI-compatible server "
                        "(vLLM / SGLang). Requires "
                        "--llm-agent <served-model-id> (e.g. Qwen-AgentWorld-35B-A3B). Forces the "
                        "openai backend for AppWorld/OfficeBench and routes τ² (litellm) via the "
                        "openai/ provider. A qualified --llm-user azure/<deployment> may still use "
                        "Azure for the τ² user simulator.")
    p.add_argument("--local-base-url", default=None,
                   help="Base URL of the local OpenAI-compatible server (default "
                        "$OPENAI_BASE_URL or http://localhost:8000/v1). Only used with --local.")
    p.add_argument("--local-api-key", default=None,
                   help="API key for the local server (default $OPENAI_API_KEY or 'EMPTY'; most "
                        "vLLM/SGLang servers accept any value). Only used with --local.")
    p.add_argument("--skip-local-healthcheck", action="store_true",
                   help="Skip the --local preflight that pings the server before running "
                        "(use if the server doesn't expose GET /v1/models).")
    p.add_argument("--temperature", type=float, default=0.7, help="agent sampling temperature (B1/B2/Ours diversity)")
    p.add_argument("--max-steps", type=int, default=100, help="max agent steps per rollout")
    p.add_argument("--rollout-retries", type=int, default=2,
                   help="Retries after one WM-interactive candidate rollout raises. If all attempts "
                        "fail, retain an explicit invalid/failed candidate so the N cell remains complete.")
    p.add_argument("--cell-retries", type=int, default=2,
                   help="Retries after a whole task/method/N cell fails during environment setup, "
                        "selection commit, or official evaluation (default: 2).")
    p.add_argument("--all-tasks", action="store_true",
                   help="τ²: run over ALL official tasks (default for --agent llm; the official "
                        "benchmark set). Scripted/world-model methods otherwise use mutating tasks only.")
    p.add_argument(
        "--simulation-only",
        action="store_true",
        help=(
            "llm_wm only: sample and select trajectories without committing or running the official "
            "evaluator; records contain simulator-predicted reward and n_commits=0"
        ),
    )
    p.add_argument("--p-correct", type=float, default=0.5, help="prob a *scripted* trajectory follows the plan")
    p.add_argument("--splits", default=None,
                   help="AppWorld splits, COMMA-separated (e.g. test_normal,test_challenge or dev). "
                        "All splits are locally scorable: every task ships its test checkers and is "
                        "scored by AppWorld.evaluate() (ground_truth_mode='minimal').")
    p.add_argument("--levels", default=None,
                   help="AppWorld difficulty levels (e.g. 1 or 1,2,3); for OfficeBench, the app-counts "
                        "to run (1,2,3 -> the 1app/2app/3app subsets), COMMA-separated.")
    p.add_argument("--domains", default=None, help="τ² domains, COMMA-separated (e.g. airline,retail).")
    p.add_argument("--subsets", default=None, help="subset ids to keep, COMMA-separated.")
    p.add_argument("--terminalbench-versions", "--tb-versions", default="1,2",
                   help="Terminal-Bench generations to run, COMMA-separated: 1,2 (default: both).")
    p.add_argument("--terminalbench-task", action="append", default=None,
                   help="Terminal-Bench task name/glob to include. Prefix with 1: or 2: to target "
                        "one generation (e.g. 1:hello-world). Repeat for multiple tasks.")
    p.add_argument(
        "--terminalbench-root",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "..", ".benchmarks", "terminalbench")),
        help="Downloaded Terminal-Bench checkout root (default: .benchmarks/terminalbench).",
    )
    p.add_argument("--terminalbench-conda-env",
                   default=os.environ.get("TTS_TERMINALBENCH_ENV", "tts-terminalbench"),
                   help="Conda environment containing tb and Harbor (default: tts-terminalbench).")
    p.add_argument("--list-subsets", action="store_true")
    p.add_argument("--report", default=None)
    p.add_argument("--recursion-mode", default="uncertainty", choices=["uncertainty", "fixed"])
    action_prediction = p.add_mutually_exclusive_group()
    action_prediction.add_argument(
        "--predict-all-actions",
        dest="predict_all_actions",
        action="store_true",
        help="B1/Ours: predict clean read observations too. B1 uses fixed context; Ours may use "
             "its bounded read-only grounding tools. This shared extension is enabled by default.",
    )
    action_prediction.add_argument(
        "--exact-clean-reads",
        dest="predict_all_actions",
        action="store_false",
        help="Restore the previous ablation where untouched clean reads execute exactly.",
    )
    p.set_defaults(predict_all_actions=True)
    p.add_argument("--tau", type=float, default=0.9)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--branching", type=int, default=2)
    p.add_argument("--node-read-limit", type=int, default=1)
    p.add_argument("--hard-cap", type=int, default=6)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-selective-invalidation", action="store_true")
    p.add_argument("--depth-sweep", action="store_true")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel tasks. Tasks are independent and LLM-bound, so >1 gives a near-"
                        "linear speed-up. AppWorld uses a PROCESS pool (it is not thread-safe); τ² "
                        "uses a THREAD pool. Default 1 (sequential).")
    p.add_argument("--aggregation-strategy", choices=["flat", "tournament"], default="flat",
                   help="Self-aggregation layout; flat preserves prior experiment behavior.")
    p.add_argument("--aggregation-group-size", type=int, default=4,
                   help="Candidates per concurrent tournament group (default 4).")
    p.add_argument("--max-tasks-per-child", type=int, default=0,
                   help="AppWorld process pool: if >0, recycle each worker after this many tasks to "
                        "bound per-worker memory growth (0 = never recycle, the default). A positive "
                        "value switches the pool to a 'forkserver' context (incompatible with the "
                        "default 'fork' start method), which also avoids inherited-lock deadlocks.")
    p.add_argument("--resume", action="store_true",
                   help="Append to (and skip tasks already present in) <out>/records.jsonl instead "
                        "of overwriting it. Each finished task is checkpointed immediately, so an "
                        "interrupted sweep can be restarted with --resume to continue where it left "
                        "off. The default (no --resume) also checkpoints incrementally but starts a "
                        "fresh records.jsonl.")
    coverage = p.add_mutually_exclusive_group()
    coverage.add_argument(
        "--allow-partial",
        dest="allow_partial",
        action="store_true",
        help="Allow a run with missing task/N cells to exit successfully and print unmatched "
             "curves (default; retained for compatibility).",
    )
    coverage.add_argument(
        "--strict-coverage",
        dest="allow_partial",
        action="store_false",
        help="Require every expected task/N cell before publishing metrics.",
    )
    p.set_defaults(allow_partial=True)
    p.add_argument("--keep-outputs", action="store_true",
                   help="Keep AppWorld per-task output dirs on disk (default: delete them as each "
                        "task finishes so large parallel runs don't accumulate temp files).")
    p.add_argument("--verbose", action="store_true",
                   help="Show the underlying tau2/litellm DEBUG logs (default: quiet — only the "
                        "progress bar and the result tables are printed).")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    args = p.parse_args()
    if getattr(args, "agent_self_aggregate", False):
        os.environ["TTS_AGENT_SELF_AGGREGATE"] = "1"
    args._success_score_explicit = args.success_score is not None
    args._predict_all_actions_explicit = any(
        flag in sys.argv[1:]
        for flag in ("--predict-all-actions", "--exact-clean-reads")
    )
    if args.success_score is None:
        args.success_score = "simulator_success"

    if args.no_verifier_reference and (args.verifier_reference or args.verifier_percheck):
        p.error("--no-verifier-reference cannot be combined with --verifier-reference or "
                "--verifier-percheck")

    if not args.verbose:
        _quiet_logging()

    if args.keep_outputs:
        os.environ["TTS_KEEP_APPWORLD_OUTPUTS"] = "1"
    os.environ["TTS_APPWORLD_WM_TOOL_MODE"] = args.appworld_wm_tool_mode
    os.environ["TTS_OFFICEBENCH_WM_TOOL_MODE"] = args.officebench_wm_tool_mode
    os.environ["TTS_TAU2_QWEN_AGENT_PROFILE"] = args.tau2_qwen_agent_profile

    # Verifier configuration -> env vars, set HERE (in the parent) so forked workers inherit them.
    if args.verifier_samples is not None:
        os.environ["TTS_APPWORLD_VERIFIER_SAMPLES"] = str(args.verifier_samples)
        os.environ["TTS_OFFICEBENCH_VERIFIER_SAMPLES"] = str(args.verifier_samples)
    if args.verifier_reference:
        os.environ["TTS_APPWORLD_VERIFIER_REFERENCE"] = "1"
        # The reference auditor yields a GRADED confidence; rank on it unless the user chose a score.
        if args.success_score == "simulator_success":
            args.success_score = "simulator_confidence"
        print(f"[verifier] reference-grounded auditor ENABLED "
              f"(TTS_APPWORLD_VERIFIER_SAMPLES={os.environ.get('TTS_APPWORLD_VERIFIER_SAMPLES','1')}, "
              f"success_score={args.success_score})")
    if args.verifier_percheck:
        os.environ["TTS_APPWORLD_VERIFIER_PERCHECK"] = "1"
        os.environ["TTS_APPWORLD_VERIFIER_REFERENCE"] = "1"  # per-check is a reference-grounded variant
        if args.success_score == "simulator_success":
            args.success_score = "simulator_confidence"
        print(f"[verifier] PER-CHECK reference auditor ENABLED (success_score={args.success_score})")
    if args.no_verifier_reference:
        os.environ["TTS_APPWORLD_VERIFIER_REFERENCE"] = "0"
        print("[verifier] reference grounding DISABLED (blind WM-only ablation)")
    if args.sandbox_prompt:
        os.environ["TTS_APPWORLD_SANDBOX_PROMPT"] = "1"
    if args.appworld_precision_prompt:
        os.environ["TTS_APPWORLD_PRECISION_PROMPT"] = "1"
        print("[agent] AppWorld precision prompt ENABLED")
    if args.appworld_agent_reference:
        os.environ["TTS_APPWORLD_AGENT_REFERENCE"] = "1"
        print("[agent] AppWorld reference-conditioned task contract ENABLED")
    if args.appworld_deterministic_dirty_reads:
        os.environ["TTS_APPWORLD_DETERMINISTIC_DIRTY_READS"] = "1"
        print("[wm] AppWorld deterministic dirty reads ENABLED")
    if args.appworld_instruction_preflights is not None:
        os.environ["TTS_APPWORLD_INSTRUCTION_PREFLIGHTS"] = (
            "1" if args.appworld_instruction_preflights else "0"
        )

    # Local, self-hosted model -> configure HERE (in the parent) so forked/spawned workers inherit
    # the OpenAI-compatible endpoint env vars (works for both the AppWorld process pool and the τ²
    # thread pool). Must run before any rollout worker is created.
    if args.local:
        from tts.llm.local import DEFAULT_LOCAL_BASE_URL, configure_local_llm, probe_local_server

        model = args.llm_agent or os.environ.get("TTS_LOCAL_MODEL")
        if not model:
            p.error("--local requires --llm-agent <served-model-id> "
                    "(e.g. --llm-agent Qwen-AgentWorld-35B-A3B)")
        if args.llm_user and args.llm_user.startswith("azure/"):
            if not (os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_API_BASE")):
                p.error("--local with --llm-user azure/<deployment> requires "
                        "AZURE_OPENAI_ENDPOINT or AZURE_API_BASE")
            os.environ["TTS_LOCAL_LLM_ALLOW_AZURE"] = "1"
        base_url = args.local_base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_LOCAL_BASE_URL
        api_key = args.local_api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        configure_local_llm(base_url=base_url, model=model, api_key=api_key)
        print(f"[llm] LOCAL OpenAI-compatible backend ENABLED -> {base_url} (model={model})")
        # Preflight: fail fast with actionable guidance instead of N cryptic per-task "Connection
        # error"s when no server is up at base_url (the #1 --local pitfall).
        if not args.skip_local_healthcheck:
            reachable, detail = probe_local_server(base_url, model=model)
            if not reachable:
                print(f"[llm] ERROR: local server not reachable — {detail}\n"
                      f"      Start it first, e.g.:\n"
                      f"        bash scripts/serve_local_model.sh <MODEL_PATH_OR_HF_ID> {model} "
                      f"{base_url.split('//',1)[-1].split('/')[0].rsplit(':',1)[-1] or '8000'}\n"
                      f"      Wait until it logs 'Uvicorn running', then re-run. "
                      f"(Bypass this check with --skip-local-healthcheck.)", file=sys.stderr)
                sys.exit(2)
            print(f"[llm] local server OK — {detail}")

    if args.report:
        print_report(load_records(args.report))
        return

    if args.env == "terminalbench":
        _run_terminalbench_cli(args)
        return

    avail = available_real_envs()
    env_name = args.env or (avail[0] if avail else None)
    if env_name is None:
        print("No real environment available in this interpreter. "
              "Activate `dw` (tau2) or `env` (appworld, with APPWORLD_ROOT), or set OFFICEBENCH_ROOT "
              "with a usable Docker daemon for officebench.")
        return
    if args.simulation_only:
        if args.agent != "llm_wm":
            p.error("--simulation-only requires --agent llm_wm")
        if not args.predict_all_actions:
            p.error("--simulation-only requires --predict-all-actions")
    if env_name == "appworld" and args.agent == "llm_wm":
        _configure_appworld_bon(args)
        if args.appworld_instruction_preflights:
            print("[agent] AppWorld replay-safe instruction preflights ENABLED")
        print(
            f"[appworld-wm] tool mode={args.appworld_wm_tool_mode}, "
            f"preground_clean={int(_env_enabled('TTS_APPWORLD_PREGROUND_CLEAN_READS', False))}, "
            f"deterministic_clean={int(_env_enabled('TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS', False))}, "
            f"program_only={int(_env_enabled('TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY', True))}, "
            f"percheck={int(_env_enabled('TTS_APPWORLD_VERIFIER_PERCHECK', False))}, "
            f"coverage_aware={int(_env_enabled('TTS_APPWORLD_VERIFIER_COVERAGE_AWARE', False))}, "
            f"unknown_credit={os.environ.get('TTS_APPWORLD_VERIFIER_UNKNOWN_CREDIT', '0.25')}, "
            f"success_score={args.success_score}"
        )
    if env_name == "officebench" and args.agent == "llm_wm":
        _configure_officebench_bon(args)
        print(
            f"[officebench-wm] tool mode={args.officebench_wm_tool_mode}, "
            f"reference={int(_env_enabled('TTS_OFFICEBENCH_VERIFIER_REFERENCE', True))}, "
            f"percheck={int(_env_enabled('TTS_OFFICEBENCH_VERIFIER_PERCHECK', False))}, "
            f"unknown_credit={os.environ.get('TTS_OFFICEBENCH_VERIFIER_UNKNOWN_CREDIT', '0.25')}, "
            f"whole_weight={os.environ.get('TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT', '0.25')}, "
            f"comparative_rerank={int(_env_enabled('TTS_OFFICEBENCH_COMPARATIVE_RERANK', False))}, "
            f"comparative_margin={_float_env('TTS_OFFICEBENCH_COMPARATIVE_CONFIDENCE_MARGIN', 0.25)}, "
            f"uniform_seeds=base+i, "
            f"verifier_samples={_positive_int_env('TTS_OFFICEBENCH_VERIFIER_SAMPLES', 1)}, "
            f"success_score={args.success_score}"
        )

    if args.list_subsets:
        for sub in env_subsets(env_name):
            try:
                print(f"  {sub:>20}  official_size={subset_size(env_name, sub)}")
            except Exception:
                print(f"  {sub}")
        return

    os.makedirs(args.out, exist_ok=True)
    methods = args.methods.split(",")
    Ns = [int(x) for x in args.N.split(",")]

    # AppWorld writes a per-open output dir tree to disk (logs + an on-disk SQLite copy). Each
    # world is cleaned up in a `finally` as it closes; additionally we namespace this run's outputs
    # with a unique base so the parent can sweep any orphans left by HARD-KILLED workers at the end.
    appworld_exp_base = None
    if env_name == "appworld" and not os.environ.get("TTS_APPWORLD_EXP_BASE"):
        import uuid as _uuid

        appworld_exp_base = f"tts_real_{_uuid.uuid4().hex[:8]}"
        os.environ["TTS_APPWORLD_EXP_BASE"] = appworld_exp_base

    # OfficeBench runs each rollout in a Docker container cleaned up in a `finally`; additionally we
    # stamp this run's containers with a unique run-id label so the parent can force-remove any
    # orphans left by HARD-KILLED workers (SIGKILL on OOM/timeout skips the per-session close()).
    officebench_run_id = None
    if env_name == "officebench" and not os.environ.get("TTS_OFFICEBENCH_RUN_ID"):
        import uuid as _uuid

        officebench_run_id = f"tts_{_uuid.uuid4().hex[:8]}"
        os.environ["TTS_OFFICEBENCH_RUN_ID"] = officebench_run_id

    specs = _specs_for(env_name, args)
    planned_specs = list(specs)
    if not planned_specs:
        print("[coverage-error] task selection is empty; check --tasks/--splits/--levels/--subsets.",
              file=sys.stderr)
        raise SystemExit(2)
    usable_only = False  # Run on all tasks, not just WM-eligible mutating tasks

    # Each task runs ALL its (method, N) cells together with a task-local rollout pool, so B0/B1/B2/
    # Ours over a task select from the SAME real-agent rollouts (fair best-of-N). Tasks are
    # independent, so --workers runs them in parallel (process pool for AppWorld, threads for τ²).
    cells = [(m, N, None, None) for m in methods for N in ([1] if m == "b0" else Ns)]

    # Incremental checkpoint + resume: each finished task's records are appended to records.jsonl
    # immediately, so an interrupted sweep (crash/OOM/hang-up/pool deadlock) keeps its progress and
    # can be continued with --resume (which skips tasks already fully present on disk).
    records_path = os.path.join(args.out, "records.jsonl")
    cell_checkpoint_dir = os.path.join(args.out, ".cell-checkpoints")
    expected_cells = _expected_cells(methods, Ns)
    prior_records: List[dict] = []
    if args.resume:
        # Recover cells written by workers before a previous parent crash/hard-kill, then canonicalize
        # them into records.jsonl before clearing the sidecars for this attempt.
        prior_records, done = _load_checkpoint(records_path, cell_checkpoint_dir)
        requested_self_aggregate = bool(
            getattr(args, "agent_self_aggregate", False))
        aggregation_mismatches = [
            record for record in prior_records
            if bool(record.get("agent_self_aggregate", False))
            != requested_self_aggregate
        ]
        if aggregation_mismatches:
            p.error(
                "--resume agent-self-aggregation mismatch. Use a new --out directory or the "
                "matching --agent-self-aggregate setting."
            )
        if env_name == "appworld":
            requested_mode = args.appworld_wm_tool_mode
            mismatched = [
                record for record in prior_records
                if record.get("appworld_wm_tool_mode", "text") != requested_mode
            ]
            if mismatched:
                recorded_modes = sorted({
                    record.get("appworld_wm_tool_mode", "text")
                    for record in mismatched
                })
                p.error(
                    f"--resume AppWorld tool-mode mismatch: records contain {recorded_modes}, "
                    f"but this run requested {requested_mode!r}. Use a new --out directory or the "
                    "matching --appworld-wm-tool-mode."
                )
            requested_coverage = _env_enabled(
                "TTS_APPWORLD_VERIFIER_COVERAGE_AWARE", False)
            coverage_mismatches = [
                record for record in prior_records
                if bool(record.get(
                    "appworld_verifier_coverage_aware", False
                )) != requested_coverage
            ]
            if coverage_mismatches:
                p.error(
                    "--resume AppWorld coverage-aware verifier mismatch. Use a new --out "
                    "directory or matching TTS_APPWORLD_VERIFIER_COVERAGE_AWARE."
                )
            requested_percheck = _env_enabled(
                "TTS_APPWORLD_VERIFIER_PERCHECK", False)
            percheck_mismatches = [
                record for record in prior_records
                if bool(record.get(
                    "appworld_verifier_percheck", False
                )) != requested_percheck
            ]
            if percheck_mismatches:
                p.error(
                    "--resume AppWorld per-check verifier mismatch. Use a new --out "
                    "directory or matching TTS_APPWORLD_VERIFIER_PERCHECK."
                )
            requested_samples = _positive_int_env(
                "TTS_APPWORLD_VERIFIER_SAMPLES", 1)
            sample_mismatches = [
                record for record in prior_records
                if int(record.get(
                    "appworld_verifier_samples", 1
                )) != requested_samples
            ]
            if sample_mismatches:
                p.error(
                    "--resume AppWorld verifier-sample mismatch. Use a new --out "
                    "directory or matching --verifier-samples."
                )
            requested_unknown_credit = _float_env(
                "TTS_APPWORLD_VERIFIER_UNKNOWN_CREDIT", 0.25)
            unknown_credit_mismatches = [
                record for record in prior_records
                if float(record.get(
                    "appworld_verifier_unknown_credit", 0.25
                )) != requested_unknown_credit
            ]
            if unknown_credit_mismatches:
                p.error(
                    "--resume AppWorld UNKNOWN-credit mismatch. Use a new --out directory "
                    "or matching TTS_APPWORLD_VERIFIER_UNKNOWN_CREDIT."
                )
            requested_margin = _float_env(
                "TTS_SELECTION_CONFIDENCE_MARGIN", 0.1)
            margin_mismatches = [
                record for record in prior_records
                if float(record.get(
                    "selection_confidence_margin", 0.1
                )) != requested_margin
            ]
            if margin_mismatches:
                p.error(
                    "--resume AppWorld confidence-margin mismatch. Use a new --out directory "
                    "or matching TTS_SELECTION_CONFIDENCE_MARGIN."
                )
            requested_preflights = _env_enabled(
                "TTS_APPWORLD_INSTRUCTION_PREFLIGHTS", False)
            preflight_mismatches = [
                record for record in prior_records
                if bool(record.get(
                    "appworld_instruction_preflights", False
                )) != requested_preflights
            ]
            if preflight_mismatches:
                p.error(
                    "--resume AppWorld instruction-preflight mismatch. Use a new --out "
                    "directory or matching --appworld-instruction-preflights setting."
                )
            score_mismatches = [
                record for record in prior_records
                if record.get("success_score", "simulator_success") != args.success_score
            ]
            if score_mismatches:
                recorded_scores = sorted({
                    record.get("success_score", "simulator_success")
                    for record in score_mismatches
                })
                p.error(
                    f"--resume AppWorld success-score mismatch: records contain "
                    f"{recorded_scores}, but this run requested {args.success_score!r}. "
                    f"Use a new --out directory or the matching --success-score."
                )
        if env_name == "officebench":
            requested_agent = getattr(args, "agent", None)
            agent_mismatches = [
                record for record in prior_records
                if record.get("solver_agent") != requested_agent
            ]
            if agent_mismatches:
                p.error(
                    f"--resume OfficeBench solver-agent mismatch: this run requested "
                    f"{requested_agent!r}. Use a new --out directory or the matching --agent."
                )
            requested_model = (
                getattr(args, "llm_agent", None)
                or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                or os.environ.get("WM_MODEL")
                or os.environ.get("TTS_AGENT_MODEL")
            )
            model_mismatches = [
                record for record in prior_records
                if record.get("llm_agent") != requested_model
            ]
            if model_mismatches:
                p.error(
                    f"--resume OfficeBench model mismatch: this run requested "
                    f"{requested_model!r}. Use a new --out directory or the matching --llm-agent."
                )
            requested_temperature = getattr(args, "temperature", None)
            temperature_mismatches = [
                record for record in prior_records
                if record.get("solver_temperature") != requested_temperature
            ]
            if temperature_mismatches:
                p.error(
                    "--resume OfficeBench temperature mismatch. Use a new --out directory "
                    "or the matching --temperature."
                )
            requested_max_steps = getattr(args, "max_steps", None)
            max_step_mismatches = [
                record for record in prior_records
                if record.get("solver_max_steps") != requested_max_steps
            ]
            if max_step_mismatches:
                p.error(
                    "--resume OfficeBench max-step mismatch. Use a new --out directory "
                    "or the matching --max-steps."
                )
            requested_seed_base = getattr(args, "seed_base", None)
            seed_mismatches = [
                record for record in prior_records
                if record.get("seed_base") != requested_seed_base
            ]
            if seed_mismatches:
                p.error(
                    "--resume OfficeBench seed-base mismatch. Use a new --out directory "
                    "or the matching --seed-base."
                )
            requested_mode = args.officebench_wm_tool_mode
            mismatched = [
                record for record in prior_records
                if record.get("officebench_wm_tool_mode", "text") != requested_mode
            ]
            if mismatched:
                recorded_modes = sorted({
                    record.get("officebench_wm_tool_mode", "text")
                    for record in mismatched
                })
                p.error(
                    f"--resume OfficeBench tool-mode mismatch: records contain {recorded_modes}, "
                    f"but this run requested {requested_mode!r}. Use a new --out directory or the "
                    "matching --officebench-wm-tool-mode."
                )
            requested_reference = _env_enabled(
                "TTS_OFFICEBENCH_VERIFIER_REFERENCE", True)
            reference_mismatches = [
                record for record in prior_records
                if bool(record.get(
                    "officebench_verifier_reference", True
                )) != requested_reference
            ]
            if reference_mismatches:
                p.error(
                    "--resume OfficeBench reference-verifier mismatch. Use a new --out "
                    "directory or matching TTS_OFFICEBENCH_VERIFIER_REFERENCE."
                )
            requested_state = _env_enabled(
                "TTS_OFFICEBENCH_VERIFIER_STATE", True)
            state_mismatches = [
                record for record in prior_records
                if bool(record.get(
                    "officebench_verifier_state", True
                )) != requested_state
            ]
            if state_mismatches:
                p.error(
                    "--resume OfficeBench state-grounding mismatch. Use a new --out "
                    "directory or matching TTS_OFFICEBENCH_VERIFIER_STATE."
                )
            requested_samples = _positive_int_env("TTS_OFFICEBENCH_VERIFIER_SAMPLES", 1)
            sample_mismatches = [
                record for record in prior_records
                if int(record.get("officebench_verifier_samples", 1)) != requested_samples
            ]
            if sample_mismatches:
                recorded_samples = sorted({
                    int(record.get("officebench_verifier_samples", 1))
                    for record in sample_mismatches
                })
                p.error(
                    f"--resume OfficeBench verifier-sample mismatch: records contain "
                    f"{recorded_samples}, but this run requested {requested_samples}. "
                    f"Use a new --out directory or matching --verifier-samples."
                )
            requested_percheck = _env_enabled(
                "TTS_OFFICEBENCH_VERIFIER_PERCHECK", False)
            percheck_mismatches = [
                record for record in prior_records
                if bool(record.get(
                    "officebench_verifier_percheck", False
                )) != requested_percheck
            ]
            if percheck_mismatches:
                p.error(
                    "--resume OfficeBench per-check verifier mismatch. Use a new --out "
                    "directory or matching TTS_OFFICEBENCH_VERIFIER_PERCHECK."
                )
            requested_atomic_cap = _positive_int_env(
                "TTS_OFFICEBENCH_VERIFIER_ATOMIC_CAP", 4)
            atomic_cap_mismatches = [
                record for record in prior_records
                if int(record.get(
                    "officebench_verifier_atomic_cap", 4
                )) != requested_atomic_cap
            ]
            if atomic_cap_mismatches:
                p.error(
                    "--resume OfficeBench atomic-cap mismatch. Use a new --out directory "
                    "or matching TTS_OFFICEBENCH_VERIFIER_ATOMIC_CAP."
                )
            requested_unknown_credit = _float_env(
                "TTS_OFFICEBENCH_VERIFIER_UNKNOWN_CREDIT", 0.25)
            unknown_credit_mismatches = [
                record for record in prior_records
                if float(record.get(
                    "officebench_verifier_unknown_credit", 0.25
                )) != requested_unknown_credit
            ]
            if unknown_credit_mismatches:
                p.error(
                    "--resume OfficeBench UNKNOWN-credit mismatch. Use a new --out directory "
                    "or matching TTS_OFFICEBENCH_VERIFIER_UNKNOWN_CREDIT."
                )
            requested_whole_weight = _float_env(
                "TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT", 0.25)
            whole_weight_mismatches = [
                record for record in prior_records
                if float(record.get(
                    "officebench_verifier_whole_weight", 0.25
                )) != requested_whole_weight
            ]
            if whole_weight_mismatches:
                p.error(
                    "--resume OfficeBench whole-audit weight mismatch. Use a new --out "
                    "directory or matching TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT."
                )
            requested_reconstruct_gold = _env_enabled(
                "TTS_OFFICEBENCH_WM_RECONSTRUCT_GOLD", False)
            reconstruct_gold_mismatches = [
                record for record in prior_records
                if bool(record.get(
                    "officebench_reconstruct_gold", False
                )) != requested_reconstruct_gold
            ]
            if reconstruct_gold_mismatches:
                p.error(
                    "--resume OfficeBench gold-reconstruction mismatch. Use a new --out "
                    "directory or matching TTS_OFFICEBENCH_WM_RECONSTRUCT_GOLD."
                )
            requested_comparative_rerank = _env_enabled(
                "TTS_OFFICEBENCH_COMPARATIVE_RERANK", False)
            comparative_rerank_mismatches = [
                record for record in prior_records
                if bool(record.get(
                    "officebench_comparative_rerank", False
                )) != requested_comparative_rerank
            ]
            if comparative_rerank_mismatches:
                p.error(
                    "--resume OfficeBench comparative-rerank mismatch. Use a new --out "
                    "directory or matching TTS_OFFICEBENCH_COMPARATIVE_RERANK."
                )
            requested_comparative_margin = _float_env(
                "TTS_OFFICEBENCH_COMPARATIVE_CONFIDENCE_MARGIN", 0.25)
            comparative_margin_mismatches = [
                record for record in prior_records
                if float(record.get(
                    "officebench_comparative_confidence_margin", 0.25
                )) != requested_comparative_margin
            ]
            if comparative_margin_mismatches:
                p.error(
                    "--resume OfficeBench comparative-confidence margin mismatch. Use a new --out "
                    "directory or matching TTS_OFFICEBENCH_COMPARATIVE_CONFIDENCE_MARGIN."
                )
            requested_margin = _float_env(
                "TTS_OFFICEBENCH_SELECTION_CONFIDENCE_MARGIN", 0.05)
            margin_mismatches = [
                record for record in prior_records
                if float(record.get(
                    "officebench_selection_confidence_margin", 0.05
                )) != requested_margin
            ]
            if margin_mismatches:
                p.error(
                    "--resume OfficeBench confidence-margin mismatch. Use a new --out directory "
                    "or matching TTS_OFFICEBENCH_SELECTION_CONFIDENCE_MARGIN."
                )
            score_mismatches = [
                record for record in prior_records
                if record.get("success_score", "simulator_success") != args.success_score
            ]
            if score_mismatches:
                recorded_scores = sorted({
                    record.get("success_score", "simulator_success")
                    for record in score_mismatches
                })
                p.error(
                    f"--resume OfficeBench success-score mismatch: records contain "
                    f"{recorded_scores}, but this run requested {args.success_score!r}. "
                    f"Use a new --out directory or the matching --success-score."
                )
        _rewrite_checkpoint(records_path, prior_records)
        shutil.rmtree(cell_checkpoint_dir, ignore_errors=True)
        before = len(specs)
        specs = [s for s in specs
                 if not expected_cells.issubset(done.get((env_name, s.subset, s.index), set()))]
        skipped = before - len(specs)
        print(f"(resume: {len(prior_records)} records on disk, skipping {skipped}/{before} "
              f"already-complete tasks, running {len(specs)} remaining)")
    else:
        # fresh run — truncate any stale checkpoint
        open(records_path, "w").close()
        shutil.rmtree(cell_checkpoint_dir, ignore_errors=True)
    os.makedirs(cell_checkpoint_dir, exist_ok=True)
    args._cell_checkpoint_dir = cell_checkpoint_dir

    ckpt_fh = open(records_path, "a", buffering=1)  # line-buffered append

    def _checkpoint(recs):
        for r in recs:
            ckpt_fh.write(json.dumps(r, default=str) + "\n")
        ckpt_fh.flush()
        os.fsync(ckpt_fh.fileno())

    try:
        new_records = _run_tasks(env_name, specs, cells, args, usable_only,
                                 desc=f"{env_name} tasks", on_task_done=_checkpoint)
    finally:
        ckpt_fh.close()
        if appworld_exp_base:
            from tts.env.real_appworld import sweep_experiment_outputs

            swept = sweep_experiment_outputs(appworld_exp_base)
            if swept:
                print(f"(swept {swept} orphaned AppWorld output dir(s))")
        if officebench_run_id:
            from tts.env.officebench_lowlevel import sweep_officebench_containers

            swept = sweep_officebench_containers(officebench_run_id)
            if swept:
                print(f"(swept {swept} orphaned OfficeBench container(s))")

    # Re-read the checkpoint so the report/aggregates reflect EVERYTHING on disk (prior + new),
    # de-duplicated per (env, spec_subset, spec_index, method, n_rollouts).
    all_records, _ = _load_checkpoint(records_path, cell_checkpoint_dir)
    if not all_records:  # checkpoint disabled/empty (e.g. no spec_index) — fall back to in-memory
        all_records = prior_records + new_records
    # Make records.jsonl self-contained: after this point users need not know about worker sidecars.
    _rewrite_checkpoint(records_path, all_records)

    run_status = getattr(args, "_last_run_status", {})
    cell_errors = _read_cell_sidecars(cell_checkpoint_dir, "error")
    manifest = _coverage_manifest(
        env_name, planned_specs, all_records, expected_cells, run_status, cell_errors)
    with open(os.path.join(args.out, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    shutil.rmtree(cell_checkpoint_dir, ignore_errors=True)
    args._cell_checkpoint_dir = None

    incomplete = manifest["partial_tasks"] + manifest["missing_tasks"]
    if incomplete and not args.allow_partial:
        print(
            f"\n[coverage-error] {incomplete}/{manifest['expected_tasks']} task(s) are missing one or "
            f"more expected cells. Refusing to publish an unequal-denominator TTS curve. "
            f"See {os.path.join(args.out, 'run_manifest.json')} and re-run with --resume, "
            "or omit --strict-coverage.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if incomplete:
        print(
            f"\n[coverage-warning] publishing PARTIAL metrics for {incomplete} task(s); "
            f"{manifest['missing_decisions']} missing decision cell(s) count as failures. "
            "Use --strict-coverage to reject incomplete runs.",
            file=sys.stderr,
        )

    metric_records = _records_with_missing_failures(
        env_name, planned_specs, all_records, expected_cells)
    print_report(metric_records)

    if args.depth_sweep:
        print("\nDepth sweep (fixed mode, success & cost vs L):")
        print(f"{'L':>3} {'success':>8} {'mean_spawns':>12} {'mean_tokens':>12} {'mean_reads':>11} {'pred_acc':>9}")
        for L in range(0, 4):
            # one L at a time so per-L aggregation stays exact; tasks within each L run in parallel
            recs = _run_tasks(env_name, specs, [("ours", max(Ns), "fixed", L)], args, usable_only,
                              desc=f"{env_name} depth-sweep L={L}")
            a = aggregate(recs, ["env"])[0] if recs else {}
            pa = a.get("prediction_accuracy")
            print(f"{L:>3} {(a.get('success_rate') or 0):>8.2f} {(a.get('mean_spawns') or 0):>12.1f} "
                  f"{(a.get('mean_tokens') or 0):>12.0f} {(a.get('mean_reads') or 0):>11.1f} "
                  f"{('%.2f' % pa) if pa is not None else '   n/a':>9}")

    dump_spawn_tree(env_name, args)

    aggs = aggregate(metric_records, ["env", "subset", "method", "n_rollouts"])
    with open(os.path.join(args.out, "aggregates.json"), "w") as fh:
        json.dump(aggs, fh, indent=2)
    write_csv(os.path.join(args.out, "aggregates.csv"), aggs)
    print(
        f"\nWrote {len(all_records)} recorded decisions and metrics over "
        f"{len(metric_records)} expected decisions ({manifest['missing_decisions']} missing "
        f"counted as failures), with {len(aggs)} aggregate rows to {args.out}/"
    )


if __name__ == "__main__":
    main()
