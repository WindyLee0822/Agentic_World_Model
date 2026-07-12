#!/usr/bin/env python3
"""Run trajectory-level TTS experiments over the REAL environments and log the full metric set.

For each real task we sample ``N`` trajectories against one shared, non-forkable task
environment (executed + scored through the **official** AppWorld / τ²-bench runners).  Each
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
    python experiments/run.py --depth-sweep --env tau2 --tasks 12
    python experiments/run.py --report results/records.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
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
    return build_dataset(env_name, tasks_per_subset=args.tasks, splits=splits, levels=levels,
                         domains=domains, subsets_filter=sub_filter)


def _cfg_for(env_name, spec, method, N, args, recursion_mode=None, L=None) -> Config:
    return Config(method=method, env=env_name, N=N, seed=args.seed_base + spec.index * 1000,
                  success_score=args.success_score,
                  solver=SolverConfig(agent=args.agent, llm_agent=args.llm_agent,
                                      llm_user=args.llm_user or args.llm_agent,
                                      temperature=args.temperature, max_steps=args.max_steps,
                                      p_correct=args.p_correct),
                  simulator=sim_config(args, recursion_mode, L))


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
    for (method, N, recursion_mode, L) in cells:
        try:
            env = make_env_from_spec(spec, usable_only=usable_only)
        except (NotImplementedError, IndexError):
            continue
        cfg = _cfg_for(env_name, spec, method, N, args, recursion_mode, L)
        rec = run_decision(env, cfg, rollout_cache=cache).record.to_dict()
        # Stamp the spec identity so a resumed run can tell which tasks are already done
        # (the record stores the resolved ``task_id``, not the enumeration ``index``).
        rec["spec_subset"] = spec.subset
        rec["spec_index"] = spec.index
        records.append(rec)
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

    def _emit(recs):
        records.extend(recs)
        if on_task_done is not None and recs:
            on_task_done(recs)

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
    # Instead we wait in slices: if NO task completes for ``stall_s`` seconds we treat the remaining
    # futures as wedged, abandon them (their records are already checkpointed), and force-terminate
    # the workers so shutdown can't block.  A healthy slow batch still produces completions and is
    # never abandoned.
    stall_s = float(os.environ.get("TTS_RUN_STALL_TIMEOUT_S", "1200") or "1200")
    ex = Executor(**pool_kwargs)
    try:
        futures = {ex.submit(_run_task_cells, p): p[1] for p in payloads}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=stall_s, return_when=FIRST_COMPLETED)
            if not done:
                print(f"\n[hang-guard] no task finished in {stall_s:.0f}s; abandoning "
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
                    print(f"\n[task {getattr(spec, 'subset', '?')} "
                          f"idx={getattr(spec, 'index', '?')}] failed: {type(err).__name__}: {err}")
                progress.update(1, postfix=getattr(futures[fut], "subset", ""))
    finally:
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
        row.update({
            "decisions": len(rs),
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


def _load_checkpoint(path):
    """Read an existing ``records.jsonl`` for resume.

    Returns ``(records, done)`` where ``records`` is the de-duplicated record list already on disk
    (last writer wins per ``(env, spec_subset, spec_index, method, n_rollouts)``) and ``done`` maps
    each task key ``(env, spec_subset, spec_index)`` to the set of ``(method, n_rollouts)`` cells it
    already has.  Malformed lines are skipped so a half-written checkpoint still resumes cleanly.
    """
    by_cell: Dict[tuple, dict] = {}
    done: Dict[tuple, set] = defaultdict(set)
    if not os.path.exists(path):
        return [], done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            si = r.get("spec_index")
            sub = r.get("spec_subset", r.get("subset"))
            task_key = (r.get("env"), sub, si)
            cell = (r.get("method"), r.get("n_rollouts"))
            by_cell[(r.get("env"), sub, si, r.get("method"), r.get("n_rollouts"))] = r
            if si is not None:
                done[task_key].add(cell)
    return list(by_cell.values()), done


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default=None, choices=["appworld", "tau2", "officebench"])
    p.add_argument("--methods", default="b0,b1,b2,ours")
    p.add_argument("--N", default="1,2,4,8")
    p.add_argument("--tasks", type=int, default=None, help="tasks PER SUBSET (default: full usable size)")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--success-score", default="simulator_success")
    p.add_argument("--verifier-reference", action="store_true",
                   help="AppWorld llm_wm: enable the REFERENCE-GROUNDED verifier (audits the agent's write "
                        "log against the task's ground-truth checklist + gold values). Sets "
                        "TTS_APPWORLD_VERIFIER_REFERENCE=1 and, unless --success-score is given explicitly, "
                        "switches ranking to simulator_confidence. NOTE: uses the task answer key -> this is "
                        "reference-grounded (tau2-style), not blind WM-only.")
    p.add_argument("--verifier-samples", type=int, default=None,
                   help="K self-consistency samples for the WM verifier (sets TTS_APPWORLD_VERIFIER_SAMPLES; "
                        "K>=2 gives a graded confidence in {0,1/K,..,1}). Default 1 (binary).")
    p.add_argument("--verifier-percheck", action="store_true",
                   help="AppWorld llm_wm: PER-CHECK reference auditor — a focused sub-pass audits each "
                        "ground-truth check in isolation (MET/UNMET/UNKNOWN) and scores graded MET/decidable. "
                        "Implies --verifier-reference; sets TTS_APPWORLD_VERIFIER_PERCHECK=1.")
    p.add_argument("--sandbox-prompt", action="store_true",
                   help="AppWorld: prepend a truthful sandbox preamble to the agent prompt so safety-aligned "
                        "models (e.g. gpt-5.2) don't REFUSE consequential actions (orders/payments). Sets "
                        "TTS_APPWORLD_SANDBOX_PROMPT=1. NOTE: deviates from the official prompt — apply "
                        "uniformly across compared models for a fair A/B.")
    p.add_argument("--agent", default="llm", choices=["llm", "llm_wm", "scripted"],
                   help="real official-prompt LLM agent on the real env (default, reproduces the "
                        "benchmark) | llm_wm: LLM solver that interacts with the world model | "
                        "scripted: offline stub")
    p.add_argument("--llm-agent", default=None, help="agent model (default: Azure deployment / WM_MODEL, e.g. gpt-4.1)")
    p.add_argument("--llm-user", default=None, help="τ² user-simulator model (default: --llm-agent)")
    p.add_argument("--local", action="store_true",
                   help="Route every LLM call to a LOCAL, OpenAI-compatible server (vLLM / SGLang) "
                        "instead of the Azure/OpenAI API — for self-hosted models. Requires "
                        "--llm-agent <served-model-id> (e.g. Qwen-AgentWorld-35B-A3B). Forces the "
                        "openai backend for AppWorld/OfficeBench and routes τ² (litellm) via the "
                        "openai/ provider; no Azure endpoint or `az login` token is needed.")
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
    p.add_argument("--all-tasks", action="store_true",
                   help="τ²: run over ALL official tasks (default for --agent llm; the official "
                        "benchmark set). Scripted/world-model methods otherwise use mutating tasks only.")
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
    p.add_argument("--list-subsets", action="store_true")
    p.add_argument("--report", default=None)
    p.add_argument("--recursion-mode", default="uncertainty", choices=["uncertainty", "fixed"])
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
    p.add_argument("--keep-outputs", action="store_true",
                   help="Keep AppWorld per-task output dirs on disk (default: delete them as each "
                        "task finishes so large parallel runs don't accumulate temp files).")
    p.add_argument("--verbose", action="store_true",
                   help="Show the underlying tau2/litellm DEBUG logs (default: quiet — only the "
                        "progress bar and the result tables are printed).")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    args = p.parse_args()

    if not args.verbose:
        _quiet_logging()

    if args.keep_outputs:
        os.environ["TTS_KEEP_APPWORLD_OUTPUTS"] = "1"

    # Verifier configuration -> env vars, set HERE (in the parent) so forked workers inherit them.
    if args.verifier_samples is not None:
        os.environ["TTS_APPWORLD_VERIFIER_SAMPLES"] = str(args.verifier_samples)
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
    if args.sandbox_prompt:
        os.environ["TTS_APPWORLD_SANDBOX_PROMPT"] = "1"
        print("[agent] sandbox preamble ENABLED (reduces consequential-action refusals)")

    # Local, self-hosted model -> configure HERE (in the parent) so forked/spawned workers inherit
    # the OpenAI-compatible endpoint env vars (works for both the AppWorld process pool and the τ²
    # thread pool). Must run before any rollout worker is created.
    if args.local:
        from tts.llm.local import DEFAULT_LOCAL_BASE_URL, configure_local_llm, probe_local_server

        model = args.llm_agent or os.environ.get("TTS_LOCAL_MODEL")
        if not model:
            p.error("--local requires --llm-agent <served-model-id> "
                    "(e.g. --llm-agent Qwen-AgentWorld-35B-A3B)")
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

    avail = available_real_envs()
    env_name = args.env or (avail[0] if avail else None)
    if env_name is None:
        print("No real environment available in this interpreter. "
              "Activate `dw` (tau2) or `env` (appworld, with APPWORLD_ROOT), or set OFFICEBENCH_ROOT "
              "with a usable Docker daemon for officebench.")
        return

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
    usable_only = False  # Run on all tasks, not just WM-eligible mutating tasks

    # Each task runs ALL its (method, N) cells together with a task-local rollout pool, so B0/B1/B2/
    # Ours over a task select from the SAME real-agent rollouts (fair best-of-N). Tasks are
    # independent, so --workers runs them in parallel (process pool for AppWorld, threads for τ²).
    cells = [(m, N, None, None) for m in methods for N in ([1] if m == "b0" else Ns)]

    # Incremental checkpoint + resume: each finished task's records are appended to records.jsonl
    # immediately, so an interrupted sweep (crash/OOM/hang-up/pool deadlock) keeps its progress and
    # can be continued with --resume (which skips tasks already fully present on disk).
    records_path = os.path.join(args.out, "records.jsonl")
    expected_cells = _expected_cells(methods, Ns)
    prior_records: List[dict] = []
    if args.resume:
        prior_records, done = _load_checkpoint(records_path)
        before = len(specs)
        specs = [s for s in specs
                 if not expected_cells.issubset(done.get((env_name, s.subset, s.index), set()))]
        skipped = before - len(specs)
        print(f"(resume: {len(prior_records)} records on disk, skipping {skipped}/{before} "
              f"already-complete tasks, running {len(specs)} remaining)")
    else:
        # fresh run — truncate any stale checkpoint
        open(records_path, "w").close()

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
    all_records, _ = _load_checkpoint(records_path)
    if not all_records:  # checkpoint disabled/empty (e.g. no spec_index) — fall back to in-memory
        all_records = prior_records + new_records

    print_report(all_records)

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

    aggs = aggregate(all_records, ["env", "subset", "method", "n_rollouts"])
    with open(os.path.join(args.out, "aggregates.json"), "w") as fh:
        json.dump(aggs, fh, indent=2)
    write_csv(os.path.join(args.out, "aggregates.csv"), aggs)
    print(f"\nWrote {len(all_records)} decisions and {len(aggs)} aggregate rows to {args.out}/")


if __name__ == "__main__":
    main()
