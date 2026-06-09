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
        records.append(run_decision(env, cfg, rollout_cache=cache).record.to_dict())
    return records


def _run_tasks(env_name, specs, cells, args, usable_only, desc) -> List[dict]:
    """Run ``cells`` for every task in ``specs``, optionally across parallel workers.

    The dominant cost is the agent's LLM rollouts (each task = many sequential network calls), so
    these are I/O-bound and parallelising across tasks gives near-linear wall-clock speed-up.
    AppWorld is **not** thread-safe (thread-affine sqlite + module-global ground-truth loader), so
    it uses a PROCESS pool; τ² is thread-safe per task, so it uses the lighter THREAD pool (which
    also shares the already-loaded tau2 registry).
    """
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    progress = _Progress(total=len(specs), desc=desc)
    records: List[dict] = []

    if workers == 1 or len(specs) <= 1:
        for spec in specs:
            records.extend(_run_task_cells((env_name, spec, cells, args, usable_only)))
            progress.update(1, postfix=getattr(spec, "subset", ""))
        progress.close()
        return records

    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

    Executor = ProcessPoolExecutor if env_name == "appworld" else ThreadPoolExecutor
    payloads = [(env_name, spec, cells, args, usable_only) for spec in specs]
    with Executor(max_workers=workers) as ex:
        futures = {ex.submit(_run_task_cells, p): p[1] for p in payloads}
        for fut in as_completed(futures):
            try:
                records.extend(fut.result())
            except Exception as err:  # noqa: BLE001 - report and continue the sweep
                print(f"\n[task {getattr(futures[fut], 'subset', '?')} "
                      f"idx={getattr(futures[fut], 'index', '?')}] failed: {type(err).__name__}: {err}")
            progress.update(1, postfix=getattr(futures[fut], "subset", ""))
    progress.close()
    return records


def run_cell(env_name, method, N, args, recursion_mode=None, L=None, rollout_cache=None,
             progress=None) -> List[dict]:
    """Single-cell runner (one method, one N) — retained for ad-hoc/programmatic use."""
    records = []
    usable_only = args.agent == "scripted" and not args.all_tasks
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


def dump_spawn_tree(env_name, args) -> None:
    if env_name != "tau2":
        return  # the read-grounded spawn tree is a τ² (entity-store) construct
    try:
        env = make_env(env_name, index=args.seed_base)
    except Exception as e:  # pragma: no cover
        print(f"(spawn-tree dump skipped: {e})")
        return
    out = run_decision(env, Config(method="ours", env=env_name, N=4, seed=args.seed_base,
                                   solver=SolverConfig(agent="scripted", p_correct=args.p_correct),
                                   simulator=sim_config(args)))
    sel = out.rollouts[out.selected_index]
    root = sel.spawn_root
    if root is not None:
        print("\nExample spawn tree (selected trajectory):")
        print(root.render_ascii())
        with open(os.path.join(args.out, "spawn_tree.json"), "w") as fh:
            json.dump(root.to_dict(), fh, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default=None, choices=["appworld", "tau2"])
    p.add_argument("--methods", default="b0,b1,b2,ours")
    p.add_argument("--N", default="1,2,4,8")
    p.add_argument("--tasks", type=int, default=None, help="tasks PER SUBSET (default: full usable size)")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--success-score", default="simulator_success")
    p.add_argument("--agent", default="llm", choices=["llm", "scripted"],
                   help="real official-prompt LLM agent (default) vs offline scripted stub")
    p.add_argument("--llm-agent", default=None, help="agent model (default: Azure deployment / WM_MODEL, e.g. gpt-4.1)")
    p.add_argument("--llm-user", default=None, help="τ² user-simulator model (default: --llm-agent)")
    p.add_argument("--temperature", type=float, default=0.0, help="agent sampling temperature (B1/B2/Ours diversity)")
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
                   help="AppWorld difficulty levels, COMMA-separated (e.g. 1 or 1,2,3).")
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
    p.add_argument("--keep-outputs", action="store_true",
                   help="Keep AppWorld per-task output dirs on disk (default: delete them as each "
                        "task finishes so large parallel runs don't accumulate temp files).")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    args = p.parse_args()

    if args.keep_outputs:
        os.environ["TTS_KEEP_APPWORLD_OUTPUTS"] = "1"

    if args.report:
        print_report(load_records(args.report))
        return

    avail = available_real_envs()
    env_name = args.env or (avail[0] if avail else None)
    if env_name is None:
        print("No real environment available in this interpreter. "
              "Activate `dw` (tau2) or `env` (appworld, with APPWORLD_ROOT).")
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

    specs = _specs_for(env_name, args)
    usable_only = args.agent == "scripted" and not args.all_tasks

    # Each task runs ALL its (method, N) cells together with a task-local rollout pool, so B0/B1/B2/
    # Ours over a task select from the SAME real-agent rollouts (fair best-of-N). Tasks are
    # independent, so --workers runs them in parallel (process pool for AppWorld, threads for τ²).
    cells = [(m, N, None, None) for m in methods for N in ([1] if m == "b0" else Ns)]
    try:
        all_records = _run_tasks(env_name, specs, cells, args, usable_only, desc=f"{env_name} tasks")
    finally:
        if appworld_exp_base:
            from tts.env.real_appworld import sweep_experiment_outputs

            swept = sweep_experiment_outputs(appworld_exp_base)
            if swept:
                print(f"(swept {swept} orphaned AppWorld output dir(s))")

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

    with open(os.path.join(args.out, "records.jsonl"), "w") as fh:
        for r in all_records:
            fh.write(json.dumps(r, default=str) + "\n")
    aggs = aggregate(all_records, ["env", "subset", "method", "n_rollouts"])
    with open(os.path.join(args.out, "aggregates.json"), "w") as fh:
        json.dump(aggs, fh, indent=2)
    write_csv(os.path.join(args.out, "aggregates.csv"), aggs)
    print(f"\nWrote {len(all_records)} decisions and {len(aggs)} aggregate rows to {args.out}/")


if __name__ == "__main__":
    main()
