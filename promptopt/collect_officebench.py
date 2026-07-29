"""Collect labeled OfficeBench verifier examples.

Unlike τ², OfficeBench has no replayable gold plan, so trajectories come from real agent rollouts.
For each task we run the official agent against the **world model** (writes predicted, reads exact —
no real mutation) via ``run_agent_rollout_wm``; the verifier call inside is intercepted by
:func:`promptopt.patching.capture_officebench` to record its inputs.  The TRUE label is then obtained
by committing that trajectory to a fresh container and scoring it with the official checks
(``commit_trajectory``).  This path needs Docker + the LLM agent, so it is slower than τ² collection.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .dataset import LabeledExample
from .patching import capture_officebench


def _config(model: Optional[str], temperature: float, max_steps: int):
    from tts.config import Config, SimulatorConfig, SolverConfig

    return Config(
        method="ours", env="officebench",
        solver=SolverConfig(agent="llm", llm_agent=model, temperature=temperature, max_steps=max_steps),
        simulator=SimulatorConfig(kind="recursive"),
    )


def collect_officebench_task(env, subset: str, task_id: str, seeds: Sequence[int], *,
                             model: Optional[str] = None, temperature: float = 0.7,
                             max_steps: int = 20, log=print) -> List[LabeledExample]:
    from tts.budget import Budget

    cfg = _config(model, temperature, max_steps)
    out: List[LabeledExample] = []
    for seed in seeds:
        sink: List[dict] = []
        try:
            with capture_officebench(sink):
                traj, _pred, _ = env.run_agent_rollout_wm(cfg, "ours", seed, Budget(max_spawns=1024))
        except Exception as exc:  # noqa: BLE001
            log(f"[officebench/{subset}] {task_id} seed {seed}: rollout failed ({type(exc).__name__}: {exc})")
            continue
        if not sink:
            log(f"[officebench/{subset}] {task_id} seed {seed}: verifier not reached")
            continue
        try:
            truth = bool(env.commit_trajectory(traj).success)
        except Exception as exc:  # noqa: BLE001
            log(f"[officebench/{subset}] {task_id} seed {seed}: commit failed ({type(exc).__name__})")
            continue
        out.append(LabeledExample(
            env="officebench", subset=subset, task_id=f"{task_id}#s{seed}", seed=seed,
            data=sink[-1]["slots"], true_label=truth,
            meta={"n_steps": len(traj.steps)}))
        log(f"[officebench/{subset}] {task_id} seed {seed}: label={'SUCCESS' if truth else 'FAIL'}")
    return out


def collect_officebench(apps: Sequence[int] = (1, 2, 3), tasks_per_app: int = 2,
                        seeds_per_task: int = 2, *, model: Optional[str] = None,
                        temperature: float = 0.7, max_steps: int = 20, log=print
                        ) -> List[LabeledExample]:
    from tts.env.datasets import TaskSpec
    from tts.env.registry import make_env_from_spec

    out: List[LabeledExample] = []
    for n in apps:
        subset = f"{n}app"
        got = 0
        for idx in range(tasks_per_app * 4):  # over-scan; some tasks error / need setup
            if got >= tasks_per_app:
                break
            try:
                env = make_env_from_spec(TaskSpec("officebench", subset, idx, level=n))
            except Exception as exc:  # noqa: BLE001
                log(f"[officebench/{subset}] task {idx}: skip ({type(exc).__name__})")
                continue
            exs = collect_officebench_task(env, subset, f"{subset}-{idx}", range(seeds_per_task),
                                           model=model, temperature=temperature,
                                           max_steps=max_steps, log=log)
            if exs:
                out.extend(exs)
                got += 1
    return out


__all__ = ["collect_officebench", "collect_officebench_task"]
