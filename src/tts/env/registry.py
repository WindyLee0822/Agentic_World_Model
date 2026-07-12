"""Env registry: build a real :class:`~tts.abcs.TaskEnvironment` from a :class:`TaskSpec`.

No concrete env class is imported by callers — they go through :func:`make_env_from_spec` /
:func:`make_env`, which lazily import the real benchmark wrappers only when needed (the two
benchmarks live in mutually-exclusive conda envs).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from ..abcs import TaskEnvironment
from .datasets import TaskSpec


@lru_cache(maxsize=None)
def _tau2_usable_tasks(domain: str):
    """Real τ² tasks of a domain that the read-grounded world model can ground on.

    A task is *usable* when it is single-control with a DB-mutating agent action (see
    :func:`tts.env.real_tau2.tau2_task_world_model_eligible`): read-only tasks score 1.0 for any
    trajectory (no flat-vs-recursive gap) and dual-control / retrieval tasks expose no faithful
    read-grounded DB target.  This filter is a world-model concern only; the **real LLM agent** (B0)
    handles every official task, so the official-benchmark path uses :func:`_tau2_all_tasks`.
    """
    from tau2.registry import registry

    from .real_tau2 import tau2_task_world_model_eligible

    tasks = registry.get_tasks_loader(domain)()
    usable = [t for t in tasks if tau2_task_world_model_eligible(domain, t)]
    return tuple(usable)


@lru_cache(maxsize=None)
def _tau2_all_tasks(domain: str):
    """All official τ² tasks of a domain (the set the official benchmark / B0 is scored over)."""
    from tau2.registry import registry

    return tuple(registry.get_tasks_loader(domain)())


def tau2_usable_size(domain: str) -> int:
    try:
        return len(_tau2_usable_tasks(domain))
    except Exception:
        from .datasets import tau2_subset_size

        return tau2_subset_size(domain)


def make_env_from_spec(spec: TaskSpec, usable_only: bool = True) -> TaskEnvironment:
    if spec.env == "tau2":
        from .real_tau2 import Tau2TaskEnv

        domain = spec.domain or spec.subset
        tasks = _tau2_usable_tasks(domain) if usable_only else _tau2_all_tasks(domain)
        if not tasks:
            raise NotImplementedError(f"no usable tau2 tasks for domain {domain!r}")
        task = tasks[spec.index % len(tasks)]
        env = Tau2TaskEnv(domain=domain, task=task, subset=spec.subset)
        env.split, env.level, env.domain = spec.split, spec.level, domain
        return env
    if spec.env == "appworld":
        from .real_appworld import AppWorldTaskEnv

        env = AppWorldTaskEnv(split=spec.split or "dev", level=spec.level, index=spec.index,
                              subset=spec.subset)
        return env
    if spec.env == "officebench":
        from .real_officebench import OfficeBenchTaskEnv

        return OfficeBenchTaskEnv(subset=spec.subset, index=spec.index)
    raise KeyError(f"unknown env {spec.env!r}")


def make_env(env_name: str, index: int = 0, domain: Optional[str] = None, **kw) -> TaskEnvironment:
    """Convenience builder for a single task (used by tests / quick probes)."""
    if env_name == "tau2":
        spec = TaskSpec("tau2", domain or "airline", index, domain=domain or "airline")
        return make_env_from_spec(spec)
    if env_name == "appworld":
        spec = TaskSpec("appworld", "dev/L1", index, split="dev", level=1)
        return make_env_from_spec(spec)
    if env_name == "officebench":
        spec = TaskSpec("officebench", "1app", index, level=1)
        return make_env_from_spec(spec)
    raise KeyError(f"unknown env {env_name!r}")


def available_real_envs() -> List[str]:
    out: List[str] = []
    try:
        from .real_tau2 import real_tau2_available

        if real_tau2_available():
            out.append("tau2")
    except Exception:
        pass
    try:
        from .real_appworld import real_appworld_available

        if real_appworld_available():
            out.append("appworld")
    except Exception:
        pass
    try:
        from .real_officebench import real_officebench_available

        if real_officebench_available():
            out.append("officebench")
    except Exception:
        pass
    return out


__all__ = ["make_env_from_spec", "make_env", "available_real_envs", "tau2_usable_size"]
