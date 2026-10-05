"""Real-benchmark dataset taxonomy: subset ids, official task counts, and enumeration.

This is the single source of truth for *which real tasks exist* in each benchmark, with no
dependency on any synthetic/analog environment.  AppWorld is organised by ``split`` x
``difficulty-level``; τ²-bench by ``domain``; OfficeBench by ``app-count`` (1/2/3 apps).  Counts
are the official benchmark numbers (AppWorld: ``appworld.load_task_ids(split, difficulty=level)``;
τ²-bench: ``tau2.registry.get_tasks_loader(domain)()``; OfficeBench: the ``tasks/<id>/subtasks/*``
tree grouped by the leading app-count digit of ``<id>``) and are used so a whole subset can be run
without the caller knowing its size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# AppWorld
# ---------------------------------------------------------------------------

#: The official benchmark test splits (the leaderboard set).  These ship the *test checkers*
#: (``evaluation.py`` + ``test_data.json`` + ``answer.json``) but withhold the *reference solution*
#: (``solution.py`` / ``required_apps.json``), so they ARE locally scorable via
#: ``AppWorld.evaluate()`` with ``ground_truth_mode="minimal"`` — only the GT-solution-replay
#: (``full`` mode) is unavailable.
APPWORLD_SPLITS = ("test_normal", "test_challenge")
#: The development / training splits; these additionally ship the reference solution (``full`` mode).
APPWORLD_LOCAL_SPLITS = ("dev", "train")
APPWORLD_LEVELS = (1, 2, 3)

#: Official AppWorld task counts per (split, difficulty-level).
APPWORLD_OFFICIAL_SIZE = {
    ("dev", 1): 30,
    ("dev", 2): 24,
    ("dev", 3): 3,
    ("train", 1): 36,
    ("train", 2): 36,
    ("train", 3): 18,
    ("test_normal", 1): 57,
    ("test_normal", 2): 48,
    ("test_normal", 3): 63,
    ("test_challenge", 1): 72,
    ("test_challenge", 2): 150,
    ("test_challenge", 3): 195,
}


def appworld_subsets() -> List[str]:
    return [f"{s}/L{l}" for s in APPWORLD_SPLITS for l in APPWORLD_LEVELS]


def parse_appworld_subset(subset: str) -> "tuple[str, int]":
    split, _, ltag = subset.partition("/")
    return split, int(ltag.lstrip("Ll"))


def appworld_subset_size(subset: str) -> int:
    key = parse_appworld_subset(subset)
    if key not in APPWORLD_OFFICIAL_SIZE:
        raise KeyError(f"unknown AppWorld subset {subset!r}")
    return APPWORLD_OFFICIAL_SIZE[key]


# ---------------------------------------------------------------------------
# τ²-bench
# ---------------------------------------------------------------------------

TAU2_DOMAINS = ("airline", "retail", "telecom", "telecom-workflow", "mock", "banking_knowledge")

#: Official τ²-bench task counts (from tau2.registry.get_tasks_loader(domain)()).
TAU2_OFFICIAL_SIZE = {
    "airline": 50,
    "retail": 114,
    "telecom": 114,
    "telecom-workflow": 114,
    "mock": 10,
    "banking_knowledge": 97,
}


def tau2_subsets() -> List[str]:
    return list(TAU2_DOMAINS)


def tau2_subset_size(domain: str) -> int:
    if domain not in TAU2_OFFICIAL_SIZE:
        raise KeyError(f"unknown tau2 domain {domain!r}; use {TAU2_DOMAINS}")
    return TAU2_OFFICIAL_SIZE[domain]


# ---------------------------------------------------------------------------
# OfficeBench
# ---------------------------------------------------------------------------

#: OfficeBench groups its tasks by how many office apps a task spans (1, 2 or 3).  A task id is
#: ``"<apps>-<n>"`` (e.g. ``"3-42"``) and its subtasks live in ``tasks/<id>/subtasks/<k>.json``;
#: the leading digit is the app-count, which is also how the official leaderboard reports scores.
OFFICEBENCH_APPS = (1, 2, 3)

#: Official OfficeBench subtask counts per app-count (the leaderboard groups: 93 / 95 / 112 = 300),
#: from the bundled ``tasks/*/subtasks/*.json`` tree.
OFFICEBENCH_OFFICIAL_SIZE = {
    "1app": 93,
    "2app": 95,
    "3app": 112,
}


def officebench_subsets() -> List[str]:
    return [f"{n}app" for n in OFFICEBENCH_APPS]


def parse_officebench_subset(subset: str) -> int:
    """``"2app" -> 2`` (the app-count that names a subset)."""
    n = subset[:-3] if subset.endswith("app") else subset
    return int(n)


def officebench_subset_size(subset: str) -> int:
    if subset not in OFFICEBENCH_OFFICIAL_SIZE:
        raise KeyError(f"unknown OfficeBench subset {subset!r}; use {officebench_subsets()}")
    return OFFICEBENCH_OFFICIAL_SIZE[subset]


# ---------------------------------------------------------------------------
# Unified accessors + dataset enumeration
# ---------------------------------------------------------------------------

REAL_ENVS = ("appworld", "tau2", "officebench")


def subsets(env: str) -> List[str]:
    if env == "appworld":
        return appworld_subsets()
    if env == "tau2":
        return tau2_subsets()
    if env == "officebench":
        return officebench_subsets()
    raise KeyError(f"unknown env {env!r}; use {REAL_ENVS}")


def subset_size(env: str, subset: str) -> int:
    if env == "appworld":
        return appworld_subset_size(subset)
    if env == "tau2":
        return tau2_subset_size(subset)
    if env == "officebench":
        return officebench_subset_size(subset)
    raise KeyError(f"unknown env {env!r}; use {REAL_ENVS}")


@dataclass(frozen=True)
class TaskSpec:
    """A real task to run: env + subset + index (and resolved split/level/domain)."""

    env: str
    subset: str
    index: int
    split: Optional[str] = None
    level: Optional[int] = None
    domain: Optional[str] = None


def build_dataset(
    env: str,
    tasks_per_subset: Optional[int] = None,
    *,
    splits: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[int]] = None,
    domains: Optional[Sequence[str]] = None,
    subsets_filter: Optional[Sequence[str]] = None,
) -> List[TaskSpec]:
    """Enumerate real :class:`TaskSpec` over the selected subsets.

    ``tasks_per_subset=None`` uses each subset's full official size, so the whole benchmark
    can be run without knowing the counts.  No filters ⇒ the whole benchmark.
    """

    def _n(sub: str) -> int:
        return tasks_per_subset if tasks_per_subset is not None else subset_size(env, sub)

    out: List[TaskSpec] = []
    if env == "appworld":
        valid_splits = APPWORLD_SPLITS + APPWORLD_LOCAL_SPLITS
        sel_splits = list(splits) if splits else list(APPWORLD_SPLITS)
        sel_levels = list(levels) if levels else list(APPWORLD_LEVELS)
        for s in sel_splits:
            if s not in valid_splits:
                raise KeyError(f"unknown AppWorld split {s!r}; use {valid_splits}")
            for lvl in sel_levels:
                if lvl not in APPWORLD_LEVELS:
                    raise KeyError(f"unknown AppWorld level {lvl!r}")
                sub = f"{s}/L{lvl}"
                if subsets_filter and sub not in subsets_filter:
                    continue
                for i in range(_n(sub)):
                    out.append(TaskSpec("appworld", sub, i, split=s, level=lvl))
    elif env == "tau2":
        sel = list(domains) if domains else list(TAU2_DOMAINS)
        for d in sel:
            if d not in TAU2_DOMAINS:
                raise KeyError(f"unknown tau2 domain {d!r}; use {TAU2_DOMAINS}")
            if subsets_filter and d not in subsets_filter:
                continue
            for i in range(_n(d)):
                out.append(TaskSpec("tau2", d, i, domain=d))
    elif env == "officebench":
        # OfficeBench subsets are app-counts; ``levels`` selects which (1/2/3 apps), mirroring how
        # ``--levels`` selects AppWorld difficulty, so the same runner flag drives both.
        sel_apps = list(levels) if levels else list(OFFICEBENCH_APPS)
        for n in sel_apps:
            if n not in OFFICEBENCH_APPS:
                raise KeyError(f"unknown OfficeBench app-count {n!r}; use {OFFICEBENCH_APPS}")
            sub = f"{n}app"
            if subsets_filter and sub not in subsets_filter:
                continue
            for i in range(_n(sub)):
                out.append(TaskSpec("officebench", sub, i, level=n))
    else:
        raise KeyError(f"unknown env {env!r}; use {REAL_ENVS}")
    return out


def dataset_size(env: str, **filters) -> int:
    return len(build_dataset(env, tasks_per_subset=None, **filters))


__all__ = [
    "REAL_ENVS",
    "APPWORLD_SPLITS",
    "APPWORLD_LOCAL_SPLITS",
    "APPWORLD_LEVELS",
    "APPWORLD_OFFICIAL_SIZE",
    "TAU2_DOMAINS",
    "TAU2_OFFICIAL_SIZE",
    "OFFICEBENCH_APPS",
    "OFFICEBENCH_OFFICIAL_SIZE",
    "TaskSpec",
    "subsets",
    "subset_size",
    "appworld_subsets",
    "appworld_subset_size",
    "tau2_subsets",
    "tau2_subset_size",
    "officebench_subsets",
    "officebench_subset_size",
    "parse_appworld_subset",
    "parse_officebench_subset",
    "build_dataset",
    "dataset_size",
]
