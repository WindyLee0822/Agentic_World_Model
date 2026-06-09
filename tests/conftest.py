"""Shared pytest fixtures: an in-memory **test double** of the trajectory env stack.

These fakes implement the trajectory ABCs (``EntityStore`` / ``ShadowEvaluator`` /
``TaskEnvironment``) over an in-memory dict so the env-agnostic machinery (recursion stop rules,
read cache, budget kill-switch, selection, single commit, simulator-never-writes) can be tested
without the real benchmark packages.

The fake task: there are several candidate *items*, each owning some *leaf* values; an item is
*feasible* iff the sum of its leaves is ≤ a cap.  The gold trajectory selects a feasible item.
A world model can only certify feasibility if it has read the item **and all its leaves** — so a
flat (``node_read_limit``-bounded) world model under-reads multi-leaf items and mis-predicts,
exactly like the real benchmarks.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Dict, List, Optional

import pytest

from tts.abcs import EntityStore, RestrictedView, ShadowEvaluator, TaskEnvironment, WorldModel
from tts.budget import Budget
from tts.cache import ReadCache
from tts.config import BudgetConfig, SimulatorConfig
from tts.trajectory import Observation, RewardInfo, Step, Trajectory
from tts.types import Action
from tts.worldmodel import make_world_model

SELECT = "select"


class FakeEntityStore(EntityStore):
    """Read-only in-memory store of items (-> leaf ids) and leaves (-> value)."""

    def __init__(self, items: Dict[str, List[str]], leaves: Dict[str, float]) -> None:
        self._items = items
        self._leaves = leaves
        self.reads = 0

    def read(self, entity_type: str, entity_id: str) -> Optional[Dict]:
        self.reads += 1
        if entity_type == "item" and entity_id in self._items:
            return {"leaf_ids": list(self._items[entity_id])}
        if entity_type == "leaf" and entity_id in self._leaves:
            return {"value": float(self._leaves[entity_id])}
        return None

    def list_ids(self, entity_type: str) -> List[str]:
        return list(self._items) if entity_type == "item" else list(self._leaves)


class FakeShadowEvaluator(ShadowEvaluator):
    """Certifies feasibility only when the item and all its leaves were read."""

    def __init__(self, gold_actions: List[Action], cap: float) -> None:
        self._gold = list(gold_actions)
        self._cap = float(cap)

    def _selected(self, actions: List[Action]) -> Optional[str]:
        for a in actions:
            if a.name == SELECT:
                return str(a.args["item_id"])
        return None

    def seed_entities(self, candidate_actions: List[Action]) -> List[tuple]:
        out: List[tuple] = []
        for a in list(candidate_actions) + self._gold:
            if a.name == SELECT:
                out.append(("item", str(a.args["item_id"])))
        return out

    def child_references(self, entity_type: str, entity_id: str, row: Dict) -> List[tuple]:
        if entity_type == "item" and row:
            return [("leaf", lid) for lid in row.get("leaf_ids", [])]
        return []

    def reward_from_shadow(self, shadow: Dict[tuple, Dict], candidate_actions: List[Action]) -> RewardInfo:
        item = self._selected(list(candidate_actions))
        if item is None:
            return RewardInfo(0.0, False, reason="no selection")
        irow = shadow.get(("item", item))
        if not irow:
            return RewardInfo(0.0, False, reason="under-read: item missing")
        leaves = irow.get("leaf_ids", [])
        if any(shadow.get(("leaf", lid)) is None for lid in leaves):
            return RewardInfo(0.0, False, reason="under-read: leaves missing")
        total = sum(float(shadow[("leaf", lid)]["value"]) for lid in leaves)
        ok = total <= self._cap
        return RewardInfo(1.0 if ok else 0.0, ok, reason=f"sum={total}<= {self._cap}")


class _FakeView(RestrictedView):
    def __init__(self, env: "FakeTaskEnv") -> None:
        self._env = env

    @property
    def task_id(self) -> str:
        return self._env.task_id

    def instruction(self) -> str:
        return "select a feasible item"

    def tool_schema(self) -> List[str]:
        return [SELECT]


class FakeTaskEnv(TaskEnvironment):
    """In-memory trajectory env: select a feasible item (sum of leaves <= cap)."""

    def __init__(self, items: Dict[str, List[float]], cap: float, task_id: str = "fake-0") -> None:
        # expand item -> leaf ids/values
        self._item_leaf_ids: Dict[str, List[str]] = {}
        self._leaves: Dict[str, float] = {}
        for iid, vals in items.items():
            ids = []
            for k, v in enumerate(vals):
                lid = f"{iid}:{k}"
                self._leaves[lid] = float(v)
                ids.append(lid)
            self._item_leaf_ids[iid] = ids
        self._cap = float(cap)
        self._totals = {iid: sum(self._leaves[l] for l in ls) for iid, ls in self._item_leaf_ids.items()}
        feasible = [iid for iid, t in self._totals.items() if t <= cap]
        self._gold_item = feasible[0] if feasible else next(iter(items))
        self._infeasible = [iid for iid, t in self._totals.items() if t > cap]
        self._task_id = task_id
        self._commits = 0
        self._committed: List[str] = []
        self.subset = "fake"
        self.split = self.level = self.domain = None

    # identity
    @property
    def env_name(self) -> str:
        return "fake_env"

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def mutating_tools(self):
        return {SELECT}

    # views / model
    def restricted_view(self) -> RestrictedView:
        return _FakeView(self)

    def entity_store(self) -> EntityStore:
        return FakeEntityStore(self._item_leaf_ids, self._leaves)

    def world_model(self, config: SimulatorConfig, budget, cache=None) -> WorldModel:
        return make_world_model(self.entity_store(), FakeShadowEvaluator(self._gold(), self._cap),
                                config, budget, BudgetConfig(), cache)

    def fork(self) -> "FakeTaskEnv":
        f = copy.deepcopy(self)
        return f

    def _gold(self) -> List[Action]:
        return [Action(SELECT, {"item_id": self._gold_item})]

    def gold_trajectory(self) -> Trajectory:
        return Trajectory(steps=[Step(a) for a in self._gold()])

    # stepping / eval / commit
    def step(self, action: Action) -> Observation:
        if action.name == SELECT:
            self._committed.append(str(action.args["item_id"]))  # mutate real state
        return Observation(content="ok")

    def _selected(self, actions: List[Action]) -> Optional[str]:
        for a in actions:
            if a.name == SELECT:
                return str(a.args["item_id"])
        return None

    def evaluate(self, trajectory: Trajectory) -> RewardInfo:
        item = self._selected(trajectory.actions)
        ok = item is not None and self._totals.get(item, float("inf")) <= self._cap
        return RewardInfo(1.0 if ok else 0.0, ok, grounded=True, reason="fake official eval")

    def commit_trajectory(self, trajectory: Trajectory) -> RewardInfo:
        for a in trajectory.actions:
            self.step(a)
        self._commits += 1
        return self.evaluate(trajectory)

    def state_hash(self) -> str:
        return hashlib.sha256(",".join(self._committed).encode()).hexdigest()

    def committed_count(self) -> int:
        return self._commits


def make_feasible_infeasible_env(task_id: str = "fake-0") -> FakeTaskEnv:
    """An env with one feasible (multi-leaf) item and several infeasible ones."""
    items = {
        "A": [1.0, 1.0, 1.0],   # sum 3 <= cap 5  -> feasible (needs all 3 leaves read)
        "B": [9.0, 9.0],         # sum 18 -> infeasible
        "C": [4.0, 4.0, 4.0],    # sum 12 -> infeasible
        "D": [10.0],             # sum 10 -> infeasible
    }
    return FakeTaskEnv(items, cap=5.0, task_id=task_id)


@pytest.fixture
def fake_env() -> FakeTaskEnv:
    return make_feasible_infeasible_env()


def make_recursive_wm(env: FakeTaskEnv, *, mode="uncertainty", L=2, tau=0.9, node_read_limit=1,
                      hard_cap=6, branching=2, budget=None):
    cfg = SimulatorConfig(kind="recursive", recursion_mode=mode, L=L, tau=tau,
                          node_read_limit=node_read_limit, hard_cap=hard_cap, branching=branching)
    return env.world_model(cfg, budget or Budget(), ReadCache())


# --------------------------------------------------------------------------- #
# A deep linear reference chain, for testing recursion *depth* precisely.
# --------------------------------------------------------------------------- #


class ChainStore(EntityStore):
    """An (effectively unbounded) chain: node k references node k+1."""

    def __init__(self, length: int = 50) -> None:
        self._length = length
        self.reads = 0

    def read(self, entity_type: str, entity_id: str):
        self.reads += 1
        k = int(entity_id)
        return {"k": k} if k < self._length else None

    def list_ids(self, entity_type: str):
        return [str(i) for i in range(self._length)]


class ChainEvaluator(ShadowEvaluator):
    def seed_entities(self, candidate_actions):
        return [("node", "0")]

    def child_references(self, entity_type, entity_id, row):
        return [("node", str(int(entity_id) + 1))]  # always one deeper

    def reward_from_shadow(self, shadow, candidate_actions):
        return RewardInfo(1.0, True, reason="chain")


def make_chain_wm(*, mode="fixed", L=2, tau=0.9, hard_cap=10, branching=1, node_read_limit=1, budget=None):
    from tts.worldmodel import make_world_model

    cfg = SimulatorConfig(kind="recursive", recursion_mode=mode, L=L, tau=tau,
                          node_read_limit=node_read_limit, hard_cap=hard_cap, branching=branching)
    wm = make_world_model(ChainStore(), ChainEvaluator(), cfg, budget or Budget(), BudgetConfig(), ReadCache())
    return wm


__all__ = [
    "FakeEntityStore", "FakeShadowEvaluator", "FakeTaskEnv",
    "make_feasible_infeasible_env", "make_recursive_wm", "make_chain_wm", "fake_env", "SELECT",
    "ChainStore", "ChainEvaluator",
]
