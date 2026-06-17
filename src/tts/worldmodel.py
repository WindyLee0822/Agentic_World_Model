"""The read-grounded world model (the *simulator*).

A world model predicts a trajectory's outcome **without touching the real environment**: it
materialises a *shadow* of the backing state by privileged, read-only reads of the
:class:`~tts.abcs.EntityStore`, then asks the env-specific :class:`~tts.abcs.ShadowEvaluator`
to predict the reward by replaying the candidate (and the gold reference) onto deep copies of
that shadow.

Two kinds share this machinery and differ only in *how far they read*:

* ``flat`` (**B1**) — one bounded read of the seed entities; no recursion, no transitive
  expansion.  Under-reads ⇒ the shadow is missing entities the trajectory touches ⇒ the replay
  cannot be validated ⇒ mis-prediction.
* ``recursive`` (**Ours**) — reads the seeds and recursively follows
  :meth:`~tts.abcs.ShadowEvaluator.child_references` to close over the touched subgraph.  The
  stop rule is configurable: ``fixed`` (descend to depth ``L``) or ``uncertainty`` (descend
  while read-coverage ``< tau``).  Every read is budgeted; every node is logged in a spawn tree.

A world model is constructed from an :class:`~tts.abcs.EntityStore` + a
:class:`~tts.abcs.ShadowEvaluator` only — never the :class:`~tts.abcs.TaskEnvironment` — so it
is structurally incapable of committing.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from .abcs import EntityStore, ShadowEvaluator, WorldModel
from .budget import Budget
from .cache import ReadCache
from .config import BudgetConfig, SimulatorConfig
from .simulator.spawn_tree import SpawnNode
from .trajectory import Observation, RewardInfo, Trajectory
from .types import Action

EntRef = Tuple[str, str]


class _Reader:
    """A privileged, read-only handle the agentic world-model subagent uses to *query* the DB.

    Each :meth:`query` is one read turn: it reads a batch of ``(type, id)`` refs (budgeted, cached,
    deep-copied) and logs them as a probe child node in the spawn tree, so the multi-turn read
    interaction is recorded.  It exposes only reads — there is no write path."""

    def __init__(self, wm: "ReadGroundedWorldModel", root: SpawnNode) -> None:
        self._wm = wm
        self._root = root

    def query(self, refs: List[EntRef], label: str = "") -> Dict[EntRef, Optional[Dict]]:
        refs = [(t, str(i)) for (t, i) in refs]
        node = SpawnNode(self._wm._nid("q"), self._root.node_id, 1,
                         label or f"query[{len(refs)}]", "probe")
        out: Dict[EntRef, Optional[Dict]] = {}
        for ref in refs:
            row = self._wm._read_entity(ref)
            node.n_reads += 1
            out[ref] = row
            self._wm._shadow[ref] = row
        node.keys_covered = sum(1 for r in refs if out.get(r) is not None)
        node.tokens, node.cost = self._wm._charge(len(refs))
        self._root.children.append(node)
        self._root.spawned = True
        return out

    def list_ids(self, entity_type: str, limit: int = 200) -> List[str]:
        return list(self._wm._store.list_ids(entity_type))[:limit]

    def probe(self, label: str) -> SpawnNode:
        """Log a read-only probe (e.g. a read-tool/KB retrieval call) in the spawn tree."""
        node = SpawnNode(self._wm._nid("p"), self._root.node_id, 1, label, "probe")
        node.n_reads = 1
        node.tokens, node.cost = self._wm._charge(1)
        self._root.children.append(node)
        self._root.spawned = True
        return node


class ReadGroundedWorldModel(WorldModel):
    """Flat or recursive read-grounded world model over an :class:`EntityStore`."""

    def __init__(
        self,
        entity_store: EntityStore,
        evaluator: ShadowEvaluator,
        config: SimulatorConfig,
        budget: Budget,
        budget_config: Optional[BudgetConfig] = None,
        cache: Optional[ReadCache] = None,
    ) -> None:
        self._store = entity_store
        self._eval = evaluator
        self._cfg = config
        self._budget = budget
        self._bc = budget_config or BudgetConfig()
        self._cache = cache if cache is not None else ReadCache(enabled=config.caching)
        self._root: Optional[SpawnNode] = None
        self._n = 0
        #: the shadow built during the last :meth:`evaluate` (kept for ``step`` reuse).
        self._shadow: Dict[EntRef, Dict] = {}

    # -- identity -------------------------------------------------------- #
    name = "world_model"

    @property
    def kind(self) -> str:
        return self._cfg.kind

    @property
    def spawn_root(self) -> Optional[SpawnNode]:
        return self._root

    def note(self, action: Action, content: str) -> None:
        """Record an externally-served observation (e.g. an exactly-executed read) in the
        evaluator's history so later predictions condition on it.  No-op if the evaluator has no
        history channel."""
        noter = getattr(self._eval, "note", None)
        if callable(noter):
            noter(action, content)

    def _nid(self, prefix: str = "n") -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    # -- privileged reads (budgeted + cached) ---------------------------- #
    def _read_entity(self, ref: EntRef) -> Optional[Dict]:
        key = f"{ref[0]}:{ref[1]}"
        cached, missing = self._cache.get_many([key])
        if key in cached:
            row = cached[key]
            return None if row.get("__absent__") else row
        row = self._store.read(ref[0], ref[1])  # deep copy, read-only
        self._budget.add_read(1)
        self._cache.put_many({key: row if row is not None else {"__absent__": True}})
        return row

    def _read_into(self, refs: List[EntRef], shadow: Dict[EntRef, Dict], node: SpawnNode) -> List[EntRef]:
        """Read ``refs`` into ``shadow``; return the child refs they transitively reference."""
        children: List[EntRef] = []
        for ref in refs:
            if ref in shadow:
                continue
            row = self._read_entity(ref)
            node.n_reads += 1
            if row is None:
                shadow[ref] = None  # mark absent (still counts as read/covered)
                continue
            shadow[ref] = row
            children.extend(self._eval.child_references(ref[0], ref[1], row))
        node.keys_covered = sum(1 for r in refs if r in shadow)
        node.tokens, node.cost = self._charge(len(refs))
        return children

    def _charge(self, n_reads: int) -> Tuple[int, float]:
        tokens = self._bc.tokens_per_node + n_reads * 0  # one node ~ one prompt
        cost = tokens * self._bc.cost_per_token
        self._budget.add_cost(tokens, cost)
        return tokens, cost

    # -- recursion stop rule --------------------------------------------- #
    def _should_recurse(self, confidence: float, depth: int) -> bool:
        if depth + 1 > self._cfg.hard_cap:
            return False
        if self._cfg.recursion_mode == "fixed":
            return depth < self._cfg.L
        # uncertainty: keep descending while not yet confident (coverage below tau)
        return confidence < self._cfg.tau

    @staticmethod
    def _dedup(refs: List[EntRef]) -> List[EntRef]:
        seen, out = set(), []
        for r in refs:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    @staticmethod
    def _split(refs: List[EntRef], branching: int) -> List[List[EntRef]]:
        branching = max(1, branching)
        n = len(refs)
        return [refs[(n * i) // branching:(n * (i + 1)) // branching] for i in range(branching)]

    # -- recursive node -------------------------------------------------- #
    def _node(self, refs: List[EntRef], depth: int, parent_id: Optional[str], shadow: Dict[EntRef, Dict]) -> SpawnNode:
        refs = [r for r in self._dedup(refs) if r not in shadow]
        node = SpawnNode(
            node_id=self._nid(),
            parent_id=parent_id,
            depth=depth,
            region=f"d{depth}[{len(refs)} refs]",
            role="simulator" if depth == 0 else "probe",
        )
        read_now = refs[: max(1, self._cfg.node_read_limit)]
        leftover = refs[max(1, self._cfg.node_read_limit):]
        child_refs = self._read_into(read_now, shadow, node)

        pending = [r for r in (leftover + child_refs) if r not in shadow]
        # coverage = fraction of this node's discovered refs already materialised
        denom = len(read_now) + len(pending)
        node.confidence = 1.0 if denom == 0 else len(read_now) / denom

        if pending and self._should_recurse(node.confidence, depth):
            node.spawned = True
            for group in self._split(pending, self._cfg.branching):
                if not group:
                    continue
                if self._budget.should_stop(extra_spawn=1):
                    node.note = "kill-switch: spawn/cost budget reached"
                    self._budget.trip()
                    break
                self._budget.add_spawn(1)
                node.children.append(self._node(group, depth + 1, node.node_id, shadow))
        return node

    # -- public API ------------------------------------------------------ #
    def _is_agentic(self) -> bool:
        """Ours (non-flat) with an agentic evaluator drives a multi-turn LLM-led read loop."""
        return bool(getattr(self._eval, "agentic", False)) and self.kind != "flat"

    def evaluate(self, trajectory: Trajectory) -> RewardInfo:
        if self._is_agentic():
            root = SpawnNode(self._nid(), None, 0, "agentic-eval", "simulator")
            self._root = root
            self._shadow = {}
            info = self._eval.reward_agentic(_Reader(self, root), trajectory.actions)
            return RewardInfo(reward=info.reward, success=info.success, grounded=False,
                              reason=info.reason)
        seeds = self._dedup(self._eval.seed_entities(trajectory.actions))
        shadow: Dict[EntRef, Dict] = {}
        if self.kind == "flat":
            # B1: a single bounded read of the seeds; no recursion / no transitive expansion.
            root = SpawnNode(self._nid(), None, 0, f"flat[{len(seeds)} refs]", "simulator")
            self._read_into(seeds[: max(1, self._cfg.node_read_limit)], shadow, root)
            covered = sum(1 for r in seeds if r in shadow)
            root.confidence = covered / max(1, len(seeds))
            self._root = root
        else:
            # Ours (non-agentic evaluators, e.g. the test double): recursive closure.
            self._root = self._node(seeds, 0, None, shadow)
        self._shadow = shadow
        info = self._eval.reward_from_shadow(shadow, trajectory.actions)
        return RewardInfo(reward=info.reward, success=info.success, grounded=False, reason=info.reason)

    def step(self, action: Action) -> Observation:
        """Predict one action's observation (no real write).
        For Ours with an agentic evaluator, the world-model subagent **proactively queries** the DB
        (multi-turn, read-only) and then emits the predicted tool result.  For B1/flat or a
        non-agentic evaluator it does a single bounded read and serves the evaluator's observation
        (or an advisory presence signal).  Nothing is ever written to the real environment.
        """
        if self._is_agentic():
            root = SpawnNode(self._nid("s"), None, 0, "agentic-step", "simulator")
            self._root = root
            try:
                content, is_error = self._eval.observe_agentic(_Reader(self, root), action)
                return Observation(content=content, error=bool(is_error))
            except Exception:  # never let observation prediction break sampling
                return Observation(content="missing-state", error=True)
        refs = self._dedup(self._eval.seed_entities([action]))
        node = SpawnNode(self._nid("s"), None, 0, f"step[{len(refs)}]", "probe")
        self._read_into(refs[: max(1, self._cfg.node_read_limit)], self._shadow, node)
        try:
            served = self._eval.observe(self._shadow, action)
        except Exception:  # never let observation prediction break sampling
            served = None
        if served is not None:
            content, is_error = served
            return Observation(content=content, error=bool(is_error))
        present = all(r in self._shadow and self._shadow[r] is not None for r in refs)
        return Observation(content=("ok" if present else "missing-state"), error=not present)

    def predict_step(self, action: Action) -> Observation:
        """Predict one action's observation, **always predicting** (never serving an exact read).

        Used for a *dirty* read — a read of an entity a prior write in this rollout already changed —
        so the observation reflects the predicted post-write state (the agentic predictor conditions
        on the running write history) rather than the unmutated real backing state.  Falls back to
        :meth:`step` for non-agentic evaluators (which already predict / serve advisory signals)."""
        if self._is_agentic():
            root = SpawnNode(self._nid("s"), None, 0, "agentic-step-predict", "simulator")
            self._root = root
            try:
                content, is_error = self._eval.observe_agentic(_Reader(self, root), action,
                                                               force_predict=True)
                return Observation(content=content, error=bool(is_error))
            except Exception:  # never let observation prediction break sampling
                return Observation(content="missing-state", error=True)
        return self.step(action)


def make_world_model(
    entity_store: EntityStore,
    evaluator: ShadowEvaluator,
    config: SimulatorConfig,
    budget: Budget,
    budget_config: Optional[BudgetConfig] = None,
    cache: Optional[ReadCache] = None,
) -> ReadGroundedWorldModel:
    if config.kind not in ("flat", "recursive"):
        raise KeyError(f"world model kind must be flat|recursive, got {config.kind!r}")
    return ReadGroundedWorldModel(entity_store, evaluator, config, budget, budget_config, cache)


__all__ = ["ReadGroundedWorldModel", "make_world_model", "EntRef"]
