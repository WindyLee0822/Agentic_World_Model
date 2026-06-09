"""The REAL τ²-bench airline environment as a trajectory-level :class:`~tts.abcs.TaskEnvironment`.

A *trajectory* is a sequence of real airline tool-calls.  Execution and scoring cohere to the
**official** τ²-bench implementation:

* stepping uses ``environment.get_response(ToolCall(...))`` (the official tool execution that
  mutates the Pydantic ``FlightDB``);
* the terminal reward is the official ``evaluate_simulation(..., EvaluationType.ENV)`` (gold vs.
  predicted DB-hash comparison) — verified to return 1.0 for a gold action sequence and 0.0
  for a perturbed one.

The read-grounded world model predicts a candidate trajectory's reward **without touching the
real env**: it materialises a shadow of the touched entities by privileged, read-only reads of
:class:`Tau2EntityStore`, then replays the candidate and the gold reference onto deep copies of
that shadow and compares ``FlightDB.get_hash()``.  A *mutating* action's required entities must
all be present for the prediction to be certified — so the flat (B1) world model, which reads
only ``node_read_limit`` entities, mis-predicts multi-entity actions (book / update) while the
recursive (Ours) world model reads the full ``reservation → user + flights`` closure and
predicts correctly.

Requires the real ``tau2`` package (Python ≥3.12 — the ``dw`` conda env).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

from ..abcs import EntityStore, RestrictedView, ShadowEvaluator, TaskEnvironment, WorldModel
from ..config import BudgetConfig, SimulatorConfig
from ..trajectory import Observation, RewardInfo, Step, Trajectory
from ..types import Action
from ..worldmodel import make_world_model

#: airline tools that mutate the backing DB (their named entities must be read to certify).
MUTATING_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_passengers",
    "update_reservation_baggages",
    "send_certificate",
}


def real_tau2_available() -> bool:
    try:
        import tau2  # noqa: F401
    except Exception:
        return False
    return True


def real_tau2_subset_size(domain: str) -> int:
    from .datasets import tau2_subset_size

    if real_tau2_available():
        try:
            from tau2.registry import registry

            return len(list(registry.get_tasks_loader(domain)()))
        except Exception:
            pass
    return tau2_subset_size(domain)


def _airline_get_environment(db=None):
    from tau2.domains.airline.environment import get_environment

    return get_environment(db=db) if db is not None else get_environment()


def _named_entities(action: Action) -> List[tuple]:
    """The backing entities an action names in its arguments (reservation / user / flights)."""
    out: List[tuple] = []
    args = action.args or {}
    if args.get("reservation_id"):
        out.append(("reservation", args["reservation_id"]))
    if args.get("user_id"):
        out.append(("user", args["user_id"]))
    for f in (args.get("flights") or []):
        if isinstance(f, dict) and f.get("flight_number"):
            out.append(("flight", f["flight_number"]))
    return out


# --------------------------------------------------------------------------- #
# Privileged, read-only entity store
# --------------------------------------------------------------------------- #


class Tau2EntityStore(EntityStore):
    """Read-only, deep-copying reads of the real airline ``FlightDB`` (privileged)."""

    def __init__(self, db) -> None:
        self._db = db

    def _bucket(self, entity_type: str):
        return {
            "reservation": self._db.reservations,
            "user": self._db.users,
            "flight": self._db.flights,
        }.get(entity_type)

    def read(self, entity_type: str, entity_id: str) -> Optional[Dict]:
        bucket = self._bucket(entity_type)
        if bucket is None or entity_id not in bucket:
            return None
        return {"model": copy.deepcopy(bucket[entity_id])}  # deep copy: cannot mutate real env

    def list_ids(self, entity_type: str) -> List[str]:
        bucket = self._bucket(entity_type)
        return list(bucket.keys()) if bucket is not None else []


# --------------------------------------------------------------------------- #
# Env-specific shadow evaluator (read-graph + read-grounded reward prediction)
# --------------------------------------------------------------------------- #


class AirlineShadowEvaluator(ShadowEvaluator):
    """Drives the read-grounded world model for the airline domain."""

    def __init__(self, gold_actions: List[Action]) -> None:
        self._gold = list(gold_actions)

    # entities a candidate (and the gold reference) start reading from
    def seed_entities(self, candidate_actions: List[Action]) -> List[tuple]:
        out: List[tuple] = []
        for a in list(candidate_actions) + self._gold:
            out.extend(_named_entities(a))
        return out

    # a reservation transitively references its user + its flights (depth-1 reads)
    def child_references(self, entity_type: str, entity_id: str, row: Dict) -> List[tuple]:
        if entity_type != "reservation" or not row:
            return []
        model = row.get("model")
        if model is None:
            return []
        refs: List[tuple] = []
        uid = getattr(model, "user_id", None)
        if uid:
            refs.append(("user", uid))
        for f in (getattr(model, "flights", None) or []):
            fn = getattr(f, "flight_number", None)
            if fn:
                refs.append(("flight", fn))
        return refs

    def _required(self, candidate_actions: List[Action]) -> List[tuple]:
        req: List[tuple] = []
        for a in list(candidate_actions) + self._gold:
            if a.name in MUTATING_TOOLS:
                req.extend(_named_entities(a))
        # de-dup, preserve order
        return list(dict.fromkeys(req))

    def _build_db(self, shadow: Dict[tuple, Dict]):
        from tau2.domains.airline.data_model import FlightDB

        flights, users, reservations = {}, {}, {}
        for (etype, eid), row in shadow.items():
            if not row or row.get("model") is None:
                continue
            model = copy.deepcopy(row["model"])
            if etype == "reservation":
                reservations[eid] = model
            elif etype == "user":
                users[eid] = model
            elif etype == "flight":
                flights[eid] = model
        return FlightDB(flights=flights, users=users, reservations=reservations)

    @staticmethod
    def _replay_hash(db, actions: List[Action]) -> str:
        from tau2.data_model.message import ToolCall

        env = _airline_get_environment(db=copy.deepcopy(db))
        for i, a in enumerate(actions):
            env.get_response(ToolCall(id=f"c{i}", name=a.name, requestor="assistant", arguments=dict(a.args)))
        return env.tools.db.get_hash()

    def reward_from_shadow(self, shadow: Dict[tuple, Dict], candidate_actions: List[Action]) -> RewardInfo:
        present = {ref for ref, row in shadow.items() if row and row.get("model") is not None}
        required = self._required(candidate_actions)
        missing = [r for r in required if r not in present]
        if missing:
            # under-read: cannot certify a mutating action whose entity was not materialised.
            return RewardInfo(reward=0.0, success=False, grounded=False, reason=f"under-read: missing {missing[0]}")
        db = self._build_db(shadow)
        try:
            cand_hash = self._replay_hash(db, list(candidate_actions))
            gold_hash = self._replay_hash(db, self._gold)
        except Exception as exc:  # missing transitive state -> cannot certify
            return RewardInfo(reward=0.0, success=False, grounded=False, reason=f"replay-error: {type(exc).__name__}")
        ok = cand_hash == gold_hash
        return RewardInfo(reward=1.0 if ok else 0.0, success=ok, grounded=False,
                          reason="read-grounded match" if ok else "read-grounded mismatch")


# --------------------------------------------------------------------------- #
# Restricted view (solver-facing)
# --------------------------------------------------------------------------- #


class _Tau2RestrictedView(RestrictedView):
    def __init__(self, env: "Tau2TaskEnv") -> None:
        self._env = env

    @property
    def task_id(self) -> str:
        return self._env.task_id

    def instruction(self) -> str:
        return self._env._instruction

    def tool_schema(self) -> List[str]:
        return list(self._env._tool_names)


# --------------------------------------------------------------------------- #
# The task environment
# --------------------------------------------------------------------------- #


class Tau2TaskEnv(TaskEnvironment):
    """One real airline task, cohering to the official τ²-bench execution + evaluation."""

    def __init__(self, domain: str = "airline", task=None, subset: Optional[str] = None,
                 _env=None) -> None:
        if not real_tau2_available():
            raise RuntimeError("real tau2 not available (needs Python>=3.12 + tau2 install)")
        if domain != "airline":
            raise NotImplementedError("the trajectory action family is wired for the airline domain")
        self.domain = domain
        self.subset = subset or domain
        self._env = _env if _env is not None else _airline_get_environment()
        self._task = task
        self._commits = 0
        gold = (task.evaluation_criteria.actions if task and task.evaluation_criteria else None) or []
        self._gold = [Action(a.name, dict(a.arguments)) for a in gold]
        self._instruction = self._derive_instruction(task)
        self._tool_names = sorted(MUTATING_TOOLS | {
            "get_reservation_details", "get_user_details", "search_direct_flight",
            "search_onestop_flight", "calculate", "list_all_airports", "transfer_to_human_agents",
        })

    @staticmethod
    def _derive_instruction(task) -> str:
        if task is None:
            return "Complete the airline customer request."
        for attr in ("description", "user_scenario", "ticket"):
            val = getattr(task, attr, None)
            if val is None:
                continue
            text = getattr(val, "instructions", None) or getattr(val, "text", None) or str(val)
            if text:
                return str(text)[:2000]
        return f"Airline task {getattr(task, 'id', '?')}"

    # -- identity / metadata -------------------------------------------- #
    @property
    def env_name(self) -> str:
        return "tau2_airline"

    @property
    def mutating_tools(self):
        return set(MUTATING_TOOLS)

    @property
    def task_id(self) -> str:
        return str(getattr(self._task, "id", "?"))

    def restricted_view(self) -> RestrictedView:
        return _Tau2RestrictedView(self)

    def entity_store(self) -> EntityStore:
        return Tau2EntityStore(self._env.tools.db)

    def world_model(self, config: SimulatorConfig, budget, cache=None) -> WorldModel:
        return make_world_model(
            self.entity_store(), AirlineShadowEvaluator(self._gold), config, budget,
            BudgetConfig(), cache,
        )

    def fork(self) -> "Tau2TaskEnv":
        forked_env = _airline_get_environment(db=copy.deepcopy(self._env.tools.db))  # B2 baseline ONLY
        return Tau2TaskEnv(domain=self.domain, task=self._task, subset=self.subset, _env=forked_env)

    def gold_trajectory(self) -> Trajectory:
        return Trajectory(steps=[Step(action=a) for a in self._gold], final_message="Done.")

    # -- stepping / evaluation / commit --------------------------------- #
    def step(self, action: Action) -> Observation:
        from tau2.data_model.message import ToolCall

        tc = ToolCall(id=f"c{self._commits}", name=action.name, requestor="assistant", arguments=dict(action.args))
        res = self._env.get_response(tc)  # official execution; mutates the real FlightDB
        content = getattr(res, "content", None)
        return Observation(content=str(content) if content is not None else "", error=False, raw=res)

    def _official_reward(self, actions: List[Action]) -> RewardInfo:
        """Official τ² evaluation of a tool-call sequence (replays on a scratch env)."""
        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        scratch = _airline_get_environment()
        messages = []
        for i, a in enumerate(actions):
            tc = ToolCall(id=f"c{i}", name=a.name, requestor="assistant", arguments=dict(a.args))
            res = scratch.get_response(tc)
            messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
            messages.append(res if isinstance(res, ToolMessage) else ToolMessage(id=f"c{i}", role="tool", content=str(res)))
        messages.append(AssistantMessage(role="assistant", content="Done.", tool_calls=None))
        sim = SimulationRun(
            id="s", task_id=self.task_id, messages=messages,
            termination_reason=TerminationReason.USER_STOP,
            start_time="2020-01-01T00:00:00", end_time="2020-01-01T00:00:01", duration=1.0,
        )
        ri = evaluate_simulation(sim, self._task, EvaluationType.ENV, solo_mode=False, domain=self.domain)
        reward = float(getattr(ri, "reward", 0.0))
        return RewardInfo(reward=reward, success=reward >= 0.999, grounded=True, reason="official tau2 ENV evaluation")

    def evaluate(self, trajectory: Trajectory) -> RewardInfo:
        return self._official_reward(trajectory.actions)

    # -- REAL official agent (B0 / sampling) ---------------------------- #
    def run_agent_rollout(self, config, seed: int):
        """Run the **official** τ² agent loop and return ``(Trajectory, RewardInfo)``.

        This drives the official runner — the LLM customer-service agent (with the official
        :data:`tau2.agent.llm_agent.AGENT_INSTRUCTION` / ``SYSTEM_PROMPT``) conversing with the
        official user simulator over the real airline environment — and scores the conversation
        with the official ``evaluate_simulation`` (``EvaluationType.ALL``).  The returned reward is
        therefore the genuine benchmark reward, so a single rollout (B0) reproduces the official
        τ²-bench number.  The realized assistant tool-calls are returned as the trajectory.
        """
        from tau2.data_model.message import AssistantMessage
        from tau2.data_model.simulation import TextRunConfig
        from tau2.evaluator.evaluator import EvaluationType
        from tau2.run import run_single_task

        from ..llm import ensure_litellm_azure_env

        solver = config.solver
        default_model = ensure_litellm_azure_env()
        llm_agent = solver.llm_agent or default_model
        llm_user = solver.llm_user or llm_agent
        llm_args = {"temperature": solver.temperature} if solver.temperature else {}
        run_cfg = TextRunConfig(
            domain=self.domain, agent="llm_agent", user="user_simulator",
            llm_agent=llm_agent, llm_args_agent=dict(llm_args),
            llm_user=llm_user, llm_args_user=dict(llm_args),
            max_steps=solver.max_steps, seed=seed,
        )
        sim = run_single_task(run_cfg, self._task, seed=seed, evaluation_type=EvaluationType.ALL)

        steps: List[Step] = []
        final_message = "Done."
        for msg in sim.messages:
            if isinstance(msg, AssistantMessage):
                if msg.content:
                    final_message = str(msg.content)
                for tc in (msg.tool_calls or []):
                    steps.append(Step(action=Action(tc.name, dict(tc.arguments or {}))))
        traj = Trajectory(steps=steps, final_message=final_message)
        reward = float(getattr(sim.reward_info, "reward", 0.0) or 0.0)
        ri = RewardInfo(reward=reward, success=reward >= 0.999, grounded=True,
                        reason="official tau2 simulation (EvaluationType.ALL)")
        return traj, ri

    def commit_trajectory(self, trajectory: Trajectory) -> RewardInfo:
        for action in trajectory.actions:
            self.step(action)  # mutate the real env (single writer chokepoint upstream)
        self._commits += 1
        return self._official_reward(trajectory.actions)

    def state_hash(self) -> str:
        return self._env.get_db_hash()

    def committed_count(self) -> int:
        return self._commits


__all__ = [
    "Tau2TaskEnv",
    "Tau2EntityStore",
    "AirlineShadowEvaluator",
    "MUTATING_TOOLS",
    "real_tau2_available",
    "real_tau2_subset_size",
]
