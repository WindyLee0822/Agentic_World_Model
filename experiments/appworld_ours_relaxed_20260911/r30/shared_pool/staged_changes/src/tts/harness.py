"""Trajectory-level decision orchestration and the swappable baselines (B0/B1/B2/Ours).

A *decision* samples ``N`` trajectories against the one shared real environment, scores each
(via its read-grounded world model, or — for B2 — an oracle on a forked snapshot), selects the
single most-promising trajectory, and commits **its whole action sequence** through the chokepoint.

    b0   → single trajectory, committed directly (no TTS, no simulation)
    b1   → flat world model      (sample N, predict reward by one bounded read, select best)
    b2   → forked snapshot oracle(sample N on per-rollout deep copies, official reward, select best)
    ours → recursive world model (sample N, read-grounded recursive prediction, select best)
"""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from .abcs import Agent, Stepper, TaskEnvironment
from .agents import ScriptedAgent
from .aggregation import aggregate_trajectories
from .budget import Budget
from .cache import ReadCache
from .commit import CommitChokepoint
from .config import Config
from .selection import Selector
from .simulator.spawn_tree import SpawnNode
from .trajectory import Observation, RewardInfo, RolloutResult, Step, Trajectory
from .types import Action, DecisionRecord

_METHOD_TO_KIND = {"b1": "flat", "ours": "recursive"}


class _NoOpStepper(Stepper):
    """A stepper that returns an empty observation (used for B0 plan execution)."""

    def step(self, action: Action) -> Observation:
        return Observation(content="", error=False)


@dataclass
class DecisionOutcome:
    record: DecisionRecord
    rollouts: List[RolloutResult]
    selected_index: int
    spawn_trees: List[SpawnNode] = field(default_factory=list)
    budget: Optional[Budget] = None


def drive(agent: Agent, view, stepper: Stepper, seed: int, max_steps: int = 64) -> Trajectory:
    """Run an agent against a stepper, producing a trajectory (the agent never touches real env)."""
    agent.reset(view, seed)
    traj = Trajectory(steps=[])
    for _ in range(max_steps):
        action = agent.next_action(traj)
        if action is None:
            break
        obs = stepper.step(action)
        traj.steps.append(Step(action=action, observation=obs))
    traj.final_message = agent.final_message(traj)
    return traj


def build_agent(env: TaskEnvironment, config: Config) -> Agent:
    """Default agent for the simulated path.

    ``llm_wm`` builds a real LLM tool-calling solver that *interacts with the world model*; every
    other (non-real-agent) mode uses the deterministic scripted stand-in following the gold plan
    w.p. ``p_correct``.
    """
    if getattr(config.solver, "agent", "scripted") == "llm_wm":
        from .llm.solver_agent import make_llm_solver_agent

        return make_llm_solver_agent(model=config.solver.llm_agent,
                                     temperature=config.solver.temperature,
                                     max_steps=config.solver.max_steps)
    plan = env.gold_trajectory().actions
    mutating = getattr(env, "mutating_tools", None)
    if mutating is None:
        from .env.real_tau2 import MUTATING_TOOLS
        mutating = set(MUTATING_TOOLS)
    return ScriptedAgent(plan=plan, mutating_names=set(mutating), p_correct=config.solver.p_correct)


def _merge_hist(hists: List[Dict[int, int]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for h in hists:
        for k, v in h.items():
            out[k] = out.get(k, 0) + v
    return out


def _solver_signature(solver) -> tuple:
    """Everything that determines how the agent is prompted/sampled (NOT the TTS method)."""
    return (
        getattr(solver, "agent", "scripted"),
        getattr(solver, "llm_agent", None),
        getattr(solver, "llm_user", None),
        getattr(solver, "temperature", 0.0),
        getattr(solver, "max_steps", 100),
        getattr(solver, "rollout_retries", 2),
    )

def _world_model_signature(simulator) -> tuple:
    """World-model settings, intentionally separate from agent settings."""

    return (
        getattr(
            simulator,
            "world_model_model",
            None,
        ),
        getattr(
            simulator,
            "world_model_base_url",
            None,
        ),
        getattr(
            simulator,
            "world_model_system_prompt_file",
            None,
        ),
        getattr(
            simulator,
            "world_model_temperature",
            0.6,
        ),
        getattr(
            simulator,
            "world_model_top_p",
            0.95,
        ),
        getattr(
            simulator,
            "world_model_top_k",
            20,
        ),
        getattr(
            simulator,
            "world_model_max_tokens",
            32768,
        ),
    )

def _agent_rollout(env: TaskEnvironment, config: Config, seed_i: int,
                   rollout_cache: Optional[Dict] = None):
    """Generate ONE real-agent rollout — the single, **method-independent** prompting path.

    The agent prompt depends only on the env + ``config.solver`` (never on ``config.method``), so
    B0/B1/B2/Ours are prompted identically.  When a ``rollout_cache`` is supplied, rollouts are
    memoised by ``(env, task, seed, solver-signature)`` so every method selects from the **same
    pool** of agent trajectories — the canonical fair best-of-``N`` comparison (and it avoids
    paying for the same LLM rollout more than once)."""
    key = (env.env_name, env.task_id, seed_i, _solver_signature(config.solver))
    if rollout_cache is not None and key in rollout_cache:
        return rollout_cache[key]
    traj, grounded = env.run_agent_rollout(config, seed_i)
    if rollout_cache is not None:
        rollout_cache[key] = (traj, grounded)
    return traj, grounded


def _real_agent_rollouts(env: TaskEnvironment, config: Config, method: str, n: int,
                         budget: Budget, cache: ReadCache,
                         rollout_cache: Optional[Dict] = None) -> List[RolloutResult]:
    """Sample ``n`` rollouts from the env's **real, official-prompt agent**.

    Generation is identical for every method (same prompt, same seeds, optionally the same shared
    pool); the method only changes the *selection signal* (``predicted``): the read-grounded
    world-model prediction (B1/Ours) where the env supports it, the oracle grounded reward (B2),
    or — for B0 — the grounded reward itself.  This keeps B1/B2/Ours a fair comparison that
    isolates the TTS method from the agent's prompting.
    """
    sim_cfg = replace(config.simulator, kind=_METHOD_TO_KIND.get(method, "none"))
    rollouts: List[RolloutResult] = []
    for i in range(n):
        traj, grounded = _agent_rollout(env, config, config.seed + i, rollout_cache)
        predicted, root = grounded, None
        if method in ("b1", "ours"):
            try:
                wm = env.world_model(sim_cfg, budget, cache)
                predicted = wm.evaluate(traj)  # predict from privileged reads (no commit)
                root = wm.spawn_root
            except NotImplementedError:
                predicted = grounded  # env without a read-grounded world model -> oracle fallback
        rollouts.append(RolloutResult(
            index=i, trajectory=traj, predicted=predicted, grounded=grounded, spawn_root=root,
            n_reads=root.total_reads() if root else 0,
            n_spawns=root.n_spawns() if root else 0,
            depth=root.max_depth() if root else 0,
            tokens=root.total_tokens() if root else 0,
            cost=root.total_cost() if root else 0.0,
        ))
    return rollouts


def _wm_agent_rollouts(env: TaskEnvironment, config: Config, method: str, n: int,
                       budget: Budget, cache: ReadCache,
                       rollout_cache: Optional[Dict] = None) -> List[RolloutResult]:
    """Sample ``n`` rollouts of the **official agent + user simulator interacting with the WM**.

    Each rollout runs the official τ² orchestrator with tool execution routed through the
    read-grounded world model (clean reads exact by default or predicted under the shared extension;
    writes predicted; the real env is never written). The selection signal is the WM's success belief; the realized
    official message log rides on the trajectory so the selected one can be committed and officially
    scored.  Generation is method-independent (same prompts/seeds); only the WM read depth differs
    (    flat for B1, recursive for Ours/B0)."""
    rollouts: List[RolloutResult] = []
    for i in range(n):
        rollout_seed = config.seed + i
        # key = (
        #     env.env_name,
        #     env.task_id,
        #     rollout_seed,
        #     _solver_signature(config.solver),
        #     method,
        #     bool(config.simulator.predict_all_actions),
        # )
        key = (
            env.env_name,
            env.task_id,
            rollout_seed,

            # agent identity
            _solver_signature(config.solver),

            method,

            # world-model identity
            _world_model_signature(
                config.simulator
            ),

            bool(
                config.simulator.predict_all_actions
            ),
        )
        if rollout_cache is not None and key in rollout_cache:
            traj, predicted, root = rollout_cache[key]
        else:
            retries = max(0, int(getattr(config.solver, "rollout_retries", 2) or 0))
            error = None
            for attempt in range(retries + 1):
                try:
                    traj, predicted, root = env.run_agent_rollout_wm(
                        config, method, rollout_seed, budget, cache)
                    error = None
                    break
                except NotImplementedError:
                    raise
                except Exception as err:  # noqa: BLE001 - preserve N with an explicit failed candidate
                    error = err
                    if attempt < retries:
                        continue
            if error is not None:
                reason = (
                    f"WM rollout seed {rollout_seed} failed after {retries + 1} attempt(s): "
                    f"{type(error).__name__}: {error}"
                )
                print(f"[rollout-error] {env.env_name}/{env.task_id}: {reason}")
                traj = Trajectory(
                    steps=[],
                    final_message=reason,
                    done=False,
                    termination_reason="rollout_error",
                )
                predicted = RewardInfo(
                    reward=0.0,
                    success=False,
                    grounded=False,
                    reason=reason,
                )
                root = None
            if rollout_cache is not None and (
                error is None or (
                    env.env_name == "appworld"
                    and method == "ours"
                    and config.simulator.predict_all_actions
                    and os.environ.get("TTS_APPWORLD_FIXED_CANDIDATE_POOL") == "1"
                )
            ):
                rollout_cache[key] = (traj, predicted, root)
        rollouts.append(RolloutResult(
            index=i, trajectory=traj, predicted=predicted, spawn_root=root,
            n_reads=root.total_reads() if root else 0,
            n_spawns=root.n_spawns() if root else 0,
            depth=root.max_depth() if root else 0,
            tokens=root.total_tokens() if root else 0,
            cost=root.total_cost() if root else 0.0,
            n_python_calls=int((predicted.metadata or {}).get("python_calls", 0)),
            n_python_failures=int((predicted.metadata or {}).get("python_failures", 0)),
            n_python_forced=int((predicted.metadata or {}).get("python_forced", 0)),
            n_python_mode_misses=int(
                (predicted.metadata or {}).get("python_mode_misses", 0)),
        ))
    return rollouts


def run_decision(env: TaskEnvironment, config: Config, agent: Optional[Agent] = None,
                 rollout_cache: Optional[Dict] = None, *, commit: bool = True) -> DecisionOutcome:
    method = config.method.lower()
    if method not in {"b0", "b1", "b2", "ours","agentworld",}:
        raise KeyError(f"unknown method {config.method!r}; choose from b0|b1|b2|ours")

    budget = Budget(max_spawns=config.budget.max_spawns, max_cost=config.budget.max_cost, max_reads=config.budget.max_reads)
    cache = ReadCache(enabled=config.simulator.caching)
    view = env.restricted_view()

    pre_hash = env.state_hash()
    t0 = time.perf_counter()
    rollouts: List[RolloutResult] = []
    n = 1 if method == "b0" else config.N

    solver_agent = getattr(config.solver, "agent", "scripted")
    if (
        method == "agentworld"
        and solver_agent != "llm_wm"
    ):
        raise ValueError(
            "agentworld requires solver.agent='llm_wm'"
        )
    use_real_agent = solver_agent == "llm" and agent is None
    use_wm_agent = solver_agent == "llm_wm" and agent is None
    self_aggregate = bool(getattr(config.solver, "self_aggregate", False)) and n > 1
    if not commit and not use_wm_agent:
        raise ValueError("commit=False is supported only for llm_wm simulation-only runs")

    if use_real_agent:
        rollouts = _real_agent_rollouts(env, config, method, n, budget, cache, rollout_cache)
        if self_aggregate:
            aggregated = aggregate_trajectories(env, config, rollouts)
            selected_index = aggregated.source_index if aggregated.source_index is not None else -1
            committed = RolloutResult(
                index=-1,
                trajectory=aggregated.trajectory,
                predicted=RewardInfo(),
            )
            chokepoint = CommitChokepoint(env)
            actual = chokepoint.commit(aggregated.trajectory)
            simulator_writes = 0 if env.state_hash() == pre_hash else 1
            return _build_outcome(
                env, config, method, rollouts, selected_index, actual,
                int(chokepoint.used), simulator_writes, budget, cache,
                time.perf_counter() - t0, committed_rollout=committed,
                selection_prediction=False, aggregation_tokens=aggregated.tokens,
                aggregation_wall_clock_s=aggregated.wall_clock_s,
            )
        if method == "b0":
            selected_index = 0
        else:
            selected_index, rollouts = Selector(config.success_score).select(rollouts)
        # the real agent already executed + officially scored each rollout: the selected rollout's
        # grounded reward IS the committed outcome (a single realized commit).
        actual = rollouts[selected_index].grounded
        simulator_writes = 0 if env.state_hash() == pre_hash else 1
        return _build_outcome(env, config, method, rollouts, selected_index, actual,
                              1, simulator_writes, budget, cache, time.perf_counter() - t0)

    if use_wm_agent:
        try:
            rollouts = _wm_agent_rollouts(env, config, method, n, budget, cache, rollout_cache)
        except NotImplementedError:
            if not commit:
                raise ValueError(
                    "simulation-only requires an environment with llm_wm rollout support")
            use_wm_agent = False  # no WM rollout for this env -> fall back to the drive path below
        else:
            # The official agent+user-sim generated each trajectory against the WM (no real write);
            # select by the WM's predicted success, then commit the chosen one to the real env.
            if self_aggregate:
                aggregated = aggregate_trajectories(env, config, rollouts)
                selected_index = (
                    aggregated.source_index if aggregated.source_index is not None else -1
                )
                committed = RolloutResult(
                    index=-1,
                    trajectory=aggregated.trajectory,
                    predicted=RewardInfo(),
                )
            else:
                refine_fn = getattr(env, "refine_rollout_scores", None)
                if callable(refine_fn):
                    refine_fn(config, rollouts)
                committed = None
            if not self_aggregate and method == "b0":
                selected_index = 0
            elif not self_aggregate:
                selected_index, rollouts = Selector(config.success_score).select(rollouts)
            simulator_writes = 0 if env.state_hash() == pre_hash else 1
            selected = committed or rollouts[selected_index]
            if not selected.trajectory.done:
                actual = RewardInfo(
                    reward=0.0,
                    success=False,
                    grounded=True,
                    reason="no valid world-model rollout was available to commit",
                )
                return _build_outcome(
                    env, config, method, rollouts, selected_index, actual,
                    0, simulator_writes, budget, cache, time.perf_counter() - t0,
                    prediction_is_grounded=commit, committed_rollout=committed,
                    selection_prediction=not self_aggregate,
                    aggregation_tokens=(aggregated.tokens if self_aggregate else 0),
                    aggregation_wall_clock_s=(aggregated.wall_clock_s if self_aggregate else 0.0),
                )
            if not commit:
                actual = RewardInfo(
                    reward=float(selected.predicted.reward),
                    success=bool(selected.predicted.success),
                    grounded=False,
                    reason="simulation-only run; no trajectory was committed or officially evaluated",
                )
                return _build_outcome(
                    env, config, method, rollouts, selected_index, actual,
                    0, simulator_writes, budget, cache, time.perf_counter() - t0,
                    prediction_is_grounded=False, committed_rollout=committed,
                    selection_prediction=not self_aggregate,
                    aggregation_tokens=(aggregated.tokens if self_aggregate else 0),
                    aggregation_wall_clock_s=(aggregated.wall_clock_s if self_aggregate else 0.0),
                )
            chokepoint = CommitChokepoint(env)
            actual = chokepoint.commit(selected.trajectory)  # the ONLY real write
            return _build_outcome(env, config, method, rollouts, selected_index, actual,
                                  int(chokepoint.used), simulator_writes, budget, cache,
                                  time.perf_counter() - t0, committed_rollout=committed,
                                  selection_prediction=not self_aggregate,
                                  aggregation_tokens=(aggregated.tokens if self_aggregate else 0),
                                  aggregation_wall_clock_s=(
                                      aggregated.wall_clock_s if self_aggregate else 0.0))

    agent = agent or build_agent(env, config)

    if method == "b0":
        # The scripted stand-in ignores observations, so B0 needs no stepper (NoOp).  The llm_wm
        # solver, however, is meant to *interact with the world model* — so drive its single rollout
        # against a (recursive) WM when the env supports one, falling back to NoOp otherwise.
        stepper: Stepper = _NoOpStepper()
        if getattr(config.solver, "agent", "scripted") == "llm_wm":
            try:
                stepper = env.world_model(replace(config.simulator, kind="recursive"), budget, cache)
            except NotImplementedError:
                stepper = _NoOpStepper()
        traj = drive(agent, view, stepper, config.seed)
        rollouts.append(RolloutResult(index=0, trajectory=traj, predicted=RewardInfo()))
        selected_index = 0
    elif method == "b2":
        for i in range(n):
            fork = env.fork()  # per-rollout deep snapshot (baseline ONLY)
            traj = drive(agent, view, fork, config.seed + i)
            predicted = fork.evaluate(traj)  # official reward on the snapshot = oracle
            rollouts.append(RolloutResult(index=i, trajectory=traj, predicted=predicted))
        if not self_aggregate:
            selected_index, rollouts = Selector(config.success_score).select(rollouts)
    else:  # b1 / ours
        kind = _METHOD_TO_KIND[method]
        sim_cfg = replace(config.simulator, kind=kind)
        for i in range(n):
            try:
                wm = env.world_model(sim_cfg, budget, cache)
            except NotImplementedError:
                # No read-grounded world model for this env/task (e.g. AppWorld, dual-control or
                # retrieval τ² task): degrade to a privileged fork + oracle reward for this rollout,
                # the same fallback contract the real-agent path uses.
                fork = env.fork()
                traj = drive(agent, view, fork, config.seed + i)
                rollouts.append(RolloutResult(index=i, trajectory=traj, predicted=fork.evaluate(traj)))
                continue
            traj = drive(agent, view, wm, config.seed + i)
            predicted = wm.evaluate(traj)
            root = wm.spawn_root
            rollouts.append(RolloutResult(
                index=i, trajectory=traj, predicted=predicted, spawn_root=root,
                n_reads=root.total_reads() if root else 0,
                n_spawns=root.n_spawns() if root else 0,
                depth=root.max_depth() if root else 0,
                tokens=root.total_tokens() if root else 0,
                cost=root.total_cost() if root else 0.0,
            ))
        if not self_aggregate:
            selected_index, rollouts = Selector(config.success_score).select(rollouts)

    # simulator-never-writes invariant: the real env is untouched during the sampling phase.
    simulator_writes = 0 if env.state_hash() == pre_hash else 1

    committed = None
    aggregation_tokens = 0
    aggregation_wall_clock_s = 0.0
    if self_aggregate:
        aggregated = aggregate_trajectories(env, config, rollouts)
        selected_index = aggregated.source_index if aggregated.source_index is not None else -1
        committed = RolloutResult(
            index=-1,
            trajectory=aggregated.trajectory,
            predicted=RewardInfo(),
        )
        aggregation_tokens = aggregated.tokens
        aggregation_wall_clock_s = aggregated.wall_clock_s
    selected = committed or rollouts[selected_index]
    chokepoint = CommitChokepoint(env)
    actual = chokepoint.commit(selected.trajectory)  # the ONLY real write
    return _build_outcome(env, config, method, rollouts, selected_index, actual,
                          int(chokepoint.used), simulator_writes, budget, cache,
                          time.perf_counter() - t0, committed_rollout=committed,
                          selection_prediction=not self_aggregate,
                          aggregation_tokens=aggregation_tokens,
                          aggregation_wall_clock_s=aggregation_wall_clock_s)


def _build_outcome(env, config, method, rollouts, selected_index, actual, n_commits,
                   simulator_writes, budget, cache, wall,
                   prediction_is_grounded: bool = True,
                   committed_rollout: Optional[RolloutResult] = None,
                   selection_prediction: bool = True,
                   aggregation_tokens: int = 0,
                   aggregation_wall_clock_s: float = 0.0) -> DecisionOutcome:
    selected = committed_rollout or rollouts[selected_index]
    n = len(rollouts)

    predicted_success: Optional[bool] = None
    prediction_correct: Optional[bool] = None
    predicted_confidence: Optional[float] = None
    if selection_prediction and method != "b0" and selected.predicted is not None:
        predicted_success = selected.predicted.success
        predicted_confidence = float(selected.predicted.reward)
        if prediction_is_grounded:
            prediction_correct = bool(selected.predicted.success) == bool(actual.success)

    spawn_trees = [r.spawn_root for r in rollouts if r.spawn_root is not None]
    record = DecisionRecord(
        method=method,
        env=env.env_name,
        subset=getattr(env, "subset", "default"),
        task_id=env.task_id,
        n_rollouts=n,
        recursion_mode=(config.simulator.recursion_mode if method == "ours" else None),
        selected_index=selected_index,
        committed_action=selected.trajectory.signature(),
        success=actual.success,
        reward=actual.reward,
        n_spawns=sum(r.n_spawns for r in rollouts),
        max_depth=max((r.depth for r in rollouts), default=0),
        depth_histogram=_merge_hist([t.depth_histogram() for t in spawn_trees]),
        n_reads=budget.n_reads,
        n_commits=n_commits,
        cache_hits=cache.hits,
        cache_misses=cache.misses,
        tokens=budget.tokens + int(aggregation_tokens),
        cost=budget.cost,
        wall_clock_s=wall,
        aggregation_wall_clock_s=aggregation_wall_clock_s,
        prediction_correct=prediction_correct,
        predicted_success=predicted_success,
        predicted_confidence=predicted_confidence,
        simulator_writes=simulator_writes,
        n_rollout_errors=sum(not rollout.trajectory.done for rollout in rollouts),
        n_python_calls=sum(rollout.n_python_calls for rollout in rollouts),
        n_python_failures=sum(rollout.n_python_failures for rollout in rollouts),
        n_python_forced=sum(rollout.n_python_forced for rollout in rollouts),
        n_python_mode_misses=sum(rollout.n_python_mode_misses for rollout in rollouts),
        selected_python_calls=selected.n_python_calls,
        selected_python_failures=selected.n_python_failures,
        split=getattr(env, "split", None),
        level=getattr(env, "level", None),
        domain=getattr(env, "domain", None),
    )
    return DecisionOutcome(record, rollouts, selected_index, spawn_trees, budget)


__all__ = ["run_decision", "drive", "build_agent", "DecisionOutcome"]
