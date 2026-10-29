"""Opt-in agent-side aggregation of a best-of-N trajectory pool.

The aggregator never sees verifier or grounded rewards.  It may compose AppWorld and
OfficeBench trajectories by referencing sampled steps.  Tau2 must retain one complete
candidate because its official evaluator needs the candidate's coherent assistant/user
message log, including dual-control user tool calls.
"""

from __future__ import annotations

import copy
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .trajectory import RolloutResult, Step, Trajectory


@dataclass(frozen=True)
class AggregationResult:
    trajectory: Trajectory
    source_index: Optional[int]
    tokens: int = 0
    wall_clock_s: float = 0.0


def _json_object(text: str) -> dict:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end < start:
            raise ValueError("aggregation response did not contain a JSON object")
        stripped = stripped[start : end + 1]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("aggregation response must be a JSON object")
    return value


def _prompt_value(value: Any, limit: int = 1_500) -> Any:
    """Keep aggregation evidence useful without allowing tool output to explode the prompt."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…[truncated]"
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(rendered) <= limit else rendered[:limit] + "…[truncated]"


def _candidate_payload(rollouts: Sequence[RolloutResult], *, observation_chars: int = 750) -> list:
    payload = []
    for candidate, rollout in enumerate(rollouts):
        steps = []
        for index, step in enumerate(rollout.trajectory.steps):
            observation = ""
            if step.observation is not None:
                observation = str(step.observation.content or "")[:observation_chars]
            steps.append(
                {
                    "step": index,
                    "action": {
                        "name": step.action.name,
                        "args": _prompt_value(step.action.args),
                    },
                    "observation": observation,
                    "observation_error": bool(
                        step.observation is not None and step.observation.error
                    ),
                }
            )
        payload.append(
            {
                "candidate": candidate,
                "done": bool(rollout.trajectory.done),
                "termination_reason": str(rollout.trajectory.termination_reason or ""),
                "final_message": rollout.trajectory.final_message[:1_000],
                "steps": steps,
            }
        )
    return payload


def _is_whole_candidate(refs: list[tuple[int, int]], rollouts: Sequence[RolloutResult]) -> Optional[int]:
    if not refs:
        return None
    candidate = refs[0][0]
    expected = [(candidate, index) for index in range(len(rollouts[candidate].trajectory.steps))]
    return candidate if refs == expected else None


def _aggregate_flat(
    env,
    config,
    rollouts: Sequence[RolloutResult],
    *,
    chat_client=None,
) -> AggregationResult:
    """Ask the solver model to turn sampled trajectories into one commit trajectory.

    Only references to sampled steps are accepted, preventing hallucinated tool names or
    arguments.  No verifier predictions, grounded scores, or success labels enter the prompt.
    """
    if not rollouts:
        raise ValueError("cannot aggregate an empty rollout pool")
    if len(rollouts) == 1:
        return AggregationResult(copy.deepcopy(rollouts[0].trajectory), 0, 0)

    if chat_client is None:
        from .llm import make_chat_client

        client_kwargs = dict(
            seed=config.seed + 9_000_000,
            temperature=0.0,
            max_tokens=config.solver.aggregation_max_tokens,
        )
        if config.solver.llm_agent:
            client_kwargs["model"] = config.solver.llm_agent
        chat_client = make_chat_client(**client_kwargs)

    env_name = str(env.env_name).lower()
    tau2_rule = (
        "For Tau2, preserve the coherent assistant/user dialogue: choose exactly one complete "
        "candidate. Return its zero-based integer index in `selected_candidate`; do not return "
        "individual step references."
        if env_name == "tau2"
        else
        "You may use one whole candidate or compose a better trajectory from referenced steps. "
        "Keep causal order, avoid repeated exploration, and include a terminal action only at the end."
    )
    system = (
        "You are the same task-solving agent performing self-aggregation over independent sampled "
        "trajectories. Produce the single trajectory that should be committed. You are not a reward "
        "verifier and receive no candidate scores. You may only reuse provided steps; never invent or "
        "edit a tool name or argument. "
        + (
            "Return JSON only with key `selected_candidate`. "
            if env_name == "tau2"
            else "Return JSON only with keys `steps` and `final_message`. `steps` must be a list "
            "of objects {\"candidate\": integer, \"step\": integer}. "
        )
        + tau2_rule
    )
    user = json.dumps(
        {
            "environment": env.env_name,
            "task_id": env.task_id,
            "instruction": env.restricted_view().instruction(),
            "max_steps": int(config.solver.max_steps),
            "candidates": _candidate_payload(rollouts),
        },
        ensure_ascii=False,
        default=str,
    )
    response = chat_client.chat(
        system,
        user,
        temperature=0.0,
        max_tokens=config.solver.aggregation_max_tokens,
    )
    tokens = int(getattr(chat_client, "last_usage", 0) or 0)
    try:
        value = _json_object(response)
    except (ValueError, TypeError, json.JSONDecodeError):
        return AggregationResult(copy.deepcopy(rollouts[0].trajectory), 0, tokens)
    if env_name == "tau2":
        selected = value.get("selected_candidate")
        # Invalid model formatting must not discard an otherwise expensive rollout pool.
        # Candidate zero is a neutral, reward-blind fallback and preserves Tau2's full dialogue.
        if not isinstance(selected, int) or not 0 <= selected < len(rollouts):
            selected = 0
        return AggregationResult(copy.deepcopy(rollouts[selected].trajectory), selected, tokens)

    raw_refs = value.get("steps")
    if not isinstance(raw_refs, list) or not raw_refs:
        return AggregationResult(copy.deepcopy(rollouts[0].trajectory), 0, tokens)
    if len(raw_refs) > int(config.solver.max_steps):
        raise ValueError("aggregated trajectory exceeds solver.max_steps")

    refs: List[tuple[int, int]] = []
    steps: List[Step] = []
    for raw in raw_refs:
        if not isinstance(raw, dict):
            continue
        candidate, step_index = raw.get("candidate"), raw.get("step")
        if not isinstance(candidate, int) or not isinstance(step_index, int):
            continue
        if candidate < 0 or candidate >= len(rollouts):
            continue
        source_steps = rollouts[candidate].trajectory.steps
        if step_index < 0 or step_index >= len(source_steps):
            continue
        refs.append((candidate, step_index))
        steps.append(copy.deepcopy(source_steps[step_index]))

    # If every reference was malformed, retain a valid sampled trajectory instead of causing the
    # entire task cell to fail. This fallback is deliberately reward-blind.
    if not steps:
        return AggregationResult(copy.deepcopy(rollouts[0].trajectory), 0, tokens)
    whole = _is_whole_candidate(refs, rollouts)
    if whole is not None:
        trajectory = copy.deepcopy(rollouts[whole].trajectory)
        requested_message = value.get("final_message")
        if isinstance(requested_message, str) and requested_message.strip():
            trajectory.final_message = requested_message.strip()
        return AggregationResult(trajectory, whole, tokens)

    final_message = value.get("final_message")
    if not isinstance(final_message, str):
        final_message = "Done."
    return AggregationResult(
        Trajectory(steps=steps, final_message=final_message, done=True),
        None,
        tokens,
    )


def _aggregate_tournament(env, config, rollouts: Sequence[RolloutResult]) -> AggregationResult:
    group_size = max(2, int(getattr(config.solver, "aggregation_group_size", 4) or 4))
    if len(rollouts) <= group_size:
        return _aggregate_flat(env, config, rollouts)
    groups = [list(rollouts[i:i + group_size]) for i in range(0, len(rollouts), group_size)]
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        results = list(pool.map(lambda group: _aggregate_flat(env, config, group), groups))
    intermediate = [
        RolloutResult(index=i, trajectory=result.trajectory, predicted=None, grounded=None)
        for i, result in enumerate(results)
    ]
    final = _aggregate_flat(env, config, intermediate)
    source_index = None
    if final.source_index is not None:
        local_index = results[final.source_index].source_index
        if local_index is not None:
            source_index = groups[final.source_index][local_index].index
    return AggregationResult(
        final.trajectory, source_index, final.tokens + sum(result.tokens for result in results)
    )


def aggregate_trajectories(
    env,
    config,
    rollouts: Sequence[RolloutResult],
    *,
    chat_client=None,
) -> AggregationResult:
    started = time.perf_counter()
    strategy = str(getattr(config.solver, "aggregation_strategy", "flat") or "flat").lower()
    if strategy == "flat" or len(rollouts) <= 1:
        result = _aggregate_flat(env, config, rollouts, chat_client=chat_client)
        return AggregationResult(
            result.trajectory, result.source_index, result.tokens, time.perf_counter() - started)
    if strategy != "tournament":
        raise ValueError(f"unknown aggregation strategy: {strategy}")
    if chat_client is not None:
        raise ValueError("parallel tournament aggregation does not accept a shared chat client")
    result = _aggregate_tournament(env, config, rollouts)
    return AggregationResult(
        result.trajectory, result.source_index, result.tokens, time.perf_counter() - started)


__all__ = ["AggregationResult", "aggregate_trajectories"]
