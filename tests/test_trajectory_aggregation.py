from __future__ import annotations

import json

import pytest

from tts.aggregation import aggregate_trajectories
from tts.config import Config, SolverConfig
from tts.trajectory import Observation, RewardInfo, RolloutResult, Step, Trajectory
from tts.types import Action


class _View:
    def instruction(self):
        return "complete the benchmark task"


class _Env:
    def __init__(self, name):
        self.env_name = name
        self.task_id = "task-0"

    def restricted_view(self):
        return _View()


class _Chat:
    def __init__(self, response):
        self.response = response
        self.last_usage = 17
        self.calls = []

    def chat(self, system, user, **kwargs):
        self.calls.append((system, json.loads(user), kwargs))
        return json.dumps(self.response)


def _rollouts():
    return [
        RolloutResult(
            0,
            Trajectory(
                steps=[
                    Step(Action("inspect", {"path": "a"}), Observation("A")),
                    Step(Action("write", {"path": "out", "value": "old"})),
                ],
                final_message="candidate zero",
                sim_messages=["dialogue-zero"],
            ),
            RewardInfo(1.0, True),
        ),
        RolloutResult(
            1,
            Trajectory(
                steps=[
                    Step(Action("inspect", {"path": "b"}), Observation("B")),
                    Step(Action("write", {"path": "out", "value": "better"})),
                ],
                final_message="candidate one",
                sim_messages=["dialogue-one"],
            ),
            RewardInfo(0.0, False),
        ),
    ]


@pytest.mark.parametrize("env_name", ["appworld", "officebench"])
def test_agent_can_compose_referenced_steps_without_seeing_verifier_scores(env_name):
    chat = _Chat({
        "steps": [
            {"candidate": 0, "step": 0},
            {"candidate": 1, "step": 1},
        ],
        "final_message": "combined",
    })
    result = aggregate_trajectories(
        _Env(env_name),
        Config(solver=SolverConfig(llm_agent="test", self_aggregate=True)),
        _rollouts(),
        chat_client=chat,
    )
    assert [step.action.args for step in result.trajectory.steps] == [
        {"path": "a"},
        {"path": "out", "value": "better"},
    ]
    assert result.source_index is None
    assert result.tokens == 17
    prompt = json.dumps(chat.calls[0][1])
    assert "predicted" not in prompt and "reward" not in prompt and "success" not in prompt


def test_tau2_preserves_one_complete_dialogue_trajectory():
    chat = _Chat({"selected_candidate": 1})
    result = aggregate_trajectories(
        _Env("tau2"),
        Config(solver=SolverConfig(llm_agent="test", self_aggregate=True)),
        _rollouts(),
        chat_client=chat,
    )
    assert result.source_index == 1
    assert result.trajectory.sim_messages == ["dialogue-one"]
    assert result.trajectory is not _rollouts()[1].trajectory


def test_tau2_malformed_choice_falls_back_to_complete_candidate():
    chat = _Chat({
        "steps": [
            {"candidate": 0, "step": 0},
            {"candidate": 1, "step": 1},
        ],
        "final_message": "invalid mix",
    })
    result = aggregate_trajectories(
        _Env("tau2"),
        Config(solver=SolverConfig(llm_agent="test", self_aggregate=True)),
        _rollouts(),
        chat_client=chat,
    )
    assert result.source_index == 0
    assert result.trajectory.sim_messages == ["dialogue-zero"]


def test_invalid_composed_references_are_ignored():
    chat = _Chat({
        "steps": [
            {"candidate": 0, "step": 0},
            {"candidate": 99, "step": 99},
            {"candidate": 1, "step": 1},
        ],
        "final_message": "combined",
    })
    result = aggregate_trajectories(
        _Env("appworld"), Config(), _rollouts(), chat_client=chat
    )
    assert [step.action.name for step in result.trajectory.steps] == ["inspect", "write"]


def test_single_sample_is_identity_without_an_llm_call():
    class _NoCall:
        def chat(self, *args, **kwargs):
            raise AssertionError("N=1 must not call the aggregator")

    result = aggregate_trajectories(
        _Env("appworld"), Config(), _rollouts()[:1], chat_client=_NoCall()
    )
    assert result.source_index == 0
    assert result.trajectory.signature() == _rollouts()[0].trajectory.signature()


def test_tau2_tournament_uses_parallel_groups_then_final(monkeypatch):
    calls = []

    class _SelectingChat:
        last_usage = 11

        def chat(self, system, user, **kwargs):
            calls.append(json.loads(user))
            return '{"selected_candidate": 0}'

    monkeypatch.setattr("tts.llm.make_chat_client", lambda **kwargs: _SelectingChat())
    rollouts = []
    for i in range(8):
        rollout = _rollouts()[i % 2]
        rollout.index = i
        rollout.trajectory.sim_messages = [f"dialogue-{i}"]
        rollouts.append(rollout)
    config = Config(solver=SolverConfig(
        llm_agent="test", self_aggregate=True,
        aggregation_strategy="tournament", aggregation_group_size=4,
    ))
    result = aggregate_trajectories(_Env("tau2"), config, rollouts)
    assert result.source_index == 0
    assert result.trajectory.sim_messages == ["dialogue-0"]
    assert len(calls) == 3  # two concurrent groups, then one final
    assert result.tokens == 33
