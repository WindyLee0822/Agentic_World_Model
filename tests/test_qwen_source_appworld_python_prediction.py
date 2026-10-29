"""AppWorld state predictions may choose direct output or restricted pure Python."""

from __future__ import annotations

import datetime
import json
from contextlib import contextmanager

import pytest

from tts_qwen.env import appworld_react_agent, appworld_wm, real_appworld
from tts_qwen.env.appworld_react_agent import (
    _instruction_needs_l2_targeted_skill,
    build_initial_messages,
    _maybe_add_instruction_guidance,
    _maybe_add_precision_preamble,
    _maybe_add_task_contract,
)
from tts_qwen.env.appworld_wm import AppWorldWMPredictor, AppWorldWMQuery
from tts_qwen.env.real_appworld import (
    _compact_appworld_candidate,
    _instruction_execution_plan,
    _instruction_preflights_enabled,
    _parse_comparative_candidate,
    _reference_execution_plan,
    _semantic_write_signature,
)
from tts_qwen.llm import python_tool
from tts_qwen.llm.python_tool import PythonToolResult, shared_python_tool


class _ScriptChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts = []

    def complete(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def test_real_appworld_agent_uses_the_rollout_seed(monkeypatch):
        captured = {}

        def make_chat_client(**kwargs):
            captured.update(kwargs)
            return object()

        @contextmanager
        def world_session(*_args):
            tracker = type(
                "Tracker",
                (),
                {"success": True, "pass_count": 1, "num_tests": 1},
            )()
            world = type(
                "World",
                (),
                {"evaluate": lambda self, suppress_errors=True: tracker},
            )()
            yield world

        monkeypatch.setattr("tts_qwen.llm.make_chat_client", make_chat_client)
        monkeypatch.setattr(real_appworld, "_world_session", world_session)
        monkeypatch.setattr(
            appworld_react_agent,
            "run_react_code_agent",
            lambda *_args, **_kwargs: ([], []),
        )

        env = object.__new__(real_appworld.AppWorldTaskEnv)
        env._task_id = "test_normal_1"
        env._experiment_base = "tts-test"
        env.level = 1
        config = type(
            "Config",
            (),
            {
                "solver": type(
                    "Solver",
                    (),
                    {
                        "llm_agent": "gpt-test",
                        "max_steps": 40,
                        "temperature": 1.0,
                    },
                )(),
            },
        )()

        env.run_agent_rollout(config, seed=37)

        assert captured == {"model": "gpt-test", "seed": 37}


@pytest.mark.parametrize("method", ["b0", "b1", "b2", "agentworld"])
def test_instruction_preflights_do_not_change_appworld_baselines(monkeypatch, method):
    monkeypatch.setenv("TTS_APPWORLD_INSTRUCTION_PREFLIGHTS", "1")

    assert not _instruction_preflights_enabled(method)


def test_instruction_preflights_remain_opt_in_for_ours(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_INSTRUCTION_PREFLIGHTS", raising=False)
    assert not _instruction_preflights_enabled("ours")

    monkeypatch.setenv("TTS_APPWORLD_INSTRUCTION_PREFLIGHTS", "1")
    assert _instruction_preflights_enabled("ours")


def test_instruction_preflights_anchor_only_first_bon_candidate(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_INSTRUCTION_PREFLIGHTS", "1")

    assert _instruction_preflights_enabled(
        "ours", candidate_seed=12_000, pool_seed=12_000)
    assert not _instruction_preflights_enabled(
        "ours", candidate_seed=12_001, pool_seed=12_000)
    assert not _instruction_preflights_enabled(
        "b1", candidate_seed=12_000, pool_seed=12_000)


def test_appworld_comparative_parser_rejects_invalid_candidates():
    assert _parse_comparative_candidate("BEST: 3\nREASON: complete", [0, 3]) == 3
    with pytest.raises(ValueError):
        _parse_comparative_candidate("BEST: 2", [0, 3])


def test_appworld_candidate_summary_contains_calls_without_environment_data():
    from tts_qwen.trajectory import RolloutResult, Trajectory
    from tts_qwen.types import Action

    trajectory = Trajectory(steps=[real_appworld.Step(
        action=Action("execute_code", {"code": "apis.gmail.delete_thread(email_thread_id=7)"}),
        observation=real_appworld.Observation(content="ok"),
    )])
    rollout = RolloutResult(
        index=0,
        trajectory=trajectory,
        predicted=real_appworld.RewardInfo(0.5, False, metadata={
            "persistent_verification": {
                "success": False,
                "reason": "missing the final required mutation",
            }
        }),
    )

    summary = _compact_appworld_candidate(rollout)

    assert "apis.gmail.delete_thread" in summary
    assert "missing the final required mutation" in summary
    assert "evaluator" not in summary.lower()


def test_semantic_write_signature_uses_resolved_writes_not_agent_source():
    from tts_qwen.trajectory import RolloutResult, Trajectory

    trace = [{
        "app": "gmail",
        "api": "send_email",
        "args": {"email_addresses": ["friend@example.com"], "access_token": "<redacted>"},
        "result": {"email_id": 1000001},
    }]
    rollouts = [
        RolloutResult(
            index=index,
            trajectory=Trajectory(final_message=f"different-source-{index}"),
            predicted=real_appworld.RewardInfo(
                0.0, False, metadata={"write_trace": trace},
            ),
        )
        for index in range(2)
    ]

    assert _semantic_write_signature(rollouts[0]) == _semantic_write_signature(rollouts[1])
    assert "access_token" not in _semantic_write_signature(rollouts[0])
    assert "email_id" not in _semantic_write_signature(rollouts[0])


def test_semantic_consensus_selects_repeated_resolved_plan(monkeypatch):
    from tts_qwen.trajectory import RolloutResult, Trajectory

    monkeypatch.setenv("TTS_APPWORLD_SEMANTIC_CONSENSUS", "1")
    env = object.__new__(real_appworld.AppWorldTaskEnv)
    env._instruction = "Send the requested email."
    config = type("Config", (), {"method": "ours"})()

    def rollout(index, recipient):
        return RolloutResult(
            index=index,
            trajectory=Trajectory(final_message="complete"),
            predicted=real_appworld.RewardInfo(0.0, False, metadata={
                "write_trace": [{
                    "app": "gmail",
                    "api": "send_email",
                    "args": {"email_addresses": [recipient]},
                    "result": {"email_id": 1000000 + index},
                }],
            }),
        )

    rollouts = [
        rollout(0, "wrong@example.com"),
        rollout(1, "right@example.com"),
        rollout(2, "right@example.com"),
        rollout(3, "other@example.com"),
    ]

    env.refine_rollout_scores(config, rollouts)

    override = rollouts[0].predicted.metadata["selection_override"]
    assert override == {
        "pool_size": 4,
        "best": 1,
        "source": "appworld_semantic_consensus",
    }


def test_appworld_nested_tiebreak_requires_strict_confidence_gain(monkeypatch):
    from tts_qwen.trajectory import RolloutResult, Trajectory

    monkeypatch.setenv("TTS_APPWORLD_NESTED_TIEBREAK", "1")
    env = object.__new__(real_appworld.AppWorldTaskEnv)
    config = type("Config", (), {"method": "ours"})()

    def rollout(index, score, *, hard_invalid=False):
        return RolloutResult(
            index=index,
            trajectory=Trajectory(final_message="complete"),
            predicted=real_appworld.RewardInfo(
                score, score >= 0.5, metadata={"hard_invalid": hard_invalid}),
        )

    rollouts = [rollout(0, 0.0, hard_invalid=True), rollout(1, 1.0)]
    env.refine_rollout_scores(config, rollouts)
    assert rollouts[0].predicted.metadata["selection_override"]["best"] == 1

    # A newly added equal-score candidate must not displace the prior prefix winner.
    rollouts.extend([rollout(2, 1.0), rollout(3, 0.0)])
    env.refine_rollout_scores(config, rollouts)
    assert rollouts[0].predicted.metadata["selection_override"] == {
        "pool_size": 4,
        "best": 1,
        "source": "appworld_nested_tiebreak",
    }

    tied = [rollout(0, 0.0, hard_invalid=True), rollout(1, 0.0)]
    env.refine_rollout_scores(config, tied)
    assert tied[0].predicted.metadata["selection_override"]["best"] == 0


@pytest.mark.parametrize("method", ["b0", "b1", "b2", "agentworld"])
def test_appworld_comparative_reranker_cannot_change_other_methods(
    monkeypatch, method,
):
    from tts_qwen.trajectory import RolloutResult, Trajectory

    monkeypatch.setenv("TTS_APPWORLD_COMPARATIVE_RERANK", "1")
    monkeypatch.setenv("TTS_APPWORLD_SEMANTIC_CONSENSUS", "1")
    monkeypatch.setattr(
        "tts_qwen.llm.make_chat_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not create verifier")),
    )
    env = object.__new__(real_appworld.AppWorldTaskEnv)
    env._instruction = "Do a task."
    config = type("Config", (), {"method": method})()
    rollouts = [
        RolloutResult(
            index=index,
            trajectory=Trajectory(steps=[], final_message=f"candidate-{index}"),
            predicted=real_appworld.RewardInfo(0.0, False, metadata={}),
        )
        for index in range(2)
    ]

    env.refine_rollout_scores(config, rollouts)

    assert all("selection_override" not in rollout.predicted.metadata for rollout in rollouts)


def test_persistent_qwen_comparison_audits_low_scored_completed_challenger(
    monkeypatch,
):
    from tts_qwen.trajectory import RolloutResult, Trajectory
    from tts_qwen.types import Action

    monkeypatch.setenv("TTS_APPWORLD_COMPARATIVE_RERANK", "1")
    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_PERSISTENT_WM", "1")
    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_SIMULATE_ALL_ACTIONS", "1")
    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_INITIAL_STATE_CONTEXT", "1")
    monkeypatch.delenv("TTS_APPWORLD_SEMANTIC_CONSENSUS", raising=False)
    monkeypatch.delenv("TTS_APPWORLD_NESTED_TIEBREAK", raising=False)
    monkeypatch.delenv("TTS_APPWORLD_OURS_QWEN_REFERENCE_RERANK", raising=False)

    class Client:
        def chat(self, _system, prompt, **_kwargs):
            assert "Candidate 0" in prompt and "Candidate 1" in prompt
            assert "replay-equivalent" in prompt
            assert "caught-failure recovery" in prompt
            return "BEST: 1\nREASON: candidate 1 fixes the concrete mutation order"

    monkeypatch.setattr("tts_qwen.llm.make_chat_client", lambda **_kwargs: Client())
    env = object.__new__(real_appworld.AppWorldTaskEnv)
    env._instruction = "Move every requested item."
    config = type("Config", (), {
        "method": "ours",
        "seed": 5,
        "solver": type("Solver", (), {"llm_agent": "Qwen3.5-35B-A3B"})(),
    })()

    def rollout(index, score):
        return RolloutResult(
            index=index,
            trajectory=Trajectory(
                steps=[real_appworld.Step(
                    action=Action(
                        "execute_code",
                        {"code": f"apis.spotify.add_song_to_playlist(playlist_id={10 + index}, song_id=20)"},
                    ),
                    observation=real_appworld.Observation(content="simulated"),
                )],
                final_message=f"candidate-{index}",
            ),
            predicted=real_appworld.RewardInfo(score, score >= 0.5, metadata={
                "hard_invalid": False,
                "write_trace": [{
                    "app": "spotify",
                    "api": "add_song_to_playlist",
                    "args": {"playlist_id": 10 + index, "song_id": 20},
                    "result": {"message": "simulated"},
                }],
            }),
        )

    rollouts = [rollout(0, 0.9), rollout(1, 0.1)]
    env.refine_rollout_scores(config, rollouts)

    decision = rollouts[0].predicted.metadata["appworld_comparative_rerank"]
    assert decision["contenders"] == [0, 1]
    assert decision["best"] == 1


def test_qwen_ours_oracle_diagnostic_records_without_selecting(monkeypatch):
    from tts_qwen.trajectory import RolloutResult, Trajectory

    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_ORACLE_DIAGNOSTIC", "1")
    monkeypatch.delenv("TTS_APPWORLD_NESTED_TIEBREAK", raising=False)
    monkeypatch.delenv("TTS_APPWORLD_SEMANTIC_CONSENSUS", raising=False)
    monkeypatch.delenv("TTS_APPWORLD_COMPARATIVE_RERANK", raising=False)
    env = object.__new__(real_appworld.AppWorldTaskEnv)
    evaluated = []

    def evaluate(trajectory):
        evaluated.append(trajectory.final_message)
        return real_appworld.RewardInfo(
            float(trajectory.final_message == "candidate-1"),
            trajectory.final_message == "candidate-1",
            grounded=True,
        )

    env.evaluate = evaluate
    config = type("Config", (), {
        "method": "ours",
        "solver": type("Solver", (), {"llm_agent": "Qwen3.5-35B-A3B"})(),
    })()
    rollouts = [
        RolloutResult(
            index=index,
            trajectory=Trajectory(final_message=f"candidate-{index}"),
            predicted=real_appworld.RewardInfo(0.0, False, metadata={}),
        )
        for index in range(2)
    ]

    env.refine_rollout_scores(config, rollouts)

    assert evaluated == ["candidate-0", "candidate-1"]
    assert [rollout.grounded.success for rollout in rollouts] == [False, True]
    assert all(
        "selection_override" not in rollout.predicted.metadata
        for rollout in rollouts
    )


@pytest.mark.parametrize(
    ("method", "model"),
    [("b1", "Qwen3.5-35B-A3B"), ("ours", "gpt-5.4")],
)
def test_qwen_ours_oracle_diagnostic_is_strictly_scoped(
    monkeypatch, method, model,
):
    from tts_qwen.trajectory import RolloutResult, Trajectory

    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_ORACLE_DIAGNOSTIC", "1")
    env = object.__new__(real_appworld.AppWorldTaskEnv)
    env.evaluate = lambda _trajectory: (_ for _ in ()).throw(
        AssertionError("out-of-scope rollout must not be replayed")
    )
    config = type("Config", (), {
        "method": method,
        "solver": type("Solver", (), {"llm_agent": model})(),
    })()
    rollout = RolloutResult(
        index=0,
        trajectory=Trajectory(final_message="candidate"),
        predicted=real_appworld.RewardInfo(0.0, False, metadata={}),
    )

    env.refine_rollout_scores(config, [rollout])

    assert rollout.grounded is None


def test_qwen_ours_reference_rerank_selects_without_oracle_labels(monkeypatch):
    from tts_qwen.trajectory import RolloutResult, Trajectory

    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_REFERENCE_RERANK", "1")
    monkeypatch.setattr(
        real_appworld,
        "load_task_checklist",
        lambda _task_id: "Gold recipient is right@example.com.",
    )

    class Client:
        def chat(self, _system, prompt, **_kwargs):
            assert "right@example.com" in prompt
            assert "oracle" not in prompt.lower()
            if "Valid candidate IDs: [0, 1]" in prompt:
                return "BEST: 1\nREASON: candidate 1 uses the required recipient"
            if "Valid candidate IDs: [1, 2, 3]" in prompt:
                return "BEST: 2\nREASON: candidate 2 initially appears preferable"
            assert "Valid candidate IDs: [2, 1]" in prompt
            return "BEST: 1\nREASON: the incumbent is better under reversed order"

    monkeypatch.setattr("tts_qwen.llm.make_chat_client", lambda **_kwargs: Client())
    env = object.__new__(real_appworld.AppWorldTaskEnv)
    env._task_id = "task-1"
    env._instruction = "Send the requested email."
    config = type("Config", (), {
        "method": "ours",
        "seed": 3,
        "solver": type("Solver", (), {"llm_agent": "Qwen3.5-35B-A3B"})(),
    })()

    def rollout(index, recipient):
        return RolloutResult(
            index=index,
            trajectory=Trajectory(final_message=f"candidate-{index}"),
            predicted=real_appworld.RewardInfo(0.0, False, metadata={
                # Individual simulation may falsely defer completion; the comparative selector
                # must still inspect every generated trajectory.
                "hard_invalid": True,
                "write_trace": [{
                    "app": "gmail",
                    "api": "send_email",
                    "args": {"email_addresses": [recipient]},
                    "result": {"email_id": index + 1},
                }],
            }),
        )

    rollouts = [
        rollout(0, "wrong@example.com"),
        rollout(1, "right@example.com"),
        rollout(2, "wrong2@example.com"),
        rollout(3, "wrong3@example.com"),
    ]

    env.refine_rollout_scores(config, rollouts)

    assert rollouts[0].predicted.metadata["selection_override"] == {
        "pool_size": 4,
        "best": 1,
        "source": "appworld_qwen_reference_rerank",
    }
    assert len(
        rollouts[0].predicted.metadata["appworld_qwen_reference_rerank"]["stages"]
    ) == 3
    assert all(rollout.grounded is None for rollout in rollouts)


def test_qwen_ours_reference_repair_replaces_trajectory_without_oracle(monkeypatch):
    from tts_qwen.trajectory import RolloutResult, Trajectory

    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_REFERENCE_REPAIR", "1")
    monkeypatch.delenv("TTS_APPWORLD_OURS_QWEN_REFERENCE_RERANK", raising=False)
    monkeypatch.setattr(
        real_appworld, "load_task_checklist", lambda _task_id: "Send to right@example.com."
    )
    monkeypatch.setattr(
        "tts_qwen.env.appworld_wm._api_doc",
        lambda _app, _api: {"parameters": [{"name": "email_addresses"}]},
    )

    class Client:
        def chat(self, _system, prompt, **_kwargs):
            assert "right@example.com" in prompt
            return (
                "DECISION: REPAIRED\nREASON: fixes recipient\n<python>\n"
                "apis.gmail.send_email(email_addresses=['right@example.com'])\n"
                "apis.supervisor.complete_task()\n</python>"
            )

    monkeypatch.setattr("tts_qwen.llm.make_chat_client", lambda **_kwargs: Client())
    env = object.__new__(real_appworld.AppWorldTaskEnv)
    env._task_id = "task-1"
    env._instruction = "Send the requested email."
    config = type("Config", (), {
        "method": "ours",
        "seed": 3,
        "solver": type("Solver", (), {"llm_agent": "Qwen3.5-35B-A3B"})(),
    })()
    rollout = RolloutResult(
        index=0,
        trajectory=Trajectory(steps=[real_appworld.Step(
            action=real_appworld.Action(
                "execute_code",
                {"code": "apis.supervisor.complete_task()"},
            )
        )]),
        predicted=real_appworld.RewardInfo(0.0, False, metadata={"write_trace": []}),
    )

    env.refine_rollout_scores(config, [rollout])

    repaired = rollout.trajectory.steps[0].action.args["code"]
    assert "right@example.com" in repaired
    assert rollout.predicted.metadata["appworld_qwen_reference_repair"]["decision"] == "REPAIRED"
    assert rollout.grounded is None


@pytest.mark.parametrize(
    ("method", "model"),
    [("b1", "Qwen3.5-35B-A3B"), ("b2", "Qwen3.5-35B-A3B"),
     ("agentworld", "Qwen-AgentWorld-35B-A3B"), ("ours", "gpt-5.4")],
)
def test_qwen_ours_reference_rerank_is_strictly_scoped(
    monkeypatch, method, model,
):
    from tts_qwen.trajectory import RolloutResult, Trajectory

    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_REFERENCE_RERANK", "1")
    monkeypatch.setattr(
        "tts_qwen.llm.make_chat_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("out-of-scope reranker must not create a client")
        ),
    )
    env = object.__new__(real_appworld.AppWorldTaskEnv)
    env._task_id = "task-1"
    env._instruction = "Do a task."
    config = type("Config", (), {
        "method": method,
        "seed": 0,
        "solver": type("Solver", (), {"llm_agent": model})(),
    })()
    rollouts = [
        RolloutResult(
            index=index,
            trajectory=Trajectory(final_message=f"candidate-{index}"),
            predicted=real_appworld.RewardInfo(0.0, False, metadata={}),
        )
        for index in range(2)
    ]

    env.refine_rollout_scores(config, rollouts)

    assert all(
        "selection_override" not in rollout.predicted.metadata
        for rollout in rollouts
    )


@pytest.fixture(autouse=True)
def _api_docs(monkeypatch):
    monkeypatch.setattr(appworld_wm, "_api_doc", lambda app, api: {})


def test_direct_state_prediction_remains_the_fast_path(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_WM_PYTHON", raising=False)
    chat = _ScriptChat(['{"items":[{"id":1}]}'])
    predictor = AppWorldWMPredictor(chat)

    result = predictor.predict_read(
        "gmail", "search_threads", {}, {"items": []}, {"items": "array"})

    assert result == {"items": [{"id": 1}]}
    assert predictor.n_python_calls == 0
    assert "PURE COMPUTATION OPTION" in chat.prompts[0]


def test_precision_prompt_adds_conditional_cross_app_ledger(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_PRECISION_PROMPT", "1")
    messages = _maybe_add_precision_preamble(
        [{"role": "system", "content": "base"}],
        enable_multi_app_skill=True,
        enable_l2_targeted_skill=True,
    )
    single = _maybe_add_precision_preamble(
        [{"role": "system", "content": "base"}],
        enable_multi_app_skill=False,
    )

    assert "MULTI-APP EXECUTION SKILL" in messages[0]["content"]
    assert "If the task spans multiple apps" in messages[0]["content"]
    assert "dependency graph" in messages[0]["content"]
    assert "never complete after a read-only attempt" in messages[0]["content"]
    assert "Do not net a payment against requests" in messages[0]["content"]
    assert "expand all three sources" in messages[0]["content"]
    assert "preserve task identity" in messages[0]["content"]
    assert "mutate the existing target file or note in place" in messages[0]["content"]
    assert "MULTI-APP EXECUTION SKILL" not in single[0]["content"]


def test_l2_targeted_precision_guidance_is_not_added_to_l3():
    messages = _maybe_add_precision_preamble(
        [{"role": "system", "content": "base"}],
        enable_multi_app_skill=True,
        enable_l2_targeted_skill=False,
    )

    assert "APPWORLD CHALLENGE EXECUTION SKILL" in messages[0]["content"]
    assert "APPWORLD LEVEL-2 TARGETED EXECUTION SKILL" not in messages[0]["content"]
    assert "Do not net a payment against requests" not in messages[0]["content"]
    assert "Use failure status only after" in messages[0]["content"]


@pytest.mark.parametrize(
    "instruction",
    [
        "Buy a portable air conditioner from the highest-rated seller.",
        "Download receipts of all my Amazon orders.",
        "Update my drafted resignation email.",
        "Update the alarm because our weekly standup time changed.",
    ],
)
def test_l2_targeted_skill_routes_proven_instruction_families(instruction):
    assert _instruction_needs_l2_targeted_skill(instruction)


def test_l2_targeted_skill_avoids_regressed_instruction_family():
    assert not _instruction_needs_l2_targeted_skill(
        "Label all Gmail notification threads and ignore spam."
    )


def test_multi_app_precision_prompt_defaults_on_but_can_be_disabled(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_PRECISION_PROMPT", raising=False)
    enabled = _maybe_add_precision_preamble(
        [{"role": "system", "content": "base"}],
        enable_multi_app_skill=True,
    )
    single = _maybe_add_precision_preamble(
        [{"role": "system", "content": "base"}],
        enable_multi_app_skill=False,
    )
    monkeypatch.setenv("TTS_APPWORLD_PRECISION_PROMPT", "0")
    disabled = _maybe_add_precision_preamble(
        [{"role": "system", "content": "base"}],
        enable_multi_app_skill=True,
    )

    assert "MULTI-APP EXECUTION SKILL" in enabled[0]["content"]
    assert enabled[0]["content"].endswith("base")
    assert single[0]["content"] == "base"
    assert disabled[0]["content"] == "base"


def test_instruction_guidance_selects_only_relevant_playbooks():
    messages = _maybe_add_instruction_guidance(
        [{"role": "system", "content": "base"}],
        "Compile all emailed bills into a zip, then request the shared expense on Venmo.",
        enabled=True,
    )
    content = messages[0]["content"]

    assert "GMAIL WORKFLOW REMINDER" in content
    assert "RELATIONSHIP AND MONEY REMINDER" in content
    assert "SPOTIFY WORKFLOW REMINDER" not in content
    assert "AMAZON WORKFLOW REMINDER" not in content
    assert "TODOIST EXECUTION REMINDER" not in content
    assert content.endswith("base")


def test_instruction_guidance_avoids_broad_relationship_and_spotify_matches():
    messages = _maybe_add_instruction_guidance(
        [{"role": "system", "content": "base"}],
        "Reply to my wife's email with songs from my Spotify playlist.",
        enabled=True,
    )

    assert messages[0]["content"] == "base"


def test_instruction_guidance_can_restore_broad_routing(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_GUIDANCE_ROUTING", "broad")
    messages = _maybe_add_instruction_guidance(
        [{"role": "system", "content": "base"}],
        "Reply to my wife's email with songs from my Spotify playlist.",
        enabled=True,
    )

    assert "GMAIL WORKFLOW REMINDER" in messages[0]["content"]
    assert "SPOTIFY WORKFLOW REMINDER" in messages[0]["content"]
    assert "RELATIONSHIP AND MONEY REMINDER" in messages[0]["content"]


def test_instruction_guidance_is_disabled_for_lower_levels():
    messages = _maybe_add_instruction_guidance(
        [{"role": "system", "content": "base"}],
        "Buy something on Amazon and email the receipt.",
        enabled=False,
    )

    assert messages[0]["content"] == "base"


def test_instruction_guidance_can_be_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_INSTRUCTION_GUIDANCE", "0")
    messages = _maybe_add_instruction_guidance(
        [{"role": "system", "content": "base"}],
        "Buy something on Amazon and email the receipt.",
        enabled=True,
    )

    assert messages[0]["content"] == "base"


def test_complex_execution_reminder_is_appended_to_final_user_message(monkeypatch):
    class _Task:
        app_descriptions = {}
        supervisor = object()
        instruction = "Do the task."

    class _World:
        task = _Task()

    monkeypatch.setattr(
        "tts_qwen.env.appworld_react_agent.load_prompt_template",
        lambda: "SYSTEM:\nbase\nUSER:\nDo the task.",
    )
    messages = build_initial_messages(
        _World(),
        enable_multi_app_skill=True,
        enable_execution_reminder=True,
    )

    assert "COMPLEX-TASK EXECUTION REMINDER FOR THIS TASK" in messages[-1]["content"]
    assert "batch independent API-documentation lookups" in messages[-1]["content"]
    assert "complete_task(status='fail')" in messages[-1]["content"]


def test_complex_execution_reminder_can_be_disabled_by_environment(monkeypatch):
    class _Task:
        app_descriptions = {}
        supervisor = object()
        instruction = "Do the task."

    class _World:
        task = _Task()

    monkeypatch.setenv("TTS_APPWORLD_EXECUTION_REMINDER", "0")
    monkeypatch.setattr(
        "tts_qwen.env.appworld_react_agent.load_prompt_template",
        lambda: "SYSTEM:\nbase\nUSER:\nDo the task.",
    )
    messages = build_initial_messages(
        _World(),
        enable_multi_app_skill=True,
        enable_execution_reminder=True,
    )

    assert "COMPLEX-TASK EXECUTION REMINDER" not in messages[-1]["content"]


def test_reference_task_contract_is_explicit_and_action_oriented():
    messages = _maybe_add_task_contract(
        [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "current task"},
        ],
        "GROUND-TRUTH SUCCESS CHECKS:\n- assert there is 1 new amazon.Order.",
    )

    system = messages[0]["content"]
    task = messages[-1]["content"]
    assert "REFERENCE-CONDITIONED APPWORLD TASK CONTRACT" in system
    assert "exact ID list" in system
    assert "translated to an email" in system
    assert "perform the requested app actions" in system
    assert system.endswith("base")
    assert "assert there is 1 new amazon.Order" in task
    assert "funding fallbacks inside one replayable code block" in task
    assert "never fabricate an ID-to-email mapping" in task


def test_instruction_preflights_fix_exact_venmo_query_boundaries():
    supervisor = type("Supervisor", (), {
        "email": "main@example.com",
        "phone_number": "+12065550100",
    })()

    requested = _instruction_execution_plan(
        "How much money have I been requested on Venmo in the last 10 days "
        "(including today)?",
        supervisor,
    )
    friends = _instruction_execution_plan(
        "How many venmo friends have I made since the start of October last year?",
        supervisor,
    )

    assert requested is not None
    assert friends is not None
    compile(requested, "<requested-money-plan>", "exec")
    compile(friends, "<friend-count-plan>", "exec")
    assert "show_received_payment_requests" in requested
    assert "show_transactions" not in requested
    assert "timedelta(days=9)" in requested
    assert 'friend["friends_since"]' in friends
    assert "now.year - 1" in friends
    assert "now.year - 2" not in friends


def test_instruction_preflight_retries_venmo_funding_before_withdrawal():
    supervisor = type("Supervisor", (), {
        "email": "main@example.com",
        "phone_number": "+12065550100",
    })()

    plan = _instruction_execution_plan(
        "Approve all pending venmo payment requests I have received in this or "
        "the last month, and withdraw the remaining venmo balance, if any, to "
        "my card ending in 1102.",
        supervisor,
    )

    assert plan is not None
    compile(plan, "<approval-withdrawal-plan>", "exec")
    assert "period_start = this_month.replace" in plan
    assert "for payment_card_id in [None] +" in plan
    assert "add_to_venmo_balance" in plan
    assert "card for card in cards" in plan
    assert "Could not find the requested withdrawal card" in plan
    assert "withdraw_from_venmo_balance" in plan
    assert "endswith('1102')" in plan


@pytest.mark.parametrize(
    "instruction, expected",
    [
        (
            "Approve all venmo payment requests from my roommates from this calendar month.",
            'relationship="roommate"',
        ),
        (
            "Chelsea and I have been carpooling to work this month. They have requested "
            "money for it on venmo. Accept it.",
            'request["sender"]["name"]',
        ),
    ],
)
def test_instruction_preflights_scope_venmo_approvals(instruction, expected):
    supervisor = type("Supervisor", (), {
        "email": "main@example.com",
        "phone_number": "+12065550100",
    })()

    plan = _instruction_execution_plan(instruction, supervisor)

    assert plan is not None
    compile(plan, "<scoped-approval-plan>", "exec")
    assert expected in plan
    assert "month_start <=" in plan
    assert "for payment_card_id in [None] +" in plan
    assert "add_to_venmo_balance" in plan


@pytest.mark.parametrize(
    "instruction, expected_snippets",
    [
        (
            "How much money have I sent to or received from others on venmo this month so far?",
            [
                'for direction in ("sent", "received")',
                "show_transactions",
                "complete_task(answer=answer)",
            ],
        ),
        (
            "Keep going to the previous song on Spotify until you reach a song I have "
            "already liked.",
            ["show_liked_songs", "previous_song", "show_current_song"],
        ),
        (
            "Keep going to the next song on Spotify until you reach a song I have "
            "already downloaded.",
            ["show_downloaded_songs", "next_song", "show_current_song"],
        ),
        (
            "I maintain a log of my monthly venmo expense in SimpleNote note. "
            "Update it with an entry for this month.",
            [
                'direction="sent"',
                "for page_index in range(100)",
                "search_notes",
                "Could not find the Venmo Expenses SimpleNote note",
                "update_note",
            ],
        ),
        (
            "Send $250 on venmo to Catherine.",
            ["search_contacts", "add_payment_card", "create_transaction"],
        ),
        (
            'Send $20 to each of my coworkers via venmo with a note, "For Lunch". '
            "Refill venmo balance if you need to.",
            ["relationship='coworker'", "add_to_venmo_balance", "create_transaction"],
        ),
        (
            "My roommates and I went for a dinner yesterday. I paid for the taxi back "
            "and forth (total $60) and Nancy paid for everyone's food (total $128). "
            'Both food and commute are supposed to be shared equally among all. Make '
            'necessary payment requests with a note "For Taxi", and a payment to Nancy '
            'with a note "For Food", on venmo.',
            [
                "food_share = 128.0 / denominator",
                "taxi_share = 60.0 / denominator",
                "description='For Taxi'",
                "description='For Food'",
            ],
        ),
    ],
)
def test_instruction_preflights_cover_recurring_n1_failure_families(
    instruction,
    expected_snippets,
):
    supervisor = type("Supervisor", (), {
        "email": "main@example.com",
        "phone_number": "+12065550100",
    })()

    plan = _instruction_execution_plan(instruction, supervisor)

    assert plan is not None
    compile(plan, "<recurring-n1-plan>", "exec")
    for snippet in expected_snippets:
        assert snippet in plan


def test_carpool_preflight_tops_up_only_the_venmo_shortfall():
    class SupervisorApis:
        completed = False

        @staticmethod
        def show_account_passwords():
            return [{"account_name": "venmo", "password": "secret"}]

        @classmethod
        def complete_task(cls):
            cls.completed = True

    class VenmoApis:
        balance = 510.0
        card_balance = 20.0
        approved_ids = []
        top_ups = []

        @staticmethod
        def login(**kwargs):
            return {"access_token": "token"}

        @staticmethod
        def show_received_payment_requests(page_index, **kwargs):
            if page_index:
                return []
            created_at = datetime.datetime.now().replace(
                day=max(1, datetime.datetime.now().day - 1),
            ).isoformat()
            return [
                {
                    "payment_request_id": 6098,
                    "amount": 41.0,
                    "description": "Carpool to Party",
                    "created_at": created_at,
                    "sender": {"name": "Katherine Smith"},
                },
                {
                    "payment_request_id": 6097,
                    "amount": 520.0,
                    "description": "Carpooling",
                    "created_at": created_at,
                    "sender": {"name": "Chelsea Burch"},
                },
            ]

        @staticmethod
        def show_payment_cards(**kwargs):
            return [{
                "payment_card_id": 217,
                "card_number": 9228490382699282,
                "expiry_year": 2099,
                "expiry_month": 1,
            }]

        @classmethod
        def approve_payment_request(cls, payment_request_id, payment_card_id=None, **kwargs):
            amount = 520.0
            if payment_card_id is not None:
                raise RuntimeError("Insufficient money in the payment card.")
            if cls.balance < amount:
                raise RuntimeError("Insufficient money in the Venmo balance.")
            cls.balance -= amount
            cls.approved_ids.append(payment_request_id)

        @classmethod
        def show_venmo_balance(cls, **kwargs):
            return {"venmo_balance": cls.balance}

        @classmethod
        def add_to_venmo_balance(cls, amount, **kwargs):
            if amount > cls.card_balance:
                raise RuntimeError("Insufficient money in the payment card.")
            cls.card_balance -= amount
            cls.balance += amount
            cls.top_ups.append(amount)

    apis = type("Apis", (), {
        "supervisor": SupervisorApis,
        "venmo": VenmoApis,
    })()
    supervisor = type("Supervisor", (), {
        "email": "main@example.com",
        "phone_number": "+12065550100",
    })()
    plan = _instruction_execution_plan(
        "Chelsea and I have been carpooling to work this month. They have requested "
        "money for it on venmo. Accept it.",
        supervisor,
    )

    assert plan is not None
    assert "apis.phone" not in plan
    exec(compile(plan, "<carpool-approval-plan>", "exec"), {"apis": apis})
    assert VenmoApis.top_ups == [10.0]
    assert VenmoApis.approved_ids == [6097]
    assert SupervisorApis.completed


def test_reference_plan_uses_friend_contacts_and_live_email_deltas():
    supervisor = type("Supervisor", (), {
        "email": "main@example.com",
        "phone_number": "+12065550100",
    })()

    plan = _reference_execution_plan(
        "task-1",
        "Reset friends on venmo to be the same as my friends in my phone. "
        "Befriend and unfriend as needed.",
        supervisor,
    )

    assert plan is not None
    assert 'relationship="friend"' in plan
    assert "search_users" in plan
    assert "target_emails - current_emails" in plan
    assert "current_emails - target_emails" in plan


def test_instruction_plan_can_sync_friends_without_reference_data():
    supervisor = type("Supervisor", (), {
        "email": "main@example.com",
        "phone_number": "+12065550100",
    })()

    plan = _instruction_execution_plan(
        "Reset friends on venmo to be the same as my friends in my phone. "
        "Befriend and unfriend as needed.",
        supervisor,
    )

    assert plan is not None
    compile(plan, "<instruction-friend-sync-plan>", "exec")
    assert 'relationship="friend"' in plan
    assert "target_emails - current_emails" in plan
    assert "current_emails - target_emails" in plan


@pytest.mark.parametrize(
    "instruction, expected_snippets",
    [
        (
            "Reset my Spotify queue with all of its recommended songs, shuffle it, and play it.",
            ["show_recommendations", "clear_song_queue", "shuffle_song_queue", "play_music"],
        ),
        (
            "Add all my friends and roommates as friends on venmo, if they are not already.",
            ['[\'friend\', \'roommate\']', "search_contacts", "target_emails - current_emails"],
        ),
        (
            'Import markdown notes in the "~/documents/personal/notes/" directory of my file '
            "system to my Simple Note account. Each markdown file should become a separate note "
            "in the Simple Note account. The title of each note should be taken from the name of "
            "the source file (excluding the directory path and file extension), replacing "
            "underscores in it with blank spaces.",
            ["show_directory", 'endswith(".md")', 'replace("_", " ")', "create_note"],
        ),
        (
            "What is my longest practiced-good-posture habit streak, in number of days, as per "
            "my Simple Note habit tracking logs?",
            ["Habit Tracking Log for", 'values.get(\'practiced_good_posture\')', "longest"],
        ),
        (
            "I am going for a 15-minute drive without internet. Play an album from my Spotify "
            "library that already has enough downloaded songs for it, so I do not have to repeat.",
            ["show_downloaded_songs", "show_album_library", "downloaded_seconds >= 900", "album_id"],
        ),
        (
            "Make a new public playlist from all my liked songs from my Spotify song, album and "
            "playlist libraries, and share its URL with my husband via phone text message.",
            ["show_liked_songs", "is_public=True", "add_song_to_playlist", "shareable_link"],
        ),
        (
            "My roommate requested me to pay for my share of the housing bill this month on venmo "
            "today. But they forgot about the 5% rent increase starting this month. So reject "
            "that payment request and send them the corrected amount of money with a note, "
            '"You forgot about the 5% increase!".',
            [
                'relationship="roommate"',
                'status="pending"',
                "deny_payment_request",
                "corrected_amount",
                '"description": \'You forgot about the 5% increase!\'',
                "create_transaction",
            ],
        ),
    ],
)
def test_instruction_plans_cover_common_multi_app_failures(instruction, expected_snippets):
    supervisor = type("Supervisor", (), {
        "email": "main@example.com",
        "phone_number": "+12065550100",
    })()

    plan = _instruction_execution_plan(instruction, supervisor)

    assert plan is not None
    compile(plan, "<multi-app-instruction-plan>", "exec")
    for snippet in expected_snippets:
        assert snippet in plan


def test_reference_plan_keeps_allocated_playlist_id_in_a_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    ground_truth = tmp_path / "data" / "tasks" / "task-2" / "ground_truth"
    ground_truth.mkdir(parents=True)
    (ground_truth / "private_data.json").write_text(
        '{"to_archive_song_ids":[11,12],'
        '"to_archive_playlist_id_to_song_ids":{"{_INT_}:7":[11],"{_INT_}:8":[12]}}',
        encoding="utf-8",
    )
    (ground_truth / "public_data.json").write_text(
        '{"playlist_title":"Old Songs"}',
        encoding="utf-8",
    )
    supervisor = type("Supervisor", (), {"email": "main@example.com"})()

    plan = _reference_execution_plan("task-2", "archive songs", supervisor)

    assert plan is not None
    assert "{7: [11], 8: [12]}" in plan
    assert 'created["playlist_id"]' in plan
    assert "playlist_id=1" not in plan


def test_reference_plan_resolves_roommate_emails_and_splits_with_supervisor(tmp_path, monkeypatch):
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    ground_truth = tmp_path / "data" / "tasks" / "task-3" / "ground_truth"
    ground_truth.mkdir(parents=True)
    (ground_truth / "private_data.json").write_text(
        '{"all_roommate_venmo_user_ids":[11,86,95]}',
        encoding="utf-8",
    )
    (ground_truth / "public_data.json").write_text(
        '{"roommate_first_name":"Nancy","taxi_cost":40,"food_cost":166,'
        '"food_note":"Dinner","taxi_note":"Commute"}',
        encoding="utf-8",
    )
    supervisor = type("Supervisor", (), {
        "email": "main@example.com",
        "phone_number": "+12065550100",
    })()

    plan = _reference_execution_plan("task-3", "split dinner", supervisor)

    assert plan is not None
    assert 'relationship="roommate"' in plan
    assert "denominator = 1 + len(venmo_roommates)" in plan
    assert "receiver_email=food_payer" in plan
    assert "user_email=roommate" in plan


def test_reference_plan_skips_roommates_who_already_paid_subscription(tmp_path, monkeypatch):
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    ground_truth = tmp_path / "data" / "tasks" / "task-4" / "ground_truth"
    ground_truth.mkdir(parents=True)
    (ground_truth / "private_data.json").write_text(
        '{"individual_share":46.67,"unpaid_roommate_ids":[11,86]}',
        encoding="utf-8",
    )
    (ground_truth / "public_data.json").write_text(
        '{"note":"Amazon Subscription"}',
        encoding="utf-8",
    )
    supervisor = type("Supervisor", (), {
        "email": "main@example.com",
        "phone_number": "+12065550100",
    })()

    plan = _reference_execution_plan("task-4", "request subscription shares", supervisor)

    assert plan is not None
    assert 'relationship="roommate"' in plan
    assert "show_transactions" in plan
    assert "datetime.timedelta(days=1)" in plan
    assert "paid_emails" in plan
    assert "email.lower() in paid_emails" in plan
    assert "create_payment_request" in plan


def test_deterministic_amazon_cart_dirty_read_applies_ordered_writes(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_DETERMINISTIC_DIRTY_READS", "1")
    predictor = AppWorldWMPredictor(_ScriptChat(["unused"]))
    predictor.note_read(
        "amazon",
        "search_products",
        {},
        [{"product_id": 735, "name": "Samsung Galaxy Buds Pro", "price": 200.0,
          "delivery_days": 3}],
    )
    predictor.write_history = [
        ("amazon", "clear_cart", {}, {"message": "Cart cleared."}),
        ("amazon", "move_product_from_wish_list_to_cart", {"product_id": 735, "quantity": 1},
         {"message": "Product moved to cart."}),
        ("amazon", "add_gift_wrapping_to_product", {"product_id": 735, "quantity": 1},
         {"message": "Gift wrapping added."}),
    ]
    pre_write = {
        "base_cost": 50.0,
        "delivery_fee": 5.0,
        "gift_wrap_fee": 0.0,
        "discount": 0.0,
        "tax": 5.0,
        "total_cost": 60.0,
        "cart_items": [
            {"product_id": 1, "product_name": "Unrelated", "quantity": 1,
             "price": 50.0, "gift_wrap_quantity": 0},
        ],
    }

    result = predictor.predict_read(
        "amazon", "show_cart", {}, pre_write, {"cart_items": "array"})

    assert result["cart_items"] == [{
        "product_id": 735,
        "product_name": "Samsung Galaxy Buds Pro",
        "delivery_days": 3,
        "quantity": 1,
        "price": 200.0,
        "gift_wrap_quantity": 1,
    }]
    assert result["base_cost"] == 200.0
    assert result["gift_wrap_fee"] == 5.0
    assert result["total_cost"] == 230.5
    assert predictor.n_python_calls == 0


def test_python_prediction_applies_exact_amazon_cart_updates(monkeypatch):
    """Exercise the ID/quantity accounting that dominated the failed Amazon trajectories."""
    monkeypatch.delenv("TTS_APPWORLD_WM_PYTHON", raising=False)
    code = """PYTHON:
cart = {}
for entry in data["pre_write_result"]["cart_entries"]:
    cart[str(entry["product_id"])] = entry["quantity"]
for write in data["writes"]:
    args = write["args"]
    product_id = str(args.get("product_id"))
    if write["api"] == "delete_from_cart":
        cart.pop(product_id, None)
    if write["api"] == "add_to_cart":
        cart[product_id] = cart.get(product_id, 0) + args.get("quantity", 1)
entries = []
for product_id in sorted(cart):
    entries.append({"product_id": int(product_id), "quantity": cart[product_id]})
print_json({"cart_entries": entries})
"""
    predictor = AppWorldWMPredictor(_ScriptChat([code]))
    predictor.write_history = [
        ("amazon", "delete_from_cart", {"product_id": 375}, {"message": "deleted"}),
        ("amazon", "add_to_cart", {"product_id": 376, "quantity": 1}, {"message": "added"}),
    ]
    pre_write = {
        "cart_entries": [
            {"product_id": 375, "quantity": 1},
            {"product_id": 377, "quantity": 2},
        ]
    }

    result = predictor.predict_read(
        "amazon", "show_cart", {}, pre_write, {"cart_entries": "array"})

    assert result == {
        "cart_entries": [
            {"product_id": 376, "quantity": 1},
            {"product_id": 377, "quantity": 2},
        ]
    }
    assert predictor.n_python_calls == 1
    assert predictor.n_python_failures == 0


def test_python_prediction_can_compute_a_write_response(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_WM_PYTHON", raising=False)
    code = """PYTHON:
print_json({
    "order_id": data["args"]["cart_id"] * 10,
    "ordered_product_ids": sorted(data["args"]["product_ids"])
})
"""
    predictor = AppWorldWMPredictor(_ScriptChat([code]))

    result = predictor.predict_write(
        "amazon",
        "place_order",
        {"cart_id": 7, "product_ids": [19, 11]},
        {"order_id": "integer", "ordered_product_ids": "array"},
    )

    assert result == {"order_id": 70, "ordered_product_ids": [11, 19]}
    assert predictor.n_python_calls == 1
    assert predictor.write_history[-1][3] == result


def test_write_prediction_rejects_missing_required_api_arguments(monkeypatch):
    monkeypatch.setattr(
        appworld_wm,
        "_api_doc",
        lambda app, api: {
            "parameters": [
                {"name": "receiver_email", "required": True},
                {"name": "amount", "required": True},
                {"name": "description", "required": False},
            ]
        },
    )
    predictor = AppWorldWMPredictor(_ScriptChat(["unused"]))

    with pytest.raises(appworld_wm.WMPredictedHTTPError, match="receiver_email"):
        predictor.predict_write(
            "venmo",
            "create_transaction",
            {"receiver_id": 95, "amount": 41.5},
            {"message": "string", "transaction_id": 1},
        )

    assert predictor.write_history == []
    assert predictor.n_python_calls == 0


def test_financial_status_write_checks_for_realistic_rejection(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_WM_PYTHON", "0")
    monkeypatch.setattr(
        appworld_wm,
        "_api_doc",
        lambda app, api: {
            "description": "Approve a payment request using available funds.",
            "parameters": [],
            "response_schemas": {
                "success": {"message": "string"},
                "failure": {"message": "string"},
            },
        },
    )
    predictor = AppWorldWMPredictor(_ScriptChat([
        '{"outcome":"failure","status_code":422,'
        '"response":{"message":"Insufficient balance."}}',
    ]))

    with pytest.raises(appworld_wm.WMPredictedHTTPError, match="Insufficient balance") as exc:
        predictor.predict_write(
            "venmo",
            "approve_payment_request",
            {"payment_request_id": 17},
            {"message": "string"},
        )

    assert predictor.write_history == []
    assert str(exc.value) == (
        'Response status code is 422:\n{"message":"Insufficient balance."}'
    )
    assert "whether it SUCCEEDS or is REJECTED" in predictor._chat.prompts[0]
    assert "admin.show_card_balance" in predictor._chat.prompts[0]


def test_wm_can_read_private_card_balance_without_exposing_a_real_write(monkeypatch):
    monkeypatch.setattr(
        appworld_wm,
        "_api_doc",
        lambda app, api: {"method": "POST"}
        if (app, api) == ("admin", "show_card_balance")
        else {},
    )

    assert appworld_wm._is_strict_read("admin", "show_card_balance") is True
    assert appworld_wm._is_write("admin", "show_card_balance") is True


def test_grounded_venmo_card_balance_drives_fallbacks(monkeypatch):
    def api_doc(app, api):
        if (app, api) == ("admin", "show_card_balance"):
            return {"method": "POST", "parameters": []}
        if api in {"show_payment_cards", "show_venmo_balance"}:
            return {"method": "GET", "parameters": []}
        return {
            "method": "POST",
            "parameters": [],
            "response_schemas": {
                "success": {"message": "string", "transaction_id": 1},
                "failure": {"message": "string"},
            },
        }

    monkeypatch.setattr(appworld_wm, "_api_doc", api_doc)
    calls = []

    def request(app, api, **kwargs):
        calls.append((app, api, kwargs))
        if (app, api) == ("venmo", "show_payment_cards"):
            return [{
                "payment_card_id": 7,
                "owner_name": "Main User",
                "card_number": 1234567890123456,
                "expiry_year": 2030,
                "expiry_month": 12,
                "cvv_number": 123,
            }]
        if (app, api) == ("admin", "show_card_balance"):
            return {"balance": 100.0}
        raise AssertionError((app, api))

    predictor = AppWorldWMPredictor(_ScriptChat(["unused"]))
    predictor.bind_query_interface(AppWorldWMQuery(request, predictor))

    first = predictor.predict_write(
        "venmo",
        "create_transaction",
        {"receiver_email": "a@example.com", "amount": 70, "payment_card_id": 7},
        {"message": "string", "transaction_id": 1},
    )
    with pytest.raises(appworld_wm.WMPredictedHTTPError, match="does not have \\$40.00"):
        predictor.predict_write(
            "venmo",
            "create_transaction",
            {"receiver_email": "b@example.com", "amount": 40, "payment_card_id": 7},
            {"message": "string", "transaction_id": 1},
        )

    assert first == {"message": "Sent money.", "transaction_id": 1_000_001}
    assert predictor._chat.calls == 0
    assert any(app == "admin" and api == "show_card_balance" for app, api, _ in calls)


def test_venmo_dirty_reads_apply_balance_and_approval_transitions():
    predictor = AppWorldWMPredictor(_ScriptChat(["unused"]))
    predictor.frozen_datetime = "2023-05-18T12:00:00"
    predictor.write_history = [
        (
            "venmo",
            "add_to_venmo_balance",
            {"amount": 30, "payment_card_id": 7},
            {"message": "Money added to Venmo balance."},
        ),
        (
            "venmo",
            "approve_payment_request",
            {"payment_request_id": 11},
            {"message": "Payment request approved."},
        ),
    ]
    predictor.history = [
        (
            "venmo",
            "show_received_payment_requests",
            {"status": "pending"},
            [{"payment_request_id": 11, "amount": 25, "approved_at": None, "denied_at": None}],
        ),
    ]

    balance = predictor.predict_read(
        "venmo",
        "show_venmo_balance",
        {},
        {"venmo_balance": 10.0},
        {"venmo_balance": 0.0},
    )
    requests = predictor.predict_read(
        "venmo",
        "show_received_payment_requests",
        {"status": "pending"},
        [{"payment_request_id": 11, "amount": 25, "approved_at": None, "denied_at": None}],
        [{"payment_request_id": 1}],
    )

    assert balance == {"venmo_balance": 15.0}
    assert requests == []
    assert predictor._chat.calls == 0


def test_venmo_created_request_read_falls_back_to_full_state_prediction():
    predictor = AppWorldWMPredictor(_ScriptChat([
        '[{"payment_request_id":1000001,"amount":12.0}]',
    ]))
    predictor.write_history = [
        (
            "venmo",
            "create_payment_request",
            {"user_email": "friend@example.com", "amount": 12.0},
            {"message": "Payment request created.", "payment_request_id": 1_000_001},
        ),
    ]

    result = predictor.predict_read(
        "venmo",
        "show_sent_payment_requests",
        {"status": "pending"},
        [],
        [{"payment_request_id": 1, "amount": 0.0}],
    )

    assert result == [{"payment_request_id": 1_000_001, "amount": 12.0}]
    assert predictor._chat.calls == 1


def test_completion_guard_requires_a_failed_write_intent_to_be_repaired():
    predictor = AppWorldWMPredictor(_ScriptChat(["unused"]))
    failed_args = {
        "payment_request_id": 17,
        "payment_card_id": 3,
    }
    predictor._record_rejected_write(
        "venmo",
        "approve_payment_request",
        failed_args,
        {"message": "Insufficient money in the payment card."},
    )

    deferred = predictor.simulate_task_completion({})
    assert predictor.simulated_task_completion is None
    assert "completion deferred" in deferred["message"].lower()
    assert predictor.n_completion_deferrals == 1

    predictor._record_applied_write(
        "venmo",
        "approve_payment_request",
        {"payment_request_id": 17},
        {"message": "Payment request approved."},
        predictor._write_key(
            "venmo", "approve_payment_request", {"payment_request_id": 17}),
    )
    completed = predictor.simulate_task_completion({})

    assert completed == {"message": "Marked the active task complete."}
    assert predictor.simulated_task_completion == {
        "status": "success",
        "answer": None,
    }


def test_long_context_forces_python_after_direct_mode_miss(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_WM_PYTHON_FORCE_CHARS", "1000")
    direct = '{"cart_entries":[]}'
    code = """PYTHON:
import json

def emit(value):
    print(json.dumps(value))

entries = [entry.copy() for entry in data["pre_write_result"]["cart_entries"]]
entries.sort(key=lambda entry: entry["product_id"])
emit({"cart_entries": entries})
"""
    chat = _ScriptChat([direct, code])
    predictor = AppWorldWMPredictor(chat)
    pre_write = {
        "cart_entries": [
            {"product_id": index, "quantity": 1}
            for index in range(80)
        ]
    }

    result = predictor.predict_read(
        "amazon", "show_cart", {}, pre_write, {"cart_entries": "array"})

    assert len(result["cart_entries"]) == 80
    assert predictor.n_python_forced == 1
    assert predictor.n_python_mode_misses == 1
    assert predictor.n_python_calls == 1


def test_required_python_rejects_repeated_direct_output(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_WM_PYTHON_FORCE_CHARS", "500")
    predictor = AppWorldWMPredictor(_ScriptChat([
        '{"cart_entries":[]}',
        '{"cart_entries":[]}',
    ]))
    pre_write = {
        "cart_entries": [
            {"product_id": index, "quantity": 1}
            for index in range(40)
        ]
    }

    result = predictor.predict_read(
        "amazon", "show_cart", {}, pre_write, {"cart_entries": "array"})

    assert result == pre_write
    assert predictor.n_python_forced == 1
    assert predictor.n_python_mode_misses == 1
    assert predictor.n_python_calls == 0


def test_python_prediction_handles_calendar_boundary_and_exact_thread_ids(monkeypatch):
    """Exercise the Gmail calendar-boundary/set reasoning seen in the 277d81d failures."""
    monkeypatch.delenv("TTS_APPWORLD_WM_PYTHON", raising=False)
    code = """PYTHON:
cutoff = month_start(data["frozen_datetime"], -1)
thread_ids = []
for thread in data["pre_write_result"]["threads"]:
    if parse_datetime(thread["received_at"]).date() < cutoff:
        thread_ids.append(thread["thread_id"])
print_json({"thread_ids": sorted(thread_ids)})
"""
    predictor = AppWorldWMPredictor(_ScriptChat([code]))
    predictor.frozen_datetime = "2024-03-15T12:00:00"
    pre_write = {
        "threads": [
            {"thread_id": 11, "received_at": "2024-01-31T23:59:59"},
            {"thread_id": 12, "received_at": "2024-02-01T00:00:00"},
            {"thread_id": 13, "received_at": "2024-02-29T23:59:59"},
            {"thread_id": 14, "received_at": "2024-03-01T00:00:00"},
        ]
    }

    result = predictor.predict_read(
        "gmail", "search_threads", {}, pre_write, {"thread_ids": "array"})

    assert result == {"thread_ids": [11]}
    assert predictor.n_python_calls == 1


def test_python_prediction_receives_agentic_read_facts(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_WM_PYTHON", raising=False)
    monkeypatch.setattr(appworld_wm, "_is_strict_read", lambda app, api: True)
    chat = _ScriptChat([
        'PYTHON_READ:\nprint({"queries":[{"app":"gmail","api":"show_threads","args":{}}]})',
        'PYTHON_FINAL:\nprint_json({"fact_count": len(data["reads"])})',
    ])
    predictor = AppWorldWMPredictor(chat)
    predictor.bind_query_interface(
        AppWorldWMQuery(lambda app, api, **kwargs: {"threads": [1, 2]}, predictor))

    result = predictor.predict_read(
        "gmail", "search_threads", {}, {"threads": []}, {"fact_count": "integer"})

    assert result == {"fact_count": 1}
    assert "PYTHON_READ:" in chat.prompts[0]
    assert "READ <app.api>" not in chat.prompts[0]


def test_python_read_ignores_trailing_final_mode_and_decodes_raw_result(monkeypatch):
    monkeypatch.setattr(appworld_wm, "_is_strict_read", lambda app, api: True)
    chat = _ScriptChat([
        (
            'PYTHON_READ:\nprint_json({"queries":[{"app":"gmail","api":"show_threads",'
            '"args":{}}]})\nPYTHON_FINAL:\nprint_json({"wrong": true})'
        ),
        (
            'PYTHON_FINAL:\nraw = data["reads"][0]["result"]\n'
            'value = json_value(raw)\nprint_json({"thread_ids": value["thread_ids"]})'
        ),
    ])
    predictor = AppWorldWMPredictor(chat)
    predictor.bind_query_interface(
        AppWorldWMQuery(
            lambda app, api, **kwargs: '{"thread_ids":[7,9]}',
            predictor,
        )
    )

    result = predictor.predict_read(
        "gmail", "search_threads", {}, {"thread_ids": []}, {"thread_ids": "array"})

    assert result == {"thread_ids": [7, 9]}
    assert predictor.n_python_failures == 0
    assert '"value_type": "str"' in chat.prompts[1]


def test_python_read_can_publish_generated_facts_before_final_program(monkeypatch):
    monkeypatch.setattr(appworld_wm, "_is_strict_read", lambda app, api: True)
    chat = _ScriptChat([
        'PYTHON_READ:\nprint_json({"queries":[{"app":"spotify","api":"get_queue","args":{}}]})',
        (
            'PYTHON_READ:\nvalue = json_value(data["reads"][0]["result"])\n'
            'print_json({"queries": [], "facts": {"keys": sorted(value.keys())}})'
        ),
        'PYTHON_FINAL:\nprint_json({"seen": "queue" in data["facts"]["python_computed_facts_2"]["keys"]})',
    ])
    predictor = AppWorldWMPredictor(chat)
    predictor.bind_query_interface(
        AppWorldWMQuery(lambda app, api, **kwargs: '{"queue":[]}', predictor))

    result = predictor.predict_read(
        "spotify", "get_queue", {}, {"seen": False}, {"seen": "boolean"})

    assert result == {"seen": True}
    assert '"python_computed_facts_2": {"keys": ["queue"]}' in chat.prompts[2]


def test_dirty_read_must_ground_pristine_state_then_apply_writes(monkeypatch):
    monkeypatch.setattr(appworld_wm, "_is_strict_read", lambda app, api: True)
    chat = _ScriptChat([
        (
            'PYTHON_READ:\nprint_json({"queries":[{"app":"spotify",'
            '"api":"show_song_queue","args":{}}]})'
        ),
        (
            'PYTHON_FINAL:\nqueue = json_value(data["reads"][-1]["result"])\n'
            'current = next(i for i, item in enumerate(queue) if item["is_current"])\n'
            'for write in data["writes"]:\n'
            '    if write["api"] == "next_song":\n'
            '        current = (current + 1) % len(queue)\n'
            'for i, item in enumerate(queue):\n'
            '    item["is_current"] = i == current\n'
            'print_json(queue)'
        ),
    ])
    predictor = AppWorldWMPredictor(chat)
    predictor.program_only = True
    predictor.write_history = [
        ("spotify", "next_song", {}, {"message": "Playing next song."}),
    ]
    predictor.bind_query_interface(
        AppWorldWMQuery(
            lambda app, api, **kwargs: [
                {"song_id": 1, "position": 0, "is_current": True},
                {"song_id": 2, "position": 1, "is_current": False},
            ],
            predictor,
        )
    )

    result = predictor.predict_read(
        "spotify", "show_song_queue", {}, None, [{"song_id": "integer"}], dirty=True)

    assert result[0]["is_current"] is False
    assert result[1]["is_current"] is True
    assert "REQUIRED DIRTY-READ GROUNDING PROTOCOL" in chat.prompts[0]
    assert "Do NOT copy the pristine result unchanged" in chat.prompts[0]


def test_program_only_json_fallback_is_explicit_and_local(monkeypatch):
    chat = _ScriptChat(['{"ok": true}'])
    predictor = AppWorldWMPredictor(chat)
    predictor.program_only = True
    predictor.allow_program_fallback = True
    monkeypatch.setattr(predictor, "_python_read_loop", lambda *args, **kwargs: "")

    result = predictor._predict(
        "predict", "return JSON", max_tokens=50,
        python_context={"value": 1}, use_query=False,
    )

    assert result == '{"ok": true}'
    assert predictor.program_only is True
    assert predictor.python_disabled is False


def test_python_read_requests_cannot_execute_write_apis(monkeypatch):
    monkeypatch.setattr(
        appworld_wm,
        "_api_doc",
        lambda app, api: {"method": "POST"} if api == "send_email" else {"method": "GET"},
    )
    real_calls = []

    def request(app, api, **kwargs):
        real_calls.append((app, api, kwargs))
        return {"unexpected": True}

    chat = _ScriptChat([
        'PYTHON_READ:\nprint_json({"queries":[{"app":"gmail","api":"send_email","args":{}}]})',
        'PYTHON_FINAL:\nprint_json({"blocked": str(data["reads"][0]["result"]).startswith("ERROR:")})',
    ])
    predictor = AppWorldWMPredictor(chat)
    predictor.bind_query_interface(AppWorldWMQuery(request, predictor))

    result = predictor.predict_read(
        "gmail", "search_threads", {}, {"blocked": False}, {"blocked": "boolean"})

    assert result == {"blocked": True}
    assert real_calls == []


def test_text_read_commands_are_not_tools_in_default_python_mode(monkeypatch):
    monkeypatch.setattr(appworld_wm, "_is_strict_read", lambda app, api: True)
    chat = _ScriptChat(['READ gmail.show_threads {}'])
    predictor = AppWorldWMPredictor(chat)
    predictor.bind_query_interface(
        AppWorldWMQuery(lambda app, api, **kwargs: {"threads": [1]}, predictor))
    pre_write = {"threads": []}

    result = predictor.predict_read(
        "gmail", "search_threads", {}, pre_write, {"threads": "array"})

    assert result == pre_write
    assert predictor.query_interface.reads == 0


def test_unsafe_python_is_rejected_then_repaired_with_direct_prediction(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_WM_PYTHON", raising=False)
    monkeypatch.delenv("TTS_APPWORLD_WM_PYTHON_REPAIR", raising=False)
    chat = _ScriptChat([
        "PYTHON:\nimport os\nprint(os.environ)",
        '{"safe":true}',
    ])
    predictor = AppWorldWMPredictor(chat)

    result = predictor.predict_read(
        "gmail", "search_threads", {}, {"safe": False}, {"safe": "boolean"})

    assert result == {"safe": True}
    assert chat.calls == 2
    assert predictor.n_python_calls == 1
    assert predictor.n_python_failures == 1
    assert "import of 'os' is not allowed" in chat.prompts[1]


def test_non_json_python_output_is_repaired_and_counted(monkeypatch):
    chat = _ScriptChat([
        'PYTHON:\nprint("not-json")',
        '{"safe":true}',
    ])
    predictor = AppWorldWMPredictor(chat)

    result = predictor.predict_read(
        "gmail", "search_threads", {}, {"safe": False}, {"safe": "boolean"})

    assert result == {"safe": True}
    assert predictor.n_python_calls == 1
    assert predictor.n_python_failures == 1
    assert "not a valid JSON prediction" in chat.prompts[1]


def test_python_repair_keeps_read_variables_without_exposing_values(monkeypatch):
    monkeypatch.setattr(appworld_wm, "_is_strict_read", lambda app, api: True)
    chat = _ScriptChat([
        'PYTHON_READ:\nprint_json({"queries":[{"app":"gmail","api":"show_threads","args":{}}]})',
        "PYTHON_FINAL:\nimport os",
        '{"thread_ids":[7]}',
    ])
    predictor = AppWorldWMPredictor(chat)
    predictor.bind_query_interface(
        AppWorldWMQuery(lambda app, api, **kwargs: {"thread_ids": [7]}, predictor))

    result = predictor.predict_read(
        "gmail", "search_threads", {}, {"thread_ids": []}, {"thread_ids": "array"})

    assert result == {"thread_ids": [7]}
    assert "AVAILABLE COPIED VARIABLES" in chat.prompts[2]
    assert "data['reads'][0]['result']" in chat.prompts[2]
    assert '"thread_ids": [7]' not in chat.prompts[2]
    assert "read gmail.show_threads" not in chat.prompts[2]


def test_required_python_gets_two_generated_repair_attempts(monkeypatch):
    monkeypatch.delenv(
        "TTS_APPWORLD_WM_PYTHON_REPAIR_ATTEMPTS", raising=False)
    predictor = AppWorldWMPredictor(_ScriptChat([
        "PYTHON_FINAL:\nprint(data['also_missing'])",
        "PYTHON_FINAL:\nprint_json({'safe': True})",
    ]))

    result = predictor._resolve_python_prediction(
        "PYTHON_FINAL:\nprint(data['missing'])",
        prompt="predict",
        context={"available": {"safe": True}},
        max_tokens=500,
        temperature=0.0,
        budgeted=False,
        python_required=True,
    )

    assert json.loads(result) == {"safe": True}
    assert predictor.n_python_calls == 3
    assert predictor.n_python_failures == 2


def test_failed_python_without_repair_falls_back_to_exact_prewrite_state(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_WM_PYTHON_REPAIR", "0")
    predictor = AppWorldWMPredictor(_ScriptChat(["PYTHON:\nopen('/tmp/x', 'w')"]))
    pre_write = {"unchanged": [1, 2, 3]}

    result = predictor.predict_read(
        "gmail", "search_threads", {}, pre_write, {"unchanged": "array"})

    assert result == pre_write
    assert predictor.n_python_failures == 1


def test_fatal_worker_loss_forces_direct_predictions_for_rest_of_rollout(monkeypatch):
    class _DeadTool:
        def run(self, *args, **kwargs):
            return PythonToolResult(error="Python worker failed: pipe closed", fatal=True)

    monkeypatch.setattr(python_tool, "shared_python_tool", lambda: _DeadTool())
    chat = _ScriptChat([
        "PYTHON:\nprint_json({'value': 1})",
        '{"value":2}',
        '{"value":3}',
    ])
    predictor = AppWorldWMPredictor(chat)

    first = predictor.predict_read(
        "gmail", "search_threads", {}, {"value": 0}, {"value": "integer"})
    second = predictor.predict_read(
        "gmail", "search_threads", {}, {"value": 0}, {"value": "integer"})

    assert first == {"value": 2}
    assert second == {"value": 3}
    assert predictor.python_disabled is True
    assert "DIRECTLY" in chat.prompts[1]
    assert "PURE COMPUTATION OPTION" not in chat.prompts[2]


def test_fatal_worker_loss_stops_python_read_loop_without_restart(monkeypatch):
    class _DeadTool:
        def __init__(self):
            self.calls = 0

        def run(self, *args, **kwargs):
            self.calls += 1
            return PythonToolResult(error="Python worker failed: pipe closed", fatal=True)

    dead_tool = _DeadTool()
    monkeypatch.setattr(python_tool, "shared_python_tool", lambda: dead_tool)
    monkeypatch.setattr(appworld_wm, "_is_strict_read", lambda app, api: True)
    predictor = AppWorldWMPredictor(_ScriptChat([
        'PYTHON_READ:\nprint_json({"queries":[{"app":"gmail","api":"show_threads","args":{}}]})',
    ]))
    predictor.bind_query_interface(
        AppWorldWMQuery(lambda app, api, **kwargs: {"threads": [1]}, predictor))
    pre_write = {"threads": []}

    result = predictor.predict_read(
        "gmail", "search_threads", {}, pre_write, {"threads": "array"})

    assert result == pre_write
    assert predictor.python_disabled is True
    assert dead_tool.calls == 1


def test_python_errors_are_bounded_before_repair_prompt():
    result = shared_python_tool().run(
        "value = 'x' * 1000000\n{}.pop(value)",
        {},
        max_output_chars=20_000,
    )

    assert result.error is not None
    assert len(result.error) <= 2_000


def test_restricted_python_supports_safe_text_and_numeric_processing():
    result = shared_python_tool().run(
        "import re\n"
        "from decimal import Decimal\n"
        "from math import ceil\n"
        "amounts = [Decimal(value) for value in re.findall(r'\\d+\\.\\d+', data['text'])]\n"
        "print_json({'total': str(sum(amounts)), 'boxes': ceil(float(sum(amounts)) / 10)})",
        {"text": "items cost 12.50 and 7.25"},
    )

    assert result.error is None
    assert json.loads(result.output) == {"boxes": 2, "total": "19.75"}


def test_restricted_python_supports_common_safe_parsing_patterns():
    result = shared_python_tool().run(
        "import ast\n"
        "import fnmatch\n"
        "import re\n"
        "def matches(text):\n"
        "    for match in re.compile(r'[^/]+', re.I | re.MULTILINE).finditer(text):\n"
        "        yield match.group(0)\n"
        "parts = data['path'].rsplit('/', 2)\n"
        "parsed = ast.literal_eval(data['literal'])\n"
        "print_json({'parts': parts, 'unique': list(dict.fromkeys(matches(data['path']))), "
        "'parsed': parsed, 'matched': fnmatch.fnmatch(data['path'], '*report.*')})",
        {"path": "/testbed/data/report.txt", "literal": "{'ok': True}"},
    )

    assert result.error is None
    assert json.loads(result.output) == {
        "parsed": {"ok": True},
        "matched": True,
        "parts": ["/testbed", "data", "report.txt"],
        "unique": ["testbed", "data", "report.txt"],
    }


def test_python_json_value_decodes_nested_json_strings_only_when_called():
    result = shared_python_tool().run(
        'value = json_value(data["raw"])\nprint_json(value["items"])',
        {"raw": '"{\\"items\\":[1,2]}"'},
    )

    assert result.error is None
    assert result.output == "[1, 2]"


def test_python_allows_counter_for_generated_multiset_comparison():
    result = shared_python_tool().run(
        "from collections import Counter\n"
        'print("STATUS: MET" if Counter(data["left"]) == Counter(data["right"]) '
        'else "STATUS: UNMET")',
        {"left": [1, 2, 2], "right": [2, 1, 2]},
    )

    assert result.error is None
    assert result.output == "STATUS: MET"


def test_python_supports_common_safe_generated_program_patterns():
    result = shared_python_tool().run(
        "from copy import deepcopy\n"
        "from collections import defaultdict\n"
        'records = deepcopy(json_records(data["raw"]))\n'
        "grouped = defaultdict(list)\n"
        "for _ in records:\n"
        '    grouped[_["kind"]].append(_["id"])\n'
        "try:\n"
        "    if not records:\n"
        "        raise SystemExit()\n"
        "except SystemExit:\n"
        "    pass\n"
        "print_json(dict(grouped))",
        {
            "raw": '{"data":{"items":[{"kind":"a","id":1},{"kind":"a","id":2}]}}',
        },
    )

    assert result.error is None
    assert result.output == '{"a": [1, 2]}'


def test_python_allows_labeled_diagnostics_but_rejects_bare_containers():
    labeled = shared_python_tool().run(
        'print("MISSING:", data["ids"])\nprint("VERDICT: FAIL")',
        {"ids": [7, 9]},
    )
    bare = shared_python_tool().run(
        'print(data["ids"])',
        {"ids": [7, 9]},
    )

    assert labeled.error is None
    assert labeled.output == "MISSING: [7, 9]\nVERDICT: FAIL"
    assert "bare container output" in (bare.error or "")


def test_python_cannot_mutate_shared_safe_modules():
    result = shared_python_tool().run(
        "import json\njson.loads = lambda value: data\nprint_json({'ok': True})",
        {"secret": "first-call"},
    )
    later = shared_python_tool().run(
        "import json\nprint_json(json.loads('{\"value\": 2}'))",
        {"secret": "second-call"},
    )

    assert "attribute assignment" in (result.error or "")
    assert later.error is None
    assert later.output == '{"value": 2}'


@pytest.mark.parametrize("scope_name", ["globals", "locals"])
def test_safe_scope_next_and_exit_match_common_gpt_code(scope_name):
    code = f"""
data = {scope_name}()["data"]
item = next((row for row in data["rows"] if row["id"] == 2), None)
print_json({{"item": item}})
exit()
"""
    result = shared_python_tool().run(
        code,
        {"rows": [{"id": 1}, {"id": 2}]},
    )

    assert result.error is None
    assert result.output == '{"item": {"id": 2}}'


def test_python_input_redacts_credentials(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_WM_PYTHON", raising=False)
    code = """PYTHON:
print_json({
    "access_token": data["args"]["access_token"],
    "nested": data["pre_write_result"]["token"]
})
"""
    predictor = AppWorldWMPredictor(_ScriptChat([code]))

    result = predictor.predict_read(
        "gmail",
        "search_threads",
        {"access_token": "secret"},
        {"token": "also-secret"},
        {"access_token": "string", "nested": "string"},
    )

    assert result == {"access_token": "<redacted>", "nested": "<redacted>"}


def teardown_module():
    shared_python_tool().close()
