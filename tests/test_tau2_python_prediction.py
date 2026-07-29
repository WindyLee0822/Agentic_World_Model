"""Restricted Python-only world-model tools for tau2."""

from __future__ import annotations

import json

from tts.env.real_tau2 import _LLMWorldModelPredictor
from tts.env.tau2_wm_python import tau2_wm_tool_mode
from tts.types import Action


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


class _Query:
    def __init__(self):
        self.read_tools = ["search_kb"]
        self.queries = []
        self.lists = []
        self.calls = []

    def query(self, refs, label="q"):
        self.queries.append((list(refs), label))
        return {
            ref: {"model": {"id": ref[1], "name": "Ada"}}
            for ref in refs
        }

    def list_ids(self, entity_type):
        self.lists.append(entity_type)
        return ["one", "two"]

    def call(self, name, args):
        self.calls.append((name, dict(args)))
        return '{"matches":[1,2]}'


def _predictor(responses):
    predictor = _LLMWorldModelPredictor("airline")
    predictor._client = _ScriptChat(responses)
    return predictor


def test_tau2_wm_tool_mode_defaults_to_python(monkeypatch):
    monkeypatch.delenv("TTS_TAU2_WM_TOOL_MODE", raising=False)
    assert tau2_wm_tool_mode() == "python"
    monkeypatch.setenv("TTS_TAU2_WM_TOOL_MODE", "unknown")
    assert tau2_wm_tool_mode() == "python"
    monkeypatch.setenv("TTS_TAU2_WM_TOOL_MODE", "text")
    assert tau2_wm_tool_mode() == "text"


def test_wm_agent_policy_reminder_is_disabled_and_guidance_is_verifier_only():
    from types import SimpleNamespace

    from tts.env.real_tau2 import (
        _AIRLINE_WM_VERIFIER_GUIDANCE,
        _WM_VERIFIER_POLICY_GUIDANCE,
        _add_wm_agent_policy_reminder,
        _wm_verifier_policy_guidance,
    )

    agent = SimpleNamespace(
        get_init_state=lambda: SimpleNamespace(system_messages=[]),
    )
    original = agent.get_init_state
    _add_wm_agent_policy_reminder(agent)
    _add_wm_agent_policy_reminder(agent)
    state = agent.get_init_state()

    assert agent.get_init_state is original
    assert state.system_messages == []
    assert _wm_verifier_policy_guidance("retail") == _WM_VERIFIER_POLICY_GUIDANCE
    assert _wm_verifier_policy_guidance("airline") == (
        f"{_WM_VERIFIER_POLICY_GUIDANCE}\n{_AIRLINE_WM_VERIFIER_GUIDANCE}")
    assert "mechanical acceptance" in _wm_verifier_policy_guidance("airline")


def test_airline_cancel_preflight_regenerates_ineligible_call():
    from types import SimpleNamespace

    from tts.env.real_tau2 import _add_airline_cancel_preflight

    cancel = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[
            SimpleNamespace(
                name="cancel_reservation",
                arguments={"reservation_id": "r1"},
            ),
        ],
    )
    refusal = SimpleNamespace(
        role="assistant",
        content="This reservation is outside the cancellation window.",
        tool_calls=None,
    )
    reservation = SimpleNamespace(
        role="tool",
        content=json.dumps({
            "reservation_id": "r1",
            "created_at": "2024-05-13T10:00:00",
            "cabin": "economy",
            "insurance": "yes",
            "flights": [{"date": "2024-05-20"}],
        }),
    )

    class _Agent:
        def __init__(self):
            self.responses = [cancel, refusal]
            self.calls = 0

        def generate_next_message(self, message, state):
            state.messages.append(message)
            response = self.responses[self.calls]
            self.calls += 1
            state.messages.append(response)
            return response, state

    agent = _Agent()
    state = SimpleNamespace(system_messages=[], messages=[reservation])
    _add_airline_cancel_preflight(agent)

    response, state = agent.generate_next_message(
        SimpleNamespace(role="user", content="Cancel it."), state)

    assert response is refusal
    assert agent.calls == 2
    assert state.messages == [
        reservation,
        SimpleNamespace(role="user", content="Cancel it."),
        refusal,
    ]


def test_airline_baggage_preflight_corrects_nonfree_count():
    from types import SimpleNamespace

    from tts.env.real_tau2 import _add_airline_baggage_preflight

    wrong = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[
            SimpleNamespace(
                name="update_reservation_baggages",
                arguments={
                    "reservation_id": "r1",
                    "total_baggages": 3,
                    "nonfree_baggages": 2,
                    "payment_id": "gift1",
                },
            ),
        ],
    )
    user_record = SimpleNamespace(
        role="tool",
        content=json.dumps({
            "user_id": "u1",
            "membership": "gold",
        }),
    )
    reservation_record = SimpleNamespace(
        role="tool",
        content=json.dumps({
            "reservation_id": "r1",
            "user_id": "u1",
            "cabin": "economy",
            "passengers": [{"first_name": "Omar"}],
        }),
    )

    class _Agent:
        def __init__(self):
            self.calls = 0

        def generate_next_message(self, message, state):
            state.messages.append(message)
            self.calls += 1
            state.messages.append(wrong)
            return wrong, state

    agent = _Agent()
    state = SimpleNamespace(
        system_messages=[],
        messages=[user_record, reservation_record],
    )
    _add_airline_baggage_preflight(agent)

    response, _ = agent.generate_next_message(
        SimpleNamespace(role="user", content="Add three bags."), state)

    assert response is wrong
    assert response.tool_calls[0].arguments["nonfree_baggages"] == 0
    assert agent.calls == 1


def test_airline_certificate_preflight_requires_completed_delay_change():
    from types import SimpleNamespace

    from tts.env.real_tau2 import _add_airline_certificate_preflight

    wrong = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[
            SimpleNamespace(
                name="send_certificate",
                arguments={"user_id": "u1", "amount": 50},
            ),
        ],
    )
    corrected = SimpleNamespace(
        role="assistant",
        content="I can’t offer delayed-flight compensation without changing or cancelling it.",
        tool_calls=None,
    )

    class _Agent:
        def __init__(self):
            self.responses = [wrong, corrected]
            self.calls = 0

        def generate_next_message(self, message, state):
            state.messages.append(message)
            response = self.responses[self.calls]
            self.calls += 1
            state.messages.append(response)
            return response, state

    agent = _Agent()
    state = SimpleNamespace(
        system_messages=[],
        messages=[
            SimpleNamespace(
                role="user",
                content="The flight was delayed, but I do not want to change or cancel it.",
            ),
        ],
    )
    _add_airline_certificate_preflight(agent)

    response, _ = agent.generate_next_message(
        SimpleNamespace(role="user", content="Can I get compensation?"), state)

    assert response is corrected
    assert agent.calls == 2


def test_python_agentic_prediction_reads_records_lists_and_tools(monkeypatch):
    monkeypatch.delenv("TTS_TAU2_WM_TOOL_MODE", raising=False)
    query = _Query()
    predictor = _predictor([
        """PYTHON_READ:
print_json({"queries": [
    {"op": "read", "entity_type": "users", "entity_id": "u1"},
    {"op": "list", "entity_type": "orders"},
    {"op": "call", "name": "search_kb", "args": {"query": "policy"}}
]})
""",
        """PYTHON_FINAL:
print(json_dumps({
    "name": data["reads"][0]["result"]["name"],
    "ids": data["reads"][1]["result"],
    "matches": data["reads"][2]["result"]["matches"]
}))
""",
    ])

    text, is_error = predictor.predict_observation_agentic(
        "airline",
        Action("book_reservation", {"user_id": "u1"}),
        [],
        query,
        ["users", "orders"],
        tool_doc="Create and return a reservation.",
    )

    assert json.loads(text) == {
        "name": "Ada",
        "ids": ["one", "two"],
        "matches": [1, 2],
    }
    assert not is_error
    assert query.queries and query.lists == ["orders"]
    assert query.calls == [("search_kb", {"query": "policy"})]
    assert predictor.n_python_calls == 2
    assert "PYTHON_READ:" in predictor._client.prompts[0]
    assert "READ <type>" not in predictor._client.prompts[0]


def test_airline_flight_update_forces_grounded_python_calculation(monkeypatch):
    monkeypatch.delenv("TTS_TAU2_WM_TOOL_MODE", raising=False)
    monkeypatch.setenv("TTS_TAU2_WM_PYTHON_FORCE_CHARS", "10")
    predictor = _predictor([
        '{"reservation_id":"r1"}',
        'PYTHON_FINAL:\nprint_json({"reservation_id": "r1", "cabin": "economy"})',
    ])

    text, is_error = predictor.predict_observation_agentic(
        "airline",
        Action("update_reservation_flights", {
            "reservation_id": "r1",
            "cabin": "economy",
            "flights": [{"flight_number": "F1", "date": "2024-05-21"}],
            "payment_id": "card1",
        }),
        [],
        _Query(),
        ["reservations", "flights"],
        seed_facts={
            "flights F1": {
                "dates": {
                    "2024-05-21": {
                        "prices": {"economy": 140, "business": 274},
                    },
                },
            },
        },
    )

    assert json.loads(text) == {"reservation_id": "r1", "cabin": "economy"}
    assert not is_error
    assert predictor.n_python_forced == 1
    assert "signed difference" in predictor._client.prompts[0]


def test_default_python_mode_rejects_legacy_text_tools(monkeypatch):
    monkeypatch.delenv("TTS_TAU2_WM_TOOL_MODE", raising=False)
    query = _Query()
    predictor = _predictor([
        "READ users u1",
        'PYTHON_FINAL:\nprint("predicted")',
    ])

    text, _ = predictor.predict_observation_agentic(
        "airline", Action("write", {}), [], query, ["users"])

    assert text == "predicted"
    assert query.queries == []
    assert "READ/LIST/CALL/TRACE are disabled" in predictor._client.prompts[1]


def test_invalid_python_read_cannot_call_write_tool(monkeypatch):
    monkeypatch.delenv("TTS_TAU2_WM_TOOL_MODE", raising=False)
    query = _Query()
    predictor = _predictor([
        """PYTHON_READ:
print_json({"queries": [
    {"op": "call", "name": "cancel_reservation", "args": {"id": "r1"}}
]})
""",
        'PYTHON_FINAL:\nprint("safe fallback")',
    ])

    text, _ = predictor.predict_observation_agentic(
        "airline", Action("write", {}), [], query, ["users"])

    assert text == "safe fallback"
    assert query.calls == []
    assert "valid read-only queries" in predictor._client.prompts[1]


def test_python_verdict_can_compare_copied_actions(monkeypatch):
    monkeypatch.delenv("TTS_TAU2_WM_TOOL_MODE", raising=False)
    predictor = _predictor([
        """PYTHON_FINAL:
same = data["candidate_actions"] == data["gold_actions"]
print_json({"verdict": "SUCCESS" if same else "FAILURE"})
""",
    ])
    actions = [Action("cancel_reservation", {"reservation_id": "r1"})]

    assert predictor.predict_success_agentic(
        "airline", actions, actions, _Query(), ["reservations"])
    assert predictor.python_metadata()["python_calls"] == 1


def test_python_verifier_receives_conversation_and_communication_criteria(monkeypatch):
    monkeypatch.delenv("TTS_TAU2_WM_TOOL_MODE", raising=False)
    predictor = _predictor([
        """PYTHON_FINAL:
ok = (
    data["evaluation_context"]["communicate_info"] == ["1628"]
    and "1,628" in data["candidate_conversation"]
)
print_json({"verdict": "SUCCESS" if ok else "FAILURE"})
""",
    ])

    assert predictor.predict_success_agentic(
        "airline",
        [],
        [],
        _Query(),
        ["reservations"],
        task_text="Report the total.",
        evaluation_context={"communicate_info": ["1628"], "nl_assertions": []},
        candidate_conversation="assistant: The total is $1,628.",
    )
    assert "WORLD-MODEL VERIFICATION GUIDANCE" in predictor._client.prompts[0]
    assert "mechanical acceptance" in predictor._client.prompts[0]


def test_reference_gate_rejects_wrong_write_args_before_llm(monkeypatch):
    from tts.env import real_tau2
    from tts.trajectory import Step, Trajectory

    class _Model:
        buckets = []

    class _Predictor:
        @staticmethod
        def predict_success_agentic(*args, **kwargs):
            raise AssertionError("deterministic mismatch should skip the LLM verifier")

        @staticmethod
        def python_metadata():
            return {}

    monkeypatch.setattr(
        real_tau2, "tau2_mutating_tools", lambda domain: {"update"})
    evaluator = real_tau2.Tau2ShadowEvaluator(
        "airline",
        [Action("update", {"payment_id": "credit_card_1234"})],
        _Model(),
        predictor=_Predictor(),
    )
    trajectory = Trajectory(
        steps=[Step(Action("update", {"payment_id": "credit_card_9999"}))])

    reward = evaluator.reward_agentic(None, trajectory)

    assert not reward.success
    assert "mutating actions" in reward.reason
    assert reward.metadata["hard_invalid"]
    assert reward.reward == 0.25


def test_reference_gate_uses_official_unordered_compare_args_semantics(monkeypatch):
    from types import SimpleNamespace

    from tts.env import real_tau2
    from tts.trajectory import Step, Trajectory

    class _Model:
        buckets = []

    class _Predictor:
        calls = 0

        @classmethod
        def predict_success_agentic(cls, *args, **kwargs):
            cls.calls += 1
            return True

        @staticmethod
        def python_metadata():
            return {}

    monkeypatch.setattr(
        real_tau2, "tau2_mutating_tools", lambda domain: {"update_a", "update_b", "extra"})
    gold_specs = [
        SimpleNamespace(
            name="update_a",
            arguments={"id": "a", "unscored": "gold"},
            compare_args=["id"],
        ),
        SimpleNamespace(
            name="update_b",
            arguments={"id": "b"},
            compare_args=None,
        ),
    ]
    evaluator = real_tau2.Tau2ShadowEvaluator(
        "airline",
        [
            Action("update_a", {"id": "a", "unscored": "gold"}),
            Action("update_b", {"id": "b"}),
        ],
        _Model(),
        predictor=_Predictor(),
        gold_action_specs=gold_specs,
    )
    trajectory = Trajectory(steps=[
        Step(Action("update_b", {"id": "b"})),
        Step(Action("extra", {"id": "ignored"})),
        Step(Action("update_a", {"id": "a", "unscored": "candidate"})),
    ])

    reward = evaluator.reward_agentic(None, trajectory)

    assert reward.success
    assert reward.metadata["reference_score"] == 1.0
    assert _Predictor.calls == 1


def test_reference_gate_requires_communicated_values_before_llm(monkeypatch):
    from types import SimpleNamespace

    from tts.env import real_tau2
    from tts.trajectory import Trajectory

    class _Model:
        buckets = []

    class _Predictor:
        @staticmethod
        def predict_success_agentic(*args, **kwargs):
            raise AssertionError("missing communication should skip the LLM verifier")

        @staticmethod
        def python_metadata():
            return {}

    monkeypatch.setattr(
        real_tau2, "tau2_mutating_tools", lambda domain: set())
    evaluator = real_tau2.Tau2ShadowEvaluator(
        "airline",
        [],
        _Model(),
        predictor=_Predictor(),
        evaluation_context={"communicate_info": ["1628"]},
    )

    reward = evaluator.reward_agentic(None, Trajectory(
        final_message="The total is $1,701.",
        sim_messages=[
            SimpleNamespace(role="tool", content='{"unrelated_value": 1628}'),
        ],
    ))

    assert not reward.success
    assert "1628" in reward.reason
    assert reward.metadata["hard_invalid"]
    assert reward.reward == 0.75


def test_action_grounding_projects_only_requested_flight_dates(monkeypatch):
    from tts.env import real_tau2

    class _Model:
        buckets = ["flights", "reservations"]

        @staticmethod
        def matches(value):
            if value == "F1":
                return [("flights", "F1")]
            if value == "r1":
                return [("reservations", "r1")]
            return []

    class _Reader:
        @staticmethod
        def query(refs, label):
            return {
                ("flights", "F1"): {
                    "model": {
                        "flight_number": "F1",
                        "dates": {
                            "2024-05-21": {"prices": {"economy": 100}},
                            "2024-05-22": {"prices": {"economy": 200}},
                        },
                    },
                },
                ("reservations", "r1"): {
                    "model": {"reservation_id": "r1", "cabin": "business"},
                },
            }

    monkeypatch.setattr(
        real_tau2, "tau2_mutating_tools", lambda domain: {"update_reservation_flights"})
    evaluator = real_tau2.Tau2ShadowEvaluator("airline", [], _Model())
    facts = evaluator._seed_action_facts(
        _Reader(),
        Action("update_reservation_flights", {
            "reservation_id": "r1",
            "flights": [{"flight_number": "F1", "date": "2024-05-21"}],
        }),
    )

    assert list(facts["flights F1"]["dates"]) == ["2024-05-21"]
    assert facts["reservations r1"]["cabin"] == "business"


def test_text_mode_restores_legacy_read_loop(monkeypatch):
    monkeypatch.setenv("TTS_TAU2_WM_TOOL_MODE", "text")
    query = _Query()
    predictor = _predictor([
        "READ users u1",
        'PREDICT: {"id":"u1","name":"Ada"}',
    ])

    text, is_error = predictor.predict_observation_agentic(
        "airline", Action("write", {}), [], query, ["users"])

    assert json.loads(text) == {"id": "u1", "name": "Ada"}
    assert not is_error
    assert query.queries
    assert predictor.n_python_calls == 0
    assert "READ <type>" in predictor._client.prompts[0]


def test_python_final_preserves_json_objects_with_text_fields(monkeypatch):
    monkeypatch.delenv("TTS_TAU2_WM_TOOL_MODE", raising=False)
    predictor = _predictor([
        'PYTHON_FINAL:\nprint_json({"text": "legitimate record field", "id": 7})',
    ])

    text, is_error = predictor.predict_observation(
        "(no records)", [], Action("get_message", {}))

    assert json.loads(text) == {"text": "legitimate record field", "id": 7}
    assert not is_error


def test_long_flat_observation_allows_direct_semantic_result(monkeypatch):
    monkeypatch.delenv("TTS_TAU2_WM_TOOL_MODE", raising=False)
    monkeypatch.setenv("TTS_TAU2_WM_PYTHON_FORCE_CHARS", "500")
    predictor = _predictor([
        '{"items":[]}',
    ])

    text, is_error = predictor.predict_observation(
        "records=" + ("x" * 1000), [], Action("update", {}))

    assert json.loads(text) == {"items": []}
    assert not is_error
    assert predictor.n_python_forced == 0
    assert predictor.n_python_mode_misses == 0
    assert predictor.n_python_calls == 0


def test_failed_python_prediction_surfaces_tool_error(monkeypatch):
    monkeypatch.delenv("TTS_TAU2_WM_TOOL_MODE", raising=False)
    predictor = _predictor([
        'PYTHON_FINAL:\nprint(data["missing"])',
    ])

    text, is_error = predictor.predict_observation(
        "(no records)", [], Action("update", {}))

    assert text.startswith("ERROR:")
    assert is_error
    assert predictor.n_python_failures >= 2


def test_tau2_reward_carries_python_telemetry(monkeypatch):
    from tts.budget import Budget
    from tts.cache import ReadCache
    from tts.config import SimulatorConfig
    from tts.env import real_tau2
    from tts.trajectory import Trajectory
    from tts.worldmodel import make_world_model

    class _Store:
        @staticmethod
        def read(entity_type, entity_id):
            return None

        @staticmethod
        def list_ids(entity_type):
            return []

    class _Model:
        buckets = []

        @staticmethod
        def matches(value):
            return []

    class _Predictor:
        @staticmethod
        def predict_success(facts, candidate_actions, gold_actions, tool_docs=""):
            return True

        @staticmethod
        def python_metadata():
            return {
                "python_calls": 3,
                "python_failures": 1,
                "python_forced": 2,
                "python_mode_misses": 1,
            }

    monkeypatch.setattr(real_tau2, "tau2_mutating_tools", lambda domain: frozenset())
    evaluator = real_tau2.Tau2ShadowEvaluator(
        "airline", [], _Model(), predictor=_Predictor())
    world_model = make_world_model(
        _Store(),
        evaluator,
        SimulatorConfig(kind="flat"),
        Budget(),
        cache=ReadCache(),
    )

    reward = world_model.evaluate(Trajectory())

    assert reward.success
    assert reward.metadata == {
        "python_calls": 3,
        "python_failures": 1,
        "python_forced": 2,
        "python_mode_misses": 1,
    }
