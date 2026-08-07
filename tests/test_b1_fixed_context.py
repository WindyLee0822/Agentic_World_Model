from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

from tts.abcs import EntityStore, ShadowEvaluator
from tts.budget import Budget
from tts.cache import ReadCache
from tts.config import BudgetConfig, SimulatorConfig, load_config
from tts.env import appworld_wm
from tts.env.appworld_wm import AppWorldWMPredictor
from tts.env.officebench_agent import run_officebench_agent
from tts.env.officebench_wm import OfficeBenchVerifier
from tts.env.officebench_wm_predict import (
    OfficeBenchWMPredictor,
    build_officebench_flat_system_prompt,
)
from tts.env.real_tau2 import _LLMWorldModelPredictor, _should_predict_tau2_read
from tts.trajectory import RewardInfo, Step, Trajectory
from tts.types import Action
from tts.worldmodel import ReadGroundedWorldModel


def test_predict_all_is_the_global_and_method_config_default():
    root = Path(__file__).resolve().parents[1]

    assert SimulatorConfig().predict_all_actions
    assert load_config(str(root / "configs" / "base.yaml")).simulator.predict_all_actions
    assert load_config(str(root / "configs" / "b1.yaml")).simulator.predict_all_actions
    assert load_config(str(root / "configs" / "ours_fixed.yaml")).simulator.predict_all_actions
    assert load_config(str(root / "configs" / "ours_uncertainty.yaml")).simulator.predict_all_actions


def test_experiment_runner_threads_predict_all_into_simulator_config():
    from experiments.run import sim_config

    args = SimpleNamespace(
        predict_all_actions=True,
        recursion_mode="uncertainty",
        tau=0.9,
        L=2,
        hard_cap=6,
        branching=2,
        node_read_limit=1,
        no_cache=False,
        no_selective_invalidation=False,
    )

    assert sim_config(args).predict_all_actions


def test_tau2_predict_all_routes_clean_and_dirty_reads_to_world_model():
    assert not _should_predict_tau2_read(
        predict_all_actions=False, read_is_dirty=False)
    assert _should_predict_tau2_read(
        predict_all_actions=False, read_is_dirty=True)
    assert _should_predict_tau2_read(
        predict_all_actions=True, read_is_dirty=False)


class _RecordingStore(EntityStore):
    def __init__(self) -> None:
        self.reads = []

    def read(self, entity_type: str, entity_id: str) -> Optional[Dict]:
        self.reads.append((entity_type, entity_id))
        return {"id": entity_id}

    def list_ids(self, entity_type: str) -> List[str]:
        return ["reference", "candidate"]


class _RecordingEvaluator(ShadowEvaluator):
    def __init__(self) -> None:
        self.seed_calls = []
        self.prepared_shadow = None
        self.observed_shadow = None

    def seed_entities(self, candidate_actions: List[Action]) -> List[tuple]:
        self.seed_calls.append(list(candidate_actions))
        if not candidate_actions:
            return [("item", "reference")]
        return [("item", str(candidate_actions[0].args["item_id"]))]

    def child_references(self, entity_type: str, entity_id: str, row: Dict) -> List[tuple]:
        return []

    def prepare_flat_context(self, shadow: Dict[tuple, Dict]) -> None:
        self.prepared_shadow = shadow

    def observe(self, shadow: Dict[tuple, Dict], action: Action):
        self.observed_shadow = dict(shadow)
        return ("predicted", False)

    def reward_from_shadow(
        self, shadow: Dict[tuple, Dict], candidate_actions: List[Action]
    ) -> RewardInfo:
        return RewardInfo(
            reward=1.0,
            success=("item", "reference") in shadow,
        )


class _CaptureChat:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        if "VERDICT" in messages[-1]["content"]:
            return "VERDICT: SUCCESS"
        return '{"reservation_id":"HATHAT"}'


def test_flat_world_model_prepares_reference_context_before_candidate_action():
    store = _RecordingStore()
    evaluator = _RecordingEvaluator()
    wm = ReadGroundedWorldModel(
        store,
        evaluator,
        SimulatorConfig(kind="flat", node_read_limit=1),
        Budget(),
        BudgetConfig(),
        ReadCache(),
    )

    assert store.reads == [("item", "reference")]
    assert evaluator.seed_calls == [[]]
    assert evaluator.prepared_shadow == {
        ("item", "reference"): {"id": "reference"}
    }

    action = Action("select", {"item_id": "candidate"})
    observation = wm.step(action)
    result = wm.evaluate(Trajectory(steps=[Step(action)]))

    assert observation.content == "predicted"
    assert result.success
    assert store.reads == [("item", "reference")]
    assert evaluator.seed_calls == [[]]
    assert evaluator.observed_shadow == {
        ("item", "reference"): {"id": "reference"}
    }
    assert wm.spawn_root is not None
    assert wm.spawn_root.region == "flat-fixed[1 refs]"


def test_tau2_flat_system_prompt_is_fixed_before_runtime_actions(monkeypatch):
    monkeypatch.setenv("TTS_TAU2_WM_TOOL_MODE", "text")
    predictor = _LLMWorldModelPredictor("airline")
    chat = _CaptureChat()
    predictor._client = chat
    predictor.configure_flat_context(
        task_text="Change the reference reservation.",
        gold_actions=[Action("update_reservation", {"reservation_id": "REF-1"})],
        tool_docs={
            "get_reservation": "Return one reservation.",
            "update_reservation": "Update and return one reservation.",
        },
        facts='reservations REF-1: {"status":"confirmed"}',
        evaluation_context={"communicate_info": ["REF-1"]},
    )

    predictor.predict_observation(
        facts="candidate-specific facts must be ignored",
        history=[],
        action=Action("update_reservation", {"reservation_id": "CANDIDATE-1"}),
        tool_doc="candidate-specific tool selection must be ignored",
    )
    predictor.predict_success(
        facts="candidate-specific facts must be ignored",
        candidate_actions=[
            Action("update_reservation", {"reservation_id": "CANDIDATE-1"})
        ],
        gold_actions=[Action("different_gold", {})],
        tool_docs="candidate-specific docs must be ignored",
    )

    first_system = chat.calls[0][0]
    second_system = chat.calls[1][0]
    assert first_system == second_system
    assert first_system["role"] == "system"
    assert "Change the reference reservation." in first_system["content"]
    assert "REF-1" in first_system["content"]
    assert "get_reservation" in first_system["content"]
    assert "CANDIDATE-1" not in first_system["content"]
    assert "candidate-specific" not in first_system["content"]

    assert chat.calls[0][-1]["role"] == "user"
    assert "CANDIDATE-1" in chat.calls[0][-1]["content"]
    assert "CURRENT AGENT TOOL CALL" in chat.calls[0][-1]["content"]
    assert "CANDIDATE-1" in chat.calls[1][-1]["content"]
    assert "CANDIDATE ACTIONS" in chat.calls[1][-1]["content"]


class _AppWorldFlatChat:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        if "VERDICT" in prompt:
            return "VERDICT: SUCCESS"
        if "READ API:" in prompt:
            return 'PYTHON_FINAL:\nprint_json({"post_write": True})'
        return 'PYTHON_FINAL:\nprint_json({"message": "Updated."})'


class _AppWorldRequester:
    def __init__(self) -> None:
        self.calls = []

    def request(self, app, api, **kwargs):
        self.calls.append((app, api, dict(kwargs)))
        return {"real": True, "app": app, "api": api}


def test_appworld_b1_reuses_fixed_prompt_and_skips_dirty_read_probe(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_WM_TOOL_MODE", "text")
    monkeypatch.setenv("TTS_APPWORLD_VERIFIER_PERCHECK", "0")
    methods = {
        ("notes", "update_note"): "POST",
        ("notes", "show_note"): "GET",
    }
    monkeypatch.setattr(
        appworld_wm,
        "_api_doc",
        lambda app, api: {
            "method": methods[(app, api)],
            "description": f"{api} documentation",
        },
    )
    monkeypatch.setattr(
        appworld_wm,
        "_success_schema",
        lambda app, api: {"message": "string"},
    )

    requester = _AppWorldRequester()
    world = type(
        "World",
        (),
        {
            "requester": requester,
            "task": type(
                "Task",
                (),
                {
                    "datetime": type(
                        "Frozen",
                        (),
                        {"isoformat": lambda self: "2026-01-01T00:00:00"},
                    )(),
                    "app_descriptions": {"notes": "Manage notes."},
                },
            )(),
        },
    )()
    chat = _AppWorldFlatChat()
    predictor = AppWorldWMPredictor(chat)
    appworld_wm.install_wm_requester(world, predictor)
    predictor.configure_flat_context(
        task_text="Update the reference note.",
        tool_docs={"notes": {"update_note": {"method": "POST"}}},
        reference_checklist="note content equals REFERENCE",
        reference_checks=(["note content equals REFERENCE"], "note_id=REF-1"),
        fixed_state={"reference_note": {"note_id": "REF-1", "content": "old"}},
    )

    world.requester.request(
        "notes", "update_note", note_id="CANDIDATE-1", content="new")
    dirty = world.requester.request(
        "notes", "show_note", note_id="CANDIDATE-1")
    confidence = predictor.predict_success(
        "candidate-specific task text",
        ["apis.notes.update_note(note_id='CANDIDATE-1')"],
        ["updated"],
        reference_checklist="candidate-specific checklist",
    )

    assert dirty == {"post_write": True}
    assert confidence == 1.0
    assert requester.calls == []
    systems = [call[0] for call in chat.calls]
    assert all(message["role"] == "system" for message in systems)
    assert all(message == systems[0] for message in systems)
    assert "REF-1" in systems[0]["content"]
    assert "update_note" in systems[0]["content"]
    assert "CANDIDATE-1" not in systems[0]["content"]
    assert any("CANDIDATE-1" in call[-1]["content"] for call in chat.calls)
    assert all(
        "candidate-specific task text" not in call[-1]["content"]
        and "candidate-specific checklist" not in call[-1]["content"]
        for call in chat.calls
    )


def test_appworld_b1_predict_all_clean_read_uses_fixed_context(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_WM_TOOL_MODE", "text")
    monkeypatch.setattr(
        appworld_wm,
        "_api_doc",
        lambda app, api: {"method": "GET", "description": "Show one note."},
    )
    monkeypatch.setattr(
        appworld_wm,
        "_success_schema",
        lambda app, api: {"note_id": "integer", "content": "string"},
    )

    requester = _AppWorldRequester()
    world = type(
        "World",
        (),
        {
            "requester": requester,
            "task": type(
                "Task",
                (),
                {
                    "datetime": type(
                        "Frozen",
                        (),
                        {"isoformat": lambda self: "2026-01-01T00:00:00"},
                    )(),
                    "app_descriptions": {"notes": "Manage notes."},
                },
            )(),
        },
    )()
    predictor = AppWorldWMPredictor(_AppWorldFlatChat())
    appworld_wm.install_wm_requester(
        world, predictor, predict_all_actions=True)
    predictor.configure_flat_context(
        task_text="Read the reference note.",
        tool_docs={"notes": {"show_note": {"method": "GET"}}},
        fixed_state={"reference_note": {"note_id": "REF-1", "content": "old"}},
        predict_all_actions=True,
    )

    result = world.requester.request("notes", "show_note", note_id="REF-1")

    assert result == {"post_write": True}
    assert requester.calls == []
    assert predictor.n_predicted_reads == 1
    assert predictor.n_dirty_reads == 0
    assert "every task-app read and write observation is predicted" in (
        predictor.flat_system_prompt or "")


class _OfficeBenchFlatChat:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def chat(self, system, user, **kwargs):
        self.calls.append((system, user))
        if "MUST use PYTHON_FINAL" in user:
            return f"PYTHON_FINAL:\nprint_json({{'text': {self.response!r}}})"
        return self.response


class _NoOfficeBenchReads:
    available_calls = ["excel.read_file"]

    def read(self, path):
        raise AssertionError("flat B1 must not read after candidate actions")

    def list(self, path):
        raise AssertionError("flat B1 must not list after candidate actions")

    def call(self, name, args):
        raise AssertionError("flat B1 must not call tools after candidate actions")

    def trace(self, path):
        raise AssertionError("flat B1 must not trace after candidate actions")


def test_officebench_b1_shares_fixed_prompt_for_prediction_and_verdict(monkeypatch):
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_PERCHECK", "0")
    config = {
        "task": "Set the reference workbook cell.",
        "username": "Bob",
        "date": "2026-01-01",
        "weekday": "Thursday",
        "time": "10:00 AM",
        "evaluation": [{
            "function": "evaluate_excel_cell_value",
            "args": {
                "file": "./data/reference.xlsx",
                "matches": [{"row": 1, "col": 1, "value": "REFERENCE"}],
            },
        }],
    }
    catalogue = {
        "available_apps": ["excel"],
        "intro": {"excel": "Read and edit workbooks."},
        "actions": {"excel": ["read_file", "set_cell"]},
        "demo": {
            "excel": {
                "read_file": "Read a workbook.",
                "set_cell": "Set one workbook cell.",
            },
        },
    }
    initial_state = {"./data/reference.xlsx": "(1, 1): old"}
    fixed_system = build_officebench_flat_system_prompt(
        task_config=config,
        catalogue=catalogue,
        initial_state=initial_state,
        gold_files={"reference.xlsx": "(1, 1): REFERENCE"},
        testbed_listing="/testbed/data/reference.xlsx",
    )

    predictor_chat = _OfficeBenchFlatChat(
        "OBSERVATION: Successfully set the cell.")
    predictor = OfficeBenchWMPredictor(predictor_chat)
    predictor.configure_flat_context(
        fixed_system, initial_state=initial_state)
    predictor.bind_query_interface(_NoOfficeBenchReads())
    action = {
        "app": "excel",
        "action": "set_cell",
        "file_path": "/testbed/data/CANDIDATE.xlsx",
        "row_idx": 1,
        "column_idx": 1,
        "text": "new",
    }
    predictor.predict_write(action)

    verifier_chat = _OfficeBenchFlatChat("SUCCESS")
    verifier = OfficeBenchVerifier(verifier_chat)
    verifier.configure_flat_context(fixed_system)
    confidence = verifier.predict_success(
        config["task"],
        [action],
        ["OBSERVATION: Successfully set the cell."],
        config=config,
        final_state={"./data/reference.xlsx": "(1, 1): REFERENCE"},
    )

    assert confidence == 1.0
    assert predictor_chat.calls[0][0] == verifier_chat.calls[0][0] == fixed_system
    assert "reference.xlsx" in fixed_system
    assert "evaluate_excel_cell_value" in fixed_system
    assert "CANDIDATE.xlsx" not in fixed_system
    assert "CANDIDATE.xlsx" in predictor_chat.calls[0][1]
    assert "CANDIDATE.xlsx" in verifier_chat.calls[0][1]


class _OfficeBenchAgentChat:
    def __init__(self, actions) -> None:
        self.actions = list(actions)

    def chat(self, system, user, **kwargs):
        return repr(self.actions.pop(0))


class _NoRuntimeReadSession:
    def __init__(self) -> None:
        self.exec_calls = []

    def apps_catalogue(self):
        return {
            "available_apps": ["excel"],
            "intro": {"excel": "Read and edit workbooks."},
            "actions": {"excel": ["read_file", "set_cell"]},
            "demo": {
                "excel": {
                    "read_file": "Read a workbook.",
                    "set_cell": "Set one workbook cell.",
                },
            },
        }

    def exec_action(self, action, workdir):
        self.exec_calls.append((action, workdir))
        raise AssertionError(
            "flat B1 dirty reads must use the pre-action baseline, not the session")

    def write_answer(self, answer):
        raise AssertionError("flat B1 must not write the answer during sampling")


def test_officebench_b1_dirty_read_uses_preaction_baseline(monkeypatch):
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    target = "/testbed/data/reference.xlsx"
    config = {
        "task": "Update and read the workbook.",
        "username": "Bob",
        "date": "2026-01-01",
        "weekday": "Thursday",
        "time": "10:00 AM",
        "evaluation": [],
    }
    fixed_system = build_officebench_flat_system_prompt(
        task_config=config,
        catalogue=_NoRuntimeReadSession().apps_catalogue(),
        initial_state={"./data/reference.xlsx": "(1, 1): old"},
    )
    predictor = OfficeBenchWMPredictor(
        _OfficeBenchFlatChat("OBSERVATION: predicted"))
    predictor.configure_flat_context(
        fixed_system,
        initial_state={"./data/reference.xlsx": "(1, 1): old"},
    )
    agent = _OfficeBenchAgentChat([
        {"app": "system", "action": "switch_app", "target_app": "excel"},
        {
            "app": "excel",
            "action": "set_cell",
            "file_path": target,
            "row_idx": 1,
            "column_idx": 1,
            "text": "new",
        },
        {"app": "excel", "action": "read_file", "file_path": target},
        {"app": "system", "action": "finish_task", "answer": "None"},
    ])
    session = _NoRuntimeReadSession()

    run = run_officebench_agent(
        session, config, agent, max_steps=6, predictor=predictor)

    assert session.exec_calls == []
    assert predictor.n_dirty_reads == 1
    assert run.final_answer == "None"


def test_officebench_b1_predict_all_clean_read_uses_fixed_context(monkeypatch):
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    target = "/testbed/data/reference.xlsx"
    config = {
        "task": "Read the workbook.",
        "username": "Bob",
        "date": "2026-01-01",
        "weekday": "Thursday",
        "time": "10:00 AM",
        "evaluation": [],
    }
    session = _NoRuntimeReadSession()
    fixed_system = build_officebench_flat_system_prompt(
        task_config=config,
        catalogue=session.apps_catalogue(),
        initial_state={"./data/reference.xlsx": "(1, 1): old"},
        predict_all_actions=True,
    )
    predictor = OfficeBenchWMPredictor(
        _OfficeBenchFlatChat("(1, 1): predicted"))
    predictor.configure_flat_context(
        fixed_system,
        initial_state={"./data/reference.xlsx": "(1, 1): old"},
    )
    agent = _OfficeBenchAgentChat([
        {"app": "system", "action": "switch_app", "target_app": "excel"},
        {"app": "excel", "action": "read_file", "file_path": target},
        {"app": "system", "action": "finish_task", "answer": "None"},
    ])

    run = run_officebench_agent(
        session,
        config,
        agent,
        max_steps=5,
        predictor=predictor,
        predict_all_actions=True,
    )

    assert session.exec_calls == []
    assert predictor.n_predicted_reads == 1
    assert predictor.n_dirty_reads == 0
    assert run.observations[1] == "(1, 1): predicted"
    assert "every app read and write observation is predicted" in fixed_system
