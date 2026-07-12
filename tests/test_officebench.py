"""Guarded tests for the REAL OfficeBench integration.

OfficeBench runs its office apps + official evaluation inside a Docker image (built on demand from
``src/tts/env/officebench_docker``), driven from the host through :mod:`tts.env.officebench_lowlevel`.
The pure-logic tests (prompt/state-machine/driver helpers/verifier) need neither Docker nor an LLM and
always run; the live tests are skipped unless the OfficeBench clone (``OFFICEBENCH_ROOT``), a usable
Docker daemon, the built app image, and (for B0) an LLM endpoint are all available.
"""

from __future__ import annotations

import os

import pytest

from tts.config import Config, SolverConfig
from tts.harness import run_decision
from tts.trajectory import Trajectory
from tts.trajectory import Step
from tts.types import Action

try:
    from tts.env.officebench_lowlevel import (
        officebench_image_available,
        real_officebench_available,
    )
except Exception:  # pragma: no cover
    real_officebench_available = lambda *a, **k: False  # noqa: E731
    officebench_image_available = lambda *a, **k: False  # noqa: E731

officebench = pytest.mark.skipif(not real_officebench_available(),
                                 reason="real OfficeBench not available (set OFFICEBENCH_ROOT + Docker)")
officebench_image = pytest.mark.skipif(
    not (real_officebench_available() and officebench_image_available()),
    reason="OfficeBench app image not built (build src/tts/env/officebench_docker)")
needs_llm = pytest.mark.skipif(
    not (os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("OPENAI_API_KEY")),
    reason="no LLM endpoint configured (set AZURE_OPENAI_ENDPOINT or OPENAI_API_KEY)",
)

# A minimal fake app catalogue (the shape OfficeBenchSession.apps_catalogue returns).
_FAKE_CATALOGUE = {
    "available_apps": ["calendar", "shell"],
    "intro": {"calendar": "calendar: manage events.", "shell": "shell: run commands."},
    "demo": {"calendar": {"create_event": "create an event: {'app':'calendar','action':'create_event',...}"},
             "shell": {"command": "run a command: {'app':'shell','action':'command','command':...}"}},
    "actions": {"calendar": ["create_event"], "shell": ["command"]},
}
_FAKE_CONFIG = {"username": "Bob", "date": "2020-05-01", "weekday": "Friday", "time": "10:00 AM",
                "task": "Add a meeting to Bob's calendar.",
                "evaluation": [{"function": "evaluate_contain",
                                "args": {"doc_type": "ics", "file": "./calendar/Bob.ics",
                                         "keywords": ["Meeting"]}}]}


# --------------------------------------------------------------------------- #
# Pure logic (no Docker, no LLM)
# --------------------------------------------------------------------------- #


def test_proc_action_and_parse():
    from tts.env.officebench_agent import _parse_action, proc_action

    assert proc_action("blah {'app': 'system', 'action': 'switch_app'} trailing") == \
        "{'app': 'system', 'action': 'switch_app'}"
    assert proc_action("no braces here") == "no braces here"
    parsed = _parse_action("{'app': 'calendar', 'action': 'create_event', 'user': ['Bob']}")
    assert parsed == {"app": "calendar", "action": "create_event", "user": "Bob"}  # single-list unwrap
    assert _parse_action("not a dict") is None


def test_build_prompt_state_machine():
    from tts.env.officebench_agent import build_prompt, construct_system_message

    sys_msg = construct_system_message(_FAKE_CONFIG, _FAKE_CATALOGUE)
    assert "Bob" in sys_msg and "calendar: manage events." in sys_msg and "shell: run commands." in sys_msg

    # no app chosen yet -> the only instruction is switch_app
    undecided = build_prompt(None, [], _FAKE_CONFIG, _FAKE_CATALOGUE)
    assert "switch_app" in undecided and "create_event" not in undecided
    # inside an app -> its action DEMOs + switch/finish
    decided = build_prompt("calendar", [({"app": "calendar"}, "ok")], _FAKE_CONFIG, _FAKE_CATALOGUE)
    assert "create_event" in decided and "finish_task" in decided and "Current apps: calendar" in decided
    # a switch to a non-existent app degrades gracefully to the undecided prompt
    assert "switch_app" in build_prompt("bogus", [], _FAKE_CONFIG, _FAKE_CATALOGUE)


def test_driver_pure_helpers():
    from tts.env import officebench_driver as drv

    assert drv._simplify_path("/testbed", "data") == "/testbed/data"
    assert drv._simplify_path("/testbed/data", "..") == "/testbed"
    assert drv._simplify_path("/a/b", "/c") == "/c"
    assert drv._minor_action_fix({"user": ["Bob"], "x": [1, 2]}) == {"user": "Bob", "x": [1, 2]}


def test_driver_read_only_reverts_incidental_creates(tmp_path):
    """A read-only exec must revert any file an official read helper creates-if-missing.

    ``calendar.list_events`` writes a blank ``<user>.ics`` and ``email.list_emails`` makedirs an
    account dir on first read; on the world-model shadow those incidental creates are rolled back so
    ``/testbed`` stays byte-for-byte pristine ("no write, even shadow").
    """
    from tts.env import officebench_driver as drv

    root = tmp_path / "testbed"
    (root / "data").mkdir(parents=True)
    (root / "data" / "keep.txt").write_text("original")

    before = drv._list_files(str(root))
    # simulate a create-if-missing read helper: a new blank calendar file appears during the read
    (root / "calendar").mkdir()
    (root / "calendar" / "Bob.ics").write_text("")
    removed = drv._remove_new_files(str(root), before)

    assert str(root / "calendar" / "Bob.ics") in removed
    assert drv._list_files(str(root)) == before  # pristine again
    assert (root / "data" / "keep.txt").read_text() == "original"  # pre-existing files untouched


def test_readonly_session_passes_read_only_flag_to_driver(monkeypatch):
    """``exec_action`` tells the in-container driver whether the session is read-only (so it can
    revert incidental read-helper writes) and still refuses genuine writes structurally."""
    import pytest as _pytest

    from tts.env.officebench_lowlevel import OfficeBenchSession

    sess = OfficeBenchSession.__new__(OfficeBenchSession)  # bypass Docker/clone construction
    sess.read_only = True
    calls = {}

    def _fake_driver(cmd, payload):
        calls["cmd"], calls["payload"] = cmd, payload
        return {"observation": "ok", "exit_code": 0, "is_cd": False, "new_workdir": None}

    monkeypatch.setattr(sess, "_driver", _fake_driver)

    sess.exec_action({"app": "shell", "action": "command", "command": "ls /testbed"}, "/")
    assert calls["cmd"] == "exec" and calls["payload"]["read_only"] is True

    # a genuine write is refused before ever reaching the driver
    calls.clear()
    with _pytest.raises(PermissionError):
        sess.exec_action({"app": "calendar", "action": "create_event", "user": "Bob"}, "/")
    assert calls == {}


def test_wm_action_classification():
    """Reads vs writes (and the file each touches) are classified so writes are never executed."""
    from tts.env.officebench_wm_predict import is_write_action, read_target, write_targets

    assert is_write_action({"app": "excel", "action": "set_cell", "file_path": "/testbed/data/s.xlsx"})
    assert not is_write_action({"app": "excel", "action": "read_file", "file_path": "./data/s.xlsx"})
    assert is_write_action({"app": "calendar", "action": "create_event", "user": "Bob"})
    assert not is_write_action({"app": "calendar", "action": "list_events", "username": "Bob"})
    assert not is_write_action({"app": "shell", "action": "command", "command": "ls /testbed/data"})
    assert is_write_action({"app": "shell", "action": "command", "command": "rm /testbed/data/x"})
    assert is_write_action({"app": "shell", "action": "command", "command": "echo hi > /testbed/o.txt"})
    assert not is_write_action({"app": "system", "action": "finish_task", "answer": "None"})
    assert write_targets({"app": "excel", "action": "set_cell", "file_path": "./data/s.xlsx"}) == ["data/s.xlsx"]
    assert read_target({"app": "excel", "action": "read_file", "file_path": "/testbed/data/s.xlsx"}) == "data/s.xlsx"


def test_wm_global_dirty_fallback():
    """An unresolved-target write (e.g. an opaque shell command) makes ALL later reads predicted."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    p = OfficeBenchWMPredictor(_StubChat("OBSERVATION: ok"))
    p.predict_write({"app": "excel", "action": "set_cell", "file_path": "/testbed/data/a.xlsx"})
    assert p.was_written("data/a.xlsx") and not p.was_written("data/b.xlsx") and not p.global_dirty
    # an opaque shell write can't be attributed to a file -> global-dirty: nothing is provably clean
    p.predict_write({"app": "shell", "action": "command", "command": "python3 munge.py"})
    assert p.global_dirty
    assert p.was_written("data/b.xlsx") and p.was_written("anything")
    assert not p.was_written(None)  # a fileless read (llm.complete_text) is never dirty


class _ScriptChat:
    """A chat stub returning queued responses in order (for driving the agent loop offline)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, system, user, **kw):
        self.calls += 1
        return self._responses.pop(0) if self._responses else repr({"app": "system", "action": "got_stuck"})


class _RecordingSession:
    """A fake OfficeBenchSession that records exec_action calls (and REFUSES writes when read_only)."""

    def __init__(self, read_only=True):
        self.read_only = read_only
        self.exec_calls = []
        self.write_answer_calls = []

    def apps_catalogue(self):
        return {"available_apps": ["excel", "calendar"],
                "intro": {"excel": "excel app", "calendar": "cal app"},
                "demo": {"excel": {"read_file": "d", "set_cell": "d"}, "calendar": {"create_event": "d"}},
                "actions": {"excel": ["read_file", "set_cell"], "calendar": ["create_event"]}}

    def exec_action(self, action, workdir):
        from tts.env.officebench_wm_predict import is_write_action

        if self.read_only and is_write_action(action):
            raise AssertionError("WRITE executed on a read-only shadow session")
        self.exec_calls.append(action)
        return {"observation": "OBSERVATION: read ok", "exit_code": 0, "is_cd": False, "new_workdir": None}

    def write_answer(self, answer):
        self.write_answer_calls.append(answer)


def test_wm_predictor_never_executes_writes():
    """In WM mode the agent's WRITE actions are predicted (recorded), never sent to exec_action."""
    from tts.env.officebench_agent import run_officebench_agent
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    # agent script: switch to excel, read (exec), set_cell (predict-write), finish
    agent = _ScriptChat([
        repr({"app": "system", "action": "switch_app", "target_app": "excel"}),
        repr({"app": "excel", "action": "read_file", "file_path": "/testbed/data/s.xlsx"}),
        repr({"app": "excel", "action": "set_cell", "file_path": "/testbed/data/s.xlsx",
              "row_idx": 1, "column_idx": 2, "text": 5}),
        repr({"app": "system", "action": "finish_task", "answer": "done"}),
    ])
    predictor = OfficeBenchWMPredictor(_StubChat("OBSERVATION: Successfully set the cell."))
    session = _RecordingSession(read_only=True)
    cfg = {"username": "Bob", "date": "2020-05-01", "weekday": "Fri", "time": "10AM",
           "task": "set a cell", "evaluation": []}
    run = run_officebench_agent(session, cfg, agent, max_steps=6, predictor=predictor)

    # the write was predicted + recorded, never executed; only the read hit exec_action
    assert predictor.n_writes == 1 and "data/s.xlsx" in predictor.applied_writes
    assert all(a.get("action") != "set_cell" for a in session.exec_calls)
    assert any(a.get("action") == "read_file" for a in session.exec_calls)
    # finish_task did NOT write answer.txt to the shadow (predicted only)
    assert session.write_answer_calls == [] and run.final_answer == "done"


def test_wm_dirty_read_uses_predictor():
    """A read of a file a prior write changed is served by predict_read, not a stale exact read."""
    from tts.env.officebench_agent import run_officebench_agent
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    agent = _ScriptChat([
        repr({"app": "system", "action": "switch_app", "target_app": "excel"}),
        repr({"app": "excel", "action": "set_cell", "file_path": "/testbed/data/s.xlsx",
              "row_idx": 1, "column_idx": 1, "text": 9}),
        repr({"app": "excel", "action": "read_file", "file_path": "/testbed/data/s.xlsx"}),
        repr({"app": "system", "action": "finish_task", "answer": "None"}),
    ])

    class _P(OfficeBenchWMPredictor):
        def __init__(self):
            super().__init__(_StubChat("x"))
            self.dirty_read_called = False

        def predict_read(self, action, pre, history=()):
            self.dirty_read_called = True
            return "OBSERVATION: predicted post-write content (1,1)=9"

    predictor = _P()
    session = _RecordingSession(read_only=True)
    cfg = {"username": "Bob", "date": "d", "weekday": "w", "time": "t", "task": "t", "evaluation": []}
    run = run_officebench_agent(session, cfg, agent, max_steps=6, predictor=predictor)
    assert predictor.dirty_read_called  # the read-after-write went through predict_read
    assert "predicted post-write" in run.observations[-2]  # the dirty-read observation


def test_wm_predicts_write_failure_leaves_state_clean(monkeypatch):
    """A write the WM predicts would FAIL is NOT applied (predicted state stays clean) but is logged."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_CHECK_WRITES", "1")
    # the world model judges this write impossible (missing input file) -> realistic failure observation
    p = OfficeBenchWMPredictor(_StubChat(
        "OBSERVATION: The file /testbed/data/missing.xlsx does not exist. Failed to read the file."))
    obs = p.predict_write({"app": "excel", "action": "set_cell", "file_path": "/testbed/data/missing.xlsx",
                           "row_idx": 1, "column_idx": 1, "text": 5})
    assert "does not exist" in obs
    assert p.n_failed_writes == 1
    assert not p.was_written("data/missing.xlsx")            # a failed write does not change the file
    assert not p.applied_writes and not p.global_dirty       # nothing recorded / no dirty collapse
    assert p.write_history and p.write_history[-1][1] == obs  # but the attempt is visible to the verifier


def test_wm_success_write_still_applies(monkeypatch):
    """A predicted SUCCESS observation is applied as before (no false-failure regressions)."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_CHECK_WRITES", "1")
    p = OfficeBenchWMPredictor(_StubChat("OBSERVATION: Successfully set the cell."))
    p.predict_write({"app": "excel", "action": "set_cell", "file_path": "/testbed/data/s.xlsx"})
    assert p.was_written("data/s.xlsx") and p.n_failed_writes == 0


def test_predicted_state_gold_backed_uses_writelog_not_reconstruction():
    """A gold-referenced written file is shown as initial + write-log — NOT an LLM reconstruction."""
    from tts.env.real_officebench import OfficeBenchTaskEnv
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    class _FakeSession:
        def read_state(self, files):
            return {f["path"]: "(1, 1): Alice\t(1, 2): 90" for f in files}

    class _P(OfficeBenchWMPredictor):
        def __init__(self):
            super().__init__(_StubChat("x"))
            self.reconstructed = False

        def predict_file_state(self, path, init):
            self.reconstructed = True
            return "FABRICATED"

    p = _P()
    p.applied_writes["data/sorted_score.xlsx"] = [
        ({"app": "excel", "action": "set_cell", "file_path": "./data/sorted_score.xlsx",
          "row_idx": 1, "column_idx": 2, "text": 95}, "OBSERVATION: Successfully set the cell.")]
    files = [{"path": "./data/sorted_score.xlsx", "doc_type": "xlsx"}]
    # _predicted_state uses no instance state, so it can be called unbound (no Docker env needed).
    out = OfficeBenchTaskEnv._predicted_state(None, _FakeSession(), p, files,
                                              gold_labels={"sorted_score.xlsx"})
    entry = out["./data/sorted_score.xlsx"]
    assert not p.reconstructed                         # gold-backed -> reconstruction skipped
    assert "INITIAL content" in entry and "WRITE actions" in entry and "set_cell" in entry


def test_predicted_state_no_gold_reconstructs():
    """A written file WITHOUT a gold reference falls back to the conservative reconstruction."""
    from tts.env.real_officebench import OfficeBenchTaskEnv
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    class _FakeSession:
        def read_state(self, files):
            return {f["path"]: "(1, 1): 10" for f in files}

    class _P(OfficeBenchWMPredictor):
        def __init__(self):
            super().__init__(_StubChat("x"))
            self.reconstructed = False

        def predict_file_state(self, path, init):
            self.reconstructed = True
            return "RECONSTRUCTED"

    p = _P()
    p.applied_writes["data/score.xlsx"] = [
        ({"app": "excel", "action": "set_cell", "file_path": "./data/score.xlsx"},
         "OBSERVATION: Successfully set the cell.")]
    files = [{"path": "./data/score.xlsx", "doc_type": "xlsx"}]
    out = OfficeBenchTaskEnv._predicted_state(None, _FakeSession(), p, files, gold_labels=set())
    assert p.reconstructed and out["./data/score.xlsx"] == "RECONSTRUCTED"


class _StubChat:
    """Records the last prompt and returns a fixed verdict (no network)."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.last_prompt = None

    def chat(self, system, user, **kw):
        self.last_prompt = user
        return self.verdict


def test_verifier_blind_and_reference(monkeypatch):
    from tts.env.officebench_wm import OfficeBenchVerifier

    actions = [{"app": "calendar", "action": "create_event", "user": "Bob"}]
    obs = ["Successfully create a new event to Bob's calendar."]

    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_SAMPLES", "1")
    succ = OfficeBenchVerifier(_StubChat("SUCCESS")).predict_success(
        _FAKE_CONFIG["task"], actions, obs, config=_FAKE_CONFIG)
    fail = OfficeBenchVerifier(_StubChat("FAILURE")).predict_success(
        _FAKE_CONFIG["task"], actions, obs, config=_FAKE_CONFIG)
    assert succ == 1.0 and fail == 0.0

    # reference grounding is ON by default -> the prompt audits against the ground-truth checks
    default_on = _StubChat("SUCCESS")
    OfficeBenchVerifier(default_on).predict_success(_FAKE_CONFIG["task"], actions, obs, config=_FAKE_CONFIG)
    assert "GROUND-TRUTH SUCCESS CHECKS" in default_on.last_prompt and "evaluate_contain" in default_on.last_prompt
    # explicit opt-out -> a blind WM-only judge (no ground-truth checks)
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_REFERENCE", "0")
    blind = _StubChat("SUCCESS")
    OfficeBenchVerifier(blind).predict_success(_FAKE_CONFIG["task"], actions, obs, config=_FAKE_CONFIG)
    assert "GROUND-TRUTH SUCCESS CHECKS" not in blind.last_prompt


def test_verifier_gold_expected_file(monkeypatch):
    """For exact_match tasks the verifier is shown the GOLD expected-output file contents."""
    from tts.env.officebench_wm import OfficeBenchVerifier, reference_files_from_config

    cfg = {"task": "sort and save sorted_score.xlsx",
           "evaluation": [{"function": "evaluate_exact_match",
                           "args": {"doc_type": "xlsx", "result_file": "./data/sorted_score.xlsx",
                                    "expected_file": "../../../../reference/sorted_score.xlsx"}}]}
    specs = reference_files_from_config(cfg)
    assert specs == [{"path": "/reference/sorted_score.xlsx", "doc_type": "xlsx",
                      "label": "sorted_score.xlsx"}]
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_REFERENCE", "1")
    chat = _StubChat("SUCCESS")
    OfficeBenchVerifier(chat).predict_success(
        cfg["task"], [{"app": "excel", "action": "set_cell"}], [""], config=cfg,
        gold_files={"sorted_score.xlsx": "(1, 1): Alice\t(1, 2): 95"})
    assert "GOLD EXPECTED OUTPUT" in chat.last_prompt and "(1, 1): Alice" in chat.last_prompt


def test_verifier_state_grounding_and_giveup():
    """The grounded verifier shows the ACTUAL resulting file state and flags give-up answers."""
    from tts.env.officebench_wm import OfficeBenchVerifier, _is_giveup, state_files_from_config

    # give-up detection (the false-positive source the analysis surfaced)
    assert _is_giveup("None") and _is_giveup("") and _is_giveup("Sorry, I could not find your score")
    assert not _is_giveup("The meeting was added to Bob's calendar")

    # the resulting file state is injected into the prompt; the answer-claim is no longer trusted alone
    grounded = _StubChat("FAILURE")
    OfficeBenchVerifier(grounded).predict_success(
        _FAKE_CONFIG["task"], [{"app": "excel", "action": "set_cell"}], [""],
        config=_FAKE_CONFIG,
        final_state={"data/score.xlsx": "(2, 2): 1000+2000000\t"},
        final_answer="None")
    assert "PREDICTED RESULTING FILE STATE" in grounded.last_prompt
    assert "1000+2000000" in grounded.last_prompt  # judge sees the literal string, not the claim
    assert "give-up/refusal" in grounded.last_prompt  # the give-up note fired for answer "None"


def test_state_files_from_config():
    """The result-file list (path + doc_type) is derived from the eval config (incl. calendar)."""
    from tts.env.officebench_wm import state_files_from_config

    cfg = {"evaluation": [
        {"function": "evaluate_excel_cell_value", "args": {"file": "data/score.xlsx", "matches": []}},
        {"function": "evaluate_calendar_no_overlap", "args": {"username": "Bob"}}]}
    files = state_files_from_config(cfg)
    assert {"path": "data/score.xlsx", "doc_type": "xlsx"} in files
    assert {"path": "./calendar/Bob.ics", "doc_type": "ics"} in files


# --------------------------------------------------------------------------- #
# Fix 1 — deterministic symbolic write-propagation (no LLM, no execution)
# --------------------------------------------------------------------------- #


def test_symbolic_set_cell_applies_exactly():
    """A sequence of set_cell writes is overlaid on the sheet exactly — other cells stay verbatim."""
    from tts.env.officebench_wm_predict import symbolic_file_state

    init = "(1, 1): Alice\t(1, 2): 90\n(2, 1): Bob\t(2, 2): 75"
    writes = [
        ({"app": "excel", "action": "set_cell", "row_idx": 1, "column_idx": 2, "text": 100}, "o"),
        ({"app": "excel", "action": "set_cell", "row_idx": 3, "column_idx": 1, "text": "Cara"}, "o"),
    ]
    out = symbolic_file_state("data/s.xlsx", init, writes)
    assert "(1, 2): 100" in out and "(1, 1): Alice" in out and "(2, 2): 75" in out
    assert "(3, 1): Cara" in out  # a new cell is appended


def test_symbolic_word_carries_literal_verbatim():
    """An in-trajectory-authored word file returns its literal contents verbatim (no re-derivation)."""
    from tts.env.officebench_wm_predict import symbolic_file_state

    writes = [
        ({"app": "word", "action": "create_new_file", "file_path": "/testbed/data/h.docx"}, "o"),
        ({"app": "word", "action": "write_to_file", "file_path": "/testbed/data/h.docx",
          "contents": "Line1\nLine2 | with a pipe"}, "o"),
    ]
    out = symbolic_file_state("data/h.docx", "[file does not exist]", writes)
    assert out == "Line1\nLine2 | with a pipe"


def test_symbolic_falls_back_when_not_modelable():
    """Mixed/under-specified/unparseable writes return None so the LLM path is used (no fabrication)."""
    from tts.env.officebench_wm_predict import symbolic_file_state

    init = "(1, 1): 10"
    # mixed apps
    assert symbolic_file_state("x", init, [
        ({"app": "excel", "action": "set_cell", "row_idx": 1, "column_idx": 1, "text": 1}, "o"),
        ({"app": "calendar", "action": "create_event", "user": "Bob"}, "o")]) is None
    # set_cell missing a coordinate
    assert symbolic_file_state("x", init, [
        ({"app": "excel", "action": "set_cell", "row_idx": 1, "text": 5}, "o")]) is None
    # set_cell against an unparseable sheet
    assert symbolic_file_state("x", "[file does not exist]", [
        ({"app": "excel", "action": "set_cell", "row_idx": 1, "column_idx": 1, "text": 5}, "o")]) is None


def test_symbolic_disabled_by_env(monkeypatch):
    """With TTS_OFFICEBENCH_WM_SYMBOLIC=0 the pure transform is bypassed (legacy LLM behaviour)."""
    from tts.env.officebench_wm_predict import symbolic_file_state

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_SYMBOLIC", "0")
    writes = [({"app": "excel", "action": "set_cell", "row_idx": 1, "column_idx": 1, "text": 5}, "o")]
    assert symbolic_file_state("x", "(1, 1): 10", writes) is None


def test_predict_file_state_symbolic_skips_llm():
    """predict_file_state uses the symbolic reconstruction and never calls the LLM for set_cell."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    class _BoomChat:
        def chat(self, *a, **k):
            raise AssertionError("LLM must not be called for an exactly-modelable file")

    p = OfficeBenchWMPredictor(_BoomChat())
    p.applied_writes["data/s.xlsx"] = [
        ({"app": "excel", "action": "set_cell", "row_idx": 1, "column_idx": 1, "text": 42}, "o")]
    out = p.predict_file_state("./data/s.xlsx", "(1, 1): 10")
    assert out == "(1, 1): 42" and p.n_llm_calls == 0


# --------------------------------------------------------------------------- #
# Fix 3 — action-stream corruption (parse recovery + rendering hygiene)
# --------------------------------------------------------------------------- #


def test_parse_action_recovers_first_of_two_objects():
    """When the model emits two JSON objects in one turn, the FIRST valid action is recovered."""
    from tts.env.officebench_agent import _parse_action

    two = ('{"app": "word", "action": "create_new_file", "file_path": "/testbed/data/h.docx"}\n'
           '{"app": "word", "action": "write_to_file", "file_path": "/testbed/data/h.docx", '
           '"contents": "a | b\\nc"}')
    action = _parse_action(two)
    assert action is not None and action["action"] == "create_new_file"


def test_parse_action_recovery_disabled(monkeypatch):
    """With sanitising off, the strict official parse still drops the multi-object turn as malformed."""
    from tts.env.officebench_agent import _parse_action

    monkeypatch.setenv("TTS_OFFICEBENCH_SANITIZE_ACTIONS", "0")
    two = '{"app": "shell", "action": "command", "command": "ls"}\n{"app": "system", "action": "got_stuck"}'
    assert _parse_action(two) is None


def test_format_history_truncates_document_content():
    """A document-sized literal argument is collapsed/truncated in the model-facing history."""
    from tts.env.officebench_agent import _format_history

    big = "X\n" * 5000
    action = {"app": "word", "action": "write_to_file", "file_path": "/d.docx", "contents": big}
    rendered = _format_history([(action, "OBSERVATION: ok")])
    assert "\n\n" not in rendered           # embedded newlines collapsed, log stays one line per step
    assert len(rendered) < 2000             # the megabyte literal is truncated, not spliced verbatim
    assert "write_to_file" in rendered


def test_signature_bounds_and_collapses_content():
    """Trajectory.signature caps each action and collapses newlines so records stay parseable."""
    from tts.trajectory import Trajectory, Step
    from tts.types import Action

    big = "Y" * 5000
    traj = Trajectory(steps=[Step(action=Action("word.write_to_file",
                                                 {"step": 0, "action": {"contents": big + "\nZ"}}))])
    sig = traj.signature()
    assert "\n" not in sig and "…" in sig and len(sig) < 500


# --------------------------------------------------------------------------- #
# Fix 4 — step-budget scaling + degenerate-loop pruning
# --------------------------------------------------------------------------- #


def test_scaled_max_steps_by_app_count(monkeypatch):
    """The default step budget keeps the 1app floor and adds headroom per extra app (capped)."""
    from tts.env.real_officebench import _scaled_max_steps

    monkeypatch.delenv("TTS_OFFICEBENCH_STEPS_FLOOR", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_STEPS_PER_APP", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_STEPS_CAP", raising=False)
    assert _scaled_max_steps(1) == 20 and _scaled_max_steps(2) == 26 and _scaled_max_steps(3) == 32
    assert _scaled_max_steps(20) == 40  # capped


def test_prune_repeated_reads_saves_budget():
    """An immediate-repeat read is reused (not re-executed) and does not consume the step budget."""
    from tts.env.officebench_agent import run_officebench_agent
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    # switch to shell, then the SAME list command 4x, then finish. Without pruning the 4 repeats would
    # eat 4 of the 6 steps; with pruning only the first hits exec_action and the budget is preserved.
    ls = repr({"app": "shell", "action": "command", "command": "ls /testbed/data"})
    agent = _ScriptChat([
        repr({"app": "system", "action": "switch_app", "target_app": "shell"}),
        ls, ls, ls, ls,
        repr({"app": "system", "action": "finish_task", "answer": "None"}),
    ])
    predictor = OfficeBenchWMPredictor(_StubChat("x"))
    session = _RecordingSession(read_only=True)
    cfg = {"username": "B", "date": "d", "weekday": "w", "time": "t", "task": "t", "evaluation": []}
    run = run_officebench_agent(session, cfg, agent, max_steps=6, predictor=predictor)

    ls_execs = [a for a in session.exec_calls if a.get("command") == "ls /testbed/data"]
    assert len(ls_execs) == 1                      # the 3 immediate repeats were pruned (not re-executed)
    assert run.final_answer == "None"              # finish still reached within the budget
    assert any("repeated action" in o for o in run.observations)


def test_prune_disabled_reexecutes(monkeypatch):
    """With TTS_OFFICEBENCH_PRUNE_LOOPS=0 repeated reads are executed every time (legacy behaviour)."""
    from tts.env.officebench_agent import run_officebench_agent
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    monkeypatch.setenv("TTS_OFFICEBENCH_PRUNE_LOOPS", "0")
    ls = repr({"app": "shell", "action": "command", "command": "ls /testbed/data"})
    agent = _ScriptChat([
        repr({"app": "system", "action": "switch_app", "target_app": "shell"}),
        ls, ls, repr({"app": "system", "action": "finish_task", "answer": "None"})])
    predictor = OfficeBenchWMPredictor(_StubChat("x"))
    session = _RecordingSession(read_only=True)
    cfg = {"username": "B", "date": "d", "weekday": "w", "time": "t", "task": "t", "evaluation": []}
    run_officebench_agent(session, cfg, agent, max_steps=6, predictor=predictor)
    ls_execs = [a for a in session.exec_calls if a.get("command") == "ls /testbed/data"]
    assert len(ls_execs) == 2                      # both reads executed (no pruning)


# --------------------------------------------------------------------------- #
# Live (Docker / LLM) — skipped unless the real OfficeBench stack is available
# --------------------------------------------------------------------------- #


@officebench
def test_officebench_task_metadata_loads():
    """The env exposes the REAL subtask instruction + the office-app tool schema (no container yet)."""
    from tts.env.real_officebench import OFFICEBENCH_APPS, OfficeBenchTaskEnv

    env = OfficeBenchTaskEnv(subset="1app", index=0)
    assert "/" in env.task_id  # "<task>/<subtask>"
    assert isinstance(env.restricted_view().instruction(), str) and env.restricted_view().instruction()
    assert env.restricted_view().tool_schema() == list(OFFICEBENCH_APPS)
    assert env.state_hash() == f"officebench:{env.task_id}"


@officebench_image
def test_officebench_official_evaluation_scores_a_deterministic_trajectory():
    """End-to-end (no LLM): replaying a correct calendar action makes the OFFICIAL checks pass.

    Drives task ``1-1/0`` ("Add a meeting to Bob's calendar ... 10:30-11:00") with a hand-built
    ``create_event`` action and asserts the official ``utils/evaluate.py`` checks (run in-container)
    pass — exercising the container, the official ``apps`` modules, the driver, and the official
    evaluation, end to end, without any model call.
    """
    from tts.env.officebench_lowlevel import load_officebench_subtasks
    from tts.env.real_officebench import OfficeBenchTaskEnv

    task_id, subtask_id = next((t, s) for (t, s) in load_officebench_subtasks(1) if (t, s) == ("1-1", "0"))
    env = OfficeBenchTaskEnv(subset="1app", task_id=task_id, subtask_id=subtask_id)

    good = {"app": "calendar", "action": "create_event", "user": "Bob", "summary": "Meeting",
            "time_start": "2024-05-17 10:30:00", "time_end": "2024-05-17 11:00:00"}
    traj = Trajectory(steps=[Step(
        action=Action("calendar.create_event", {"step": 0, "action": good}))])
    ri = env.evaluate(traj)
    assert ri.grounded and ri.success and ri.reward == 1.0

    # an empty trajectory must be SCORED (and fail), proving the official evaluator actually ran
    empty = env.evaluate(Trajectory(steps=[]))
    assert empty.grounded and empty.success is False and "OfficeBench evaluate" in empty.reason


@officebench_image
@needs_llm
def test_officebench_b0_real_agent_runs():
    """B0 drives the official OfficeBench agent against the real container and scores via evaluate()."""
    from tts.env.real_officebench import OfficeBenchTaskEnv

    env = OfficeBenchTaskEnv(subset="1app", task_id="1-1", subtask_id="0")
    cfg = Config(method="b0", env="officebench", N=1, seed=0,
                 solver=SolverConfig(agent="llm", max_steps=15, temperature=0.0))
    out = run_decision(env, cfg)
    r = out.record
    assert r.n_commits == 1 and r.simulator_writes == 0
    assert r.reward in (0.0, 1.0)  # official OfficeBench execution-based score


@officebench_image
def test_officebench_readonly_session_refuses_writes():
    """A read-only (world-model) session executes reads but REFUSES to execute any write action."""
    from tts.env.officebench_lowlevel import OfficeBenchSession

    s = OfficeBenchSession("1-1", "0", phase="rotest", read_only=True).start()
    try:
        # a read executes fine
        r = s.exec_action({"app": "shell", "action": "command", "command": "ls /testbed/data"}, "/")
        assert r["exit_code"] == 0
        # a write action is refused structurally (never reaches the container)
        with pytest.raises(PermissionError):
            s.exec_action({"app": "calendar", "action": "create_event", "user": "Bob",
                           "summary": "M", "time_start": "2024-05-17 10:30:00",
                           "time_end": "2024-05-17 11:00:00"}, "/")
    finally:
        s.close()


@officebench_image
def test_officebench_readonly_exact_read_leaves_testbed_pristine():
    """An exact READ whose official helper create-if-missing (``calendar.list_events`` writes a blank
    ``<user>.ics``) must leave the read-only shadow ``/testbed`` byte-for-byte pristine — the driver
    reverts the incidental create, so a "read" never mutates the environment during sampling/judging."""
    from tts.env.officebench_lowlevel import OfficeBenchSession

    s = OfficeBenchSession("1-1", "0", phase="roexact", read_only=True).start()
    try:
        before = s.testbed_digest()
        # list_events on a user with no existing calendar would otherwise CREATE a blank .ics
        r = s.exec_action({"app": "calendar", "action": "list_events",
                           "username": "NoSuchUser_pristine"}, "/")
        assert r["exit_code"] == 0  # the read still executed and returned an observation
        assert s.testbed_digest() == before  # ...but no file was left behind on the shadow
    finally:
        s.close()


@officebench_image
def test_officebench_sweep_reaps_only_its_run(monkeypatch):
    """The container sweeper force-removes only the containers of its own run-id (orphan safety net)."""
    from tts.env.officebench_lowlevel import OfficeBenchSession, sweep_officebench_containers

    monkeypatch.setenv("TTS_OFFICEBENCH_RUN_ID", "tts_sweeptest_A")
    a = OfficeBenchSession("1-1", "0", phase="sweepA").start()  # started with run-id A, not closed
    monkeypatch.setenv("TTS_OFFICEBENCH_RUN_ID", "tts_sweeptest_B")
    b = OfficeBenchSession("1-1", "0", phase="sweepB").start()
    try:
        # sweeping run B removes B's container but leaves A's
        assert sweep_officebench_containers("tts_sweeptest_B") == 1
        b._started = False  # already force-removed
        # A survives until swept explicitly
        assert sweep_officebench_containers("tts_sweeptest_A") == 1
        a._started = False
    finally:
        a.close()
        b.close()


@officebench_image
@needs_llm
def test_officebench_wm_rollout_leaves_testbed_pristine():
    """PROOF of write-safety: an ``llm_wm`` sampling rollout never mutates the shadow ``/testbed``.

    The predictive world model executes reads exactly but PREDICTS every write, so the shadow
    container's file tree must be byte-for-byte identical before and after the rollout — the real,
    non-trivial write-safety guarantee (``state_hash`` is a constant and cannot show this).
    """
    from tts.env.officebench_agent import run_officebench_agent
    from tts.env.officebench_lowlevel import OfficeBenchSession, load_subtask_config
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor
    from tts.llm import make_chat_client

    cfg = load_subtask_config("1-4", "0")  # an excel write task
    s = OfficeBenchSession("1-4", "0", phase="pristine", read_only=True).start()
    try:
        before = s.testbed_digest()
        predictor = OfficeBenchWMPredictor(make_chat_client())
        run_officebench_agent(s, cfg, make_chat_client(), max_steps=12, temperature=0.7,
                              predictor=predictor)
        after = s.testbed_digest()
        assert before == after and before  # shadow /testbed unchanged despite the agent "writing"
        assert predictor.n_writes >= 1  # the agent DID issue writes — they were predicted, not run
    finally:
        s.close()

