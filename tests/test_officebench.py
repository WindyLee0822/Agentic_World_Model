"""Guarded tests for the REAL OfficeBench integration.

OfficeBench runs its office apps + official evaluation inside a Docker image (built on demand from
``src/tts/env/officebench_docker``), driven from the host through :mod:`tts.env.officebench_lowlevel`.
The pure-logic tests (prompt/state-machine/driver helpers/verifier) need neither Docker nor an LLM and
always run; the live tests are skipped unless the OfficeBench clone (``OFFICEBENCH_ROOT``), a usable
Docker daemon, the built app image, and (for B0) an LLM endpoint are all available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tts.config import Config, SolverConfig
from tts.harness import run_decision
from tts.trajectory import RewardInfo, RolloutResult, Step, Trajectory
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


def test_transition_log_indexes_applied_and_rejected_effects():
    from tts.simulator.transition_log import TransitionLog

    log = TransitionLog()
    log.record({"action": "write"}, "ok", ["data/a.txt"], applied=True)
    log.record({"action": "write"}, "failed", ["data/b.txt"], applied=False)
    assert log.resources() == ["data/a.txt"]
    assert len(log.entries("data/a.txt", applied_only=True)) == 1
    assert "APPLIED" in log.render("data/a.txt")
    assert "REJECTED" in log.render("data/b.txt")


def test_officebench_wm_tool_mode_defaults_to_python(monkeypatch):
    from tts.env.officebench_wm_python import officebench_wm_tool_mode

    monkeypatch.delenv("TTS_OFFICEBENCH_WM_TOOL_MODE", raising=False)
    assert officebench_wm_tool_mode() == "python"
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "python")
    assert officebench_wm_tool_mode() == "python"
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    assert officebench_wm_tool_mode() == "text"


def test_officebench_comparative_parser():
    from tts.env.real_officebench import _parse_comparative_best

    assert _parse_comparative_best("BEST: 2\nREASON: complete", 4) == 2
    with pytest.raises(ValueError):
        _parse_comparative_best("BEST: 4", 4)


def test_officebench_comparative_reranks_ambiguous_nested_pools(monkeypatch):
    from tts.env.real_officebench import OfficeBenchTaskEnv

    calls = []
    responses = iter([
        "BEST: 1\nREASON: candidate 1 is more complete",
        "BEST: 3\nREASON: candidate 3 is more complete",
    ])

    class _FakeClient:
        def chat(self, system, user, **kwargs):
            calls.append((system, user, kwargs))
            return next(responses)

    monkeypatch.setenv("TTS_OFFICEBENCH_COMPARATIVE_RERANK", "1")
    monkeypatch.setattr("tts.llm.make_chat_client", lambda **kwargs: _FakeClient())
    env = OfficeBenchTaskEnv.__new__(OfficeBenchTaskEnv)
    env._instruction = "Create the requested artifact."
    env._config = {"evaluation": [{"function": "evaluate_exists"}]}
    rollouts = [
        RolloutResult(index, Trajectory(steps=[]), RewardInfo(reward=0.75))
        for index in range(2)
    ]

    env.refine_rollout_scores(
        Config(seed=7, solver=SolverConfig(llm_agent="gpt-5.4")),
        rollouts,
    )

    assert len(calls) == 1
    assert [rollout.predicted.reward for rollout in rollouts] == [0.75, 0.75]
    assert rollouts[1].predicted.metadata["comparative_rerank"]["best"] == 1
    assert rollouts[1].predicted.metadata["selection_override"]["best"] == 1

    rollouts.extend([
        RolloutResult(index, Trajectory(steps=[]), RewardInfo(reward=0.75))
        for index in range(2, 4)
    ])
    env.refine_rollout_scores(
        Config(seed=7, solver=SolverConfig(llm_agent="gpt-5.4")),
        rollouts,
    )

    assert len(calls) == 2
    assert [rollout.predicted.reward for rollout in rollouts] == [0.75] * 4
    decision = rollouts[3].predicted.metadata["comparative_rerank"]
    assert decision["incumbent"] == 1
    assert decision["contenders"] == [1, 2, 3]
    assert decision["best"] == 3


def test_officebench_comparative_always_audits_valid_incumbent(monkeypatch):
    """A noisy high-confidence challenger cannot replace the lower-N winner without comparison."""
    from tts.env.real_officebench import OfficeBenchTaskEnv

    prompts = []

    class _FakeClient:
        def chat(self, system, user, **kwargs):
            prompts.append(user)
            return "BEST: 3\nREASON: the incumbent preserves the exact required value"

    monkeypatch.setenv("TTS_OFFICEBENCH_COMPARATIVE_RERANK", "1")
    monkeypatch.setattr("tts.llm.make_chat_client", lambda **kwargs: _FakeClient())
    env = OfficeBenchTaskEnv.__new__(OfficeBenchTaskEnv)
    env._instruction = "Return the exact value from the document."
    env._config = {"evaluation": [{"function": "evaluate_contain"}]}
    rewards = [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]
    rollouts = [
        RolloutResult(index, Trajectory(steps=[]), RewardInfo(reward=reward))
        for index, reward in enumerate(rewards)
    ]
    rollouts[0].predicted.metadata["comparative_rerank"] = {
        "pool_size": 4,
        "best": 3,
    }

    env.refine_rollout_scores(
        Config(seed=7, solver=SolverConfig(llm_agent="gpt-5.4")),
        rollouts,
    )

    decision = rollouts[-1].predicted.metadata["comparative_rerank"]
    assert decision["incumbent"] == 3
    assert decision["native_best"] == 7
    assert decision["contenders"] == [3, 7]
    assert decision["best"] == 3
    assert "### Candidate 3" in prompts[0] and "### Candidate 7" in prompts[0]


def test_officebench_python_disabled_uses_direct_prompt(monkeypatch):
    from tts.env.officebench_wm_python import run_python_prediction_loop

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_PYTHON", "0")
    responses = iter([
        'PYTHON_FINAL:\nprint_json({"text": "not executed"})',
        "OBSERVATION: direct fallback",
    ])
    prompts = []

    def complete(prompt, _max_tokens):
        prompts.append(prompt)
        return next(responses)

    result = run_python_prediction_loop(
        complete,
        None,
        "Predict the observation.",
        "PREDICT: <observation>",
        context={"initial_content": "hello"},
    )

    assert result.valid and result.output == "OBSERVATION: direct fallback"
    assert result.python_calls == 0 and result.python_mode_misses == 1
    assert "PURE COMPUTATION OPTION" not in prompts[0]


def test_officebench_python_parser_uses_one_protocol_block():
    from tts.env.officebench_wm_python import _extract_python_action

    response = (
        'PYTHON_READ:\nprint_json({"queries": []})\n'
        'PYTHON_FINAL:\nprint_json({"text": "should not join programs"})'
    )
    mode, code = _extract_python_action(response)

    assert mode == "read"
    assert code == 'print_json({"queries": []})'


def test_officebench_python_retries_failed_repairs(monkeypatch):
    from tts.env.officebench_wm_python import run_python_prediction_loop

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_PYTHON_REPAIR_ATTEMPTS", "2")
    responses = iter([
        "PYTHON_FINAL:\nraise ValueError('first')",
        "not a program",
        'PYTHON_FINAL:\nprint_json({"text": data["value"]})',
    ])

    result = run_python_prediction_loop(
        lambda _prompt, _max_tokens: next(responses),
        None,
        "Predict the observation.",
        "PREDICT: <observation>",
        context={"value": "OBSERVATION: repaired"},
        force_always=True,
    )

    assert result.valid
    assert result.output == "OBSERVATION: repaired"
    assert result.python_calls == 2
    assert result.python_failures == 1


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
    from tts.env.officebench_wm_predict import _doc_type

    assert drv._simplify_path("/testbed", "data") == "/testbed/data"
    assert drv._simplify_path("/testbed/data", "..") == "/testbed"
    assert drv._simplify_path("/a/b", "/c") == "/c"
    assert drv._minor_action_fix({"user": ["Bob"], "x": [1, 2]}) == {"user": "Bob", "x": [1, 2]}
    assert _doc_type("/testbed/data/scan.JPG") == "image"
    assert _doc_type("/testbed/data/report.xlsx") == "xlsx"


def test_driver_reads_images_through_officebench_ocr(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tts.env import officebench_driver as drv

    image = tmp_path / "notice.png"
    image.write_bytes(b"not decoded directly")
    fake_reader = SimpleNamespace(ocr_recognize_file=lambda path: f"OCR:{path}")
    fake_apps = SimpleNamespace(
        ocr_app=SimpleNamespace(ocr_recognize_file=fake_reader),
    )
    monkeypatch.setitem(__import__("sys").modules, "apps", fake_apps)
    monkeypatch.setitem(__import__("sys").modules, "apps.ocr_app", fake_apps.ocr_app)

    assert drv._read_one_file(str(image), "image") == f"OCR:{image}"


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
    before_dirs = drv._list_dirs(str(root))
    # simulate a create-if-missing read helper: a new blank calendar file appears during the read
    (root / "calendar").mkdir()
    (root / "calendar" / "Bob.ics").write_text("")
    removed = drv._remove_new_files(str(root), before, before_dirs)

    assert str(root / "calendar" / "Bob.ics") in removed
    assert str(root / "calendar") in removed
    assert drv._list_files(str(root)) == before  # pristine again
    assert drv._list_dirs(str(root)) == before_dirs
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


def test_predict_all_session_refuses_every_action():
    from tts.env.officebench_lowlevel import OfficeBenchSession

    session = OfficeBenchSession.__new__(OfficeBenchSession)
    session.read_only = True
    session.actions_disabled = True

    with pytest.raises(PermissionError, match="predict-all"):
        session.exec_action(
            {"app": "shell", "action": "command", "command": "ls /testbed"},
            "/",
        )


def test_local_backend_is_available_without_docker(tmp_path, monkeypatch):
    from tts.env import officebench_lowlevel as lowlevel

    root = tmp_path / "OfficeBench"
    (root / "tasks").mkdir(parents=True)
    (root / "apps").mkdir()
    (root / "utils").mkdir()
    (root / "utils" / "evaluate.py").write_text("", encoding="utf-8")

    monkeypatch.setenv("TTS_OFFICEBENCH_BACKEND", "local")
    monkeypatch.setattr(lowlevel, "docker_available", lambda: False)

    assert lowlevel.officebench_backend() == "local"
    assert lowlevel.real_officebench_available(str(root))


def test_local_backend_materializes_and_cleans_task_without_docker(tmp_path, monkeypatch):
    from tts.env import officebench_lowlevel as lowlevel

    root = tmp_path / "OfficeBench"
    (root / "apps").mkdir(parents=True)
    (root / "apps" / "__init__.py").write_text("", encoding="utf-8")
    (root / "utils").mkdir()
    (root / "utils" / "evaluate.py").write_text("", encoding="utf-8")
    testbed = root / "tasks" / "1-1" / "testbed" / "data"
    testbed.mkdir(parents=True)
    (testbed / "input.txt").write_text("hello", encoding="utf-8")

    monkeypatch.setenv("TTS_OFFICEBENCH_BACKEND", "local")
    monkeypatch.setenv("TTS_OFFICEBENCH_LOCAL_TMP", str(tmp_path))
    monkeypatch.setattr(
        lowlevel,
        "ensure_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local backend must not touch Docker")),
    )

    session = lowlevel.OfficeBenchSession("1-1", "0", root=str(root)).start()
    local_root = session._local_root
    assert local_root is not None
    assert (
        os.path.join(local_root, "testbed", "data", "input.txt")
    )
    assert open(
        os.path.join(local_root, "testbed", "data", "input.txt"),
        encoding="utf-8",
    ).read() == "hello"

    session.close()
    assert not os.path.exists(local_root)


def test_local_backend_driver_uses_unprivileged_proot_binds(tmp_path, monkeypatch):
    import json
    import subprocess

    from tts.env import officebench_lowlevel as lowlevel

    local_root = tmp_path / "session"
    (local_root / "testbed").mkdir(parents=True)
    (local_root / "reference").mkdir()
    session = lowlevel.OfficeBenchSession.__new__(lowlevel.OfficeBenchSession)
    session._local_root = str(local_root)
    session.exec_timeout_s = 30.0
    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        payload = {"entries": [], "missing": False}
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"OFFICEBENCH_RESULT={json.dumps(payload)}\n",
            stderr="",
        )

    monkeypatch.setattr(
        lowlevel.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in ("proot", "tts-no-network") else None,
    )
    monkeypatch.setattr(lowlevel.subprocess, "run", fake_run)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "must-not-leak")

    assert session._driver_local("list_state", {"path": "/testbed"}) == {
        "entries": [],
        "missing": False,
    }
    command = calls["command"]
    assert command[:2] == ["/usr/bin/tts-no-network", "/usr/bin/proot"]
    assert command[command.index("-r") + 1] == str(local_root)
    assert "/officebench_driver.py" in command
    assert str(Path(__file__).parents[1]) not in command
    assert "/proc" not in command
    assert "AZURE_OPENAI_API_KEY" not in calls["kwargs"]["env"]


def test_local_backend_recursive_object_listing(tmp_path):
    from tts.env import officebench_driver as driver

    root = tmp_path / "testbed"
    nested = root / "data" / "nested"
    nested.mkdir(parents=True)
    (root / "data" / "top.txt").write_text("top", encoding="utf-8")
    (nested / "deep.txt").write_text("deep", encoding="utf-8")

    original = driver._TESTBED
    driver._TESTBED = str(root)
    try:
        result = driver.handle_list_state({
            "path": str(root / "data"),
            "recursive": True,
        })
    finally:
        driver._TESTBED = original

    assert [entry["path"] for entry in result["entries"]] == [
        str(nested),
        str(root / "data" / "top.txt"),
        str(nested / "deep.txt"),
    ]


def test_wm_action_classification():
    """Reads, inputs and outputs use the official action argument names."""
    from tts.env.officebench_wm_predict import (
        action_inputs,
        action_execution_kind,
        is_write_action,
        read_target,
        shell_command_kind,
        write_targets,
    )

    assert is_write_action({"app": "excel", "action": "set_cell", "file_path": "/testbed/data/s.xlsx"})
    assert not is_write_action({"app": "excel", "action": "read_file", "file_path": "./data/s.xlsx"})
    assert is_write_action({"app": "calendar", "action": "create_event", "user": "Bob"})
    assert not is_write_action({"app": "calendar", "action": "list_events", "username": "Bob"})
    assert not is_write_action({"app": "shell", "action": "command", "command": "ls /testbed/data"})
    assert not is_write_action({"app": "shell", "action": "command", "command": "cd /testbed/data"})
    assert is_write_action({"app": "shell", "action": "command",
                            "command": "cd /testbed/data && rm x.txt"})
    assert is_write_action({"app": "shell", "action": "command", "command": "cd"})
    assert is_write_action({"app": "shell", "action": "command",
                            "command": "ls /testbed/data; rm /testbed/data/x"})
    assert is_write_action({"app": "shell", "action": "command",
                            "command": "find /testbed -delete"})
    assert not is_write_action({
        "app": "shell", "action": "command",
        "command": "find /testbed/data -maxdepth 2 -type f 2>/dev/null | sort | sed -n '1,200p'",
    })
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "sed -f /testbed/data/script.sed /testbed/data/input.txt",
    }) == "blocked"
    assert not is_write_action({
        "app": "shell", "action": "command",
        "command": "ls -la /testbed/data && file /testbed/data/a.docx && wc -c /testbed/data/a.docx",
    })
    assert is_write_action({"app": "shell", "action": "command",
                            "command": "sort -o /testbed/data/out.txt /testbed/data/in.txt"})
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "sort -o /testbed/data/out.txt /testbed/data/in.txt",
    }) == "write"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "cat /testbed/data/in.txt | sort -o /testbed/data/out.txt",
    }) == "write"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "sort -ro /testbed/data/out.txt /testbed/data/in.txt",
    }) == "write"
    assert is_write_action({"app": "shell", "action": "command",
                            "command": "/testbed/data/ls /testbed"})
    assert is_write_action({"app": "shell", "action": "command",
                            "command": "cd `rm /testbed/data/x`"})
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "python -c \"import os; print(os.listdir('/testbed/data'))\"",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command", "command": "cat /reference/gold.xlsx",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command", "command": "cat ~/.bashrc",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "file --files-from=/reference/gold.txt",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "wc --files0-from=/reference/gold.txt",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command", "command": "cd /reference",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "grep -f /reference/patterns /testbed/data/in.txt",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "grep -Ff /reference/patterns /testbed/data/in.txt",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "grep --regexp=. /reference/gold.txt",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "sed --expression=p /reference/gold.txt",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command", "command": "grep -aR ''",
    }, "/") == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command", "command": "du",
    }, "/") == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "tail --follow /testbed/data/log.txt",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "find -L /reference -type f",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "cat /testbed/*/../../reference/gold.xlsx",
    }) == "blocked"
    assert shell_command_kind({
        "app": "shell", "action": "command",
        "command": "file -bC -m /testbed/data/magic",
    }) == "write"
    assert action_execution_kind({
        "app": "excel", "action": "read_file", "file_path": "/reference/gold.xlsx",
    }) == "blocked"
    assert action_execution_kind({
        "app": "llm", "action": "complete_text", "prompt": "Summarize this document.",
    }) == "read"
    assert action_execution_kind({
        "app": "llm", "action": "complete_text",
        "prompt": "''' ; cat /reference/gold.xlsx ; echo '''",
    }) == "write"
    assert write_targets({
        "app": "llm", "action": "complete_text",
        "prompt": "''' ; cat /reference/gold.xlsx ; echo '''",
    }) == []
    assert is_write_action({"app": "shell", "action": "command", "command": "rm /testbed/data/x"})
    assert is_write_action({"app": "shell", "action": "command", "command": "echo hi > /testbed/o.txt"})
    assert not is_write_action({"app": "system", "action": "finish_task", "answer": "None"})
    assert write_targets({"app": "excel", "action": "set_cell", "file_path": "./data/s.xlsx"}) == ["data/s.xlsx"]
    assert read_target({"app": "excel", "action": "read_file", "file_path": "/testbed/data/s.xlsx"}) == "data/s.xlsx"
    word_pdf = {"app": "word", "action": "convert_to_pdf",
                "word_file_path": "/testbed/data/a.docx", "pdf_file_path": "/testbed/data/a.pdf"}
    assert action_inputs(word_pdf) == ["data/a.docx"]
    assert write_targets(word_pdf) == ["data/a.pdf"]
    pdf_image = {"app": "pdf", "action": "convert_to_image",
                 "pdf_file_path": "/testbed/data/a.pdf", "image_file_path": "/testbed/data/a.jpg"}
    assert action_inputs(pdf_image) == ["data/a.pdf"]
    assert write_targets(pdf_image) == ["data/a.jpg"]
    assert write_targets({"app": "shell", "action": "command", "command": "rm x.txt"},
                         "/testbed/data") == ["data/x.txt"]
    assert write_targets({"app": "shell", "action": "command",
                          "command": "rm /testbed/data/x.txt && touch /testbed/data/x.txt"}) == ["<shell>"]
    assert write_targets({"app": "shell", "action": "command",
                          "command": "mv /testbed/data/old.txt /testbed/data/new.txt"}) == [
                              "data/new.txt"]
    assert write_targets({"app": "shell", "action": "command",
                          "command": "cp /testbed/data/old.txt /testbed/data/new.txt"}) == [
                              "data/new.txt"]
    assert write_targets({"app": "shell", "action": "command",
                          "command": "cp /testbed/data/old.txt /testbed/data/"}) == [
                              "data/old.txt"]
    assert write_targets({"app": "shell", "action": "command",
                          "command": "cp -rt /testbed/data/dest /testbed/data/src"}) == [
                              "data/dest/src"]
    assert write_targets({"app": "shell", "action": "command",
                          "command": "cp --target-directory=/testbed/data/dest /testbed/data/src"}) == [
                              "data/dest/src"]
    assert write_targets({"app": "shell", "action": "command",
                          "command": "sort -o /testbed/data/out.txt /testbed/data/in.txt"}) == [
                              "data/out.txt"]
    assert write_targets({"app": "shell", "action": "command",
                          "command": "sort -ro /testbed/data/out.txt /testbed/data/in.txt"}) == [
                              "data/out.txt"]
    assert write_targets({"app": "shell", "action": "command",
                          "command": "printf x >> /testbed/data/a.txt"}) == ["data/a.txt"]
    assert read_target({"app": "pdf", "action": "read_file",
                        "pdf_file_path": "/testbed/data/a.pdf"}) == "data/a.pdf"
    assert read_target({"app": "shell", "action": "command", "command": "cat report.txt"},
                       "/testbed/data") == "data/report.txt"
    assert read_target({"app": "shell", "action": "command", "command": "cat < report.txt"},
                       "/testbed/data") == "data/report.txt"
    assert read_target({
        "app": "shell", "action": "command",
        "command": "grep -f patterns.txt input.txt",
    }, "/testbed/data") == "data"
    assert read_target({"app": "shell", "action": "command",
                        "command": "cat a.txt b.txt"}, "/testbed/data") == "data"
    assert read_target({
        "app": "shell", "action": "command",
        "command": "find /testbed/data -type f 2>/dev/null | sort | sed -n '1,20p'",
    }, "/") == "data"
    assert shell_command_kind({
        "app": "shell", "action": "command", "command": "ls /testbed/data/*.txt",
    }, "/") == "blocked"
    assert read_target({"app": "shell", "action": "command", "command": "ls ."},
                       "/testbed/data") == "data"
    assert read_target({"app": "shell", "action": "command", "command": "ls /testbed"},
                       "/") == ""


def test_wm_query_exposes_only_read_only_testbed_tools():
    """READ/LIST/CALL inspect testbed state and reject writes or paths outside it."""
    from tts.env.officebench_wm_predict import OfficeBenchWMQuery, _parse_agentic

    class _Session:
        def __init__(self):
            self.exec_calls = []

        def apps_catalogue(self):
            return {
                "actions": {
                    "excel": ["read_file", "set_cell"],
                    "shell": ["command"],
                    "system": ["finish_task"],
                }
            }

        def read_state(self, files):
            return {files[0]["path"]: "(1, 1): 10"}

        def exec_action(self, action, workdir):
            self.exec_calls.append((action, workdir))
            return {"observation": "read result"}

    session = _Session()
    q = OfficeBenchWMQuery(session, lambda: "/testbed/data")
    assert q.available_calls == ["excel.read_file"]
    assert q.read("./s.xlsx") == "(1, 1): 10"
    assert q.list("/testbed/data") == "read result"
    assert q.call("excel.read_file", {"file_path": "./s.xlsx"}) == "read result"
    assert session.exec_calls[-1][0]["file_path"] == "/testbed/data/s.xlsx"
    assert "requires one JSON object" in q.call("excel.read_file", ["not", "an", "object"])
    assert _parse_agentic('CALL excel.read_file ["not", "an", "object"]')[2] == [
        ("excel.read_file", None)]
    assert "not an available read-only tool" in q.call(
        "excel.set_cell", {"file_path": "./data/s.xlsx", "row_idx": 1, "column_idx": 1})
    assert "unsafe argument characters" in q.call(
        "excel.read_file", {"file_path": "s.xlsx; rm /testbed/data/s.xlsx"})
    assert "unsafe argument characters" in q.call(
        "excel.read_file", {"file_path": "s.xlsx", "sheet_name": "x; rm /testbed/data/s.xlsx"})
    assert "unsafe argument characters" in q.call(
        "excel.read_file", {"file_path": "s.xlsx", "username": "../../reference/probe"})
    assert "only files under /testbed" in q.read("/reference/gold.xlsx")
    assert "only directories under /testbed" in q.list("../../reference")
    assert all(call[0]["action"] != "set_cell" for call in session.exec_calls)


def test_wm_query_predict_all_reads_objects_without_executing_actions():
    """Predict-all exposes immutable objects as variables and disables app/shell action execution."""
    from tts.env.officebench_wm_predict import OfficeBenchWMQuery

    class _Session:
        def __init__(self):
            self.exec_calls = []

        def apps_catalogue(self):
            return {"actions": {"excel": ["read_file", "set_cell"], "shell": ["command"]}}

        def read_state(self, files):
            return {files[0]["path"]: "(1, 1): 10"}

        def list_state(self, path):
            return {
                "entries": [
                    {"name": "s.xlsx", "path": f"{path}/s.xlsx", "type": "file"},
                ],
                "missing": False,
            }

        def exec_action(self, action, workdir):
            self.exec_calls.append((action, workdir))
            raise AssertionError("predict-all must not execute OfficeBench actions")

    session = _Session()
    query = OfficeBenchWMQuery(
        session,
        lambda: "/testbed/data",
        allow_action_execution=False,
    )

    assert query.available_calls == []
    assert query.read("./s.xlsx") == "(1, 1): 10"
    listing = query.list("/testbed/data")
    assert '"name": "s.xlsx"' in listing
    assert "disabled when every action must be predicted" in query.call(
        "excel.read_file", {"file_path": "./s.xlsx"})
    assert session.exec_calls == []


def test_wm_query_tools_are_overlay_aware():
    """READ/LIST/CALL/TRACE expose predicted resources instead of stale pristine absence."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor, OfficeBenchWMQuery

    class _Session:
        def apps_catalogue(self):
            return {"actions": {"pdf": ["read_file"], "word": ["convert_to_pdf"]}}

        def read_state(self, files):
            path = files[0]["path"]
            content = "SOURCE DOCUMENT" if path.endswith(".docx") else "[file does not exist]"
            return {path: content}

        def exec_action(self, action, workdir):
            return {"observation": "base listing"}

    session = _Session()
    predictor = OfficeBenchWMPredictor(
        _StubChat("PREDICT: OBSERVATION: Successfully converted the file."))
    query = OfficeBenchWMQuery(session, lambda: "/")
    predictor.bind_query_interface(query)
    predictor.predict_write({
        "app": "word",
        "action": "convert_to_pdf",
        "word_file_path": "/testbed/data/a.docx",
        "pdf_file_path": "/testbed/data/a.pdf",
    })

    logical = query.read("/testbed/data/a.pdf")
    assert "LOGICAL OVERLAY RESOURCE: data/a.pdf" in logical
    assert "data/a.docx" in logical and "convert_to_pdf" in logical
    assert "data/a.pdf" in query.list("/testbed/data")
    assert "APPLIED" in query.trace("/testbed/data/a.pdf")
    assert "LOGICAL OVERLAY RESOURCE" in query.call(
        "pdf.read_file", {"pdf_file_path": "/testbed/data/a.pdf"})


def test_wm_relative_paths_follow_current_workdir():
    """Action footprints and query tools resolve relative paths from the agent's current directory."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor, OfficeBenchWMQuery

    class _Session:
        def apps_catalogue(self):
            return {"actions": {"excel": ["read_file", "set_cell"]}}

        def read_state(self, files):
            return {files[0]["path"]: "(1, 1): old"}

        def exec_action(self, action, workdir):
            return {"observation": "exact read"}

    predictor = OfficeBenchWMPredictor(_StubChat("OBSERVATION: Successfully set the cell."))
    predictor.bind_query_interface(OfficeBenchWMQuery(_Session(), lambda: "/testbed/data"))
    predictor.predict_write({
        "app": "excel", "action": "set_cell", "file_path": "scores.xlsx",
        "row_idx": 1, "column_idx": 1, "text": "new",
    })

    assert "data/scores.xlsx" in predictor.applied_writes
    assert "scores.xlsx" not in predictor.applied_writes
    assert predictor.was_written("data")
    assert predictor.was_written("")


def test_wm_delete_tombstone_overrides_pristine_state():
    """A successful predicted delete hides the pristine file from READ/LIST until a later write."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor, OfficeBenchWMQuery

    class _Session:
        def __init__(self):
            self.exec_calls = 0

        def apps_catalogue(self):
            return {"actions": {"shell": ["command"]}}

        def read_state(self, files):
            return {files[0]["path"]: "PRISTINE"}

        def exec_action(self, action, workdir):
            self.exec_calls += 1
            return {"observation": "s.xlsx\nother.xlsx"}

    predictor = OfficeBenchWMPredictor(_StubChat("OBSERVATION: Successfully executed shell.command."))
    session = _Session()
    query = OfficeBenchWMQuery(session, lambda: "/testbed/data")
    predictor.bind_query_interface(query)
    predictor.predict_write({"app": "shell", "action": "command", "command": "rm s.xlsx"})

    assert "data/s.xlsx" in predictor.deleted_paths
    assert "[file does not exist]" in query.read("s.xlsx")
    listing = query.list(".")
    assert "PREDICTED DELETIONS" in listing and "data/s.xlsx" in listing
    assert "data/s.xlsx" not in predictor.overlay_paths()

    predictor.predict_write({
        "app": "shell", "action": "command", "command": "rm -r archive",
    })
    assert predictor.was_written("data/archive/old.xlsx")
    assert "[file does not exist]" in predictor.logical_read("data/archive/old.xlsx", "PRISTINE")
    calls_before = session.exec_calls
    assert "[directory does not exist]" in query.list("archive/subdir")
    assert session.exec_calls == calls_before

    predictor.predict_write({
        "app": "word", "action": "create_new_file",
        "file_path": "/testbed/data/archive/new.docx",
    })
    assert predictor.is_deleted("data/archive")  # Word does not create missing parent dirs
    assert not predictor.is_deleted("data/archive/new.docx")
    revived_listing = query.list("archive")
    assert "[directory does not exist]" in revived_listing
    assert session.exec_calls == calls_before

    predictor.predict_write({
        "app": "shell", "action": "command", "command": "python3 restore.py",
    })
    assert predictor.global_dirty and not predictor.deleted_paths
    assert "current state may differ" in predictor.logical_read("data/s.xlsx", "PRISTINE")


def test_wm_move_tombstones_source(monkeypatch):
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    predictor = OfficeBenchWMPredictor(
        _StubChat("PREDICT: OBSERVATION: Successfully moved the file."))
    predictor.predict_write({
        "app": "shell",
        "action": "command",
        "command": "mv /testbed/data/old.txt /testbed/data/new.txt",
    })

    assert predictor.is_deleted("data/old.txt")
    assert predictor.was_written("data/old.txt")
    assert predictor.was_written("data/new.txt")
    assert "data/old.txt" not in predictor.overlay_paths()


def test_wm_opaque_write_qualifies_pristine_reads():
    """After an unattributed write, logical READ labels pristine content as potentially stale."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    predictor = OfficeBenchWMPredictor(_StubChat("OBSERVATION: ok"))
    predictor.predict_write({"app": "shell", "action": "command", "command": "python3 munge.py"})
    logical = predictor.logical_read("data/s.xlsx", "OLD")
    assert "PRISTINE CONTENT (current state may differ" in logical
    assert "python3 munge.py" in logical

    predictor.applied_writes["data/s.xlsx"] = [
        ({"app": "excel", "action": "set_cell", "row_idx": 1, "column_idx": 1, "text": "new"}, "ok")]
    assert "current state may differ" in predictor.logical_read("data/s.xlsx", "(1, 1): old")

    compound = OfficeBenchWMPredictor(_StubChat("OBSERVATION: ok"))
    compound.predict_write({
        "app": "shell", "action": "command",
        "command": "rm /testbed/data/x.txt && touch /testbed/data/x.txt",
    })
    assert compound.global_dirty and not compound.deleted_paths


def test_wm_directory_copy_dirties_descendants():
    """A copied directory makes every destination descendant part of the logical overlay."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    predictor = OfficeBenchWMPredictor(_StubChat("OBSERVATION: Successfully copied directory."))
    predictor.predict_write({
        "app": "shell", "action": "command",
        "command": "cp -r /testbed/data/src /testbed/data/dest/",
    })

    assert "data/dest/src" in predictor.applied_writes
    assert predictor.was_written("data/dest/src/report.txt")
    assert "cp -r" in predictor.file_write_log("data/dest/src/report.txt")


def test_wm_exact_delete_and_answer_bypass_llm_reconstruction():
    """Exact overlay transitions stay exact in dirty reads and final-state rendering."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    class _NoPrediction:
        def chat(self, *args, **kwargs):
            raise AssertionError("exact overlay state must not call the LLM")

    predictor = OfficeBenchWMPredictor(_NoPrediction())
    predictor.applied_writes["data/old.txt"] = [
        ({"app": "shell", "action": "command", "command": "rm /testbed/data/old.txt"}, "ok")]
    predictor.deleted_paths.add("data/old.txt")
    assert predictor.predict_read(
        {"app": "word", "action": "read_file", "file_path": "/testbed/data/old.txt"},
        "PRISTINE",
    ) == "OBSERVATION: The file /testbed/data/old.txt does not exist. Failed to read the file."
    assert predictor.predict_file_state("data/old.txt", "PRISTINE") == "[file does not exist]"

    predictor.record_answer('O\'Brien said "done"')
    assert predictor.predict_file_state(
        "data/answer.txt", "[file does not exist]") == "OBrien said done\n"


class _GroundingQuery:
    """In-memory READ/LIST/CALL/TRACE stub for the agentic WM loop."""

    available_calls = ["excel.read_file"]

    def __init__(self):
        self.read_paths = []
        self.list_paths = []
        self.tool_calls = []
        self.trace_paths = []

    def read(self, path):
        self.read_paths.append(path)
        return f"content of {path}"

    def list(self, path):
        self.list_paths.append(path)
        return "a.xlsx\nb.xlsx"

    def call(self, name, args):
        self.tool_calls.append((name, args))
        return "OBSERVATION: exact app read"

    def trace(self, path):
        self.trace_paths.append(path)
        return "APPLIED write"


def test_wm_agentic_loop_uses_read_list_call_before_prediction(monkeypatch):
    """The OfficeBench WM can gather evidence over multiple turns before predicting a write."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    chat = _ScriptChat([
        "LIST /testbed/data",
        "READ /testbed/data/s.xlsx",
        'CALL excel.read_file {"file_path": "/testbed/data/s.xlsx"}',
        "TRACE /testbed/data/s.xlsx",
        "PREDICT: OBSERVATION: Successfully set the cell.",
    ])
    query = _GroundingQuery()
    predictor = OfficeBenchWMPredictor(chat)
    predictor.bind_query_interface(query)
    obs = predictor.predict_write({
        "app": "excel", "action": "set_cell", "file_path": "/testbed/data/s.xlsx",
        "row_idx": 1, "column_idx": 1, "text": 5,
    })

    assert obs == "OBSERVATION: Successfully set the cell."
    assert query.list_paths == ["/testbed/data"]
    assert query.read_paths == ["/testbed/data/s.xlsx"]
    assert query.tool_calls == [
        ("excel.read_file", {"file_path": "/testbed/data/s.xlsx"})]
    assert query.trace_paths == ["/testbed/data/s.xlsx"]
    assert chat.calls == 5


def test_wm_predict_all_clean_read_uses_prompted_world_model(monkeypatch):
    """A clean agent read is grounded by a tool call, then generated by the world model."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    chat = _ScriptChat([
        'CALL excel.read_file {"file_path":"/testbed/data/s.xlsx"}',
        "PREDICT: OBSERVATION: generated app read",
    ])
    query = _GroundingQuery()
    predictor = OfficeBenchWMPredictor(chat)
    predictor.bind_query_interface(query)

    observation = predictor.predict_read(
        {
            "app": "excel",
            "action": "read_file",
            "file_path": "/testbed/data/s.xlsx",
        },
        None,
        dirty=False,
    )

    assert observation == "OBSERVATION: generated app read"
    assert query.tool_calls == [
        ("excel.read_file", {"file_path": "/testbed/data/s.xlsx"})]
    assert predictor.n_predicted_reads == 1
    assert predictor.n_dirty_reads == 0
    assert chat.calls == 2
    assert "HIGHEST-PRIORITY CLEAN-READ RULE" in chat.prompts[0]


def test_wm_clean_read_python_protocol_copies_host_observation():
    from tts.env.officebench_wm_python import _python_protocol

    prompt = _python_protocol(
        {
            "grounded_clean_read_required": True,
            "grounded_clean_read": "OBSERVATION: exact",
        },
        required=False,
        context_chars=100,
        output_kind="text",
        allow_reads=True,
        allow_calls=False,
    )

    assert "EXACT GROUNDED CLEAN-READ PROTOCOL" in prompt
    assert "Return a PYTHON_FINAL program" in prompt
    assert "data['grounded_clean_read']" in prompt
    assert '"op":"call"' not in prompt


def test_wm_agentic_loop_forces_prediction_after_read_budget(monkeypatch):
    """As in τ², the read-turn limit is followed by one forced prediction call."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    chat = _ScriptChat([
        "READ /testbed/data/a.xlsx",
        "READ /testbed/data/b.xlsx",
        "PREDICT: OBSERVATION: write result",
    ])
    query = _GroundingQuery()
    predictor = OfficeBenchWMPredictor(chat)
    predictor.MAX_TURNS = 2
    predictor.bind_query_interface(query)
    obs = predictor.predict_write({"app": "word", "action": "write_to_file",
                                   "file_path": "/testbed/data/out.docx", "contents": "x"})

    assert query.read_paths == ["/testbed/data/a.xlsx", "/testbed/data/b.xlsx"]
    assert obs == "OBSERVATION: write result"
    assert chat.calls == 3


def test_wm_python_loop_uses_copied_read_results(monkeypatch):
    """Python mode requests host-mediated reads and computes the final observation from copied data."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "python")
    chat = _ScriptChat([
        (
            "PYTHON_READ:\n"
            'print_json({"queries": [{"op": "read", "path": "/testbed/data/s.xlsx"}]})'
        ),
        (
            "PYTHON_FINAL:\n"
            'print_json({"text": "OBSERVATION: Successfully set the cell."})'
        ),
    ])
    query = _GroundingQuery()
    predictor = OfficeBenchWMPredictor(chat)
    predictor.bind_query_interface(query)
    obs = predictor.predict_write({
        "app": "excel", "action": "set_cell", "file_path": "/testbed/data/s.xlsx",
        "row_idx": 1, "column_idx": 1, "text": 5,
    })

    assert obs == "OBSERVATION: Successfully set the cell."
    assert query.read_paths == ["/testbed/data/s.xlsx"]
    assert predictor.n_python_calls == 2
    assert predictor.n_python_failures == 0
    assert "content of /testbed/data/s.xlsx" not in chat.prompts[1]
    assert "data['reads'][0]['result']" in chat.prompts[1]


def test_wm_python_forces_long_file_transform(monkeypatch):
    """Long copied state rejects direct regeneration and requires a PYTHON_FINAL transform."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "python")
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_PYTHON_FORCE_CHARS", "10")
    initial = "  unchanged-\ntrailing  "
    chat = _ScriptChat([
        "DIRECT OUTPUT IS NOT ACCEPTED",
        'PYTHON_FINAL:\nprint_json({"text": data["initial_content"]})',
    ])
    predictor = OfficeBenchWMPredictor(chat)
    predictor.applied_writes["data/out.txt"] = [(
        {
            "app": "shell",
            "action": "command",
            "command": "cp /testbed/data/in.txt /testbed/data/out.txt",
        },
        "OBSERVATION: copied",
    )]

    assert predictor.predict_file_state("data/out.txt", initial) == initial
    assert predictor.n_python_forced == 1
    assert predictor.n_python_mode_misses == 1
    assert predictor.n_python_calls == 1


def test_wm_program_only_invalid_python_is_a_rollout_error(monkeypatch):
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor, OfficeBenchSimulationError

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "python")
    predictor = OfficeBenchWMPredictor(_ScriptChat([
        "not a Python program",
        "still not a Python program",
    ]))
    predictor.program_only = True

    with pytest.raises(OfficeBenchSimulationError, match="PYTHON_FINAL"):
        predictor.predict_write({
            "app": "excel",
            "action": "set_cell",
            "file_path": "/testbed/data/s.xlsx",
            "row_idx": 1,
            "column_idx": 1,
            "text": "new",
        })

    assert predictor.n_failed_writes == 0
    assert predictor.write_history == []
    assert predictor.transition_log.entries() == []
    assert not predictor.was_written("data/s.xlsx")


def test_wm_python_preserves_valid_empty_file_results(monkeypatch):
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "python")
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_PYTHON_FORCE_CHARS", "10")
    action = {
        "app": "shell",
        "action": "command",
        "command": "cp /testbed/data/in.txt /testbed/data/out.txt",
    }

    final_predictor = OfficeBenchWMPredictor(_ScriptChat([
        "DIRECT OUTPUT IS NOT ACCEPTED",
        'PYTHON_FINAL:\nprint_json({"text": ""})',
    ]))
    final_predictor.applied_writes["data/out.txt"] = [(action, "OBSERVATION: copied")]
    assert final_predictor.predict_file_state("data/out.txt", "non-empty initial") == ""

    read_predictor = OfficeBenchWMPredictor(_ScriptChat([
        "DIRECT OUTPUT IS NOT ACCEPTED",
        'PYTHON_FINAL:\nprint_json({"text": ""})',
    ]))
    read_predictor.applied_writes["data/out.txt"] = [(action, "OBSERVATION: copied")]
    assert read_predictor.predict_read(
        {"app": "word", "action": "read_file", "file_path": "/testbed/data/out.txt"},
        "non-empty initial",
    ) == ""


def test_predict_all_shell_inventory_is_typed_for_generated_python(monkeypatch):
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor, OfficeBenchWMQuery

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "python")

    class _Session:
        def apps_catalogue(self):
            return {"actions": {"shell": ["command"]}}

        def list_state(self, path, *, recursive=False):
            assert recursive
            return {
                "entries": [
                    {
                        "name": "CS161 Final.xlsx",
                        "path": f"{path}/CS161 Final.xlsx",
                        "type": "file",
                    },
                ],
                "missing": False,
            }

    predictor = OfficeBenchWMPredictor(_ScriptChat([
        (
            "PYTHON_FINAL:\n"
            "records = data['immutable_objects']\n"
            "stages = data['shell_stages']\n"
            "print_json({'text': records[0]['path'] if stages[0][0] == 'find' else ''})"
        ),
    ]))
    predictor.program_only = True
    predictor.bind_query_interface(OfficeBenchWMQuery(
        _Session(),
        lambda: "/testbed",
        allow_action_execution=False,
    ))

    observation = predictor.predict_read(
        {
            "app": "shell",
            "action": "command",
            "command": "find /testbed/data -type f | sort",
        },
        None,
        dirty=False,
    )

    assert observation == "/testbed/data/CS161 Final.xlsx"
    assert predictor.n_python_failures == 0


@pytest.mark.parametrize("workdir", ["/", "/testbed", "/testbed/data"])
def test_predict_all_shell_inventory_emulates_common_discovery_pipeline(monkeypatch, workdir):
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor, OfficeBenchWMQuery

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "python")

    class _Session:
        def apps_catalogue(self):
            return {"actions": {"shell": ["command"]}}

        def list_state(self, path, *, recursive=False):
            assert recursive
            assert path == "/testbed/data"
            return {
                "entries": [
                    {"name": "notes.txt", "path": f"{path}/notes.txt", "type": "file"},
                    {
                        "name": "meeting_notification.jpg",
                        "path": f"{path}/meeting_notification.jpg",
                        "type": "file",
                    },
                    {"name": "nested", "path": f"{path}/nested", "type": "directory"},
                ],
                "missing": False,
            }

    predictor = OfficeBenchWMPredictor(_ScriptChat([]))
    predictor.program_only = True
    predictor.bind_query_interface(OfficeBenchWMQuery(
        _Session(),
        lambda: workdir,
        allow_action_execution=False,
    ))

    observation = predictor.predict_read(
        {
            "app": "shell",
            "action": "command",
            "command": (
                "find /testbed/data -maxdepth 3 -type f | sort | "
                "grep -iE '\\.(png|jpg)$|meeting' | head -20"
            ),
        },
        None,
        dirty=False,
    )

    assert observation == "/testbed/data/meeting_notification.jpg"
    assert predictor.n_python_calls == 0


def test_readonly_shell_allows_quoted_regex_anchor_but_blocks_expansion():
    from tts.env.officebench_wm_predict import (
        _emulate_inventory_pipeline,
        _readonly_shell_stages,
    )

    assert _readonly_shell_stages(
        "find /testbed/data -type f | grep -E '\\.xlsx$' | sort"
    )
    assert _readonly_shell_stages(
        "find /testbed/data -type f | grep '\\.png\\|\\.jpg$' | sort"
    )
    assert _readonly_shell_stages(
        'find /testbed/data -type f | grep -E "\\.xlsx$" | sort'
    )
    assert _readonly_shell_stages("echo $HOME") is None
    assert _readonly_shell_stages('echo "$HOME"') is None
    assert _readonly_shell_stages("echo `pwd`") is None
    assert _readonly_shell_stages(
        "find /testbed/data -type f || grep x"
    ) is None
    assert _readonly_shell_stages(
        "sort --files0-from=/testbed/data/names"
    ) is None
    assert _readonly_shell_stages(
        "wc --files0-from=/testbed/data/names"
    ) is None
    objects = [
        {"name": "meeting.jpg", "path": "/testbed/data/meeting.jpg", "type": "file"},
        {"name": "other.jpg", "path": "/testbed/data/other.jpg", "type": "file"},
    ]
    assert _emulate_inventory_pipeline(
        [["find", "/testbed/data", "-type", "f", "-name", "*.jpg", "-name", "meeting*"]],
        objects,
    ) == "/testbed/data/meeting.jpg"
    assert _emulate_inventory_pipeline(
        [["ls", "-la", "/testbed/data"]],
        objects,
    ) is None
    assert _emulate_inventory_pipeline(
        [["find", "/testbed/data", "-type", "f"], ["grep", "-F", ".jpg"]],
        objects,
    ) == "/testbed/data/meeting.jpg\n/testbed/data/other.jpg"
    assert _emulate_inventory_pipeline(
        [["find", ".", "-type", "f"]],
        objects,
        workdir="/testbed/data",
    ) == "/testbed/data/meeting.jpg\n/testbed/data/other.jpg"


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
        self.prompts = []

    def chat(self, system, user, **kw):
        self.calls += 1
        self.prompts.append(user)
        return self._responses.pop(0) if self._responses else repr({"app": "system", "action": "got_stuck"})


def test_parse_action_normalizes_python_ellipsis():
    from tts.env.officebench_agent import _parse_action

    action = _parse_action(
        "{'app': 'excel', 'action': 'set_cell', 'text': ..., "
        "'nested': {'values': [1, ...]}}"
    )

    assert action == {
        "app": "excel",
        "action": "set_cell",
        "text": "...",
        "nested": {"values": [1, "..."]},
    }


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
    run = run_officebench_agent(
        session, cfg, agent, max_steps=6, predictor=predictor,
        predict_all_actions=False,
    )

    # the write was predicted + recorded, never executed; only the read hit exec_action
    assert predictor.n_writes == 1 and "data/s.xlsx" in predictor.applied_writes
    assert all(a.get("action") != "set_cell" for a in session.exec_calls)
    assert any(a.get("action") == "read_file" for a in session.exec_calls)
    # finish_task did NOT write answer.txt to the shadow (predicted only)
    assert session.write_answer_calls == [] and run.final_answer == "done"
    assert "data/answer.txt" in predictor.applied_writes
    assert predictor.logical_read("data/answer.txt", "[file does not exist]") == "done\n"
    assert "finish_task" in predictor.trace_text("data/answer.txt")


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
    run = run_officebench_agent(
        session, cfg, agent, max_steps=6, predictor=predictor,
        predict_all_actions=False,
    )
    assert predictor.dirty_read_called  # the read-after-write went through predict_read
    assert "predicted post-write" in run.observations[-2]  # the dirty-read observation


def test_wm_cd_executes_exactly_even_after_opaque_write():
    """Directory changes are read-only control flow and remain executable after global-dirty writes."""
    from tts.env.officebench_agent import run_officebench_agent
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    class _CdSession(_RecordingSession):
        def exec_action(self, action, workdir):
            result = super().exec_action(action, workdir)
            if action.get("command") == "cd /testbed/data":
                result.update({"is_cd": True, "new_workdir": "/testbed/data"})
            return result

    agent = _ScriptChat([
        repr({"app": "system", "action": "switch_app", "target_app": "shell"}),
        repr({"app": "shell", "action": "command",
              "command": "python3 -c \"open('/testbed/data/x.txt', 'w').write('x')\""}),
        repr({"app": "shell", "action": "command", "command": "cd /testbed/data"}),
        repr({"app": "system", "action": "finish_task", "answer": "None"}),
    ])
    predictor = OfficeBenchWMPredictor(_StubChat("OBSERVATION: ok"))
    session = _CdSession(read_only=True)
    cfg = {"username": "B", "date": "d", "weekday": "w", "time": "t", "task": "t", "evaluation": []}
    run_officebench_agent(
        session, cfg, agent, max_steps=6, predictor=predictor,
        predict_all_actions=False,
    )

    assert predictor.global_dirty
    assert any(a.get("command") == "cd /testbed/data" for a in session.exec_calls)


def test_wm_blocks_ambiguous_shell_inspection_without_dirtying_state():
    """Unprovable inspection is neither executed nor recorded as a predicted write."""
    from tts.env.officebench_agent import run_officebench_agent
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    command = "python -c \"import os; print(os.listdir('/testbed/data'))\""
    agent = _ScriptChat([
        repr({"app": "system", "action": "switch_app", "target_app": "shell"}),
        repr({"app": "shell", "action": "command", "command": command}),
        repr({"app": "system", "action": "finish_task", "answer": "None"}),
    ])
    predictor = OfficeBenchWMPredictor(_StubChat("OBSERVATION: should not be used"))
    session = _RecordingSession(read_only=True)
    cfg = {"username": "B", "date": "d", "weekday": "w", "time": "t", "task": "t", "evaluation": []}
    run = run_officebench_agent(session, cfg, agent, max_steps=5, predictor=predictor)

    assert all(a.get("command") != command for a in session.exec_calls)
    assert not predictor.global_dirty and predictor.n_writes == 0
    assert any("cannot be proven read-only" in observation for observation in run.observations)


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


def test_predicted_state_gold_backed_materializes_exact_edits_without_llm():
    """Materialize exact edits to gold-backed files without asking an LLM to rewrite them."""
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
    assert entry == "(1, 1): Alice\t(1, 2): 95\t\n"


def test_predicted_state_gold_backed_ignores_global_dirty_reconstruction():
    from tts.env.real_officebench import OfficeBenchTaskEnv
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    class _FakeSession:
        def read_state(self, files):
            return {f["path"]: "(1, 1): Alice" for f in files}

    class _P(OfficeBenchWMPredictor):
        def __init__(self):
            super().__init__(_StubChat("x"))
            self.reconstructed = False

        def predict_file_state(self, path, init):
            self.reconstructed = True
            return "FABRICATED"

    predictor = _P()
    predictor.global_dirty = True
    predictor.applied_writes["data/result.xlsx"] = [
        ({"app": "shell", "action": "command", "command": "python3 transform.py"},
         "OBSERVATION: Successfully executed.")]

    out = OfficeBenchTaskEnv._predicted_state(
        None,
        _FakeSession(),
        predictor,
        [{"path": "./data/result.xlsx", "doc_type": "xlsx"}],
        gold_labels={"result.xlsx"},
    )

    assert not predictor.reconstructed
    assert "INITIAL content" in out["./data/result.xlsx"]
    assert "transform.py" in out["./data/result.xlsx"]


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


def test_predicted_state_respects_recursive_tombstones():
    """Verifier input routes files below a deleted directory through the logical overlay."""
    from tts.env.real_officebench import OfficeBenchTaskEnv
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    class _FakeSession:
        def read_state(self, files):
            return {f["path"]: "PRISTINE" for f in files}

    predictor = OfficeBenchWMPredictor(_StubChat("OBSERVATION: Successfully removed directory."))
    predictor.predict_write({
        "app": "shell", "action": "command", "command": "rm -r /testbed/data/archive",
    })
    files = [{"path": "./data/archive/old.txt", "doc_type": "txt"}]
    out = OfficeBenchTaskEnv._predicted_state(
        None, _FakeSession(), predictor, files, gold_labels={"old.txt"})
    assert out["./data/archive/old.txt"] == "[file does not exist]"


class _StubChat:
    """Records the last prompt and returns a fixed verdict (no network)."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.last_prompt = None

    def chat(self, system, user, **kw):
        self.last_prompt = user
        return self.verdict


def test_verifier_deterministic_prechecks_reject_proven_failures(monkeypatch):
    monkeypatch.setenv('TTS_OFFICEBENCH_WM_TOOL_MODE', 'text')
    from tts.env.officebench_wm import OfficeBenchVerifier

    class _NoChat:
        def chat(self, *args, **kwargs):
            raise AssertionError("deterministic failure must bypass the LLM verifier")

    config = {"evaluation": [{
        "function": "evaluate_exact_match",
        "args": {"result_file": "./data/result.pdf", "expected_file": "reference/result.pdf"},
    }]}

    missing = OfficeBenchVerifier(_NoChat())
    assert missing.predict_success(
        "create result.pdf", [], [], config=config,
        final_state={"./data/result.pdf": "[file does not exist]"},
    ) == 0.0
    assert "required output" in missing.last_precheck_failure

    empty_output = OfficeBenchVerifier(_NoChat())
    empty_config = {"evaluation": [{
        "function": "evaluate_contain",
        "args": {"file": "./data/result.docx", "keywords": ["hello"]},
    }]}
    actions = [{"app": "word", "action": "create_new_file",
                "file_path": "/testbed/data/result.docx"}]
    assert empty_output.predict_success(
        "write hello", actions, ["Successfully created"],
        config=empty_config, final_state={"./data/result.docx": ""},
    ) == 0.0
    assert "left empty" in empty_output.last_precheck_failure

    empty_redirect = OfficeBenchVerifier(_NoChat())
    redirect = {
        "app": "shell", "action": "command",
        "command": ": > /testbed/data/result.docx",
    }
    assert empty_redirect.predict_success(
        "write hello", [redirect], ["Successfully executed"],
        config=empty_config, final_state={"./data/result.docx": ""},
    ) == 0.0
    assert "left empty" in empty_redirect.last_precheck_failure

    overwrite_append = OfficeBenchVerifier(_StubChat("SUCCESS"))
    append_config = {
        "task": "Add the sentence hello to the end of result.docx.",
        "evaluation": [{
            "function": "evaluate_contain",
            "args": {"file": "./data/result.docx", "keywords": ["hello"]},
        }],
    }
    overwrite = {
        "app": "word", "action": "write_to_file",
        "file_path": "/testbed/data/result.docx", "contents": "hello",
    }
    assert overwrite_append.predict_success(
        append_config["task"], [overwrite], ["Successfully wrote"],
        config=append_config, final_state={"./data/result.docx": "hello"},
    ) == 1.0
    assert overwrite_append.last_precheck_failure is None

    comma_cell = OfficeBenchVerifier(_NoChat())
    comma_config = {
        "task": "Write a formatted amount.",
        "evaluation": [{
            "function": "evaluate_excel_cell_value",
            "args": {
                "file": "./data/result.xlsx",
                "matches": [{"row": "1", "col": "1", "value": "2,100,000"}],
            },
        }],
    }
    comma_write = {
        "app": "excel", "action": "set_cell",
        "file_path": "/testbed/data/result.xlsx",
        "row_idx": "1", "column_idx": "1", "text": "2,100,000",
    }
    assert comma_cell.predict_success(
        comma_config["task"], [comma_write], ["Successfully wrote"],
        config=comma_config, final_state={"./data/result.xlsx": "2,100,000"},
    ) == 0.0
    assert "comma-containing text" in comma_cell.last_precheck_failure


def test_verifier_prechecks_allow_substantive_output(monkeypatch):
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    from tts.env.officebench_wm import OfficeBenchVerifier

    config = {"evaluation": [{
        "function": "evaluate_contain",
        "args": {"file": "./data/result.docx", "keywords": ["hello"]},
    }]}
    chat = _StubChat("SUCCESS")
    verifier = OfficeBenchVerifier(chat)
    actions = [
        {"app": "word", "action": "create_new_file",
         "file_path": "/testbed/data/result.docx"},
        {"app": "word", "action": "write_to_file",
         "file_path": "/testbed/data/result.docx", "contents": "hello"},
        {"app": "word", "action": "write_to_file",
         "file_path": "/testbed/data/result.docx", "contents": ""},
    ]
    assert verifier.predict_success(
        "write hello", actions,
        ["Successfully created", "Successfully wrote", "Successfully wrote"],
        config=config, final_state={"./data/result.docx": "hello"},
    ) == 1.0
    assert verifier.last_precheck_failure is None and chat.last_prompt is not None

    # Unknown state evidence and shell-produced binary files are deferred to the LLM, not rejected.
    unknown = _StubChat("SUCCESS")
    unknown_verifier = OfficeBenchVerifier(unknown)
    binary_action = {
        "app": "shell", "action": "command",
        "command": "cat /testbed/data/source.pdf > /testbed/data/result.pdf",
    }
    assert unknown_verifier.predict_success(
        "copy the PDF", [binary_action], ["Successfully executed"],
        config={"evaluation": [{
            "function": "evaluate_exact_match",
            "args": {"result_file": "./data/result.pdf", "expected_file": "reference/result.pdf"},
        }]},
        final_state={"WRITE LOG (predicted)": "APPLIED shell.command"},
    ) == 1.0
    assert unknown_verifier.last_precheck_failure is None and unknown.last_prompt is not None

    copied = _StubChat("SUCCESS")
    copied_verifier = OfficeBenchVerifier(copied)
    copied_actions = [
        {"app": "shell", "action": "command",
         "command": ": > /testbed/data/result.txt"},
        {"app": "shell", "action": "command",
         "command": "cp /testbed/data/source.txt /testbed/data/result.txt"},
    ]
    assert copied_verifier.predict_success(
        "copy hello", copied_actions, ["Successfully executed", "Successfully executed"],
        config={"evaluation": [{
            "function": "evaluate_contain",
            "args": {"file": "./data/result.txt", "keywords": ["hello"]},
        }]},
        final_state={"./data/result.txt": "hello"},
    ) == 1.0
    assert copied_verifier.last_precheck_failure is None

    negative = _StubChat("SUCCESS")
    negative_verifier = OfficeBenchVerifier(negative)
    assert negative_verifier.predict_success(
        "ensure forbidden text is absent",
        [{"app": "word", "action": "create_new_file",
          "file_path": "/testbed/data/result.txt"}],
        ["Successfully created"],
        config={"evaluation": [{
            "function": "evaluate_not_contain",
            "args": {"file": "./data/result.txt", "keywords": ["forbidden"]},
        }]},
        final_state={"./data/result.txt": ""},
    ) == 1.0
    assert negative_verifier.last_precheck_failure is None
    assert negative_verifier.predict_success(
        "ensure forbidden text is absent", [], [],
        config={"evaluation": [{
            "function": "evaluate_not_contain",
            "args": {"file": "./data/result.txt", "keywords": ["forbidden"]},
        }]},
        final_state={"./data/result.txt": "[file does not exist]"},
    ) == 1.0
    assert negative_verifier.last_precheck_failure is None


def test_verifier_blind_and_reference(monkeypatch):
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
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

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "python")
    python_blind = _ScriptChat([
        (
            "PYTHON_FINAL:\n"
            'print_json({"verdict": "SUCCESS" if data["config"] == {} '
            'and data["requirements"] == [] else "FAILURE"})'
        ),
    ])
    confidence = OfficeBenchVerifier(python_blind).predict_success(
        _FAKE_CONFIG["task"],
        actions,
        obs,
        config=_FAKE_CONFIG,
        final_state={"./calendar/Bob.ics": "[file does not exist]"},
    )
    assert confidence == 1.0


def test_verifier_self_consistency_uses_distinct_reproducible_seeds(monkeypatch):
    from tts.env.officebench_wm import OfficeBenchVerifier

    class _SeededChat:
        def __init__(self):
            self.seed = None
            self.seen = []
            self.prompts = []

        def chat(self, _system, user, **_kwargs):
            self.seen.append(self.seed)
            self.prompts.append(user)
            return "SUCCESS" if self.seed % 2 == 0 else "FAILURE"

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_SAMPLES", "3")
    chat = _SeededChat()
    confidence = OfficeBenchVerifier(chat, sample_seed=20).predict_success(
        _FAKE_CONFIG["task"],
        [{"app": "calendar", "action": "create_event", "user": "Bob"}],
        ["Successfully created"],
        config=_FAKE_CONFIG,
    )

    assert confidence == pytest.approx(2 / 3)
    assert chat.seen == [20, 21, 22]
    assert len(set(chat.prompts)) == 3
    assert chat.seed is None


def test_verifier_atomic_requirements_split_small_lists():
    from tts.env.officebench_wm import _atomic_requirements

    checks = _atomic_requirements({
        "evaluation": [
            {
                "function": "evaluate_contain",
                "args": {"file": "a.txt", "keywords": ["a", "b", "c"]},
            },
            {
                "function": "evaluate_excel_cell_value",
                "args": {"file": "b.xlsx", "matches": list(range(5))},
            },
        ],
    })

    assert [check["args"]["keywords"] for check in checks[:3]] == [["a"], ["b"], ["c"]]
    assert checks[3]["args"]["matches"] == list(range(5))


def test_verifier_percheck_keeps_partial_coverage_separate(monkeypatch):
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    from tts.env.officebench_wm import OfficeBenchVerifier

    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_PERCHECK", "1")
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_UNKNOWN_CREDIT", "0.25")
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT", "0")
    chat = _ScriptChat(["STATUS: MET", "STATUS: UNKNOWN", "STATUS: UNMET"])
    verifier = OfficeBenchVerifier(chat)
    confidence = verifier.predict_success(
        "write three keywords",
        [{"app": "word", "action": "write_to_file"}],
        ["Successfully wrote"],
        config={
            "evaluation": [{
                "function": "evaluate_contain",
                "args": {"file": "a.txt", "keywords": ["a", "b", "c"]},
            }],
        },
        final_state={"a.txt": "a"},
    )

    assert confidence == 0.0
    assert verifier.last_percheck_breakdown == {
        "met": 1,
        "unmet": 1,
        "unknown": 1,
        "total": 3,
        "unknown_credit": 0.25,
        "atomic_score": pytest.approx((1 + 0.25) / 3),
        "whole_score": 0.0,
    }


def test_verifier_percheck_uses_whole_task_calibration(monkeypatch):
    from tts.env.officebench_wm import OfficeBenchVerifier

    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_PERCHECK", "1")
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT", "0.25")
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_PYTHON", "0")
    chat = _ScriptChat(["STATUS: MET", "FAILURE"])
    verifier = OfficeBenchVerifier(chat)

    confidence = verifier.predict_success(
        "create a.txt",
        [{"app": "word", "action": "create_new_file"}],
        ["Successfully created"],
        config={
            "evaluation": [{
                "function": "evaluate_file_exist",
                "args": {"file": "a.txt"},
            }],
        },
        final_state={"a.txt": "content"},
    )

    assert confidence == 0.0
    assert verifier.last_percheck_breakdown["atomic_score"] == 1.0
    assert verifier.last_percheck_breakdown["whole_score"] == 0.0


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-4.1", "DeepSeek-V4-Pro"])
def test_officebench_repaired_defaults_and_overrides(monkeypatch, model):
    monkeypatch.delenv("TTS_OFFICEBENCH_VERIFIER_TYPED", raising=False)
    monkeypatch.delenv("TTS_SELECTION_CONFIDENCE_MARGIN", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_WM_PYTHON_TIMEOUT_S", raising=False)
    from argparse import Namespace

    from experiments.run import _configure_officebench_bon

    monkeypatch.delenv("TTS_OFFICEBENCH_VERIFIER_SAMPLES", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_VERIFIER_PERCHECK", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_VERIFIER_UNKNOWN_CREDIT", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_COMPARATIVE_RERANK", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_COMPARATIVE_CONFIDENCE_MARGIN", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_SELECTION_CONFIDENCE_MARGIN", raising=False)
    args = Namespace(
        agent="llm_wm",
        methods="ours",
        llm_agent=model,
        verifier_samples=None,
        success_score="simulator_success",
        _success_score_explicit=False,
        predict_all_actions=True,
        _predict_all_actions_explicit=False,
        simulation_only=False,
    )
    _configure_officebench_bon(args)

    assert os.environ["TTS_OFFICEBENCH_VERIFIER_SAMPLES"] == "5"
    assert os.environ["TTS_OFFICEBENCH_VERIFIER_PERCHECK"] == "1"
    assert os.environ["TTS_OFFICEBENCH_VERIFIER_UNKNOWN_CREDIT"] == "0.25"
    assert os.environ["TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT"] == "0.25"
    assert os.environ["TTS_OFFICEBENCH_COMPARATIVE_RERANK"] == "1"
    assert os.environ["TTS_OFFICEBENCH_COMPARATIVE_CONFIDENCE_MARGIN"] == "0.25"
    assert args.success_score == "simulator_confidence_guarded"
    assert args.predict_all_actions is True
    assert os.environ["TTS_OFFICEBENCH_VERIFIER_TYPED"] == "1"
    assert os.environ["TTS_OFFICEBENCH_WM_PYTHON_TIMEOUT_S"] == "5"
    assert os.environ["TTS_SELECTION_CONFIDENCE_MARGIN"] == "0.05"

    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_SAMPLES", "7")
    explicit = Namespace(
        agent="llm_wm",
        methods="ours",
        llm_agent=model,
        verifier_samples=7,
        success_score="simulator_success",
        _success_score_explicit=True,
        predict_all_actions=False,
        _predict_all_actions_explicit=True,
        simulation_only=False,
    )
    _configure_officebench_bon(explicit)
    assert os.environ["TTS_OFFICEBENCH_VERIFIER_SAMPLES"] == "7"
    assert explicit.success_score == "simulator_success"
    assert explicit.predict_all_actions is False

    monkeypatch.delenv("TTS_OFFICEBENCH_VERIFIER_SAMPLES", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", model)
    deployment_only = Namespace(
        agent="llm_wm",
        methods="ours",
        llm_agent=None,
        verifier_samples=None,
        success_score="simulator_success",
        _success_score_explicit=False,
        predict_all_actions=True,
        _predict_all_actions_explicit=False,
        simulation_only=False,
    )
    _configure_officebench_bon(deployment_only)
    assert os.environ["TTS_OFFICEBENCH_VERIFIER_SAMPLES"] == "5"


@pytest.mark.parametrize("model,methods,agent", [
    ("DeepSeek-V4-Pro", "b1", "llm_wm"),
    ("qwen3", "ours", "llm_wm"),
    ("DeepSeek-V4-Pro", "ours", "llm"),
])
def test_officebench_repaired_profile_scope(monkeypatch, model, methods, agent):
    from argparse import Namespace
    from experiments.run import _configure_officebench_bon

    monkeypatch.delenv("TTS_OFFICEBENCH_VERIFIER_TYPED", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_COMPARATIVE_RERANK", raising=False)
    args = Namespace(agent=agent, methods=methods, llm_agent=model,
                     predict_all_actions=False, success_score="simulator_success")
    _configure_officebench_bon(args)
    assert args.predict_all_actions is False
    assert args.success_score != "simulator_confidence_guarded"
    assert "TTS_OFFICEBENCH_COMPARATIVE_RERANK" not in os.environ


def test_officebench_repaired_profile_respects_environment(monkeypatch):
    from argparse import Namespace
    from experiments.run import _configure_officebench_bon

    overrides = {
        "TTS_OFFICEBENCH_VERIFIER_TYPED": "0",
        "TTS_OFFICEBENCH_VERIFIER_PERCHECK": "0",
        "TTS_OFFICEBENCH_VERIFIER_SAMPLES": "2",
        "TTS_OFFICEBENCH_VERIFIER_UNKNOWN_CREDIT": "0",
        "TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT": "0",
        "TTS_OFFICEBENCH_COMPARATIVE_RERANK": "0",
        "TTS_OFFICEBENCH_COMPARATIVE_CONFIDENCE_MARGIN": "0.4",
        "TTS_SELECTION_CONFIDENCE_MARGIN": "0.3",
        "TTS_OFFICEBENCH_WM_PYTHON_TIMEOUT_S": "9",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    args = Namespace(agent="llm_wm", methods="b1,ours", llm_agent="DeepSeek-V4-Pro",
                     predict_all_actions=True, success_score="simulator_success")
    _configure_officebench_bon(args)
    assert args.predict_all_actions is True
    assert all(os.environ[key] == value for key, value in overrides.items())


def test_officebench_repairs_switch_app_shorthand():
    """A model-addressed switch is normalized to the documented system action."""
    from tts.env.officebench_agent import run_officebench_agent

    agent = _ScriptChat([
        repr({"app": "excel", "action": "switch_app"}),
        repr({"app": "system", "action": "finish_task", "answer": "None"}),
    ])
    session = _RecordingSession(read_only=False)
    cfg = {
        "username": "Bob",
        "date": "d",
        "weekday": "w",
        "time": "t",
        "task": "inspect the spreadsheet",
        "evaluation": [],
    }

    run = run_officebench_agent(session, cfg, agent, max_steps=3)

    assert run.actions[0] == {
        "app": "system",
        "action": "switch_app",
        "target_app": "excel",
    }
    assert run.observations[0] == "Successfully switched to app: excel"
    assert run.done and run.final_answer == "None"


def test_verifier_gold_expected_file(monkeypatch):
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_TYPED", "0")
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
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


def test_verifier_state_grounding_and_giveup(monkeypatch):
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_TYPED", "0")
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
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
        final_state={
            "data/score.xlsx": "(2, 2): 1000+2000000\t",
            "./calendar/Bob.ics": "BEGIN:VCALENDAR\nEND:VCALENDAR",
        },
        final_answer="None")
    assert "PREDICTED RESULTING FILE STATE" in grounded.last_prompt
    assert "1000+2000000" in grounded.last_prompt  # judge sees the literal string, not the claim
    assert "give-up/refusal" in grounded.last_prompt  # the give-up note fired for answer "None"


def test_verifier_agentic_loop_can_gather_read_only_evidence(monkeypatch):
    """The final OfficeBench WM verdict uses the same bounded grounding protocol."""
    from tts.env.officebench_wm import OfficeBenchVerifier

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_SAMPLES", "1")
    chat = _ScriptChat([
        "READ /testbed/data/result.xlsx",
        "VERDICT: SUCCESS",
    ])
    query = _GroundingQuery()
    verifier = OfficeBenchVerifier(chat)
    verifier.bind_query_interface(query)
    confidence = verifier.predict_success(
        "update the sheet", [{"app": "excel", "action": "set_cell"}],
        ["OBSERVATION: Successfully set the cell."])

    assert confidence == 1.0
    assert query.read_paths == ["/testbed/data/result.xlsx"]
    assert chat.calls == 2


def test_verifier_python_loop_uses_host_mediated_evidence(monkeypatch):
    from tts.env.officebench_wm import OfficeBenchVerifier

    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "python")
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_SAMPLES", "1")
    chat = _ScriptChat([
        (
            "PYTHON_READ:\n"
            'print_json({"queries": [{"op": "read", "path": "/testbed/data/result.xlsx"}]})'
        ),
        'PYTHON_FINAL:\nprint_json({"verdict": "SUCCESS"})',
    ])
    query = _GroundingQuery()
    verifier = OfficeBenchVerifier(chat)
    verifier.bind_query_interface(query)
    confidence = verifier.predict_success(
        "update the sheet",
        [{"app": "excel", "action": "set_cell"}],
        ["OBSERVATION: Successfully set the cell."],
    )

    assert confidence == 1.0
    assert query.read_paths == ["/testbed/data/result.xlsx"]
    assert verifier.n_python_calls == 2
    assert verifier.n_python_failures == 0


def test_verifier_private_percheck_and_whole_votes(monkeypatch):
    monkeypatch.setenv("TTS_OFFICEBENCH_VERIFIER_TYPED", "0")
    from tts.env.officebench_wm import OfficeBenchVerifier

    monkeypatch.setenv('TTS_OFFICEBENCH_WM_TOOL_MODE', 'python')
    monkeypatch.setenv('TTS_OFFICEBENCH_WM_PYTHON', '1')
    monkeypatch.setenv('TTS_OFFICEBENCH_VERIFIER_PERCHECK', '1')
    monkeypatch.setenv('TTS_OFFICEBENCH_VERIFIER_SAMPLES', '5')
    secret = 'PRIVATE_CHECK_VALUE_731'
    email_check = {'function': 'evaluate_contain', 'args': {
        'doc_type': 'email', 'username': secret, 'keywords': [secret]}}
    email_prompt = OfficeBenchVerifier(None)._build_prompt(
        secret, [], [], {'evaluation': [email_check]}, status_check=email_check)
    assert 'This is an EMAIL check' in email_prompt
    assert secret not in email_prompt

    class Chat:
        def __init__(self):
            self.calls = 0

        def chat(self, system, prompt, **kwargs):
            assert secret not in system + prompt
            assert 'successful recovery' in system + prompt
            self.calls += 1
            if self.calls == 1:
                return (
                    'PYTHON_FINAL:\n'
                    'args = data["status_check"]["args"]\n'
                    'met = all(k in data["final_state"][args["file"]] '
                    'for k in args["keywords"])\n'
                    'print_json({"text": "STATUS: MET" if met else "STATUS: UNMET"})'
                )
            return (
                'PYTHON_FINAL:\n'
                'args = data["config"]["evaluation"][0]["args"]\n'
                'met = all(k in data["final_state"][args["file"]] '
                'for k in args["keywords"])\n'
                'print_json({"verdict": "SUCCESS" if met else "FAILURE"})'
            )

    for content, expected_calls, expected_score in ((secret, 6, 1.0), ('wrong', 1, 0.0)):
        chat = Chat()
        verifier = OfficeBenchVerifier(chat)
        score = verifier.predict_success(
            secret, [], [], final_answer='None',
            config={'evaluation': [{'function': 'evaluate_contain',
                'args': {'file': 'data/private.txt', 'keywords': [secret]}}]},
            final_state={'data/private.txt': content}, gold_files={'gold': secret})
        assert score == expected_score
        assert chat.calls == expected_calls
        assert verifier.n_python_calls == expected_calls
        assert verifier.n_python_failures == 0


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


def test_symbolic_new_workbook_create_set_delete():
    """A new workbook is an empty symbolic grid that supports subsequent cell edits exactly."""
    from tts.env.officebench_wm_predict import symbolic_file_state

    writes = [
        ({"app": "excel", "action": "create_new_file", "file_path": "/testbed/data/new.xlsx"}, "o"),
        ({"app": "excel", "action": "set_cell", "row_idx": "1", "column_idx": "1",
          "text": "Name"}, "o"),
        ({"app": "excel", "action": "set_cell", "row_idx": 1, "column_idx": 2,
          "text": "Alice"}, "o"),
        ({"app": "excel", "action": "delete_cell", "row_idx": 1, "column_idx": 1}, "o"),
    ]
    out = symbolic_file_state("data/new.xlsx", "[file does not exist]", writes)
    assert out == "(1, 1): [Empty Cell]\t(1, 2): Alice\t\n"


def test_symbolic_create_existing_file_is_noop():
    """OfficeBench create_new_file preserves an existing workbook instead of truncating it."""
    from tts.env.officebench_wm_predict import symbolic_file_state

    initial = "(1, 1): keep"
    writes = [(
        {"app": "excel", "action": "create_new_file", "file_path": "/testbed/data/existing.xlsx"},
        "OBSERVATION: File already exists.",
    )]
    assert symbolic_file_state("data/existing.xlsx", initial, writes) == initial


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

    overwrite = [(
        {"app": "word", "action": "create_new_file", "file_path": "/testbed/data/h.docx"}, "o")]
    assert symbolic_file_state("data/h.docx", "OLD CONTENT", overwrite) == ""


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


@pytest.mark.parametrize("program_only", [False, True])
def test_predict_file_state_symbolic_skips_llm(program_only):
    """predict_file_state uses the symbolic reconstruction and never calls the LLM for set_cell."""
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    class _BoomChat:
        def chat(self, *a, **k):
            raise AssertionError("LLM must not be called for an exactly-modelable file")

    p = OfficeBenchWMPredictor(_BoomChat())
    p.program_only = program_only
    p.applied_writes["data/s.xlsx"] = [
        ({"app": "excel", "action": "set_cell", "row_idx": 1, "column_idx": 1, "text": 42}, "o")]
    out = p.predict_file_state("./data/s.xlsx", "(1, 1): 10")
    assert out == "(1, 1): 42\t\n" and p.n_llm_calls == 0


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


def test_default_step_budget_is_fixed_at_40(monkeypatch):
    """Every OfficeBench subset has the same 40 productive-step budget."""
    import inspect

    from tts.env.officebench_agent import run_officebench_agent
    from tts.env.real_officebench import _rollout_max_steps, _scaled_max_steps

    monkeypatch.delenv("TTS_OFFICEBENCH_STEPS_FLOOR", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_STEPS_PER_APP", raising=False)
    monkeypatch.delenv("TTS_OFFICEBENCH_STEPS_CAP", raising=False)
    assert _scaled_max_steps(1) == 40
    assert _scaled_max_steps(2) == 40
    assert _scaled_max_steps(3) == 40
    assert _rollout_max_steps(48, 3) == 40
    assert _rollout_max_steps(60, 1) == 40
    assert _rollout_max_steps(60, 2) == 40
    assert _rollout_max_steps(60, 3) == 40
    assert inspect.signature(run_officebench_agent).parameters["max_steps"].default == 40


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
    run = run_officebench_agent(
        session, cfg, agent, max_steps=6, predictor=predictor,
        predict_all_actions=False,
    )

    ls_execs = [a for a in session.exec_calls if a.get("command") == "ls /testbed/data"]
    assert len(ls_execs) == 1                      # the 3 immediate repeats were pruned (not re-executed)
    assert run.final_answer == "None"              # finish still reached within the budget
    assert len(set(run.observations[1:5])) == 1
    assert all("repeated action" not in o for o in run.observations)


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
    run_officebench_agent(
        session, cfg, agent, max_steps=6, predictor=predictor,
        predict_all_actions=False,
    )
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


@pytest.mark.parametrize('listing', [
    {'error': 'inventory timed out'},
    {'missing': True},
    {'_timeout': True},
    {},
])
def test_predict_all_failed_inventory_is_not_an_empty_directory(listing):
    from tts.env.officebench_wm_predict import (
        OfficeBenchSimulationError, OfficeBenchWMPredictor, OfficeBenchWMQuery,
    )

    class Session:
        def apps_catalogue(self):
            return {}

        def list_state(self, path, **kwargs):
            return listing

    predictor = OfficeBenchWMPredictor(_ScriptChat([]))
    predictor.program_only = True
    predictor.bind_query_interface(OfficeBenchWMQuery(
        Session(), lambda: '/', allow_action_execution=False))
    with pytest.raises(OfficeBenchSimulationError, match='inventory'):
        predictor.predict_read(
            {'app': 'shell', 'action': 'command', 'command': 'find /testbed/data -type f'},
            None, dirty=False)


def test_predict_all_empty_inventory_remains_a_valid_empty_read():
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor, OfficeBenchWMQuery

    class Session:
        def apps_catalogue(self):
            return {}

        def list_state(self, path, **kwargs):
            assert path == '/testbed'
            return {'entries': [], 'missing': False}

    predictor = OfficeBenchWMPredictor(_ScriptChat([]))
    predictor.program_only = True
    predictor.bind_query_interface(OfficeBenchWMQuery(
        Session(), lambda: '/', allow_action_execution=False))
    assert predictor.predict_read(
        {'app': 'shell', 'action': 'command', 'command': 'find /testbed -type f'},
        None, dirty=False) == ''


def test_predict_all_symbolic_dirty_read_preserves_unedited_cells():
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor, OfficeBenchWMQuery

    initial = '(1, 1): old\n(2, 1): ' + 'x' * 8000 + 'TAIL'

    class Session:
        def apps_catalogue(self):
            return {}

        def read_state(self, specs):
            assert specs[0]['path'] == '/testbed/data/s.xlsx'
            return {specs[0]['path']: initial}

    predictor = OfficeBenchWMPredictor(_ScriptChat([]))
    predictor.program_only = True
    predictor.bind_query_interface(OfficeBenchWMQuery(
        Session(), lambda: '/', allow_action_execution=False))
    predictor.applied_writes['data/s.xlsx'] = [(
        {'app': 'excel', 'action': 'set_cell', 'row_idx': 1, 'column_idx': 1, 'text': 'new'},
        'OBSERVATION: Successfully set cell',
    )]
    observation = predictor.predict_read(
        {'app': 'excel', 'action': 'read_file', 'file_path': '/testbed/data/s.xlsx'},
        None, dirty=True)
    assert observation == 'OBSERVATION: The following is the table from the excel file:\n' + initial.replace('(1, 1): old\n', '(1, 1): new\t\n') + '\t\n'
    assert predictor.n_llm_calls == 0


def test_driver_and_verifier_preserve_complete_file_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("TTS_OFFICEBENCH_WM_TOOL_MODE", "text")
    from tts.env import officebench_driver as driver
    from tts.env.officebench_wm import OfficeBenchVerifier

    text = 'x' * 9000 + 'REQUIRED_TAIL'
    (tmp_path / 'data').mkdir()
    (tmp_path / 'calendar').mkdir()
    (tmp_path / 'data' / 'answer.txt').write_text(text)
    (tmp_path / 'calendar' / 'Bob.ics').write_text(text)
    monkeypatch.setattr(driver, '_TESTBED', str(tmp_path))
    monkeypatch.setattr(driver, '_ensure_path', lambda: None)
    for extras in (False, True):
        state = driver.handle_read_state({
            'files': [{'path': 'data/answer.txt', 'doc_type': 'txt'}], 'extras': extras,
        })['state']
        assert state['data/answer.txt'] == text
        if extras:
            assert state['calendar/Bob.ics'] == text
    monkeypatch.setenv('TTS_OFFICEBENCH_VERIFIER_REFERENCE', '1')
    prompt = OfficeBenchVerifier(_ScriptChat([]))._build_prompt(
        'Check the tail', [], [], {},
        final_state={'out': text}, gold_files={'gold': text})
    assert prompt.count(text) == 2


def test_write_prediction_has_program_budget_and_preserves_native_failure(monkeypatch):
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    monkeypatch.delenv('TTS_OFFICEBENCH_WM_WRITE_TOKENS', raising=False)

    class Chat:
        def chat(self, system, prompt, **kwargs):
            assert kwargs['max_tokens'] >= 2048
            return 'PYTHON_FINAL:\nprint_json({"text": "OBSERVATION: Failed to write text to /testbed/data/s.xlsx"})'

    predictor = OfficeBenchWMPredictor(Chat())
    predictor.program_only = True
    observation = predictor.predict_write({
        'app': 'excel', 'action': 'set_cell', 'file_path': '/testbed/data/s.xlsx',
        'row_idx': -1, 'column_idx': 1, 'text': 'x',
    })
    assert 'Failed to write' in observation
    assert predictor.n_failed_writes == 1
    assert not predictor.was_written('data/s.xlsx')
    assert predictor.transition_log.entries()[0].applied is False


def test_final_state_simulation_failure_does_not_fall_back_to_pristine(monkeypatch):
    from tts.env.officebench_wm_predict import OfficeBenchSimulationError, OfficeBenchWMPredictor

    predictor = OfficeBenchWMPredictor(_ScriptChat([]))
    predictor.global_dirty = True
    monkeypatch.setattr(predictor, '_predict', lambda *a, **kw: None)
    with pytest.raises(OfficeBenchSimulationError, match='final state is unknown'):
        predictor.predict_file_state('data/out.txt', 'old')


def test_reconstruction_logs_preserve_all_writes_and_literal_tails():
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor

    predictor = OfficeBenchWMPredictor(_ScriptChat([]))
    content = 'x' * 8000 + 'REQUIRED_TAIL'
    for i in range(45):
        action = {'app': 'word', 'action': 'write_to_file', 'contents': content + str(i)}
        predictor.transition_log.record(action, 'ok', ['data/out.docx'], applied=True)
        predictor.write_history.append((action, 'ok'))
    for rendered in (predictor.trace_text('data/out.docx'),
                     predictor.file_write_log('data/out.docx'),
                     predictor.predicted_write_log()):
        assert rendered.count('REQUIRED_TAIL') == 45
        assert 'REQUIRED_TAIL44' in rendered
    assert len(predictor._write_history_data()) == 45


def test_fixed_context_preserves_files_or_reports_budget_exhaustion(monkeypatch):
    from tts.env.officebench_wm_predict import (
        OfficeBenchSimulationError, build_officebench_flat_system_prompt,
    )

    text = 'x' * 9000 + 'REQUIRED_TAIL'
    kwargs = dict(task_config={}, catalogue={}, initial_state={'file': text})
    monkeypatch.setenv('TTS_OFFICEBENCH_B1_CONTEXT_CHARS', '20000')
    assert text in build_officebench_flat_system_prompt(**kwargs)
    monkeypatch.setenv('TTS_OFFICEBENCH_B1_CONTEXT_CHARS', '1000')
    with pytest.raises(OfficeBenchSimulationError, match='rather than truncate'):
        build_officebench_flat_system_prompt(**kwargs)
