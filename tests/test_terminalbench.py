"""Terminal-Bench command loop and write-safe world-model tests."""

from __future__ import annotations

import asyncio
import re

import pytest

from tts.env.terminalbench_agent import (
    TerminalCommandResult,
    run_terminalbench_agent,
    run_terminalbench_tts,
)
from tts.env.terminalbench_wm import (
    TerminalBenchWMPredictor,
    shell_command_is_read_only,
)


class _Chat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.last_usage = 1

    def chat(self, *_args, **_kwargs):
        if not self.responses:
            raise AssertionError("unexpected chat call")
        return self.responses.pop(0)


class _Executor:
    def __init__(self):
        self.commands = []

    async def initial_cwd(self):
        return "/app"

    async def execute(self, command, cwd):
        self.commands.append((command, cwd))
        return TerminalCommandResult(
            stdout=f"ran {command}",
            return_code=0,
            cwd=cwd,
        )


def test_terminalbench_read_only_classifier_is_conservative():
    assert shell_command_is_read_only("cat /app/input.txt")
    assert shell_command_is_read_only("find /app -maxdepth 2 -type f | sort")
    assert shell_command_is_read_only("git status --short")
    assert shell_command_is_read_only("cd /app")
    assert shell_command_is_read_only("sed -n '1,240p' -- /app/input.txt")
    assert shell_command_is_read_only("sed 's/a/b/g' /app/input.txt")

    assert not shell_command_is_read_only("echo hi > /app/output.txt")
    assert not shell_command_is_read_only("echo hi &> /app/output.txt")
    assert not shell_command_is_read_only("cat /app/input.txt &>> /app/output.txt")
    assert not shell_command_is_read_only("echo hi |& tee /app/output.txt")
    assert not shell_command_is_read_only("sed -i 's/a/b/' /app/file")
    assert not shell_command_is_read_only("sed 's/a/b/w /app/output.txt' /app/file")
    assert not shell_command_is_read_only("sed 's/a/b/e' /app/file")
    assert not shell_command_is_read_only("sed 'W /app/output.txt' /app/file")
    assert not shell_command_is_read_only("sed --expression='w /app/output.txt' /app/file")
    assert not shell_command_is_read_only("find /app -delete")
    assert not shell_command_is_read_only("git branch new-branch")
    assert not shell_command_is_read_only("git tag release")
    assert not shell_command_is_read_only("git diff --output=/app/output.txt")
    assert not shell_command_is_read_only("git grep -O touch pattern")
    assert not shell_command_is_read_only("date --set tomorrow")
    assert not shell_command_is_read_only("printf -v result value")
    assert not shell_command_is_read_only("rg --pre 'touch /tmp/x' pattern /app")
    assert not shell_command_is_read_only("tree -o /app/output.txt /app")
    assert not shell_command_is_read_only("python -c 'open(\"/tmp/x\",\"w\").write(\"x\")'")
    assert not shell_command_is_read_only("cat $(touch /tmp/x)")


def test_terminalbench_agent_runs_one_command_at_a_time():
    executor = _Executor()
    chat = _Chat([
        '{"command":"cat /app/input.txt"}',
        '{"finish":true,"summary":"done"}',
    ])

    run = asyncio.run(run_terminalbench_agent(
        "Inspect the input.",
        executor,
        chat,
        max_steps=4,
    ))

    assert run.commands == ["cat /app/input.txt"]
    assert run.done is True
    assert run.final_answer == "done"
    assert executor.commands == [("cat /app/input.txt", "/app")]


def test_terminalbench_world_model_executes_only_pristine_reads():
    real = _Executor()
    chat = _Chat([
        "OBSERVATION:\nreturn_code: 0\nstdout: predicted write",
        "OBSERVATION:\nreturn_code: 0\nstdout: predicted dirty read",
    ])
    predictor = TerminalBenchWMPredictor(
        "Create an output.",
        real,
        chat,
        recursive=False,
    )

    async def scenario():
        first = await predictor.execute("cat /app/input.txt", "/app")
        write = await predictor.execute("echo hi > /app/output.txt", "/app")
        dirty = await predictor.execute("cat /app/output.txt", "/app")
        return first, write, dirty

    first, write, dirty = asyncio.run(scenario())
    assert first.predicted is False
    assert write.predicted is True
    assert dirty.predicted is True
    assert real.commands == [("cat /app/input.txt", "/app")]
    assert len(predictor.transitions.entries()) == 2


def test_terminalbench_world_model_uses_shared_agentic_tool_loop():
    real = _Executor()
    chat = _Chat([
        "\n".join([
            "LIST /app",
            "READ /app/input.txt",
            "CALL cat /app/input.txt",
            "TRACE /app",
        ]),
        "OBSERVATION:\nreturn_code: 0\nstdout: predicted write",
    ])
    predictor = TerminalBenchWMPredictor(
        "Create an output.",
        real,
        chat,
        recursive=True,
        max_turns=2,
    )

    result = asyncio.run(predictor.execute("echo hi > /app/output.txt", "/app"))

    assert result.predicted is True
    assert predictor.query.reads == 1
    assert predictor.query.lists == 1
    assert predictor.query.calls == 1
    assert predictor.query.traces == 1
    assert predictor.probe_calls == 4
    assert [command for command, _ in real.commands] == [
        "find /app -maxdepth 2 -mindepth 1 -printf '%y %p\\n' | sort | head -240",
        "sed -n '1,240p' -- /app/input.txt",
        "cat /app/input.txt",
    ]


def test_terminalbench_read_tool_combines_pristine_state_with_predicted_transitions():
    real = _Executor()
    predictor = TerminalBenchWMPredictor(
        "Create an output.",
        real,
        _Chat(["OBSERVATION:\nreturn_code: 0\nstdout: predicted write"]),
        recursive=False,
    )

    async def scenario():
        await predictor.execute("echo hi > /app/output.txt", "/app")
        return await predictor.query.read("/app/output.txt", "/app")

    evidence = asyncio.run(scenario())

    assert "ran sed -n" in evidence
    assert "PREDICTED LOGICAL TRANSITIONS" in evidence
    assert "echo hi > /app/output.txt" in evidence


def test_terminalbench_tts_selects_then_commits_once():
    real = _Executor()
    wm_responses = {
        10: [
            "OBSERVATION:\nreturn_code: 0\nstdout: weak candidate",
            "VERDICT: FAILURE\nCONFIDENCE: 0.9",
        ],
        11: [
            "OBSERVATION:\nreturn_code: 0\nstdout: strong candidate",
            "VERDICT: SUCCESS\nCONFIDENCE: 0.9",
        ],
    }

    def chat_factory(seed, temperature):
        if temperature > 0:
            command = "echo weak > /app/out" if seed == 10 else "echo strong > /app/out"
            return _Chat([
                f'{{"command":"{command}"}}',
                '{"finish":true,"summary":"done"}',
            ])
        return _Chat(wm_responses[seed])

    decision = asyncio.run(run_terminalbench_tts(
        "Write the strongest result.",
        real,
        chat_factory,
        settings={
            "method": "ours",
            "n": 2,
            "seed": 10,
            "temperature": 0.7,
            "max_steps": 4,
            "wm_turns": 2,
            "verifier_samples": 1,
        },
    ))

    assert decision.selected_index == 1
    assert [command for command, _ in real.commands] == ["echo strong > /app/out"]


def test_terminalbench_tts_never_selects_an_incomplete_rollout():
    real = _Executor()

    def chat_factory(seed, temperature):
        if temperature > 0:
            if seed == 10:
                return _Chat(['{"command":"echo partial > /app/out"}', '{"command":"'])
            return _Chat([
                '{"command":"echo complete > /app/out"}',
                '{"finish":true,"summary":"done"}',
            ])
        return _Chat([
            "OBSERVATION:\nreturn_code: 0",
            "VERDICT: FAILURE\nSUCCESS_PROBABILITY: 0.01",
        ])

    decision = asyncio.run(run_terminalbench_tts(
        "Write a result.",
        real,
        chat_factory,
        settings={
            "method": "ours",
            "n": 2,
            "seed": 10,
            "temperature": 0.7,
            "max_steps": 4,
            "wm_turns": 1,
            "verifier_samples": 1,
        },
    ))

    assert decision.rollouts[0].run.done is False
    assert decision.rollouts[0].confidence == 0.0
    assert decision.selected_index == 1
    assert [command for command, _ in real.commands] == ["echo complete > /app/out"]


def test_terminalbench_legacy_executor_anchors_each_command_to_logical_cwd():
    pytest.importorskip("terminal_bench")
    from tts.env.terminalbench_legacy import _LegacyExecutor

    class _Session:
        def __init__(self):
            self.sent = []

        def send_keys(self, keys, **_kwargs):
            self.sent.append(keys[0])

        def get_incremental_output(self):
            marker = re.search(r"(__TTS_[0-9a-f]+__)RC", self.sent[-1]).group(1)
            cwd = "/work" if len(self.sent) == 1 else "/tmp"
            echoed = self.sent[-1].replace(
                f"{marker}END",
                f"{marker[:20]}\n{marker[20:]}END",
            )
            return (
                f"root@container:/app# {echoed}\n"
                f"{marker}RC:0\n{marker}PWD:{cwd}\n{marker}END\n"
            )

    async def scenario():
        session = _Session()
        executor = _LegacyExecutor(session, timeout_sec=5)
        initial_cwd = await executor.initial_cwd()
        result = await executor.execute("cd /tmp", initial_cwd)
        return session, initial_cwd, result

    session, initial_cwd, result = asyncio.run(scenario())
    assert initial_cwd == "/work"
    assert result.cwd == "/tmp"
    assert session.sent[1].startswith("cd -- /work && cd /tmp;")
    assert "__tts_rc=$?" in result.stdout
