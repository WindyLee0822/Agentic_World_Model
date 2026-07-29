"""Shared command-loop solver for Terminal-Bench 1 and 2.

The official harnesses own sandbox construction and verification. This module owns the agent loop:
one shell command per turn, the same prompt for real and world-model execution, and one final replay
of the selected trajectory into the official task container.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Optional, Protocol


@dataclass
class TerminalCommandResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    cwd: Optional[str] = None
    predicted: bool = False

    def observation(self, cap: int = 12000) -> str:
        parts = [f"return_code={self.return_code}"]
        if self.cwd:
            parts.append(f"cwd={self.cwd}")
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        if self.predicted:
            parts.append("(world-model prediction; command was not executed)")
        text = "\n".join(parts)
        return text if len(text) <= cap else text[:cap] + "\n...[truncated]"


class TerminalExecutor(Protocol):
    async def initial_cwd(self) -> str: ...

    async def execute(self, command: str, cwd: str) -> TerminalCommandResult: ...


@dataclass
class TerminalAgentRun:
    commands: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    return_codes: List[int] = field(default_factory=list)
    final_answer: str = ""
    done: bool = False
    n_steps: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TerminalBenchRollout:
    index: int
    run: TerminalAgentRun
    confidence: float
    n_reads: int = 0
    n_spawns: int = 0
    max_depth: int = 0


@dataclass
class TerminalBenchDecision:
    method: str
    selected_index: int
    rollouts: List[TerminalBenchRollout]
    committed: bool
    commit_results: List[TerminalCommandResult] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(rollout.run.total_tokens for rollout in self.rollouts)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "selected_index": self.selected_index,
            "committed": self.committed,
            "rollouts": [
                {
                    "index": rollout.index,
                    "confidence": rollout.confidence,
                    "n_reads": rollout.n_reads,
                    "n_spawns": rollout.n_spawns,
                    "max_depth": rollout.max_depth,
                    "run": rollout.run.to_dict(),
                }
                for rollout in self.rollouts
            ],
            "commit_results": [asdict(result) for result in self.commit_results],
        }


_SYSTEM = """You are a terminal agent operating inside an isolated benchmark container.
Solve the task by issuing exactly one shell command per turn. Inspect before editing, keep commands
non-interactive, and do not start a long-lived foreground process. Keep each command compact and
use additional turns instead of embedding a large program in one response. Return only one JSON object:

{"command": "<one shell command>"}

When the task is complete, return:

{"finish": true, "summary": "<brief completion summary>"}

Do not wrap the JSON in markdown and do not emit multiple commands as separate JSON objects."""


def _history_text(run: TerminalAgentRun, cap: int = 24000) -> str:
    lines = []
    for index, command in enumerate(run.commands):
        observation = run.observations[index] if index < len(run.observations) else ""
        lines.append(f"STEP {index}\nCOMMAND: {command}\nOBSERVATION:\n{observation}")
    text = "\n\n".join(lines)
    return text if len(text) <= cap else text[-cap:]


def _prompt(instruction: str, cwd: str, run: TerminalAgentRun) -> str:
    history = _history_text(run) or "(no commands yet)"
    return (
        f"TASK:\n{instruction}\n\nCURRENT WORKING DIRECTORY:\n{cwd}\n\n"
        f"COMMAND HISTORY:\n{history}\n\nReturn the next JSON action."
    )


def _extract_object(text: str) -> Optional[dict]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _chat_tokens(chat: Any) -> int:
    try:
        return max(0, int(getattr(chat, "last_usage", 0) or 0))
    except (TypeError, ValueError):
        return 0


async def run_terminalbench_agent(
    instruction: str,
    executor: TerminalExecutor,
    chat: Any,
    *,
    max_steps: int = 40,
    temperature: float = 0.7,
    max_tokens: int = 8000,
) -> TerminalAgentRun:
    """Run the shared one-command loop against a real executor or a world model."""
    run = TerminalAgentRun()
    cwd = await executor.initial_cwd()
    repeated = 0
    previous_command: Optional[str] = None

    for _ in range(max(1, max_steps)):
        response = chat.chat(
            _SYSTEM,
            _prompt(instruction, cwd, run),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        run.total_tokens += _chat_tokens(chat)
        action = _extract_object(response or "")
        if not action:
            run.final_answer = f"Unparseable agent response: {(response or '')[:500]}"
            break
        if action.get("finish") is True:
            run.final_answer = str(action.get("summary", "Task complete."))
            run.done = True
            break
        command = action.get("command")
        if not isinstance(command, str) or not command.strip():
            run.final_answer = f"Invalid command action: {action}"
            break
        command = command.strip()

        if command == previous_command:
            repeated += 1
        else:
            repeated = 0
        previous_command = command
        if repeated >= 2:
            result = TerminalCommandResult(
                stdout="Repeated command suppressed. Choose a different action.",
                return_code=1,
                cwd=cwd,
                predicted=True,
            )
        else:
            result = await executor.execute(command, cwd)
        cwd = result.cwd or cwd
        run.commands.append(command)
        run.observations.append(result.observation())
        run.return_codes.append(result.return_code)
        run.n_steps += 1

    if not run.final_answer:
        run.final_answer = "Stopped at the command-step limit."
    return run


async def replay_terminal_commands(
    executor: TerminalExecutor,
    commands: List[str],
) -> List[TerminalCommandResult]:
    """Commit a selected trajectory exactly once to the official task environment."""
    cwd = await executor.initial_cwd()
    results: List[TerminalCommandResult] = []
    for command in commands:
        result = await executor.execute(command, cwd)
        cwd = result.cwd or cwd
        results.append(result)
    return results


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def terminalbench_settings(overrides: Optional[dict] = None) -> dict:
    """Resolve harness-neutral TTS settings from env vars and optional agent kwargs."""
    overrides = overrides or {}
    method = str(overrides.get(
        "method", os.environ.get("TTS_TERMINALBENCH_METHOD", "ours"))).lower()
    if method not in {"b0", "b1", "ours"}:
        raise ValueError("Terminal-Bench method must be b0, b1, or ours")
    return {
        "method": method,
        "n": max(1, int(overrides.get(
            "n", os.environ.get("TTS_TERMINALBENCH_N", "4")))),
        "max_steps": max(1, int(overrides.get(
            "max_steps", os.environ.get("TTS_TERMINALBENCH_MAX_STEPS", "40")))),
        "temperature": float(overrides.get(
            "temperature", os.environ.get("TTS_TERMINALBENCH_TEMPERATURE", "0.7"))),
        "seed": int(overrides.get(
            "seed", os.environ.get("TTS_TERMINALBENCH_SEED", "0"))),
        "command_timeout": _env_int("TTS_TERMINALBENCH_COMMAND_TIMEOUT_S", 300),
        "wm_turns": _env_int("TTS_TERMINALBENCH_WM_TURNS", 5),
        "verifier_samples": _env_int("TTS_TERMINALBENCH_VERIFIER_SAMPLES", 1),
        "max_tokens": _env_int("TTS_TERMINALBENCH_AGENT_MAX_TOKENS", 8000),
        "wm_max_tokens": _env_int("TTS_TERMINALBENCH_WM_MAX_TOKENS", 1000),
        "verifier_temperature": _env_float(
            "TTS_TERMINALBENCH_VERIFIER_TEMPERATURE", 0.0),
    }


async def run_terminalbench_tts(
    instruction: str,
    real_executor: TerminalExecutor,
    chat_factory: Callable[[int, float], Any],
    *,
    settings: Optional[dict] = None,
) -> TerminalBenchDecision:
    """Sample with the world model, select one trajectory, and commit it once."""
    cfg = terminalbench_settings(settings)
    method = cfg["method"]
    if method == "b0":
        run = await run_terminalbench_agent(
            instruction,
            real_executor,
            chat_factory(cfg["seed"], cfg["temperature"]),
            max_steps=cfg["max_steps"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        return TerminalBenchDecision(
            method="b0",
            selected_index=0,
            rollouts=[TerminalBenchRollout(0, run, 1.0)],
            committed=True,
        )

    from .terminalbench_wm import TerminalBenchWMPredictor, TerminalBenchVerifier

    rollouts: List[TerminalBenchRollout] = []
    n = cfg["n"]
    for index in range(n):
        seed = cfg["seed"] + index
        wm_chat = chat_factory(seed, 0.0)
        predictor = TerminalBenchWMPredictor(
            instruction,
            real_executor,
            wm_chat,
            recursive=method == "ours",
            max_turns=cfg["wm_turns"],
            max_tokens=cfg["wm_max_tokens"],
        )
        run = await run_terminalbench_agent(
            instruction,
            predictor,
            chat_factory(seed, cfg["temperature"]),
            max_steps=cfg["max_steps"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        verifier = TerminalBenchVerifier(
            instruction,
            predictor,
            wm_chat,
            samples=cfg["verifier_samples"],
            temperature=cfg["verifier_temperature"],
        )
        confidence = await verifier.predict_success(run)
        run.total_tokens += predictor.total_tokens
        rollouts.append(TerminalBenchRollout(
            index,
            run,
            confidence,
            n_reads=predictor.exact_reads + predictor.probe_reads,
            n_spawns=predictor.probe_calls,
            max_depth=1 if predictor.probe_calls else 0,
        ))

    selected = max(
        rollouts,
        key=lambda rollout: (
            rollout.run.done and bool(rollout.run.commands),
            rollout.confidence,
            -rollout.index,
        ),
    )
    eligible = selected.run.done and bool(selected.run.commands)
    commit_results = (
        await replay_terminal_commands(real_executor, selected.run.commands)
        if eligible
        else []
    )
    return TerminalBenchDecision(
        method=method,
        selected_index=selected.index,
        rollouts=rollouts,
        committed=eligible,
        commit_results=commit_results,
    )


__all__ = [
    "TerminalAgentRun",
    "TerminalBenchDecision",
    "TerminalBenchRollout",
    "TerminalCommandResult",
    "TerminalExecutor",
    "replay_terminal_commands",
    "run_terminalbench_agent",
    "run_terminalbench_tts",
    "terminalbench_settings",
]
