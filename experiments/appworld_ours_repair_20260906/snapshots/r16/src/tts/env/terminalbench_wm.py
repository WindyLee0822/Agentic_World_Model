"""Read-only, agentic world model for Terminal-Bench command trajectories."""

from __future__ import annotations

import posixpath
import re
import shlex
from typing import Any, List, Optional

from ..simulator.transition_log import TransitionLog
from .terminalbench_agent import TerminalAgentRun, TerminalCommandResult, TerminalExecutor

_READ_COMMANDS = {
    "[", "basename", "cat", "cmp", "cut", "date", "df", "diff", "dirname", "du",
    "echo", "env", "file", "find", "git", "grep", "head", "id", "jq", "ls",
    "md5sum", "printenv", "printf", "ps", "pwd", "readlink", "realpath", "rg",
    "sed", "sha1sum", "sha256sum", "sort", "stat", "strings", "tail", "test",
    "tr", "tree", "type", "uname", "wc", "which", "whoami",
}
_GIT_READ_SUBCOMMANDS = {
    "diff", "grep", "log", "ls-files", "rev-parse", "show", "status",
}
_SEPARATORS = {"|", "||", "&&", ";", "&"}
_REDIRECTIONS = {"<", "<<", "<<<", ">", ">>", ">|", "<>", ">&", "<&"}
_SHELL_PUNCTUATION = re.compile(r"^[;&|<>]+$")


def _tokens(command: str) -> Optional[List[str]]:
    if not command or re.search(r"[\r\n`]|\$\(|\$\{|<\(|>\(", command):
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return None


def standalone_cd_target(command: str) -> Optional[str]:
    tokens = _tokens(command)
    if tokens and len(tokens) == 2 and tokens[0] == "cd" and tokens[1] not in _SEPARATORS:
        return tokens[1]
    return None


def _sed_substitution_is_read_only(script: str) -> bool:
    if len(script) < 4 or script[0] != "s":
        return False
    delimiter = script[1]
    if delimiter.isalnum() or delimiter in {"\\", "\n"}:
        return False
    index = 2
    for _ in range(2):
        escaped = False
        while index < len(script):
            character = script[index]
            index += 1
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == delimiter:
                break
        else:
            return False
    flags = script[index:]
    return bool(re.fullmatch(r"[0-9gIpimM]*", flags))


def _sed_script_is_read_only(script: str) -> bool:
    script = script.strip()
    if not script or "\n" in script or ";" in script:
        return False
    if re.fullmatch(r"(?:\d+|\$)(?:,(?:\d+|\$))?[pqd=]", script):
        return True
    if re.fullmatch(r"/(?:\\.|[^/])+/[pqd=]", script):
        return True
    return _sed_substitution_is_read_only(script)


def _sed_is_read_only(args: List[str]) -> bool:
    scripts: List[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--":
            index += 1
            break
        if argument in {
            "-n", "--quiet", "--silent", "-E", "-r", "--regexp-extended",
            "-u", "--unbuffered", "-z", "--null-data", "-s", "--separate",
        }:
            index += 1
            continue
        if argument in {"-e", "--expression"}:
            if index + 1 >= len(args):
                return False
            scripts.append(args[index + 1])
            index += 2
            continue
        if argument.startswith("--expression="):
            scripts.append(argument.partition("=")[2])
            index += 1
            continue
        if argument.startswith("-"):
            return False
        if not scripts:
            scripts.append(argument)
            index += 1
        break
    return bool(scripts) and all(_sed_script_is_read_only(script) for script in scripts)


def _stage_is_read(stage: List[str]) -> bool:
    if not stage or "/" in stage[0] or stage[0] not in _READ_COMMANDS:
        return False
    command, args = stage[0], stage[1:]
    if command == "git":
        non_options = [arg for arg in args if not arg.startswith("-")]
        if not non_options or non_options[0] not in _GIT_READ_SUBCOMMANDS:
            return False
        return not any(
            arg == "-O"
            or arg.startswith("--output")
            or arg.startswith("--ext-diff")
            or arg.startswith("--textconv")
            or arg.startswith("--open-files-in-pager")
            for arg in args
        )
    if command == "find" and any(
        arg in {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint",
                "-fprint0", "-fprintf", "-fls"}
        for arg in args
    ):
        return False
    if command == "sed":
        return _sed_is_read_only(args)
    if command == "sort" and any(
        arg.startswith("-o")
        or arg.startswith("--output")
        or arg.startswith("--temporary-directory")
        or arg.startswith("--compress-program")
        for arg in args
    ):
        return False
    if command == "date" and any(
        arg.startswith("-s") or arg.startswith("--set") for arg in args
    ):
        return False
    if command == "printf" and any(
        arg == "-v" or arg.startswith("--var") for arg in args
    ):
        return False
    if command == "rg" and any(
        arg == "--pre" or arg.startswith("--pre=") for arg in args
    ):
        return False
    if command == "tree" and any(
        arg.startswith("-o") or arg.startswith("--output") for arg in args
    ):
        return False
    if command == "env" and args:
        return False
    return True


def shell_command_is_read_only(command: str) -> bool:
    """True only when every shell stage is statically proven non-mutating."""
    if standalone_cd_target(command) is not None:
        return True
    tokens = _tokens(command)
    if not tokens:
        return False
    stages: List[List[str]] = []
    stage: List[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _SEPARATORS:
            if token not in {"|", "&&"} or not _stage_is_read(stage):
                return False
            stages.append(stage)
            stage = []
            index += 1
            continue
        if token in _REDIRECTIONS:
            if index + 1 >= len(tokens):
                return False
            target = tokens[index + 1]
            if token == "<":
                pass
            elif target != "/dev/null":
                return False
            index += 2
            continue
        if _SHELL_PUNCTUATION.fullmatch(token):
            return False
        stage.append(token)
        index += 1
    if not _stage_is_read(stage):
        return False
    stages.append(stage)
    return bool(stages)


def _resource_hint(command: str, cwd: str) -> List[str]:
    tokens = _tokens(command) or []
    paths = []
    for token in tokens[1:]:
        if token.startswith("-") or token in _SEPARATORS or token in _REDIRECTIONS:
            continue
        if token in {".", ".."} or "/" in token:
            paths.append(posixpath.normpath(
                token if token.startswith("/") else posixpath.join(cwd, token)))
    return paths[:8] or [cwd]


def _parse_return_code(text: str) -> int:
    match = re.search(r"return[_ ]code\s*[:=]\s*(-?\d+)", text or "", re.I)
    return int(match.group(1)) if match else 0


def _parse_confidence(text: str) -> float:
    match = re.search(
        r"(?:success[_ ]probability|success[_ ]score)\s*[:=]\s*"
        r"(0(?:\.\d+)?|1(?:\.0+)?)",
        text or "",
        re.I,
    )
    if match:
        return min(1.0, max(0.0, float(match.group(1))))
    lower = (text or "").strip().lower()
    match = re.search(
        r"(?:confidence|score)\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        text or "",
        re.I,
    )
    if match:
        confidence = min(1.0, max(0.0, float(match.group(1))))
        if "failure" in lower:
            return 1.0 - confidence
        return confidence
    if "failure" in lower:
        return 0.0
    if "success" in lower:
        return 1.0
    return 0.0


class TerminalBenchWMQuery:
    """Read-only grounding interface for the Terminal-Bench world-model subagent.

    It exposes the same paper-level tools as AppWorld and OfficeBench:
    ``READ`` logical resource state, ``LIST`` the namespace, ``CALL`` a safe read-only domain
    operation, and ``TRACE`` predicted transitions. ``RUN`` remains an undocumented compatibility
    alias for ``CALL``.
    """

    def __init__(self, executor: TerminalExecutor, predictor: "TerminalBenchWMPredictor") -> None:
        self._executor = executor
        self._predictor = predictor
        self.reads = 0
        self.lists = 0
        self.calls = 0
        self.traces = 0

    def _logical_evidence(self, observation: str, resources: List[str]) -> str:
        traces = []
        for resource in dict.fromkeys(resources):
            if self._predictor.transitions.entries(resource, applied_only=True):
                traces.append(
                    f"{resource}:\n{self._predictor.transitions.render(resource)}"
                )
        if not traces:
            return observation
        return (
            observation
            + "\n\nPREDICTED LOGICAL TRANSITIONS "
            "(apply these over the pristine evidence):\n"
            + "\n".join(traces)
        )

    async def _execute_read(
        self,
        command: str,
        cwd: str,
        resources: List[str],
    ) -> str:
        self._predictor.probe_calls += 1
        self._predictor.probe_reads += 1
        result = await self._executor.execute(command, cwd)
        return self._logical_evidence(result.observation(cap=10000), resources)

    async def read(self, path: str, cwd: str) -> str:
        path = str(path or "").strip()
        if not path:
            return "ERROR: READ requires a path"
        self.reads += 1
        command = f"sed -n '1,240p' -- {shlex.quote(path)}"
        resource = posixpath.normpath(
            path if path.startswith("/") else posixpath.join(cwd, path)
        )
        return await self._execute_read(command, cwd, [resource])

    async def list(self, path: str, cwd: str) -> str:
        self.lists += 1
        target = str(path or ".").strip() or "."
        command = (
            f"find {shlex.quote(target)} -maxdepth 2 -mindepth 1 "
            "-printf '%y %p\\n' | sort | head -240"
        )
        directory = posixpath.normpath(
            target if target.startswith("/") else posixpath.join(cwd, target)
        )
        resources = [
            resource
            for resource in self._predictor.transitions.resources()
            if resource == directory or resource.startswith(directory.rstrip("/") + "/")
        ]
        return await self._execute_read(command, cwd, resources)

    async def call(self, command: str, cwd: str) -> str:
        command = str(command or "").strip()
        if not shell_command_is_read_only(command):
            return "ERROR: CALL accepts only statically proven read-only commands"
        self.calls += 1
        return await self._execute_read(command, cwd, _resource_hint(command, cwd))

    def trace(self, resource: str) -> str:
        self.traces += 1
        self._predictor.probe_calls += 1
        return self._predictor.transitions.render(str(resource or "").strip() or None)

    async def __call__(self, request: str, cwd: str) -> str:
        request = str(request or "").strip()
        verb, _, argument = request.partition(" ")
        verb = verb.upper()
        argument = argument.strip()
        if verb == "READ":
            return await self.read(argument, cwd)
        if verb == "LIST":
            return await self.list(argument, cwd)
        if verb in {"CALL", "RUN"}:
            return await self.call(argument, cwd)
        if verb == "TRACE":
            return self.trace(argument)
        return "ERROR: use READ <path>, LIST <path>, CALL <read-only command>, or TRACE [resource]"


def _parse_agentic(text: str, terminal: str):
    """Parse one Terminal-Bench WM turn into shared grounding tools or a terminal answer."""
    if re.search(rf"(?im)^\s*{re.escape(terminal)}", text or ""):
        return [], [], [], [], text
    reads: List[str] = []
    lists: List[str] = []
    calls: List[str] = []
    traces: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        command = stripped.upper()
        if command.startswith("READ "):
            path = stripped[5:].strip().strip("\"'")
            if path:
                reads.append(path)
        elif command.startswith("LIST "):
            path = stripped[5:].strip().strip("\"'")
            if path:
                lists.append(path)
        elif command.startswith("CALL "):
            shell_command = stripped[5:].strip()
            if shell_command:
                calls.append(shell_command)
        elif command.startswith("RUN "):
            shell_command = stripped[4:].strip()
            if shell_command:
                calls.append(shell_command)
        elif command.startswith("TRACE"):
            traces.append(stripped[5:].strip().strip("\"'"))
    return reads, lists, calls, traces, None


async def run_agentic_read_loop(
    complete,
    query: TerminalBenchWMQuery,
    header: str,
    final_instr: str,
    cwd: str,
    *,
    terminal: str,
    max_turns: int = 5,
    max_tokens: int = 1000,
) -> str:
    """Run the AppWorld/OfficeBench-style ``READ``/``LIST``/``CALL``/``TRACE`` loop."""
    known: dict[str, str] = {}
    protocol = (
        "You may inspect the CURRENT LOGICAL Terminal-Bench state before answering. The real "
        "container remains pristine except for statically proven reads; predicted writes are recorded "
        "in an append-only transition log. Each turn output one or more grounding commands:\n"
        "  READ <path>                    (read a file from the pristine container)\n"
        "  LIST <path>                    (list the pristine namespace)\n"
        "  CALL <read-only shell command> (invoke a statically proven read-only operation)\n"
        "  TRACE [resource]               (inspect predicted transitions)\n"
        f"OR, when enough evidence is gathered, output:\n  {final_instr}\n"
        f"Use only the minimum evidence needed; you have at most {max_turns} read turns."
    )
    for _turn in range(max_turns):
        facts = "\n".join(f"{key}: {value}" for key, value in known.items()) or "(nothing read yet)"
        response = complete(
            f"{header}\n\n{protocol}\n\nFacts gathered so far:\n{facts}\n\nYour move:",
            max_tokens,
        )
        reads, lists, calls, traces, terminal_response = _parse_agentic(response, terminal)
        if terminal_response is not None:
            return terminal_response
        acted = False
        for path in lists[:3]:
            known[f"list {path}"] = await query.list(path, cwd)
            acted = True
        for path in reads[:3]:
            known[f"read {path}"] = await query.read(path, cwd)
            acted = True
        for command in calls[:4]:
            known[f"call {command}"] = await query.call(command, cwd)
            acted = True
        for resource in traces[:3]:
            known[f"trace {resource}"] = query.trace(resource)
            acted = True
        if not acted:
            return response

    facts = "\n".join(f"{key}: {value}" for key, value in known.items()) or "(nothing read)"
    return complete(
        f"{header}\n\nFacts gathered:\n{facts}\n\nYou are out of read turns. {final_instr}",
        max_tokens,
    )


class TerminalBenchWMPredictor:
    """Execute pristine reads exactly; predict writes and every dirty read."""

    def __init__(
        self,
        instruction: str,
        executor: TerminalExecutor,
        chat: Any,
        *,
        recursive: bool,
        max_turns: int = 5,
        max_tokens: int = 1000,
    ) -> None:
        self.instruction = instruction
        self.executor = executor
        self.chat = chat
        self.recursive = recursive
        self.max_turns = max(1, max_turns)
        self.max_tokens = max_tokens
        self.transitions = TransitionLog()
        self.query = TerminalBenchWMQuery(executor, self)
        self._dirty = False
        self._history: List[tuple[str, str]] = []
        self.total_tokens = 0
        self.exact_reads = 0
        self.probe_reads = 0
        self.probe_calls = 0

    async def initial_cwd(self) -> str:
        return await self.executor.initial_cwd()

    async def execute(self, command: str, cwd: str) -> TerminalCommandResult:
        exact_read = shell_command_is_read_only(command) and not self._dirty
        if exact_read:
            result = await self.executor.execute(command, cwd)
            self.exact_reads += 1
            self._history.append((command, result.observation()))
            return result

        predicted = await self._predict_observation(command, cwd)
        target = standalone_cd_target(command)
        next_cwd = cwd
        if target and predicted.return_code == 0:
            next_cwd = posixpath.normpath(
                target if target.startswith("/") else posixpath.join(cwd, target))
        predicted.cwd = next_cwd
        self._dirty = True
        observation = predicted.observation()
        self.transitions.record(
            command,
            observation,
            _resource_hint(command, cwd),
            applied=predicted.return_code == 0,
        )
        self._history.append((command, observation))
        return predicted

    async def _predict_observation(self, command: str, cwd: str) -> TerminalCommandResult:
        history = "\n".join(
            f"{index}. {cmd} -> {obs[:1200]}"
            for index, (cmd, obs) in enumerate(self._history[-12:])
        ) or "(none)"
        prompt = (
            f"TASK:\n{self.instruction}\n\nCURRENT CWD: {cwd}\n"
            f"COMMAND TO SIMULATE (DO NOT EXECUTE):\n{command}\n\n"
            f"PRIOR TRAJECTORY:\n{history}\n\n"
            "Predict the command result and its effect on the logical container state."
        )
        final_instr = (
            "OBSERVATION:\nreturn_code: <integer>\nstdout: <text>\nstderr: <text>\n"
            "Be concise and do not claim the command was executed."
        )
        text = await self._agentic_completion(
            prompt,
            cwd,
            terminal="OBSERVATION:",
            final_instr=final_instr,
        )
        return TerminalCommandResult(
            stdout=text,
            return_code=_parse_return_code(text),
            cwd=cwd,
            predicted=True,
        )

    async def _agentic_completion(
        self,
        prompt: str,
        cwd: str,
        *,
        terminal: str,
        final_instr: str,
        temperature: float = 0.0,
    ) -> str:
        def complete(user_prompt: str, max_tokens: int) -> str:
            response = self.chat.chat(
                "You are a conservative read-grounded terminal world model. Never execute a command "
                "that may mutate state. Use the provided read-only probes when evidence is needed.",
                user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.total_tokens += int(getattr(self.chat, "last_usage", 0) or 0)
            return response

        if not self.recursive:
            return complete(f"{prompt}\n\nEnd with:\n{final_instr}", self.max_tokens)
        return await run_agentic_read_loop(
            complete,
            self.query,
            prompt,
            final_instr,
            cwd,
            terminal=terminal,
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
        )


class TerminalBenchVerifier:
    """Agentic success judge over predicted command transitions."""

    def __init__(
        self,
        instruction: str,
        predictor: TerminalBenchWMPredictor,
        chat: Any,
        *,
        samples: int = 1,
        temperature: float = 0.0,
    ) -> None:
        self.instruction = instruction
        self.predictor = predictor
        self.chat = chat
        self.samples = max(1, samples)
        self.temperature = temperature
        self.total_tokens = 0

    async def predict_success(self, run: TerminalAgentRun) -> float:
        if not run.done or not run.commands:
            return 0.0
        trajectory = "\n".join(
            f"{index}. COMMAND: {command}\nOBSERVATION: {run.observations[index][:3000]}"
            for index, command in enumerate(run.commands)
        )
        prompt = (
            f"TASK:\n{self.instruction}\n\nPREDICTED TRAJECTORY:\n{trajectory}\n\n"
            f"PREDICTED TRANSITION LOG:\n{self.predictor.transitions.render()}\n\n"
            f"AGENT FINAL MESSAGE:\n{run.final_answer}\n\n"
            "Judge the predicted resulting container state, not the agent's claim. The official tests "
            "are not available, so be conservative."
        )
        scores = []
        for _ in range(self.samples):
            text = await self.predictor._agentic_completion(
                prompt,
                await self.predictor.initial_cwd(),
                terminal="VERDICT:",
                final_instr=(
                    "VERDICT: SUCCESS or VERDICT: FAILURE\n"
                    "SUCCESS_PROBABILITY: <number from 0 to 1>"
                ),
                temperature=self.temperature,
            )
            scores.append(_parse_confidence(text))
        return sum(scores) / len(scores)


__all__ = [
    "TerminalBenchVerifier",
    "TerminalBenchWMPredictor",
    "TerminalBenchWMQuery",
    "run_agentic_read_loop",
    "shell_command_is_read_only",
    "standalone_cd_target",
]
