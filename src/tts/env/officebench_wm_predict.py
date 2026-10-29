"""Predictive, write-safe world model for OfficeBench (coherent with the τ²/AppWorld contract).

During sampling the agent interacts with a **world model**, not the real environment. Reads are exact
only in the explicit clean-read ablation; under predict-all, recursive Ours grounds through immutable
object reads/listings and has the world model generate the result, while flat B1 predicts from fixed
context. Candidate actions are **LLM-predicted and NEVER executed**, and a **read after a write**
returns predicted post-write state.
The real environment is therefore never mutated while the ``N`` candidates are drawn; only the
selected trajectory is committed.

As in τ², nontrivial predictions use a bounded world-model subagent over immutable object ``READ`` and
``LIST`` operations plus ``TRACE`` over the predicted transition log. The exact-clean-read ablation may
also expose ``CALL`` for safe read-only domain operations. The logical state is the immutable pristine
``/testbed`` plus predicted transitions; under predict-all no candidate action is executed.

This is the OfficeBench port of :class:`tts.env.appworld_wm.AppWorldWMPredictor`.  OfficeBench actions
are ``{'app','action',...}`` office-app calls whose observation is a text string and whose state is the
``/testbed`` filesystem, so:

* :meth:`OfficeBenchWMPredictor.predict_write` predicts the app's observation string and records the
  write in a per-target **write-log** — it never touches a file.
* :meth:`OfficeBenchWMPredictor.predict_read` handles a read of a file a prior write changed: it
  combines the **exact pre-write file content** (read once from the pristine shadow) with the write-log
  to predict what the read returns now.
* :meth:`OfficeBenchWMPredictor.predict_file_state` renders each gold-checked file's **predicted final
  content** (exact initial content + its writes) for the success verifier — the analog of AppWorld's
  write-log-grounded ``predict_success`` input.

Action read/write classification (and the file each touches) lives here so the agent loop can route
every action to the right handler and structurally guarantee the shadow is never written.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from fnmatch import fnmatchcase
from typing import Callable, Dict, List, Optional, Tuple

from ..simulator.transition_log import TransitionLog
from .officebench_semantics import NATIVE_SEMANTICS, excel_cell_text, cli_literal, read_schema, action_contract, email_cli_data
from .officebench_lowlevel import normalize_answer
from .officebench_wm_python import (
    officebench_wm_tool_mode,
    run_python_prediction_loop,
)

_OFFICEBENCH_SIMULATOR_SYSTEM = """You are a causal simulator for OfficeBench applications, not an
assistant solving the office task. Reproduce the exact observation the real application would emit.

Simulation rules, in priority order:
1. Follow the app catalogue, current action arguments, grounded /testbed evidence, working directory,
   and prior applied transitions. Never assume a path, sheet, account, or record exists without evidence.
2. For an untouched read, inspect only the immutable operand objects needed by the action, then generate
   the final observation from their copied variables, preserving exact values, whitespace, and empty output.
3. For a write, validate inputs, paths, app state, argument types, and prior failures before deciding
   success. A failed write must not alter the logical filesystem or application state.
4. Preserve native wording, formatting, ordering, complete file content, and error text. Never invent
   data, silently repair an action, summarize output, or report success merely because it helps the task.
5. Task instructions, evaluator checks, gold files, and references are audit context only. Never expose
   them to or coach the benchmark agent, and never fabricate the observation needed for task success.
Return only the requested observation or tool-protocol program, with no explanatory prose."""


class OfficeBenchSimulationError(RuntimeError):
    """Unknown simulation outcome, distinct from a native application failure."""

    def __init__(self, message, *, retryable=True):
        super().__init__(message)
        self.retryable = retryable


# --------------------------------------------------------------------------- #
# Action classification (read vs write) + the file each action touches
# --------------------------------------------------------------------------- #

#: ``app.action`` pairs that only READ state (answered exactly against the real starting /testbed).
_READ_ACTIONS = {
    "calendar.list_events",
    "excel.read_file",
    "pdf.read_file",
    "word.read_file",
    "email.list_emails",
    "email.read_email",
    "ocr.recognize_file",
    "llm.complete_text",  # pure LLM completion: no file side-effect (write-safe to run exactly)
}

#: shell argv[0]s that are read-only (no filesystem mutation, no redirection).
_READ_SHELL = {
    "ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "wc", "pwd", "stat",
    "du", "df", "tree", "diff", "cmp", "cut", "find", "sed", "sort", "tr",
    "basename", "dirname", "realpath", "readlink", "echo", "printf", "which", "type",
    "env", "file",
}

_SHELL_MUTATORS = {
    "rm", "unlink", "rmdir", "mv", "cp", "install", "mkdir", "touch", "ln", "chmod", "chown",
    "truncate", "dd", "tee", "convert", "magick", "libreoffice", "soffice",
}
_SHELL_SEPARATORS = {"|", "||", "&&", ";"}


def _short_option_present(args: List[str], option: str) -> bool:
    return any(
        arg.startswith("-") and not arg.startswith("--") and option in arg[1:]
        for arg in args
    )


def _output_option_value(args: List[str]) -> Optional[str]:
    for index, arg in enumerate(args):
        if arg == "-o" or arg == "--output":
            return args[index + 1] if index + 1 < len(args) else None
        if arg.startswith("--output="):
            return arg.split("=", 1)[1]
        if arg.startswith("-") and not arg.startswith("--") and "o" in arg[1:]:
            suffix = arg[arg.index("o") + 1:]
            return suffix or (args[index + 1] if index + 1 < len(args) else None)
    return None


def _target_directory_option(args: List[str]) -> Tuple[Optional[str], set[int]]:
    for index, arg in enumerate(args):
        if arg in ("-t", "--target-directory"):
            if index + 1 < len(args):
                return args[index + 1], {index, index + 1}
            return None, {index}
        if arg.startswith("--target-directory="):
            return arg.split("=", 1)[1], {index}
        if arg.startswith("-") and not arg.startswith("--") and "t" in arg[1:]:
            suffix = arg[arg.index("t") + 1:]
            if suffix:
                return suffix, {index}
            if index + 1 < len(args):
                return args[index + 1], {index, index + 1}
            return None, {index}
    return None, set()


def _option_path_values(tokens: List[str]) -> List[str]:
    command, args = tokens[0], tokens[1:]
    specs = {
        "grep": (("f",), ("--file", "--exclude-from")),
        "egrep": (("f",), ("--file", "--exclude-from")),
        "fgrep": (("f",), ("--file", "--exclude-from")),
        "find": ((), ("-files0-from",)),
        "file": (("m", "f"), ("--magic-file", "--files-from")),
        "sort": ((), ("--files0-from",)),
        "diff": ((), ("--from-file", "--to-file")),
        "du": ((), ("--files0-from",)),
        "wc": ((), ("--files0-from",)),
    }
    shorts, options = specs.get(command, ((), ()))
    out = []
    for index, arg in enumerate(args):
        for short in shorts:
            if arg.startswith("-") and not arg.startswith("--") and short in arg[1:]:
                suffix = arg[arg.index(short) + 1:]
                if suffix:
                    out.append(suffix)
                elif index + 1 < len(args):
                    out.append(args[index + 1])
        for option in options:
            if arg == option and index + 1 < len(args):
                out.append(args[index + 1])
            elif arg.startswith(option + "="):
                out.append(arg.split("=", 1)[1])
    return out


def shell_cd_target(command: str) -> Optional[str]:
    """Return the target of one standalone ``cd <path>`` command, otherwise ``None``."""
    if re.search(r"[\r\n;&|<>`$*?\[\]{}()]", str(command)):
        return None
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return None
    if len(tokens) != 2 or tokens[0] != "cd":
        return None
    if re.search(r"[;&|<>]", str(command)):
        return None
    return tokens[1]


def _shell_tokens(command: str) -> Optional[List[str]]:
    try:
        lexer = shlex.shlex(
            str(command), posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return None


def _sed_stage_is_read(tokens: List[str]) -> bool:
    scripts = []
    positional = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in ("-n", "--quiet", "--silent", "-E", "-r", "--regexp-extended"):
            index += 1
            continue
        if token in ("-e", "--expression"):
            if index + 1 >= len(tokens):
                return False
            scripts.append(tokens[index + 1])
            index += 2
            continue
        if token.startswith("--expression="):
            scripts.append(token.split("=", 1)[1])
            index += 1
            continue
        if token.startswith("-"):
            return False
        positional.append(token)
        index += 1
    if not scripts:
        if not positional:
            return False
        scripts.append(positional.pop(0))
    for script in scripts:
        script = script.strip()
        if re.fullmatch(r"(?:\d+(?:,\d+)?|/[^/]*/)?[pdq]", script):
            continue
        if not script.startswith("s") or len(script) < 4:
            return False
        delimiter = script[1]
        positions = []
        escaped = False
        for index, char in enumerate(script[2:], start=2):
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == delimiter:
                positions.append(index)
                if len(positions) == 2:
                    break
        if len(positions) < 2:
            return False
        flags = script[positions[1] + 1:]
        if any(flag not in "gIp0123456789" for flag in flags):
            return False
    return True


def _shell_stage_is_read(tokens: List[str]) -> bool:
    if not tokens or "/" in tokens[0] or tokens[0] not in _READ_SHELL:
        return False
    command = tokens[0]
    args = tokens[1:]
    if command == "find" and any(
            arg in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint",
                    "-fprint0", "-fprintf", "-fls") for arg in args):
        return False
    if command == "find" and any(
            arg == "-files0-from" or arg.startswith("-files0-from=") for arg in args):
        return False
    if command == "find" and args and args[0].startswith("-"):
        return False
    if command == "sed" and not _sed_stage_is_read(tokens):
        return False
    if command == "sort" and any(
            _short_option_present(args, "o")
            or arg.startswith("--output")
            or arg.startswith("--compress-program")
            or arg.startswith("--files0-from")
            or arg == "-T"
            or arg.startswith("--temporary-directory") for arg in args):
        return False
    if command == "tree" and any(
            _short_option_present(args, "o") or arg.startswith("--output") for arg in args):
        return False
    if command == "tail" and any(
            _short_option_present(args, "f")
            or _short_option_present(args, "F")
            or arg == "--retry"
            or arg == "--follow"
            or arg.startswith("--follow=")
            for arg in args):
        return False
    if command == "diff" and any(
            arg.startswith("--output") for arg in args):
        return False
    if command == "file" and any(
            _short_option_present(args, "C")
            or arg == "--compile"
            or _short_option_present(args, "f")
            or arg.startswith("--files-from")
            for arg in args):
        return False
    if command in ("du", "wc") and any(
            arg.startswith("--files0-from") for arg in args):
        return False
    if command == "env" and args:
        return False
    return True


def _shell_stage_operands(tokens: List[str]) -> List[str]:
    command_name = tokens[0]
    args = [token for token in tokens[1:] if token != "--"]
    non_options = [token for token in args if not token.startswith("-")]
    if command_name == "find":
        roots = []
        for token in args:
            if token.startswith("-") or token in ("(", ")", "!"):
                break
            roots.append(token)
        return roots or ["."]
    if command_name in ("ls", "tree"):
        return non_options or ["."]
    if command_name in ("grep", "egrep", "fgrep", "sed"):
        positional = []
        pattern_supplied = False
        consume_next = False
        after_options = False
        pattern_options = (
            {"-e", "--regexp", "-f", "--file"}
            if command_name != "sed"
            else {"-e", "--expression", "-f", "--file"}
        )
        for token in tokens[1:]:
            if consume_next:
                consume_next = False
                continue
            if token == "--":
                after_options = True
                continue
            if not after_options and token in pattern_options:
                pattern_supplied = True
                consume_next = True
                continue
            if not after_options and any(
                    token.startswith(option + "=")
                    for option in pattern_options if option.startswith("--")):
                pattern_supplied = True
                continue
            if not after_options and token.startswith("-") and not token.startswith("--"):
                option_chars = token[1:]
                for option in ("e", "f"):
                    if option not in option_chars:
                        continue
                    pattern_supplied = True
                    suffix = option_chars[option_chars.index(option) + 1:]
                    if not suffix:
                        consume_next = True
                    break
                continue
            if not after_options and token.startswith("-"):
                continue
            positional.append(token)
        file_operands = positional if pattern_supplied else positional[1:]
        if command_name != "sed" and any(
                token == "--recursive"
                or (token.startswith("-") and not token.startswith("--")
                    and any(flag in token[1:] for flag in ("r", "R")))
                for token in tokens[1:]
        ) and not file_operands:
            return ["."]
        return file_operands
    if command_name in (
            "cat", "head", "tail", "wc", "stat", "file", "du", "df", "diff", "cmp",
            "sort", "cut", "realpath", "readlink"):
        if command_name == "du" and not non_options:
            return ["."]
        return non_options
    return []


def _shell_operand_scope(path: str, workdir: str) -> Optional[str]:
    if path == "-":
        return None
    if path.startswith("~") or any(char in path for char in "*?["):
        return None
    return _testbed_path(path, workdir)


def _has_unquoted_shell_expansion(command: str) -> bool:
    """Return whether a command contains expansion syntax outside single quotes."""
    quote = None
    escaped = False
    text = str(command)

    def starts_expansion(index: int) -> bool:
        if index + 1 >= len(text):
            return False
        follower = text[index + 1]
        return follower.isalnum() or follower in "({?*@$#!-_"

    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote == "'":
            if char == "'":
                quote = None
            continue
        if quote == '"':
            if char == '"':
                quote = None
            elif char == "`":
                return True
            elif char == "$" and starts_expansion(index):
                return True
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "`":
            return True
        if char == "$" and starts_expansion(index):
            return True
    return quote is not None


def _readonly_shell_stages(
        command: str, workdir: str = "/testbed") -> Optional[List[List[str]]]:
    """Parse a shell command only when every stage is provably non-mutating."""
    if not command or re.search(r"[\r\n]", str(command)) or _has_unquoted_shell_expansion(command):
        return None
    cd_target = shell_cd_target(command)
    if cd_target is not None:
        return (
            [["cd", cd_target]]
            if _shell_operand_scope(cd_target, workdir) is not None else None
        )
    tokens = _shell_tokens(command)
    if not tokens:
        return None
    stages: List[List[str]] = []
    stage: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _SHELL_SEPARATORS:
            if token not in ("|", "&&"):
                return None
            if not stage or not _shell_stage_is_read(stage):
                return None
            stages.append(stage)
            stage = []
            i += 1
            continue
        if token == "&":
            return None
        if token in ("<", "<<", "<<<", ">", ">>", ">|", "<>"):
            if i + 1 >= len(tokens):
                return None
            target = tokens[i + 1]
            fd = stage[-1] if stage and stage[-1].isdigit() else None
            if fd is not None:
                stage.pop()
            if token == "<":
                if _shell_operand_scope(target, workdir) is None:
                    return None
            elif target != "/dev/null":
                return None
            i += 2
            continue
        if token in (">&", "<&"):
            if i + 1 >= len(tokens) or not tokens[i + 1].isdigit():
                return None
            if stage and stage[-1].isdigit():
                stage.pop()
            i += 2
            continue
        if token and all(char in ";&|<>" for char in token):
            return None
        stage.append(token)
        i += 1
    if not stage or not _shell_stage_is_read(stage):
        return None
    stages.append(stage)
    for read_stage in stages:
        for operand in _shell_stage_operands(read_stage) + _option_path_values(read_stage):
            if _shell_operand_scope(operand, workdir) is None:
                return None
    return stages


def _emulate_inventory_pipeline(
    stages: List[List[str]],
    objects: List[Dict],
    workdir: str = "/testbed",
) -> Optional[str]:
    """Evaluate common read-only discovery pipelines from a typed immutable inventory."""
    if not stages or not objects:
        return "" if stages else None

    lines: List[str] = []
    first = stages[0]
    if first[0] == "find":
        roots = _shell_stage_operands(first)
        if len(roots) > 1:
            return None
        root = _testbed_path(roots[0] if roots else ".", workdir)
        if root is None:
            return None
        root = os.path.normpath(root)
        wanted_type = None
        max_depth = None
        min_depth = 0
        name_patterns: List[Tuple[str, bool]] = []
        name_or = False
        index = 1
        while index < len(first):
            token = first[index]
            if token == "-type" and index + 1 < len(first):
                wanted_type = {"f": "file", "d": "directory"}.get(first[index + 1])
                if wanted_type is None:
                    return None
                index += 2
                continue
            if token in ("-maxdepth", "-mindepth") and index + 1 < len(first):
                try:
                    depth = int(first[index + 1])
                except ValueError:
                    return None
                if token == "-maxdepth":
                    max_depth = depth
                else:
                    min_depth = depth
                index += 2
                continue
            if token in ("-name", "-iname") and index + 1 < len(first):
                name_patterns.append((first[index + 1], token == "-iname"))
                index += 2
                continue
            if token == "-o":
                name_or = True
                index += 1
                continue
            if token in (root, "-print"):
                index += 1
                continue
            if token in ("-print0", "-a", "(", ")", "!"):
                return None
            if token.startswith("-"):
                return None
            index += 1

        find_objects = (sorted(objects, key=lambda obj: obj['native_order'])
                        if all(isinstance(obj.get('native_order'), int) for obj in objects) else objects)
        for record in find_objects:
            path = os.path.normpath(str(record.get("path", "")))
            if path != root and not path.startswith(root.rstrip("/") + "/"):
                continue
            relative = os.path.relpath(path, root)
            depth = 0 if path == root else len(relative.split(os.sep))
            if depth < min_depth or (max_depth is not None and depth > max_depth):
                continue
            if wanted_type and record.get("type") != wanted_type:
                continue
            name = os.path.basename(path)
            if name_patterns:
                matches = [
                    fnmatchcase(
                        name.lower() if insensitive else name,
                        pattern.lower() if insensitive else pattern,
                    )
                    for pattern, insensitive in name_patterns
                ]
                if not (any(matches) if name_or else all(matches)):
                    continue
            lines.append(path)
    elif first[0] == "ls":
        options = [
            char
            for token in first[1:]
            if token.startswith("-") and not token.startswith("--")
            for char in token[1:]
        ]
        long_options = [token for token in first[1:] if token.startswith("--")]
        if any(char not in ("1", "R") for char in options) or long_options:
            return None
        operands = _shell_stage_operands(first)
        if len(operands) > 1 or "R" in options:
            return None
        root = _testbed_path(operands[0] if operands else ".", workdir)
        if root is None:
            return None
        root = os.path.normpath(root)
        recursive = "R" in options
        for record in objects:
            path = os.path.normpath(str(record.get("path", "")))
            parent = os.path.dirname(path)
            if path == root and record.get('type') == 'file':
                lines.append(operands[0] if operands else path)
                continue
            if (recursive and path.startswith(root.rstrip("/") + "/")) or parent == root:
                lines.append(path if recursive else os.path.basename(path))
    else:
        return None

    for stage in stages[1:]:
        command = stage[0]
        args = stage[1:]
        if command == "sort":
            if any(
                token not in ("-r", "--reverse")
                and not (
                    token.startswith("-")
                    and not token.startswith("--")
                    and set(token[1:]) <= {"r"}
                )
                for token in args
            ):
                return None
            reverse = "-r" in args or "--reverse" in args
            lines = sorted(lines, reverse=reverse)
            continue
        if command in ("grep", "egrep", "fgrep"):
            allowed_short = {"i", "v", "E", "F"}
            allowed_long = {
                "--ignore-case", "--invert-match", "--extended-regexp", "--fixed-strings",
            }
            if any(
                (
                    token.startswith("--") and token not in allowed_long
                ) or (
                    token.startswith("-")
                    and not token.startswith("--")
                    and not set(token[1:]) <= allowed_short
                )
                for token in args
                if token.startswith("-")
            ):
                return None
            insensitive = any(
                token in ("-i", "--ignore-case")
                or (token.startswith("-") and not token.startswith("--") and "i" in token[1:])
                for token in args
            )
            invert = any(
                token in ("-v", "--invert-match")
                or (token.startswith("-") and not token.startswith("--") and "v" in token[1:])
                for token in args
            )
            extended = command in ("egrep",) or any(
                token in ("-E", "--extended-regexp")
                or (token.startswith("-") and not token.startswith("--") and "E" in token[1:])
                for token in args
            )
            fixed = command == "fgrep" or any(
                token in ("-F", "--fixed-strings")
                or (token.startswith("-") and not token.startswith("--") and "F" in token[1:])
                for token in args
            )
            positional = [token for token in args if not token.startswith("-")]
            if len(positional) != 1:
                return None
            pattern = positional[0]
            flags = re.I if insensitive else 0
            if fixed:
                expression = re.escape(pattern)
            elif extended:
                expression = pattern
            else:
                # GNU grep basic regex uses escaped alternation/grouping while Python uses the
                # unescaped forms. Preserve ordinary escaped characters such as ``\.``.
                expression = (
                    pattern
                    .replace(r"\|", "|")
                    .replace(r"\(", "(")
                    .replace(r"\)", ")")
                    .replace(r"\+", "+")
                    .replace(r"\?", "?")
                )
            try:
                matcher = re.compile(expression, flags)
            except re.error:
                return None
            lines = [line for line in lines if bool(matcher.search(line)) != invert]
            continue
        if command in ("head", "tail"):
            count = 10
            consumed = set()
            for index, token in enumerate(args):
                value = None
                if token == "-n" and index + 1 < len(args):
                    value = args[index + 1]
                    consumed.update((index, index + 1))
                elif re.fullmatch(r"-\d+", token):
                    value = token[1:]
                    consumed.add(index)
                if value is not None:
                    if value.startswith("+"):
                        return None
                    try:
                        count = max(0, int(value))
                    except ValueError:
                        return None
                    break
            if any(index not in consumed for index in range(len(args))):
                return None
            lines = lines[:count] if command == "head" else lines[-count:]
            continue
        if command == "sed":
            if "-n" not in args and "--quiet" not in args and "--silent" not in args:
                return None
            scripts = [
                token for token in args
                if not token.startswith("-") and token not in ("--",)
            ]
            if not scripts:
                return None
            script = scripts[0]
            match = re.fullmatch(r"(\d+)(?:,(\d+))?p", script)
            if match:
                start = int(match.group(1))
                end = int(match.group(2) or start)
                lines = lines[max(0, start - 1):end]
                continue
            match = re.fullmatch(r"/(.*)/p", script)
            if match:
                try:
                    matcher = re.compile(match.group(1))
                except re.error:
                    return None
                lines = [line for line in lines if matcher.search(line)]
                continue
            return None
        if command == "wc" and args == ["-l"]:
            lines = [str(len(lines))]
            continue
        return None
    return "\n".join(lines)


def _shell_is_read(command: str, workdir: str = "/testbed") -> bool:
    """Whether every shell stage is statically proven to leave environment state unchanged."""
    return _readonly_shell_stages(command, workdir) is not None


def _shell_stage_is_definite_write(tokens: List[str]) -> bool:
    if not tokens:
        return False
    name, args = tokens[0], tokens[1:]
    if name in _SHELL_MUTATORS:
        return True
    if name == "sort" and any(
            _short_option_present(args, "o") or arg.startswith("--output") for arg in args):
        return True
    if name == "tree" and any(
            _short_option_present(args, "o") or arg.startswith("--output") for arg in args):
        return True
    if name == "diff" and any(arg.startswith("--output") for arg in args):
        return True
    if name == "file" and (
            _short_option_present(args, "C") or "--compile" in args):
        return True
    if name == "sed" and any(
            _short_option_present(args, "i") or arg.startswith("--in-place")
            for arg in args):
        return True
    if name == "find" and any(
            arg in ("-delete", "-fprint", "-fprint0", "-fprintf", "-fls")
            for arg in args):
        return True
    if name in ("python", "python3"):
        code = " ".join(tokens).lower()
        return bool(re.search(
            r"\.write\s*\(|\.writelines\s*\(|\.save\s*\(|"
            r"open\s*\([^)]*,\s*['\"][wax+]|"
            r"os\.(?:remove|unlink|rename|replace|mkdir|makedirs)\s*\(|"
            r"shutil\.(?:copy|copy2|move|rmtree)\s*\(|"
            r"\.(?:write_text|write_bytes|unlink|mkdir|rename|replace|touch|chmod)\s*\(",
            code,
        ))
    return False


def _shell_is_definite_write(command: str) -> bool:
    tokens = _shell_tokens(command) or []
    if any(token in _SHELL_MUTATORS for token in tokens):
        return True
    for i, token in enumerate(tokens[:-1]):
        if token in (">", ">>", ">|", "<>") and tokens[i + 1] != "/dev/null":
            return True
    stages: List[List[str]] = []
    stage: List[str] = []
    for token in tokens:
        if token in _SHELL_SEPARATORS or token == "&":
            if stage:
                stages.append(stage)
            stage = []
        elif token not in ("<", "<<", "<<<", ">", ">>", ">|", "<>", ">&", "<&"):
            stage.append(token)
    if stage:
        stages.append(stage)
    return any(_shell_stage_is_definite_write(stage) for stage in stages)


def shell_command_kind(action: Dict, workdir: str = "/testbed") -> str:
    """Classify shell actions as exact ``read``, predicted ``write``, or safe ``blocked``."""
    command = action.get("command", "")
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    if _shell_is_read(str(command), workdir):
        return "read"
    if _shell_is_definite_write(str(command)):
        return "write"
    return "blocked"


def action_execution_kind(action: Dict, workdir: str = "/testbed") -> str:
    """Return ``read``, ``write``, or ``blocked`` without executing the action."""
    app = action.get("app")
    act = action.get("action")
    if app == "system":
        return "read"
    if app == "shell":
        return shell_command_kind(action, workdir)
    if app == "llm" and act == "complete_text":
        prompt = action.get("prompt")
        # OfficeBench interpolates this value into a shell command delimited by single quotes.
        # Unsafe prompts are predicted by the WM instead of reaching that shell.
        return (
            "read"
            if isinstance(prompt, str) and "'" not in prompt and "\x00" not in prompt
            else "write"
        )
    if f"{app}.{act}" not in _READ_ACTIONS:
        return "write"
    for key in _PATH_ARGUMENT_KEYS:
        if action.get(key) and _testbed_path(str(action[key]), workdir) is None:
            return "blocked"
    for key in ("username", "user", "email_id"):
        if action.get(key) is not None and not _safe_call_value(action[key]):
            return "blocked"
    return "read"


def is_write_action(action: Dict, workdir: str = "/testbed") -> bool:
    """True if executing ``action`` would MUTATE the environment (so the WM must predict, not run it)."""
    return action_execution_kind(action, workdir) != "read"


def _norm_path(path: str, workdir: str = "/testbed") -> str:
    """Canonical resource key relative to ``/testbed``, resolving ``.``/``..`` and workdir."""
    safe = _testbed_path(path, workdir)
    if safe is None:
        return ""
    return "" if safe == "/testbed" else os.path.relpath(safe, "/testbed")


def _derive_output(path: str, ext: str, workdir: str = "/testbed") -> str:
    base = _norm_path(path, workdir)
    stem = base.rsplit(".", 1)[0] if "." in base else base
    return f"{stem}.{ext}"


_PATH_ARGUMENT_KEYS = (
    "file_path",
    "pdf_file_path",
    "word_file_path",
    "excel_file_path",
    "image_file_path",
    "path",
    "input_file",
    "output_file",
)


def _shell_delete_spec(action: Dict, workdir: str) -> Optional[Tuple[List[str], bool]]:
    command = action.get("command", "")
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    if re.search(r"[;&|<>]", str(command)):
        return None
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return None
    if not tokens or tokens[0] not in ("rm", "unlink", "rmdir"):
        return None
    paths = []
    recursive = tokens[0] == "rmdir"
    after_options = False
    for token in tokens[1:]:
        if token == "--":
            after_options = True
            continue
        if not after_options and token.startswith("-"):
            if "r" in token[1:] or "R" in token[1:]:
                recursive = True
            continue
        if any(char in token for char in "*?[]{}$`"):
            return None
        paths.append(_norm_path(token, workdir))
    return (paths, recursive) if paths and all(paths) else None


def _shell_delete_paths(action: Dict, workdir: str) -> List[str]:
    spec = _shell_delete_spec(action, workdir)
    return spec[0] if spec is not None else []


def _shell_move_sources(action: Dict, workdir: str) -> List[str]:
    """Return exact source resources removed by one simple successful ``mv`` command."""
    command = action.get("command", "")
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    if re.search(r"[;&|<>]", str(command)):
        return []
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return []
    if not tokens or tokens[0] != "mv":
        return []
    args = tokens[1:]
    target_dir, consumed = _target_directory_option(args)
    operands = [
        arg for index, arg in enumerate(args)
        if index not in consumed and arg != "--" and not arg.startswith("-")
    ]
    sources = operands if target_dir else operands[:-1]
    if not sources or any(any(char in source for char in "*?[]{}$`~") for source in sources):
        return []
    normalized = [_norm_path(source, workdir) for source in sources]
    return normalized if all(normalized) else []


def _shell_read_target(action: Dict, workdir: str) -> Optional[str]:
    """Resolve a conservative scope containing every operand of one safe shell read."""
    command = action.get("command", "")
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    command = str(command)
    stages = _readonly_shell_stages(command, workdir)
    if stages is None or shell_cd_target(command) is not None:
        return None
    operands: List[str] = []
    for tokens in stages:
        operands.extend(_shell_stage_operands(tokens))
        operands.extend(_option_path_values(tokens))
    command_tokens = _shell_tokens(command) or []
    for index, token in enumerate(command_tokens[:-1]):
        if token == "<":
            operands.append(command_tokens[index + 1])
    resolved = []
    for operand in operands:
        safe = _shell_operand_scope(operand, workdir)
        if safe is not None:
            resolved.append(safe)
    if not resolved:
        return None
    return _norm_path(os.path.commonpath(resolved), "/testbed")


def _safe_call_value(value, *, allow_path: bool = False) -> bool:
    """Reject values that an unquoted OfficeBench command constructor could interpret as shell."""
    if value is None or isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, str):
        if re.search(r"[\x00\r\n;&|<>`$*?\[\]{}()'\"\\]", value):
            return False
        return allow_path or ("/" not in value and "~" not in value and value not in (".", ".."))
    if isinstance(value, list):
        return all(_safe_call_value(item, allow_path=allow_path) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str)
            and _safe_call_value(key)
            and _safe_call_value(item, allow_path=allow_path)
            for key, item in value.items()
        )
    return False


def _path_arg(action: Dict, *keys: str, workdir: str = "/testbed") -> str:
    for key in keys:
        if action.get(key):
            return _norm_path(action[key], workdir)
    return ""


def _canonical_action_paths(action: Dict, workdir: str) -> Dict:
    """Copy an action with every recognized path resolved for stable transition provenance."""
    out = dict(action)
    for key in _PATH_ARGUMENT_KEYS:
        if out.get(key):
            safe = _testbed_path(str(out[key]), workdir)
            if safe is not None:
                out[key] = safe
    return out


def _creates_parent_dirs(action: Dict) -> bool:
    """Whether a successful action creates missing parent directories for its output."""
    return action.get("app") == "calendar" and action.get("action") == "create_event"


def _parent_resources(target: str) -> List[str]:
    parts = _norm_path(target).split("/")[:-1]
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def action_inputs(action: Dict, workdir: str = "/testbed") -> List[str]:
    """Logical resources an action reads as inputs before producing its result."""
    app, act = action.get("app"), action.get("action")
    if app in ("word", "excel") and act == "create_new_file":
        target = _path_arg(action, "file_path", workdir=workdir)
        # The output need not exist. Only its parent is a required input resource.
        return [target.rsplit("/", 1)[0] if "/" in target else "/testbed"] if target else []
    if app in ("word", "excel") and act == "convert_to_pdf":
        key = "word_file_path" if app == "word" else "excel_file_path"
        path = _path_arg(action, key, "file_path", workdir=workdir)
        return [path] if path else []
    if app == "pdf" and act in ("convert_to_image", "convert_to_word"):
        path = _path_arg(action, "pdf_file_path", "file_path", workdir=workdir)
        return [path] if path else []
    target = read_target(action, workdir)
    return [target] if target else []


def write_targets(action: Dict, workdir: str = "/testbed") -> List[str]:
    """Logical resources changed by a write action (its domain-neutral state footprint)."""
    app, act = action.get("app"), action.get("action")
    a = action
    if app == "llm" and act == "complete_text":
        return []
    if app in ("excel", "word"):
        if act == "convert_to_pdf":
            output = _path_arg(a, "pdf_file_path", workdir=workdir)
            source_key = "excel_file_path" if app == "excel" else "word_file_path"
            source = _path_arg(a, source_key, "file_path", workdir=workdir)
            if output and source:
                return [os.path.join(os.path.dirname(output), os.path.splitext(os.path.basename(source))[0] + '.pdf')]
            return [_derive_output(source, "pdf", "/testbed")] if source else ["<unknown>"]
        if a.get("file_path"):
            return [_norm_path(a["file_path"], workdir)]
    if app == "pdf":
        if act == "convert_to_image":
            output = _path_arg(a, "image_file_path", workdir=workdir)
            source = _path_arg(a, "pdf_file_path", "file_path", workdir=workdir)
            return [output] if output else (
                [_derive_output(source, "png", "/testbed")] if source else ["<unknown>"])
        if act == "convert_to_word":
            output = _path_arg(a, "word_file_path", workdir=workdir)
            source = _path_arg(a, "pdf_file_path", "file_path", workdir=workdir)
            return [output] if output else (
                [_derive_output(source, "docx", "/testbed")] if source else ["<unknown>"])
    if app == "calendar" and act in ("create_event", "delete_event"):
        user = a.get("user") or a.get("username") or "user"
        return [f"calendar/{user}.ics"]
    if app == "email" and act == "send_email":
        out = []
        for who in (email_cli_data(a)['recipient'], a.get("sender")):
            if who:
                who = str(who).split("@")[0]
                out.append(f"emails/{who}")
        return out or ["emails/"]
    if app == "shell":
        cmd = a.get("command", "")
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        if re.search(r"[;&|]", str(cmd)):
            return ["<shell>"]
        deleted = _shell_delete_paths(a, workdir)
        if deleted:
            return deleted
        if _deletes_targets(a):
            return ["<shell>"]
        try:
            tokens = shlex.split(str(cmd))
        except ValueError:
            tokens = []
        if tokens:
            name, args = tokens[0], tokens[1:]
            target_dir, consumed = _target_directory_option(args)
            operands = [
                arg for index, arg in enumerate(args)
                if index not in consumed and not arg.startswith("-")
            ]
            if name in ("cp", "mv", "install", "ln", "convert", "magick") and operands:
                if target_dir and name in ("cp", "mv", "install", "ln"):
                    targets = [
                        _norm_path(os.path.join(target_dir, os.path.basename(source)), workdir)
                        for source in operands
                    ]
                    return targets if all(targets) else ["<shell>"]
                destination = operands[-1]
                sources = operands[:-1]
                if (destination.endswith("/") or len(sources) > 1) and sources:
                    targets = [
                        _norm_path(os.path.join(destination, os.path.basename(source)), workdir)
                        for source in sources
                    ]
                    return targets if all(targets) else ["<shell>"]
                target = _norm_path(destination, workdir)
                return [target] if target else ["<shell>"]
            if name in ("touch", "mkdir") and operands:
                targets = [_norm_path(path, workdir) for path in operands]
                return targets if all(targets) else ["<shell>"]
            if name in ("sort", "tree"):
                output = _output_option_value(args)
                if output:
                    target = _norm_path(output, workdir)
                    return [target] if target else ["<shell>"]
            if name == "diff":
                for arg in args:
                    if arg.startswith("--output="):
                        target = _norm_path(arg.split("=", 1)[1], workdir)
                        return [target] if target else ["<shell>"]
        m = re.search(r"(?:>>|>)\s*(\S+)", cmd)
        if m:
            target = _norm_path(m.group(1), workdir)
            return [target] if target else ["<shell>"]
        return ["<shell>"]
    # any other write: fall back to a file_path arg if present
    return [_norm_path(a["file_path"], workdir)] if a.get("file_path") else ["<unknown>"]


def read_target(action: Dict, workdir: str = "/testbed") -> Optional[str]:
    """The canonical file a READ action reads (to detect a read-after-write), or ``None``."""
    app, act = action.get("app"), action.get("action")
    a = action
    if app == "pdf" and act == "read_file":
        path = _path_arg(a, "pdf_file_path", "file_path", workdir=workdir)
        return path or None
    if app in ("excel", "word", "ocr"):
        path = _path_arg(a, "file_path", "image_file_path", workdir=workdir)
        return path or None
    if app == "calendar" and act == "list_events":
        return f"calendar/{a.get('username') or a.get('user') or 'user'}.ics"
    if app == "email" and act in ("list_emails", "read_email"):
        return f"emails/{a.get('username') or a.get('user') or 'user'}"
    if app == "shell":
        return _shell_read_target(a, workdir)
    return None


def _deletes_targets(action: Dict) -> bool:
    if action.get("app") != "shell":
        return False
    command = action.get("command", "")
    if isinstance(command, list):
        command = " ".join(command)
    try:
        first = os.path.basename(shlex.split(str(command))[0])
    except (ValueError, IndexError):
        return False
    return first in ("rm", "unlink", "rmdir")


# --------------------------------------------------------------------------- #
# The predictor
# --------------------------------------------------------------------------- #


def _action_str(action: Dict) -> str:
    import json

    return json.dumps(action, sort_keys=True, default=str)[:300]


def _int_env(name: str, default: int) -> int:
    """A positive-int tuning knob from the environment (falls back to ``default``)."""
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _doc_type(path: str) -> str:
    """Infer the OfficeBench reader to use for a direct ``READ <path>`` grounding command."""
    ext = os.path.splitext(str(path).lower())[1]
    return {
        ".xlsx": "xlsx",
        ".doc": "doc",
        ".docx": "docx",
        ".pdf": "pdf",
        ".ics": "ics",
        ".bmp": "image",
        ".jpeg": "image",
        ".jpg": "image",
        ".png": "image",
        ".tif": "image",
        ".tiff": "image",
    }.get(ext, "txt")


def _testbed_path(path: str, workdir: str = "/testbed") -> Optional[str]:
    """Resolve a path under ``/testbed`` and reject traversal or access to reference/system files."""
    raw = str(path or "").strip()
    if not raw or re.search(r"[\x00\r\n;&|<>`$*?\[\]{}()'\"\\]", raw):
        return None
    if os.path.isabs(raw):
        resolved = os.path.normpath(raw)
    else:
        base = os.path.normpath(workdir or "/testbed")
        if base != "/testbed" and not base.startswith("/testbed/"):
            return None
        resolved = os.path.normpath(os.path.join(base, raw))
    if resolved != "/testbed" and not resolved.startswith("/testbed/"):
        return None
    return resolved


class OfficeBenchWMQuery:
    """Read-only grounding interface exposed to the OfficeBench world-model subagent.

    It exposes four domain-neutral simulation tools:

    * ``READ`` — current logical resource state (predicted overlay first, pristine state second);
    * ``LIST`` — pristine namespace plus resources created in the predicted overlay;
    * ``CALL`` — an optional whitelisted read-only app operation for exact-read mode only;
    * ``TRACE`` — the ordered transition log for one resource.
    """

    def __init__(
        self,
        session,
        workdir: Callable[[], str],
        *,
        allow_action_execution: bool = True,
    ) -> None:
        self._session = session
        self._workdir = workdir
        self.allow_action_execution = bool(allow_action_execution)
        self.reads = 0
        self.lists = 0
        self.calls = 0
        self.traces = 0
        self._logical_read = None
        self._trace = None
        self._overlay_paths = None
        self._deleted_paths = None
        self._recursive_deleted_paths = None
        self._revived_paths = None
        self._is_deleted = None
        self._is_dirty = None
        self._raw_only = False
        catalogue = session.apps_catalogue()
        available = []
        for app, actions in (catalogue.get("actions", {}) or {}).items():
            for action in actions:
                name = f"{app}.{action}"
                probe = {"app": app, "action": action}
                if app not in ("system", "shell") and not is_write_action(probe):
                    available.append(name)
        self.available_calls = (
            list(dict.fromkeys(available)) if self.allow_action_execution else []
        )

    def bind_simulation(self, *, logical_read, trace, overlay_paths, deleted_paths,
                        recursive_deleted_paths, revived_paths, is_deleted, is_dirty,
                        raw_only: bool = False) -> None:
        """Bind domain state operations after the predictor and query interface are constructed."""
        self._logical_read = logical_read
        self._trace = trace
        self._overlay_paths = overlay_paths
        self._deleted_paths = deleted_paths
        self._recursive_deleted_paths = recursive_deleted_paths
        self._revived_paths = revived_paths
        self._is_deleted = is_deleted
        self._is_dirty = is_dirty
        self._raw_only = bool(raw_only)

    def workdir(self) -> str:
        return self._workdir() or "/"

    def _read_pristine(self, safe_path: str) -> str:
        try:
            state = self._session.read_state(
                [{"path": safe_path, "doc_type": _doc_type(safe_path)}])
            if safe_path not in state:
                raise ValueError("grounding returned no file state")
            content = str(state[safe_path])
            if content.startswith("[unreadable:") or content == "[OCR failed]":
                raise ValueError(content)
            return content
        except Exception as exc:
            raise OfficeBenchSimulationError(
                f"Unable to read immutable evidence {safe_path}: {exc}"
            ) from exc

    def read_raw(self, path: str) -> str:
        """Return untouched pristine evidence for generated Python."""
        path = str(path or "").strip()
        if not path:
            return "ERROR: READ requires a file path"
        safe_path = _testbed_path(path, self.workdir())
        if safe_path is None:
            return "ERROR: READ may access only files under /testbed"
        self.reads += 1
        return self._read_pristine(safe_path)

    def read(self, path: str) -> str:
        """Read the current logical state: overlay first, pristine state if untouched."""
        path = str(path or "").strip()
        if not path:
            return "ERROR: READ requires a file path"
        safe_path = _testbed_path(path, self.workdir())
        if safe_path is None:
            return "ERROR: READ may access only files under /testbed"
        self.reads += 1
        pristine = self._read_pristine(safe_path)
        target = _norm_path(safe_path)
        if self._logical_read is not None:
            return str(self._logical_read(target, pristine))
        return pristine

    def list(self, path: str, *, recursive: bool = False) -> str:
        """List the pristine directory plus predicted overlay resources below it."""
        path = str(path or "/testbed").strip() or "/testbed"
        safe_path = _testbed_path(path, self.workdir())
        if safe_path is None:
            return "ERROR: LIST may access only directories under /testbed"
        self.lists += 1
        directory = _norm_path(safe_path)
        recursive_roots = (
            self._recursive_deleted_paths()
            if self._recursive_deleted_paths is not None else []
        )
        covering_deletions = [
            root for root in recursive_roots
            if directory == root or directory.startswith(root.rstrip("/") + "/")
        ]
        if self._is_deleted is not None and self._is_deleted(directory):
            detail = "\n".join(f"  {p}" for p in covering_deletions) or "  (deleted resource)"
            return f"[directory does not exist]\nPREDICTED DELETION:\n{detail}"
        if covering_deletions:
            overlay = [
                p for p in (self._overlay_paths() if self._overlay_paths is not None else [])
                if p == directory or p.startswith(directory.rstrip("/") + "/")
            ]
            revived = [
                p for p in (self._revived_paths() if self._revived_paths is not None else [])
                if p == directory or p.startswith(directory.rstrip("/") + "/")
            ]
            logical = list(dict.fromkeys(overlay + revived))
            entries = "\n".join(f"  {p}" for p in logical) or "  (empty)"
            return (
                "PREDICTED LOGICAL DIRECTORY "
                "(pristine descendants were removed recursively):\n" + entries
            )
        list_state = getattr(self._session, "list_state", None)
        if callable(list_state):
            try:
                result = (
                    list_state(safe_path, recursive=True)
                    if recursive else list_state(safe_path)
                )
                if result.get("missing"):
                    base = "[directory does not exist]"
                elif result.get("error"):
                    base = f"ERROR: {result['error']}"
                elif result.get("_timeout") or not isinstance(result.get("entries"), list):
                    base = "ERROR: immutable listing returned no complete inventory"
                else:
                    base = json.dumps(
                        result["entries"],
                        ensure_ascii=True,
                        sort_keys=True,
                    )
            except Exception as exc:
                return f"ERROR: {type(exc).__name__}: {exc}"
        elif self.allow_action_execution:
            action = {
                "app": "shell",
                "action": "command",
                "command": f"ls -la -- {shlex.quote(safe_path)}",
            }
            try:
                result = self._session.exec_action(action, self.workdir())
                base = str(result.get("observation", result))
            except Exception as exc:
                return f"ERROR: {type(exc).__name__}: {exc}"
        else:
            return "ERROR: immutable object listing is unavailable"
        if self._raw_only:
            return base
        if self._overlay_paths is None:
            return base
        directory = directory.rstrip("/")
        overlay = [
            p for p in self._overlay_paths()
            if p not in ("<shell>", "<unknown>")
            and (not directory or p == directory or p.startswith(directory + "/"))
        ]
        deleted = [
            p for p in (self._deleted_paths() if self._deleted_paths is not None else [])
            if not directory or p == directory or p.startswith(directory + "/")
        ]
        if self._revived_paths is not None:
            revived = set(self._revived_paths())
            deleted = [p for p in deleted if p not in revived]
        if not overlay and not deleted:
            return base
        sections = [base]
        if overlay:
            sections.append("PREDICTED OVERLAY ENTRIES:\n" + "\n".join(f"  {p}" for p in overlay))
        if deleted:
            sections.append(
                "PREDICTED DELETIONS (these override any pristine listing):\n"
                + "\n".join(f"  {p}" for p in deleted)
            )
        return "\n".join(sections)

    def runtime_capabilities(self):
        getter = getattr(self._session, 'runtime_capabilities', None)
        return getter() if callable(getter) else {}

    def call(self, name: str, args: Dict) -> str:
        """Invoke a whitelisted read action, using the overlay instead of stale state when dirty."""
        if not self.allow_action_execution:
            return "ERROR: CALL is disabled when every action must be predicted"
        if name not in self.available_calls or "." not in name:
            return f"ERROR: tool {name!r} is not an available read-only tool"
        app, action_name = name.split(".", 1)
        if not isinstance(args, dict):
            return f"ERROR: CALL {name!r} requires one JSON object of arguments"
        if not all(
            _safe_call_value(key)
            and _safe_call_value(value, allow_path=key in _PATH_ARGUMENT_KEYS)
            for key, value in args.items()
        ):
            return f"ERROR: CALL {name!r} contains unsafe argument characters"
        action = dict(args)
        action["app"] = app
        action["action"] = action_name
        for key in _PATH_ARGUMENT_KEYS:
            if action.get(key):
                safe_path = _testbed_path(str(action[key]), self.workdir())
                if safe_path is None:
                    return f"ERROR: tool {name!r} may access only paths under /testbed"
                action[key] = safe_path
        if is_write_action(action):
            return f"ERROR: tool {name!r} is not read-only for these arguments"
        self.calls += 1
        target = read_target(action, self.workdir())
        if (
            not self._raw_only
            and target
            and self._is_dirty is not None
            and self._is_dirty(target)
        ):
            return self.read(f"/testbed/{target}")
        try:
            result = self._session.exec_action(action, self.workdir())
            return str(result.get("observation", result))
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    def trace(self, path: str) -> str:
        """Return the append-only predicted transition history for one logical resource."""
        safe_path = _testbed_path(str(path or "").strip(), self.workdir())
        if safe_path is None:
            return "ERROR: TRACE may access only resources under /testbed"
        self.traces += 1
        if self._trace is None:
            return "  (no transition log bound)"
        return str(self._trace(_norm_path(safe_path)))


def _parse_agentic(text: str):
    """Parse an OfficeBench WM turn into grounding commands or a terminal prediction/verdict."""
    up = (text or "").upper()
    pi = up.find("PREDICT:")
    if pi != -1:
        return [], [], [], [], ("predict", text[pi + len("PREDICT:"):].strip())
    vi = up.find("VERDICT:")
    if vi != -1:
        answer = up[vi + len("VERDICT:"):].strip()
        verdict = "SUCCESS" if answer.startswith("SUCCESS") else "FAILURE"
        return [], [], [], [], ("verdict", verdict)
    reads, lists, calls, traces = [], [], [], []
    for line in (text or "").splitlines():
        s = line.strip()
        u = s.upper()
        if u.startswith("READ "):
            path = s[5:].strip().strip('"\'')
            if path:
                reads.append(path)
        elif u.startswith("LIST "):
            path = s[5:].strip().strip('"\'')
            if path:
                lists.append(path)
        elif u.startswith("CALL "):
            rest = s[5:].strip()
            parts = rest.split(maxsplit=1)
            name = parts[0] if parts else ""
            args = {}
            if len(parts) == 2:
                try:
                    parsed = json.loads(parts[1])
                    args = parsed if isinstance(parsed, dict) else None
                except Exception:
                    args = None
            if name:
                calls.append((name, args))
        elif u.startswith("TRACE "):
            path = s[6:].strip().strip('"\'')
            if path:
                traces.append(path)
    return reads, lists, calls, traces, None


def run_agentic_read_loop(complete: Callable[[str, int], str], query: OfficeBenchWMQuery,
                           header: str, final_instr: str, *, max_turns: int = 6,
                           max_tokens: int = 1500):
    """Run the bounded OfficeBench ``READ``/``LIST``/``CALL``/``TRACE`` grounding loop."""
    known: Dict[str, str] = {}
    calls = ", ".join(query.available_calls) or "(none)"
    protocol = (
        "You may inspect the CURRENT LOGICAL OfficeBench state before answering. It is the immutable "
        "pristine state plus an automatic log of predicted writes; never assume a pristine missing file "
        "is absent if the overlay created it. Each turn output one or more grounding commands:\n"
        "  READ <path>                    (read logical state: overlay first, pristine second)\n"
        "  LIST <path>                    (list pristine and overlay entries)\n"
        "  CALL <app.action> {json args}  (invoke a read-only action against logical state)\n"
        "  TRACE <path>                   (inspect predicted transitions for a resource)\n"
        f"Available CALL tools: {calls}\n"
        f"OR, when enough evidence is gathered, output:\n  {final_instr}\n"
        f"Use only the minimum evidence needed; you have at most {max_turns} read turns."
    )
    for _turn in range(max_turns):
        facts = "\n".join(f"{key}: {value}" for key, value in known.items()) or "(nothing read yet)"
        response = complete(f"{header}\n\n{protocol}\n\nFacts gathered so far:\n{facts}\n\nYour move:",
                            max_tokens)
        reads, lists, tool_calls, traces, terminal = _parse_agentic(response)
        if terminal is not None:
            return terminal
        acted = False
        for path in lists[:3]:
            known[f"list {path}"] = query.list(path)
            acted = True
        for path in reads[:3]:
            known[f"read {path}"] = query.read(path)
            acted = True
        for name, args in tool_calls[:4]:
            known[f"call {name}({args})"] = query.call(name, args)
            acted = True
        for path in traces[:3]:
            known[f"trace {path}"] = query.trace(path)
            acted = True
        if not acted:
            return ("predict", (response or "").strip())

    facts = "\n".join(f"{key}: {value}" for key, value in known.items()) or "(nothing read)"
    response = complete(
        f"{header}\n\nFacts gathered:\n{facts}\n\nYou are out of read turns. {final_instr}",
        max_tokens,
    )
    _, _, _, _, terminal = _parse_agentic(response)
    return terminal if terminal is not None else ("predict", (response or "").strip())


#: Output budget for a predicted read / reconstructed file (must be able to emit the larger inputs).
_READ_MAX_TOKENS = _int_env("TTS_OFFICEBENCH_WM_READ_TOKENS", 3000)
_STATE_MAX_TOKENS = _int_env("TTS_OFFICEBENCH_WM_STATE_TOKENS", 3000)


def _wm_check_writes() -> bool:
    """Whether the WM predicts write SUCCESS-or-FAILURE (default ON) vs the old assume-success stub.

    When on, a write's observation is predicted from the trajectory so far (which files exist, prior
    errors), so an impossible write (missing input file, bad path) yields the app's realistic FAILURE
    observation and does NOT mutate the predicted state — instead of silently rubber-stamping success.
    Set ``TTS_OFFICEBENCH_WM_CHECK_WRITES=0`` to reproduce the old always-succeeds behaviour.
    """
    return os.environ.get("TTS_OFFICEBENCH_WM_CHECK_WRITES", "1") not in ("0", "false", "False")


def _wm_symbolic() -> bool:
    """Whether structured writes are propagated with EXACT pure functions (default ON) not the LLM.

    Multi-app OfficeBench trajectories fail mostly because every write is LLM-predicted/never executed,
    so a downstream read/verify consumes an LLM-*re-derived* (drifting, truncated, sometimes corrupted)
    version of an upstream file.  When on, a written file whose writes are all *fully determined by
    their action arguments* (``excel.set_cell``; ``word.create_new_file``/``write_to_file`` carrying the
    literal ``contents``) is reconstructed by a deterministic transform instead of the LLM — carrying
    the literal forward verbatim and applying cell edits exactly, which removes that drift without ever
    executing the write.  Any write the transform cannot model exactly falls back to the LLM path.
    Set ``TTS_OFFICEBENCH_WM_SYMBOLIC=0`` to reproduce the always-LLM behaviour.
    """
    return os.environ.get("TTS_OFFICEBENCH_WM_SYMBOLIC", "1") not in ("0", "false", "False")


# --------------------------------------------------------------------------- #
# Deterministic (execution-free) symbolic reconstruction of a written file
# --------------------------------------------------------------------------- #

_CELL_RE = re.compile(r"\((\d+),\s*(\d+)\): ?([^\t\n]*)")


def _parse_excel_grid(content: str) -> Dict[Tuple[int, int], str]:
    """Parse the official ``excel_read_file`` rendering (``(r, c): value`` tokens) into a cell map."""
    grid: Dict[Tuple[int, int], str] = {}
    for m in _CELL_RE.finditer(content or ""):
        grid[(int(m.group(1)), int(m.group(2)))] = m.group(3)
    return grid


def _render_excel_grid(grid: Dict[Tuple[int, int], str]) -> str:
    """Render a cell map back into the official ``(r, c): value`` form (rows by ``\\n``, cells by tab)."""
    if not grid:
        return ""
    return "".join(
        "".join(f"({r}, {c}): {grid.get((r, c), '[Empty Cell]')}\t"
                for c in range(1, max(c for _, c in grid) + 1)) + "\n"
        for r in range(1, max(r for r, _ in grid) + 1)
    )


def _content_arg(action: Dict) -> Optional[str]:
    """The literal file content an authoring word action carries (``contents``/``content``/``text``)."""
    for key in ("contents", "content", "text"):
        if key in action and action[key] is not None:
            return str(action[key])
    return None


def _apply_excel_writes(initial: str, writes: List[Tuple[Dict, str]]) -> Optional[str]:
    """Exactly apply create/set/delete operations to a workbook, including a newly created file.

    A ``create_new_file`` transition initializes an empty grid, so subsequent cell writes no longer
    require an already parseable pristine workbook. Unsupported or underspecified operations return
    ``None`` and use the LLM fallback.
    """
    grid = _parse_excel_grid(initial)
    exists = bool((initial or "").strip()) and (initial or "").strip() != "[file does not exist]"
    modelable = bool(grid)
    changed = False
    for action, _obs in writes:
        act = action.get("action")
        if act == "create_new_file":
            if not exists:
                grid = {}
                exists = True
                modelable = True
                changed = True
            continue
        if not modelable:
            return None
        try:
            row = int(action["row_idx"])
            col = int(action["column_idx"])
        except (KeyError, TypeError, ValueError):
            return None
        if act == "delete_cell":
            grid[(row, col)] = "[Empty Cell]"
            changed = True
        elif act == "set_cell":
            val = action.get("text", action.get("value", ""))
            try:
                if cli_literal(val) is None:
                    # openpyxl.cell(value=None) leaves an existing value unchanged.
                    grid.setdefault((row, col), "[Empty Cell]")
                    changed = True
                    continue
            except ValueError:
                return None
            rendered = excel_cell_text(val)
            if rendered is None or row < 1 or col < 1:
                return None
            grid[(row, col)] = rendered
            changed = True
        else:
            return None
    return _render_excel_grid(grid) if changed else initial


def _apply_word_writes(initial: str, writes: List[Tuple[Dict, str]]) -> Optional[str]:
    """Apply native quote stripping and paragraph append/create semantics to copied text."""
    missing = initial.strip() == "[file does not exist]"
    parts: List[str] = [] if missing or not initial else initial.split("\n")
    for action, _obs in writes:
        content = _content_arg(action)
        if action.get("action") == "create_new_file":
            parts = []
            missing = False
        else:  # write_to_file
            if missing:
                return None
            if content is None:
                return None
            # Native construct_action strips quotes, then Fire parses the argument.
            value = cli_literal(content.replace("'", "").replace('"', ''))
            if not isinstance(value, str):
                return None
            parts.append(value)
    return "\n".join(parts)


def symbolic_file_state(target: str, initial: str, writes: List[Tuple[Dict, str]]) -> Optional[str]:
    """Deterministic post-write content of ``target`` when every write is exactly modelable, else ``None``.

    Pure function (no LLM, no execution): dispatches to the per-app transform only when *all* writes to
    the file are of one exactly-reconstructable kind; any other mix returns ``None`` so the caller uses
    its LLM fallback.  This is the execution-free analog of replaying the writes onto the file.
    """
    if not _wm_symbolic() or not writes:
        return None
    if all(a.get("app") == "excel"
           and a.get("action") in ("create_new_file", "set_cell", "delete_cell")
           for a, _ in writes):
        return _apply_excel_writes(initial, writes)
    if all(a.get("app") == "word" and a.get("action") in ("create_new_file", "write_to_file")
           for a, _ in writes):
        return _apply_word_writes(initial, writes)
    return None


#: Substrings the official OfficeBench apps print when an action FAILS (so a predicted write observation
#: carrying one of these is a failed write: it must not change the predicted file state).
_FAILURE_MARKERS = (
    "does not exist", "failed to", "no such file", "not found", "cannot ", "could not",
    "is not a", "invalid", "error:", "traceback", "no module named", "permission denied",
    "malformed", "unable to", "already exists",
)


def _looks_like_failure(observation: str) -> bool:
    """True if a predicted observation reads like an app error (a write that did NOT take effect)."""
    o = (observation or "").lower()
    if not o:
        return False
    if "success" in o:  # apps prefix successful writes with 'Successfully ...' — never a failure
        return False
    return any(m in o for m in _FAILURE_MARKERS)




class OfficeBenchWMPredictor:
    """Agentically predicts OfficeBench observations and final file state without executing writes."""

    MAX_TURNS = 6

    def __init__(self, chat_client) -> None:
        self._chat = chat_client
        self._query: Optional[OfficeBenchWMQuery] = None
        self._flat_system_prompt: Optional[str] = None
        self._flat_initial_state: Dict[str, str] = {}
        self.transition_log = TransitionLog()
        #: target (normalised path) -> list of (action, predicted_observation) applied to it, in order.
        self.applied_writes: Dict[str, List[Tuple[Dict, str]]] = {}
        #: exact pre-write content read once per target (for dirty-read grounding).
        self.pre_write_content: Dict[str, str] = {}
        self.write_history: List[Tuple[Dict, str]] = []
        self.conversion_sources: Dict[str, Dict] = {}
        self.file_snapshots: Dict[str, Dict] = {}
        self.logical_directories: set[str] = set()
        self.deleted_paths: set[str] = set()
        self.recursive_deleted_paths: set[str] = set()
        self.revived_paths: set[str] = set()
        #: set once a write with an UNRESOLVED target (e.g. an unparseable shell write) occurs — from
        #: then on EVERY read is treated as dirty (predicted), because we cannot prove it is untouched.
        #: Mirrors τ²'s ``global_writes`` fallback for ref-less (device/global) writes.
        self.global_dirty = False
        self.n_writes = 0
        self.n_failed_writes = 0
        self.n_reads = 0
        self.n_predicted_reads = 0
        self.n_dirty_reads = 0
        self.n_llm_calls = 0
        self.n_python_calls = 0
        self.n_python_failures = 0
        self.n_python_forced = 0
        self.n_python_mode_misses = 0
        self.program_only = False

    def bind_query_interface(self, query: OfficeBenchWMQuery) -> None:
        """Give the WM subagent read-only access to the logical simulated environment."""
        self._query = query
        bind = getattr(query, "bind_simulation", None)
        if callable(bind):
            bind(
                logical_read=self.logical_read,
                trace=self.trace_text,
                overlay_paths=self.overlay_paths,
                deleted_paths=lambda: sorted(self.deleted_paths),
                recursive_deleted_paths=lambda: sorted(self.recursive_deleted_paths),
                revived_paths=lambda: sorted(self.revived_paths),
                is_deleted=self.is_deleted,
                is_dirty=self.was_written,
                raw_only=self.program_only,
            )

    @property
    def uses_fixed_context(self) -> bool:
        return False

    @property
    def flat_system_prompt(self) -> Optional[str]:
        return self._flat_system_prompt


    def fixed_pre_write_content(self, target: Optional[str]) -> Optional[str]:
        """Return a baseline selected before actions, without reading the live shadow."""
        return None
        return self._flat_initial_state.get(
            _norm_path(target),
            "[not included in the fixed pre-action context]",
        )

    @property
    def query_interface(self) -> Optional[OfficeBenchWMQuery]:
        return self._query

    def _record_python_result(self, result) -> None:
        self.n_python_calls += result.python_calls
        self.n_python_failures += result.python_failures
        self.n_python_forced += result.python_forced
        self.n_python_mode_misses += result.python_mode_misses

    def _current_workdir(self) -> str:
        workdir = getattr(self._query, "workdir", None)
        return workdir() if callable(workdir) else "/testbed"

    # -- bookkeeping ---------------------------------------------------- #
    def was_written(self, target: Optional[str]) -> bool:
        # A specific written target OR the global-dirty fallback (an unresolved write happened) makes a
        # read dirty -> predicted, never a stale exact read.  A ``None`` target (e.g. llm.complete_text)
        # has no file, so it is never dirty.
        if target is None:
            return False
        target = _norm_path(target)
        prefix = "" if not target else target.rstrip("/") + "/"
        return (
            self.global_dirty
            or target in self.applied_writes
            or self._is_deleted(target)
            or any(path.startswith(prefix) for path in self.overlay_paths())
            or any(
                target.startswith(path.rstrip("/") + "/")
                for path in self.overlay_paths() if path)
            or any(path.startswith(prefix) for path in self.deleted_paths)
        )

    def _is_deleted(self, target: str) -> bool:
        target = _norm_path(target)
        if target in self.revived_paths:
            return False
        if target in self.deleted_paths:
            return True
        return any(
            target.startswith(root.rstrip("/") + "/")
            for root in self.recursive_deleted_paths
        )

    def is_deleted(self, target: str) -> bool:
        """Whether a concrete resource is hidden by an exact or recursive tombstone."""
        return self._is_deleted(target)

    def note_read(self, target: Optional[str], content: str) -> None:
        """Cache the EXACT pre-write content of a target the first time it's read (dirty-read anchor)."""
        self.n_reads += 1
        if target and target not in self.pre_write_content:
            self.pre_write_content[target] = content

    def _writes_text(self, target: str) -> str:
        target = _norm_path(target)
        prefix = "" if not target else target.rstrip("/") + "/"
        related = lambda resource: (  # noqa: E731 - compact relation used twice below
            resource == target
            or resource.startswith(prefix)
            or (bool(resource) and target.startswith(resource.rstrip("/") + "/"))
        )
        entries = [
            entry for entry in self.transition_log.entries(applied_only=True)
            if any(related(resource) for resource in entry.resources)
        ]
        if entries:
            return "\n".join(
                f"  - {json.dumps(entry.action, default=str)} -> [{entry.observation}]"
                for entry in entries
            )
        # Compatibility for tests/callers that seed ``applied_writes`` directly.
        items = [
            item for resource, resource_items in self.applied_writes.items()
            if related(resource)
            for item in resource_items
        ]
        return "\n".join(
            f"  - {json.dumps(a, default=str)} -> [{o}]" for a, o in items
        ) or "  (none)"

    def _related_writes(self, target: str) -> List[Tuple[Dict, str]]:
        target = _norm_path(target)
        prefix = "" if not target else target.rstrip("/") + "/"
        entries = self.transition_log.entries(applied_only=True)
        if entries:
            # One transition can affect several resources. Preserve global order and include
            # it once, rather than regrouping/repeating it through the per-resource index.
            return [(entry.action, entry.observation) for entry in entries if any(
                resource == target or resource.startswith(prefix)
                or (bool(resource) and target.startswith(resource.rstrip('/') + '/'))
                for resource in entry.resources)]
        return [
            item for resource, items in self.applied_writes.items()
            if (
                resource == target
                or resource.startswith(prefix)
                or (bool(resource) and target.startswith(resource.rstrip("/") + "/"))
            )
            for item in items
        ]

    def file_write_log(self, path: str) -> str:
        """Human-readable log of the WRITE actions applied to ``path`` (for reference-grounded verify)."""
        target = _norm_path(path)
        entries = self.transition_log.entries(target)
        return self._complete_trace(entries) if entries else self._writes_text(target)

    @staticmethod
    def _complete_trace(entries) -> str:
        """State reconstruction needs complete write literals, not a display-sized preview."""
        return "\n".join(
            f"  - {'APPLIED' if entry.applied else 'REJECTED'} "
            f"{json.dumps(entry.action, default=str)} -> [{entry.observation}]"
            for entry in entries
        )

    def overlay_paths(self) -> List[str]:
        """Resources created or changed by applied predicted transitions."""
        resources = list(self.transition_log.resources(applied_only=True))
        for target in self.applied_writes:
            if target not in resources and target not in self.deleted_paths:
                resources.append(target)
        return [resource for resource in resources if not self._is_deleted(resource)]

    def trace_text(self, path: str) -> str:
        """Render the automatic transition log for the paper-level ``TRACE`` tool."""
        target = _norm_path(path)
        entries = self.transition_log.entries(target)
        if entries:
            return self._complete_trace(entries)
        return self._writes_text(target)

    def _exact_overlay_content(self, target: str, pristine_content: str) -> Optional[str]:
        """Materialize transitions whose resulting content is deterministic."""
        target = _norm_path(target)
        if self._is_deleted(target):
            return "[file does not exist]"
        writes = self.applied_writes.get(target)
        if not writes:
            return None
        for action, _ in reversed(writes):
            if action.get("action") == "finish_task":
                return str(action.get("answer", "None")) + "\n"
        if self.global_dirty:
            return None
        if target in self.file_snapshots:
            snapshot = self.file_snapshots[target]
            tail = writes[snapshot['offset']:]
            return (symbolic_file_state(target, snapshot['content'], tail)
                    if tail else snapshot['content'])
        return symbolic_file_state(target, pristine_content, writes)

    def logical_read(self, target: str, pristine_content: str) -> str:
        """Return current logical state without recursively invoking another LLM prediction.

        Exactly modelable transitions are materialized symbolically. Otherwise the tool returns the
        pristine content plus transition provenance, which tells the subagent that the resource exists
        in the overlay and what must be applied instead of serving stale pristine state.
        """
        target = _norm_path(target)
        writes = self._related_writes(target)
        if self._is_deleted(target):
            return f"[file does not exist]\nPREDICTED TRANSITIONS:\n{self.trace_text(target)}"
        if not writes:
            if self.global_dirty:
                return (
                    f"PRISTINE CONTENT (current state may differ because an opaque write occurred):\n"
                    f"{pristine_content}\n\nGLOBAL PREDICTED TRANSITIONS:\n{self.predicted_write_log()}"
                )
            return pristine_content
        if self.global_dirty and not any(
                action.get("action") == "finish_task" for action, _ in writes):
            return (
                f"PRISTINE CONTENT (current state may differ because an opaque write occurred):\n"
                f"{pristine_content}\n\nTARGET PREDICTED TRANSITIONS:\n{self.trace_text(target)}\n\n"
                f"GLOBAL PREDICTED TRANSITIONS:\n{self.predicted_write_log()}"
            )
        exact = self._exact_overlay_content(target, pristine_content)
        if exact is not None:
            return exact
        sources = list(dict.fromkeys(
            source for action, _ in writes for source in action_inputs(action) if source
        ))
        source_text = ", ".join(sources) or "(none recorded)"
        return (
            f"LOGICAL OVERLAY RESOURCE: {target}\n"
            f"PRISTINE CONTENT BEFORE PREDICTED WRITES:\n{pristine_content}\n\n"
            f"SOURCE RESOURCES: {source_text}\n"
            f"PREDICTED TRANSITIONS:\n{self.trace_text(target)}"
        )

    def _history_text(self, k: int = 8) -> str:
        tail = self.write_history[-k:]
        return "\n".join(f"  - {_action_str(a)} -> [{o[:100]}]" for a, o in tail) or "  (none)"

    def _agent_context_text(self, history: List[Tuple], k: int = 10) -> str:
        """Compact tail of the realized (action -> observation) log, so the write predictor can judge
        whether the write would actually succeed (which files exist, whether earlier steps errored)."""
        if not history:
            return "  (no prior steps)"
        lines = []
        for a, o in list(history)[-k:]:
            a_txt = _action_str(a) if isinstance(a, dict) else str(a)[:300]
            lines.append(f"  - {a_txt} -> [{str(o)[:200]}]")
        return "\n".join(lines)

    @staticmethod
    def _agent_context_data(history: List[Tuple], k: int = 20) -> List[Dict]:
        return [
            {
                "action": a if isinstance(a, dict) else {"raw": str(a)[:1000]},
                "observation": str(o),
            }
            for a, o in list(history)[-k:]
        ]

    def _write_history_data(self) -> List[Dict]:
        entries = self.transition_log.entries()
        if entries:
            return [{'action': entry.action, 'observation': entry.observation,
                     'applied': entry.applied, 'resources': list(entry.resources)}
                    for entry in entries]
        return [
            {"action": action, "observation": observation,
             "applied": not _looks_like_failure(observation)}
            for action, observation in self.write_history
        ]

    def _current_state_data(self, target: str, pristine: Optional[str] = None) -> Dict:
        """Copied logical evidence for Python, never rendered into the model prompt.

        Unsupported transitions remain explicitly unknown. This does not execute candidate
        actions or use native post-write state.
        """
        target = _norm_path(target)
        if pristine is None:
            if self._query is not None:
                pristine = self._query.read_raw(f'/testbed/{target}')
        if pristine is None:
            return {'status': 'unknown', 'exists': None, 'content': None}
        schema = read_schema(target, pristine)
        writes = self._related_writes(target)
        exact = self._exact_overlay_content(target, pristine)
        if exact is None and not writes and not self.global_dirty and schema['status'] != 'tool_error':
            if not str(pristine).startswith('[not included in the fixed pre-action context]'):
                exact = pristine
        if exact is not None:
            missing = exact.strip() == '[file does not exist]'
            return {'status': 'exact', 'state_version': 'predicted_current',
                    'exists': not missing, 'content': None if missing else exact}
        return {'status': 'unknown', 'state_version': 'predicted_current', 'exists': None,
                'content': None, 'pristine_content': pristine,
                'conversion_source': self.conversion_sources.get(target),
                'transitions': [{'action': a, 'observation': o} for a, o in writes]}

    def _mailbox_messages(self, target: str):
        """Pure MIME parsing and send transitions over copied pristine messages.

        No native action or filesystem mutation. Unsupported transitions stay unknown.
        """
        if self.global_dirty or self._query is None or self.uses_fixed_context:
            return None
        writes = self._related_writes(target)
        if any(a.get('app') != 'email' or a.get('action') != 'send_email' for a, _ in writes):
            return None
        from email import policy
        from email.parser import Parser
        from email.message import EmailMessage
        try:
            listing = self._query.list('/testbed/' + target)
            entries = [] if listing.startswith('[directory does not exist]') else json.loads(listing)
            if not isinstance(entries, list):
                return None
            messages = {}
            for item in sorted(entries, key=lambda e: e.get('native_order', 0)):
                name = item['name']
                if item.get('type') == 'file' and name.endswith('.eml') and not name.startswith('.'):
                    raw = self._query.read_raw(item['path'])
                    messages[name] = Parser(policy=policy.default).parsestr(raw)
            for action, _ in writes:
                subject = cli_literal(str(action['subject']).replace("'", '').replace('"', ''))
                content = cli_literal(str(action['content']).replace("'", '').replace('"', ''))
                if not isinstance(subject, str) or not isinstance(content, str) or '/' in subject:
                    return None
                msg = EmailMessage()
                msg['From'] = str(action['sender']).split('@')[0] + '@example.com'
                recipient = email_cli_data(action)['recipient']
                if not isinstance(recipient, str):
                    return None
                msg['To'] = recipient.split('@')[0] + '@example.com'
                msg['Subject'] = subject
                msg.set_content(content)
                messages[subject + '.eml'] = msg
            return messages
        except (ValueError, TypeError, KeyError, AttributeError):
            return None

    @staticmethod
    def _email_body(msg):
        parts = list(msg.iter_parts()) if msg.is_multipart() else [msg]
        return '\n'.join(part.get_payload(decode=True).decode(part.get_content_charset(), errors='replace')
                         for part in parts if not msg.is_multipart()
                         or part.get_content_type() in ('text/plain', 'text/html'))

    def _mailbox_state(self, target: str, limit: int = -1) -> Optional[str]:
        messages = self._mailbox_messages(target)
        if messages is None:
            return None
        blocks = []
        for name, msg in messages.items():
            text = self._email_body(msg)
            if limit != -1:
                text = text[:limit] + '...'
            blocks.append(f"Email ID: {name}\nFrom: {msg['From']}\nTo: {msg['To']}\nSubject: {msg['Subject']}\nContent: {text}\n" + '-' * 50 + '\n')
        return ''.join(blocks) if blocks else 'No emails found.'

    def _predict_shell(self, action):
        """Common single-command filesystem transitions over copied inputs only."""
        if not self.program_only or self.global_dirty or self._query is None or action.get('app') != 'shell':
            return None, {}, set()
        command = action.get('command', '')
        if not isinstance(command, str) or re.search(r"[;&|<>$`\\*?]", command):
            return None, {}, set()
        try:
            argv = shlex.split(command)
        except ValueError:
            return None, {}, set()
        if not argv:
            return None, {}, set()
        snapshots, directories = {}, set()
        if argv[0] in ('mv', 'cp') and len(argv) == 3 and not argv[1].startswith('-'):
            source, dest = (_testbed_path(p, self._current_workdir()) for p in argv[1:])
            if not source or not dest or source == dest:
                return None, {}, set()
            # Bound this adapter to file-to-file operations and existing parent directories.
            listing = self._query.list(dest.rsplit('/', 1)[0])
            try:
                entries = json.loads(listing)
            except (ValueError, TypeError):
                return None, {}, set()
            if not isinstance(entries, list) or any(e.get('path') == dest and e.get('type') == 'directory' for e in entries):
                return None, {}, set()
            state = self._current_state_data(_norm_path(source))
            if state.get('status') != 'exact' or state.get('exists') is not True:
                return None, {}, set()
            snapshots[_norm_path(dest)] = state['content']
        elif len(argv) == 3 and argv[:2] == ['mkdir', '-p']:
            target = _testbed_path(argv[2], self._current_workdir())
            if target is None:
                return None, {}, set()
            # Refuse to certify success if any existing prefix is a regular file.
            parts = target.removeprefix('/testbed/').split('/') if target != '/testbed' else []
            parent = '/testbed'
            for part in parts:
                raw = self._query.list(parent)
                try:
                    entries = json.loads(raw)
                except (ValueError, TypeError):
                    if not str(raw).startswith('[directory does not exist]') or (
                            _norm_path(parent) not in directories | self.logical_directories):
                        return None, {}, set()
                    entries = None
                if entries is not None and not isinstance(entries, list):
                    return None, {}, set()
                if entries is not None and any(e.get('name') == part and e.get('type') == 'file' for e in entries):
                    return None, {}, set()
                parent += '/' + part
                directories.add(_norm_path(parent))
        else:
            return None, {}, set()
        return f'Successfully executed command: {command}. The output was [].', snapshots, directories

    def _predict_creation(self, action: Dict) -> Optional[str]:
        """Pure creation transition from immutable inventory and the predicted overlay."""
        if (not self.program_only or self.global_dirty or self._query is None
                or action.get("app") not in ("word", "excel")
                or action.get("action") != "create_new_file"):
            return None
        raw = str(action.get("file_path", ""))
        # Native constructors do not quote paths. Leave complex shell arguments to the WM.
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", raw):
            return None
        safe = _testbed_path(raw, self._current_workdir())
        if safe is None:
            return None
        target = _norm_path(safe)
        parent = safe.rsplit("/", 1)[0]
        parent_target = _norm_path(parent)
        failed = f"Failed to create new file {raw}"
        if self._is_deleted(parent_target):
            return failed
        listing = self._query.list(parent)
        if parent_target in self.logical_directories and str(listing).startswith("[directory does not exist]"):
            listing = "[]"
        if str(listing).startswith("[directory does not exist]"):
            return None if self.was_written(parent_target) else failed
        try:
            entries = json.loads(listing)
        except (TypeError, ValueError):
            return None
        if not isinstance(entries, list):
            return None
        if any(e.get("path") == parent and e.get("type") == "file"
               for e in entries if isinstance(e, dict)):
            return failed
        matching = [e for e in entries if isinstance(e, dict)
                    and _norm_path(str(e.get("path", ""))) == target]
        exists = not self._is_deleted(target) and (bool(matching) or target in self.applied_writes)
        if action["app"] == "excel" and exists:
            return f"File {raw} already exists"
        if exists and any(e.get("type") == "directory" for e in matching):
            return failed
        return f"Successfully create new file {raw}"

    # -- prediction ----------------------------------------------------- #
    def predict_write(self, action: Dict, history: List[Tuple] = ()) -> str:
        """Predict the observation string a WRITE action returns, and record it — never executes it.

        Unless disabled (``TTS_OFFICEBENCH_WM_CHECK_WRITES=0``), the write is NOT presumed to succeed:
        the predictor judges success/failure from the trajectory so far and, on a predicted FAILURE,
        records nothing (the predicted file state is left unchanged) so a broken write can no longer be
        silently rubber-stamped into the write-log/state.
        """
        self.n_writes += 1
        app, act = action.get("app"), action.get("action")
        workdir = self._current_workdir()
        delete_spec = _shell_delete_spec(action, workdir)
        move_sources = _shell_move_sources(action, workdir)
        recorded_action = _canonical_action_paths(action, workdir)
        inputs = action_inputs(action, workdir)
        targets = write_targets(action, workdir)
        footprint = (
            f"Action state footprint: inputs={inputs or '(none)'}, "
            f"outputs={targets or '(unknown)'}. "
            "READ/LIST are pristine evidence. Apply prior predicted writes from TRACE or "
            "data['writes'] to derive CURRENT logical state before checking preconditions."
        )
        if app in ("word", "excel") and act == "create_new_file":
            footprint += (
                " This action CREATES its output; a missing output is expected, not failure. "
                "The inputs above are parent DIRECTORIES: inspect them with LIST, not READ. "
                "A valid LIST array (even []) establishes that the queried directory exists; its entries "
                "contain children, not the directory itself. Word overwrites an existing document; "
                "Excel reports an existing workbook and leaves it unchanged."
            )
        if _wm_check_writes():
            prompt = (
                f"You are a faithful world model of the OfficeBench '{app}' app. Predict the observation "
                f"the real app prints for the WRITE action '{app}.{act}' with arguments "
                f"{_action_str(action)}.\n{footprint}\n\n"
                "The trajectory and prior write observations are available only through the copied "
                "`data['history']` and `data['writes']` variables. Generated Python must inspect or "
                "transform their contents.\n\n"
                f"Decide whether this write would actually SUCCEED given that context and the app's "
                f"semantics. If a required input/target file is missing, the path is wrong, an argument "
                f"is invalid, or a prior step already failed, return the app's realistic FAILURE "
                f"observation (e.g. 'OBSERVATION: The file /testbed/data/x.xlsx does not exist. Failed to "
                f"read the file.'). Otherwise return the SUCCESS observation matching the app's wording "
                f"(e.g. 'OBSERVATION: Successfully create a new event to Bob's calendar.'). Return ONLY "
                f"that single-line observation string. No prose, no explanation."
            )
        else:
            prompt = (
                f"You are a world model of the OfficeBench '{app}' app. The agent called the WRITE action "
                f"'{app}.{act}' with arguments {_action_str(action)}. {footprint} "
                f"Assume it succeeds on a normal "
                f"filesystem. Return ONLY the single-line observation string the real app prints, matching "
                f"its wording (office apps print e.g. 'OBSERVATION: Successfully create a new event to "
                f"Bob's calendar.'). No prose, no explanation."
            )
        current_state = {}
        obs, shell_snapshots, shell_directories = self._predict_shell(action)
        if obs is None:
            obs = self._predict_creation(action)
        if obs is None:
            if self.program_only and app in ('excel', 'word') and act in ('set_cell', 'delete_cell', 'write_to_file', 'convert_to_pdf'):
                current_state = {target: self._current_state_data(target)
                                 for target in dict.fromkeys(targets + inputs)}
            obs = self._predict(
                prompt + '\n\n' + action_contract(action),
                "PREDICT: <the exact single-line observation returned by the write action>",
                max_tokens=_int_env("TTS_OFFICEBENCH_WM_WRITE_TOKENS", 2048),
                python_context={
                    "kind": "write_observation",
                    "action": recorded_action,
                    "observation_action": dict(action),
                    "current_state": current_state,
                    "email_cli": email_cli_data(action) if app == 'email' and act == 'send_email' else None,
                    "enforce_observation_contract": self.program_only,
                    "inputs": inputs,
                    "targets": targets,
                    "history": self._agent_context_data(history),
                    "writes": self._write_history_data(),
                },
                python_force_always=self.program_only,
            )
        if obs is None:
            raise OfficeBenchSimulationError(
                f"Unable to simulate {app}.{act}; application outcome is unknown"
            )
        # A predicted FAILURE leaves the environment unchanged: record it in the write-history (so the
        # verifier can see the attempted-and-failed write) but do NOT mark any target as written, and do
        # NOT trip global-dirty — a write that did not take effect cannot make a later read dirty.
        resources = list(dict.fromkeys(targets + move_sources))
        if _wm_check_writes() and _looks_like_failure(obs):
            self.n_failed_writes += 1
            self.write_history.append((recorded_action, obs))
            self.transition_log.record(recorded_action, obs, resources, applied=False)
            return obs
        for tgt in targets:
            self.conversion_sources.pop(tgt, None)
            self.applied_writes.setdefault(tgt, []).append((recorded_action, obs))
            if tgt in ("<shell>", "<unknown>"):
                # an unresolved write target: we can no longer certify any file is untouched, so from
                # here every read is predicted (τ²'s global-dirty fallback for ref-less writes).
                self.global_dirty = True
        self.file_snapshots.update({target: {'content': content, 'offset': len(self.applied_writes[target])}
                                    for target, content in shell_snapshots.items()})
        self.logical_directories.update(shell_directories)
        if app in ('word', 'excel') and act == 'convert_to_pdf' and inputs and current_state:
            # Freeze the source AT conversion, not when final state is later queried.
            # This is evidence for prediction, not a claim of exact PDF layout or success.
            import copy
            for target in targets:
                self.conversion_sources[target] = {
                    'action': dict(recorded_action), 'path': inputs[0],
                    'state': copy.deepcopy(current_state.get(inputs[0], {})),
                }
        if self.global_dirty:
            self.conversion_sources.clear()
            self.file_snapshots.clear()
            self.logical_directories.clear()
            # An opaque transition may recreate or remove any resource, so old tombstones are no
            # longer definitive; subsequent reads use the qualified global-dirty representation.
            self.deleted_paths.clear()
            self.recursive_deleted_paths.clear()
            self.revived_paths.clear()
        elif delete_spec is not None:
            deleted, recursive = delete_spec
            self.deleted_paths.update(deleted)
            if recursive:
                self.recursive_deleted_paths.update(deleted)
            for path in deleted:
                self.revived_paths = {
                    revived for revived in self.revived_paths
                    if revived != path and not revived.startswith(path.rstrip("/") + "/")
                }
        elif move_sources:
            self.deleted_paths.update(move_sources)
            self.recursive_deleted_paths.update(move_sources)
            for path in move_sources:
                self.revived_paths = {
                    revived for revived in self.revived_paths
                    if revived != path and not revived.startswith(path.rstrip("/") + "/")
                }
        else:
            for target in targets:
                self.deleted_paths.discard(target)
                self.recursive_deleted_paths.discard(target)
                if target not in ("<shell>", "<unknown>"):
                    self.revived_paths.add(target)
                    if _creates_parent_dirs(recorded_action):
                        self.revived_paths.update(_parent_resources(target))
        self.write_history.append((recorded_action, obs))
        self.transition_log.record(recorded_action, obs, resources, applied=True)
        return obs

    def record_answer(self, answer: str) -> None:
        """Record ``finish_task`` as a predicted answer-file transition without touching /testbed."""
        normalized = normalize_answer(answer)
        action = {"app": "system", "action": "finish_task", "answer": normalized}
        observation = "Task finished"
        target = "data/answer.txt"
        self.applied_writes.setdefault(target, []).append((action, observation))
        self.deleted_paths.discard(target)
        self.recursive_deleted_paths.discard(target)
        self.revived_paths.add(target)
        self.write_history.append((action, observation))
        self.transition_log.record(action, observation, [target], applied=True)

    def predict_read(
        self,
        action: Dict,
        pre_write_content: Optional[str],
        history: List[Tuple] = (),
        *,
        dirty: bool = True,
    ) -> str:
        """Predict a clean or dirty read observation without executing the agent action."""
        self.n_predicted_reads += 1
        target = read_target(action, self._current_workdir()) or ""
        if dirty:
            self.n_dirty_reads += 1
        app, act = action.get("app"), action.get("action")
        if self.program_only and app == 'llm' and act == 'complete_text':
            getter = getattr(self._query, 'runtime_capabilities', None)
            caps = getter() if callable(getter) else {}
            prompt_text = action.get('prompt')
            if (caps.get('llm_key_file_present') is False and isinstance(prompt_text, str)
                    and "'" not in prompt_text and '\x00' not in prompt_text):
                return "Error: [Errno 2] No such file or directory: '/openai_key.txt'"
        if self.program_only and app == 'email' and act == 'read_email':
            messages = self._mailbox_messages(target)
            if messages is not None:
                username = action.get('username')
                email_id = str(action.get('email_id', ''))
                name = email_id if email_id.endswith('.eml') else email_id + '.eml'
                msg = messages.get(name)
                if msg is None:
                    return f'Failed to read email {email_id} for {username}.'
                text = self._email_body(msg)
                return (f"Successfully read email {email_id} for {username}:\n"
                        f"From: {msg['From']}\nTo: {msg['To']}\nSubject: {msg['Subject']}\n"
                        f"Content: {text}...\n").strip()
        if self.program_only and app == 'email' and act == 'list_emails':
            mailbox = self._mailbox_state(target, 20)
            if mailbox is not None:
                username = action.get('username')
                return (f'No emails found for {username}.' if mailbox == 'No emails found.'
                        else f'Successfully list emails for {username}:\n{mailbox}'.strip())
        if dirty and app in ("word", "excel") and act == "read_file":
            # Symbolic transitions are pure computations, also valid under predict-all.
            # Use the actual pristine baseline rather than treating an uncached file as empty.
            if pre_write_content is None and self._query is not None:
                pre_write_content = self._query.read_raw(f"/testbed/{target}")
            if pre_write_content is not None:
                exact = self._exact_overlay_content(target, pre_write_content)
                if exact is not None:
                    if exact == "[file does not exist]":
                        return f"OBSERVATION: The file {action.get('file_path', target)} does not exist. Failed to read the file."
                    prefix = ("OBSERVATION: The following is the table from the excel file:\n"
                              if app == "excel" else "OBSERVATION: The following is the content from the word file:" + ("\n" if exact else ""))
                    return prefix + exact
        # target-specific writes, or — when the read is dirty only via the global-dirty fallback — the
        # full write history (so a globally-dirty read still sees what may have changed the file).

        baseline = (
            "Pristine evidence is available only through `data['pre_write_content']`."
            if pre_write_content is not None
            else
            "Use bounded read-only grounding tools; their untouched outputs are available only through "
            "`data['reads'][index]['result']` variables."
        )
        state_instruction = (
            "Earlier predicted writes may have changed the target. Apply those transitions to pristine "
            "evidence and return the logical post-write result."
            if dirty else
            (
                "This clean read is not executed directly for the agent. The host has copied the exact "
                "read-only action observation into the grounding evidence below; generate the "
                "agent-visible response by reproducing it exactly."
                if pre_write_content is not None else
                "This clean read is not executed directly for the agent. Predict its exact observation "
                "from the available fixed context or read-only grounding evidence."
            )
        )
        tool_name = f"{app}.{act}"
        grounded_clean_read = bool(not dirty and pre_write_content is not None)
        can_call_exact = bool(
            not dirty
            and not grounded_clean_read
            and not self.uses_fixed_context
            and self._query is not None
            and tool_name in set(getattr(self._query, "available_calls", []) or [])
        )
        exact_read_contract = ""
        if grounded_clean_read:
            exact_read_contract = (
                "\nHIGHEST-PRIORITY CLEAN-READ RULE:\n"
                "- The PRISTINE / FIXED evidence below is the exact native observation.\n"
                "- Generate the final response by copying it VERBATIM, preserving exact wording, "
                "formatting, whitespace, ordering, empty output, and failure text.\n"
                "- Do not make another read, infer a different value, summarize, or repair it.\n"
            )
        elif can_call_exact:
            if self.program_only or officebench_wm_tool_mode() == "python":
                first_read = (
                    "On the FIRST tool turn, issue PYTHON_READ with exactly one `call` query"
                )
            else:
                first_read = "On the FIRST tool turn, issue exactly one CALL command"
            exact_read_contract = (
                "\nHIGHEST-PRIORITY CLEAN-READ RULE:\n"
                f"- {first_read} for "
                f"{tool_name!r}, using the current action arguments unchanged except `app`/`action`.\n"
                "- Do not infer the observation from file contents or history before making that call.\n"
                "- Return the host tool result VERBATIM. Preserve exact wording, formatting, whitespace, "
                "empty values, and failure text; do not summarize or repair it.\n"
            )
        elif not dirty and not self.uses_fixed_context and self._query is not None:
            first_read = (
                "use PYTHON_READ with READ/LIST operations"
                if self.program_only or officebench_wm_tool_mode() == "python"
                else "issue READ/LIST commands"
            )
            exact_read_contract = (
                "\nHIGHEST-PRIORITY CLEAN-READ RULE:\n"
                f"- This read has no identical CALL tool. On the FIRST tool turn, {first_read} over "
                "every operand needed by the current action.\n"
                "- Reproduce the native read observation exactly from those host results, including "
                "formatting, ordering, counts, empty output, and error wording. Do not guess without "
                "grounding first.\n"
            )
        immutable_objects = None
        immutable_objects_error = None
        shell_stages = None
        shell_command = str(action.get("command") or "").strip()
        shell_listing_read = bool(
            app == "shell"
            and re.match(r"^(?:find|ls)\b", shell_command)
        )
        if (
            not dirty
            and shell_listing_read
            and self._query is not None
            and read_target(action, self._current_workdir()) is not None
        ):
            raw_immutable_objects = self._query.list(
                f"/testbed/{target}",
                recursive=True,
            )
            try:
                parsed_objects = json.loads(raw_immutable_objects)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed_objects = None
            if isinstance(parsed_objects, list):
                immutable_objects = parsed_objects
            else:
                raise OfficeBenchSimulationError(
                    f"Unable to ground shell inventory for /testbed/{target}: "
                    f"{str(raw_immutable_objects)[:1000]}"
                )
            parsed_stages = _readonly_shell_stages(
                shell_command,
                self._current_workdir(),
            )
            shell_stages = parsed_stages if parsed_stages is not None else []
            if immutable_objects_error is None:
                emulated = _emulate_inventory_pipeline(
                    shell_stages,
                    immutable_objects,
                    self._current_workdir(),
                )
                if emulated is not None:
                    return emulated
            exact_read_contract = (
                "\nHIGHEST-PRIORITY CLEAN-READ RULE:\n"
                "- The host copied a recursive immutable inventory as a list of "
                "`{'name','path','type'}` records in `data['immutable_objects']` and parsed the "
                "pipeline argv into `data['shell_stages']`.\n"
                "- Use PYTHON_FINAL to reproduce the requested read-only shell pipeline from that "
                "inventory. Do not request repeated one-level LIST operations or guess paths.\n"
            )
        operand_reads = []
        if (not dirty and not self.uses_fixed_context and self._query is not None
                and not grounded_clean_read and not can_call_exact and target
                and app in ("excel", "word", "pdf", "ocr", "calendar")):
            path = f"/testbed/{target}"
            content = self._query.read_raw(path)
            operand_reads = [{"op": "read", "path": path, "result": content,
                              **read_schema(path, content)}]
            exact_read_contract = (
                "\nThe immutable operand is already grounded in data['reads'][0]['result'], "
                "with its representation schema. Return PYTHON_FINAL to compute the full native "
                "observation from that copied value. Do not reinterpret it as bytes or omit app wording.\n"
            )
        program_output = self.program_only or officebench_wm_tool_mode() == "python"
        processing_instruction = (
            "Do not transcribe, truncate, summarize, or otherwise process evidence text outside "
            "generated Python. Return a PYTHON_FINAL program whose printed value is the exact content "
            "or observation the read would print."
            if program_output else
            "Use the bounded READ/LIST/CALL/TRACE protocol when grounding is needed, then return the "
            "exact content or observation in the requested PREDICT format."
        )
        prompt = (
            f"You are a world model of the OfficeBench '{app}' app. The agent called the READ action "
            f"'{app}.{act}' ({_action_str(action)}). {state_instruction}"
            f"{exact_read_contract}\n"
            f"PRISTINE / FIXED evidence:\n{baseline}\n\n"
            "Applied writes and history are available only through `data['writes']` and "
            f"`data['history']`. {processing_instruction}"
        )
        out = self._predict(
            prompt + '\n\n' + action_contract(action),
            "PREDICT: <the exact resulting file content or read observation, with no prose>",
            max_tokens=_READ_MAX_TOKENS,
            python_context={
                "kind": "dirty_read" if dirty else "clean_read",
                "action": _canonical_action_paths(action, self._current_workdir()),
                "observation_action": dict(action),
                "enforce_observation_contract": self.program_only,
                "current_state": ({target: self._current_state_data(target, pre_write_content)}
                                  if dirty and target else {}),
                "target": target,
                "reads": operand_reads,
                "pre_write_content": pre_write_content,
                "grounded_clean_read": (
                    pre_write_content if grounded_clean_read else None
                ),
                "grounded_clean_read_required": grounded_clean_read,
                "exact_read_required": bool(exact_read_contract),
                "exact_read_call": (
                    {
                        "name": tool_name,
                        "args": {
                            key: value
                            for key, value in action.items()
                            if key not in ("app", "action")
                        },
                    }
                    if can_call_exact else None
                ),
                "immutable_objects": immutable_objects,
                "immutable_objects_error": immutable_objects_error,
                "shell_stages": shell_stages,
                "writes": [
                    {"action": write, "observation": observation}
                    for write, observation in self._related_writes(target)
                ],
                "history": self._agent_context_data(history),
            },
            python_force_long=True,
            python_force_always=self.program_only,
        )
        if out is not None:
            return out
        raise OfficeBenchSimulationError(
            f"Unable to simulate {app}.{act}; application observation is unknown"
        )

    def predict_file_state(self, path: str, initial_content: str) -> str:
        """Render a gold-checked file's PREDICTED final content (initial + its writes) for the verifier.

        Only used when the file has NO gold reference to compare against (e.g. cell-value / keyword
        checks); reference-grounded files are judged from initial + write-log directly (no fabrication).
        The prompt is deliberately conservative — apply only write-justified changes, keep everything
        else verbatim — so the reconstruction is as trustworthy as possible without executing writes.
        """
        target = _norm_path(path)
        if target.startswith('emails/'):
            mailbox = self._mailbox_state(target)
            if mailbox is not None:
                return mailbox
        writes = self._related_writes(target)
        if not writes and not self._is_deleted(target) and not self.global_dirty:
            return initial_content
        exact = self._exact_overlay_content(target, initial_content)
        if exact is not None:
            return exact
        app_hint = writes[0][0].get("app", "office") if writes else "office"
        prompt = (
            f"You are a world model of an OfficeBench '{app_hint}' file. Given the initial content of "
            f"file '{target}' and the WRITE actions the agent applied to it, predict the EXACT FINAL "
            "content of the file. Initial content and writes are available only through "
            "`data['initial_content']`, `data['writes']`, and `data['global_writes']`; generated Python "
            "must perform every copy or transformation.\n\n"
            "WRITE SCHEMA: each entry['action'] is a DICTIONARY with 'app', 'action', and argument "
            "fields directly inside it (not an action name string and not an 'args' wrapper). "
            "Skip entries with applied=False. A conversion may supply data['conversion_source']: "
            "its state is the copied logical INPUT at conversion time, including prior edits. "
            "SCHEMA: snapshot = data['conversion_source']; input_record = snapshot['state']; "
            "input_text = input_record['content'] (a decoded STRING, not bytes or a dict). "
            "For Word conversion this is paragraph text: preserve its characters and predict only "
            "PDF whitespace/layout. For Excel it is the documented decoded cell representation. "
            "Use that snapshot, not pristine output absence or later source edits. Predict decoded "
            "PDF text from this evidence; preserve input values and distinguish unknown layout "
            "from missing content. Never execute LibreOffice, native apps, subprocesses, file I/O "
            "or binary encoders. Only pure computation over copied values is permitted.\n\n"
            f"Change ONLY the cells/lines a listed write clearly determines (e.g. a set_cell write sets "
            f"exactly that cell to that value); copy every other part of the file verbatim from the "
            "initial content and never invent values. Return a PYTHON_FINAL program whose printed value "
            "is the exact final file content."
        )
        out = self._predict(
            prompt,
            "PREDICT: <the exact final file content, with no prose>",
            max_tokens=_STATE_MAX_TOKENS,
            python_context={
                "kind": "final_file_state",
                "enforce_observation_contract": self.program_only,
                "target": target,
                "initial_content": initial_content,
                "conversion_source": self.conversion_sources.get(target),
                "writes": [
                    {"action": write, "observation": observation}
                    for write, observation in writes
                ],
                "global_writes": self._write_history_data(),
                "global_dirty": self.global_dirty,
            },
            python_force_long=True,
            python_force_always=self.program_only,
        )
        if out is None:
            raise OfficeBenchSimulationError(
                f"Unable to reconstruct /testbed/{target}; final state is unknown"
            )
        return out

    def predicted_write_log(self) -> str:
        """Complete transitions for reconstruction and verification, including write literals."""
        if self.transition_log.entries():
            return self._complete_trace(self.transition_log.entries())
        lines = [
            f"  - {json.dumps(a, default=str)} -> [{o}]" for a, o in self.write_history
        ]
        return "\n".join(lines) or "  (no writes)"

    def _predict(
        self,
        prompt: str,
        final_instr: str,
        *,
        max_tokens: int,
        python_context: Optional[Dict] = None,
        python_force_long: bool = False,
        python_force_always: bool = False,
    ) -> Optional[str]:
        """Use the agentic grounding loop when a pristine-session query interface is available."""
        query = self._query
        if (
            (self.program_only or officebench_wm_tool_mode() == "python")
            and python_context is not None
        ):
            result = run_python_prediction_loop(
                lambda text, budget: self._complete(text, max_tokens=budget),
                query,
                prompt,
                final_instr,
                context=python_context,
                max_turns=self.MAX_TURNS,
                max_tokens=max_tokens,
                output_kind="text",
                force_long=python_force_long,
                force_always=python_force_always,
            )
            self._record_python_result(result)
            if not result.valid:
                raise OfficeBenchSimulationError(
                    f"Python simulation failed for {python_context.get('kind', 'prediction')}: "
                    f"{result.error or 'prediction protocol exhausted without a valid result'}",
                    retryable=False,
                )
            return result.output
        if query is None:
            output = self._complete(prompt, max_tokens=max_tokens)
            return output if output else None
        terminal = run_agentic_read_loop(
            lambda text, budget: self._complete(text, max_tokens=budget),
            query,
            prompt,
            final_instr,
            max_turns=self.MAX_TURNS,
            max_tokens=max_tokens,
        )
        output = (terminal[1] if terminal else "").strip()
        return output if output else None

    def _complete(self, prompt: str, *, max_tokens: int) -> str:
        try:
            self.n_llm_calls += 1
            system = (
                self._flat_system_prompt
                or _OFFICEBENCH_SIMULATOR_SYSTEM
            )
            return (self._chat.chat(system,
                                    prompt, temperature=0.0, max_tokens=max_tokens) or "").strip()
        except Exception:  # noqa: BLE001 - a prediction failure must never crash the rollout
            return ""


__all__ = [
    "OfficeBenchWMPredictor",
    "OfficeBenchWMQuery",
    "action_inputs",
    "is_write_action",
    "run_agentic_read_loop",
    "write_targets",
    "read_target",
    "symbolic_file_state",
]
