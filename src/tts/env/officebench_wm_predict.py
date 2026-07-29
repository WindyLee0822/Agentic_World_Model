"""Predictive, write-safe world model for OfficeBench (coherent with the τ²/AppWorld contract).

The read-grounded TTS premise is that during sampling the agent interacts with a **world model**, not
the real environment: **read** actions are answered exactly (read-only) against the real starting
state, **write** actions are **LLM-predicted and NEVER executed**, and a **read after a write** returns
the *predicted* post-write state (the τ² "dirty read").  The real environment is therefore never
mutated while the ``N`` candidates are drawn; only the single selected trajectory is committed (really
executed) at the chokepoint.

As in τ², nontrivial predictions use a bounded world-model subagent with four domain-neutral,
read-only simulation tools: ``READ`` current logical state, ``LIST`` the logical namespace, ``CALL``
a safe read-only domain operation, and ``TRACE`` the automatic transition log. The logical state is
the immutable pristine ``/testbed`` plus predicted transitions; no candidate write is executed.

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
from typing import Callable, Dict, List, Optional, Tuple

from ..simulator.transition_log import TransitionLog
from .officebench_lowlevel import normalize_answer
from .officebench_wm_python import (
    officebench_wm_tool_mode,
    run_python_prediction_loop,
)

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
    if command == "find" and args and args[0].startswith("-"):
        return False
    if command == "sed" and not _sed_stage_is_read(tokens):
        return False
    if command == "sort" and any(
            _short_option_present(args, "o")
            or arg.startswith("--output")
            or arg.startswith("--compress-program")
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
            _short_option_present(args, "C") or arg == "--compile" for arg in args):
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


def _readonly_shell_stages(
        command: str, workdir: str = "/testbed") -> Optional[List[List[str]]]:
    """Parse a shell command only when every stage is provably non-mutating."""
    if not command or re.search(r"[\r\n`$]", str(command)):
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
    return (
        action.get("app") in ("word", "excel")
        and action.get("action") == "create_new_file"
    )


def _parent_resources(target: str) -> List[str]:
    parts = _norm_path(target).split("/")[:-1]
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def action_inputs(action: Dict, workdir: str = "/testbed") -> List[str]:
    """Logical resources an action reads as inputs before producing its result."""
    app, act = action.get("app"), action.get("action")
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
            if output:
                return [output]
            source_key = "excel_file_path" if app == "excel" else "word_file_path"
            source = _path_arg(a, source_key, "file_path", workdir=workdir)
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
        for who in (a.get("recipient"), a.get("sender")):
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
    * ``CALL`` — a whitelisted read-only app operation, routed through the overlay when dirty;
    * ``TRACE`` — the ordered transition log for one resource.
    """

    def __init__(self, session, workdir: Callable[[], str]) -> None:
        self._session = session
        self._workdir = workdir
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
        catalogue = session.apps_catalogue()
        available = []
        for app, actions in (catalogue.get("actions", {}) or {}).items():
            for action in actions:
                name = f"{app}.{action}"
                probe = {"app": app, "action": action}
                if app not in ("system", "shell") and not is_write_action(probe):
                    available.append(name)
        self.available_calls = list(dict.fromkeys(available))

    def bind_simulation(self, *, logical_read, trace, overlay_paths, deleted_paths,
                        recursive_deleted_paths, revived_paths, is_deleted, is_dirty) -> None:
        """Bind domain state operations after the predictor and query interface are constructed."""
        self._logical_read = logical_read
        self._trace = trace
        self._overlay_paths = overlay_paths
        self._deleted_paths = deleted_paths
        self._recursive_deleted_paths = recursive_deleted_paths
        self._revived_paths = revived_paths
        self._is_deleted = is_deleted
        self._is_dirty = is_dirty

    def workdir(self) -> str:
        return self._workdir() or "/"

    def _read_pristine(self, safe_path: str) -> str:
        try:
            state = self._session.read_state(
                [{"path": safe_path, "doc_type": _doc_type(safe_path)}])
            return str(state.get(safe_path, "[file does not exist]"))[:_READ_CONTENT_CHARS]
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

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
            return str(self._logical_read(target, pristine))[:_READ_CONTENT_CHARS]
        return pristine

    def list(self, path: str) -> str:
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
        action = {
            "app": "shell",
            "action": "command",
            "command": f"ls -la -- {shlex.quote(safe_path)}",
        }
        try:
            result = self._session.exec_action(action, self.workdir())
            base = str(result.get("observation", result))[:_READ_CONTENT_CHARS]
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
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

    def call(self, name: str, args: Dict) -> str:
        """Invoke a whitelisted read action, using the overlay instead of stale state when dirty."""
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
        if target and self._is_dirty is not None and self._is_dirty(target):
            return self.read(f"/testbed/{target}")
        try:
            result = self._session.exec_action(action, self.workdir())
            return str(result.get("observation", result))[:_READ_CONTENT_CHARS]
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
        return str(self._trace(_norm_path(safe_path)))[:_READ_CONTENT_CHARS]


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


#: How much exact file content to feed the predictor.  Real OfficeBench files render LONG (measured on
#: the task corpus: docx median ~1.8k / p90 ~13k chars, pdf median ~0.6k / p90 ~12k, xlsx p90 ~1.3k /
#: max ~15k), so the previous 2000-char cap truncated ~9-26% of files (and the informative tail of the
#: big ones).  Default 6000 keeps the vast majority intact; override with the env vars.
_READ_CONTENT_CHARS = _int_env("TTS_OFFICEBENCH_WM_READ_CHARS", 6000)
_STATE_CONTENT_CHARS = _int_env("TTS_OFFICEBENCH_WM_STATE_CHARS", 6000)
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

_CELL_RE = re.compile(r"\((\d+),\s*(\d+)\):\s*([^\t\n]*)")


def _parse_excel_grid(content: str) -> Dict[Tuple[int, int], str]:
    """Parse the official ``excel_read_file`` rendering (``(r, c): value`` tokens) into a cell map."""
    grid: Dict[Tuple[int, int], str] = {}
    for m in _CELL_RE.finditer(content or ""):
        grid[(int(m.group(1)), int(m.group(2)))] = m.group(3).rstrip()
    return grid


def _render_excel_grid(grid: Dict[Tuple[int, int], str]) -> str:
    """Render a cell map back into the official ``(r, c): value`` form (rows by ``\\n``, cells by tab)."""
    from itertools import groupby

    lines = []
    for row, cells in groupby(sorted(grid), key=lambda rc: rc[0]):
        lines.append("\t".join(f"({row}, {c}): {grid[(row, c)]}" for (_, c) in cells))
    return "\n".join(lines)


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
            grid[(row, col)] = str(val)
            changed = True
        else:
            return None
    return _render_excel_grid(grid) if changed else initial


def _apply_word_writes(initial: str, writes: List[Tuple[Dict, str]]) -> Optional[str]:
    """Reconstruct a word file authored in-trajectory by carrying its literal ``contents`` verbatim.

    Only handled when the file is created/empty in the trajectory (so its content is fully determined
    by the write literals); a write_to_file with no literal content, or an append onto pre-existing
    real content whose overwrite/append semantics we cannot be sure of, returns ``None`` (→ LLM).
    """
    created = any(a.get("action") == "create_new_file" for a, _ in writes)
    init_empty = (not initial) or initial.strip() in ("", "[file does not exist]")
    if not (created or init_empty):
        return None
    parts: List[str] = []
    for action, _obs in writes:
        content = _content_arg(action)
        if action.get("action") == "create_new_file":
            parts = []
            if content:
                parts.append(content)
        else:  # write_to_file
            if content is None:
                return None
            parts.append(content)
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
        self.transition_log = TransitionLog()
        #: target (normalised path) -> list of (action, predicted_observation) applied to it, in order.
        self.applied_writes: Dict[str, List[Tuple[Dict, str]]] = {}
        #: exact pre-write content read once per target (for dirty-read grounding).
        self.pre_write_content: Dict[str, str] = {}
        self.write_history: List[Tuple[Dict, str]] = []
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
        self.n_dirty_reads = 0
        self.n_llm_calls = 0
        self.n_python_calls = 0
        self.n_python_failures = 0
        self.n_python_forced = 0
        self.n_python_mode_misses = 0

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
                f"  - {_action_str(entry.action)} -> [{entry.observation[:200]}]"
                for entry in entries
            )
        # Compatibility for tests/callers that seed ``applied_writes`` directly.
        items = [
            item for resource, resource_items in self.applied_writes.items()
            if related(resource)
            for item in resource_items
        ]
        return "\n".join(f"  - {_action_str(a)} -> [{o[:200]}]" for a, o in items) or "  (none)"

    def _related_writes(self, target: str) -> List[Tuple[Dict, str]]:
        target = _norm_path(target)
        prefix = "" if not target else target.rstrip("/") + "/"
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
        return self.transition_log.render(target) if entries else self._writes_text(target)

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
            return self.transition_log.render(target)
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

    def _write_history_data(self, k: int = 40) -> List[Dict]:
        return [
            {"action": action, "observation": observation}
            for action, observation in self.write_history[-k:]
        ]

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
            "Use READ for current logical state and TRACE for prior predicted transitions."
        )
        if _wm_check_writes():
            prompt = (
                f"You are a faithful world model of the OfficeBench '{app}' app. Predict the observation "
                f"the real app prints for the WRITE action '{app}.{act}' with arguments "
                f"{_action_str(action)}.\n{footprint}\n\n"
                f"TRAJECTORY SO FAR (action -> observation; shows which files/dirs exist and whether "
                f"earlier steps errored):\n{self._agent_context_text(history)}\n\n"
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
        obs = self._predict(
            prompt,
            "PREDICT: <the exact single-line observation returned by the write action>",
            max_tokens=120,
            python_context={
                "kind": "write_observation",
                "action": recorded_action,
                "inputs": inputs,
                "targets": targets,
                "history": self._agent_context_data(history),
                "writes": self._write_history_data(),
            },
        ) or f"OBSERVATION: Successfully executed {app}.{act}."
        obs = obs.strip().splitlines()[0] if obs.strip() else obs
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
            self.applied_writes.setdefault(tgt, []).append((recorded_action, obs))
            if tgt in ("<shell>", "<unknown>"):
                # an unresolved write target: we can no longer certify any file is untouched, so from
                # here every read is predicted (τ²'s global-dirty fallback for ref-less writes).
                self.global_dirty = True
        if self.global_dirty:
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

    def predict_read(self, action: Dict, pre_write_content: str, history: List[Tuple] = ()) -> str:
        """A read AFTER a write to the same file: predict the post-write content it returns now."""
        self.n_dirty_reads += 1
        target = read_target(action, self._current_workdir()) or ""
        app, act = action.get("app"), action.get("action")
        # Deterministic short-circuit: if every write to this file is exactly modelable, apply them with
        # a pure function (no LLM, no drift) instead of asking the LLM to re-derive the post-write file.
        exact = self._exact_overlay_content(target, pre_write_content)
        if exact is not None:
            return exact
        # target-specific writes, or — when the read is dirty only via the global-dirty fallback — the
        # full write history (so a globally-dirty read still sees what may have changed the file).
        writes_text = self._history_text(k=20) if self.global_dirty else self._writes_text(target)
        prompt = (
            f"You are a world model of the OfficeBench '{app}' app's files. The agent READS via "
            f"'{app}.{act}' ({_action_str(action)}) a file that earlier WRITE actions may have changed. "
            f"Predict the EXACT content the read returns NOW by applying those writes to the pre-write "
            f"content.\n\n"
            f"PRE-WRITE content (what the real file held BEFORE the writes below):\n"
            f"{pre_write_content[:_READ_CONTENT_CHARS]}\n\n"
            f"WRITE actions applied since (in order):\n{writes_text}\n\n"
            f"Apply ONLY changes the listed writes clearly justify; leave every other part of the file "
            f"exactly as in the pre-write content, and never invent data. If none of the writes affect "
            f"this file, return the pre-write content UNCHANGED. Return ONLY the resulting file "
            f"content/observation as the read would print it. No prose."
        )
        out = self._predict(
            prompt,
            "PREDICT: <the exact resulting file content or read observation, with no prose>",
            max_tokens=_READ_MAX_TOKENS,
            python_context={
                "kind": "dirty_read",
                "action": _canonical_action_paths(action, self._current_workdir()),
                "target": target,
                "pre_write_content": pre_write_content,
                "writes": [
                    {"action": write, "observation": observation}
                    for write, observation in self._related_writes(target)
                ],
                "history": self._agent_context_data(history),
            },
            python_force_long=True,
        )
        return out if out is not None else self.logical_read(target, pre_write_content)

    def predict_file_state(self, path: str, initial_content: str) -> str:
        """Render a gold-checked file's PREDICTED final content (initial + its writes) for the verifier.

        Only used when the file has NO gold reference to compare against (e.g. cell-value / keyword
        checks); reference-grounded files are judged from initial + write-log directly (no fabrication).
        The prompt is deliberately conservative — apply only write-justified changes, keep everything
        else verbatim — so the reconstruction is as trustworthy as possible without executing writes.
        """
        target = _norm_path(path)
        writes = self._related_writes(target)
        if not writes and not self._is_deleted(target) and not self.global_dirty:
            return initial_content  # untouched by the trajectory
        exact = self._exact_overlay_content(target, initial_content)
        if exact is not None:
            return exact  # exact deterministic reconstruction (no LLM re-derivation, no drift)
        app_hint = writes[0][0].get("app", "office") if writes else "office"
        writes_text = self._history_text(k=40) if self.global_dirty else self._writes_text(target)
        prompt = (
            f"You are a world model of an OfficeBench '{app_hint}' file. Given the initial content of "
            f"file '{target}' and the WRITE actions the agent applied to it, predict the EXACT FINAL "
            f"content of the file.\n\n"
            f"INITIAL content:\n{initial_content[:_STATE_CONTENT_CHARS]}\n\n"
            f"WRITE actions applied (in order):\n{writes_text}\n\n"
            f"Change ONLY the cells/lines a listed write clearly determines (e.g. a set_cell write sets "
            f"exactly that cell to that value); copy every other part of the file verbatim from the "
            f"initial content and never invent values. Return ONLY the final file content as it would be "
            f"read back. No prose."
        )
        out = self._predict(
            prompt,
            "PREDICT: <the exact final file content, with no prose>",
            max_tokens=_STATE_MAX_TOKENS,
            python_context={
                "kind": "final_file_state",
                "target": target,
                "initial_content": initial_content,
                "writes": [
                    {"action": write, "observation": observation}
                    for write, observation in writes
                ],
                "global_writes": self._write_history_data(),
                "global_dirty": self.global_dirty,
            },
            python_force_long=True,
        )
        return out if out is not None else self.logical_read(target, initial_content)

    def predicted_write_log(self, cap: int = 40) -> str:
        """A compact rendering of every predicted write (for the verifier when no gold file is checked)."""
        if self.transition_log.entries():
            return self.transition_log.render(cap=cap)
        lines = [f"  - {_action_str(a)} -> [{o[:200]}]" for a, o in self.write_history[:cap]]
        return "\n".join(lines) or "  (no writes)"

    def _predict(
        self,
        prompt: str,
        final_instr: str,
        *,
        max_tokens: int,
        python_context: Optional[Dict] = None,
        python_force_long: bool = False,
    ) -> Optional[str]:
        """Use the agentic grounding loop when a pristine-session query interface is available."""
        if officebench_wm_tool_mode() == "python" and python_context is not None:
            result = run_python_prediction_loop(
                lambda text, budget: self._complete(text, max_tokens=budget),
                self._query,
                prompt,
                final_instr,
                context=python_context,
                max_turns=self.MAX_TURNS,
                max_tokens=max_tokens,
                output_kind="text",
                force_long=python_force_long,
            )
            self._record_python_result(result)
            return result.output if result.valid else None
        if self._query is None:
            output = self._complete(prompt, max_tokens=max_tokens)
            return output if output else None
        terminal = run_agentic_read_loop(
            lambda text, budget: self._complete(text, max_tokens=budget),
            self._query,
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
            return (self._chat.chat("You are a faithful world model. Answer exactly, no prose.",
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
