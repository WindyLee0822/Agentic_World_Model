"""Restricted pure-Python computation for world-model predictions.

The model may use this helper to transform copied JSON-like evidence into an exact prediction.  The
program never receives an environment object, requester, filesystem handle, network client, or process
environment.  It runs in a dedicated helper process with a small builtin/method allowlist, bounded
input/output, an AST size limit, and a wall-clock timeout.
"""

from __future__ import annotations

import ast
import atexit
import builtins
import datetime as datetime_module
import io
import json
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional


@dataclass(frozen=True)
class PythonToolResult:
    output: str = ""
    error: Optional[str] = None
    fatal: bool = False


_ALLOWED_NODES = {
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.FunctionDef,
    ast.Lambda,
    ast.Return,
    ast.arguments,
    ast.arg,
    ast.Import,
    ast.ImportFrom,
    ast.alias,
    ast.For,
    ast.If,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Call,
    ast.keyword,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Subscript,
    ast.Slice,
    ast.Attribute,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.And,
    ast.Or,
    ast.Not,
    ast.UAdd,
    ast.USub,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
}

_SAFE_CALL_NAMES = {
    "abs",
    "all",
    "any",
    "bool",
    "date_add",
    "date",
    "datetime",
    "dict",
    "dumps",
    "enumerate",
    "float",
    "globals",
    "int",
    "isinstance",
    "json_dumps",
    "json_loads",
    "len",
    "list",
    "locals",
    "loads",
    "max",
    "min",
    "month_end",
    "month_start",
    "next",
    "parse_date",
    "parse_datetime",
    "print",
    "print_json",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "timedelta",
    "tuple",
    "zip",
    "exit",
}

_SAFE_METHODS = {
    "add",
    "append",
    "clear",
    "copy",
    "count",
    "date",
    "discard",
    "endswith",
    "extend",
    "fromisoformat",
    "get",
    "index",
    "insert",
    "isoformat",
    "items",
    "join",
    "keys",
    "lower",
    "lstrip",
    "pop",
    "remove",
    "replace",
    "reverse",
    "rstrip",
    "setdefault",
    "sort",
    "split",
    "splitlines",
    "startswith",
    "strip",
    "title",
    "upper",
    "update",
    "values",
    "dumps",
    "loads",
    "strptime",
}

_SAFE_ATTRIBUTES = _SAFE_METHODS | {
    "day",
    "datetime",
    "days",
    "hour",
    "microsecond",
    "minute",
    "month",
    "second",
    "seconds",
    "timedelta",
    "year",
}

_FORBIDDEN_NAMES = {
    "__builtins__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "help",
    "input",
    "memoryview",
    "open",
    "os",
    "quit",
    "setattr",
    "socket",
    "subprocess",
    "sys",
    "type",
    "vars",
}


class _Validator(ast.NodeVisitor):
    def __init__(self, max_nodes: int, function_names: set[str]) -> None:
        self.max_nodes = max_nodes
        self.function_names = function_names
        self.nodes = 0

    def generic_visit(self, node) -> None:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise ValueError(f"program exceeds {self.max_nodes} AST nodes")
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"{type(node).__name__} is not allowed")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_") or node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"name {node.id!r} is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr not in _SAFE_ATTRIBUTES:
            raise ValueError(f"attribute {node.attr!r} is not allowed")
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("attribute assignment/deletion is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_CALL_NAMES and node.func.id not in self.function_names:
                raise ValueError(f"call to {node.func.id!r} is not allowed")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr not in _SAFE_METHODS:
                raise ValueError(f"method {node.func.attr!r} is not allowed")
        else:
            raise ValueError("indirect calls are not allowed")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("_"):
            raise ValueError(f"function {node.name!r} is not allowed")
        if node.decorator_list:
            raise ValueError("function decorators are not allowed")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in ("datetime", "json"):
                raise ValueError(f"import of {alias.name!r} is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        allowed = {
            "datetime": {"date", "datetime", "timedelta"},
            "json": {"dumps", "loads"},
        }
        if node.level or node.module not in allowed:
            raise ValueError(f"import from {node.module!r} is not allowed")
        for alias in node.names:
            if alias.name not in allowed[node.module]:
                raise ValueError(
                    f"import of {node.module}.{alias.name} is not allowed")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, int) and abs(node.value) > 10_000_000:
            raise ValueError("integer literal is too large")
        if isinstance(node.value, (str, bytes)) and len(node.value) > 100_000:
            raise ValueError("string literal is too large")
        self.generic_visit(node)


def _bounded_sequence(value, *, limit: int = 100_000):
    if isinstance(value, range):
        if len(value) > limit:
            raise ValueError(f"sequence exceeds {limit} items")
        return value
    items = list(value)
    if len(items) > limit:
        raise ValueError(f"sequence exceeds {limit} items")
    return items


def _safe_range(*args):
    value = range(*args)
    if len(value) > 100_000:
        raise ValueError("range exceeds 100000 items")
    return value


def _safe_sorted(value, *args, **kwargs):
    return sorted(_bounded_sequence(value), *args, **kwargs)


def _safe_sum(value, start=0):
    return sum(_bounded_sequence(value), start)


def _safe_min(value, *args, **kwargs):
    if args:
        return min((value, *args), **kwargs)
    return min(_bounded_sequence(value), **kwargs)


def _safe_max(value, *args, **kwargs):
    if args:
        return max((value, *args), **kwargs)
    return max(_bounded_sequence(value), **kwargs)


def _parse_datetime(value: Any) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def _parse_date(value: Any) -> date:
    text = str(value).strip()
    if "T" in text or " " in text:
        return _parse_datetime(text).date()
    return date.fromisoformat(text[:10])


def _date_add(value: Any, days: int = 0):
    parsed = _parse_datetime(value) if ("T" in str(value) or " " in str(value)) else _parse_date(value)
    return parsed + timedelta(days=int(days))


def _month_start(value: Any, offset: int = 0) -> date:
    current = _parse_date(value)
    month_index = current.year * 12 + current.month - 1 + int(offset)
    return date(month_index // 12, month_index % 12 + 1, 1)


def _month_end(value: Any, offset: int = 0) -> date:
    return _month_start(value, int(offset) + 1) - timedelta(days=1)


class _ProgramExit(Exception):
    pass


def _bounded_json(value: Any, limit: int) -> str:
    chunks = []
    length = 0
    encoder = json.JSONEncoder(ensure_ascii=True, sort_keys=True, default=str)
    for chunk in encoder.iterencode(value):
        length += len(chunk)
        if length > limit:
            raise ValueError(f"JSON output exceeds {limit} characters")
        chunks.append(chunk)
    return "".join(chunks)


def _error_text(exc: Exception, limit: int) -> str:
    parts = []
    for arg in getattr(exc, "args", ())[:3]:
        if isinstance(arg, str):
            parts.append(arg[:500])
        elif isinstance(arg, (int, float, bool, type(None))):
            parts.append(str(arg))
        else:
            parts.append(f"<{type(arg).__name__}>")
    detail = ", ".join(parts)
    text = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return text[:max(200, min(limit, 2_000))]


def _execute_restricted(
    code: str,
    data: Any,
    *,
    max_output_chars: int,
    max_nodes: int,
    max_steps: int,
    timeout_s: float,
    allow_bare_container_output: bool,
) -> PythonToolResult:
    try:
        tree = ast.parse(code, mode="exec")
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        _Validator(max_nodes, function_names).visit(tree)
    except Exception as exc:
        return PythonToolResult(error=f"invalid program: {exc}"[:2_000])

    output = io.StringIO()

    def safe_print(*values, sep=" ", end="\n"):
        rendered = []
        has_scalar_label = any(
            not isinstance(value, (dict, list, set, tuple))
            for value in values
        )
        for value in values:
            if isinstance(value, (dict, list, set, tuple)):
                if not has_scalar_label:
                    if not allow_bare_container_output:
                        raise ValueError("use print_json(...) for bare container output")
                    rendered.append(_bounded_json(value, max_output_chars))
                    continue
                # LLM-generated verifier programs commonly print labelled diagnostics such as
                # ``print("MISSING:", missing_ids)`` before the final verdict. These diagnostics are
                # bounded here and discarded by the caller; bare containers remain JSON-only so a
                # state prediction cannot accidentally accept a Python repr as its result.
                rendered.append(str(value))
                continue
            rendered.append(str(value))
        text = str(sep).join(rendered) + str(end)
        if output.tell() + len(text) > max_output_chars:
            raise ValueError(f"output exceeds {max_output_chars} characters")
        output.write(text)

    def print_json(value):
        text = _bounded_json(value, max_output_chars - output.tell())
        if output.tell() + len(text) + 1 > max_output_chars:
            raise ValueError(f"output exceeds {max_output_chars} characters")
        output.write(text + "\n")

    def json_dumps(value):
        return _bounded_json(value, max_output_chars)

    def json_loads(value):
        return json.loads(str(value))

    def safe_scope():
        return {"data": data}

    def safe_exit(*_args):
        raise _ProgramExit()

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            raise ImportError("relative imports are unavailable")
        if name == "json":
            return json
        if name == "datetime":
            return datetime_module
        if name in {"_locale", "_strptime", "calendar", "locale", "re", "time"}:
            # Internal dependencies lazily imported by ``datetime.strptime``. Model code cannot
            # request them directly because the AST validator permits only ``json``/``datetime``.
            return builtins.__import__(name, globals, locals, fromlist, level)
        raise ImportError(f"module {name!r} is unavailable")

    safe_builtins = {
        "__import__": safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "globals": safe_scope,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "locals": safe_scope,
        "max": _safe_max,
        "min": _safe_min,
        "next": next,
        "print": safe_print,
        "range": _safe_range,
        "round": round,
        "set": set,
        "sorted": _safe_sorted,
        "str": str,
        "sum": _safe_sum,
        "tuple": tuple,
        "zip": zip,
        "exit": safe_exit,
    }
    globals_dict = {
        "__builtins__": safe_builtins,
        "data": data,
        "date_add": _date_add,
        "json_dumps": json_dumps,
        "json_loads": json_loads,
        "month_end": _month_end,
        "month_start": _month_start,
        "parse_date": _parse_date,
        "parse_datetime": _parse_datetime,
        "print_json": print_json,
        "timedelta": timedelta,
    }

    deadline = time.monotonic() + max(0.05, float(timeout_s))
    steps = 0

    def trace(_frame, event, _arg):
        nonlocal steps
        if event == "line":
            steps += 1
            if steps > max_steps:
                raise TimeoutError(f"program exceeds {max_steps} executed lines")
            if time.monotonic() > deadline:
                raise TimeoutError(f"program exceeds {timeout_s:.2f}s")
        return trace

    import sys

    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        exec(compile(tree, "<wm-python>", "exec"), globals_dict, globals_dict)
    except _ProgramExit:
        pass
    except Exception as exc:
        return PythonToolResult(error=_error_text(exc, max_output_chars))
    finally:
        sys.settrace(previous_trace)

    text = output.getvalue().strip()
    if not text:
        return PythonToolResult(error="program produced no output")
    return PythonToolResult(output=text)


def _apply_worker_limits() -> None:
    """Bound additional address space and prohibit file creation in the helper process."""
    try:
        import resource

        extra_mb = max(32, int(os.environ.get("TTS_WM_PYTHON_MEMORY_MB", "128")))
        current_vms = 0
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmSize:"):
                        current_vms = int(line.split()[1]) * 1024
                        break
        except Exception:
            current_vms = 0
        if current_vms:
            address_limit = current_vms + extra_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def _worker(conn, inherited_parent_conn=None) -> None:
    temp_dir = tempfile.mkdtemp(prefix="tts-wm-python-")
    try:
        # Under POSIX ``fork`` the child inherits both ends of the pipe. Closing the parent's end here
        # is essential: if the benchmark worker is SIGTERM/SIGKILLed, ``conn.recv`` then observes EOF
        # and this helper exits instead of becoming an orphan that keeps process-pool descriptors alive.
        if inherited_parent_conn is not None:
            inherited_parent_conn.close()
        _apply_worker_limits()
        os.environ.clear()
        os.chdir(temp_dir)
        while True:
            message = conn.recv()
            if message is None:
                return
            request_id, code, data, limits = message
            result = _execute_restricted(code, data, **limits)
            conn.send((request_id, result))
    except EOFError:
        return
    finally:
        conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


class PurePythonTool:
    """A reusable isolated worker for restricted world-model computation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owner_pid = os.getpid()
        self._process = None
        self._conn = None
        self._request_id = 0

    def prepare(self) -> bool:
        with self._lock:
            self._reset_after_fork_locked()
            if self._process is not None and self._process.is_alive():
                return True
            self._close_locked()
            try:
                # AppWorld runs on Linux; ``fork`` starts before its SafetyGuard and avoids both the
                # sandbox's subprocess block and the high per-rollout cost of a fresh interpreter.
                # The child immediately clears its environment and exposes only restricted exec globals.
                method = "fork" if os.name == "posix" else "spawn"
                context = multiprocessing.get_context(method)
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker, args=(child, parent), daemon=True)
                process.start()
                child.close()
            except Exception:
                return False
            self._conn = parent
            self._process = process
            return True

    def run(
        self,
        code: str,
        data: Any,
        *,
        timeout_s: float = 1.0,
        max_input_chars: int = 200_000,
        max_output_chars: int = 20_000,
        max_code_chars: int = 12_000,
        max_nodes: int = 2_000,
        max_steps: int = 100_000,
        allow_bare_container_output: bool = False,
    ) -> PythonToolResult:
        if len(code) > max_code_chars:
            return PythonToolResult(error=f"program exceeds {max_code_chars} characters")
        try:
            serialized = _bounded_json(data, max_input_chars)
            copied = json.loads(serialized)
        except Exception as exc:
            return PythonToolResult(error=f"invalid Python input: {exc}"[:2_000])

        with self._lock:
            if not self.prepare():
                return PythonToolResult(error="Python worker could not start", fatal=True)
            self._request_id += 1
            request_id = self._request_id
            limits = {
                "max_output_chars": max_output_chars,
                "max_nodes": max_nodes,
                "max_steps": max_steps,
                "timeout_s": timeout_s,
                "allow_bare_container_output": allow_bare_container_output,
            }
            try:
                self._conn.send((request_id, code, copied, limits))
                if not self._conn.poll(max(0.1, timeout_s) + 0.25):
                    self._close_locked()
                    return PythonToolResult(
                        error=f"program exceeded {timeout_s:.2f}s", fatal=True)
                response_id, result = self._conn.recv()
            except Exception as exc:
                self._close_locked()
                return PythonToolResult(
                    error=f"Python worker failed: {exc}"[:2_000], fatal=True)
            if response_id != request_id or not isinstance(result, PythonToolResult):
                self._close_locked()
                return PythonToolResult(
                    error="Python worker returned an invalid response", fatal=True)
            return result

    def close(self) -> None:
        with self._lock:
            self._reset_after_fork_locked()
            self._close_locked()

    def _reset_after_fork_locked(self) -> None:
        """Discard inherited pipe/process handles when a benchmark worker was forked."""
        pid = os.getpid()
        if pid == self._owner_pid:
            return
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._owner_pid = pid
        self._process = None
        self._conn = None
        self._request_id = 0

    def _close_locked(self) -> None:
        process, conn = self._process, self._conn
        self._process = None
        self._conn = None
        if conn is not None:
            try:
                conn.send(None)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        if process is not None:
            try:
                process.join(timeout=0.2)
            except Exception:
                pass
            if process.is_alive():
                try:
                    process.terminate()
                    process.join(timeout=0.5)
                except Exception:
                    pass


_SHARED_TOOL = PurePythonTool()
atexit.register(_SHARED_TOOL.close)


def shared_python_tool() -> PurePythonTool:
    return _SHARED_TOOL


__all__ = ["PurePythonTool", "PythonToolResult", "shared_python_tool"]
