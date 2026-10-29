"""In-container driver for the REAL OfficeBench benchmark (runs INSIDE the docker container).

This script is **never imported on the host** — it is copied into the OfficeBench docker container
(which ships the official app dependencies and has the official ``apps`` package at ``/apps`` and the
official ``utils`` package at ``/utils``) and executed there via ``docker exec ... python3
/officebench_driver.py <cmd>``.  Keeping every OfficeBench-specific import inside the container means
the host needs no OfficeBench Python deps — only Docker — exactly the way the official runner places
all app code in the image.

It faithfully reproduces the official OfficeBench action execution
(:meth:`utils.env.OfficeAgentEnv.exec_action`) and the official evaluation
(:func:`evaluation.evaluate_output` / ``utils.evaluate.*``), so a rollout driven through it scores
identically to the official benchmark.

Sub-commands (each reads a JSON request on stdin and prints exactly one
``OFFICEBENCH_RESULT=<json>`` line on stdout):

* ``apps``         → the app catalogue ({app: INTRO}, {app: {action: DEMO}}, action lists) used to
  build the official agent prompt on the host.
* ``exec``         → run one action ``{"action": {...}, "workdir": "/"}`` and return its observation.
* ``write_answer`` → write a ``finish_task`` answer to ``/testbed/data/answer.txt`` (official behaviour).
* ``evaluate``     → run a task's ground-truth ``evaluation`` config against ``/testbed`` and return
  per-check pass/fail (the official, execution-based score).
* ``list_state``   → list immutable testbed objects without dispatching an OfficeBench action.

Only the Python standard library is imported at module top; the OfficeBench ``apps`` / ``utils``
packages are imported lazily inside the handlers, so importing this file on the host (where those
packages are absent) never fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_TESTBED = "/testbed"
_RESULT_PREFIX = "OFFICEBENCH_RESULT="


def _emit(payload: dict) -> None:
    sys.stdout.write(_RESULT_PREFIX + json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def _ensure_path() -> None:
    # ``/`` on sys.path makes the container's top-level ``apps`` and ``utils`` packages importable.
    if "/" not in sys.path:
        sys.path.insert(0, "/")


def _simplify_path(current: str, changed: str) -> str:
    """Resolve a ``cd`` target against the current workdir (port of OfficeAgentEnv.simplify_path)."""
    if not changed:
        return current
    if changed[0] == "/":
        current = ""
    path = []
    for segment in (current + "/" + changed).split("/"):
        if segment == "..":
            if path:
                path.pop()
        elif segment and segment != ".":
            path.append(segment)
    return "/" + "/".join(path)


def _minor_action_fix(action: dict) -> dict:
    out = {}
    for k, v in action.items():
        out[k] = v[0] if isinstance(v, list) and len(v) == 1 else v
    return out


def _list_files(root: str) -> set:
    """The set of regular files currently under ``root`` (for read-only incidental-write detection)."""
    import os

    out = set()
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            out.add(os.path.join(dirpath, name))
    return out


def _list_dirs(root: str) -> set:
    """The set of directories currently under ``root`` (excluding ``root`` itself)."""
    import os

    return {
        os.path.join(dirpath, name)
        for dirpath, dirs, _files in os.walk(root)
        for name in dirs
    }


def _remove_new_files(root: str, before: set, before_dirs: set | None = None) -> list:
    """Delete files/directories created since the snapshots; return what was removed.

    Some official OfficeBench *read* helpers create-if-missing (``calendar.list_events`` writes a blank
    ``<user>.ics``; ``email.list_emails`` makedirs an account dir), so an exact read on the world-model
    shadow could otherwise leave a file behind.  Reverting those incidental creates keeps the read-only
    shadow ``/testbed`` byte-for-byte pristine — the "no write, even shadow" invariant.
    """
    import os

    removed = []
    for path in _list_files(root) - before:
        try:
            os.remove(path)
            removed.append(path)
        except OSError:
            pass
    if before_dirs is not None:
        for path in sorted(_list_dirs(root) - before_dirs, key=len, reverse=True):
            try:
                os.rmdir(path)
                removed.append(path)
            except OSError:
                pass
    return removed


def handle_apps() -> dict:
    """Return the official app catalogue (INTRO + per-action DEMO) for prompt construction."""
    _ensure_path()
    import apps  # noqa: WPS433 (in-container only)

    intros = {name: getattr(mod, "INTRO", name) for name, mod in apps.AVAILABLE_APPS.items()}
    demos: dict = {}
    actions: dict = {}
    for app, action_map in apps.AVAILABLE_ACTIONS.items():
        demos[app] = {a: getattr(mod, "DEMO", a) for a, mod in action_map.items()}
        actions[app] = list(action_map.keys())
    return {"available_apps": list(apps.AVAILABLE_APPS.keys()), "intro": intros,
            "demo": demos, "actions": actions}


def handle_exec(req: dict) -> dict:
    """Execute ONE app/shell action inside the container (port of OfficeAgentEnv.exec_action).

    Only the ``app``/``shell`` branch lives here — the ``system`` actions (switch_app / finish_task /
    got_stuck) are handled on the host, where the agent's app/workdir state is tracked.
    """
    _ensure_path()
    import apps  # noqa: WPS433

    action = _minor_action_fix(req["action"])
    workdir = req.get("workdir") or "/"
    # On the world-model shadow the session is read-only: capture the file tree so any file an exact
    # read incidentally creates (create-if-missing read helpers) can be reverted, keeping it pristine.
    read_only = bool(req.get("read_only"))
    pre_files = _list_files(_TESTBED) if read_only else None
    pre_dirs = _list_dirs(_TESTBED) if read_only else None
    is_cd = False
    new_workdir = None

    if action.get("app") == "shell":
        command = action.get("command", "")
        if isinstance(command, list):
            command = " ".join(command)
        is_cd = command.strip().startswith("cd")
        if is_cd:
            cd_arg = command[command.index("cd ") + 3:].strip()
            new_workdir = _simplify_path(workdir, cd_arg)
            command = f"cd {new_workdir}"
    else:
        action_module = apps.AVAILABLE_ACTIONS[action["app"]][action["action"]]
        command = action_module.construct_action(workdir, args=action)

    proc = subprocess.run(["/bin/bash", "-c", command], cwd=workdir,
                          capture_output=True, text=True)
    exit_code = proc.returncode
    raw = (proc.stdout or "") + (proc.stderr or "")
    observation = raw.split("OBSERVATION:")[-1].strip()
    if observation == "" and action.get("app") == "shell":
        observation = f"Successfully executed command: {command}. The output was [{raw}]."

    if is_cd and exit_code != 0:
        new_workdir = None  # the cd failed — keep the old workdir
    if read_only and pre_files is not None and pre_dirs is not None:
        _remove_new_files(
            _TESTBED, pre_files, pre_dirs)  # revert any incidental read-helper writes
    return {"observation": observation, "exit_code": exit_code,
            "is_cd": is_cd, "new_workdir": new_workdir, "command": command}


def handle_write_answer(req: dict) -> dict:
    """Write a ``finish_task`` answer to ``/testbed/data/answer.txt`` (official behaviour)."""
    import os

    answer = str(req.get("answer", "None")).replace('"', "").replace("'", "")
    os.makedirs(f"{_TESTBED}/data", exist_ok=True)
    with open(f"{_TESTBED}/data/answer.txt", "w") as fh:
        fh.write(answer + "\n")
    return {"observation": "Task finished", "written": answer}


def _read_one_file(path: str, doc_type: str) -> str:
    """Read one result file with the official in-container helper for its ``doc_type`` (read-only)."""
    import os

    if not os.path.exists(path):
        return "[file does not exist]"
    if os.path.isdir(path):
        return "ERROR: operand is a directory; use LIST instead of READ"
    try:
        if doc_type == "image":
            from apps.ocr_app import ocr_recognize_file
            text = ocr_recognize_file.ocr_recognize_file(path)
            return text if text is not None else "[OCR failed]"
        if doc_type == "xlsx":
            from apps.excel_app import excel_read_file
            return excel_read_file.excel_read_file(path)
        if doc_type in ("doc", "docx"):
            from apps.word_app import word_read_file
            return word_read_file.word_read_file_into_string(path)
        if doc_type == "pdf":
            from apps.pdf_app import pdf_read_file
            return pdf_read_file.read_pdf_to_string(path)
        # ics / txt / anything else: raw text
        with open(path, errors="replace") as fh:
            return fh.read()
    except Exception as exc:  # noqa: BLE001
        return f"[unreadable: {type(exc).__name__}: {exc}]"


def handle_read_state(req: dict) -> dict:
    """Return the ACTUAL post-trajectory ``/testbed`` state (read-only), for the grounded verifier.

    Reads the concrete result files the ground-truth checks reference — resolved on the host from the
    eval config and passed as ``files`` ``[{path, doc_type}, ...]`` — plus the agent's ``answer.txt``,
    any user calendars, and the email accounts, using the official ``apps`` read helpers.  This lets
    the world-model verifier judge from the real resulting state instead of the agent's self-report,
    while never executing the candidate or running the official boolean checks (that is the oracle).
    """
    _ensure_path()
    import glob
    import os

    out: Dict[str, str] = {}
    for spec in req.get("files", []) or []:
        path = spec.get("path", "")
        if not path:
            continue
        abspath = path if path.startswith("/") else os.path.join(_TESTBED, path.lstrip("./"))
        out[path] = _read_one_file(abspath, spec.get("doc_type", "txt"))

    # ``extras`` (default on) also dumps the agent's answer + calendars + emails, for state grounding.
    # Reference-file reads pass ``extras=False`` so only the requested gold files are returned.
    if not req.get("extras", True):
        return {"state": out}

    # The agent's written answer (give-ups / wrong answers show up here).
    ans = f"{_TESTBED}/data/answer.txt"
    if os.path.exists(ans):
        out["data/answer.txt"] = _read_one_file(ans, "txt")

    # Calendars + emails referenced by username-style checks.
    for ics in glob.glob(f"{_TESTBED}/calendar/*.ics"):
        out[os.path.relpath(ics, _TESTBED)] = _read_one_file(ics, "ics")
    try:
        from apps.email_app import email_list_emails
        for acct in glob.glob(f"{_TESTBED}/emails/*"):
            if os.path.isdir(acct):
                user = os.path.basename(acct)
                out[f"emails/{user}"] = str(email_list_emails.list_emails(user, _TESTBED, -1))
    except Exception:  # noqa: BLE001
        pass
    return {"state": out}


def handle_list_state(req: dict) -> dict:
    """List one testbed directory without invoking the shell app or any OfficeBench action."""
    path = os.path.normpath(str(req.get("path") or _TESTBED))
    recursive = bool(req.get("recursive"))
    if path != _TESTBED and not path.startswith(_TESTBED + os.sep):
        return {"error": "path must be under /testbed", "entries": []}
    if not os.path.exists(path):
        return {"entries": [], "missing": True}
    if not os.path.isdir(path):
        return {"entries": [{"name": os.path.basename(path), "path": path, "type": "file"}],
                "missing": False, "operand_type": "file"}
    entries = []
    if recursive:
        # Keep the existing sorted inventory API, but retain native DFS/readdir
        # order so a predicted `find` need not hallucinate alphabetical output.
        native_order = {}
        def visit(directory):
            with os.scandir(directory) as children:
                for child in children:
                    native_order[child.path] = len(native_order)
                    if child.is_dir(follow_symlinks=False):
                        visit(child.path)
        visit(path)
        for root, dirs, files in os.walk(path):
            dirs.sort()
            files.sort()
            for name in dirs:
                child = os.path.join(root, name)
                entries.append({"name": name, "path": child, "type": "directory", "native_order": native_order[child]})
            for name in files:
                child = os.path.join(root, name)
                entries.append({"name": name, "path": child, "type": "file", "native_order": native_order[child]})
    else:
        names = os.listdir(path)
        native_order = {name: index for index, name in enumerate(names)}
        for name in sorted(names):
            child = os.path.join(path, name)
            entries.append({
                "name": name,
                "path": child,
                "type": "directory" if os.path.isdir(child) else "file",
                "native_order": native_order[name],
            })
    return {"entries": entries, "missing": False}


def handle_evaluate(req: dict) -> dict:
    """Run a task's ground-truth ``evaluation`` config against ``/testbed`` (the official score).

    Mirrors :func:`evaluation.evaluate_output`: each item ``{"function","args"}`` is dispatched to the
    official ``utils.evaluate`` checker against the testbed dir; the task passes iff every check passes.
    """
    _ensure_path()
    from utils.evaluate import (  # noqa: WPS433
        evaluate_calendar_no_overlap,
        evaluate_contain,
        evaluate_diff_contain_text,
        evaluate_excel_cell_comparator,
        evaluate_excel_cell_value,
        evaluate_exact_match,
        evaluate_file_exist,
        evaluate_file_not_exist,
        evaluate_not_contain,
    )

    fns = {
        "evaluate_contain": evaluate_contain,
        "evaluate_not_contain": evaluate_not_contain,
        "evaluate_file_exist": evaluate_file_exist,
        "evaluate_file_not_exist": evaluate_file_not_exist,
        "evaluate_diff_contain_text": evaluate_diff_contain_text,
        "evaluate_excel_cell_value": evaluate_excel_cell_value,
        "evaluate_excel_cell_comparator": evaluate_excel_cell_comparator,
        "evaluate_exact_match": evaluate_exact_match,
        "evaluate_calendar_no_overlap": evaluate_calendar_no_overlap,
    }
    testbed_dir = req.get("testbed_dir", _TESTBED)
    checks = []
    for item in req.get("evaluation", []):
        fn_name = item.get("function")
        args = item.get("args", {})
        fn = fns.get(fn_name)
        if fn is None:
            checks.append({"function": fn_name, "pass": False, "error": "unknown function"})
            continue
        try:
            ok = bool(fn(testbed_dir, args))
            checks.append({"function": fn_name, "pass": ok})
        except Exception as exc:  # noqa: BLE001 - faithful to official (a crashing check fails)
            checks.append({"function": fn_name, "pass": False, "error": str(exc)})
    passed = bool(checks) and all(c["pass"] for c in checks)
    return {"passed": passed, "num_checks": len(checks),
            "num_passed": sum(1 for c in checks if c["pass"]), "checks": checks}


def handle_capabilities(req):
    # Presence metadata only: never read or return credential contents.
    import os
    return {'llm_key_file_present': os.path.isfile('/openai_key.txt')}


_HANDLERS = {
    "apps": lambda req: handle_apps(),
    "capabilities": handle_capabilities,
    "exec": handle_exec,
    "write_answer": handle_write_answer,
    "evaluate": handle_evaluate,
    "read_state": handle_read_state,
    "list_state": handle_list_state,
}


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    handler = _HANDLERS.get(cmd)
    if handler is None:
        _emit({"error": f"unknown command {cmd!r}", "commands": list(_HANDLERS)})
        return 2
    raw = sys.stdin.read()
    req = json.loads(raw) if raw.strip() else {}
    try:
        _emit(handler(req))
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
