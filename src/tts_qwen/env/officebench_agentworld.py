"""Qwen-AgentWorld transition model for OfficeBench.

This module is deliberately separate from :mod:`officebench_agent`: the benchmark
agent and its prompts stay unchanged, while this adapter supplies the predictor
interface that the existing agent loop already uses for world-model rollouts.
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, Iterable, Optional

from ..llm.agentworld import AgentWorldSession, build_agentworld_system_prompt


_CONFIDENCE_RE = re.compile(
    r"(?:confidence|probability|score)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

# Behavioural details that the official catalogue DEMOs omit but the in-container
# OfficeBench implementations enforce.  Supplying them prevents AgentWorld from
# hallucinating a successful transition that the committed action will reject.
_ACTION_NOTES = {
    "word.write_to_file": (
        "The target document MUST already exist. If it does not exist, return a failed observation "
        "and do not create it; create_new_file must be called first."
    ),
    "word.convert_to_pdf": "The source Word document must exist before conversion.",
    "excel.set_cell": "The target spreadsheet must exist before a cell can be set.",
    "excel.convert_to_pdf": "The source spreadsheet must exist before conversion.",
    "pdf.convert_to_image": "The source PDF must exist before conversion.",
    "pdf.convert_to_word": "The source PDF must exist before conversion.",
    "email.send_email": (
        "sender and recipient are account names (an @domain suffix is stripped). A .eml suffix is "
        "not stripped and would name a different account. The email is stored under the recipient "
        "and sender account directories."
    ),
}


def officebench_agentworld_tool_docs(catalogue: Dict) -> Dict[str, str]:
    """Translate the official OfficeBench catalogue to MCP-style tool docs."""
    docs: Dict[str, str] = {}
    intros = catalogue.get("intro", {}) or {}
    demos = catalogue.get("demo", {}) or {}
    for app in catalogue.get("available_apps", []) or []:
        for action in (catalogue.get("actions", {}) or {}).get(app, []) or []:
            name = f"{app}.{action}"
            docs[name] = (
                f"{intros.get(app, app)}\n"
                f"OfficeBench action contract: {(demos.get(app, {}) or {}).get(action, action)}\n"
                + (f"Exact environment semantics: {_ACTION_NOTES[name]}\n" if name in _ACTION_NOTES else "")
                + "Predict the observation and update the simulated office/filesystem state. "
                "Paths are rooted in /testbed unless the action says otherwise."
            )
    docs["system.finish_task"] = (
        "Record that the OfficeBench agent finished. Arguments: answer (string). "
        "Update /testbed/data/answer.txt in simulated state using that answer."
    )
    docs["__agentworld__.verify_task"] = (
        "Judge whether the task is fully satisfied in the CURRENT SIMULATED OfficeBench state. "
        "Use the entire preceding action/observation history and inspect all requested deliverables, "
        "calendar events, emails, and answer content. Return exactly one JSON object with keys "
        '"success" (boolean), "confidence" (number from 0 to 1), and "reason" (string).'
    )
    return docs


def _doc_type(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    return {
        ".xlsx": "xlsx", ".xls": "xlsx", ".doc": "doc", ".docx": "docx",
        ".pdf": "pdf", ".png": "image", ".jpg": "image", ".jpeg": "image",
        ".bmp": "image", ".tif": "image", ".tiff": "image", ".ics": "ics",
    }.get(ext, "txt")


def build_officebench_initial_state(
    session,
    *,
    max_files: int = 30,
    max_chars_per_file: int = 4000,
    max_total_chars: int = 40000,
) -> Dict[str, Any]:
    """Read a bounded pristine task snapshot, excluding OfficeBench gold references."""
    listing = session.list_state("/testbed", recursive=True)
    entries = list(listing.get("entries", []) or [])
    all_files = [
        str(entry.get("path"))
        for entry in entries
        if entry.get("type") == "file" and str(entry.get("path", "")).startswith("/testbed/")
    ]
    files = all_files[:max_files]
    specs = [{"path": path, "doc_type": _doc_type(path)} for path in files]
    try:
        contents = session.read_state(specs) if specs else {}
    except Exception:  # a corrupt source artifact should not abort the whole rollout
        contents = {}

    compact: Dict[str, str] = {}
    used = 0
    for path in files:
        value = str(contents.get(path, "[content unavailable]"))[:max_chars_per_file]
        if used + len(value) > max_total_chars:
            break
        compact[path] = value
        used += len(value)

    # read_state also supplies calendars and email accounts, which may not appear as ordinary files.
    for path, raw in contents.items():
        if path in compact:
            continue
        value = str(raw)[:max_chars_per_file]
        if used + len(value) > max_total_chars:
            break
        compact[str(path)] = value
        used += len(value)

    return {
        "filesystem": entries[:200],
        "contents": compact,
        "snapshot_complete": len(all_files) <= max_files and len(entries) <= 200,
    }


def build_officebench_agentworld_prompt(
    catalogue: Dict,
    config: Dict,
    initial_state: Dict[str, Any],
    *,
    template_file: str,
) -> str:
    prompt = build_agentworld_system_prompt(
        officebench_agentworld_tool_docs(catalogue), template_file=template_file)
    context = {
        "task": config.get("task", ""),
        "user": config.get("username", ""),
        "date": config.get("date", ""),
        "weekday": config.get("weekday", ""),
        "time": config.get("time", ""),
        "initial_state": initial_state,
    }
    return (
        prompt
        + "\n\n# OfficeBench Task and Initial Environment State\n"
        + "This is a bounded snapshot of the pristine task environment, not a gold/reference output. "
          "Maintain changes cumulatively across turns. An unlisted path may still exist.\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
    )


def _parse_object(text: str) -> Dict[str, Any]:
    candidate = str(text or "").strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
    starts = [pos for pos in (candidate.find("{"), candidate.find("[")) if pos >= 0]
    if starts:
        candidate = candidate[min(starts):]
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {}


def parse_agentworld_confidence(raw: str) -> tuple[float, Dict[str, Any]]:
    result = _parse_object(raw)
    value: Any = next(
        (result[key] for key in ("confidence", "probability", "score") if key in result), None)
    if value is None and isinstance(result.get("success"), bool):
        value = 1.0 if result["success"] else 0.0
    if value is None:
        match = _CONFIDENCE_RE.search(str(raw or ""))
        value = match.group(1) if match else 0.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence > 1.0 and confidence <= 100.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence)), result


class OfficeBenchAgentWorldPredictor:
    """Minimal predictor protocol consumed by ``run_officebench_agent``."""

    uses_fixed_context = False

    def __init__(self, session: AgentWorldSession, *, valid_tools: Optional[Iterable[str]] = None) -> None:
        self.session = session
        self.valid_tools = set(valid_tools or [])
        self.program_only = True
        self.pre_write_content: Dict[str, str] = {}
        self.n_transition_calls = 0
        self.n_failures = 0
        self._written = set()

    def bind_query_interface(self, query) -> None:
        del query

    @staticmethod
    def _call(action: Dict) -> tuple[str, Dict[str, Any]]:
        app = str(action.get("app", "unknown"))
        act = str(action.get("action", "unknown"))
        args = {key: value for key, value in action.items() if key not in ("app", "action")}
        return f"{app}.{act}", args

    def _predict(self, action: Dict) -> str:
        name, args = self._call(action)
        self.n_transition_calls += 1
        if self.valid_tools and name not in self.valid_tools:
            observation = (
                f"OBSERVATION: Unknown OfficeBench action {name}. Use one of the exact documented "
                "actions for the current app."
            )
            self.session.observe_tool(name, args, observation)
            return observation
        try:
            return self.session.predict_tool(name, args)
        except Exception as exc:  # keep the agent loop alive and make failure visible
            self.n_failures += 1
            return f"ERROR: AgentWorld {type(exc).__name__}: {exc}"

    def predict_write(self, action: Dict, history: Iterable) -> str:
        del history
        for key in ("file_path", "word_file_path", "pdf_file_path", "image_file_path"):
            if action.get(key):
                self._written.add(os.path.normpath(str(action[key])))
        return self._predict(action)

    def predict_read(self, action: Dict, pre_content=None, history=None, dirty=False) -> str:
        del pre_content, history, dirty
        return self._predict(action)

    def was_written(self, target: Optional[str]) -> bool:
        return bool(target and os.path.normpath(str(target)) in self._written)

    def fixed_pre_write_content(self, target: str):
        return self.pre_write_content.get(target)

    def note_read(self, target: str, observation: str) -> None:
        del target, observation

    def record_answer(self, answer: str) -> None:
        self.n_transition_calls += 1
        try:
            self.session.predict_tool("system.finish_task", {"answer": answer})
        except Exception:
            self.n_failures += 1

    def verify_task(self, instruction: str, final_answer: Optional[str]) -> tuple[float, Dict, str]:
        raw = self.session.predict_tool(
            "__agentworld__.verify_task",
            {"instruction": instruction, "final_answer": final_answer},
            max_tokens=min(self.session.max_tokens, 16384),
        )
        confidence, parsed = parse_agentworld_confidence(raw)
        return confidence, parsed, raw


__all__ = [
    "OfficeBenchAgentWorldPredictor",
    "build_officebench_agentworld_prompt",
    "build_officebench_initial_state",
    "officebench_agentworld_tool_docs",
    "parse_agentworld_confidence",
]
