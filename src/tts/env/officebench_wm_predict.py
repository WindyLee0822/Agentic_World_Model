"""Predictive, write-safe world model for OfficeBench (coherent with the τ²/AppWorld contract).

The read-grounded TTS premise is that during sampling the agent interacts with a **world model**, not
the real environment: **read** actions are answered exactly (read-only) against the real starting
state, **write** actions are **LLM-predicted and NEVER executed**, and a **read after a write** returns
the *predicted* post-write state (the τ² "dirty read").  The real environment is therefore never
mutated while the ``N`` candidates are drawn; only the single selected trajectory is committed (really
executed) at the chokepoint.

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

import os
import re
from typing import Dict, List, Optional, Tuple

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
    "ls", "cat", "head", "tail", "find", "grep", "egrep", "fgrep", "wc", "pwd", "stat", "file",
    "du", "df", "tree", "diff", "cmp", "less", "more", "sort", "uniq", "cut", "awk", "sed",
    "basename", "dirname", "realpath", "readlink", "echo", "printf", "date", "which", "type",
}


def _shell_is_read(command: str) -> bool:
    """A shell command is a READ iff it has no redirection/pipe-to-writer and every stage is read-only.

    Conservative: anything with output redirection (``>``/``>>``), an in-place edit (``sed -i``), or a
    non-whitelisted command is treated as a WRITE (predicted, never executed) — so misclassification can
    only ever make us *predict* (safe), never *execute a mutation* (unsafe).
    """
    if not command:
        return False
    if ">" in command or ">>" in command:
        return False
    if re.search(r"\bsed\b[^|]*\s-\S*i", command):  # sed -i / -Ei ... (in-place edit)
        return False
    for stage in command.split("|"):
        toks = stage.strip().split()
        if not toks:
            continue
        cmd = os.path.basename(toks[0])
        if cmd not in _READ_SHELL:
            return False
    return True


def is_write_action(action: Dict) -> bool:
    """True if executing ``action`` would MUTATE the environment (so the WM must predict, not run it)."""
    app = action.get("app")
    act = action.get("action")
    if app == "system":
        return False  # switch_app / finish_task / got_stuck have no /testbed effect during sampling
    if app == "shell":
        cmd = action.get("command", "")
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        return not _shell_is_read(cmd)
    return f"{app}.{act}" not in _READ_ACTIONS


def _norm_path(path: str) -> str:
    """Canonicalise a testbed path for write/read target matching (strip /testbed, ./ and leading /)."""
    if not path:
        return ""
    p = str(path).strip()
    for prefix in ("/testbed/", "testbed/", "./"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    return p.lstrip("/")


def _derive_output(path: str, ext: str) -> str:
    base = _norm_path(path)
    stem = base.rsplit(".", 1)[0] if "." in base else base
    return f"{stem}.{ext}"


def write_targets(action: Dict) -> List[str]:
    """The canonical file(s) a WRITE action changes (so a later read of one is a dirty read)."""
    app, act = action.get("app"), action.get("action")
    a = action
    if app in ("excel", "word"):
        if act == "convert_to_pdf":
            return [_derive_output(a.get("file_path", ""), "pdf")]
        if a.get("file_path"):
            return [_norm_path(a["file_path"])]
    if app == "pdf":
        if act == "convert_to_image":
            return [_derive_output(a.get("file_path", ""), "png")]
        if act == "convert_to_word":
            return [_derive_output(a.get("file_path", ""), "docx")]
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
        m = re.search(r"(?:>|>>)\s*(\S+)", cmd) or re.search(r"/testbed/\S+", cmd)
        if m:
            return [_norm_path(m.group(1) if m.lastindex else m.group(0))]
        return ["<shell>"]
    # any other write: fall back to a file_path arg if present
    return [_norm_path(a["file_path"])] if a.get("file_path") else ["<unknown>"]


def read_target(action: Dict) -> Optional[str]:
    """The canonical file a READ action reads (to detect a read-after-write), or ``None``."""
    app, act = action.get("app"), action.get("action")
    a = action
    if app in ("excel", "word", "pdf", "ocr") and a.get("file_path"):
        return _norm_path(a["file_path"])
    if app == "calendar" and act == "list_events":
        return f"calendar/{a.get('username') or a.get('user') or 'user'}.ics"
    if app == "email" and act in ("list_emails", "read_email"):
        return f"emails/{a.get('username') or a.get('user') or 'user'}"
    if app == "shell":
        cmd = a.get("command", "")
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        m = re.search(r"/testbed/\S+", cmd)
        return _norm_path(m.group(0)) if m else None
    return None


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


def _apply_set_cells(initial: str, writes: List[Tuple[Dict, str]]) -> Optional[str]:
    """Exactly apply a sequence of ``excel.set_cell`` writes to the rendered sheet, or ``None``.

    Returns ``None`` (→ LLM fallback) if the starting content is not a parseable grid or a write is not
    fully specified (missing/invalid row/col), so we never fabricate a sheet from nothing.
    """
    grid = _parse_excel_grid(initial)
    if not grid:  # cannot ground an edit against a sheet we could not parse
        return None
    for action, _obs in writes:
        try:
            row = int(action["row_idx"])
            col = int(action["column_idx"])
        except (KeyError, TypeError, ValueError):
            return None
        val = action.get("text", action.get("value", ""))
        grid[(row, col)] = str(val)
    return _render_excel_grid(grid)


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
            parts = []  # a (re)create resets the file
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
    if all(a.get("app") == "excel" and a.get("action") == "set_cell" for a, _ in writes):
        return _apply_set_cells(initial, writes)
    if all(a.get("app") == "word" and a.get("action") in ("create_new_file", "write_to_file")
           for a, _ in writes):
        return _apply_word_writes(initial, writes)
    return None


#: Substrings the official OfficeBench apps print when an action FAILS (so a predicted write observation
#: carrying one of these is a failed write: it must not change the predicted file state).
_FAILURE_MARKERS = (
    "does not exist", "failed to", "no such file", "not found", "cannot ", "could not",
    "is not a", "invalid", "error:", "traceback", "no module named", "permission denied",
    "malformed", "unable to",
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
    """Predicts OfficeBench write observations, dirty reads, and predicted final file state (no exec)."""

    def __init__(self, chat_client) -> None:
        self._chat = chat_client
        #: target (normalised path) -> list of (action, predicted_observation) applied to it, in order.
        self.applied_writes: Dict[str, List[Tuple[Dict, str]]] = {}
        #: exact pre-write content read once per target (for dirty-read grounding).
        self.pre_write_content: Dict[str, str] = {}
        self.write_history: List[Tuple[Dict, str]] = []
        #: set once a write with an UNRESOLVED target (e.g. an unparseable shell write) occurs — from
        #: then on EVERY read is treated as dirty (predicted), because we cannot prove it is untouched.
        #: Mirrors τ²'s ``global_writes`` fallback for ref-less (device/global) writes.
        self.global_dirty = False
        self.n_writes = 0
        self.n_failed_writes = 0
        self.n_reads = 0
        self.n_dirty_reads = 0
        self.n_llm_calls = 0

    # -- bookkeeping ---------------------------------------------------- #
    def was_written(self, target: Optional[str]) -> bool:
        # A specific written target OR the global-dirty fallback (an unresolved write happened) makes a
        # read dirty -> predicted, never a stale exact read.  A ``None`` target (e.g. llm.complete_text)
        # has no file, so it is never dirty.
        if target is None:
            return False
        return self.global_dirty or target in self.applied_writes

    def note_read(self, target: Optional[str], content: str) -> None:
        """Cache the EXACT pre-write content of a target the first time it's read (dirty-read anchor)."""
        self.n_reads += 1
        if target and target not in self.pre_write_content:
            self.pre_write_content[target] = content

    def _writes_text(self, target: str) -> str:
        items = self.applied_writes.get(target, [])
        return "\n".join(f"  - {_action_str(a)} -> [{o[:200]}]" for a, o in items) or "  (none)"

    def file_write_log(self, path: str) -> str:
        """Human-readable log of the WRITE actions applied to ``path`` (for reference-grounded verify)."""
        return self._writes_text(_norm_path(path))

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
        if _wm_check_writes():
            prompt = (
                f"You are a faithful world model of the OfficeBench '{app}' app. Predict the observation "
                f"the real app prints for the WRITE action '{app}.{act}' with arguments "
                f"{_action_str(action)}.\n\n"
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
                f"'{app}.{act}' with arguments {_action_str(action)}. Assume it succeeds on a normal "
                f"filesystem. Return ONLY the single-line observation string the real app prints, matching "
                f"its wording (office apps print e.g. 'OBSERVATION: Successfully create a new event to "
                f"Bob's calendar.'). No prose, no explanation."
            )
        obs = self._complete(prompt, max_tokens=120) or f"OBSERVATION: Successfully executed {app}.{act}."
        obs = obs.strip().splitlines()[0] if obs.strip() else obs
        # A predicted FAILURE leaves the environment unchanged: record it in the write-history (so the
        # verifier can see the attempted-and-failed write) but do NOT mark any target as written, and do
        # NOT trip global-dirty — a write that did not take effect cannot make a later read dirty.
        if _wm_check_writes() and _looks_like_failure(obs):
            self.n_failed_writes += 1
            self.write_history.append((action, obs))
            return obs
        targets = write_targets(action)
        for tgt in targets:
            self.applied_writes.setdefault(tgt, []).append((action, obs))
            if tgt in ("<shell>", "<unknown>"):
                # an unresolved write target: we can no longer certify any file is untouched, so from
                # here every read is predicted (τ²'s global-dirty fallback for ref-less writes).
                self.global_dirty = True
        self.write_history.append((action, obs))
        return obs

    def predict_read(self, action: Dict, pre_write_content: str, history: List[Tuple] = ()) -> str:
        """A read AFTER a write to the same file: predict the post-write content it returns now."""
        self.n_dirty_reads += 1
        target = read_target(action) or ""
        app, act = action.get("app"), action.get("action")
        # Deterministic short-circuit: if every write to this file is exactly modelable, apply them with
        # a pure function (no LLM, no drift) instead of asking the LLM to re-derive the post-write file.
        target_writes = self.applied_writes.get(target)
        if target_writes:
            sym = symbolic_file_state(target, pre_write_content, target_writes)
            if sym is not None:
                return sym
        # target-specific writes, or — when the read is dirty only via the global-dirty fallback — the
        # full write history (so a globally-dirty read still sees what may have changed the file).
        writes_text = self._writes_text(target)
        if writes_text.strip() == "(none)" and self.global_dirty:
            writes_text = self._history_text(k=20)
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
        out = self._complete(prompt, max_tokens=_READ_MAX_TOKENS)
        return out if out else pre_write_content

    def predict_file_state(self, path: str, initial_content: str) -> str:
        """Render a gold-checked file's PREDICTED final content (initial + its writes) for the verifier.

        Only used when the file has NO gold reference to compare against (e.g. cell-value / keyword
        checks); reference-grounded files are judged from initial + write-log directly (no fabrication).
        The prompt is deliberately conservative — apply only write-justified changes, keep everything
        else verbatim — so the reconstruction is as trustworthy as possible without executing writes.
        """
        target = _norm_path(path)
        writes = self.applied_writes.get(target)
        if not writes:
            return initial_content  # untouched by the trajectory
        sym = symbolic_file_state(target, initial_content, writes)
        if sym is not None:
            return sym  # exact deterministic reconstruction (no LLM re-derivation, no drift)
        app_hint = writes[0][0].get("app", "office")
        prompt = (
            f"You are a world model of an OfficeBench '{app_hint}' file. Given the initial content of "
            f"file '{target}' and the WRITE actions the agent applied to it, predict the EXACT FINAL "
            f"content of the file.\n\n"
            f"INITIAL content:\n{initial_content[:_STATE_CONTENT_CHARS]}\n\n"
            f"WRITE actions applied (in order):\n{self._writes_text(target)}\n\n"
            f"Change ONLY the cells/lines a listed write clearly determines (e.g. a set_cell write sets "
            f"exactly that cell to that value); copy every other part of the file verbatim from the "
            f"initial content and never invent values. Return ONLY the final file content as it would be "
            f"read back. No prose."
        )
        out = self._complete(prompt, max_tokens=_STATE_MAX_TOKENS)
        return out if out else initial_content

    def predicted_write_log(self, cap: int = 40) -> str:
        """A compact rendering of every predicted write (for the verifier when no gold file is checked)."""
        lines = [f"  - {_action_str(a)} -> [{o[:200]}]" for a, o in self.write_history[:cap]]
        return "\n".join(lines) or "  (no writes)"

    def _complete(self, prompt: str, *, max_tokens: int) -> str:
        try:
            self.n_llm_calls += 1
            return (self._chat.chat("You are a faithful world model. Answer exactly, no prose.",
                                    prompt, temperature=0.0, max_tokens=max_tokens) or "").strip()
        except Exception:  # noqa: BLE001 - a prediction failure must never crash the rollout
            return ""


__all__ = [
    "OfficeBenchWMPredictor",
    "is_write_action",
    "write_targets",
    "read_target",
    "symbolic_file_state",
]
