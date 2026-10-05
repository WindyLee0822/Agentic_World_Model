"""The **official** OfficeBench agent loop (faithful re-implementation).

OfficeBench's reference agent (``utils/policies.py::LLMPolicy``) solves a task as an app
state-machine: a constant system message introduces every app, and at each step a single
``system`` + ``user`` prompt — the ``user`` half carrying the running step history — asks the model
for exactly **one** JSON action ``{'app': ..., 'action': ..., ...}``.  Until an app is chosen the only
move is ``switch_app``; once inside an app the prompt lists that app's action ``DEMO``s plus
``switch_app`` / ``finish_task``.  The loop ends on ``finish_task`` / ``got_stuck`` (or a 5-step
no-progress window, which forces ``got_stuck``) or the step budget.

This module reproduces that prompt and loop verbatim (``PROMPT_DICT`` /
``construct_system_message`` / ``build_prompt`` / ``proc_action`` are ports of the official ones),
with two transport-only differences:

* the app catalogue (each app's ``INTRO`` and each action's ``DEMO``) is fetched from the running
  container via :meth:`tts.env.officebench_lowlevel.OfficeBenchSession.apps_catalogue` rather than by
  importing OfficeBench's ``apps`` package on the host;
* the model is called through :class:`tts.llm.chat_client.ChatClient` (single ``system`` + ``user``
  turn, ``temperature``/``max_tokens`` matching the official defaults) instead of the OfficeBench
  ``ChatGPT`` wrapper.

The prompt preserves the official catalogue and state machine, with one validation reminder added:
the model must switch apps first and use only the exact action names/arguments shown for the current
app. The container still executes every committed action through the official ``apps`` modules.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _bool_env(name: str, default: bool = True) -> bool:
    """A boolean tuning knob (``0``/``false`` off, anything else on)."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val not in ("0", "false", "False")


def _int_env(name: str, default: int) -> int:
    """A non-negative-int tuning knob (falls back to ``default``)."""
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


#: Max chars any single argument value / rendered action is allowed to contribute to the running
#: step-history the model reads back.  A ``word.write_to_file`` can carry a whole document (OCR text,
#: source code) as a literal ``contents`` arg; splicing that verbatim into ``##History`` corrupts the
#: prompt the model conditions on and cascades into malformed follow-up actions (observed burning the
#: 3app step budget).  Truncating the rendering — never the executed action — keeps the log readable.
_HISTORY_ARG_CHARS = _int_env("TTS_OFFICEBENCH_HISTORY_ARG_CHARS", 200)


def _sanitize_actions() -> bool:
    """Whether to harden action parsing/rendering against file-content corruption (default ON).

    Turns on: (1) a fallback that recovers the FIRST balanced ``{...}`` action when the model emits two
    JSON objects in one turn (the strict official parse drops it as malformed, wasting the step and
    leaking the second object's file content into the trajectory); (2) newline-collapsing/truncating
    rendering of actions in the running history so a literal document argument cannot corrupt the
    prompt.  Set ``TTS_OFFICEBENCH_SANITIZE_ACTIONS=0`` for the strict official behaviour.
    """
    return _bool_env("TTS_OFFICEBENCH_SANITIZE_ACTIONS", True)


def _prune_loops() -> bool:
    """Whether an immediate-repeat READ is pruned so it doesn't consume the step budget (default ON).

    A degenerate rollout re-issues the same read (e.g. ``shell.list_dir /testbed``) and spins until the
    official 5-identical window forces ``got_stuck`` — burning steps that longer multi-app tasks need.
    When on, a read identical to the immediately preceding action returns the prior (unchanged) result
    with a nudge and does NOT count against the productive-step budget (the official give-up window and
    a hard iteration ceiling still bound the loop).  Set ``TTS_OFFICEBENCH_PRUNE_LOOPS=0`` to disable.
    """
    return _bool_env("TTS_OFFICEBENCH_PRUNE_LOOPS", True)


def _extract_first_object(text: str) -> Optional[str]:
    """Return the first brace-balanced ``{...}`` substring of ``text`` (quote/escape aware), or ``None``.

    Recovers a single valid action when the model emitted several (``proc_action`` keeps the whole
    ``{...}`` span, which then fails to parse); only the first object is used, mirroring "one action at
    a time".
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    quote = ""
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_str = False
        elif ch in ("'", '"'):
            in_str = True
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _render_action(action: object) -> str:
    """Render an action for the model-facing history: one line, long argument values truncated.

    Only affects the *displayed* history (and, via the same helper, the committed-action log) — the
    executed action dict is untouched — so a document-sized literal argument can no longer corrupt the
    prompt or the record.
    """
    if not _sanitize_actions():
        return str(action)
    cap = _HISTORY_ARG_CHARS
    if isinstance(action, dict):
        def _clip(v):
            if isinstance(v, dict):
                return {k: _clip(sv) for k, sv in v.items()}
            s = str(v)
            s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            return s[:cap] + "…" if len(s) > cap else s

        rendered = ", ".join(f"{k!r}: {_clip(v)!r}" if not isinstance(v, dict) else f"{k!r}: {_clip(v)}"
                             for k, v in action.items())
        return "{" + rendered + "}"
    s = str(action).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return s[:cap] + "…" if len(s) > cap else s


# Verbatim copy of utils/policies.py::PROMPT_DICT (the official agent prompts).
PROMPT_DICT = {
    "prompt_undecided_app": (
        "##Task: {task}\n"
        "##Available apps: {available_apps}\n"
        "##Instruction:\n"
        " - choose an app from the avaiblable apps: {{'app': 'system', 'action': 'switch_app', 'target_app': [THE_APP_YOU_CHOOSE]}}\n"
        "##Command:"
    ),
    "prompt_undecided_app_w_history": (
        "##Task: {task}\n"
        "##History:\n"
        "{history}"
        "##Available apps: {available_apps}\n"
        "##Instruction:\n"
        " - choose an app from the avaiblable apps: {{'app': 'system', 'action': 'switch_app', 'target_app': [THE_APP_YOU_CHOOSE]}}\n"
        "##Command:"
    ),
    "prompt_decided_app": (
        "##Task: {task}\n"
        "##History:\n"
        "{history}"
        "##Current apps: {current_app}\n"
        "##Instruction: Choose one action from the list as the next step.\n"
        "{detailed_instruction}"
        " - switch to another app among {available_apps}: {{'app': 'system', 'action': 'switch_app', 'target_app': [THE_APP_YOU_CHOOSE]}}\n"
        " - finish the task with your answer as None if the task is not a question: {{'app': 'system', 'action': 'finish_task', 'answer': 'None'}}\n"
        " - finish the task with your answer if the task is a question: {{'app': 'system', 'action': 'finish_task', 'answer': [ANSWER]}}\n"
        "##Command:"
    ),
}

_ACTION_WINDOW_SIZE = 5


@dataclass
class AgentRun:
    """The realized OfficeBench rollout (faithful to the official ``env.history``)."""

    actions: List[Dict] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    history: List[Tuple[object, str]] = field(default_factory=list)
    final_answer: Optional[str] = None
    done: bool = False
    n_steps: int = 0


def construct_system_message(config: Dict, catalogue: Dict) -> str:
    """Port of LLMPolicy.construct_system_message (the constant system prompt listing every app)."""
    intros = catalogue["intro"]
    app_introduction = ""
    for app in catalogue["available_apps"]:
        app_introduction += f" - {intros.get(app, app)}\n"
    username = config["username"]
    date = config["date"]
    weekday = config["weekday"]
    time = config["time"]
    return (
        f"Today is {date} ({weekday}). The current time is {time}. You are an AI assistant for user {username}.\n"
        "You can help solve the task step by step.\n"
        "You can interact with an operation system and use apps to solve the task.\n"
        "You must follow the instructions and use the given json format to call APIs.\n"
        "You can only generate one action at a time.\n"
        "Before using an app action, switch to that app. Use ONLY the exact action names and argument "
        "fields shown for the current app; never invent or rename an action.\n"
        "You can find files for your task in `/testbed/data`. For inspection, use only non-mutating "
        "shell commands such as ls, find, cat, grep, sed without -i, sort, head, tail, file, and wc. "
        "Do not use Python, xargs, sh, output redirection, or mutation commands merely to inspect files.\n"
        "For multi-app tasks, first identify the exact source file, then read or recognize its actual "
        "content with the appropriate office app before copying, summarizing, converting, or emailing "
        "it. Copy the discovered path exactly, including its extension; never substitute a guessed "
        "filename or change .jpg to .png. Never invent source content from a filename or task wording.\n"
        "Complete every requested deliverable and location before finish_task. In particular, an email "
        "request means the email body must contain the requested source information because send_email "
        "does not attach files. After any failed action, correct its exact arguments or path instead of "
        "finishing as if it succeeded. After a successful write, proceed to the next requested app or "
        "finish the task; do not spend the remaining budget repeatedly searching for the predicted output.\n"
        "Representative workflows:\n"
        " - Image to document/PDF: discover the exact image path, switch to ocr and recognize that exact "
        "path, then switch to word to create/write the extracted summary and convert it if requested.\n"
        " - Spreadsheet/document to PDF plus email: read the exact source first, convert that same source "
        "to the requested PDF path, then send an email whose content includes the relevant values from "
        "the read observation; mentioning a PDF path is not an attachment.\n"
        " - Output under a named folder: create the folder first if it does not exist, then use the exact "
        "requested destination path.\n"
        "If a file search returns no matches, do not repeatedly vary only head, tail, or sed limits on "
        "the same search. Broaden the search once, then use an exact discovered path or change apps.\n"
        "You have following apps installed in the system:\n"
        f"{app_introduction}"
    ).strip()


def _format_history(history: List[Tuple[object, str]]) -> str:
    out = ""
    for i, (action, observation) in enumerate(history):
        action_txt = _render_action(action)
        if observation:
            observation = observation.replace("\n", "\\n")
            out += f" - Step {i}: {action_txt} -> [{observation}]\n"
        else:
            out += f" - Step {i}: {action_txt}\n"
    return out


def build_prompt(current_app: Optional[str], history: List[Tuple[object, str]],
                 config: Dict, catalogue: Dict) -> str:
    """Port of LLMPolicy.build_prompt (the per-step user prompt)."""
    available_apps = catalogue["available_apps"]
    # Tolerate a model that switched to a non-existent app: treat it as 'no app chosen yet'.
    if current_app is not None and current_app not in catalogue["actions"]:
        current_app = None

    if current_app is None:
        if not history:
            return PROMPT_DICT["prompt_undecided_app"].format_map({
                "task": config["task"], "available_apps": available_apps})
        return PROMPT_DICT["prompt_undecided_app_w_history"].format_map({
            "task": config["task"], "history": _format_history(history),
            "available_apps": available_apps})

    detailed_instruction = ""
    for action in catalogue["actions"][current_app]:
        detailed_instruction += f" - {catalogue['demo'][current_app][action]}\n"
    other_apps = [app for app in available_apps if app != current_app]
    return PROMPT_DICT["prompt_decided_app"].format_map({
        "task": config["task"], "history": _format_history(history),
        "current_app": current_app, "detailed_instruction": detailed_instruction,
        "available_apps": other_apps})


def proc_action(action: str) -> str:
    """Port of LLMPolicy.proc_action: keep only the outermost ``{...}`` JSON action."""
    if "{" not in action or "}" not in action:
        return action
    left = action.find("{")
    right = action.rfind("}") + 1
    return action[left:right]


def _parse_action(action_string: str) -> Optional[Dict]:
    try:
        action = ast.literal_eval(action_string)
    except Exception:
        action = None
    # Fallback: the model emitted several JSON objects in one turn (or trailing prose), so the strict
    # parse failed — recover the FIRST brace-balanced action instead of dropping the whole step.
    if not isinstance(action, dict) and _sanitize_actions():
        first = _extract_first_object(action_string)
        if first is not None and first != action_string:
            try:
                action = ast.literal_eval(first)
            except Exception:
                action = None
    if not isinstance(action, dict):
        return None

    def normalize(value):
        if value is Ellipsis:
            return "..."
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    # minor fix: unwrap single-element lists (mirrors OfficeAgentEnv._minor_action_fix)
    action = normalize(action)
    return {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in action.items()}


def run_officebench_agent(session, config: Dict, chat_client, *, max_steps: int = 40,
                          temperature: float = 0.0, max_tokens: int = 1024,
                          predictor=None, predict_all_actions: bool = True) -> AgentRun:
    """Drive the official OfficeBench agent loop against a live :class:`OfficeBenchSession`.

    ``system`` actions (switch_app / finish_task / got_stuck) are resolved here, where the agent's
    app + workdir state lives.  Non-system actions are dispatched two ways:

    * ``predictor is None`` (B0 / real path): every action is **executed** in-container via
      ``session.exec_action`` — the faithful official behaviour, used for the committed rollout.
    * ``predictor`` set (``llm_wm`` sampling): the action is routed through the **world model**. Reads
      are predicted by default; ``predict_all_actions=False`` restores exact execution for untouched
      clean reads. A **WRITE is LLM-predicted and NEVER executed**, so the shadow environment is never
      mutated (the τ²/AppWorld contract).

    Returns an :class:`AgentRun` carrying the realized action/observation history so the trajectory can
    be committed (and, on the WM path, judged by the success verifier).
    """
    from .officebench_wm_predict import (
        OfficeBenchWMQuery,
        action_execution_kind,
        is_write_action,
        read_target,
        shell_cd_target,
        _looks_like_failure,
    )

    catalogue = session.apps_catalogue()
    system_message = construct_system_message(config, catalogue)

    current_app: Optional[str] = None
    workdir = "/"
    history: List[Tuple[object, str]] = []
    action_window: List[str] = []
    run = AgentRun()
    if predictor is not None:
        predictor.program_only = bool(predict_all_actions)
        bind_query = getattr(predictor, "bind_query_interface", None)
        if callable(bind_query):
            bind_query(OfficeBenchWMQuery(
                session,
                lambda: workdir,
                allow_action_execution=not predict_all_actions,
            ))

    # ``max_steps`` bounds the *productive* steps; pruned degenerate reads (Fix 4) don't count against
    # it, so a separate hard iteration ceiling guarantees termination regardless.
    prune = _prune_loops()
    productive = 0
    iters = 0
    hard_cap = max_steps * 2 + _ACTION_WINDOW_SIZE
    last_action_string: Optional[str] = None

    while productive < max_steps and iters < hard_cap:
        iters += 1
        prompt = build_prompt(current_app, history, config, catalogue)
        response = chat_client.chat(system_message, prompt, temperature=temperature,
                                    max_tokens=max_tokens)
        action_string = proc_action(response or "")
        if action_string == "":
            action_string = response or ""

        # 5-step no-progress window -> force got_stuck (port of LLMPolicy.forward)
        action_window.append(action_string)
        action_window = action_window[-_ACTION_WINDOW_SIZE:]
        if (len(action_window) >= _ACTION_WINDOW_SIZE
                and all(a == action_window[0] for a in action_window)):
            action_string = repr({"app": "system", "action": "got_stuck"})

        action = _parse_action(action_string)

        # Fix 4: prune an immediate-repeat READ (same action as the previous step) — its result cannot
        # have changed, so reuse it with a nudge and do NOT spend a productive step on it.
        if (prune and action is not None and last_action_string is not None
                and action_string == last_action_string
                and action.get("app") not in (None, "system")
                and not is_write_action(action, workdir)):
            prev_obs = run.observations[-1] if run.observations else ""
            observation = (f"OBSERVATION: (repeated action — result unchanged) "
                           f"{str(prev_obs)[:200]}. Take a different action to make progress.")
            history.append((action, observation))
            run.actions.append(action)
            run.observations.append(prev_obs)
            continue

        run.n_steps += 1
        productive += 1

        if action is None:
            observation = "Malformed action! You must follow the given action format! Try a different action."
            raw = action_string[:_HISTORY_ARG_CHARS] if _sanitize_actions() else action_string
            history.append((action_string, observation))
            run.actions.append({"raw": raw})
            run.observations.append(observation)
            last_action_string = action_string
            continue

        # GPT occasionally emits the natural shorthand
        # {"app": "shell", "action": "switch_app"} instead of the documented system action.
        # Recover the intended target rather than reporting a successful switch to ``None`` and
        # trapping the rollout in the undecided-app prompt.
        if (
            action.get("action") == "switch_app"
            and not action.get("target_app")
            and action.get("app") in catalogue["available_apps"]
        ):
            action = {
                **action,
                "app": "system",
                "target_app": action["app"],
            }

        act_name = action.get("action")
        app = action.get("app")

        # switch_app may be addressed to any app; the official code sets current_app to the target
        # (an unknown target degrades to the 'no app chosen' prompt via build_prompt).
        if act_name == "switch_app":
            target_app = action.get("target_app")
            if target_app not in catalogue["available_apps"]:
                current_app = None
                observation = (
                    "Malformed switch_app action: target_app must name one of "
                    f"{catalogue['available_apps']}. Try again with the documented system action."
                )
            else:
                current_app = target_app
                observation = f"Successfully switched to app: {current_app}"
        elif act_name == "finish_task":
            answer = action.get("answer", "None")
            # In WM (predictor) mode the answer is a PREDICTED write to answer.txt (recorded, not
            # written to the shadow); the real write happens only when the trajectory is committed.
            if predictor is None:
                session.write_answer(str(answer))
            else:
                record_answer = getattr(predictor, "record_answer", None)
                if callable(record_answer):
                    record_answer(str(answer))
            run.final_answer = str(answer)
            observation = "Task finished"
            run.done = True
        elif act_name == "got_stuck":
            observation = "Task failed"
            run.done = True
        elif predictor is not None:
            # WORLD-MODEL routing: reads exact (dirty-reads predicted), writes predicted, no real write.
            execution_kind = action_execution_kind(action, workdir)
            if execution_kind == "blocked":
                observation = (
                    "OBSERVATION: This action was not executed because it cannot be proven read-only "
                    "within /testbed. Inspect files using only /testbed paths and safe commands such as "
                    "ls, find without -delete/-exec, cat, grep, sed without -i/e/w, sort, head, tail, "
                    "file, or wc; do not use Python, xargs, sh, or output redirection for inspection."
                )
            elif execution_kind == "write":
                observation = predictor.predict_write(action, history)
            else:
                tgt = read_target(action, workdir)
                is_cd = (action.get("app") == "shell"
                         and shell_cd_target(str(action.get("command", ""))) is not None)
                if predict_all_actions:
                    dirty = predictor.was_written(tgt) and not is_cd
                    if predictor.uses_fixed_context:
                        pre = predictor.fixed_pre_write_content(tgt)
                    else:
                        pre = None
                    observation = predictor.predict_read(
                        action, pre, history, dirty=dirty)
                    if is_cd and not _looks_like_failure(observation):
                        cd_target = shell_cd_target(str(action.get("command", "")))
                        if cd_target:
                            workdir = os.path.normpath(
                                cd_target
                                if os.path.isabs(cd_target)
                                else os.path.join(workdir, cd_target)
                            )
                elif predictor.was_written(tgt) and not is_cd:
                    pre = predictor.pre_write_content.get(tgt)
                    if pre is None:
                        fixed_pre_write = getattr(
                            predictor, "fixed_pre_write_content", None)
                        if callable(fixed_pre_write):
                            pre = fixed_pre_write(tgt)
                    if pre is None:
                        pre = session.exec_action(action, workdir).get("observation", "")
                    observation = predictor.predict_read(action, pre, history)
                else:
                    result = session.exec_action(action, workdir)  # exact, read-only (never mutated)
                    observation = result.get("observation", "")
                    predictor.note_read(tgt, observation)
                    if result.get("is_cd") and result.get("new_workdir"):
                        workdir = result["new_workdir"]
        else:
            result = session.exec_action(action, workdir)
            observation = result.get("observation", "")
            if result.get("is_cd") and result.get("new_workdir"):
                workdir = result["new_workdir"]
        if predictor is not None and action.get("app") != "system":
            from .officebench_semantics import normalize_agent_observation
            observation = normalize_agent_observation(action, observation)
        history.append((action, observation))
        run.actions.append(action)
        run.observations.append(observation)
        last_action_string = action_string
        if run.done:
            break

    run.history = history
    return run


__all__ = ["AgentRun", "PROMPT_DICT", "construct_system_message", "build_prompt",
           "proc_action", "run_officebench_agent"]
