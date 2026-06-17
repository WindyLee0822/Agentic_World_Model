"""The **official** AppWorld ReAct code agent loop (faithful re-implementation).

AppWorld's reference agent solves a task by a multi-step REPL conversation: the model writes a
single ``python`` code block, the environment executes it and returns stdout, and the loop repeats
until the agent calls ``apis.supervisor.complete_task`` (detected by ``world.task_completed()``) or
the step budget is exhausted.  The terminal reward is the official ``world.evaluate()`` (Task Goal
Completion — every test passes).

This module reproduces that loop and the official prompt exactly:

* the prompt is the bundled ``prompts/appworld_react_code_agent.txt`` (verbatim copy of the
  official ``experiments/prompts/react_code_agent/instructions.txt``);
* the prompt is split into role-tagged messages with the same ``(USER|ASSISTANT|SYSTEM):`` parser;
* only the **first** ``python`` code block per turn is executed (``ignore_multiple_calls``);
* observations are wrapped as ``Output:\n```\n...\n``` `` user turns.

The only difference from the official ``SimplifiedReActCodeAgent`` is the transport: we render the
template without Jinja2 (env ``env`` has no jinja2) via targeted placeholder substitution, and we
call the model through :class:`tts.llm.chat_client.ChatClient` (openai + azure-identity) instead of
the experiments' language-model wrapper.  The *content* of the prompt and loop is identical.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "appworld_react_code_agent.txt")
_FULL_CODE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_PARTIAL_CODE_RE = re.compile(r".*```python\n(.*)", re.DOTALL)
_ROLE_RE = re.compile(r"(USER|ASSISTANT|SYSTEM):\n", re.IGNORECASE)


def load_prompt_template() -> str:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as fh:
        return fh.read().lstrip()


def _render(template: str, *, main_user, app_descriptions: str, instruction: str) -> str:
    subs = {
        r"\{\{\s*main_user\.first_name\s*\}\}": getattr(main_user, "first_name", ""),
        r"\{\{\s*main_user\.last_name\s*\}\}": getattr(main_user, "last_name", ""),
        r"\{\{\s*main_user\.email\s*\}\}": getattr(main_user, "email", ""),
        r"\{\{\s*main_user\.phone_number\s*\}\}": getattr(main_user, "phone_number", ""),
        r"\{\{\s*app_descriptions\s*\}\}": app_descriptions,
        r"\{\{\s*instruction\s*\}\}": instruction,
    }
    out = template
    for pattern, value in subs.items():
        out = re.sub(pattern, lambda _m, v=value: str(v), out)
    return out


def _text_to_messages(text: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    last_start = 0
    for m in _ROLE_RE.finditer(text):
        last_end = m.span()[0]
        if messages:
            messages[-1]["content"] = text[last_start:last_end]
        messages.append({"role": m.group(1).lower(), "content": ""})
        last_start = m.span()[1]
    if messages:
        messages[-1]["content"] = text[last_start:]
    return messages


def _extract_first_code(text: str) -> str:
    m = _FULL_CODE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _PARTIAL_CODE_RE.match(text)
    if m:
        return m.group(1).strip()
    return ""


def build_initial_messages(world) -> List[Dict[str, str]]:
    """Render the official prompt for ``world``'s task into role-tagged chat messages."""
    template = load_prompt_template()
    app_descriptions = json.dumps(
        [{"name": k, "description": v} for k, v in world.task.app_descriptions.items()],
        indent=1,
    )
    text = _render(
        template,
        main_user=world.task.supervisor,
        app_descriptions=app_descriptions,
        instruction=world.task.instruction,
    ) + "\n\n"
    return _text_to_messages(text)


def run_react_code_agent(
    world,
    chat_client,
    *,
    max_steps: int = 40,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    wall_clock_budget_s: Optional[float] = None,
) -> Tuple[List[str], List[str]]:
    """Run the official ReAct loop on an open :class:`appworld.AppWorld`.

    Returns ``(code_blocks, outputs)`` — the executed code per step and the environment stdout —
    so the caller can build a trajectory.  Stops on ``world.task_completed()``, ``max_steps``, or
    ``wall_clock_budget_s`` (a hard wall-clock backstop: a real agent can emit a non-terminating
    API loop that AppWorld's own per-cell timeout fails to abort, so the loop is also bounded here).
    """
    messages = build_initial_messages(world)
    code_blocks: List[str] = []
    outputs: List[str] = []
    deadline = (time.monotonic() + wall_clock_budget_s) if wall_clock_budget_s and wall_clock_budget_s > 0 else None
    for _ in range(max_steps):
        if deadline is not None and time.monotonic() > deadline:
            break  # hard wall-clock backstop (e.g. a non-terminating agent loop)
        content = chat_client.complete(messages, temperature=temperature, max_tokens=max_tokens)
        code = _extract_first_code(content or "")
        messages.append({"role": "assistant", "content": (content or "") + "\n\n"})
        if not code:
            break  # no actionable code: the agent has stalled
        output = world.execute(code)
        code_blocks.append(code)
        outputs.append(output)
        tail = "\n" if not output.endswith("\n") else ""
        messages.append({"role": "user", "content": f"Output:\n```\n{output}{tail}```\n\n"})
        if world.task_completed():
            break
    return code_blocks, outputs


__all__ = ["run_react_code_agent", "build_initial_messages", "load_prompt_template"]
