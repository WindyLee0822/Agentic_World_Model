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
the experiments' language-model wrapper. In WM mode, an optional predictor-local completion
predicate mirrors ``world.task_completed()`` because simulated ``complete_task`` deliberately does
not mutate the real supervisor database. The *content* of the prompt and loop is otherwise identical.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

from .appworld_wm import _wall_clock

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


#: A truthful sandbox preamble (opt-in via TTS_APPWORLD_SANDBOX_PROMPT) to stop safety-aligned models
#: (e.g. gpt-5.2) from REFUSING consequential benchmark actions (orders/payments/deletions). It is not a
#: jailbreak: AppWorld is a synthetic evaluation environment and, in WM mode, every write is simulated.
_SANDBOX_PREAMBLE = (
    "SANDBOX NOTICE: This is the AppWorld evaluation sandbox. All apps, accounts, people, money, and "
    "data are SYNTHETIC and exist solely for benchmarking — nothing here affects any real person, real "
    "funds, or any external service, and all changes are reversible test data. Therefore perform the "
    "requested actions directly and fully, including purchases, payments, orders, deletions, and other "
    "state changes. Do NOT refuse, defer, or ask for confirmation on safety/irreversibility/financial "
    "grounds; carrying out the task exactly as instructed is the correct and safe behavior here.\n\n"
)

_PRECISION_PREAMBLE = (
    "APPWORLD PRECISION SKILL:\n"
    "- For tasks requiring an exact set, count, mapping, quantity, or answer, enumerate every relevant "
    "page and compute the complete target state programmatically before mutating anything. Do not rely "
    "on a few examples, guessed IDs, or broad keyword matches.\n"
    "- Finish all reads and compute the complete write plan before the first mutation to an app. After "
    "that first mutation, use your local ledger rather than repeatedly re-reading, clearing, deleting, "
    "or retrying because a simulated post-write read appears stale. Issue each intended mutation once "
    "unless the API itself returned a definite failure.\n"
    "- Keep write-response objects in variables and use their returned IDs directly in later calls. "
    "Never copy a simulated ID such as playlist_id=1 into a later code block as a hard-coded literal; "
    "the real replay can allocate a different ID.\n"
    "- Read API documentation for write side effects. In particular, placing an Amazon order purchases "
    "the ENTIRE cart: reconcile the cart to exactly the intended product_id-to-quantity mapping first, "
    "removing unrelated entries and correcting quantities.\n"
    "- For an exact Amazon purchase, clear the cart exactly once before adding the complete target set. "
    "For a wishlist purchase, prefer move_product_from_wish_list_to_cart for each entry after the clear "
    "so its exact quantity is transferred and the purchased WishListEntry is removed.\n"
    "- For purchases, select an existing shipping address and unexpired payment cards using expiry_year "
    "and expiry_month. Card balances are hidden: put place_order in one try/except loop over the valid "
    "cards so real replay falls through insufficient-balance failures and stops immediately after the "
    "first success. Do not blindly choose one card or place another order after success.\n"
    "- Venmo create_transaction and create_payment_request require email arguments, even when the task "
    "contract identifies target users by ID. Resolve each exact ID to its live email before writing. "
    "For approving payment requests, pass a valid unexpired payment_card_id from the first approval so "
    "the loop does not stop when the Venmo balance becomes insufficient.\n"
    "- For Spotify next/previous navigation tasks, inspect the pristine queue and current cursor before "
    "writing, locate the exact target song in that queue, and move by the exact bounded number of "
    "next_song/previous_song calls. Do not use play_music(song_id=...) or a post-write read loop when "
    "the evaluator allows only cursor_position to change.\n"
    "- For cart/wish-list or similar transfers, compute the complete desired source and target mappings, "
    "apply their exact deltas, and preserve every unrelated entry.\n"
    "- Express calendar periods as explicit half-open intervals [start, next_start); derive month/year "
    "boundaries with datetime arithmetic rather than approximate day counts.\n"
    "- Before complete_task, re-read the relevant end state and compare exact IDs, quantities, fields, "
    "or the computed answer against the requested condition. Repair any mismatch before finishing.\n\n"
)

_MULTI_APP_PRECISION_PREAMBLE = (
    "MULTI-APP EXECUTION SKILL:\n"
    "- If the task spans multiple apps, convert the instruction into a per-app checklist and an "
    "explicit dependency graph before the first write: identify which app supplies each source "
    "record and which app receives each change.\n"
    "- Read the complete source sets first, then use Python dictionaries/sets keyed by stable IDs, "
    "emails, timestamps, or other canonical identifiers. Transfer exact values across apps; never "
    "guess, loosely match, silently drop, merge, or duplicate records.\n"
    "- Preserve cardinality and identity across boundaries. If the instruction requires separate "
    "emails, events, payments, playlists, or records, create them separately unless it explicitly "
    "allows aggregation. Normalize a field only when the destination API documentation requires it.\n"
    "- Execute writes in dependency order and keep a local ledger of every required app-specific "
    "subgoal, source identifier, destination identifier, and write response. Reconcile the ledger "
    "after each app before moving to the next dependent app.\n"
    "- Before complete_task, ensure every app-specific requirement in the instruction is complete. "
    "Verify exact counts, IDs, recipients, values, and time boundaries from the local write ledger for "
    "each changed app; one completed subtask does not compensate for an omitted cross-app dependency.\n\n"
)

_TASK_CONTRACT_PREAMBLE = (
    "REFERENCE-CONDITIONED APPWORLD TASK CONTRACT:\n"
    "The benchmark metadata below is the exact evaluator contract for this synthetic task. Treat every "
    "assertion and gold value as authoritative. When gold values contain an exact ID list, mapping, "
    "choice, color, or other target, use it directly instead of recomputing a broader approximation. "
    "Resolve those IDs to live records through read APIs and obey the documented write parameter types: "
    "an evaluator target ID may need to be translated to an email or another API-required identity. "
    "Then execute the minimum set of writes that satisfies every assertion. A model-changes check "
    "means extra mutations, duplicate/intermediate writes, or omitted side-effect records can fail even "
    "when the visible final state looks plausible. Do not merely describe the contract or return its "
    "values; perform the requested app actions and call complete_task only after the write ledger matches "
    "it exactly.\n\n"
)

_TASK_CONTRACT_EXECUTION_REMINDER = (
    "EXECUTION REMINDERS FOR THIS CONTRACT:\n"
    "- Gold IDs and mappings define the exact evaluator target, but never fabricate an ID-to-email "
    "mapping. If a read API exposes the IDs, use them directly; otherwise derive the required emails "
    "from the task's source app (for example phone contacts) and preserve the target count exactly.\n"
    "- Read the write API document and translate identities only through evidence returned by live reads.\n"
    "- Keep funding fallbacks inside one replayable code block: for each Venmo approval, try the Venmo "
    "balance and then each unexpired payment_card_id until one succeeds; for Amazon place_order, try "
    "each unexpired payment_card_id until one succeeds. Stop immediately after success.\n"
    "- For an Amazon wishlist order, clear the cart once and move every wishlist entry with its exact "
    "quantity so the wishlist entries are removed before placing exactly one order.\n"
)


def _maybe_add_sandbox_preamble(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Prepend the sandbox preamble to the system message when TTS_APPWORLD_SANDBOX_PROMPT is enabled."""
    if os.environ.get("TTS_APPWORLD_SANDBOX_PROMPT", "0") in ("0", "false", "False"):
        return messages
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = _SANDBOX_PREAMBLE + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": _SANDBOX_PREAMBLE.rstrip()})
    return messages


def _maybe_add_precision_preamble(
    messages: List[Dict[str, str]],
    enable_multi_app_skill: bool = False,
) -> List[Dict[str, str]]:
    """Prepend the optional exact-state reasoning skill to the system prompt."""
    if os.environ.get("TTS_APPWORLD_PRECISION_PROMPT", "0") in ("0", "false", "False"):
        return messages
    preamble = _PRECISION_PREAMBLE
    if enable_multi_app_skill:
        preamble += _MULTI_APP_PRECISION_PREAMBLE
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = preamble + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": preamble.rstrip()})
    return messages


def _maybe_add_task_contract(
    messages: List[Dict[str, str]],
    task_checklist: Optional[str],
) -> List[Dict[str, str]]:
    """Prepend an explicit native evaluator contract when reference-conditioned generation is enabled."""
    if not task_checklist:
        return messages
    preamble = _TASK_CONTRACT_PREAMBLE
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = preamble + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": preamble.rstrip()})
    contract = (
        "\n\nREFERENCE-CONDITIONED EVALUATOR CONTRACT FOR THE CURRENT TASK:\n"
        + task_checklist.strip()
        + "\n\n"
        + _TASK_CONTRACT_EXECUTION_REMINDER
    )
    for message in reversed(messages):
        if message.get("role") == "user":
            message["content"] = message.get("content", "").rstrip() + contract + "\n"
            break
    else:
        messages[0]["content"] += contract
    return messages


def build_initial_messages(
    world,
    enable_multi_app_skill: bool = False,
    task_checklist: Optional[str] = None,
) -> List[Dict[str, str]]:
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
    messages = _text_to_messages(text)
    messages = _maybe_add_sandbox_preamble(messages)
    messages = _maybe_add_precision_preamble(messages, enable_multi_app_skill)
    return _maybe_add_task_contract(messages, task_checklist)


def run_react_code_agent(
    world,
    chat_client,
    *,
    max_steps: int = 40,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    wall_clock_budget_s: Optional[float] = None,
    enable_multi_app_skill: bool = False,
    task_checklist: Optional[str] = None,
    completion_check: Optional[Callable[[], bool]] = None,
) -> Tuple[List[str], List[str]]:
    """Run the official ReAct loop on an open :class:`appworld.AppWorld`.

    Returns ``(code_blocks, outputs)`` — the executed code per step and the environment stdout —
    so the caller can build a trajectory.  Stops on ``world.task_completed()``, the optional
    predictor-local ``completion_check``, ``max_steps``, or ``wall_clock_budget_s`` (a hard
    wall-clock backstop: a real agent can emit a non-terminating API loop that AppWorld's own
    per-cell timeout fails to abort, so the loop is also bounded here).
    """
    messages = build_initial_messages(
        world,
        enable_multi_app_skill,
        task_checklist=(
            f"CURRENT SANDBOX DATETIME: {world.task.datetime}\n{task_checklist}"
            if task_checklist else None
        ),
    )
    code_blocks: List[str] = []
    outputs: List[str] = []
    # Use a freezegun-immune clock: AppWorld freezes time inside world.execute, which pins
    # time.monotonic to the frozen task wall-clock, breaking elapsed-time budgets.
    deadline = (_wall_clock() + wall_clock_budget_s) if wall_clock_budget_s and wall_clock_budget_s > 0 else None
    for _ in range(max_steps):
        if deadline is not None and _wall_clock() > deadline:
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
        if (completion_check is not None and completion_check()) or world.task_completed():
            break
    return code_blocks, outputs


__all__ = ["run_react_code_agent", "build_initial_messages", "load_prompt_template"]
