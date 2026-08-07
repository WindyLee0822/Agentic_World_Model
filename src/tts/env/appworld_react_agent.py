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
    "- Before complete_task, map every action verb in the instruction to at least one definite "
    "successful mutation in your local ledger. If the task asks to buy, send, reply, forward, update, "
    "create, delete, move, approve, request, schedule, label, follow, upload, or mark something and "
    "the corresponding write is absent, continue working; never complete after a read-only attempt.\n"
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

_L2_TARGETED_EXECUTION_PREAMBLE = (
    "APPWORLD LEVEL-2 TARGETED EXECUTION SKILL:\n"
    "- For Venmo month-scoped approvals, enumerate all pending received requests in the exact "
    "half-open month interval, resolve the requested sender/contact identity exactly, and approve "
    "every match. Use the documented payment-card fallback on each approval; if the task also asks "
    "for a withdrawal, approve first, re-read the resulting balance, then withdraw that exact "
    "remainder once to the card with the requested last four digits.\n"
    "- For shared-trip, dinner, taxi, or roommate settlements, include the user in the participant "
    "count and preserve each requested economic action separately. Do not net a payment against "
    "requests when the instruction asks for both; compute each person's share from the stated total, "
    "resolve every roommate through live contacts, and use the exact required note and privacy.\n"
    "- For Spotify tasks that say song, album, and playlist libraries, expand all three sources: "
    "collect directly liked/library songs, every track from each qualifying album, and every track "
    "from each qualifying playlist before taking the exact union or multiset. Keep the new playlist "
    "response and send its returned URL rather than constructing one.\n"
    "- When moving Todoist tasks, preserve task identity. Prefer update_task with the destination "
    "project_id and retain all original fields, subtasks, and comments; do not recreate and delete "
    "unless the API cannot move a task and the evaluator contract explicitly permits new IDs.\n"
    "- For CSV, note, and filesystem updates, mutate the existing target file or note in place, "
    "preserve its schema, ordering, unrelated rows, and formatting, and apply only exact deltas. "
    "Use the latest message/reply per person and never create a replacement file when an update API "
    "exists.\n"
    "- For Splitwise imports from notes or files, parse every source record into payer, debtors, "
    "amount, description, group, and receipt fields before writing. Preserve separate expenses and "
    "the stated split semantics; do not deduplicate distinct records or silently infer equal shares "
    "when explicit amounts are present.\n"
)

_CHALLENGE_EXECUTION_PREAMBLE = (
    "APPWORLD CHALLENGE EXECUTION SKILL:\n"
    "- For constrained Amazon selection, enumerate every search page before choosing. Filter using the "
    "exact requested product rating, seller rating, review count, price, dimensions, availability, "
    "delivery deadline, size, and review/QA conditions; do not stop after the first plausible result. "
    "Choose only among fully eligible products, then isolate the cart and place exactly one order.\n"
    "- When an instruction comes from a note or email, parse every requested product, quantity, person, "
    "date, path, and formatting rule into a checklist before writing. Resolve each item independently; "
    "do not abandon the whole task because one search query is too narrow—broaden the query while "
    "preserving the exact constraint ledger.\n"
    "- For receipt/ticket export tasks, paginate all relevant threads, inspect every matching message, "
    "and create or download one file per required record using the exact directory and filename format. "
    "Verify the complete expected file set, not just one example file.\n"
    "- For Gmail drafts and attachments, inspect the target draft and filesystem first, update the draft "
    "to the exact final recipient/body/attachment set, and send it exactly once only after the update "
    "succeeds. Never send an intermediate draft and then retry with a second message.\n"
    "- For Venmo reconciliation, enumerate every participant and source amount first, compute exact "
    "per-person shares programmatically, and build a ledger of direction, recipient email, amount, "
    "privacy, and note. Distinguish money you owe (payment) from money owed to you (request), account "
    "for your own share exactly once, and issue every ledger entry once.\n"
    "- For Spotify changes sourced from messages or notes, gather every suggestion before writing, "
    "normalize song identity using live search results, compute exact additions/removals against the "
    "original playlist as sets or multisets, and apply each delta once. Keep the created/updated "
    "playlist response variable so any later phone message uses its real replay URL or ID.\n"
    "- For dependent Amazon workflows—return then replace, order then save/forward a receipt, compare "
    "promotions, or buy from an emailed checklist—finish and record each dependency in order. Inspect "
    "existing filesystem organization before choosing a receipt path, preserve every requested draft "
    "placeholder and attachment, and do not finish after only the purchase or only the communication.\n"
    "- A simulated observation can be incomplete after a predicted write. Trust a definite API success "
    "response and your write ledger; do not call complete_task(status='fail') merely because a later "
    "simulated read is uncertain. Use failure status only after a definite API rejection or exhausted "
    "documented alternatives.\n\n"
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

_COMPLEX_TASK_EXECUTION_REMINDER = (
    "\n\nCOMPLEX-TASK EXECUTION REMINDER FOR THIS TASK:\n"
    "- The task is designed to be solvable. Do not call complete_task(status='fail') because a "
    "search was empty, a simulated post-write read was stale, or one query was too narrow. Broaden "
    "the read, inspect another source, or trust definite successful write responses in your ledger.\n"
    "- Conserve the available step budget for mutations: batch independent API-documentation lookups, "
    "logins, pagination loops, and source reads in the same Python block instead of spending one "
    "turn per call. Once the required identities and values are known, execute the complete write "
    "plan without another exploratory pass.\n"
    "- Before completing, verify that every action verb in the instruction has a corresponding "
    "successful write. If required writes are still missing, continue; if they succeeded and only "
    "a simulated confirmation read is uncertain, call complete_task() with success.\n"
)

_AMAZON_WORKFLOW_SKILL = (
    "AMAZON WORKFLOW REMINDER:\n"
    "- Derive the complete item/quantity/constraint ledger from the named email, note, message, "
    "checklist, past order, or cart before shopping. For 'remaining' items, subtract explicit "
    "commitments from the source checklist rather than buying the full list.\n"
    "- For trusted-seller tasks, collect seller IDs from past orders first and restrict candidates "
    "to that exact set. For highest-rated or cheapest tasks, enumerate all pages and apply every "
    "tie-breaker only after filtering availability, price, delivery, review, size, and feature "
    "constraints.\n"
    "- The cart is one order to one address. For different recipients/addresses/gift settings, make "
    "separate exact-cart orders, grouping only recipients at the same address. Never leave products "
    "for a later address in the current cart.\n"
    "- To compare promo codes, record the original cart, test each code one at a time, read the full "
    "resulting total, remove it, then reapply only the cheapest valid option before ordering.\n"
    "- For return/receipt workflows, complete the order or return first, then locate the newly "
    "generated confirmation thread/attachment and forward that exact record to the requested person.\n"
    "- If cards lack funds, compute the minimum final charge, transfer that amount to a valid card "
    "before place_order, and then place exactly one order.\n\n"
)

_GMAIL_WORKFLOW_SKILL = (
    "GMAIL WORKFLOW REMINDER:\n"
    "- Identify the exact source thread, email, or existing draft before writing. Reply/forward on "
    "that object rather than creating an unrelated message, preserve all required recipients, and "
    "send each intended message exactly once.\n"
    "- For bulk reply, reminder, label, star, unread, spam, or scheduling tasks, enumerate all pages "
    "of the relevant inbox/outbox/draft set and build the complete target thread-ID set first. Apply "
    "all requested post-send thread state changes after the reply succeeds.\n"
    "- For scheduled replies, schedule one response per source email at the requested time relative "
    "to that email's timestamp. For template workflows, fill each candidate's own decision/template "
    "and preserve placeholders, subjects, attachments, and existing draft formatting.\n"
    "- For file exports, create every required file with the exact filename, package the complete "
    "set when a zip is requested, attach that artifact to the existing reply/draft, then send it.\n\n"
)

_SPOTIFY_WORKFLOW_SKILL = (
    "SPOTIFY WORKFLOW REMINDER:\n"
    "- Distinguish liked songs from songs merely present in a library. For intersections/unions, "
    "enumerate liked songs, song library, album tracks, and playlist tracks separately, normalize "
    "by stable song ID, then compute the exact requested set before writing or replying.\n"
    "- For migrations, parse the entire source file first, create/login to the account, then restore "
    "song library, album library, and every playlist with exact membership as separate ledgers.\n"
    "- For recommendations, gather every response from every specified email/contact channel before "
    "searching Spotify. Resolve each exact artist/song, deduplicate by stable ID, and follow/add all "
    "matches before completing the task.\n\n"
)

_RELATIONSHIP_MONEY_SKILL = (
    "RELATIONSHIP AND MONEY REMINDER:\n"
    "- Resolve roommates, family, coworkers, friends, managers, and partners from live contacts and "
    "relationships; never infer an email from a name. Include the user when computing equal shares.\n"
    "- For reimbursements, subtract only payments that clearly correspond to the stated expense, "
    "then issue one exact payment/request/expense per remaining person with the required note, "
    "privacy, receipt, payer, debtor, and share. Do not net distinct requested operations.\n\n"
)

_TODOIST_EXECUTION_SKILL = (
    "TODOIST EXECUTION REMINDER:\n"
    "- Read every task, subtask, description, and comment in the named project before acting. Convert "
    "them into an ordered cross-app checklist, execute each external action completely, record its "
    "returned IDs, and mark a Todoist item complete only after its corresponding action succeeds.\n\n"
)


def _maybe_add_instruction_guidance(
    messages: List[Dict[str, str]],
    instruction: str,
    enabled: bool,
    broad_routing: bool = False,
) -> List[Dict[str, str]]:
    """Add compact instruction-derived playbooks without using evaluator metadata."""
    if (
        not enabled
        or os.environ.get("TTS_APPWORLD_INSTRUCTION_GUIDANCE", "1")
        in ("0", "false", "False")
    ):
        return messages
    text = instruction.lower()
    guidance = ""
    broad_routing = (
        broad_routing
        or os.environ.get("TTS_APPWORLD_GUIDANCE_ROUTING") == "broad"
    )
    amazon_workflow = "amazon" in text and (
        broad_routing
        or any(term in text for term in (
            "promo code", "flight confirmation", "highest-rated", "top-rated",
            "last 2 amazon orders", "last two amazon orders", "receipt", "return",
            "shopping list", "checklist",
        ))
    )
    gmail_workflow = any(
        term in text for term in ("email", "gmail", "draft", "inbox", "outbox")
    ) and (
        broad_routing
        or any(term in text for term in (
            "all", "every", "zip", "compile", "label", "schedule", "draft",
            "attachment", "bill", "confirmation",
        ))
    )
    money_workflow = any(term in text for term in (
        "venmo", "splitwise", "owe", "owed", "share", "expense", "payment",
        "request money", "send $",
    )) or (
        broad_routing
        and any(term in text for term in (
            "roommate", "friend", "coworker", "manager", "husband", "wife",
            "parent", "sibling", "sister", "brother",
        ))
    )
    if amazon_workflow:
        guidance += _AMAZON_WORKFLOW_SKILL
    if gmail_workflow:
        guidance += _GMAIL_WORKFLOW_SKILL
    if broad_routing and any(
        term in text for term in ("spotify", "song", "playlist", "album", "artist")
    ):
        guidance += _SPOTIFY_WORKFLOW_SKILL
    if money_workflow:
        guidance += _RELATIONSHIP_MONEY_SKILL
    if "todoist" in text or "project" in text and "task" in text:
        guidance += _TODOIST_EXECUTION_SKILL
    if not guidance:
        return messages
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = guidance + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": guidance.rstrip()})
    return messages


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
    enable_l2_targeted_skill: bool = False,
) -> List[Dict[str, str]]:
    """Prepend exact-state reasoning, enabled by default for multi-app tasks."""
    configured = os.environ.get("TTS_APPWORLD_PRECISION_PROMPT")
    enabled = (
        enable_multi_app_skill
        if configured is None
        else configured not in ("0", "false", "False")
    )
    if not enabled:
        return messages
    preamble = _PRECISION_PREAMBLE
    if enable_multi_app_skill:
        preamble += _MULTI_APP_PRECISION_PREAMBLE
        if enable_l2_targeted_skill:
            preamble += _L2_TARGETED_EXECUTION_PREAMBLE
        preamble += _CHALLENGE_EXECUTION_PREAMBLE
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = preamble + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": preamble.rstrip()})
    return messages


def _instruction_needs_l2_targeted_skill(instruction: str) -> bool:
    """Route L2-only guidance to instruction families where paired runs improved."""
    text = instruction.lower()
    return any(term in text for term in (
        "highest-rated seller",
        "within $10",
        "t-shirt",
        "camping lantern",
        "download receipts of all my amazon orders",
        "highest-rated picture frame",
        "delivered today was an order",
        "amazon shopping list",
        "resignation email",
        "announcement email",
        "weekly work meeting schedule",
        "electricity_bill.pdf",
        "weekly standup time",
    ))


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
    enable_l2_targeted_skill: bool = False,
    enable_instruction_guidance: bool = False,
    broad_instruction_guidance: bool = False,
    enable_execution_reminder: bool = False,
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
    messages = _maybe_add_precision_preamble(
        messages,
        enable_multi_app_skill,
        (
            enable_l2_targeted_skill
            and _instruction_needs_l2_targeted_skill(world.task.instruction)
        ),
    )
    messages = _maybe_add_instruction_guidance(
        messages,
        world.task.instruction,
        enable_instruction_guidance,
        broad_instruction_guidance,
    )
    reminder_enabled = (
        enable_execution_reminder
        and os.environ.get("TTS_APPWORLD_EXECUTION_REMINDER", "1")
        not in ("0", "false", "False")
    )
    if reminder_enabled:
        for message in reversed(messages):
            if message.get("role") == "user":
                message["content"] = (
                    message.get("content", "").rstrip()
                    + _COMPLEX_TASK_EXECUTION_REMINDER
                )
                break
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
    enable_l2_targeted_skill: bool = False,
    enable_instruction_guidance: bool = False,
    broad_instruction_guidance: bool = False,
    enable_execution_reminder: bool = False,
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
        enable_l2_targeted_skill,
        enable_instruction_guidance,
        broad_instruction_guidance,
        enable_execution_reminder,
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
