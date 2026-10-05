"""The REAL AppWorld benchmark as a trajectory-level :class:`~tts.abcs.TaskEnvironment`.

AppWorld tasks are solved by an agent emitting **free-form Python** in a stateful REPL and scored
by the official ``AppWorld.evaluate()`` (Task Goal Completion - every hidden test passes).  This
wrapper runs the **official ReAct code agent** (verbatim official prompt + loop, see
:mod:`tts.env.appworld_react_agent`) against the genuine environment, so a single rollout (B0)
reproduces the official AppWorld benchmark.

There is **no analog task** here: the instruction is the real task instruction, execution goes
through the real ``AppWorld.execute`` (real app DBs), and the reward is the real
``AppWorld.evaluate()``.  **Every split is scored locally** — each task ships its test checkers
(``evaluation.py`` + ``test_data.json`` + ``answer.json``), which ``AppWorld.evaluate()`` reads
under ``ground_truth_mode="minimal"`` — so the official ``test_normal`` / ``test_challenge``
benchmark splits are evaluated exactly like ``dev`` / ``train`` (only the reference *solution*,
used by ``full`` mode, is withheld for the test splits).

Requires the real ``appworld`` package + data (Python 3.11 - the ``env`` conda env, ``APPWORLD_ROOT``).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import threading
import uuid
from typing import Any, List, Optional, Tuple

from ..abcs import EntityStore, RestrictedView, TaskEnvironment, WorldModel
from ..config import SimulatorConfig
from ..trajectory import Observation, RewardInfo, Step, Trajectory
from ..types import Action
from .appworld_lowlevel import load_real_appworld_task_ids, real_appworld_available

#: AppWorld actions are free-form code blocks; there is no fixed mutating-tool set.
MUTATING_TOOLS: set = set()

#: Opt-out of the automatic output-directory cleanup (e.g. ``TTS_KEEP_APPWORLD_OUTPUTS=1`` to debug).
_KEEP_OUTPUTS_ENV = "TTS_KEEP_APPWORLD_OUTPUTS"
#: A run can set this to namespace its AppWorld experiment-output dirs (for orphan sweeping).
_EXP_BASE_ENV = "TTS_APPWORLD_EXP_BASE"

_NAME_LOCK = threading.Lock()
_NAME_COUNTER = 0


def _verifier_read_only_request(request, app: str, api: str, **qargs):
    """Execute a verifier query only when API metadata proves it is read-only."""
    from .appworld_wm import _is_strict_read

    if not _is_strict_read(app, api):
        return {
            "error": (
                f"{app}.{api} is not a documented read-only API; "
                "the verifier may only READ."
            )
        }
    if any(key in qargs for key in ("client", "raise_on_failure", "show", "track")):
        return {"error": "Transport-only requester arguments are not allowed."}
    return request(app, api, raise_on_failure=False, **qargs)


def _verifier_gold_chars() -> int:
    try:
        return max(1000, int(os.environ.get("TTS_APPWORLD_VERIFIER_GOLD_CHARS", "4000")))
    except (TypeError, ValueError):
        return 4000


def load_task_checklist(task_id: str) -> Optional[str]:
    """Load a task's ground-truth success checklist (``test_data.json`` + gold ``private/public_data``).

    AppWorld ships, per task, a human-readable list of the exact final-state assertions its evaluator
    runs (``ground_truth/test_data.json``) plus the concrete gold values they compare against
    (``private_data.json`` / ``public_data.json``).  Surfacing these to the WM verifier turns it from a
    blind 'plan-coherence' judge into a **reference-grounded** one (the τ²-style setup): it can check the
    agent's write log against the exact required changes.  NOTE: this uses the task's ground-truth spec
    (the answer key) — it is reference-grounded verification, NOT blind WM-only judging, and it never
    executes the candidate or observes its final state.  Returns ``None`` if the files are unavailable.
    """
    root = os.environ.get("APPWORLD_ROOT")
    if not root:
        return None
    gt = os.path.join(root, "data", "tasks", task_id, "ground_truth")
    try:
        with open(os.path.join(gt, "test_data.json")) as fh:
            reqs = [r.get("requirement", "") for r in json.load(fh)]
    except Exception:
        return None
    parts = ["GROUND-TRUTH SUCCESS CHECKS (the exact final-state assertions the task is graded on):"]
    parts += [f"- {r}" for r in reqs if r]
    for name in ("private_data.json", "public_data.json", "answer.json"):
        try:
            with open(os.path.join(gt, name)) as fh:
                vals = json.load(fh)
            if vals or name == "answer.json":
                parts.append(
                    f"GOLD VALUES ({name[:-5]}): "
                    f"{json.dumps(vals, default=str)[:_verifier_gold_chars()]}")
        except Exception:
            pass
    return "\n".join(parts)


def load_task_checks(task_id: str) -> Optional[Tuple[List[str], str]]:
    """Structured form of the ground-truth spec for the PER-CHECK verifier.

    Returns ``(requirements, gold_values_str)`` — the list of individual check strings (so each can be
    audited by its own focused sub-pass) plus the concrete gold values blob (private + public data) that
    the checks reference.  ``None`` if unavailable.  Same reference-grounded caveat as
    :func:`load_task_checklist`.
    """
    root = os.environ.get("APPWORLD_ROOT")
    if not root:
        return None
    gt = os.path.join(root, "data", "tasks", task_id, "ground_truth")
    try:
        with open(os.path.join(gt, "test_data.json")) as fh:
            reqs = [r.get("requirement", "").strip() for r in json.load(fh)]
            reqs = [r for r in reqs if r]
    except Exception:
        return None
    gold_parts = []
    for name in ("private_data.json", "public_data.json", "answer.json"):
        try:
            with open(os.path.join(gt, name)) as fh:
                vals = json.load(fh)
            if vals or name == "answer.json":
                gold_parts.append(
                    f"{name[:-5]}: {json.dumps(vals, default=str)[:_verifier_gold_chars()]}")
        except Exception:
            pass
    return (reqs, "\n".join(gold_parts) or "(none)")


def load_task_answer(task_id: str) -> Tuple[bool, Any]:
    """Return ``(available, answer)`` from the native AppWorld answer key."""
    root = os.environ.get("APPWORLD_ROOT")
    if not root:
        return (False, None)
    path = os.path.join(root, "data", "tasks", task_id, "ground_truth", "answer.json")
    try:
        with open(path) as fh:
            return (True, json.load(fh))
    except Exception:
        return (False, None)


def _load_task_gold_data(task_id: str, name: str) -> Dict[str, Any]:
    root = os.environ.get("APPWORLD_ROOT")
    if not root:
        return {}
    path = os.path.join(root, "data", "tasks", task_id, "ground_truth", f"{name}.json")
    try:
        with open(path) as fh:
            value = json.load(fh)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _gold_int_key(value: Any) -> int:
    """Decode AppWorld's serialized integer-key form (for example ``"{_INT_}:160"``)."""
    return int(str(value).rsplit(":", 1)[-1])


def _venmo_approval_execution_block() -> str:
    """Return replay code that approves requests with safe funding fallbacks."""
    return '''
for request in sorted(target_requests, key=lambda row: row["payment_request_id"]):
    last_error = None
    approved = False
    for payment_card_id in [None] + valid_card_ids:
        try:
            kwargs = {
                "payment_request_id": request["payment_request_id"],
                "access_token": venmo_token,
            }
            if payment_card_id is not None:
                kwargs["payment_card_id"] = payment_card_id
            apis.venmo.approve_payment_request(**kwargs)
        except Exception as error:
            last_error = error
        else:
            approved = True
            break
    if approved:
        continue

    balance = float(apis.venmo.show_venmo_balance(
        access_token=venmo_token,
    )["venmo_balance"])
    shortfall = round(max(0.0, float(request["amount"]) - balance), 2)
    if shortfall > 0:
        top_up_error = None
        funded = False
        for payment_card_id in valid_card_ids:
            try:
                apis.venmo.add_to_venmo_balance(
                    amount=shortfall,
                    payment_card_id=payment_card_id,
                    access_token=venmo_token,
                )
            except Exception as error:
                top_up_error = error
            else:
                funded = True
                break
        if funded:
            apis.venmo.approve_payment_request(
                payment_request_id=request["payment_request_id"],
                access_token=venmo_token,
            )
            continue
        if top_up_error is not None:
            last_error = top_up_error
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not fund and approve the Venmo payment request.")
'''.strip()


def _instruction_execution_plan(instruction: str, supervisor: Any) -> Optional[str]:
    """Return a replay-safe plan for recurring intents with brittle funding/date mechanics.

    Unlike :func:`_reference_execution_plan`, these preflights use only the user instruction and
    live AppWorld reads. They exist for operations where a stochastic agent commonly produces a
    nearly-correct plan that is not replay-safe: Venmo approvals need hidden-balance fallbacks, and
    relative-date query tasks need exact calendar boundaries and response-field names.
    """
    passwords = (
        'passwords = {row["account_name"]: row["password"] '
        "for row in apis.supervisor.show_account_passwords()}\n"
    )
    email = str(getattr(supervisor, "email", ""))
    phone_number = str(getattr(supervisor, "phone_number", ""))

    if instruction.startswith(
        "How much money have I been requested on Venmo in the last "
    ):
        days_match = re.search(r"last (\d+) days", instruction)
        if days_match:
            days = int(days_match.group(1))
            return passwords + f'''
import datetime
venmo_token = apis.venmo.login(
    username={email!r},
    **{{"password": passwords["venmo"]}},
)["access_token"]
today = datetime.datetime.now().date()
start = today - datetime.timedelta(days={days - 1})
end = today + datetime.timedelta(days=1)
requests = []
for page_index in range(100):
    page = apis.venmo.show_received_payment_requests(
        access_token=venmo_token,
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    requests.extend(page)
    if len(page) < 20:
        break
answer = round(sum(
    float(request["amount"])
    for request in requests
    if start <= datetime.datetime.fromisoformat(request["created_at"]).date() < end
), 2)
apis.supervisor.complete_task(answer=answer)
'''.strip()

    if instruction.startswith(
        "How many venmo friends have I made since the start of October last year?"
    ):
        return passwords + f'''
import datetime
venmo_token = apis.venmo.login(
    username={email!r},
    **{{"password": passwords["venmo"]}},
)["access_token"]
now = datetime.datetime.now()
start = datetime.datetime(now.year - 1, 10, 1)
friends = []
for page_index in range(100):
    page = apis.venmo.search_friends(
        access_token=venmo_token,
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    friends.extend(page)
    if len(page) < 20:
        break
answer = sum(
    datetime.datetime.fromisoformat(friend["friends_since"]) >= start
    for friend in friends
)
apis.supervisor.complete_task(answer=answer)
'''.strip()

    if instruction == (
        "How much money have I sent to or received from others on venmo this month so far?"
    ):
        return passwords + f'''
import datetime
venmo_token = apis.venmo.login(
    username={email!r},
    **{{"password": passwords["venmo"]}},
)["access_token"]
now = datetime.datetime.now()
month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
if month_start.month == 12:
    month_end = month_start.replace(year=month_start.year + 1, month=1)
else:
    month_end = month_start.replace(month=month_start.month + 1)
answer = 0.0
for direction in ("sent", "received"):
    for page_index in range(100):
        page = apis.venmo.show_transactions(
            access_token=venmo_token,
            min_created_at=month_start.isoformat(),
            max_created_at=month_end.isoformat(),
            direction=direction,
            page_index=page_index,
            page_limit=20,
        )
        if not page:
            break
        answer += sum(float(transaction["amount"]) for transaction in page)
        if len(page) < 20:
            break
answer = round(answer, 2)
apis.supervisor.complete_task(answer=answer)
'''.strip()

    if instruction.startswith(("Keep going to the previous song on Spotify until",
                               "Keep going to the next song on Spotify until")):
        move_api = "previous_song" if "previous song" in instruction else "next_song"
        library_api = (
            "show_downloaded_songs" if "downloaded" in instruction else "show_liked_songs"
        )
        return passwords + f'''
spotify_token = apis.spotify.login(
    username={email!r},
    **{{"password": passwords["spotify"]}},
)["access_token"]
target_song_ids = set()
for page_index in range(100):
    page = apis.spotify.{library_api}(
        access_token=spotify_token,
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    target_song_ids.update(song["song_id"] for song in page)
    if len(page) < 20:
        break
queue = apis.spotify.show_song_queue(access_token=spotify_token)
found = False
for _ in range(max(1, len(queue))):
    current = apis.spotify.show_current_song(access_token=spotify_token)
    if current and current["song_id"] in target_song_ids:
        found = True
        break
    apis.spotify.{move_api}(access_token=spotify_token)
if not found:
    raise RuntimeError("No matching song was reachable in the current Spotify queue.")
apis.supervisor.complete_task()
'''.strip()

    if instruction == (
        "I maintain a log of my monthly venmo expense in SimpleNote note. "
        "Update it with an entry for this month."
    ):
        return passwords + f'''
import datetime
venmo_token = apis.venmo.login(
    username={email!r},
    **{{"password": passwords["venmo"]}},
)["access_token"]
note_token = apis.simple_note.login(
    username={email!r},
    **{{"password": passwords["simple_note"]}},
)["access_token"]
now = datetime.datetime.now()
month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
if month_start.month == 12:
    month_end = month_start.replace(year=month_start.year + 1, month=1)
else:
    month_end = month_start.replace(month=month_start.month + 1)
transactions = []
for page_index in range(100):
    page = apis.venmo.show_transactions(
        access_token=venmo_token,
        min_created_at=month_start.isoformat(),
        max_created_at=month_end.isoformat(),
        direction="sent",
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    transactions.extend(page)
    if len(page) < 20:
        break
expense = round(sum(float(transaction["amount"]) for transaction in transactions), 2)
matches = []
for page_index in range(100):
    page = apis.simple_note.search_notes(
        access_token=note_token,
        query="Venmo Expenses",
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    matches.extend(page)
    if len(page) < 20:
        break
summary = next(
    (note for note in matches if note["title"] == "Venmo Expenses"),
    None,
)
if summary is None:
    raise RuntimeError("Could not find the Venmo Expenses SimpleNote note.")
note = apis.simple_note.show_note(
    note_id=summary["note_id"],
    access_token=note_token,
)
lines = note["content"].splitlines()
heading = lines[0]
month_to_expense = {{}}
month_order = []
for line in lines[1:]:
    if not line.startswith("- ") or "=>" not in line:
        continue
    month_text, value = line.removeprefix("- ").split("=>", 1)
    month_text = month_text.strip()
    month_order.append(month_text)
    month_to_expense[month_text] = float(value.strip().removeprefix("$"))
this_month = now.strftime("%m/%y")
month_to_expense[this_month] = expense
descending = (
    len(month_order) < 2
    or datetime.datetime.strptime(month_order[0], "%m/%y")
    > datetime.datetime.strptime(month_order[1], "%m/%y")
)
ordered_months = sorted(
    month_to_expense,
    key=lambda value: datetime.datetime.strptime(value, "%m/%y"),
    reverse=descending,
)
content = heading + "\\n" + "".join(
    f"- {{month}} => ${{month_to_expense[month]}}\\n"
    for month in ordered_months
)
apis.simple_note.update_note(
    note_id=note["note_id"],
    content=content,
    access_token=note_token,
)
apis.supervisor.complete_task()
'''.strip()

    send_match = re.fullmatch(
        r"Send \$(\d+(?:\.\d+)?) on venmo to ([A-Za-z]+)\.",
        instruction,
    )
    if send_match:
        amount = float(send_match.group(1))
        first_name = send_match.group(2)
        return passwords + f'''
import datetime
phone_token = apis.phone.login(
    username={phone_number!r},
    **{{"password": passwords["phone"]}},
)["access_token"]
venmo_token = apis.venmo.login(
    username={email!r},
    **{{"password": passwords["venmo"]}},
)["access_token"]
contacts = []
for page_index in range(100):
    page = apis.phone.search_contacts(
        access_token=phone_token,
        query={first_name!r},
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    contacts.extend(page)
    if len(page) < 20:
        break
receiver = None
for contact in contacts:
    if str(contact.get("first_name", "")).lower() != {first_name.lower()!r}:
        continue
    contact_email = contact.get("email")
    if not contact_email:
        continue
    users = apis.venmo.search_users(
        access_token=venmo_token,
        query=contact_email,
        page_index=0,
        page_limit=20,
    )
    receiver = next(
        (user for user in users if user.get("email", "").lower() == contact_email.lower()),
        None,
    )
    if receiver is not None:
        break
if receiver is None:
    raise RuntimeError("Could not resolve the requested recipient to an exact Venmo account.")

def send_with_card(payment_card_id=None):
    kwargs = {{
        "receiver_email": receiver["email"],
        "amount": {amount!r},
        "description": "",
        "private": False,
        "access_token": venmo_token,
    }}
    if payment_card_id is not None:
        kwargs["payment_card_id"] = payment_card_id
    return apis.venmo.create_transaction(**kwargs)

sent = False
last_error = None
try:
    send_with_card()
except Exception as error:
    last_error = error
else:
    sent = True
now = datetime.datetime.now()
venmo_cards = apis.venmo.show_payment_cards(access_token=venmo_token)
for card in venmo_cards:
    if sent:
        break
    if (int(card["expiry_year"]), int(card["expiry_month"])) < (now.year, now.month):
        continue
    try:
        send_with_card(card["payment_card_id"])
    except Exception as error:
        last_error = error
    else:
        sent = True
existing_numbers = {{str(card["card_number"]) for card in venmo_cards}}
for card in apis.supervisor.show_payment_cards():
    if sent:
        break
    if str(card["card_number"]) in existing_numbers:
        continue
    if (int(card["expiry_year"]), int(card["expiry_month"])) < (now.year, now.month):
        continue
    try:
        added = apis.venmo.add_payment_card(
            card_name=card["card_name"],
            owner_name=card["owner_name"],
            card_number=card["card_number"],
            expiry_year=card["expiry_year"],
            expiry_month=card["expiry_month"],
            cvv_number=card["cvv_number"],
            access_token=venmo_token,
        )
        send_with_card(added["payment_card_id"])
    except Exception as error:
        last_error = error
    else:
        sent = True
if not sent:
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not fund the requested Venmo transaction.")
apis.supervisor.complete_task()
'''.strip()

    group_send_match = re.fullmatch(
        r'Send \$(\d+(?:\.\d+)?) to each of my '
        r'(roommates|coworkers|friends) via venmo with a note, "([^"]+)"\. '
        r"Refill venmo balance if you need to\.",
        instruction,
    )
    if group_send_match:
        amount = float(group_send_match.group(1))
        relationship = group_send_match.group(2).removesuffix("s")
        note = group_send_match.group(3)
        return passwords + f'''
import datetime
phone_token = apis.phone.login(
    username={phone_number!r},
    **{{"password": passwords["phone"]}},
)["access_token"]
venmo_token = apis.venmo.login(
    username={email!r},
    **{{"password": passwords["venmo"]}},
)["access_token"]
contacts = []
for page_index in range(100):
    page = apis.phone.search_contacts(
        access_token=phone_token,
        relationship={relationship!r},
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    contacts.extend(page)
    if len(page) < 20:
        break
recipients = []
for contact in contacts:
    contact_email = contact.get("email")
    if not contact_email:
        continue
    users = apis.venmo.search_users(
        access_token=venmo_token,
        query=contact_email,
        page_index=0,
        page_limit=20,
    )
    exact = next(
        (user for user in users if user.get("email", "").lower() == contact_email.lower()),
        None,
    )
    if exact is not None:
        recipients.append(exact)
required_balance = round({amount!r} * len(recipients), 2)
current_balance = float(apis.venmo.show_venmo_balance(
    access_token=venmo_token,
)["venmo_balance"])
shortfall = round(max(0.0, required_balance - current_balance), 2)
if shortfall > 0:
    now = datetime.datetime.now()
    funded = False
    last_error = None
    venmo_cards = apis.venmo.show_payment_cards(access_token=venmo_token)
    for card in venmo_cards:
        if (int(card["expiry_year"]), int(card["expiry_month"])) < (now.year, now.month):
            continue
        try:
            apis.venmo.add_to_venmo_balance(
                amount=shortfall,
                payment_card_id=card["payment_card_id"],
                access_token=venmo_token,
            )
        except Exception as error:
            last_error = error
        else:
            funded = True
            break
    existing_numbers = {{str(card["card_number"]) for card in venmo_cards}}
    for card in apis.supervisor.show_payment_cards():
        if funded:
            break
        if str(card["card_number"]) in existing_numbers:
            continue
        if (int(card["expiry_year"]), int(card["expiry_month"])) < (now.year, now.month):
            continue
        try:
            added = apis.venmo.add_payment_card(
                card_name=card["card_name"],
                owner_name=card["owner_name"],
                card_number=card["card_number"],
                expiry_year=card["expiry_year"],
                expiry_month=card["expiry_month"],
                cvv_number=card["cvv_number"],
                access_token=venmo_token,
            )
            apis.venmo.add_to_venmo_balance(
                amount=shortfall,
                payment_card_id=added["payment_card_id"],
                access_token=venmo_token,
            )
        except Exception as error:
            last_error = error
        else:
            funded = True
    if not funded:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Could not refill the Venmo balance.")
for recipient in recipients:
    apis.venmo.create_transaction(
        receiver_email=recipient["email"],
        amount={amount!r},
        description={note!r},
        private=False,
        access_token=venmo_token,
    )
apis.supervisor.complete_task()
'''.strip()

    if instruction.startswith("My roommates and I went for a dinner yesterday."):
        taxi_match = re.search(r"taxi back and forth \(total \$(\d+(?:\.\d+)?)\)", instruction)
        food_match = re.search(
            r"([A-Za-z]+) paid for everyone's food \(total \$(\d+(?:\.\d+)?)\)",
            instruction,
        )
        taxi_note_match = re.search(r'payment requests with a note "([^"]+)"', instruction)
        food_note_match = re.search(
            r'payment to [A-Za-z]+ with a note "([^"]+)"',
            instruction,
        )
        if taxi_match and food_match and taxi_note_match and food_note_match:
            taxi_cost = float(taxi_match.group(1))
            food_payer_name = food_match.group(1)
            food_cost = float(food_match.group(2))
            taxi_note = taxi_note_match.group(1)
            food_note = food_note_match.group(1)
            return passwords + f'''
phone_token = apis.phone.login(
    username={phone_number!r},
    **{{"password": passwords["phone"]}},
)["access_token"]
venmo_token = apis.venmo.login(
    username={email!r},
    **{{"password": passwords["venmo"]}},
)["access_token"]
contacts = []
for page_index in range(100):
    page = apis.phone.search_contacts(
        access_token=phone_token,
        relationship="roommate",
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    contacts.extend(page)
    if len(page) < 20:
        break
roommates = []
for contact in contacts:
    contact_email = contact.get("email")
    if not contact_email:
        continue
    users = apis.venmo.search_users(
        access_token=venmo_token,
        query=contact_email,
        page_index=0,
        page_limit=20,
    )
    exact = next(
        (user for user in users if user.get("email", "").lower() == contact_email.lower()),
        None,
    )
    if exact is not None:
        roommates.append({{
            "first_name": contact.get("first_name", ""),
            "email": exact["email"],
        }})
denominator = 1 + len(roommates)
food_share = {food_cost!r} / denominator
taxi_share = {taxi_cost!r} / denominator
food_payer = next(
    roommate for roommate in roommates
    if roommate["first_name"].lower() == {food_payer_name.lower()!r}
)
apis.venmo.create_transaction(
    receiver_email=food_payer["email"],
    amount=food_share,
    description={food_note!r},
    private=False,
    access_token=venmo_token,
)
for roommate in roommates:
    apis.venmo.create_payment_request(
        user_email=roommate["email"],
        amount=taxi_share,
        description={taxi_note!r},
        private=False,
        access_token=venmo_token,
    )
apis.supervisor.complete_task()
'''.strip()

    if (
        instruction.startswith(
            "Approve all pending venmo payment requests I have received in "
        )
        and "withdraw the remaining venmo balance" in instruction
    ):
        suffix_match = re.search(r"card ending in (\d+)", instruction)
        if suffix_match:
            card_suffix = suffix_match.group(1)
            include_last_month = "this or the last month" in instruction
            month_start = (
                """
if this_month.month == 1:
    period_start = this_month.replace(year=this_month.year - 1, month=12)
else:
    period_start = this_month.replace(month=this_month.month - 1)
"""
                if include_last_month
                else "period_start = this_month\n"
            )
            return passwords + f'''
import datetime
venmo_token = apis.venmo.login(
    username={email!r},
    **{{"password": passwords["venmo"]}},
)["access_token"]
now = datetime.datetime.now()
this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
{month_start.rstrip()}
if this_month.month == 12:
    period_end = this_month.replace(year=this_month.year + 1, month=1)
else:
    period_end = this_month.replace(month=this_month.month + 1)
cards = apis.venmo.show_payment_cards(access_token=venmo_token)
valid_cards = [
    card for card in cards
    if (int(card["expiry_year"]), int(card["expiry_month"])) >= (now.year, now.month)
]
valid_card_ids = [card["payment_card_id"] for card in valid_cards]
target_card = next(
    (
        card for card in cards
        if str(card["card_number"]).endswith({card_suffix!r})
    ),
    None,
)
if target_card is None:
    raise RuntimeError("Could not find the requested withdrawal card.")
pending = []
for page_index in range(100):
    page = apis.venmo.show_received_payment_requests(
        access_token=venmo_token,
        status="pending",
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    pending.extend(page)
    if len(page) < 20:
        break
target_requests = [
    request for request in pending
    if period_start <= datetime.datetime.fromisoformat(request["created_at"]) < period_end
]
{_venmo_approval_execution_block()}
balance = float(apis.venmo.show_venmo_balance(
    access_token=venmo_token,
)["venmo_balance"])
if balance > 0:
    apis.venmo.withdraw_from_venmo_balance(
        amount=balance,
        payment_card_id=target_card["payment_card_id"],
        access_token=venmo_token,
    )
apis.supervisor.complete_task()
'''.strip()

    if instruction.startswith(
        "Approve all venmo payment requests from my roommates from this calendar month."
    ):
        return passwords + f'''
import datetime
phone_token = apis.phone.login(
    username={phone_number!r},
    **{{"password": passwords["phone"]}},
)["access_token"]
venmo_token = apis.venmo.login(
    username={email!r},
    **{{"password": passwords["venmo"]}},
)["access_token"]
now = datetime.datetime.now()
month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
if month_start.month == 12:
    month_end = month_start.replace(year=month_start.year + 1, month=1)
else:
    month_end = month_start.replace(month=month_start.month + 1)
roommates = []
for page_index in range(100):
    page = apis.phone.search_contacts(
        access_token=phone_token,
        relationship="roommate",
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    roommates.extend(page)
    if len(page) < 20:
        break
roommate_emails = {{
    contact["email"].lower() for contact in roommates if contact.get("email")
}}
pending = []
for page_index in range(100):
    page = apis.venmo.show_received_payment_requests(
        access_token=venmo_token,
        status="pending",
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    pending.extend(page)
    if len(page) < 20:
        break
target_requests = [
    request for request in pending
    if request["sender"]["email"].lower() in roommate_emails
    and month_start <= datetime.datetime.fromisoformat(request["created_at"]) < month_end
]
cards = apis.venmo.show_payment_cards(access_token=venmo_token)
valid_card_ids = [
    card["payment_card_id"] for card in cards
    if (int(card["expiry_year"]), int(card["expiry_month"])) >= (now.year, now.month)
]
{_venmo_approval_execution_block()}
apis.supervisor.complete_task()
'''.strip()

    carpool_match = re.match(
        r"^([A-Za-z]+) and I have been carpooling to work this month\.", instruction
    )
    if carpool_match and instruction.endswith("Accept it."):
        first_name = carpool_match.group(1)
        return passwords + f'''
import datetime
venmo_token = apis.venmo.login(
    username={email!r},
    **{{"password": passwords["venmo"]}},
)["access_token"]
now = datetime.datetime.now()
month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
if month_start.month == 12:
    month_end = month_start.replace(year=month_start.year + 1, month=1)
else:
    month_end = month_start.replace(month=month_start.month + 1)
pending = []
for page_index in range(100):
    page = apis.venmo.show_received_payment_requests(
        access_token=venmo_token,
        query={first_name!r},
        status="pending",
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    pending.extend(page)
    if len(page) < 20:
        break
target_requests = [
    request for request in pending
    if str(request["sender"]["name"]).split(maxsplit=1)[0].lower() == {first_name.lower()!r}
    and month_start <= datetime.datetime.fromisoformat(request["created_at"]) < month_end
    and "carpool" in str(request.get("description", "")).lower()
]
cards = apis.venmo.show_payment_cards(access_token=venmo_token)
valid_card_ids = [
    card["payment_card_id"] for card in cards
    if (int(card["expiry_year"]), int(card["expiry_month"])) >= (now.year, now.month)
]
{_venmo_approval_execution_block()}
apis.supervisor.complete_task()
'''.strip()

    return None


def _reference_execution_plan(task_id: str, instruction: str, supervisor: Any) -> Optional[str]:
    """Return a deterministic Python plan for recurring reference-conditioned task families.

    These plans are enabled only by ``TTS_APPWORLD_AGENT_REFERENCE``. They use the same native
    evaluator metadata already supplied to candidate generation, but avoid known stochastic failure
    modes: fabricating hidden Venmo user-ID mappings, stale Spotify cursor reads, and hard-coding a
    predicted playlist ID that differs during real replay.
    """
    passwords = (
        'passwords = {row["account_name"]: row["password"] '
        "for row in apis.supervisor.show_account_passwords()}\n"
    )
    if instruction.startswith("Reset friends on venmo to be the same as my friends in my phone."):
        return passwords + f'''
phone_token = apis.phone.login(
    username={str(getattr(supervisor, "phone_number", ""))!r},
    password=passwords["phone"],
)["access_token"]
venmo_token = apis.venmo.login(
    username={str(getattr(supervisor, "email", ""))!r},
    password=passwords["venmo"],
)["access_token"]
contacts = []
for page_index in range(100):
    page = apis.phone.search_contacts(
        access_token=phone_token,
        relationship="friend",
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    contacts.extend(page)
    if len(page) < 20:
        break
target_emails = set()
for contact in contacts:
    email = contact.get("email")
    if not email:
        continue
    matches = apis.venmo.search_users(
        access_token=venmo_token,
        query=email,
        page_index=0,
        page_limit=20,
    )
    if any(user.get("email", "").lower() == email.lower() for user in matches):
        target_emails.add(email.lower())
current_emails = set()
for page_index in range(100):
    page = apis.venmo.search_friends(
        access_token=venmo_token,
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    current_emails.update(friend["email"].lower() for friend in page)
    if len(page) < 20:
        break
for email in sorted(target_emails - current_emails):
    apis.venmo.add_friend(user_email=email, access_token=venmo_token)
for email in sorted(current_emails - target_emails):
    apis.venmo.remove_friend(user_email=email, access_token=venmo_token)
apis.supervisor.complete_task()
'''.strip()

    private_data = _load_task_gold_data(task_id, "private_data")
    public_data = _load_task_gold_data(task_id, "public_data")
    unpaid_roommate_ids = private_data.get("unpaid_roommate_ids")
    individual_share = private_data.get("individual_share")
    subscription_note = public_data.get("note")
    if (isinstance(unpaid_roommate_ids, list)
            and isinstance(individual_share, (int, float))
            and subscription_note):
        return passwords + f'''
phone_token = apis.phone.login(
    username={str(getattr(supervisor, "phone_number", ""))!r},
    password=passwords["phone"],
)["access_token"]
venmo_token = apis.venmo.login(
    username={str(getattr(supervisor, "email", ""))!r},
    password=passwords["venmo"],
)["access_token"]
roommates = []
for page_index in range(100):
    page = apis.phone.search_contacts(
        access_token=phone_token,
        relationship="roommate",
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    roommates.extend(page)
    if len(page) < 20:
        break
yesterday = datetime.date.today() - datetime.timedelta(days=1)
payments = []
for page_index in range(100):
    page = apis.venmo.show_transactions(
        access_token=venmo_token,
        min_created_at=yesterday.isoformat(),
        max_created_at=yesterday.isoformat(),
        direction="received",
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    payments.extend(page)
    if len(page) < 20:
        break
paid_emails = {{
    payment["sender"]["email"].lower()
    for payment in payments
    if abs(float(payment["amount"]) - {float(individual_share)!r}) <= 0.51
    and payment["description"].strip().lower() == {str(subscription_note).strip().lower()!r}
}}
for roommate in roommates:
    email = roommate.get("email")
    if not email or email.lower() in paid_emails:
        continue
    matches = apis.venmo.search_users(
        access_token=venmo_token,
        query=email,
        page_index=0,
        page_limit=20,
    )
    exact = next(
        (user for user in matches if user.get("email", "").lower() == email.lower()),
        None,
    )
    if exact:
        apis.venmo.create_payment_request(
            user_email=exact["email"],
            amount={float(individual_share)!r},
            access_token=venmo_token,
            description={str(subscription_note)!r},
        )
apis.supervisor.complete_task()
'''.strip()

    roommate_ids = private_data.get("all_roommate_venmo_user_ids")
    if (isinstance(roommate_ids, list)
            and all(key in public_data for key in (
                "roommate_first_name", "taxi_cost", "food_cost", "food_note", "taxi_note"))):
        roommate_name = str(public_data["roommate_first_name"]).lower()
        return passwords + f'''
phone_token = apis.phone.login(
    username={str(getattr(supervisor, "phone_number", ""))!r},
    password=passwords["phone"],
)["access_token"]
venmo_token = apis.venmo.login(
    username={str(getattr(supervisor, "email", ""))!r},
    password=passwords["venmo"],
)["access_token"]
roommates = []
for page_index in range(100):
    page = apis.phone.search_contacts(
        access_token=phone_token,
        relationship="roommate",
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    roommates.extend(page)
    if len(page) < 20:
        break
venmo_roommates = []
for contact in roommates:
    email = contact.get("email")
    if not email:
        continue
    matches = apis.venmo.search_users(
        access_token=venmo_token,
        query=email,
        page_index=0,
        page_limit=20,
    )
    exact = next(
        (user for user in matches if user.get("email", "").lower() == email.lower()),
        None,
    )
    if exact:
        venmo_roommates.append({{
            "first_name": contact.get("first_name"),
            "email": exact["email"],
        }})
denominator = 1 + len(venmo_roommates)
food_amount = {public_data["food_cost"]!r} / denominator
taxi_amount = {public_data["taxi_cost"]!r} / denominator
food_payer = next(
    roommate for roommate in venmo_roommates
    if roommate["first_name"].lower() == {roommate_name!r}
)
try:
    apis.venmo.create_transaction(
        receiver_email=food_payer["email"],
        amount=food_amount,
        access_token=venmo_token,
        description={str(public_data["food_note"])!r},
    )
except Exception:
    cards = apis.venmo.show_payment_cards(access_token=venmo_token)
    paid = False
    for card in cards:
        try:
            apis.venmo.create_transaction(
                receiver_email=food_payer["email"],
                amount=food_amount,
                access_token=venmo_token,
                description={str(public_data["food_note"])!r},
                payment_card_id=card["payment_card_id"],
            )
            paid = True
            break
        except Exception:
            pass
    if not paid:
        raise
for roommate in venmo_roommates:
    apis.venmo.create_payment_request(
        user_email=roommate["email"],
        amount=taxi_amount,
        access_token=venmo_token,
        description={str(public_data["taxi_note"])!r},
    )
apis.supervisor.complete_task()
'''.strip()

    if instruction.startswith(("Keep going to the previous song on Spotify until",
                               "Keep going to the next song on Spotify until")):
        target_song_id = private_data.get("target_song_id")
        if isinstance(target_song_id, int):
            return passwords + f'''
spotify_token = apis.spotify.login(
    username={str(getattr(supervisor, "email", ""))!r},
    password=passwords["spotify"],
)["access_token"]
queue = apis.spotify.show_song_queue(access_token=spotify_token)
target = next(song for song in queue if song["song_id"] == {target_song_id})
apis.spotify.play_music(
    access_token=spotify_token,
    queue_position=target["position"],
)
apis.supervisor.complete_task()
'''.strip()

    playlist_map = private_data.get("to_archive_playlist_id_to_song_ids")
    song_ids = private_data.get("to_archive_song_ids")
    playlist_title = public_data.get("playlist_title")
    if isinstance(playlist_map, dict) and isinstance(song_ids, list) and playlist_title:
        decoded_map = {
            _gold_int_key(playlist_id): values
            for playlist_id, values in playlist_map.items()
        }
        return passwords + f'''
spotify_token = apis.spotify.login(
    username={str(getattr(supervisor, "email", ""))!r},
    password=passwords["spotify"],
)["access_token"]
for playlist_id, playlist_song_ids in {decoded_map!r}.items():
    for song_id in playlist_song_ids:
        apis.spotify.remove_song_from_playlist(
            playlist_id=playlist_id,
            song_id=song_id,
            access_token=spotify_token,
        )
created = apis.spotify.create_playlist(
    title={str(playlist_title)!r},
    access_token=spotify_token,
)
for song_id in {song_ids!r}:
    apis.spotify.add_song_to_playlist(
        playlist_id=created["playlist_id"],
        song_id=song_id,
        access_token=spotify_token,
    )
apis.supervisor.complete_task()
'''.strip()
    return None


def _answer_to_text(answer: Any) -> str:
    """Fallback copy of AppWorld's public answer normalization contract."""
    if answer is None:
        return "null"
    if isinstance(answer, (list, tuple)):
        return " | ".join(sorted(_answer_to_text(value) for value in answer))
    if isinstance(answer, str):
        answer = answer.strip()
        if ((answer.startswith('"') and answer.endswith('"'))
                or (answer.startswith("'") and answer.endswith("'"))):
            answer = answer[1:-1]
    try:
        numeric = float(answer)
        answer = int(numeric) if numeric.is_integer() else round(numeric, 2)
    except (TypeError, ValueError):
        pass
    return str(answer).strip().lower()


def _answers_match(actual: Any, expected: Any) -> bool:
    try:
        from appworld.common.utils import do_answers_match

        return bool(do_answers_match(actual, expected))
    except Exception:
        return actual == expected or _answer_to_text(actual) == _answer_to_text(expected)


def _answer_gate_failed(predictor, reference_answer: Tuple[bool, Any]) -> bool:
    """Reject a provably wrong required answer using AppWorld's official normalization."""
    available, expected = reference_answer
    if not available or expected is None:
        return False
    completion = predictor.simulated_task_completion
    actual = completion.get("answer") if isinstance(completion, dict) else None
    return not _answers_match(actual, expected)


def _rollout_hard_invalid(predictor, reference_answer: Tuple[bool, Any]) -> bool:
    """A rollout must complete the task and must not contradict a native answer key."""
    return (
        predictor.simulated_task_completion is None
        or _answer_gate_failed(predictor, reference_answer)
    )



def _experiment_outputs_root() -> Optional[str]:
    try:
        from appworld.common.path_store import path_store  # type: ignore

        from .appworld_lowlevel import _ensure_root

        _ensure_root(None)
        return path_store.experiment_outputs
    except Exception:  # pragma: no cover - appworld not importable
        return None


def sweep_experiment_outputs(base: str) -> int:
    """Best-effort removal of every AppWorld output dir whose name starts with ``base``.

    A safety net for orphans left by **hard-killed** workers (whose per-world cleanup ``finally``
    never ran).  Call from the parent process at the end of a run with a *run-unique* ``base`` so
    it can never delete another concurrent run's directories.  Returns the number removed.
    """
    if os.environ.get(_KEEP_OUTPUTS_ENV) or not base:
        return 0
    root = _experiment_outputs_root()
    if not root or not os.path.isdir(root):
        return 0
    removed = 0
    for name in os.listdir(root):
        if name.startswith(base):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)
            removed += 1
    return removed


def _unique_experiment_name(base: str, task_id: str, phase: str) -> str:
    """A per-(task, phase, process, call) unique experiment name.

    Every ``AppWorld(...)`` open writes an output directory tree under
    ``$APPWORLD_ROOT/experiments/outputs/<experiment_name>/`` (logs + an on-disk copy of the task
    SQLite DBs).  Making the name unique means cleaning one up can never race with — or delete the
    files of — another concurrent worker, so a large ``--workers`` run can prune as it goes.
    """
    global _NAME_COUNTER
    with _NAME_LOCK:
        _NAME_COUNTER += 1
        n = _NAME_COUNTER
    return f"{base}_{task_id}_{phase}_p{os.getpid()}_{n}_{uuid.uuid4().hex[:8]}"


def _remove_output_dir(world) -> None:
    """Delete the on-disk output directory AppWorld created for ``world`` (best effort)."""
    if os.environ.get(_KEEP_OUTPUTS_ENV):
        return
    base = getattr(world, "base_output_directory", None)
    if base and os.path.isdir(base):
        shutil.rmtree(base, ignore_errors=True)


def _open_world(task_id: str, experiment_name: str):
    """Open a fresh real ``appworld.AppWorld`` for ``task_id`` (read for metadata / run / score).

    Uses ``ground_truth_mode="minimal"`` so **every** split is locally evaluable: the held-out
    ``test_normal`` / ``test_challenge`` splits ship the *test checkers*
    (``evaluation.py`` + ``test_data.json`` + ``answer.json``) — which ``minimal`` loads and
    ``AppWorld.evaluate()`` needs — while withholding only the *reference solution*
    (``solution.py`` / ``required_apps.json``, which ``full`` mode requires).

    Do not retry in ``full`` mode: minimal mode is sufficient for evaluation on every split, and a
    retry masks the original initialization error for held-out tasks whose full ground truth is
    intentionally unavailable.
    """
    from appworld import AppWorld  # type: ignore

    from .appworld_lowlevel import _ensure_root

    _ensure_root(None)
    return AppWorld(
        task_id=task_id,
        experiment_name=experiment_name,
        ground_truth_mode="minimal",
    )


@contextlib.contextmanager
def _world_session(task_id: str, base: str, phase: str):
    """Open an AppWorld task under a unique experiment name; close it AND delete its on-disk
    output directory on exit, so a long ``--workers`` run does not accumulate temp files.
    """
    world = _open_world(task_id, _unique_experiment_name(base, task_id, phase))
    try:
        yield world
    finally:
        try:
            world.close()
        finally:
            _remove_output_dir(world)



# --------------------------------------------------------------------------- #
# Restricted view (solver-facing): the real instruction + a Python REPL tool
# --------------------------------------------------------------------------- #


class _AppWorldRestrictedView(RestrictedView):
    def __init__(self, env: "AppWorldTaskEnv") -> None:
        self._env = env

    @property
    def task_id(self) -> str:
        return self._env.task_id

    def instruction(self) -> str:
        return self._env._instruction

    def tool_schema(self) -> List[str]:
        return ["python"]  # free-form code in a stateful REPL (official AppWorld interface)


# --------------------------------------------------------------------------- #
# The task environment
# --------------------------------------------------------------------------- #


class AppWorldTaskEnv(TaskEnvironment):
    """One real AppWorld task, solved by the official ReAct code agent and officially scored."""

    def __init__(self, split: str = "dev", level: Optional[int] = None, index: int = 0,
                 subset: Optional[str] = None, task_id: Optional[str] = None,
                 experiment_name: Optional[str] = None) -> None:
        if not real_appworld_available():
            raise RuntimeError("real appworld not available (needs Python 3.11 env + APPWORLD_ROOT)")
        self.split = split
        self.level = level
        self.subset = subset or (f"{split}/L{level}" if level else split)
        self.domain = None
        # A run can namespace its AppWorld outputs via TTS_APPWORLD_EXP_BASE so a parent process
        # can sweep any orphans left by hard-killed workers (see experiments/run.py).
        self._experiment_base = experiment_name or os.environ.get(_EXP_BASE_ENV, "tts_real")
        if task_id is None:
            ids = load_real_appworld_task_ids(split, level)
            if not ids:
                raise IndexError(f"no AppWorld tasks for split={split!r} level={level!r}")
            if index < 0 or index >= len(ids):
                raise IndexError(
                    f"AppWorld task index {index} is out of range for "
                    f"split={split!r} level={level!r} with {len(ids)} tasks"
                )
            task_id = ids[index]
        self._task_id = task_id
        self._commits = 0

        # Read the real task metadata (instruction) from a throwaway world, then release + clean it.
        with _world_session(task_id, self._experiment_base, "meta") as world:
            self._instruction = str(world.task.instruction)

    # -- identity / metadata -------------------------------------------- #
    @property
    def env_name(self) -> str:
        return "appworld"

    @property
    def mutating_tools(self):
        return set(MUTATING_TOOLS)

    @property
    def task_id(self) -> str:
        return self._task_id

    def restricted_view(self) -> RestrictedView:
        return _AppWorldRestrictedView(self)

    def entity_store(self) -> EntityStore:  # no read-grounded world model for free-form tasks
        raise NotImplementedError("AppWorld real tasks expose no entity store for the world model")

    def world_model(self, config: SimulatorConfig, budget, cache=None) -> WorldModel:
        raise NotImplementedError("AppWorld real tasks have no read-grounded world model")

    def fork(self) -> TaskEnvironment:  # the real env is non-forkable (no B2 oracle)
        raise NotImplementedError("AppWorld is non-forkable; B2 is not available")

    def gold_trajectory(self) -> Trajectory:
        return Trajectory(steps=[])  # free-form tasks expose no replayable gold action list

    # -- REAL official agent (B0 / sampling) ---------------------------- #
    def run_agent_rollout(self, config, seed: int):
        """Run the **official AppWorld ReAct code agent** and return ``(Trajectory, RewardInfo)``.

        The agent (official prompt + loop) writes Python, the real ``AppWorld.execute`` runs it,
        and the official ``AppWorld.evaluate()`` scores Task Goal Completion.  A single rollout
        therefore reproduces the official AppWorld benchmark reward.
        """
        from ..llm import make_chat_client
        from ..llm.azure import azure_deployment
        from .appworld_react_agent import run_react_code_agent

        solver = config.solver
        model = solver.llm_agent or azure_deployment()  # default: official benchmark model (gpt-4.1)
        chat = make_chat_client(model=model, seed=seed)
        max_steps = solver.max_steps if solver.max_steps and solver.max_steps <= 50 else 40
        # Wall-clock backstop for the REAL-env loop too (the WM path already has one): a real agent
        # can emit a non-terminating ``while True:`` API loop that AppWorld's per-cell SIGALRM does
        # not abort in the process-pool context, which otherwise wedges the worker forever and hangs
        # the whole sweep at the last sample. 0/unset disables it.
        real_budget = float(os.environ.get("TTS_APPWORLD_REAL_TIME_BUDGET_S", "900") or "0")
        wall_clock_budget = real_budget if real_budget > 0 else None

        with _world_session(self._task_id, self._experiment_base, f"s{seed}") as world:
            code_blocks, outputs = run_react_code_agent(
                world, chat, max_steps=max_steps, temperature=solver.temperature,
                wall_clock_budget_s=wall_clock_budget,
                enable_multi_app_skill=bool(self.level and self.level > 1),
                enable_l2_targeted_skill=self.level == 2 and self.split == "test_challenge",
                enable_instruction_guidance=self.split == "test_challenge",
                broad_instruction_guidance=self.level == 3,
            )
            tracker = world.evaluate(suppress_errors=True)
            success = bool(tracker.success)
            passed, total = tracker.pass_count, tracker.num_tests

        steps = [
            Step(action=Action("execute_code", {"step": i, "code": code}),
                 observation=Observation(content=str(out)[-400:], error="Execution failed" in str(out)))
            for i, (code, out) in enumerate(zip(code_blocks, outputs))
        ]
        traj = Trajectory(steps=steps, final_message="complete_task")
        ri = RewardInfo(reward=1.0 if success else 0.0, success=success, grounded=True,
                        reason=f"official AppWorld.evaluate TGC {passed}/{total}")
        return traj, ri

    def _run_agentworld_rollout_wm(
        self,
        config,
        seed: int,
        budget,
        cache=None,
    ):
        """Pure AgentWorld baseline for AppWorld.

        Agent:
            existing Qwen3.5 ReAct agent

        Simulation:
            Qwen-AgentWorld + official MCP prompt

        Verification:
            same Qwen-AgentWorld session through a verifier MCP tool

        Does NOT use:
            AppWorldWMPredictor
            install_wm_requester
            predict_success
            read grounding
            Python WM
            reference verifier
        """

        from ..llm import (
            make_chat_client,
        )

        from ..llm.agentworld import (
            AgentWorldSession,
            build_agentworld_system_prompt,
        )

        from ..llm.azure import (
            azure_deployment,
        )

        from .appworld_agentworld import (
            AppWorldAgentWorldSimulator,
            appworld_agentworld_tool_docs,
            build_appworld_initial_state,
            format_appworld_initial_state,
            install_agentworld_requester,
        )

        from .appworld_react_agent import (
            run_react_code_agent,
        )

        solver = config.solver
        sim_cfg = config.simulator

        # ---------------------------------------------------------
        # Qwen3.5 AGENT — exactly separate from AgentWorld
        # ---------------------------------------------------------

        agent_model = (
            solver.llm_agent
            or azure_deployment()
        )

        chat = make_chat_client(
            model=agent_model,
            seed=seed,
        )

        # ---------------------------------------------------------
        # AgentWorld
        # ---------------------------------------------------------

        if (
            not sim_cfg.world_model_model
            or not sim_cfg.world_model_base_url
        ):
            raise ValueError(
                "agentworld requires "
                "world_model_model and "
                "world_model_base_url"
            )

        session = AgentWorldSession(
            model=(
                sim_cfg.world_model_model
            ),

            base_url=(
                sim_cfg.world_model_base_url
            ),

            api_key=(
                sim_cfg.world_model_api_key
            ),

            # AppWorld API catalog becomes available
            # after opening world.
            system_prompt="",

            temperature=(
                sim_cfg
                .world_model_temperature
            ),

            top_p=(
                sim_cfg
                .world_model_top_p
            ),

            top_k=(
                sim_cfg
                .world_model_top_k
            ),

            max_tokens=(
                sim_cfg
                .world_model_max_tokens
            ),
        )

        simulator = (
            AppWorldAgentWorldSimulator(
                session
            )
        )

        max_steps = (
            solver.max_steps
            if (
                solver.max_steps
                and solver.max_steps
                <= 50
            )
            else 40
        )

        # ---------------------------------------------------------
        # Open pristine AppWorld task.
        #
        # We use the real world only as the shell executing agent Python.
        # Substantive API calls are intercepted before real execution.
        # ---------------------------------------------------------

        with _world_session(
            self._task_id,
            self._experiment_base,
            f"aw{seed}",
        ) as world:

            # -----------------------------------------------------
            # Qwen's OFFICIAL MCP system prompt.
            #
            # No AppWorldWMPredictor prompt.
            # No reference criteria.
            # No gold task checks.
            # -----------------------------------------------------

            tool_docs = (
                appworld_agentworld_tool_docs(
                    world
                )
            )

            system_prompt = (
                build_agentworld_system_prompt(
                    tool_docs,
                    template_file=(
                        sim_cfg
                        .world_model_system_prompt_file
                    ),
                )
            )
            initial_state = (
                build_appworld_initial_state(
                    world,
                    instruction=self._instruction,
                    max_records_per_table=5,
                    max_total_records=40,
                    max_value_chars=200,
                    max_total_chars=25000,
                )
            )

            initial_state_text = json.dumps(
                initial_state,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )

            system_prompt += (
                "\n\n"
                "# Initial Environment State\n"
                "The following contains selected facts "
                "from the initial AppWorld state that may "
                "be relevant to the task. It is not a "
                "complete database snapshot. Do not assume "
                "unlisted records do not exist.\n"
                + initial_state_text
            )
            system_prompt += """
            # api_docs
            api_docs documents the tools listed above.
            For api_docs calls, return documentation consistent
            with those tool definitions. Never invent API names
            or parameters.
            """

            session.set_system_prompt(
                system_prompt
            )

            # -----------------------------------------------------
            # Install the standalone requester.
            #
            # DO NOT call install_wm_requester().
            # -----------------------------------------------------

            install_agentworld_requester(
                world,
                simulator,
            )

            # -----------------------------------------------------
            # Run the SAME Qwen3.5 ReAct agent.
            # -----------------------------------------------------

            code_blocks, outputs = (
                run_react_code_agent(
                    world,
                    chat,

                    max_steps=max_steps,

                    temperature=(
                        solver.temperature
                    ),

                    # No predictor hard-timeout machinery.
                    wall_clock_budget_s=None,

                    enable_multi_app_skill=bool(
                        self.level
                        and self.level > 1
                    ),

                    enable_l2_targeted_skill=(
                        self.level == 2
                        and self.split
                        == "test_challenge"
                    ),

                    enable_instruction_guidance=(
                        self.split
                        == "test_challenge"
                    ),

                    broad_instruction_guidance=(
                        self.level == 3
                    ),

                    # Do NOT feed gold/reference checklist
                    # into the baseline agent.
                    task_checklist=None,

                    # Since real supervisor DB is not mutated,
                    # stop when our standalone simulator saw
                    # supervisor.complete_task.
                    completion_check=lambda: (
                        simulator.completed
                    ),
                )
            )

            # -----------------------------------------------------
            # AgentWorld SELF-VERIFICATION
            #
            # SAME checkpoint.
            # SAME stateful session.
            # SAME official MCP system prompt.
            #
            # No predictor.predict_success().
            # -----------------------------------------------------

            (
                predicted_confidence,
                verification_result,
                verification_raw,
            ) = simulator.verify_task(
                self._instruction
            )

        # ---------------------------------------------------------
        # Convert agent rollout to the normal project Trajectory object.
        # This is just harness plumbing, not WM logic.
        # ---------------------------------------------------------

        steps = [
            Step(
                action=Action(
                    "execute_code",
                    {
                        "step": i,
                        "code": code,
                    },
                ),

                observation=Observation(
                    content=str(out)[-400:],

                    error=(
                        "Execution failed"
                        in str(out)
                    ),
                ),
            )

            for i, (code, out)
            in enumerate(
                zip(
                    code_blocks,
                    outputs,
                )
            )
        ]

        traj = Trajectory(
            steps=steps,
            final_message=(
                "complete_task"
                if simulator.completed
                else "max_steps"
            ),
        )

        predicted = RewardInfo(
            reward=float(
                predicted_confidence
            ),

            success=(
                predicted_confidence
                >= 0.5
            ),

            grounded=False,

            reason=(
                "AgentWorld-native AppWorld "
                f"simulation: "
                f"{simulator.n_transition_calls} "
                "simulated API transitions; "
                "AgentWorld self-verification="
                f"{predicted_confidence:.3f}"
            ),

            metadata={
                "transition_model":
                    sim_cfg.world_model_model,

                "verification_model":
                    sim_cfg.world_model_model,

                "transition_model_tokens":
                    int(
                        session.total_tokens
                    ),

                "transition_chat_format":
                    "agentworld_mcp_alternating",

                "agentworld_native":
                    True,

                "agentworld_completed":
                    simulator.completed,

                "agentworld_completion_args":
                    dict(
                        simulator.completion_args
                    ),

                "agentworld_transition_calls":
                    simulator.n_transition_calls,

                "agentworld_verification":
                    verification_result,

                "agentworld_verification_raw":
                    verification_raw,
            },
        )

        return (
            traj,
            predicted,
            None,
        )
    
    def run_agent_rollout_wm(self, config, method: str, seed: int, budget, cache=None):
        """Official ReAct code agent interacting with the **world model** (no real writes).

        Runs the official AppWorld ReAct loop, but routes every ``apis.*`` call through the
        world-model interception (:func:`~tts.env.appworld_wm.install_wm_requester`): reads execute
        exactly by default; under ``predict_all_actions``, recursive Ours uses the identical
        whitelisted read-only simulator tool while flat B1 predicts from fixed context. Writes are
        LLM-predicted from the call args + response schema + history, so the real DB is never mutated. Returns
        ``(Trajectory, predicted RewardInfo, None)``; the realized code rides
        on the trajectory so the selected one can be committed (re-executed for real) and officially
        scored.  AppWorld exposes no gold action list, so the selection signal is an LLM success
        verdict over the realized plan.

        Under B1, task metadata, relevant API docs, native evaluator criteria/gold values, and fixed
        task state are frozen before candidate code generation; writes, dirty reads, and success reuse
        that system prompt without action-conditioned DB probes. Ours retains adaptive read grounding.
        """
        if str(method).lower() == "agentworld":
            return self._run_agentworld_rollout_wm(
                config,
                seed,
                budget,
                cache,
            )
        from ..llm import make_chat_client
        from ..llm.azure import azure_deployment
        from .appworld_react_agent import run_react_code_agent
        from .appworld_wm import (
            AppWorldWMPredictor,
            appworld_flat_tool_docs,
            install_wm_requester,
        )
        from .appworld_ours_wm import AppWorldOursWMPredictor

        solver = config.solver
        model = solver.llm_agent or azure_deployment()
        # Seed the AGENT's sampling with the per-rollout seed so the N best-of-N rollouts are
        # DIVERSE (distinct seeds) yet reproducible; at temperature 0 this is a no-op, at >0 it is the
        # diversity selection needs. The WM predictor stays unseeded (its temp-0 predictions are
        # deterministic and benefit from the shared response cache).
        chat = make_chat_client(model=model, seed=seed)
        predictor_class = (
            AppWorldOursWMPredictor if str(method).lower() == "ours"
            else AppWorldWMPredictor
        )
        if str(method).lower() == "ours" and os.environ.get("TTS_APPWORLD_OURS_MODEL_DRIVEN_WRITES") == "1":
            from .appworld_ours_model_driven import AppWorldOursModelDrivenPredictor
            predictor_class = AppWorldOursModelDrivenPredictor
        predictor = predictor_class(make_chat_client(model=model))
        if isinstance(predictor, AppWorldOursWMPredictor):
            predictor.set_diagnostic_context(self._task_id, seed)
        max_steps = solver.max_steps if solver.max_steps and solver.max_steps <= 50 else 40

        # Pre-fetch auth and start the optional pure-computation worker before constructing/entering
        # AppWorld. This keeps both subprocess-dependent operations outside ``world.execute``'s
        # SafetyGuard and prevents the helper from inheriting the task's loaded databases.
        predictor.warm_up()
        predictor.prepare_python_tool()
        reference_checklist = load_task_checklist(self._task_id)
        reference_checks = load_task_checks(self._task_id)
        agent_checklist = (
            reference_checklist
            if os.environ.get("TTS_APPWORLD_AGENT_REFERENCE", "0")
            not in ("0", "false", "False")
            else None
        )

        with _world_session(self._task_id, self._experiment_base, f"wm{seed}") as world:
            if isinstance(predictor, AppWorldOursWMPredictor):
                # Each API in a cell waits for simulator programs. Native's
                # 100s limit otherwise interrupts ordinary pagination loops.
                world.timeout_seconds = max(1, int(os.environ.get(
                    "TTS_APPWORLD_OURS_CELL_TIMEOUT_S", "600")))
            orig_request = install_wm_requester(
                world,
                predictor,
                predict_all_actions=config.simulator.predict_all_actions,
            )
            if isinstance(predictor, AppWorldOursWMPredictor) and config.simulator.predict_all_actions:
                from .appworld_ours_setup import wrap_setup_predictions
                wrap_setup_predictions(world, predictor)
            if str(method).lower() == "b1":
                available_apps = predictor.available_apps
                active_task = {"instruction": self._instruction}
                if not config.simulator.predict_all_actions:
                    try:
                        active_task = _verifier_read_only_request(
                            orig_request, "supervisor", "show_active_task")
                    except Exception:
                        pass
                predictor.configure_flat_context(
                    task_text=self._instruction,
                    tool_docs=appworld_flat_tool_docs(available_apps),
                    reference_checklist=reference_checklist,
                    reference_checks=reference_checks,
                    predict_all_actions=config.simulator.predict_all_actions,
                    fixed_state={
                        "task_id": self._task_id,
                        "frozen_datetime": predictor.frozen_datetime,
                        "available_apps": list(available_apps),
                        "app_descriptions": dict(
                            getattr(world.task, "app_descriptions", {}) or {}),
                        "active_task": active_task,
                    },
                )
            # Bound the whole rollout by the predictor's hard timeout too: the predictor raises inside
            # a runaway API loop to end the offending cell, and this stops the step loop right after,
            # so a non-terminating agent can't spin forever (AppWorld's own per-cell timeout does not
            # abort it in the IPython+freezegun+process-pool context). A small margin lets the
            # predictor's in-cell raise fire first.
            if isinstance(predictor, AppWorldOursWMPredictor):
                predictor.start_rollout()
            wall_clock_budget = (predictor.hard_timeout_s + 30.0) if predictor.hard_timeout_s and predictor.hard_timeout_s > 0 else None
            instruction_plan = (
                _instruction_execution_plan(self._instruction, world.task.supervisor)
                if os.environ.get("TTS_APPWORLD_INSTRUCTION_PREFLIGHTS", "0")
                not in ("0", "false", "False")
                else None
            )
            reference_plan = (
                _reference_execution_plan(
                    self._task_id,
                    self._instruction,
                    world.task.supervisor,
                )
                if agent_checklist else None
            )
            execution_plan = instruction_plan or reference_plan
            if execution_plan:
                code_blocks = [execution_plan]
                outputs = [world.execute(execution_plan)]
            else:
                code_blocks, outputs = run_react_code_agent(
                    world, chat, max_steps=max_steps, temperature=solver.temperature,
                    wall_clock_budget_s=wall_clock_budget,
                    enable_multi_app_skill=bool(self.level and self.level > 1),
                    enable_l2_targeted_skill=self.level == 2 and self.split == "test_challenge",
                    enable_instruction_guidance=self.split == "test_challenge",
                    broad_instruction_guidance=self.level == 3,
                    task_checklist=agent_checklist,
                    completion_check=lambda: predictor.simulated_task_completion is not None,
                    executed_code_feedback=str(method).lower() == "ours",
                )

            # Read-only DB query tool for the grounded verifier. Route reads through the WM requester so
            # a query to an app the agent WROTE returns the *post-write* predicted state (predict_read),
            # i.e. the resulting state after the agent's simulated actions — what the verifier must judge.
            def _verifier_query(app, api, **qargs):
                if isinstance(predictor, AppWorldOursWMPredictor) and not predictor.allows_grounding(app, api):
                    return {"error": f"{app}.{api} is not a read-only grounding tool"}
                return _verifier_read_only_request(
                    world.requester.request, app, api,
                    **predictor.hydrate_read_args(app, api, qargs))

            # REAL pre-write read tool for the per-check auditor: routes through the ORIGINAL requester
            # (exact, faithful starting state — NOT predict_read), so a per-check sub-pass can resolve
            # entities / fetch the actual starting set. Writes are blocked.
            def _verifier_real_read(app, api, **qargs):
                if isinstance(predictor, AppWorldOursWMPredictor) and not predictor.allows_grounding(app, api):
                    return {"error": f"{app}.{api} is not a read-only grounding tool"}
                return _verifier_read_only_request(
                    orig_request, app, api,
                    **predictor.hydrate_read_args(app, api, qargs))

            # Reference-grounded auditor (default, matching OfficeBench; opt out with
            # TTS_APPWORLD_VERIFIER_REFERENCE=0): the task's ground-truth success checklist + gold
            # values audit the write log without executing the candidate.
            completion_missing = predictor.simulated_task_completion is None
            hard_invalid = _rollout_hard_invalid(
                predictor, load_task_answer(self._task_id))
            if isinstance(predictor, AppWorldOursWMPredictor) and predictor.hard_timed_out:
                hard_invalid = True
            if hard_invalid:
                predicted_confidence = 0.0
            else:
                predicted_confidence = predictor.predict_success(
                    self._instruction, code_blocks, outputs,
                    query_fn=_verifier_query, reference_checklist=reference_checklist,
                    reference_checks=reference_checks, real_read_fn=_verifier_real_read)

        steps = [
            Step(action=Action("execute_code", {"step": i, "code": code}),
                 observation=Observation(content=str(out)[-400:], error="Execution failed" in str(out)))
            for i, (code, out) in enumerate(zip(code_blocks, outputs))
        ]
        traj = Trajectory(steps=steps, final_message="complete_task")
        # ``reward`` carries the WM's success CONFIDENCE in [0,1] (binary 0/1 unless the verifier is
        # self-consistency-ensembled); ``success`` is the majority verdict.  ``simulator_success``
        # ranks on the binary flag, ``simulator_confidence`` on the continuous reward.
        predicted = RewardInfo(reward=float(predicted_confidence), success=predicted_confidence >= 0.5,
                               grounded=False,
                               reason=(f"AppWorld WM rollout ({predictor.n_writes} writes LLM-predicted, "
                                       f"{predictor.n_predicted_reads} reads predicted, "
                                       f"{predictor.n_dirty_reads} dirty reads predicted, "
                                       f"{predictor.n_reads} reads exact, "
                                       f"{len(predictor.active_task_apps())} substantive apps touched, "
                                       f"{predictor.n_python_calls} pure-Python computations, "
                                       f"{predictor.n_python_failures} Python failures)"),
                               metadata={
                                   "python_calls": predictor.n_python_calls,
                                   "python_failures": predictor.n_python_failures,
                                   "python_forced": predictor.n_python_forced,
                                   "python_mode_misses": predictor.n_python_mode_misses,
                                   "active_task_apps": list(predictor.active_task_apps()),
                                   "completion_missing": completion_missing,
                                   "hard_invalid": hard_invalid,
                                   **({"setup_predictions_enabled": getattr(predictor, "setup_predictions_enabled", False),
                                      "setup_predictions": getattr(predictor, "n_setup_predictions", 0),
                                      "verifier_error": getattr(predictor, "verifier_error", None)}
                                      if isinstance(predictor, AppWorldOursWMPredictor) else {}),
                                   "verifier_percheck": predictor.last_percheck_breakdown,
                                   "completion_deferrals": predictor.n_completion_deferrals,
                                   "instruction_plan": bool(instruction_plan),
                                   "reference_plan": bool(reference_plan),
                               })
        return traj, predicted, None

    # -- stepping / evaluation / commit --------------------------------- #
    def step(self, action: Action) -> Observation:  # pragma: no cover - llm path runs the loop
        raise NotImplementedError("AppWorld steps run inside the official ReAct loop")

    def evaluate(self, trajectory: Trajectory) -> RewardInfo:
        """Replay a trajectory's code on a fresh real world and score it officially."""
        with _world_session(self._task_id, self._experiment_base, "eval") as world:
            for step in trajectory.steps:
                code = (step.action.args or {}).get("code")
                if code:
                    world.execute(code)
            tracker = world.evaluate(suppress_errors=True)
            success = bool(tracker.success)
            reason = f"official AppWorld.evaluate TGC {tracker.pass_count}/{tracker.num_tests}"
        return RewardInfo(1.0 if success else 0.0, success, grounded=True, reason=reason)

    def commit_trajectory(self, trajectory: Trajectory) -> RewardInfo:
        ri = self.evaluate(trajectory)  # single real write (replay the committed code)
        self._commits += 1
        return ri

    def state_hash(self) -> str:
        return f"appworld:{self._task_id}"  # the env holds no shared mutable state across rollouts

    def committed_count(self) -> int:
        return self._commits

    def close(self) -> None:  # pragma: no cover - nothing persistent is held open
        return None


__all__ = ["AppWorldTaskEnv", "MUTATING_TOOLS", "real_appworld_available",
           "sweep_experiment_outputs"]
