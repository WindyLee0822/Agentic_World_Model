"""Standalone Qwen-AgentWorld simulator for AppWorld.

This baseline intentionally does NOT use AppWorldWMPredictor,
install_wm_requester, read grounding, recursive WM, Python WM,
or the project's success verifier.

Qwen-AgentWorld itself:
1. simulates every substantive AppWorld API transition;
2. maintains the simulated state through its native MCP history;
3. produces the final trajectory-success score.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict

from ..llm.agentworld import AgentWorldSession


_HTTP_ERROR_RE = re.compile(
    r"Response\s+status\s+code\s+is\s+(\d+)\s*:\s*(.*)\Z",
    re.IGNORECASE | re.DOTALL,
)

_FENCED_RE = re.compile(
    r"^```(?:json|python|text)?\s*(.*?)\s*```$",
    re.IGNORECASE | re.DOTALL,
)


import json
import re
from typing import Any


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in",
    "on", "at", "with", "from", "by", "my", "me", "i",
    "please", "find", "show", "get", "make", "create",
    "update", "delete", "remove", "add",
    "all", "have", "has", "had", "that", "then", "not",
}


import json
from typing import Any, Dict, Optional


def _public_api_docs(
    app_name: str,
) -> dict[str, dict[str, Any]]:
    from appworld.requester import (
        _get_api_name_to_doc,
    )

    if not isinstance(app_name, str):
        return {}

    app_name = app_name.strip()

    if not app_name:
        return {}

    try:
        docs = _get_api_name_to_doc(
            app_name,
            include_private_apis=False,
        )
    except Exception:
        return {}

    if not isinstance(docs, dict):
        return {}

    return {
        str(name): doc
        for name, doc in docs.items()
        if isinstance(doc, dict)
    }

def _static_reference_for_action(
    world,
    app_name: str,
    api_name: str,
    arguments,
) -> str:
    """Return only the exact static schema for the current API.

    This does NOT execute the API and does NOT reveal any dynamic
    environment observation. It only gives AgentWorld the precise
    definition/schema of the API it is currently simulating.
    """

    app_name = str(
        app_name or ""
    ).strip()

    api_name = str(
        api_name or ""
    ).strip()

    if not app_name or not api_name:
        return ""

    current_docs = (
        _public_api_docs(
            app_name
        )
    )

    current_doc = (
        current_docs.get(
            api_name
        )
    )

    if not isinstance(
        current_doc,
        dict,
    ):
        return ""

    return (
        "Exact static schema for the current AppWorld API. "
        "Use this only to determine the API semantics and "
        "the required observation format. "
        "This is NOT an environment observation.\n"
        + _agentworld_tool_definition(
            api_name,
            current_doc,
        )
    )


def _instruction_terms(
    instruction: str,
) -> set[str]:
    words = re.findall(
        r"[A-Za-z0-9@._+-]+",
        instruction.lower(),
    )

    return {
        word if "@" in word else word.strip("._+-")
        for word in words
        if len(word.strip("._+-")) >= 3
        and word.strip("._+-") not in _STOPWORDS
    }


def _record_to_dict(
    record,
) -> dict:
    try:
        return record.to_dict(
            keep_computed=False
        )
    except Exception:
        try:
            return record.dict()
        except Exception:
            return {
                "value": str(record)
            }


def _compact_value(
    value: Any,
    max_chars: int = 200,
):
    if hasattr(value, "isoformat"):
        try:
            value = value.isoformat()
        except Exception:
            pass

    if isinstance(value, str):
        value = " ".join(
            value.split()
        )

        if len(value) > max_chars:
            value = value[:max_chars] + "…"

    return value


def expand_task_relevant_product_variants(
    world,
    app_name: str,
    app_state: dict,
    instruction: str,
    *,
    max_variants: int = 60,
    max_value_chars: int = 4_000,
) -> int:
    """Complete product families needed for a fixed-snapshot variant choice.

    AppWorld stores each size/color combination as a separate ``Product`` row.
    A normal foreign-key closure therefore reaches the purchased product but not
    its sibling variants.  When the task explicitly asks for a size/color/variant
    choice, add siblings of the task-named product and of products in the most
    recent selected order.  This only reads the pristine pre-action model snapshot;
    it never executes (or inspects the result of) a candidate action.

    The caller decides whether to enable this augmentation.  In particular, the
    strict Ours+Qwen persistent-WM path calls it behind its dedicated opt-in flag,
    so the AgentWorld baseline and other datasets/methods retain their old packet.
    """

    instruction_lower = str(instruction or "").lower()
    if not any(
        cue in instruction_lower
        for cue in (
            "size", "color", "colour", "variant", "variation",
            "same item", "same product", "next available",
        )
    ):
        return 0

    product_table = next(
        (name for name in app_state if str(name).lower() == "product"), None
    )
    if product_table is None:
        return 0

    existing_products = list(app_state.get(product_table) or [])
    if not existing_products:
        return 0

    try:
        app_models = getattr(world.models, app_name)
        product_model = getattr(app_models, product_table)
        source_records = product_model.all()
    except Exception:
        return 0

    source_by_id = {}
    for record in source_records:
        row = _record_to_dict(record)
        if row.get("id") is not None:
            source_by_id[row["id"]] = row

    anchor_ids = set()

    # Prefer a product whose name is explicitly present in the task.  Generic
    # catalog words are ignored so "flip-flops" alone cannot select every brand.
    generic_product_terms = {
        "item", "product", "same", "next", "available", "size", "color",
        "colour", "combination", "variant", "variation", "buy", "purchase",
        "bought", "return", "initiate", "replace", "replacement", "order",
        "recent", "latest", "last", "most", "amazon", "prefer", "otherwise",
        "small", "medium", "large", "larger", "extra-large", "bit", "too",
    }
    instruction_terms = _instruction_terms(instruction) - generic_product_terms
    name_scores = []
    for row in existing_products:
        name_terms = _instruction_terms(str(row.get("name") or ""))
        score = len(instruction_terms & name_terms)
        name_scores.append((score, row.get("id")))
    best_name_score = max((score for score, _ in name_scores), default=0)
    if best_name_score >= 2:
        anchor_ids.update(
            product_id
            for score, product_id in name_scores
            if score == best_name_score and product_id is not None
        )

    # "Most recent/latest/last order" tasks also have an unambiguous relational
    # anchor even when the product is not named in the instruction.
    if re.search(r"\b(?:most recent|latest|last)\b", instruction_lower):
        order_table = next(
            (name for name in app_state if str(name).lower() == "order"), None
        )
        item_table = next(
            (name for name in app_state if str(name).lower() == "orderitem"), None
        )
        orders = list(app_state.get(order_table) or []) if order_table else []
        items = list(app_state.get(item_table) or []) if item_table else []
        if orders and items:
            def order_recency(row):
                date_values = [
                    str(value)
                    for field, value in row.items()
                    if value is not None
                    and any(token in str(field).lower() for token in ("date", "_at"))
                ]
                return (max(date_values, default=""), row.get("id", -1))

            # Resolve "last flip-flops" (or another category/name) through
            # Order -> OrderItem -> Product before applying recency.  Selecting
            # the latest order overall is wrong when newer unrelated purchases
            # exist, which is common in AppWorld's dense Amazon histories.
            product_terms = instruction_terms
            matching_product_ids = {
                product_id
                for product_id, product_row in source_by_id.items()
                if product_terms
                and any(
                    term in (
                        str(product_row.get("name") or "")
                        + " "
                        + str(product_row.get("product_type") or "")
                    ).lower()
                    for term in product_terms
                )
            }
            matching_order_ids = {
                row.get("order_id")
                for row in items
                if row.get("product_id") in matching_product_ids
                and row.get("order_id") is not None
            }
            candidate_orders = [
                row for row in orders if row.get("id") in matching_order_ids
            ] or orders
            recent_order_id = max(candidate_orders, key=order_recency).get("id")
            anchor_ids.update(
                row.get("product_id")
                for row in items
                if row.get("order_id") == recent_order_id
                and row.get("product_id") is not None
            )

    anchors = [source_by_id[value] for value in anchor_ids if value in source_by_id]
    family_keys = {
        (str(row.get("name") or "").strip().lower(), row.get("seller_id"))
        for row in anchors
        if row.get("has_variations")
        and str(row.get("name") or "").strip()
    }
    if not family_keys:
        return 0

    siblings = [
        row
        for row in source_by_id.values()
        if (str(row.get("name") or "").strip().lower(), row.get("seller_id"))
        in family_keys
    ]
    siblings.sort(key=lambda row: (str(row.get("name") or ""), row.get("id", -1)))
    siblings = siblings[:max_variants]

    family_summaries = []
    for family_name, family_seller_id in sorted(family_keys):
        family_rows = [
            row for row in siblings
            if str(row.get("name") or "").strip().lower() == family_name
            and row.get("seller_id") == family_seller_id
        ]
        family_summaries.append({
            "name": family_rows[0].get("name") if family_rows else family_name,
            "seller_id": family_seller_id,
            "variants": [
                {
                    "product_id": row.get("id"),
                    "relative_size": row.get("relative_size"),
                    "color": row.get("color"),
                    "inventory_quantity": row.get("inventory_quantity"),
                }
                for row in family_rows
            ],
        })

    # Keep full rows already selected by the normal relational packet, but do
    # not duplicate every sibling's descriptions/dimensions/reviews.  The compact
    # family index below preserves every fact needed to select a purchasable
    # variant while avoiding context overflow on dense product catalogs.
    anchor_id_set = {row.get("id") for row in anchors}
    app_state[product_table] = sorted(
        existing_products,
        key=lambda row: row.get("id") not in anchor_id_set,
    )
    # This is a derived compressed index over pristine Product rows, not an
    # additional source of state.  It makes omitted same-name sibling rows
    # explicit to a language-model simulator that otherwise invents variant IDs.
    app_state["ProductVariantFamily"] = family_summaries
    return len([row for row in siblings if row.get("id") not in {
        existing.get("id") for existing in existing_products
    }])


def expand_file_referenced_spotify_songs(
    world,
    merged_state: dict,
    *,
    max_songs: int = 60,
    max_value_chars: int = 4_000,
) -> int:
    """Bridge task-relevant file contents to frozen Spotify Song rows.

    Playlist foreign keys recover songs already in playlists, but omit a title
    named by an input file when it is not currently in one.  Exact title matches
    are copied from the same pre-action model snapshot; no candidate action or
    API result is executed or inspected.
    """
    apps = (merged_state or {}).get("apps") or {}
    spotify_state = apps.get("spotify")
    file_state = apps.get("file_system")
    if not isinstance(spotify_state, dict) or not isinstance(file_state, dict):
        return 0

    content = "\n".join(
        str(row.get("content") or "")
        for row in (file_state.get("File") or [])
        if isinstance(row, dict)
    ).casefold()
    if not content:
        return 0

    existing_ids = {
        row.get("id")
        for row in (spotify_state.get("Song") or [])
        if isinstance(row, dict)
    }
    try:
        source_rows = world.models.spotify.Song.all()
    except Exception:
        return 0

    added = 0
    for record in source_rows:
        row = _record_to_dict(record)
        title = str(row.get("title") or "").strip()
        if not title or title.casefold() not in content or row.get("id") in existing_ids:
            continue
        spotify_state.setdefault("Song", []).append({
            str(key): _compact_value(value, max_value_chars)
            for key, value in row.items()
        } | {
            "_retrieval_provenance": (
                "file_reference_catalog_match_only_not_playlist_membership"
            )
        })
        existing_ids.add(row.get("id"))
        added += 1
        if added >= max_songs:
            break
    return added

def build_appworld_initial_state(
    world,
    instruction: str,
    *,
    extra_terms=None,
    preferred_apps=None,
    max_records_per_table: int = 5,
    max_total_records: int = 40,
    max_value_chars: int = 200,
    max_total_chars: int = 25000,
    relational_closure: bool = False,
) -> dict:
    """
    Build a SMALL task-relevant initial state for AgentWorld.

    Retrieval is conditioned on:
      1. the original task instruction;
      2. terms from the retrieved task-relevant APIs;
      3. the apps selected by API retrieval.

    Important:
      - no agent action is executed here;
      - later substantive actions are still simulated;
      - this is only an initial grounding packet.
    """

    task_terms = _instruction_terms(
        instruction
    )
    instruction_lower = str(instruction or "").lower()
    terms = set(task_terms)

    # Add terms derived from the task-specific API retrieval.
    for term in (
        extra_terms
        or []
    ):
        text = str(
            term
        ).lower().strip()

        if (
            len(text) >= 3
            and text
            not in _STOPWORDS
        ):
            terms.add(
                text
            )

    state = {
        "datetime": None,
        "supervisor": {},
        "apps": {},
    }

    # ---------------------------------------------------------
    # Time
    # ---------------------------------------------------------

    try:
        state[
            "datetime"
        ] = str(
            world.task.datetime
        )
    except Exception:
        pass
    current_date_text = str(state.get("datetime") or "")[:10]
    current_month_text = current_date_text[:7]
    month_sensitive = any(
        phrase in instruction_lower
        for phrase in ("this month", "calendar month", "month so far")
    )
    day_sensitive = any(
        phrase in instruction_lower for phrase in ("today", "current date")
    )
    recent_sensitive = any(
        term in instruction_lower for term in ("last ", "latest", "recent", "newest")
    )
    broad_coverage = bool(re.search(r"\b(?:all|every)\b", instruction_lower)) or any(
        phrase in instruction_lower
        for phrase in ("how much", "total", "this month", "calendar month")
    )

    # ---------------------------------------------------------
    # Supervisor identity
    # ---------------------------------------------------------

    supervisor_values = set()
    supervisor_identity_values = set()

    try:
        supervisor = (
            world.task.supervisor
        )

        if isinstance(
            supervisor,
            dict,
        ):
            supervisor_dict = dict(
                supervisor
            )

        else:
            try:
                supervisor_dict = (
                    supervisor.dict()
                )

            except Exception:
                supervisor_dict = {
                    "value":
                        str(
                            supervisor
                        )
                }

        state[
            "supervisor"
        ] = {
            str(k):
                _compact_value(
                    v,
                    max_value_chars,
                )
            for k, v
            in supervisor_dict.items()
        }

        for value in (
            supervisor_dict.values()
        ):

            if isinstance(
                value,
                (str, int),
            ):

                text = str(
                    value
                ).lower().strip()

                if text:
                    supervisor_values.add(
                        text
                    )

        for key in (
            "email",
            "phone_number",
            "phone",
        ):
            value = supervisor_dict.get(key)
            if isinstance(value, (str, int)):
                text = str(value).lower().strip()
                if text:
                    supervisor_identity_values.add(text)

    except Exception:
        pass

    # ---------------------------------------------------------
    # Apps available in this task
    # ---------------------------------------------------------

    try:
        all_app_names = list(
            (
                world.task
                .app_descriptions
                or {}
            ).keys()
        )

    except Exception:
        all_app_names = []

    all_app_names = [
        str(app)
        for app in all_app_names
        if str(app) not in {
            "api_docs",
            "admin",
        }
    ]

    # ---------------------------------------------------------
    # Put retrieved/relevant apps first so they consume the
    # limited initial-state budget before irrelevant apps.
    # ---------------------------------------------------------

    app_names = []

    for app in (
        preferred_apps
        or []
    ):

        app = str(
            app
        )

        if (
            app in {
                "api_docs",
                "admin",
            }
        ):
            continue

        if (
            app not in app_names
        ):
            app_names.append(
                app
            )

    if not app_names:
        app_names.extend(all_app_names)

    if (
        "supervisor"
        not in app_names
    ):
        app_names.insert(
            0,
            "supervisor",
        )

    selected_total = 0

    # Tables whose task-user rows are especially useful.
    identity_table_terms = {
        "user",
        "account",
        "profile",
        "address",
        "paymentcard",
        "payment_card",
        "credential",
        "password",
    }

    # Faster lookup for whether an app came from task retrieval.
    preferred_app_set = {
        str(app)
        for app in (
            preferred_apps
            or []
        )
    }
    for app_name in app_names:

        if (
            selected_total
            >= max_total_records
        ):
            break

        try:
            app_models = getattr(
                world.models,
                app_name,
            )

        except Exception:
            continue

        try:
            model_names = (
                app_models
                .SQLModel
                .model_names()
            )

        except Exception:
            continue

        direct_link_models = set()
        if relational_closure:
            for candidate_name in model_names:
                try:
                    sample_rows = getattr(app_models, candidate_name).all()
                    if not sample_rows:
                        continue
                    sample = _record_to_dict(sample_rows[0])
                    if any(
                        key == "user_id"
                        or key.endswith("_user_id")
                        or key in {"owner_id", "creator_id", "sender_id", "receiver_id"}
                        for key in map(str, sample)
                    ):
                        direct_link_models.add(candidate_name)
                except Exception:
                    pass

        semantic_aliases = {
            "order": {"buy", "bought", "purchase", "purchased", "last", "recent"},
            "orderitem": {"buy", "bought", "purchase", "purchased", "last", "recent"},
            "cart": {"buy", "purchase", "replacement", "replace", "order"},
            "cartentry": {"buy", "purchase", "replacement", "replace", "order"},
            "transaction": {"money", "sent", "received", "paid", "payment"},
            "paymentrequest": {"request", "approve", "decline", "roommate"},
            "address": {"address", "home", "work", "deliver", "delivery"},
            "paymentcard": {"card", "pay", "payment", "buy", "purchase"},
            "productreturn": {"return", "refund", "replace", "replacement"},
            "contact": {"contact", "roommate", "roommates", "friend", "family"},
            "contactrelationship": {"contact", "roommate", "roommates", "friend", "family"},
        }

        def model_task_relevant(name):
            normalized = str(name).lower()
            if (
                relational_closure
                and normalized == "transaction"
                and "payment request" in instruction_lower
                and "transaction" not in instruction_lower
            ):
                return False
            normalized_terms = {
                normalized,
                normalized.rstrip("s"),
                normalized.replace("item", ""),
            }
            return any(
                term in normalized_terms
                or any(term in value or value in term for value in normalized_terms)
                for term in task_terms
                if len(term) >= 3
            ) or bool(
                task_terms & semantic_aliases.get(normalized, set())
            )

        def relational_priority(name):
            if not relational_closure:
                return 0
            normalized = str(name).lower()
            if normalized == "user":
                return 100
            task_relevant = model_task_relevant(name)
            # Current-state/relationship tables must precede large historical
            # tables. Otherwise (for example) a user's Orders consume the fixed
            # packet before Cart/Product, making exact IDs impossible to infer.
            current_state = {
                "cart", "cartitem", "wishlist", "wishlistitem",
                "playlist", "playlistsong", "paymentrequest", "transaction",
                "contact", "contactrelationship", "friend", "friendship",
                "address", "paymentcard", "email", "thread", "event",
                "file", "directory", "task", "project", "section",
            }
            relevance_bonus = 200 if task_relevant else 0
            current_bonus = 100 if normalized in current_state else 0
            core = {
                "playlist", "album", "project", "section", "task", "order",
                "email", "thread", "event", "file", "directory", "note",
                "product", "paymentrequest", "transaction", "contact",
            }
            links = {
                "playlistsong", "orderitem", "projectcollaboratorlink",
                "emailrecipient", "eventattendee", "taskcollaboratorlink",
            }
            if normalized in core and name in direct_link_models:
                return relevance_bonus + current_bonus + 90
            if normalized in links:
                return relevance_bonus + current_bonus + 80
            if normalized in core:
                return relevance_bonus + current_bonus + 70
            if name in direct_link_models:
                return relevance_bonus + current_bonus + 50
            return relevance_bonus + current_bonus

        model_names = sorted(
            model_names,
            key=lambda name: (
                relational_priority(name),
                str(name).lower() == "user",
                relational_closure and name in direct_link_models,
                str(name).lower() == "musicplayer",
                sum(
                    1
                    for term in task_terms
                    if (
                        term in str(name).lower()
                        or term.rstrip("s").rstrip("d")
                        in str(name).lower()
                    )
                ),
            ),
            reverse=True,
        )

        app_state = {}
        referenced_ids: dict[str, set] = {}

        # Resolve the task user's per-app primary key once, then use it to
        # retrieve rows from relational tables (MusicPlayer, SongLike, etc.).
        # Lexical matching alone cannot connect those rows to an email/name.
        task_user_ids = set()
        try:
            user_model = getattr(
                app_models,
                "User",
            )
            for user_record in user_model.all():
                user_row = _record_to_dict(
                    user_record
                )
                user_text = json.dumps(
                    user_row,
                    ensure_ascii=False,
                    default=str,
                ).lower()
                identity_values = (
                    supervisor_identity_values
                    or supervisor_values
                )
                if any(
                    len(value) >= 3
                    and value in user_text
                    for value in identity_values
                ):
                    user_id = user_row.get("id")
                    if user_id is not None:
                        task_user_ids.add(user_id)
        except Exception:
            pass

        for model_name in model_names:

            if (
                selected_total
                >= max_total_records
            ):
                break

            normalized_model_name = str(model_name).lower()
            if (
                relational_closure
                and normalized_model_name != "user"
                and not model_task_relevant(model_name)
                and not referenced_ids.get(normalized_model_name.rstrip("s"))
            ):
                continue

            try:
                model_cls = getattr(
                    app_models,
                    model_name,
                )

                records = (
                    model_cls.all()
                )

            except Exception:
                continue

            if not records:
                continue

            scored = []

            for record in records:

                row = (
                    _record_to_dict(
                        record
                    )
                )

                row_text = (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        default=str,
                    )
                    .lower()
                )

                score = 0

                direct_user_link = row.get("user_id") in task_user_ids
                if relational_closure and not direct_user_link:
                    direct_user_link = any(
                        value in task_user_ids
                        for field, value in row.items()
                        if str(field) == "user_id"
                        or str(field).endswith("_user_id")
                        or str(field) in {
                            "owner_id", "creator_id", "sender_id", "receiver_id"
                        }
                    )

                if (
                    direct_user_link
                    or (
                        str(model_name).lower() == "user"
                        and row.get("id") in task_user_ids
                    )
                ):
                    score += 30

                if relational_closure:
                    for field_name, field_value in row.items():
                        field_name = str(field_name).lower()
                        if not field_name.endswith("_id"):
                            continue
                        target = field_name[:-3].split("_")[-1]
                        if field_value in referenced_ids.get(target, set()):
                            score += 40

                if (
                    row.get("id")
                    in referenced_ids.get(
                        str(model_name).lower(),
                        set(),
                    )
                ):
                    score += 40

                # ---------------------------------------------
                # A. Belongs to / references supervisor.
                # ---------------------------------------------

                for value in (
                    supervisor_identity_values
                    or supervisor_values
                ):

                    if (
                        len(value) >= 3
                        and value
                        in row_text
                    ):
                        score += 10

                # ---------------------------------------------
                # B. Matches task/retrieved API terminology.
                # ---------------------------------------------

                matched_terms = sum(
                    1
                    for term in task_terms
                    if term
                    in row_text
                )

                score += (
                    3
                    * matched_terms
                )

                if relational_closure and (month_sensitive or day_sensitive or recent_sensitive):
                    date_texts = [
                        str(value)
                        for field, value in row.items()
                        if value is not None
                        and any(token in str(field).lower() for token in ("date", "_at"))
                    ]
                    if month_sensitive and current_month_text and any(
                        text.startswith(current_month_text) for text in date_texts
                    ):
                        score += 80
                    if day_sensitive and current_date_text and any(
                        text.startswith(current_date_text) for text in date_texts
                    ):
                        score += 100
                    if recent_sensitive and date_texts:
                        newest = max(date_texts)
                        digits = re.sub(r"\D", "", newest[:10])
                        if digits:
                            score += int(digits) / 1_000_000

                # ---------------------------------------------
                # C. Identity/account tables are useful seeds.
                # ---------------------------------------------

                normalized_model = (
                    str(
                        model_name
                    )
                    .lower()
                    .replace(
                        "-",
                        "",
                    )
                )

                if score > 0 and any(
                    term
                    in normalized_model
                    for term
                    in identity_table_terms
                ):
                    score += 2

                # ---------------------------------------------
                # D. Slight preference for apps selected by
                #    task-conditioned tool retrieval.
                #
                # Do not make this alone sufficient to expose a
                # record; it only breaks ties among relevant rows.
                # ---------------------------------------------

                if (
                    app_name
                    in preferred_app_set
                    and score > 0
                ):
                    score += 2

                # Only keep rows for which we have some
                # grounding reason.
                if score <= 0:
                    continue

                scored.append(
                    (
                        score,
                        row,
                    )
                )

            if not scored:
                continue

            scored.sort(
                key=lambda item:
                    item[0],
                reverse=True,
            )

            selected_rows = []

            selected_candidates = scored[
                :max_records_per_table
            ]
            if relational_closure and str(model_name).lower() == "user" and task_user_ids:
                needed_user_ids = set(task_user_ids) | set(referenced_ids.get("user", set()))
                task_user_candidates = [
                    item for item in scored if item[1].get("id") in needed_user_ids
                ]
                if task_user_candidates:
                    selected_candidates = task_user_candidates
            if str(model_name).lower() != "user":
                model_is_task_relevant = model_task_relevant(model_name)
                linked_candidates = [
                    item
                    for item in scored
                    if (
                        item[1].get("user_id") in task_user_ids
                        or (
                            relational_closure
                            and any(
                                value in task_user_ids
                                for field, value in item[1].items()
                                if str(field) == "user_id"
                                or str(field).endswith("_user_id")
                                or str(field) in {
                                    "owner_id", "creator_id", "sender_id", "receiver_id"
                                }
                            )
                        )
                    )
                ]
                if linked_candidates:
                    selected_candidates = linked_candidates[
                        :max(
                            max_records_per_table,
                            (
                                60
                                if model_is_task_relevant and broad_coverage
                                else 20
                                if model_is_task_relevant
                                else max_records_per_table
                            ),
                        )
                    ]

                referenced_candidates = [
                    item
                    for item in scored
                    if (
                        item[1].get("id")
                        in referenced_ids.get(str(model_name).lower(), set())
                    )
                ]
                reverse_links = {
                    "orderitem": ("order_id", "order"),
                    "playlistsong": ("playlist_id", "playlist"),
                    "emailrecipient": ("email_id", "email"),
                    "eventattendee": ("event_id", "event"),
                    "taskcollaboratorlink": ("task_id", "task"),
                    "projectcollaboratorlink": ("project_id", "project"),
                }
                reverse_link = reverse_links.get(normalized_model_name)
                if reverse_link:
                    field_name, parent_name = reverse_link
                    parent_ids = referenced_ids.get(parent_name, set())
                    child_candidates = [
                        item for item in scored if item[1].get(field_name) in parent_ids
                    ]
                    if child_candidates:
                        referenced_candidates = child_candidates
                if referenced_candidates:
                    selected_candidates = referenced_candidates[
                        :max(max_records_per_table, 40)
                    ]

            selected_candidates = selected_candidates[
                :max(0, max_total_records - selected_total)
            ]

            for (
                score,
                row,
            ) in selected_candidates:

                compact_row = {
                    str(key):
                        _compact_value(
                            value,
                            max_value_chars,
                        )
                    for key, value
                    in row.items()
                }

                selected_rows.append(
                    compact_row
                )

                if relational_closure and row.get("id") is not None:
                    referenced_ids.setdefault(
                        str(model_name).lower().rstrip("s"), set()
                    ).add(row["id"])

                for field_name, field_value in row.items():
                    field_name = str(field_name).lower()
                    target = None
                    if field_name.endswith("song_ids"):
                        target = "song"
                    elif field_name.endswith("album_ids"):
                        target = "album"
                    elif field_name.endswith("playlist_ids"):
                        target = "playlist"
                    elif field_name in {
                        "user_id", "sender_id", "receiver_id", "owner_id", "creator_id"
                    }:
                        target = "user"
                    elif field_name.endswith("_id"):
                        target = field_name[:-3].split("_")[-1]
                    if target is None or (target == "user" and not relational_closure):
                        continue
                    values = (
                        field_value
                        if isinstance(field_value, (list, tuple, set))
                        else [field_value]
                    )
                    referenced_ids.setdefault(target, set()).update(
                        value
                        for value in values
                        if value is not None
                    )

            if selected_rows:

                app_state[
                    model_name
                ] = selected_rows

                selected_total += len(
                    selected_rows
                )

        if app_state:

            state[
                "apps"
            ][
                app_name
            ] = app_state

    # ---------------------------------------------------------
    # Hard character budget.
    # ---------------------------------------------------------

    while True:

        text = json.dumps(
            state,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        if (
            len(text)
            <= max_total_chars
        ):
            break

        # Remove one record from the currently largest table.
        largest = None

        for (
            app_name,
            tables,
        ) in state[
            "apps"
        ].items():

            for (
                table_name,
                rows,
            ) in tables.items():

                if not rows:
                    continue

                candidate = (
                    len(
                        json.dumps(
                            rows,
                            ensure_ascii=False,
                            default=str,
                        )
                    ),
                    app_name,
                    table_name,
                )

                if (
                    largest is None
                    or candidate[0]
                    > largest[0]
                ):
                    largest = (
                        candidate
                    )

        if largest is None:
            break

        (
            _,
            app_name,
            table_name,
        ) = largest

        state[
            "apps"
        ][
            app_name
        ][
            table_name
        ].pop()

        if not state[
            "apps"
        ][
            app_name
        ][
            table_name
        ]:

            del state[
                "apps"
            ][
                app_name
            ][
                table_name
            ]

        if not state[
            "apps"
        ][
            app_name
        ]:

            del state[
                "apps"
            ][
                app_name
            ]

    return state



def format_appworld_initial_state(
    state: dict,
) -> str:

    return json.dumps(
        state,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


# ---------------------------------------------------------------------
# AppWorld metadata
# ---------------------------------------------------------------------


def _api_docs(
    app_name: str,
) -> dict:

    from appworld.requester import (
        _get_api_name_to_doc,
    )

    try:
        return (
            _get_api_name_to_doc(
                app_name,
                include_private_apis=False,
            )
            or {}
        )

    except Exception:
        return {}

def _available_app_names(
    world,
) -> list[str]:

    try:
        descriptions = (
            world.task.app_descriptions
            or {}
        )

        names = [
            str(name)
            for name in descriptions.keys()
        ]

    except Exception:
        names = []

    # These are both agent-visible AppWorld apps.
    for required in (
        "api_docs",
        "supervisor",
    ):
        if required not in names:
            names.append(required)

    return names

def _short_text(text: Any, limit: int = 160) -> str:
    text = " ".join(
        str(text or "").split()
    )

    if len(text) <= limit:
        return text

    return text[: limit - 1] + "…"


# Arguments that add almost no semantic information but appear everywhere.
_OMIT_ARGS = {
    "access_token",
    "token",
}


def _compact_parameters(
    parameters: Any,
) -> str:
    """
    Compact parameter schema.

    Output example:
        query:string!,page_index:integer,page_limit:integer

    `!` means required.
    """

    if not parameters:
        return ""

    items = []

    # AppWorld may expose parameters as either:
    #   1. dict[name -> schema]
    #   2. list[parameter-dict]
    if isinstance(parameters, dict):

        normalized = []

        for name, value in parameters.items():

            if isinstance(value, dict):
                value = dict(value)
                value.setdefault(
                    "name",
                    name,
                )
            else:
                value = {
                    "name": name,
                    "type": str(value),
                }

            normalized.append(
                value
            )

    elif isinstance(parameters, list):

        normalized = parameters

    else:
        return ""

    for param in normalized:

        if not isinstance(
            param,
            dict,
        ):
            continue

        name = str(
            param.get(
                "name",
                "",
            )
        ).strip()

        if not name:
            continue

        # Keep access_token too if ALL actions are simulated.
        # Don't silently remove model-visible parameters here.

        schema = (
            param.get("schema")
            or {}
        )

        type_name = (
            param.get("type")
            or schema.get("type")
            or "any"
        )

        # Normalize verbose Python-ish representations.
        type_name = str(
            type_name
        ).replace(
            "typing.",
            "",
        )

        required = bool(
            param.get(
                "required",
                False,
            )
        )

        items.append(
            f"{name}:{type_name}"
            f"{'!' if required else ''}"
        )

    return ",".join(
        items
    )

def _doc_terms(
    api_name: str,
    doc: dict,
) -> set[str]:

    pieces = [
        str(api_name),
        str(doc.get("description", "")),
    ]

    parameters = (
        doc.get("parameters")
        or []
    )

    for param in parameters:
        if not isinstance(param, dict):
            continue

        pieces.append(
            str(param.get("name", ""))
        )

        pieces.append(
            str(param.get("description", ""))
        )

    # Response field names are especially useful.
    response_schemas = (
        doc.get("response_schemas")
        or {}
    )

    pieces.append(
        json.dumps(
            response_schemas,
            ensure_ascii=False,
            default=str,
        )
    )

    return _instruction_terms(
        " ".join(pieces)
    )

def _agentworld_tool_definition(
    api_name: str,
    doc: dict,
) -> str:
    """Keep information AgentWorld actually needs.

    In particular:
      - description
      - parameters
      - RESPONSE SCHEMA

    Drop examples / HTTP boilerplate / unnecessary metadata.
    """

    parameters = []

    for param in (
        doc.get("parameters")
        or []
    ):
        if not isinstance(
            param,
            dict,
        ):
            continue

        item = {
            "name": param.get("name"),
            "required": bool(
                param.get(
                    "required",
                    False,
                )
            ),
        }

        if param.get("description"):
            item["description"] = (
                param["description"]
            )

        if param.get("schema"):
            item["schema"] = (
                param["schema"]
            )

        elif param.get("type"):
            item["type"] = (
                param["type"]
            )

        parameters.append(
            item
        )

    payload = {
        "description": (
            doc.get(
                "description",
                "",
            )
        ),

        "parameters":
            parameters,

        # THIS IS CRITICAL.
        "response_schemas": (
            doc.get(
                "response_schemas",
                {},
            )
        ),
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )

def _score_api_for_task(
    instruction_terms: set[str],
    api_name: str,
    doc: dict,
) -> float:

    name_terms = _instruction_terms(
        api_name.replace(
            "_",
            " ",
        )
    )

    description_terms = (
        _instruction_terms(
            str(
                doc.get(
                    "description",
                    "",
                )
            )
        )
    )

    all_terms = _doc_terms(
        api_name,
        doc,
    )

    score = 0.0

    # API-name match is strongest.
    score += (
        10.0
        * len(
            instruction_terms
            & name_terms
        )
    )

    # Natural-language API description.
    score += (
        5.0
        * len(
            instruction_terms
            & description_terms
        )
    )

    # Parameter / response field match.
    score += (
        2.0
        * len(
            instruction_terms
            & all_terms
        )
    )

    return score

def _score_app_for_task(
    instruction: str,
    instruction_terms: set[str],
    app_name: str,
    app_description: str,
) -> float:

    score = 0.0

    if (
        app_name.lower()
        in instruction.lower()
    ):
        score += 100.0

    desc_terms = (
        _instruction_terms(
            str(app_description)
        )
    )

    score += (
        5.0
        * len(
            instruction_terms
            & desc_terms
        )
    )

    return score

def _agentworld_tool_definition(
    api_name: str,
    doc: dict,
) -> str:
    """Compact but semantically useful API definition for AgentWorld.

    Keep:
      - description
      - parameters
      - success/error response schemas

    Drop:
      - examples
      - unnecessary HTTP metadata
      - verbose AppWorld-internal fields
    """

    parameters = []

    raw_parameters = (
        doc.get("parameters")
        or []
    )

    # AppWorld normally gives a list, but tolerate dict form.
    if isinstance(
        raw_parameters,
        dict,
    ):
        normalized_parameters = []

        for name, value in (
            raw_parameters.items()
        ):
            if isinstance(
                value,
                dict,
            ):
                item = dict(value)
                item.setdefault(
                    "name",
                    str(name),
                )
            else:
                item = {
                    "name": str(name),
                    "type": str(value),
                }

            normalized_parameters.append(
                item
            )

    elif isinstance(
        raw_parameters,
        list,
    ):
        normalized_parameters = (
            raw_parameters
        )

    else:
        normalized_parameters = []

    for param in normalized_parameters:

        if not isinstance(
            param,
            dict,
        ):
            continue

        name = str(
            param.get(
                "name",
                "",
            )
            or ""
        ).strip()

        if not name:
            continue

        item = {
            "name": name,
            "required": bool(
                param.get(
                    "required",
                    False,
                )
            ),
        }

        description = (
            param.get(
                "description"
            )
        )

        if description:
            item[
                "description"
            ] = str(
                description
            )

        schema = (
            param.get(
                "schema"
            )
        )

        if schema:
            item[
                "schema"
            ] = schema

        elif param.get(
            "type"
        ):
            item[
                "type"
            ] = param.get(
                "type"
            )

        parameters.append(
            item
        )

    payload = {
        "name": str(api_name),

        "description": str(
            doc.get(
                "description",
                "",
            )
            or ""
        ),

        "parameters":
            parameters,

        # Critical for AgentWorld:
        # tell it exactly what Python/JSON shape to produce.
        "response_schemas": (
            doc.get(
                "response_schemas",
                {},
            )
            or {}
        ),
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _doc_terms(
    api_name: str,
    doc: dict,
) -> set[str]:
    """Terms used only for task-conditioned tool retrieval."""

    pieces = [
        str(api_name).replace(
            "_",
            " ",
        ),

        str(
            doc.get(
                "description",
                "",
            )
            or ""
        ),
    ]

    parameters = (
        doc.get(
            "parameters"
        )
        or []
    )

    if isinstance(
        parameters,
        dict,
    ):
        parameters = [
            {
                "name": name,
                **(
                    value
                    if isinstance(
                        value,
                        dict,
                    )
                    else {
                        "type": str(
                            value
                        )
                    }
                ),
            }
            for name, value
            in parameters.items()
        ]

    if isinstance(
        parameters,
        list,
    ):
        for param in parameters:

            if not isinstance(
                param,
                dict,
            ):
                continue

            pieces.append(
                str(
                    param.get(
                        "name",
                        "",
                    )
                ).replace(
                    "_",
                    " ",
                )
            )

            pieces.append(
                str(
                    param.get(
                        "description",
                        "",
                    )
                    or ""
                )
            )

    return _instruction_terms(
        " ".join(
            pieces
        )
    )


def _score_api_for_task(
    instruction_terms: set[str],
    api_name: str,
    doc: dict,
) -> float:
    """Lexically rank one API against the current task."""

    api_name_text = (
        str(api_name)
        .replace(
            "_",
            " ",
        )
    )

    name_terms = (
        _instruction_terms(
            api_name_text
        )
    )

    description_terms = (
        _instruction_terms(
            str(
                doc.get(
                    "description",
                    "",
                )
                or ""
            )
        )
    )

    all_terms = (
        _doc_terms(
            api_name,
            doc,
        )
    )

    score = 0.0

    # API-name overlap is strongest.
    score += (
        10.0
        * len(
            instruction_terms
            & name_terms
        )
    )

    # Description overlap is also strong.
    score += (
        5.0
        * len(
            instruction_terms
            & description_terms
        )
    )

    # Parameter overlap is weaker but useful.
    score += (
        2.0
        * len(
            instruction_terms
            & all_terms
        )
    )

    return score


def _score_app_for_task(
    instruction: str,
    instruction_terms: set[str],
    app_name: str,
    app_description,
) -> float:
    """Rank one AppWorld app against the current task."""

    app_name = str(
        app_name
    )

    instruction_lower = (
        instruction.lower()
    )

    score = 0.0

    # Explicit app mention is overwhelmingly informative.
    if (
        app_name.lower()
        in instruction_lower
    ):
        score += 100.0

    app_name_terms = (
        _instruction_terms(
            app_name.replace(
                "_",
                " ",
            )
        )
    )

    description_terms = (
        _instruction_terms(
            str(
                app_description
                or ""
            )
        )
    )

    score += (
        10.0
        * len(
            instruction_terms
            & app_name_terms
        )
    )

    score += (
        5.0
        * len(
            instruction_terms
            & description_terms
        )
    )

    return score


def _neighbor_score(
    source_name: str,
    source_doc: dict,
    target_name: str,
    target_doc: dict,
) -> float:
    """Similarity between two APIs in the same app."""

    source_terms = (
        _doc_terms(
            source_name,
            source_doc,
        )
    )

    target_terms = (
        _doc_terms(
            target_name,
            target_doc,
        )
    )

    return float(
        len(
            source_terms
            & target_terms
        )
    )


def appworld_agentworld_tool_docs(
    world,
    instruction: str,
    *,
    max_apps: int = 5,
    max_apis_per_app: int = 6,
    max_total_apis: int = 24,
    neighbors_per_api: int = 1,
    required_apps=None,
    required_apis=None,
) -> tuple[
    Dict[str, str],
    dict,
]:
    """Retrieve a compact task-specific API catalog for AgentWorld.

    Strategy:
      1. rank apps using the task instruction;
      2. rank APIs inside relevant apps;
      3. add required supervisor/auth dependencies;
      4. add a few same-app neighboring APIs;
      5. keep the exact current API schema available separately
         through _static_reference_for_action().
    """

    instruction = str(
        instruction or ""
    )

    instruction_terms = (
        _instruction_terms(
            instruction
        )
    )

    # ---------------------------------------------------------
    # 1. Obtain apps available in this AppWorld task.
    # ---------------------------------------------------------

    try:
        app_descriptions = dict(
            world.task.app_descriptions
            or {}
        )
    except Exception:
        app_descriptions = {}

    available_apps = [
        str(app)
        for app in app_descriptions.keys()
        if str(app) not in {
            "api_docs",
            "admin",
        }
    ]

    # ---------------------------------------------------------
    # 2. Rank apps.
    # ---------------------------------------------------------

    ranked_apps = []

    for app_name in available_apps:

        score = (
            _score_app_for_task(
                instruction,
                instruction_terms,
                app_name,
                app_descriptions.get(
                    app_name,
                    "",
                ),
            )
        )

        ranked_apps.append(
            (
                score,
                app_name,
            )
        )

    ranked_apps.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    selected_apps = []

    # Prefer positively matched apps.
    for score, app_name in ranked_apps:

        if score <= 0:
            continue

        if (
            app_name
            not in selected_apps
        ):
            selected_apps.append(
                app_name
            )

        if (
            len(selected_apps)
            >= max_apps
        ):
            break

    # Do not dilute a strong task match with unrelated apps.  If the
    # instruction has no lexical match at all, retain the highest-ranked
    # app as a conservative fallback.
    if not selected_apps and ranked_apps:
        selected_apps.append(
            ranked_apps[0][1]
        )

    # Some relations are described semantically rather than by app name.  For
    # example, a Venmo task referring to "roommates" needs Phone contacts.
    # Callers may require those apps explicitly so the tool catalog and fixed
    # state packet cannot drift apart.  The default remains unchanged for the
    # standalone AgentWorld baseline and every other method/dataset.
    for app_name in required_apps or ():
        app_name = str(app_name)
        if app_name in available_apps and app_name not in selected_apps:
            selected_apps.append(app_name)

    # Supervisor is needed for credentials/completion.
    if (
        "supervisor"
        not in selected_apps
    ):
        selected_apps.append(
            "supervisor"
        )

    # ---------------------------------------------------------
    # 3. Rank APIs inside selected apps.
    # ---------------------------------------------------------

    selected_docs: dict[
        str,
        dict,
    ] = {}

    seed_api_names = set()

    substantive_apps = [
        app_name
        for app_name in selected_apps
        if app_name != "supervisor"
    ]
    per_app_limit = max(
        max_apis_per_app,
        min(
            12,
            max_total_apis // max(1, len(substantive_apps)),
        ),
    )

    for app_name in substantive_apps:

        docs = (
            _public_api_docs(
                app_name
            )
        )

        scored_apis = []

        for api_name, doc in (
            docs.items()
        ):

            score = (
                _score_api_for_task(
                    instruction_terms,
                    api_name,
                    doc,
                )
            )

            scored_apis.append(
                (
                    score,
                    str(api_name),
                    doc,
                )
            )

        scored_apis.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        # Keep the best APIs from each relevant app.
        for (
            score,
            api_name,
            doc,
        ) in scored_apis[
            :per_app_limit
        ]:

            full_name = (
                f"{app_name}.{api_name}"
            )

            if (
                full_name
                in selected_docs
            ):
                continue

            selected_docs[
                full_name
            ] = doc

            seed_api_names.add(
                full_name
            )

            if (
                len(selected_docs)
                >= max_total_apis
            ):
                break

        if (
            len(selected_docs)
            >= max_total_apis
        ):
            break

    # ---------------------------------------------------------
    # 4. Deterministic dependencies.
    # ---------------------------------------------------------

    dependency_names = {
        "supervisor.show_active_task",
        "supervisor.show_account_passwords",
        "supervisor.complete_task",
    }

    # Any app whose selected APIs require access_token
    # should expose its login API too.
    for full_name in list(
        selected_docs.keys()
    ):

        app_name, api_name = (
            full_name.split(
                ".",
                1,
            )
        )

        doc = (
            selected_docs[
                full_name
            ]
        )

        parameters_text = (
            json.dumps(
                doc.get(
                    "parameters",
                    [],
                ),
                ensure_ascii=False,
                default=str,
            ).lower()
        )

        if (
            "access_token"
            in parameters_text
        ):
            app_docs = (
                _public_api_docs(
                    app_name
                )
            )

            if "login" in app_docs:
                dependency_names.add(
                    f"{app_name}.login"
                )

    for full_name in sorted(
        dependency_names
    ):

        if (
            full_name
            in selected_docs
        ):
            continue

        app_name, api_name = (
            full_name.split(
                ".",
                1,
            )
        )

        doc = (
            _public_api_docs(
                app_name
            ).get(
                api_name
            )
        )

        if isinstance(
            doc,
            dict,
        ):
            selected_docs[
                full_name
            ] = doc

    # Required cross-app schemas are still ordinary public tool definitions;
    # this only prevents lexical retrieval from omitting the bridge API that
    # motivated ``required_apps`` (for example roommate -> Phone contacts).
    for full_name in required_apis or ():
        full_name = str(full_name)
        if full_name in selected_docs or "." not in full_name:
            continue
        app_name, api_name = full_name.split(".", 1)
        if app_name not in available_apps:
            continue
        doc = _public_api_docs(app_name).get(api_name)
        if isinstance(doc, dict):
            selected_docs[full_name] = doc

    # ---------------------------------------------------------
    # 5. Same-app neighbor expansion.
    #
    # This helps cover workflows such as:
    # search -> inspect -> modify
    # ---------------------------------------------------------

    if neighbors_per_api > 0:

        neighbor_additions = {}

        # Only expand from the originally retrieved APIs.
        # Do not recursively expand dependencies forever.
        for source_full_name in sorted(
            seed_api_names
        ):

            if (
                source_full_name
                not in selected_docs
            ):
                continue

            app_name, source_api = (
                source_full_name.split(
                    ".",
                    1,
                )
            )

            source_doc = (
                selected_docs[
                    source_full_name
                ]
            )

            app_docs = (
                _public_api_docs(
                    app_name
                )
            )

            candidates = []

            for (
                target_api,
                target_doc,
            ) in app_docs.items():

                target_full_name = (
                    f"{app_name}.{target_api}"
                )

                if (
                    target_full_name
                    in selected_docs
                ):
                    continue

                score = (
                    _neighbor_score(
                        source_api,
                        source_doc,
                        str(target_api),
                        target_doc,
                    )
                )

                if score <= 0:
                    continue

                candidates.append(
                    (
                        score,
                        str(target_api),
                        target_doc,
                    )
                )

            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                ),
                reverse=True,
            )

            for (
                score,
                target_api,
                target_doc,
            ) in candidates[
                :neighbors_per_api
            ]:

                target_full_name = (
                    f"{app_name}.{target_api}"
                )

                neighbor_additions[
                    target_full_name
                ] = target_doc

        # Neighbor expansion is allowed to go a little above the
        # seed budget, but keep it bounded.
        neighbor_cap = len(selected_docs) + max(
            4,
            len(seed_api_names) // 3,
        )

        for (
            full_name,
            doc,
        ) in neighbor_additions.items():

            if (
                full_name
                in selected_docs
            ):
                continue

            if (
                len(selected_docs)
                >= neighbor_cap
            ):
                break

            selected_docs[
                full_name
            ] = doc

    # ---------------------------------------------------------
    # 6. Convert selected docs to AgentWorld prompt definitions.
    # ---------------------------------------------------------

    out: Dict[str, str] = {}

    for full_name, doc in sorted(
        selected_docs.items()
    ):

        _, api_name = (
            full_name.split(
                ".",
                1,
            )
        )

        out[
            full_name
        ] = (
            _agentworld_tool_definition(
                api_name,
                doc,
            )
        )

    out[
        "__agentworld__.verify_task"
    ] = (
        '{"name":"verify_task","description":"Inspect the full simulated '
        'trajectory and current virtual state. Estimate whether every part of '
        'the user instruction is satisfied. Return a calibrated probability; '
        'task completion alone is not evidence of success.","parameters":['
        '{"name":"instruction","required":true,"type":"string"},'
        '{"name":"completion_args","required":true,"type":"object"},'
        '{"name":"task_was_marked_complete","required":true,"type":"boolean"}'
        '],"response_schemas":{"success":{"success_probability":0.73,'
        '"success":true,"reason":"brief evidence grounded in the simulated state"}}}'
    )

    # ---------------------------------------------------------
    # 7. Build retrieval metadata for initial-state retrieval.
    # ---------------------------------------------------------

    retrieval_terms = set(
        instruction_terms
    )

    for full_name, doc in (
        selected_docs.items()
    ):

        app_name, api_name = (
            full_name.split(
                ".",
                1,
            )
        )

        retrieval_terms.update(
            _instruction_terms(
                app_name.replace(
                    "_",
                    " ",
                )
            )
        )

        retrieval_terms.update(
            _doc_terms(
                api_name,
                doc,
            )
        )

    # Remove extremely generic schema terms that tend to match
    # almost every database row.
    retrieval_terms -= {
        "string",
        "integer",
        "number",
        "boolean",
        "object",
        "array",
        "null",
        "true",
        "false",
        "response",
        "success",
        "error",
        "required",
        "optional",
    }

    retrieval_info = {
        "apps":
            selected_apps,

        "tools":
            sorted(
                selected_docs.keys()
            ),

        "terms":
            sorted(
                retrieval_terms
            ),
    }

    return (
        out,
        retrieval_info,
    )

# ---------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------


def _strip_fence(text: str) -> str:

    text = str(
        text or ""
    ).strip()

    match = _FENCED_RE.match(text)

    return (
        match.group(1).strip()
        if match
        else text
    )


def parse_agentworld_result(
    text: str,
) -> Any:
    """Convert AgentWorld textual observations into AppWorld-native values.

    AgentWorld may return:
      - raw JSON
      - Python-literal-like output
      - JSON encoded as a JSON string
      - MCP {"type": "text", "text": "..."} wrappers
      - MCP {"content": [{"type": "text", "text": "..."}]} wrappers

    AppWorld's Python agent needs the underlying native list/dict/scalar,
    not the MCP transport wrapper.
    """

    value: Any = _strip_fence(
        text
    )

    # A few layers are enough for:
    #
    #   "{\"access_token\":\"...\"}"
    #
    # or:
    #
    #   {"type":"text",
    #    "text":"[{\"account_name\":...}]"}
    #
    for _ in range(5):

        # -------------------------------------------------
        # Text -> parsed Python value
        # -------------------------------------------------

        if isinstance(
            value,
            str,
        ):
            cleaned = _strip_fence(
                value
            ).strip()

            if not cleaned:
                return ""

            try:
                parsed = json.loads(
                    cleaned
                )

            except Exception:
                try:
                    parsed = ast.literal_eval(
                        cleaned
                    )

                except Exception:
                    # Genuine plain-text observation.
                    return cleaned

            # Prevent useless infinite reparsing.
            if (
                isinstance(parsed, str)
                and parsed == cleaned
            ):
                return cleaned

            value = parsed
            continue

        # -------------------------------------------------
        # MCP text wrapper:
        #
        # {"type":"text","text":"..."}
        # -------------------------------------------------

        if isinstance(
            value,
            dict,
        ):
            if (
                value.get("type") == "text"
                and isinstance(
                    value.get("text"),
                    str,
                )
                and set(value.keys()).issubset(
                    {"type", "text"}
                )
            ):
                value = value["text"]
                continue

            # ---------------------------------------------
            # Standard MCP content envelope:
            #
            # {
            #   "content": [
            #       {"type":"text","text":"..."}
            #   ]
            # }
            # ---------------------------------------------

            content = value.get(
                "content"
            )

            if (
                isinstance(content, list)
                and len(content) == 1
                and isinstance(
                    content[0],
                    dict,
                )
                and content[0].get(
                    "type"
                ) == "text"
                and isinstance(
                    content[0].get(
                        "text"
                    ),
                    str,
                )
                and set(
                    value.keys()
                ).issubset(
                    {
                        "content",
                        "isError",
                        "structuredContent",
                    }
                )
            ):
                structured = value.get(
                    "structuredContent"
                )

                if structured is not None:
                    value = structured
                else:
                    value = content[0][
                        "text"
                    ]

                continue

        # Already a native Python object.
        return value

    return value

# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class AgentWorldHTTPError(Exception):

    def __init__(
        self,
        status_code: int,
        body: Any,
    ):

        self.status_code = int(
            status_code
        )

        self.body = body

        super().__init__(
            self.__str__()
        )

    def __str__(self):

        if isinstance(
            self.body,
            (dict, list),
        ):

            body = json.dumps(
                self.body,
                ensure_ascii=False,
            )

        else:

            body = str(
                self.body
            )

        return (
            f"Response status code is "
            f"{self.status_code}:\n"
            f"{body}"
        )


def _explicit_http_error(
    text: str,
):

    cleaned = _strip_fence(text)

    match = _HTTP_ERROR_RE.match(
        cleaned
    )

    if not match:
        return None

    status = int(
        match.group(1)
    )

    body = parse_agentworld_result(
        match.group(2).strip()
    )

    return status, body


def _looks_like_error(
    result: Any,
) -> bool:

    if not isinstance(
        result,
        dict,
    ):
        return False

    status = str(
        result.get(
            "status",
            "",
        )
    ).lower()

    if status in {
        "error",
        "failed",
        "failure",
    }:
        return True

    if any(
        key in result
        for key in (
            "error",
            "errors",
            "detail",
        )
    ):
        return True

    message = str(
        result.get(
            "message",
            "",
        )
    ).lower()

    return any(
        token in message
        for token in (
            "not found",
            "invalid",
            "unauthorized",
            "not authorized",
            "forbidden",
            "insufficient",
            "failed",
            "failure",
            "cannot ",
            "can't ",
        )
    )


# ---------------------------------------------------------------------
# Standalone AgentWorld simulator
# ---------------------------------------------------------------------
def _normalize_agentworld_result_shape(
    app: str,
    api: str,
    result: Any,
    raw: str,
) -> Any:
    """Normalize common transport envelopes, then enforce AppWorld's top-level type."""

    docs = _public_api_docs(
        app
    )

    doc = docs.get(
        api
    )

    if not isinstance(
        doc,
        dict,
    ):
        return result

    response_schemas = (
        doc.get(
            "response_schemas"
        )
        or {}
    )

    if not isinstance(
        response_schemas,
        dict,
    ):
        return result

    expected = response_schemas.get(
        "success"
    )

    # AgentWorld is MCP-trained and occasionally retains the HTTP success
    # envelope around an otherwise correct payload.  AppWorld's requester
    # strips that envelope before exposing a value to Python, so mirror that
    # behavior here.
    if isinstance(result, dict) and "success" in result:
        payload = result.get("success")
        if (
            isinstance(expected, list)
            and isinstance(payload, list)
        ) or (
            isinstance(expected, dict)
            and isinstance(payload, dict)
        ):
            result = payload

    # Common learned-simulator envelopes. Accept these only when the enclosed
    # value has the exact documented top-level type; this repairs formatting,
    # not semantics, and never consults AppWorld state.
    if isinstance(result, dict):
        preferred_keys = [
            str(api).removeprefix("show_").removeprefix("search_"),
            "data",
            "items",
            "results",
            "records",
        ]
        if isinstance(expected, list):
            list_values = [value for value in result.values() if isinstance(value, list)]
            for key in preferred_keys:
                if isinstance(result.get(key), list):
                    result = result[key]
                    break
            else:
                if len(list_values) == 1:
                    result = list_values[0]
        elif isinstance(expected, dict):
            for key in ("data", "result", "record"):
                if isinstance(result.get(key), dict):
                    result = result[key]
                    break

    # Repair only the top-level list container. All fields and values remain
    # model-predicted; this mirrors AppWorld's response decoding rather than
    # adding environment information.
    if isinstance(expected, list):
        if isinstance(result, dict) and not _looks_like_error(result):
            result = [result]
        elif isinstance(result, str) and not result.strip():
            result = []

    if app == "supervisor" and api == "show_account_passwords" and isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and isinstance(item.get("account_name"), str):
                item["account_name"] = item["account_name"].lower()

    wrong_shape = False

    if isinstance(
        expected,
        list,
    ):
        wrong_shape = not isinstance(
            result,
            list,
        )

    elif isinstance(
        expected,
        dict,
    ):
        wrong_shape = not isinstance(
            result,
            dict,
        )

    if wrong_shape:
        raw_preview = str(
            raw
        )[:1000]

        raise RuntimeError(
            "AgentWorld returned the wrong AppWorld response type "
            f"for {app}.{api}: "
            f"expected {type(expected).__name__}, "
            f"got {type(result).__name__}. "
            f"Raw AgentWorld output: {raw_preview!r}"
        )

    return result

class AppWorldAgentWorldSimulator:
    """Pure AgentWorld AppWorld simulator.

    This class has no dependency on AppWorldWMPredictor.
    """

    def __init__(
        self,
        session: AgentWorldSession,
        *,
        strict_task_audit: bool = False,
    ):

        self.session = session
        self.strict_task_audit = bool(strict_task_audit)

        self.completed = False

        self.completion_args = {}

        self.n_transition_calls = 0

        self.verification_raw = None

        self.verification_result = None

    def predict_api(
        self,
        app: str,
        api: str,
        arguments: dict,
        *,
        reference_context: str | None = None,
        max_tokens: int | None = None,
    ):
        args = dict(arguments or {})

        self.n_transition_calls += 1

        # Detect completion from the ACTION itself.
        if (
            app == "supervisor"
            and api == "complete_task"
        ):
            self.completed = True
            self.completion_args = args

        raw = self.session.predict_tool(
            f"{app}.{api}",
            args,
            reference_context=reference_context,
            max_tokens=max_tokens,
        )

        # IMPORTANT:
        # KEEP EXACTLY WHATEVER PARSING CODE YOU HAD HERE
        # BEFORE adding reference_context.
        #
        # Do NOT add another parse_agentworld_result(raw)
        # if that wasn't here previously.

        result = parse_agentworld_result(raw)  # use YOUR existing helper
        # AgentWorld sometimes predicts the same textual HTTP-error
        # representation used by real AppWorld.
        #
        # Do not expose that as a normal string result to generated
        # Python code.
        if isinstance(
            result,
            str,
        ):
            explicit_error = (
                _explicit_http_error(
                    result
                )
            )

            if explicit_error is not None:
                status_code, body = (
                    explicit_error
                )

                raise AgentWorldHTTPError(
                    status_code,
                    body,
                )

        # Structured failure without an explicit HTTP prefix.
        if _looks_like_error(
            result
        ):
            raise AgentWorldHTTPError(
                422,
                result,
            )

        # The strict persistent AppWorld path must preserve Python control flow
        # for learned duplicate/non-membership failures.  These phrases are
        # normal 422 responses in AppWorld but do not contain the generic words
        # recognized above.  Keep the standalone AgentWorld baseline's parser
        # unchanged by applying the extension only to the opt-in strict mode.
        if self.strict_task_audit and isinstance(result, dict):
            strict_message = str(result.get("message") or "").lower()
            if any(
                phrase in strict_message
                for phrase in (
                    "already exists",
                    "already in playlist",
                    "not in playlist",
                    "does not exist",
                )
            ):
                raise AgentWorldHTTPError(422, result)

        result = _normalize_agentworld_result_shape(
            app,
            api,
            result,
            raw,
        )
        return result


    def verify_task(
        self,
        instruction: str,
        *,
        mutation_context: str | None = None,
    ):
        """Ask the SAME AgentWorld session to score the simulated state."""

        arguments = {
                "instruction":
                    instruction,

                "completion_args":
                    self.completion_args,

                "task_was_marked_complete":
                    self.completed,
            }
        verification_options = {}
        if self.strict_task_audit:
            verification_options = {
                "reference_context": (
                "This internal verifier must inspect the preceding simulated "
                "tool history and current virtual state. Return exactly one "
                "JSON object with keys success_probability (a number from 0 "
                "to 1), success (boolean), and reason (string). Score every "
                "requirement using the compact fixed task evidence supplied "
                "with the ledger when present; its file contents and records "
                "are authoritative and must not be contradicted from memory. "
                "requirement in the instruction; do not treat complete_task "
                "as proof of success. For instructions containing all/every, "
                "verify exact set coverage rather than one example. A single "
                "call to an API that documents page_index and page_limit sees only its "
                "documented default page when page_limit is omitted; APIs without "
                "pagination (such as show_song_queue) return their entire collection. "
                "Never apply an invented five-row limit to an unpaginated API. When the fixed "
                "evidence contains more eligible rows, require pagination until "
                "exhaustion or one explicit page_limit large enough to cover the "
                "set. Never credit an oversized hallucinated single-page response. Moving a "
                "cart or wishlist item must move its full initial quantity; a "
                "default quantity of one is incomplete when more existed. A "
                "purchase task must not include unrelated pre-existing cart "
                "items. Amazon wishlist-to-cart operations are moves rather "
                "than copies: each successful move consumes the corresponding "
                "wishlist quantity. If the cart is subsequently cleared, a "
                "second attempt based on the stale cached wishlist cannot "
                "re-add those consumed rows. Treat such a trajectory as "
                "incomplete unless a later successful operation genuinely "
                "restores every required cart item. For a task that moves items selected by a reference "
                "list or filter, the required target set is the exact "
                "intersection of that reference with the initial source "
                "collection: do not reward inserting reference items that "
                "were absent from every source. If one entity occurs in "
                "multiple source containers, require a successful removal "
                "from each container but only one target insertion unless "
                "the instruction explicitly asks to preserve duplicates. "
                "Treat malformed calls as unmet. A failed write makes the "
                "task incomplete only when that write was required and no "
                "later successful mutation covers it; a caught attempt on an "
                "already-approved, denied, or otherwise irrelevant record "
                "does not negate all required successful writes."
                " Repeated writes to the same target do not substitute for "
                "distinct required targets. Count a mutation only when its "
                "predicted result indicates success. Before assigning the "
                "probability, internally audit three concrete sets from the "
                "fixed initial state and ordered ledger: required source "
                "memberships, successful source removals/updates, and "
                "successful destination insertions. Compare their distinct "
                "entity IDs and per-container multiplicities. A visible "
                "missing or extra required mutation must score below 0.5, "
                "even when the trajectory otherwise looks coherent. For "
                "Python loops, an uncaught API exception terminates that code "
                "block immediately: do not count loop iterations or writes "
                "after the failing call. In particular, adding a song to an "
                "archive playlist does not prove it was removed from any source "
                "playlist; require each successful source-scoped removal in the "
                "ordered ledger. Printed counters and later read results are not "
                "substitutes for missing successful mutation entries. For "
                "payment-request approval tasks, only initially pending "
                "requests (both approved_at and denied_at null) are eligible; "
                "already approved or denied requests are not missing work. "
                "For each successful Venmo approval funded by a payment card, "
                "debit that card's running balance by the request amount in "
                "ledger order. An insufficient or expired-card attempt is not "
                "a successful approval; require a later successful retry with "
                "a viable funding source. "
                "For Amazon orders, require a predicted successful place_order "
                "after comparing the final cart total with the chosen card's "
                "running balance. An insufficient-card attempt needs a later "
                "successful retry and does not itself place an order. "
                "For received requests, first require receiver_id to equal the "
                "task user's Venmo User.id; do not count outgoing rows where "
                "the task user is sender. Then join each sender_id to a Venmo "
                "User email and the Phone contact relationship before applying "
                "the instruction's relationship and date filters. "
                "A Song row marked file_reference_catalog_match_only_not_"
                "playlist_membership is searchable catalog evidence only and "
                "must not be treated as belonging to any playlist without an "
                "explicit PlaylistSong row or playlist song_ids membership. "
                "For Spotify, duplicate insertion means prior membership in the "
                "same target playlist. A newly created playlist is empty, and a "
                "failed malformed add call creates no membership."
                + (
                    "\n\n" + mutation_context
                    if mutation_context
                    else ""
                )
                ),
                "max_tokens": min(self.session.max_tokens, 3072),
            }

        raw = self.session.predict_tool(
            "__agentworld__.verify_task",
            arguments,
            **verification_options,
        )

        self.verification_raw = raw

        result = (
            parse_agentworld_result(
                raw
            )
        )

        self.verification_result = (
            result
        )

        probability = None

        if isinstance(
            result,
            dict,
        ):

            for key in (
                "success_probability",
                "probability",
                "score",
                "confidence",
                "confidence_score",
            ):

                if key in result:
                    try:
                        probability = float(
                            result[key]
                        )
                    except Exception:
                        pass

                    if probability is not None:
                        break

            if (
                probability is None
                and isinstance(
                    result.get("success"),
                    bool,
                )
            ):

                probability = (
                    1.0
                    if result["success"]
                    else 0.0
                )

        # Conservative fallback instead of crashing the rollout.
        if probability is None:
            probability = 0.0

        # The strict simulator asks for both a calibrated probability and a
        # categorical judgment.  Keep those two fields internally consistent
        # instead of treating an explicit ``success: false`` as a passing score.
        # This consumes only the learned verifier's own output and no benchmark
        # contract or environment feedback.
        if (
            self.strict_task_audit
            and isinstance(result, dict)
            and result.get("success") is False
        ):
            probability = min(probability, 0.49)

        if 1.0 < probability <= 100.0:
            probability /= 100.0

        probability = max(
            0.0,
            min(
                1.0,
                probability,
            ),
        )

        return (
            probability,
            result,
            raw,
        )


# ---------------------------------------------------------------------
# AppWorld requester interception
# ---------------------------------------------------------------------


def install_agentworld_requester(
    world,
    simulator,
):
    """Intercept AppWorld calls for the standalone AgentWorld baseline.

    Control-plane/static calls are answered exactly:
      - api_docs.*
      - supervisor reads (completion remains simulated)
      - login (side-effect-free authentication)

    All substantive environment calls are simulated by AgentWorld.

    The exact static schema for the current substantive API is provided
    to AgentWorld as reference_context.
    """

    orig_request = (
        world.requester.request
    )

    def agentworld_request(
        _app_name,
        _api_name,
        client=None,
        raise_on_failure=None,
        show=False,
        track=True,
        **data,
    ):

        app_name = str(
            _app_name
        )

        api_name = str(
            _api_name
        )

        args = dict(
            data or {}
        )

        # -----------------------------------------------------
        # 1. api_docs.* is static tool metadata.
        #
        # Do NOT ask AgentWorld to hallucinate the action space.
        # Execute it against the pristine AppWorld requester,
        # then insert exactly what the agent saw into AgentWorld's
        # trajectory history.
        # -----------------------------------------------------

        if (
            app_name == "api_docs"
            or (
                app_name == "supervisor"
                and api_name != "complete_task"
            )
            or api_name == "login"
        ):

            result = (
                orig_request(
                    app_name,
                    api_name,
                    client=client,
                    raise_on_failure=
                        raise_on_failure,
                    show=show,
                    track=track,
                    **args,
                )
            )

            simulator.session.observe_tool(
                f"{app_name}.{api_name}",
                args,
                result,
            )

            return result

        # -----------------------------------------------------
        # 3. Everything substantive still goes through
        #    Qwen-AgentWorld.
        #
        # Give it the exact schema of THIS API only.
        # -----------------------------------------------------

        reference_context = (
            _static_reference_for_action(
                world,
                app_name,
                api_name,
                args,
            )
        )

        try:

            return (
                simulator.predict_api(
                    app_name,
                    api_name,
                    args,
                    reference_context=
                        reference_context,
                )
            )

        except AgentWorldHTTPError as exc:

            if (
                raise_on_failure
                is False
            ):
                return exc.body

            raise Exception(
                str(exc)
            ) from None

    world.requester.request = (
        agentworld_request
    )

    return orig_request
__all__ = [
    "AppWorldAgentWorldSimulator",
    "appworld_agentworld_tool_docs",
    "install_agentworld_requester",
    "parse_agentworld_result",
]
