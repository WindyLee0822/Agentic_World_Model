"""Inference adapters used only by AppWorld Ours/Qwen strict simulation."""

from __future__ import annotations

import json
import hashlib
import ast
import copy
import os
import re

from ..llm.agentworld import AgentWorldSession


_READ_PROGRAM_CACHE = {}


def validate_predicted_schema(value, schema, path="response", *, allow_null_fields=False):
    """Validate documented JSON examples without consulting environment state."""
    if schema is None:
        return
    # response_schemas contains examples, not a non-null JSON Schema. Native
    # fields (e.g. Alarm.repeat_days) may be null even when the example is a list.
    if allow_null_fields and value is None and path != "response":
        return
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object, got {type(value).__name__}")
        missing = set(schema) - set(value)
        if missing:
            raise ValueError(f"{path} is missing documented fields: {sorted(missing)}")
        for key, expected in schema.items():
            validate_predicted_schema(value[key], expected, f"{path}.{key}",
                                      allow_null_fields=allow_null_fields)
    elif isinstance(schema, list):
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list, got {type(value).__name__}")
        if schema:
            for index, item in enumerate(value):
                validate_predicted_schema(item, schema[0], f"{path}[{index}]",
                                          allow_null_fields=allow_null_fields)
    elif value is not None:
        expected_type = type(schema)
        valid = isinstance(value, expected_type)
        if expected_type in (int, float):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid:
            raise ValueError(f"{path} must be {expected_type.__name__}, got {type(value).__name__}")


def validate_api_prediction(value, doc, name=None, *, allow_null_fields=False):
    schemas = doc.get("response_schemas", {})
    expected = schemas.get("success", schemas)
    if isinstance(value, dict) and set(value) == {"message"} and "failure" in schemas:
        expected = schemas["failure"]
    if expected:
        validate_predicted_schema(value, expected, allow_null_fields=allow_null_fields)
    if name == "spotify.show_song_queue" and isinstance(value, list):
        positions = [row.get("position") for row in value if isinstance(row, dict)]
        if positions != list(range(len(value))):
            raise ValueError(
                "Queue positions must be consecutive zero-based indices 0..len(queue)-1, "
                "matching remove_song_from_queue. The schema example 1 is not an index base."
            )


def complete_fixed_snapshot(world, packet, tool_docs, instruction):
    """Complete task-relevant ownership and foreign-key edges before sampling.

    This is initial evidence retrieval, not an API simulator: it takes no action
    arguments and is never called by a candidate or a verifier.
    """
    from .appworld_agentworld import _record_to_dict

    def words(value):
        return set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", value).lower()))

    def stem(value):
        return re.sub(r"[^a-z]", "", value.lower()).rstrip("s")

    identity = {str(packet.get("supervisor", {}).get(k, "")).lower()
                for k in ("email", "phone_number")}
    identity.discard("")
    funding_cards = []
    admin_models = getattr(world.models, "admin", None)
    if admin_models is not None:
        owners = {_record_to_dict(row).get("id") for row in admin_models.MainUser.all()
                  if any(str(_record_to_dict(row).get(k, "")).lower() in identity
                         for k in ("email", "phone_number"))}
        funding_cards = [
            {key: row.get(key) for key in ("card_number", "balance", "expiry_year", "expiry_month")}
            for row in (_record_to_dict(record) for record in admin_models.PaymentCard.all())
            if row.get("main_user_id") in owners
        ]
    supervisor_models = getattr(world.models, "supervisor", None)
    if supervisor_models is not None:
        profiles = [_record_to_dict(row) for row in supervisor_models.Supervisor.all()]
        profiles = [row for row in profiles if any(
            str(row.get(k, "")).lower() in identity for k in ("email", "phone_number"))]
        if profiles:
            profile = profiles[0]
            packet["supervisor"].update({k: v.isoformat() if hasattr(v, "isoformat") else v
                                         for k, v in profile.items() if k != "id"})
            supervisor_state = packet["apps"].setdefault("supervisor", {})
            supervisor_state["Supervisor"] = profiles
            for table in ("Address", "PaymentCard", "AccountPassword"):
                supervisor_state[table] = [
                    _record_to_dict(row) for row in getattr(supervisor_models, table).all()
                    if table == "AccountPassword" or getattr(row, "supervisor_id", None) == profile.get("id")
                ]
    user_roles = {"user", "friend", "sender", "receiver", "owner", "creator", "payer", "payee", "author"}
    ignored = {"show", "get", "search", "list", "entry", "global", "user", "create", "delete", "update", "to", "from", "the", "my"}
    for app, state in packet.get("apps", {}).items():
        if app in {"amazon", "venmo", "supervisor"}:
            state["FundingCard"] = funding_cards
        models = getattr(world.models, app)
        names = list(models.SQLModel.model_names())
        task_words = words(instruction) - ignored
        task_words |= {
            "phone": {"contact"},
            "file_system": {"file", "directory"},
            "spotify": {"music", "player"},
        }.get(app, set())
        task_stems = {stem(word) for word in task_words}
        if "friend" in task_stems:
            task_stems.add("friendship")
        rows_by_name = {}

        def rows(name):
            if name not in rows_by_name:
                rows_by_name[name] = []
                for record in getattr(models, name).all():
                    row = _record_to_dict(record)
                    if app == "file_system" and name in {"File", "Directory"}:
                        path = str(row.get("path") or "")
                        match = re.match(r"^(/home/[^/]+)(?:/|$)", path)
                        if match:
                            row["home_directory"] = match.group(1)
                            row["tilde_path"] = "~" + path[len(match.group(1)):]
                    rows_by_name[name].append(row)
            return rows_by_name[name]

        own_ids = {row.get("id") for row in rows("User")
                   if any(str(row.get(k, "")).lower() in identity for k in ("email", "phone_number"))} if "User" in names else set()
        if not own_ids:
            continue

        def merge(name, additions):
            target = state.setdefault(name, [])
            existing = {row.get("id"): index for index, row in enumerate(target)}
            for row in additions:
                # Whitespace is data in file contents, notes and messages.
                # Restore exact frozen values rather than the generic retriever's
                # whitespace-collapsed display summaries.
                value = {str(k): v.isoformat() if hasattr(v, "isoformat") else v
                         for k, v in row.items()}
                if row.get("id") in existing:
                    target[existing[row.get("id")]] = value
                    continue
                existing[row.get("id")] = len(target)
                target.append(value)

        merge("User", [row for row in rows("User") if row.get("id") in own_ids])
        sequence_tables = {
            "Order", "OrderItem", "PaymentCard", "Address", "Playlist", "PlaylistSong",
            "PaymentRequest", "Transaction", "Friendship", "File", "Directory", "Note",
        }
        state["IdSequence"] = [
            {"table": name, "next_id": max((int(row.get("id") or 0) for row in rows(name)), default=0) + 1}
            for name in names if name in sequence_tables
        ]
        # Public catalogs are searchable global state, not user-owned rows.
        # Freeze them before any action so learned programs can perform complete
        # comparisons without inventing omitted search results.
        for name in {
            "amazon": ("Product", "Seller"),
            "spotify": ("Song", "Album", "Artist"),
            "venmo": ("User",),
        }.get(app, ()):
            if name in names:
                merge(name, rows(name))

        for name in names:
            if name == "User":
                continue
            terms = {stem(word) for word in words(name)} - ignored
            if name not in state and not terms.intersection(task_stems):
                continue
            merge(name, [row for row in rows(name) if any(
                row.get(key) in own_ids for key in
                ("user_id", "owner_id", "creator_id", "sender_id", "receiver_id"))])

        # Follow both scalar and list IDs, including role aliases and compound
        # model names. Friendship.friend_id points to User, not to Friend.
        lookup = {stem(name): name for name in names}
        for _ in range(4):
            before = sum(map(len, state.values()))
            if os.environ.get("TTS_APPWORLD_OURS_QWEN_COMPLETE_CHILDREN", "0") == "1":
                # Initial retrieval caps child tables even after this function
                # restores all owned parents. Complete containment edges as well
                # as forward references, before any candidate action is sampled.
                # These are structural collections, not reverse catalog links
                # such as Product -> every customer's OrderItem.
                children = {
                    "OrderItem": (("order_id", "Order"),),
                    "PlaylistSong": (("playlist_id", "Playlist"),),
                    "Section": (("project_id", "Project"),),
                    "Task": (("project_id", "Project"),),
                    "SubTask": (("task_id", "Task"),),
                    "TaskLabelLink": (("task_id", "Task"),),
                    "TaskComment": (("task_id", "Task"),),
                    "TaskCollaboratorLink": (("task_id", "Task"),),
                    "ProjectCollaboratorLink": (("project_id", "Project"),),
                    "EventAttendee": (("event_id", "Event"),),
                    "Attachment": (("email_id", "Email"), ("draft_id", "Draft"),
                                   ("task_comment_id", "TaskComment")),
                }
                for child, links in children.items():
                    if child not in names:
                        continue
                    parent_ids = {
                        field: {row["id"] for row in state.get(parent, [])
                                if row.get("id") is not None}
                        for field, parent in links
                    }
                    if any(parent_ids.values()):
                        merge(child, [row for row in rows(child) if any(
                            row.get(field) in ids for field, ids in parent_ids.items()
                        )])
            wanted = {}
            for table_rows in list(state.values()):
                for row in table_rows:
                    for field, value in row.items():
                        if field.endswith("_ids") and isinstance(value, list):
                            base, values = field[:-4], value
                        elif field.endswith("_id") and value is not None:
                            base, values = field[:-3], [value]
                        else:
                            continue
                        parts = base.split("_")
                        target = "User" if base in user_roles else next(
                            (lookup[stem("_".join(parts[index:]))] for index in range(len(parts))
                             if stem("_".join(parts[index:])) in lookup), None)
                        if target:
                            wanted.setdefault(target, set()).update(v for v in values if isinstance(v, (str, int)))
            for target, ids in wanted.items():
                merge(target, [row for row in rows(target) if row.get("id") in ids])
            if sum(map(len, state.values())) == before:
                break


class NonThinkingChat:
    """Keep bounded ReAct completions available to the code parser."""

    def __init__(self, chat):
        self.chat = chat

    def complete(self, messages, **kwargs):
        # Solver turns run outside world.execute's SafetyGuard. A timed-out
        # computation worker cannot be restarted from inside a simulated API
        # call, but can be recovered here before the next code cell.
        from ..llm.python_tool import shared_python_tool

        shared_python_tool().prepare()
        thinking = os.environ.get("TTS_APPWORLD_OURS_QWEN_SOLVER_THINKING", "0") == "1"
        extra = dict(kwargs.pop("extra_body", None) or {})
        extra["chat_template_kwargs"] = {
            **extra.get("chat_template_kwargs", {}), "enable_thinking": thinking,
        }
        messages = [dict(message) for message in messages]
        reminder = (
            "\nExecution discipline: inspect api_docs.show_api_doc before using an "
            "unfamiliar endpoint, especially login and writes. Copy its exact "
            "parameter names and response fields. Keep login_result['access_token'] "
            "in a variable; never hardcode tokens or guessed record identities. "
            "Fetch all pages when the task asks for all/every; compute sets and "
            "amounts in Python. If an API call fails, inspect its documentation "
            "and repair the call before completing the task. Do not suppress "
            "exceptions or mark a partially executed task complete.\n"
            "Before writes, turn every constraint in the task into an explicit "
            "Python filter or set operation. 'Friends in my phone' means contacts "
            "with the friend relationship, not the whole contact book. Respect "
            "relationship, category, date, quantity, seller and payment constraints. "
            "For a purchase of a subset of the cart, arrange that only the requested "
            "items are ordered and preserve unrelated items. Never retype a returned "
            "list or its IDs as literals: compute from response variables so the code "
            "works for every returned record. Use the documented zero-based queue "
            "positions. Finish only after the complete set of required changes.\n"
        )
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += reminder
        else:
            messages.insert(0, {"role": "system", "content": reminder})
        code_budget = kwargs.get("max_tokens") or 2048
        if thinking:
            kwargs["max_tokens"] = max(code_budget, 8192)
        result = self.chat.complete(messages, extra_body=extra, **kwargs)
        if thinking and not re.search(r"```python\s*\n", result or ""):
            # A reasoning-only completion is not an executable solver action.
            # Recover within this turn using the same simulated observations.
            extra["chat_template_kwargs"]["enable_thinking"] = False
            kwargs["max_tokens"] = code_budget
            messages.append({"role": "user", "content": (
                "Return the next executable Python code block only. Use the "
                "observations above and the documented apis interface."
            )})
            result = self.chat.complete(messages, extra_body=extra, **kwargs)
        return result


class QwenAppWorldSession(AgentWorldSession):
    """Retrieve bounded evidence from a fixed packet, never from live APIs.

    The action's public API contract arrives in reference_context. Avoid a
    second, incomplete tool registry and a full multi-app database in the system
    prompt: both confused parameter validation and overflowed the 32K window.
    The ordered simulated mutation ledger is supplied by the requester.
    """

    def __init__(self, *, initial_state, **kwargs):
        self.initial_state = initial_state
        self.computed_reads = os.environ.get("TTS_APPWORLD_OURS_QWEN_COMPUTED_READS", "0") == "1"
        self.computed_verifier = os.environ.get("TTS_APPWORLD_OURS_QWEN_COMPUTED_VERIFIER", "0") == "1"
        self.evidence_guard = os.environ.get("TTS_APPWORLD_OURS_QWEN_EVIDENCE_GUARD", "0") == "1"
        self.transitions = []
        self.computation_errors = []
        self.n_computations = 0
        self.computation_trace = []
        self.instructions = (
            "You simulate AppWorld API calls. Every response is a prediction; "
            "you have no tools and cannot execute real actions. Return only the "
            "JSON API response, without MCP envelopes, prose, or code fences. "
            "The current call's supplied public parameters and response_schemas "
            "are authoritative, including defaults and field names. Do not "
            "treat numeric schema examples as literal IDs, limits or index bases. "
            "Spotify queue positions are zero-based, consistent with the "
            "remove_song_from_queue position parameter. Do not "
            "invent Python/Pydantic validation errors, alternative signatures, "
            "unavailable tools, or authentication requirements. An omitted API "
            "from earlier documentation is not an unavailable API. Use the "
            "fixed snapshot plus successful simulated mutations in order to "
            "predict the current state. Failed calls never mutate state. "
            "Opaque simulated access tokens issued by prior successful logins "
            "remain valid. A phone login uses the documented username field; "
            "do not invent a phone_number argument. Preserve exact IDs, names, "
            "amounts, file contents and collection membership from evidence. "
            "Unlisted rows may exist; never invent matching records to make "
            "the task succeed. For verification judge the task against the "
            "simulated mutations, not the agent's claims of completion.\n"
        )
        kwargs["system_prompt"] = self.instructions
        kwargs["enable_thinking"] = False
        super().__init__(**kwargs)

    def _snapshot(self, app, arguments, limit=36000):
        packet = {
            "datetime": self.initial_state.get("datetime"),
            "supervisor": self.initial_state.get("supervisor", {}),
            "apps": {},
        }
        apps = self.initial_state.get("apps", {})
        selected = {app: apps[app]} if app in apps else apps
        needles = [str(v).lower() for k, v in arguments.items()
                   if k not in {"access_token", "page_index", "page_limit"}
                   and isinstance(v, (str, int)) and str(v)]
        tables = []
        for name, state in selected.items():
            packet["apps"][name] = {}
            for table, rows in state.items():
                if not isinstance(rows, list):
                    continue
                ordered = sorted(rows, key=lambda row: sum(
                    needle in json.dumps(row, default=str).lower()
                    for needle in needles), reverse=True)
                packet["apps"][name][table] = []
                tables.append((packet["apps"][name][table], ordered))
        # Round-robin tables so large catalogs cannot hide credentials, join
        # rows, cards or files. Keep whole records and valid JSON throughout.
        used = len(json.dumps(packet, default=str))
        for index in range(max((len(rows) for _, rows in tables), default=0)):
            for target, rows in tables:
                if index >= len(rows):
                    continue
                size = len(json.dumps(rows[index], ensure_ascii=False, default=str)) + 1
                if used + size <= limit:
                    target.append(rows[index])
                    used += size
        return json.dumps(packet, ensure_ascii=False, separators=(",", ":"), default=str)

    def _transitions(self, app):
        """One explicit ledger format for direct and computed predictions."""
        return [{
            **copy.deepcopy(row),
            "app": row["name"].split(".", 1)[0],
            "api": row["name"].split(".", 1)[1],
            "args": copy.deepcopy(row["arguments"]),
            "result": copy.deepcopy(row["observation"]),
            "predicted_result": copy.deepcopy(row["observation"]),
        } for row in self.transitions
            if app == "__agentworld__" or row["name"].split(".", 1)[0] == app]

    @staticmethod
    def _ledger_text(transitions):
        # Omit duplicate aliases from the prompt, but retain every transition.
        # The full ledger remains available even when conversation turns age out.
        return json.dumps([
            {key: row[key] for key in ("name", "app", "api", "arguments", "observation")}
            for row in transitions
        ], ensure_ascii=False, separators=(",", ":"), default=str)

    def predict_tool(self, name, arguments, **kwargs):
        app, api = name.split(".", 1)
        self.messages[0]["content"] = (
            self.instructions + "\nFixed pre-action snapshot (possibly partial):\n"
            + self._snapshot(name.split(".", 1)[0], arguments)
            + "\nOrdered simulated calls for this app (failed calls change nothing):\n"
            + self._ledger_text(self._transitions(app))
        )
        from .appworld_wm import _api_doc

        doc = _api_doc(app, api) if app != "__agentworld__" else {}
        if ((self.computed_reads and doc.get("method") == "GET")
                or (self.computed_verifier and name == "__agentworld__.verify_task")):
            result = self._compute_read(name, arguments, doc, kwargs.get("reference_context", ""))
            if result is not None:
                self.observe_tool(name, arguments, result)
                return json.dumps(result, ensure_ascii=False, default=str)
        from .appworld_agentworld import parse_agentworld_result

        for attempt in range(3):
            accepted_history = copy.deepcopy(self.messages)
            raw = super().predict_tool(name, arguments, **kwargs)
            try:
                validate_api_prediction(parse_agentworld_result(raw), doc, name,
                                        allow_null_fields=self.evidence_guard)
                break
            except ValueError as exc:
                # AgentWorld records its output before returning. An invalid
                # response is a rejected proposal, not an observed transition.
                self.messages = accepted_history
                if attempt == 2:
                    raise RuntimeError(f"Invalid simulated schema for {name}: {exc}") from exc
                kwargs = {**kwargs, "reference_context": (
                    str(kwargs.get("reference_context") or "")
                    + f"\nYour previous prediction had an invalid schema: {exc}. "
                    + "Repair the simulated response using the documented contract. "
                    + "The rejected prediction did not change state.\n"
                    + json.dumps(doc, default=str)
                )}
        if doc.get("method") in {"POST", "PUT", "PATCH", "DELETE"}:
            from .appworld_agentworld import parse_agentworld_result

            self.transitions.append({"name": name, "arguments": copy.deepcopy(arguments),
                                     "observation": parse_agentworld_result(raw)})
        return raw

    def _compute_read(self, name, arguments, doc, reference):
        """Let Qwen predict a read program; execute only on copied JSON data."""
        from ..llm.python_tool import shared_python_tool

        app = name.split(".", 1)[0]
        verification = name == "__agentworld__.verify_task"
        state = {"datetime": self.initial_state.get("datetime"),
                 "supervisor": self.initial_state.get("supervisor", {}),
                 "apps": (dict(self.initial_state.get("apps", {})) if verification else
                          {app: self.initial_state.get("apps", {}).get(app, {})})}
        state["apps"].setdefault("supervisor", self.initial_state.get("apps", {}).get("supervisor", {}))
        transitions = self._transitions(app)
        data = {"state": state, "arguments": arguments, "transitions": transitions,
                "app": app, "api": name.split(".", 1)[1], "supervisor": state["supervisor"]}
        data["next_ids"] = {row["table"]: row["next_id"]
                            for row in state["apps"].get(app, {}).get("IdSequence", [])}
        if app == "file_system":
            home = next((row.get("home_directory") for table in ("File", "Directory")
                         for row in state["apps"][app].get(table, [])
                         if row.get("home_directory")), None)
            if home:
                arguments = {key: home + value[1:] if key.endswith("_path")
                             and isinstance(value, str) and value.startswith("~") else value
                             for key, value in arguments.items()}
                data["arguments"] = arguments
        identity = {str(state["supervisor"].get(key, "")).lower()
                    for key in ("email", "phone_number")}
        identity.discard("")
        data["task_users"] = {
            app_name: next((row for row in tables.get("User", []) if any(
                str(row.get(key, "")).lower() in identity for key in ("email", "phone_number"))), None)
            for app_name, tables in state["apps"].items()
        }
        inventory = {app_name: {table: {"count": len(rows), "fields": list(rows[0]) if rows else []}
                               for table, rows in tables.items()}
                     for app_name, tables in state["apps"].items()}
        messages = [{"role": "system", "content": (
            "Predict an AppWorld API observation by writing a short Python program. "
            "The variable data is ALREADY injected by the worker. NEVER initialize, "
            "overwrite, redefine or shadow data. Start with state = data['state']. "
            "You are a world model, not the task-solving agent. The program receives "
            "data['state'] with the frozen snapshot, data['arguments'] with this call, "
            "and data['transitions'] with prior simulated mutations (name, arguments, "
            "observation). The supervisor is a flat profile at state['supervisor']; "
            "all table rows live under state['apps'][app][table]. Transition aliases "
            "are app, api, args, result, predicted_result. "
            "name is the FULL app.api string; compare t['app'] to the app name "
            "and t['api'] to the endpoint name. Never compare t['name'] to just "
            "an app name. Successful writes often return only a message or a new ID: "
            "apply the actual arguments to the matching table rows, not observation "
            "as a replacement database row.\nWork on the already "
            "copied inputs; do not deepcopy catalogs. Use only these copied JSON values: no real APIs, imports "
            "of appworld, filesystem, database, network or tools. Basic Python plus "
            "json, re, datetime, collections, copy and math are allowed. Print exactly "
            "one JSON response using print(json.dumps(result)). Return code in a "
            "python fence. Keep code concise, with no comments or restating the task. "
            "Date strftime is unavailable; use ISO strings or date-field f-strings. "
            "Use set operators (&, |, -) and all(x in b for x in a) instead of "
            "intersection/union/issubset methods. Use from copy import deepcopy "
            "if a copy is needed. Avoid huge numeric sentinels; use None or min/max "
            "with default. Fields absent in the inventory must be derived from "
            "the supplied relations, never assumed present.\n"
            "Do not manually transcribe records or output hardcoded "
            "lists: filter, join, sort and paginate data programmatically. Resolve "
            "the task user by matching supervisor email/phone to the app User rows. "
            "Apply successful prior simulated mutations to a local copy before "
            "reading it. Failed calls change nothing. Use the documented response "
            "field names, including renaming database id to contact_id/song_id/etc. "
            "Do not invent data or rows to fill missing evidence. Token handles from "
            "simulated logins are valid. Preserve file whitespace. Queue positions "
            "are zero-based. Only APIs documenting pagination have a page limit. "
            "All frozen rows are available in data even when omitted from the "
            "prompt sample.\n"
            + ("This is INTERNAL VERIFICATION, not an environment API. Compute "
               "whether the task in data['arguments']['instruction'] was fulfilled "
               "by applying the simulated mutation ledger to the fixed initial state. "
               "Return {success_probability: number, success: bool, reason: string}. "
               "Use exact set comparisons and Python arithmetic, not verbal guesses. "
               "Descending queue removals preserve lower indices: simulate list.pop. "
               "Existing correct friends need no add call; add_friend establishes a "
               "friendship immediately and remove_friend deletes it. Shared expenses "
               "include the task user as well as the listed other participants. "
               "Do not require impossible confirmation beyond the documented API. "
               "Complete_task alone never proves success. Score the required final "
               "state, including preservation of unrelated entities.\n"
               if verification else "")
        )}, {"role": "user", "content": (
            f"API: {name}\nArguments: {json.dumps(arguments)}\n"
            f"The flat supervisor profile is exactly: {json.dumps(state['supervisor'], default=str)}\n"
            f"Resolved task users, available as data['task_users']: {json.dumps(data['task_users'], default=str)}\n"
            "Use data['task_users'][app_name]['id'] for that app's owner ID. IDs are app-specific; "
            "do not use the Supervisor table ID as an app User ID. state['supervisor'] is already "
            "a flat profile: it has no nested Supervisor key. For verification, the provided "
            "answer is data['arguments']['completion_args'].get('answer'); task_was_marked_complete "
            "is directly in data['arguments'], NOT inside completion_args.\n"
            f"Table inventory under state['apps']: {json.dumps(inventory)}\n"
            "Exact ordered data['transitions'] (aliases args/result/predicted_result "
            "refer to arguments/observation):\n"
            f"{self._ledger_text(transitions)}\n"
            f"Snapshot sample: {self._snapshot(app, arguments, limit=12000)}\n"
            f"Current API contract: {json.dumps(doc, default=str)}\n"
            f"API contract and transition evidence:\n{reference}"
        )}]
        cache_key = hashlib.sha256(json.dumps(
            [self.model, self.base_url, messages, data, self.evidence_guard], sort_keys=True, default=str,
        ).encode()).hexdigest()
        cached_code = _READ_PROGRAM_CACHE.get(cache_key)
        for _ in range(3):
            if cached_code is not None:
                code = cached_code
                raw = f"```python\n{code}\n```"
            else:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0,
                    max_tokens=4096, extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                self.total_tokens += int(getattr(response.usage, "total_tokens", 0) or 0)
                raw = response.choices[0].message.content or ""
                match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.S)
                code = match.group(1) if match else raw
            self.n_computations += 1
            from ..llm.python_tool import PythonToolResult
            try:
                tree = ast.parse(code)
                if any(isinstance(node, ast.Name) and node.id == "data"
                       and isinstance(node.ctx, ast.Store) for node in ast.walk(tree)):
                    raise ValueError("data is already provided; do not assign to data or embed a replacement snapshot. Read data['state'] and data['arguments'].")
                result = shared_python_tool().run(
                    ("import json\n" + code) if self.evidence_guard else code,
                    data, timeout_s=10, max_input_chars=8_000_000,
                    max_output_chars=100_000, max_code_chars=24000, max_steps=5_000_000,
                    evidence_schema_guard=self.evidence_guard,
                )
            except (SyntaxError, ValueError) as exc:
                result = PythonToolResult(error=str(exc))
            error = result.error
            trace = {"name": name, "arguments": arguments,
                                           "code": code, "error": error,
                                           "cache_hit": cached_code is not None}
            self.computation_trace.append(trace)
            if not error:
                try:
                    value = json.loads(result.output)
                    validate_api_prediction(value, doc, name,
                                            allow_null_fields=self.evidence_guard)
                    if len(_READ_PROGRAM_CACHE) >= 256:
                        _READ_PROGRAM_CACHE.pop(next(iter(_READ_PROGRAM_CACHE)))
                    _READ_PROGRAM_CACHE[cache_key] = code
                    return value
                except (ValueError, TypeError) as exc:
                    error = str(exc)
            trace["error"] = error
            self.computation_errors.append(f"{name}: {error}")
            cached_code = None
            _READ_PROGRAM_CACHE.pop(cache_key, None)
            if result.fatal:
                # Regenerating code cannot repair a dead worker inside the
                # execution sandbox. Fall back to the learned JSON predictor;
                # NonThinkingChat recovers the worker before the next cell.
                return None
            messages.extend([{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"Pure Python computation failed: {error}. Repair the program; print one JSON value."}])
        return None
