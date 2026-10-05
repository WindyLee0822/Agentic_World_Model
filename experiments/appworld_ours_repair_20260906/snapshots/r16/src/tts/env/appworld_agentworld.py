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
    """Static documentation only.

    Never executes an AppWorld action.
    Always returns a STRING.
    """

    app_name = str(
        app_name or ""
    )

    api_name = str(
        api_name or ""
    )

    if isinstance(arguments, dict):
        args = dict(arguments)
    else:
        args = {}

    reference = {}

    # -----------------------------------------------------
    # Definition of current tool.
    # -----------------------------------------------------

    current_docs = _public_api_docs(
        app_name
    )

    current_doc = current_docs.get(
        api_name
    )

    if isinstance(current_doc, dict):
        reference[
            "current_tool_definition"
        ] = current_doc

    # -----------------------------------------------------
    # api_docs.show_api_doc
    # -----------------------------------------------------

    if (
        app_name == "api_docs"
        and api_name == "show_api_doc"
    ):
        target_app = str(
            args.get(
                "app_name",
                "",
            )
            or ""
        )

        target_api = str(
            args.get(
                "api_name",
                "",
            )
            or ""
        )

        target_docs = (
            _public_api_docs(
                target_app
            )
        )

        target_doc = (
            target_docs.get(
                target_api
            )
        )

        if isinstance(
            target_doc,
            dict,
        ):
            reference[
                "requested_tool_definition"
            ] = target_doc

        else:
            reference[
                "requested_tool_definition"
            ] = {
                "exists": False,
                "app_name": target_app,
                "api_name": target_api,
            }

    # -----------------------------------------------------
    # api_docs.show_api_descriptions
    # -----------------------------------------------------

    elif (
        app_name == "api_docs"
        and api_name
        == "show_api_descriptions"
    ):
        target_app = str(
            args.get(
                "app_name",
                "",
            )
            or ""
        )

        target_docs = (
            _public_api_docs(
                target_app
            )
        )

        descriptions = []

        for name, doc in (
            target_docs.items()
        ):
            # Already guaranteed by
            # _public_api_docs(), but keep safe.
            if not isinstance(
                doc,
                dict,
            ):
                continue

            descriptions.append(
                {
                    "name": str(name),
                    "description": str(
                        doc.get(
                            "description",
                            "",
                        )
                        or ""
                    ),
                }
            )

        reference[
            "requested_app_apis"
        ] = descriptions

    # -----------------------------------------------------
    # api_docs.show_app_descriptions
    # -----------------------------------------------------

    elif (
        app_name == "api_docs"
        and api_name
        == "show_app_descriptions"
    ):
        try:
            descriptions = (
                world.task
                .app_descriptions
            )
        except Exception:
            descriptions = None

        # Don't assume AppWorld represents this
        # as any particular container type.
        if isinstance(
            descriptions,
            (dict, list),
        ):
            reference[
                "available_apps"
            ] = descriptions

    if not reference:
        return ""

    try:
        payload = json.dumps(
            reference,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        return ""

    return (
        "Static AppWorld tool metadata for "
        "simulating the current call. "
        "This is documentation, not a real "
        "environment observation.\n"
        + payload
    )

def _instruction_terms(
    instruction: str,
) -> set[str]:
    words = re.findall(
        r"[A-Za-z0-9@._+-]+",
        instruction.lower(),
    )

    return {
        word
        for word in words
        if len(word) >= 3
        and word not in _STOPWORDS
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

def build_appworld_initial_state(
    world,
    instruction: str,
    *,
    max_records_per_table: int = 5,
    max_total_records: int = 40,
    max_value_chars: int = 200,
    max_total_chars: int = 25000,
) -> dict:
    """
    Build a SMALL task-relevant initial state for AgentWorld.

    Important:
      - no agent action is executed here
      - all later actions are still simulated
      - only task-relevant initial DB facts are exposed
    """

    terms = _instruction_terms(
        instruction
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
        state["datetime"] = str(
            world.task.datetime
        )
    except Exception:
        pass

    # ---------------------------------------------------------
    # Supervisor identity
    # ---------------------------------------------------------

    supervisor_values = set()

    try:
        supervisor = world.task.supervisor

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
                    "value": str(
                        supervisor
                    )
                }

        state["supervisor"] = {
            str(k): _compact_value(
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

    except Exception:
        pass

    # ---------------------------------------------------------
    # Apps available for this task only
    # ---------------------------------------------------------

    try:
        app_names = list(
            (
                world.task.app_descriptions
                or {}
            ).keys()
        )
    except Exception:
        app_names = []

    app_names = [
        str(app)
        for app in app_names
        if app not in {
            "api_docs",
            "admin",
        }
    ]

    if "supervisor" not in app_names:
        app_names.append(
            "supervisor"
        )

    selected_total = 0

    # Tables whose task-user rows are particularly useful.
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

    for app_name in app_names:

        if selected_total >= max_total_records:
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

        app_state = {}

        for model_name in model_names:

            if selected_total >= max_total_records:
                break

            try:
                model_cls = getattr(
                    app_models,
                    model_name,
                )

                records = model_cls.all()
            except Exception:
                continue

            if not records:
                continue

            scored = []

            for record in records:

                row = _record_to_dict(
                    record
                )

                row_text = json.dumps(
                    row,
                    ensure_ascii=False,
                    default=str,
                ).lower()

                score = 0

                # ---------------------------------------------
                # A. Belongs to / references supervisor
                # ---------------------------------------------

                for value in supervisor_values:

                    if (
                        len(value) >= 3
                        and value in row_text
                    ):
                        score += 10

                # ---------------------------------------------
                # B. Mentioned by task instruction
                # ---------------------------------------------

                matched_terms = sum(
                    1
                    for term in terms
                    if term in row_text
                )

                score += (
                    3 * matched_terms
                )

                # ---------------------------------------------
                # C. Identity/account tables are useful seeds
                # ---------------------------------------------

                normalized_model = (
                    model_name
                    .lower()
                    .replace("-", "")
                )

                if any(
                    term
                    in normalized_model
                    for term
                    in identity_table_terms
                ):
                    score += 2

                # Only retain something that has some reason
                # to be in the initial state.
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
                key=lambda x: x[0],
                reverse=True,
            )

            selected_rows = []

            for score, row in scored[
                :max_records_per_table
            ]:

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

            if selected_rows:

                app_state[
                    model_name
                ] = selected_rows

                selected_total += len(
                    selected_rows
                )

        if app_state:
            state["apps"][
                app_name
            ] = app_state

    # ---------------------------------------------------------
    # Hard safety budget
    # ---------------------------------------------------------

    while True:

        text = json.dumps(
            state,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        if len(text) <= max_total_chars:
            break

        # Remove one record from the largest remaining table.
        largest = None

        for app_name, tables in (
            state["apps"].items()
        ):
            for table_name, rows in (
                tables.items()
            ):
                if not rows:
                    continue

                candidate = (
                    len(
                        json.dumps(
                            rows,
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
                    largest = candidate

        if largest is None:
            break

        _, app_name, table_name = (
            largest
        )

        state["apps"][
            app_name
        ][
            table_name
        ].pop()

        if not state["apps"][
            app_name
        ][
            table_name
        ]:
            del state["apps"][
                app_name
            ][
                table_name
            ]

        if not state["apps"][
            app_name
        ]:
            del state["apps"][
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

def appworld_agentworld_tool_docs(
    world,
) -> Dict[str, str]:
    """Ultra-compact AppWorld tool definitions.

    Format:
        spotify.search(query,limit,offset)
        spotify.follow_artist(artist_id)
        gmail.search_emails(query)

    No descriptions, schemas, HTTP metadata, types, examples,
    or authentication parameters.
    """

    out: Dict[str, str] = {}

    for app in _available_app_names(world):

        docs = _api_docs(app)

        for api, doc in sorted(
            docs.items()
        ):

            name = f"{app}.{api}"

            params = _compact_parameters(
                doc.get("parameters") or []
            )

            out[name] = (
                f"{name}({params})"
            )

    # Keep verification tool equally compact.
    out[
        "__agentworld__.verify_task"
    ] = (
        "__agentworld__.verify_task("
        "instruction,completion_args,"
        "task_was_marked_complete)"
    )

    return out
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

    cleaned = _strip_fence(text)

    if not cleaned:
        return ""

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    try:
        value = ast.literal_eval(
            cleaned
        )
    except Exception:
        return cleaned

    if (
        isinstance(
            value,
            (
                dict,
                list,
                tuple,
                str,
                int,
                float,
                bool,
            ),
        )
        or value is None
    ):
        return value

    return cleaned


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


class AppWorldAgentWorldSimulator:
    """Pure AgentWorld AppWorld simulator.

    This class has no dependency on AppWorldWMPredictor.
    """

    def __init__(
        self,
        session: AgentWorldSession,
    ):

        self.session = session

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
        )

        # IMPORTANT:
        # KEEP EXACTLY WHATEVER PARSING CODE YOU HAD HERE
        # BEFORE adding reference_context.
        #
        # Do NOT add another parse_agentworld_result(raw)
        # if that wasn't here previously.

        result = parse_agentworld_result(raw)  # use YOUR existing helper

        return result
    
    def verify_task(
        self,
        instruction: str,
    ):
        """Ask the SAME AgentWorld session to score the simulated state."""

        raw = self.session.predict_tool(
            "__agentworld__.verify_task",
            {
                "instruction":
                    instruction,

                "completion_args":
                    self.completion_args,

                "task_was_marked_complete":
                    self.completed,
            },
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

    # Save only so caller can restore later.
    # Do NOT call it during AgentWorld rollout.
    orig_request = (
        world.requester.request
    )

    def agentworld_request(
        app_name,
        api_name,
        client=None,
        raise_on_failure=None,
        show=False,
        track=True,
        **data,
    ):

        # ---------------------------------------------
        # STATIC metadata retrieval only.
        #
        # This does NOT execute the action.
        # ---------------------------------------------

        reference_context = (
            _static_reference_for_action(
                world,
                str(app_name),
                str(api_name),
                dict(data or {}),
            )
        )

        try:

            # EVERY action reaches AgentWorld.
            return simulator.predict_api(
                str(app_name),
                str(api_name),
                dict(data or {}),
                reference_context=
                    reference_context,
            )

        except AgentWorldHTTPError as exc:

            if raise_on_failure is False:
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