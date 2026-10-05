"""AppWorld Ours-only repairs; baseline predictors keep their original behavior."""

from __future__ import annotations

import json
import copy
from dataclasses import replace
import os
from pathlib import Path
import re
import traceback
import uuid

from .appworld_wm import (
    AppWorldWMPredictor, WMHardTimeout, _api_docs, _env_float, _env_int, _wall_clock,
    _parse_json,
    _success_schema,
    _api_doc,
    _is_success_verdict,
)


# Audited against AppWorld a072b7a (data 0.2): these GET endpoints create a
# file_system record and increment a download counter in the source app.
_DOWNLOAD_WRITES = frozenset({
    ("amazon", "download_order_receipt"),
    ("amazon", "download_prime_subscription_receipt"),
    ("gmail", "download_attachment"),
    ("todoist", "download_attachment"),
    ("spotify", "download_premium_subscription_receipt"),
    ("splitwise", "download_expense_receipt_file"),
    ("splitwise", "download_payment_receipt_file"),
    ("venmo", "download_bank_transfer_receipt"),
    ("venmo", "download_transaction_receipt"),
})


class _DiagnosticChat:
    """Optional host-side trace; nothing is appended to the solver's observations."""

    def __init__(self, client, directory):
        self.client = client
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Open before entering AppWorld's SafetyGuard, like the computation worker.
        self.stream = open(Path(directory) / f"wm-{os.getpid()}-{uuid.uuid4().hex}.jsonl", "w")
        self.context = {}

    def complete(self, messages, **kwargs):
        started = _wall_clock()
        event = {**self.context, "max_tokens": kwargs.get("max_tokens")}
        try:
            result = self.client.complete(messages, **kwargs)
            prefix = str(result or "")[:1600]
            event.update(response_prefix=re.sub(
                r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "<redacted>", prefix),
                response_chars=len(result or ""))
            return result
        except Exception as exc:
            event.update(error_type=type(exc).__name__, error=str(exc)[:1500],
                         traceback=traceback.format_exc()[-5000:])
            raise
        finally:
            event["elapsed_s"] = _wall_clock() - started
            self.stream.write(json.dumps(event, default=str) + "\n")
            self.stream.flush()


class AppWorldOursWMPredictor(AppWorldWMPredictor):
    """Keep observation generation and candidate verification inside the simulator.

    Predict-all needs multiple model turns per API call. Its budget must accommodate
    those turns, and exhaustion must abort the candidate rather than fabricate a
    successful write or return an empty/stale read. No candidate writes execute.
    """

    MAX_TURNS = 10  # Grounding, schema inspection and dependent reads need separate rounds.

    def __init__(self, chat_client, **kwargs):
        from .appworld_ours_chat import isolated_chat

        chat_client = isolated_chat(chat_client)
        diagnostic_dir = os.environ.get("TTS_APPWORLD_OURS_DIAGNOSTICS_DIR")
        if diagnostic_dir:
            chat_client = _DiagnosticChat(chat_client, diagnostic_dir)
        kwargs.setdefault("time_budget_s", _env_float(
            "TTS_APPWORLD_OURS_WM_TIME_BUDGET_S",
            _env_float("TTS_APPWORLD_WM_TIME_BUDGET_S", 1800.0)))
        kwargs.setdefault("hard_timeout_s", _env_float(
            "TTS_APPWORLD_OURS_WM_HARD_TIMEOUT_S",
            _env_float("TTS_APPWORLD_WM_HARD_TIMEOUT_S", 2400.0)))
        super().__init__(chat_client, **kwargs)
        self._logical_read_cache = {}

    def set_diagnostic_context(self, task_id, seed):
        if isinstance(self._chat, _DiagnosticChat):
            self._chat.context = {"task_id": task_id, "seed": seed}

    def bind_query_interface(self, query):
        native = getattr(query, "_request", None)
        requester = getattr(native, "__self__", None)
        function = getattr(native, "__func__", None)
        if function is not None and hasattr(requester, "raise_on_extra_parameters"):
            # AppWorld's public-parameter checker rejects private API names even
            # though its routing metadata includes them. Give only the simulator
            # a separate requester view for this audited read; retain explicit
            # parameter validation and leave the solver's requester untouched.
            private_reader = copy.copy(requester)
            private_reader.raise_on_extra_parameters = False

            def read_with_private_balance(app, api, **kwargs):
                if (app, api) != ("admin", "show_card_balance"):
                    return native(app, api, **kwargs)
                if not self.allows_grounding(app, api):
                    raise ValueError(f"{app}.{api} is not read-only")
                allowed = {p["name"] for p in _api_doc(app, api).get("parameters", [])}
                extra = set(kwargs) - allowed - {"client", "raise_on_failure", "show", "track"}
                if extra:
                    raise ValueError(f"Unexpected parameters for {app}.{api}: {sorted(extra)}")
                return function(private_reader, app, api, **kwargs)

            query._request = read_with_private_balance
        super().bind_query_interface(query)

    def start_rollout(self):
        """Exclude authentication, Python-worker startup and world loading from budgets."""
        now = _wall_clock()
        self._deadline = now + self.time_budget_s if self.time_budget_s > 0 else None
        self._hard_deadline = now + self.hard_timeout_s if self.hard_timeout_s > 0 else None

    def predicted_write(self, app, api, args, schema):
        """Satisfy the requester callback using the actual predictive transition method."""
        return self.predict_write(app, api, args, schema)

    @staticmethod
    def _write_intent_key(app, api, args):
        # A declined checkout retried with another card is the same purchase intent.
        # Keep other target arguments, including the shipping address, distinct.
        if (app, api) == ("amazon", "place_order"):
            args = {key: value for key, value in (args or {}).items()
                    if key != "payment_card_id"}
        return AppWorldWMPredictor._write_intent_key(app, api, args)

    def _predict_grounded_venmo_funding(self, api, args, schema):
        # The inherited shortcut returns hand-built observations and synthetic
        # IDs without invoking the simulator program, even in program-only mode.
        return None

    def _reference_verdict(self, instruction, checklist, temperature):
        prompt = (
            "Audit the AppWorld task by reconstructing the candidate's final logical state in "
            "generated Python. No candidate write was executed. Read tools provide exact PRISTINE "
            "state; apply the ordered successful writes in data['writes'] to that baseline. "
            "Compare the resulting values with EVERY evaluator requirement and gold value.\n"
            "SUCCESS requires every requirement to be established. An UNKNOWN or missing "
            "read is not a successful check. Gather needed evidence using PYTHON_READ, including "
            "initial collections when writes identify positions or when a complete set is required. "
            "Do not infer initial contents from the number of writes or from the expected final set. "
            "Do not trust a candidate's claim that it fetched all pages. Inspect the full relevant "
            "collection and compute exact missing/extra elements or value differences in Python. "
            "Bookkeeping checks may be inferred from the documented effects of the actual applied "
            "writes, but a substantive unknown must be investigated and cannot count as a pass. "
            "When removing rows from an ordered collection, update dependent positions and "
            "current-item cursors. Do not preserve stale per-row is_current/is_playing annotations "
            "after deleting the current row. Track enclosing control state separately and apply "
            "explicit control calls: a successful play call on a nonempty remaining queue makes "
            "the player active even when its previous current row was removed. "
            "If a required check remains unproven after the read budget, return FAIL.\n\n"
            f"TASK:\n{instruction}\n\nCHECKS AND GOLD:\n{checklist}\n\n"
            "All write arguments and results are copied in data['writes']; all grounding results "
            "are in data['reads']. Use those variables instead of transcribing guessed state. "
            "Inspect schemas with a generated facts round when necessary. End with a generated "
            "PYTHON_FINAL program printing VERDICT: SUCCESS or VERDICT: FAIL."
        )
        response = self._predict(
            prompt, "VERDICT: SUCCESS or VERDICT: FAIL", temperature=temperature,
            max_tokens=4096, budgeted=False,
            python_context=self._verifier_python_context(instruction, checklist=checklist),
            python_output_kind="verdict",
        )
        return _is_success_verdict(response) if response else None

    def predict_success(self, *args, **kwargs):
        self.verifier_error = None
        try:
            return super().predict_success(*args, **kwargs)
        except RuntimeError as exc:
            if not str(exc).startswith((
                "program-only AppWorld prediction", "Simulator program did not produce"
            )):
                raise
            # Verification failure is not a new generation attempt. Preserve the
            # already simulated candidate, assigning conservative zero confidence.
            self.verifier_error = str(exc)
            return 0.0

    def _read_write_precondition(self, app, api, args):
        if not self.allows_grounding(app, api):
            raise ValueError(f"{app}.{api} is not a read-only grounding tool")
        if app in self.dirty_apps:
            key = self._logical_read_key(app, api, args)
            if key in self._logical_read_cache:
                return copy.deepcopy(self._logical_read_cache[key])
            # This nested prediction grounds against pristine data internally.
            # Its generated program supplies the logical value to the enclosing
            # write predictor; no candidate mutation executes.
            return self.predict_read(app, api, args, None, _success_schema(app, api), dirty=True)
        return self._query.read_value(f"{app}.{api}", args)

    def _logical_read_key(self, app, api, args):
        return (len(self.write_history), app, api, json.dumps(
            self.hydrate_read_args(app, api, args), sort_keys=True, default=str))

    def predict_read(self, app, api, args, pre_write_result, schema, *, dirty=True):
        result = super().predict_read(app, api, args, pre_write_result, schema, dirty=dirty)
        # Only generated observations enter this cache. A later applied write
        # invalidates them, including transitions affecting a different app.
        version = len(self.write_history)
        self._logical_read_cache = {
            key: value for key, value in self._logical_read_cache.items() if key[0] == version
        }
        self._logical_read_cache[self._logical_read_key(app, api, args)] = copy.deepcopy(result)
        return result

    def is_write_api(self, app, api):
        return (app, api) in _DOWNLOAD_WRITES or super().is_write_api(app, api)

    def allows_grounding(self, app, api):
        return (app, api) not in _DOWNLOAD_WRITES and super().allows_grounding(app, api)

    def _record_applied_write(self, app, api, args, result, key):
        response = super()._record_applied_write(app, api, args, result, key)
        if (app, api) in _DOWNLOAD_WRITES:
            self.dirty_apps.add("file_system")
        return response

    def _complete(self, prompt, *, max_tokens, temperature=0.0, budgeted=True):
        # The inherited 700/800-token verdict/write budgets were sized for JSON.
        # Python programs (including read requests and repair) need room to reach
        # their final print; a truncated program supplies no simulator verdict.
        minimum = _env_int(
            "TTS_APPWORLD_OURS_WM_PROGRAM_TOKENS" if budgeted
            else "TTS_APPWORLD_OURS_WM_VERIFIER_TOKENS",
            2048 if budgeted else 4096,
        )
        return super()._complete(
            prompt, max_tokens=max(max_tokens, minimum),
            temperature=temperature, budgeted=budgeted)

    def _predict(self, prompt, final_instr, **kwargs):
        # The Python protocol previously offered read tools without a catalog.
        # Models guessed endpoints (e.g. venmo.show_user_public_info), received
        # plain-text permission errors, then repeatedly tried json.loads(error).
        # Supply documentation, never database values or an executed write.
        context = kwargs.get("python_context") or {}
        mode = context.get("mode")
        if kwargs.get("python_output_kind") in {"verdict", "status"}:
            prompt += (
                "\nVERIFIER INPUT SCHEMA: data['checklist'] is a STRING containing all checks; "
                "data['check'] is a STRING containing one check, or an empty string. "
                "There is NO data['check']['checks']. data['gold'] is a dictionary with "
                "'raw' (string) and 'sections' (dictionary). Section values are already JSON-decoded "
                "when possible: use json_value(value), not unconditional json.loads(value). "
                "data['writes'] and data['reads'] are lists of records with app, api, name, args "
                "and result. Inspect those variables in a PYTHON_READ facts round before "
                "assuming additional keys or nesting.\n"
            )
        logical_preconditions = (
            mode == "write_with_rejection" and self._query is not None
            and kwargs.get("python_read_fn") is None
        )
        if logical_preconditions:
            kwargs["python_read_fn"] = self._read_write_precondition
        if mode in {"write_response", "write_with_rejection"}:
            context = {**context, "writes": [
                {"app": app, "api": api, "name": f"{app}.{api}", "args": args, "result": result}
                for app, api, args, result in self.write_history
            ]}
            kwargs["python_context"] = context
        if mode in {"write_with_rejection", "dirty_read"}:
            prompt += (
                "\n\nLOGICAL STATE AND SCHEMA INSPECTION:\n"
                + ("For THIS write validation, read tools return CURRENT LOGICAL state: "
                   "a nested simulator predicts any read affected by prior writes. "
                   "Use these read values directly for preconditions; do NOT apply prior writes "
                   "to them a second time. " if logical_preconditions else
                   "Grounding tools return PRISTINE state, before ALL simulated writes. ")
                +
                "data['writes'] contains only successful applied transitions, in order. "
                "Before your final prediction, use a PYTHON_READ facts round to inspect "
                "the relevant entries in data['writes'] and the actual shape/keys of each "
                "needed read value. Print compact facts computed by your program, including "
                "relevant write arguments and response fields. Do not assume the history is empty. "
                "Never guess a response key or use .get(guessed_key, []) as evidence of an empty set. "
                "For a PRISTINE baseline only, apply the relevant ordered transitions in generated "
                "Python BEFORE checking preconditions or returning a dirty read. For example, "
                "an empty pristine cart does not imply an empty logical cart after an add-product "
                "transition. Likewise a newly added card/file/record can exist only in the "
                "predicted transitions. Resolve uncertain fields with read API documentation "
                "and computed facts, not invented failures.\n"
                "An inspection program must print ONE JSON envelope, never labeled debug prints. "
                "For example, after gathering reads:\n"
                "PYTHON_READ:\n"
                "print_json({'queries': [], 'facts': {'applied_writes': data['writes'], "
                "'read_values': [{'name': r['name'], 'value': json_value(r['result'])} "
                "for r in data['reads']]}})\n"
                "Inspect these returned facts before choosing field names or computing predicates.\n"
                "The current call's parameters are directly in data['args'], e.g. "
                "data['args']['payment_card_id']; there is no nested args key. "
                "You may issue another PYTHON_READ whenever evidence is missing; a turn number "
                "does not require a premature PYTHON_FINAL.\n"
            )
        if context.get("mode") == "dirty_read" and context.get("app") == "file_system":
            # Include cross-app file creation when the next read predicts a file
            # listing or its content. The entries remain ordered model predictions.
            context = {**context, "writes": [
                {"app": app, "api": api, "name": f"{app}.{api}", "args": args, "result": result}
                for app, api, args, result in self.write_history
                if app == "file_system" or (app, api) in _DOWNLOAD_WRITES
            ]}
            kwargs["python_context"] = context
        apps = (context["app"],) if context.get("app") else self.active_task_apps()
        if mode == "write_with_rejection":
            # Financial preconditions may require this existing read-only
            # private API even when the current write belongs to another app.
            apps = tuple(dict.fromkeys((*apps, "admin")))
        catalog = []
        for app in apps:
            try:
                docs = _api_docs(app)
            except ImportError:
                docs = {}
            for api, doc in docs.items():
                if (app, api) in _DOWNLOAD_WRITES or str(doc.get("method", "")).upper() not in {"GET", "HEAD", "OPTIONS"}:
                    continue
                parameters = ", ".join(
                    str(p["name"]) + (" [required]" if p.get("required") else "")
                    for p in doc.get("parameters", []) if isinstance(p, dict) and p.get("name"))
                catalog.append(f"{app}.{api}({parameters}): {doc.get('description', '')[:160]}")
        if catalog:
            prompt += (
                "\n\nDOCUMENTED READ TOOLS (use these exact names and parameter names):\n"
                + "\n".join(catalog)
                + "\nFor full documentation, request api_docs.show_api_doc with app_name and api_name. "
                "A denied or invalid tool request is not evidence that a user/record is absent. "
                "Inspect a read value using json_value; plain-text ERROR results are not JSON. "
                "If evidence is unclear, use PYTHON_READ to print computed facts before a final program.\n"
                "Use copied variables for identifiers and payment-card fields; do not transcribe "
                "them into integer literals (the computation sandbox bounds integer literals). "
            )
        if mode == "write_with_rejection":
            prompt += (
                "For admin.show_card_balance, copy ALL documented required fields from the "
                "actual payment-card value, including owner_name, card_number, expiry_year, "
                "expiry_month and cvv_number. A tool validation error is not a business-rule rejection.\n"
            )
        response = super()._predict(prompt, final_instr, **kwargs)
        if mode in {"clean_read", "dirty_read", "write_response", "write_with_rejection"}:
            parsed = _parse_json(response)
            if parsed is None:
                raise RuntimeError("Simulator program did not produce a usable API observation")
            if mode == "write_with_rejection":
                if not isinstance(parsed, dict) or (
                    "outcome" in parsed and not isinstance(parsed.get("response"), (dict, list))
                ):
                    raise RuntimeError("Simulator program did not produce a write response envelope")
        return response

    def _record_python_facts(self, facts, computed_facts, turn):
        super()._record_python_facts(facts, computed_facts, turn)
        if isinstance(computed_facts, dict):
            # These values were already computed by the simulator's Python.
            # Carry their names forward exactly, alongside the round snapshots.
            # Otherwise programs using data['facts']['account'] fail because the
            # host silently relocated 'account' under python_computed_facts_N.
            facts.update({key: value for key, value in computed_facts.items()
                          if not str(key).startswith("python_")})

    def _requires_read_inspection(self, context, output_kind):
        return output_kind in {"verdict", "status"}

    @staticmethod
    def _parse_python_read_payload(output):
        try:
            parsed = json.loads(output)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            payload = AppWorldWMPredictor._parse_python_read_payload(output)
            if payload is not None:
                return payload
        if isinstance(parsed, dict):
            if "queries" in parsed or "facts" in parsed:
                return None  # Malformed tool requests still require repair.
            return [], parsed  # A generated inspection dictionary is already facts.
        if isinstance(output, str) and output.strip():
            # Debug-print inspections are generated computation, not API requests.
            # Preserve their stdout verbatim; never extract or execute embedded calls.
            return [], {"generated_stdout": output}
        return None

    def _run_python_prediction(self, code, context, **kwargs):
        result = super()._run_python_prediction(code, context, **kwargs)
        if result.error and "attribute '__name__' is not allowed" in result.error:
            result = replace(result, error=result.error + (
                "; use str(type(value)) or isinstance(value, dict/list) to inspect types. "
                "Do not access __name__ or any other dunder attribute."
            ))
        elif result.error and "Delete is not allowed" in result.error:
            result = replace(result, error=result.error + (
                "; use list.pop(index) for a list removal instead of del."
            ))
        elif result.error and "Assert is not allowed" in result.error:
            result = replace(result, error=result.error + (
                "; compute the condition as a Boolean and print the corresponding verdict "
                "with if/else instead of using assert."
            ))
        elif result.error and "attribute 'JSONDecodeError' is not allowed" in result.error:
            result = replace(result, error=result.error + (
                "; use json_value(value), or catch ValueError and TypeError instead."
            ))
        return result

    def _python_tool_limits(self):
        limits = super()._python_tool_limits()
        for field, suffix, default in (
            ("timeout_s", "TIMEOUT_S", 3.0),
            ("max_input_chars", "MAX_INPUT_CHARS", 2_000_000),
            ("max_output_chars", "MAX_OUTPUT_CHARS", 1_000_000),
            ("max_code_chars", "MAX_CODE_CHARS", 24_000),
            ("max_nodes", "MAX_AST_NODES", 4_000),
        ):
            # A native page of orders can exceed the old 20K output cap. Preserve
            # the complete generated result, with finite limits and the same
            # AST, import, network and filesystem restrictions as other methods.
            if f"TTS_APPWORLD_WM_PYTHON_{suffix}" in os.environ:
                default = limits[field]
            read_env = _env_float if field == "timeout_s" else _env_int
            limits[field] = read_env(f"TTS_APPWORLD_OURS_WM_PYTHON_{suffix}", default)
        return limits

    def _over_budget(self):
        if super()._over_budget():
            self.hard_timed_out = True
            raise WMHardTimeout(
                "AppWorld Ours prediction budget exhausted; refusing an unpredicted observation")
        return False
