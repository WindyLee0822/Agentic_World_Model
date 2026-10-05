"""Opt-in AppWorld Ours predictions without optimistic write-response shortcuts."""
import json
import re

from .appworld_ours_wm import AppWorldOursWMPredictor, _DiagnosticChat
from .appworld_wm import _api_doc, _failure_schema, _parse_json, WMPredictedHTTPError


class AppWorldOursModelDrivenPredictor(AppWorldOursWMPredictor):
    predicts_request_errors = True
    preserve_final_read_evidence = True

    def _requires_read_inspection(self, context, output_kind):
        # A generated final program can inspect the same copied evidence itself.
        # Do not discard it just because it omitted a separate facts-only round.
        return False

    @staticmethod
    def _python_protocol(context, **kwargs):
        protocol = AppWorldOursWMPredictor._python_protocol(context, **kwargs)
        return protocol + (
            "\nCONTEXT ACCESS: The printed TOOL FACTS AND ERRORS dictionary is "
            "data['facts'], not the top level of data. A fact named file_exists_result "
            "is data['facts']['file_exists_result']; it is not data['file_exists_result']. "
            "Read values are json_value(data['reads'][i]['result']); locate the matching "
            "record by its app, api and args. The request is data['args']. Successful "
            "transitions are data['writes'], with args and result, not result_summary. "
            "Inspect actual keys instead of inventing aliases or wrapper fields.\n"
            "PYTHON HELPERS: json_value and print_json are already defined. Use "
            "str(type(value)) or isinstance, not type(value).__name__. Use json_value "
            "instead of defining a JSON decoder; catch ValueError/TypeError if needed. "
            "Use Boolean conditions and if/else, not assert, dir, hash or __import__.\n"
            "INSPECTION IS OPTIONAL: You may inspect evidence and compute the verdict "
            "in one PYTHON_FINAL program. Request a facts-only PYTHON_READ round when "
            "you need to see intermediate values, not merely to satisfy a protocol. "
            "This overrides instructions requiring a separate inspection round; "
            "required grounding reads and the final output format still apply.\n"
        )

    @staticmethod
    def _parse_python_read_payload(output):
        # Accept a terminal request envelope after generated inspection output.
        # Never search arbitrary embedded text for calls. Endpoint execution is
        # still subject to the existing read-only query interface and limits.
        if isinstance(output, str):
            prefix, separator, last = output.rstrip().rpartition("\n")
            if separator and prefix.strip():
                try:
                    terminal = json.loads(last)
                except ValueError:
                    terminal = None
                if isinstance(terminal, dict) and "queries" in terminal:
                    terminal["facts"] = {
                        "generated_stdout": prefix,
                        "terminal_facts": terminal.get("facts"),
                    }
                    return AppWorldOursWMPredictor._parse_python_read_payload(
                        json.dumps(terminal))
        return AppWorldOursWMPredictor._parse_python_read_payload(output)

    def _diagnostic_event(self, **event):
        if not isinstance(self._chat, _DiagnosticChat):
            return
        try:
            encoded = json.dumps({**self._chat.context, **event}, default=str)
            encoded = re.sub(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "<redacted>", encoded)
            self._chat.stream.write(encoded + "\n")
            self._chat.stream.flush()
        except (OSError, ValueError):
            pass  # Diagnostics must not change an observation or verdict.

    def _complete(self, prompt, **kwargs):
        response = super()._complete(prompt, **kwargs)
        self._diagnostic_event(event="complete_response", response=str(response)[:32000],
                               response_chars=len(response or ""))
        return response

    def _run_python_prediction(self, code, context, **kwargs):
        result = super()._run_python_prediction(code, context, **kwargs)
        if result.error:
            self._diagnostic_event(event="program_error", error=result.error,
                                   fatal=result.fatal, code=code[:24000],
                                   mode=context.get("mode"), app=context.get("app"),
                                   api=context.get("api"), context_keys=list(context))
        return result

    def _predict(self, prompt, final_instr, **kwargs):
        context = kwargs.get("python_context") or {}
        if context.get("mode") == "write_with_rejection":
            contract = {key: context.get(key) for key in (
                "api_documentation", "success_schema", "failure_schema")}
            prompt += (
                "\nCURRENT ENDPOINT CONTRACT (static documentation, not an executed response):\n"
                + json.dumps(contract, ensure_ascii=True, default=str)
                + "\nMatch the documented response fields and types. Example IDs in a "
                "schema describe types, not the actual ID to return. Derive predicted "
                "values from available state evidence and simulated transitions.\n"
            )
        if kwargs.get("python_output_kind") in {"verdict", "status"}:
            prompt += (
                "\nVERIFIER STATE SEMANTICS: Grounding queries return PRISTINE state, "
                "before the candidate's simulated writes. The writes were never applied "
                "to the native database. In generated Python, construct logical final "
                "state by applying the ordered successful entries in data['writes'] "
                "to relevant pristine evidence before checking final-state criteria. "
                "A newly created record is normally absent from pristine query results; "
                "that absence alone does not mean its simulated creation failed. "
                "Likewise a deleted record can still appear in pristine results. "
                "Derive effects from API semantics and actual write arguments, not "
                "from the expected gold state. Track missing, extra and incorrect "
                "changes; a success response alone is not proof of task correctness. "
                "Do not apply transitions twice to evidence already explicitly "
                "identified as a computed logical final state.\n"
                "COMPARE FINAL-STATE MEANING: API parameters and stored model fields "
                "can encode the same value differently. In generated Python, derive "
                "the stored field using documented API semantics before comparing "
                "it with the expected value (for example a Boolean option versus a "
                "text enum, or an email parameter versus a resolved user ID). Inspect "
                "the expected value's type and relevant read schema. Do not compare "
                "different representations directly or coerce arbitrary values with "
                "bool(). Resolve recipient identities from the actual write arguments "
                "and matching read evidence; never assign a candidate field from gold "
                "and then compare that assigned field with gold. Gold specifies the "
                "expected state only. Preserve all requested checks during repairs.\n"
            )
        if context.get("mode") in {"clean_read", "dirty_read"}:
            prompt += (
                "\nREAD OUTPUT SERIALIZATION: An untouched read result can be a JSON "
                "object, encoded JSON, or plain-text error evidence from an invalid "
                "request. Use json_value(value) to decode only when possible; do not "
                "unconditionally json.loads a string. Serialize the final predicted "
                "value with print_json(value), including when that value is a string. "
                "Printing an unquoted error string is not valid JSON. JSON string "
                "quoting is transport serialization, not an extra response wrapper. "
                "Preserve the predicted value and do not invent an empty object "
                "when the evidence is an error.\n"
            )
        try:
            return super()._predict(prompt, final_instr, **kwargs)
        except Exception as exc:
            self._diagnostic_event(event="prediction_error", error_type=type(exc).__name__,
                                   error=str(exc), mode=context.get("mode"),
                                   app=context.get("app"), api=context.get("api"))
            raise

    def requires_operational_rejection(self):
        # Answer correctness and goal satisfaction belong to the simulator verifier.
        return self.hard_timed_out

    def predict_write(self, app, api, args, schema):
        self.n_writes += 1
        self.first_write_seen = True
        self._check_hard_deadline()
        self._over_budget()  # Ours raises; never fabricate an observation on exhaustion.
        key = self._write_key(app, api, args)
        context = {
            "mode": "write_with_rejection", "app": app, "api": api,
            "args": args, "api_documentation": _api_doc(app, api),
            "success_schema": schema, "failure_schema": _failure_schema(app, api),
            "is_repeat": key in self.applied_writes,
            "frozen_datetime": self.frozen_datetime,
            "history": [{"app": a, "api": b, "name": f"{a}.{b}",
                         "args": c, "result": d} for a, b, c, d in self.history],
        }
        response = self._predict(
            "Predict the exact response to data['app'].data['api'] with data['args']. "
            "The supplied Python variable data already contains the request and evidence; "
            "use it directly, without replacing it with a copied or invented dictionary. "
            "data['api_documentation'] is ONE dictionary for this exact endpoint, "
            "not a list of endpoints: inspect its keys and access fields by key. "
            "Its response_schemas dictionary contains success and failure schemas; "
            "data['success_schema'] and data['failure_schema'] also supply those shapes. "
            "A failure schema describes the response shape, not a list of rejection "
            "conditions. Do not assume a failure_responses field or iterate dictionary "
            "keys as endpoint objects. Inspect parameter definitions and descriptions. "
            "Decide whether this call succeeds or fails from its arguments, documented "
            "preconditions, and current logical state. Do not invent prerequisites from "
            "real-world expectations about an app. Fields returned by a read do not "
            "by themselves establish restrictions on this write. Ground each rejection "
            "condition in the endpoint contract and relevant state evidence. "
            "Use read-only evidence tools to "
            "resolve missing facts. First-time calls can fail; repeated calls can succeed. "
            "Predict parameter-validation errors as well as state-dependent errors. "
            "Do not execute the write. Return an envelope with outcome ('success' or "
            "'failure'), integer HTTP status_code, and response (the exact JSON body).",
            'PREDICT: {"outcome":"success"|"failure","status_code":<integer>,"response":<JSON>}',
            max_tokens=1200, python_context=context, python_force=self.program_only,
        )
        envelope = _parse_json(response)
        if not isinstance(envelope, dict) or not {"outcome", "status_code", "response"} <= envelope.keys():
            raise RuntimeError("Simulator did not predict a complete write response envelope")
        outcome, status = envelope["outcome"], envelope["status_code"]
        if type(status) is not int or not (
            outcome == "success" and 200 <= status < 300
            or outcome == "failure" and 400 <= status < 600
        ):
            raise RuntimeError("Simulator predicted an inconsistent write response envelope")
        body = envelope["response"]
        if outcome == "failure":
            self._record_rejected_write(app, api, args, body)
            raise WMPredictedHTTPError(status, body)
        return self._record_applied_write(app, api, args, body, key)
