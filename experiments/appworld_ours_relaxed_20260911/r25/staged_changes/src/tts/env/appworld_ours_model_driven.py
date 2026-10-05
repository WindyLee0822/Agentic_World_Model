"""Opt-in AppWorld Ours predictions without optimistic write-response shortcuts."""
from .appworld_ours_wm import AppWorldOursWMPredictor
from .appworld_wm import _api_doc, _failure_schema, _parse_json, WMPredictedHTTPError


class AppWorldOursModelDrivenPredictor(AppWorldOursWMPredictor):
    predicts_request_errors = True
    preserve_final_read_evidence = True

    def _predict(self, prompt, final_instr, **kwargs):
        context = kwargs.get("python_context") or {}
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
        return super()._predict(prompt, final_instr, **kwargs)

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
