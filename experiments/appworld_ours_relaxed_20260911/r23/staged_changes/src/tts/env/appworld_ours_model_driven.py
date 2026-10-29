"""Opt-in AppWorld Ours predictions without optimistic write-response shortcuts."""
from .appworld_ours_wm import AppWorldOursWMPredictor
from .appworld_wm import _api_doc, _failure_schema, _parse_json, WMPredictedHTTPError


class AppWorldOursModelDrivenPredictor(AppWorldOursWMPredictor):
    predicts_request_errors = True
    preserve_final_read_evidence = True

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
            "Inspect the full API documentation and success/failure schemas in data. "
            "Decide whether this call succeeds or fails from its arguments, documented "
            "preconditions, and current logical state. Use read-only evidence tools to "
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
