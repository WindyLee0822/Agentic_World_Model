"""Boundary tests for the isolated experiment; no live-model claims."""

import sys
import json
import types
import unittest
from unittest.mock import patch

from routing import wrap_setup_predictions


class Predictor:
    program_only = True

    def __init__(self):
        self.calls = []
        self.fail = False

    def predict_read(self, app, api, args, evidence, schema, *, dirty):
        self.calls.append((app, api, args, evidence, dirty))
        if self.fail:
            raise RuntimeError("prediction failed")
        return json.dumps({"generated": json.loads(evidence)})

    def note_read(self, *args):
        pass


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {
            "tts.env.appworld_ours_wm": types.SimpleNamespace(AppWorldOursWMPredictor=Predictor),
            "tts.env.appworld_wm": types.SimpleNamespace(_success_schema=lambda *args: {}),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict("os.environ", {"TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.events = []

        def existing(app, api, **kwargs):
            self.events.append((app, api))
            return {"status": "complete"} if api == "complete_task" else {"access_token": "opaque"}

        self.world = types.SimpleNamespace(requester=types.SimpleNamespace(request=existing))
        self.predictor = Predictor()

    def test_setup_observations_are_generated_and_completion_uses_logical_evidence(self):
        wrap_setup_predictions(self.world, self.predictor)
        for app, api in [("supervisor", "show_active_task"), ("api_docs", "show_api_doc"),
                         ("gmail", "login"), ("supervisor", "complete_task")]:
            result = self.world.requester.request(app, api)
            self.assertIn("generated", result)
        self.assertEqual(len(self.predictor.calls), 4)
        self.assertEqual(json.loads(self.predictor.calls[-1][3]), {"status": "complete"})
        self.assertEqual(len(self.events), 4)

    def test_native_keyword_binding(self):
        wrap_setup_predictions(self.world, self.predictor)
        result = self.world.requester.request(_app_name="gmail", _api_name="login", username="fixture")
        self.assertIn("generated", result)
        self.assertEqual(self.predictor.calls[0][2], {"username": "fixture"})

    def test_failure_does_not_return_exact_evidence(self):
        wrap_setup_predictions(self.world, self.predictor)
        self.predictor.fail = True
        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            self.world.requester.request("gmail", "login")

    def test_substantive_routing_and_non_ours_are_preserved(self):
        prior = self.world.requester.request
        with self.assertRaises(TypeError):
            wrap_setup_predictions(self.world, object())
        self.assertIs(self.world.requester.request, prior)
        wrap_setup_predictions(self.world, self.predictor)
        self.assertEqual(self.world.requester.request("gmail", "send_email"), {"access_token": "opaque"})
        self.assertEqual(self.predictor.calls, [])

    def test_native_api_error_is_prediction_evidence(self):
        def rejected(*args, **kwargs):
            raise Exception('Response status code is 401:\n{"message":"invalid credentials"}')

        self.world.requester.request = rejected
        self.predictor.predict_read = lambda *args, **kwargs: json.dumps("generated authentication error")
        wrap_setup_predictions(self.world, self.predictor)
        with self.assertRaisesRegex(Exception, "^generated authentication error$"):
            self.world.requester.request("gmail", "login")

    def test_infrastructure_error_is_not_converted_into_an_observation(self):
        failure = TimeoutError("transport unavailable")

        def unavailable(*args, **kwargs):
            raise failure

        self.world.requester.request = unavailable
        wrap_setup_predictions(self.world, self.predictor)
        with self.assertRaises(TimeoutError) as caught:
            self.world.requester.request("gmail", "login")
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.predictor.calls, [])


if __name__ == "__main__":
    unittest.main()
