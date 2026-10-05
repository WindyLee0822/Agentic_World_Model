"""Exercise actual Ours Python execution with scripted model programs."""

from types import SimpleNamespace

import pytest

from tts.env.appworld_ours_wm import AppWorldOursWMPredictor
from routing import wrap_setup_predictions


class ScriptChat:
    def __init__(self, program):
        self.program = program
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        return self.program


@pytest.mark.parametrize('app,api,evidence', [
    ('api_docs', 'show_api_doc', {'parameters': [{'name': 'access_token'}]}),
    ('gmail', 'login', {'access_token': 'opaque.token-value', 'token_type': 'bearer'}),
    ('supervisor', 'show_active_task', {'status': 'complete', 'answer': None}),
    ('supervisor', 'show_account_passwords', [{'app': 'gmail', 'password': 'synthetic-test-password'}]),
    ('supervisor', 'complete_task', None),
])
def test_actual_program_produces_setup_response(monkeypatch, app, api, evidence):
    monkeypatch.setenv('TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS', '0')
    chat = ScriptChat("PYTHON_FINAL:\nprint_json(data['grounded_clean_read'])")
    predictor = AppWorldOursWMPredictor(chat)
    predictor.program_only = True
    predictor.start_rollout()
    world = SimpleNamespace(requester=SimpleNamespace(request=lambda *a, **k: evidence))
    wrap_setup_predictions(world, predictor)
    assert world.requester.request(app, api) == evidence
    assert chat.calls > 0
    assert predictor.n_python_calls > 0
    assert predictor.n_predicted_reads == 1


def test_malformed_model_output_cannot_return_setup_evidence(monkeypatch):
    monkeypatch.setenv('TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS', '0')
    predictor = AppWorldOursWMPredictor(ScriptChat('not a program'), max_llm_calls=2)
    predictor.program_only = True
    predictor.start_rollout()
    world = SimpleNamespace(requester=SimpleNamespace(request=lambda *a, **k: {'token': 'secret'}))
    wrap_setup_predictions(world, predictor)
    with pytest.raises((RuntimeError, TimeoutError)):
        world.requester.request('gmail', 'login')
