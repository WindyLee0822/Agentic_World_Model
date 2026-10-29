import json
import pytest
from tts.env import appworld_ours_model_driven as m
from tts.env.appworld_ours_wm import AppWorldOursWMPredictor

@pytest.fixture
def predictor(monkeypatch):
    p = m.AppWorldOursModelDrivenPredictor(None)
    p.start_rollout()
    p.program_only = True
    monkeypatch.setattr(m, '_api_doc', lambda *a: {'parameters': [{'name': 'required', 'required': True}], 'description': 'Full documentation'})
    monkeypatch.setattr(m, '_failure_schema', lambda *a: {'message': 'string'})
    return p

@pytest.mark.parametrize('args', [{}, {'required': 'value'}])
def test_first_trivial_write_can_be_model_rejected(predictor, monkeypatch, args):
    seen = []
    def predict(*a, **kw):
        seen.append(kw)
        return json.dumps({'outcome': 'failure', 'status_code': 422, 'response': {'message': 'Model rejection'}})
    monkeypatch.setattr(predictor, '_predict', predict)
    with pytest.raises(m.WMPredictedHTTPError):
        predictor.predicted_write('test', 'write', args, {'message': 'string'})
    assert len(seen) == 1
    assert seen[0]['python_context']['args'] == args
    assert seen[0]['python_context']['api_documentation']['parameters'][0]['required']
    assert seen[0]['python_context']['mode'] == 'write_with_rejection'
    assert seen[0]['python_force'] is True
    assert not predictor.applied_writes

@pytest.mark.parametrize('response', ['garbage', '{}', '{"outcome":"success","status_code":422,"response":{}}'])
def test_malformed_prediction_never_synthesizes_success(predictor, monkeypatch, response):
    monkeypatch.setattr(predictor, '_predict', lambda *a, **k: response)
    with pytest.raises(RuntimeError):
        predictor.predicted_write('test', 'write', {}, {})
    assert not predictor.applied_writes


def test_model_success_updates_logical_history(predictor, monkeypatch):
    body = {'message': 'Generated success'}
    monkeypatch.setattr(predictor, '_predict', lambda *a, **k: json.dumps({'outcome':'success','status_code':200,'response':body}))
    assert predictor.predicted_write('test', 'write', {}, {}) == body
    assert len(predictor.write_history) == 1
    assert predictor.applied_writes


def test_default_ours_predictor_keeps_original_write_method():
    assert AppWorldOursWMPredictor.predict_write is not m.AppWorldOursModelDrivenPredictor.predict_write

@pytest.mark.parametrize('outcome,status', [('success', 200), ('failure', 422)])
def test_actual_simulator_program_decides_write(predictor, outcome, status):
    class Chat:
        def complete(self, *a, **kw):
            return 'PYTHON_FINAL:\nprint_json(' + repr({'outcome': outcome, 'status_code': status, 'response': {'message': 'Predicted by program'}}) + ')'
    predictor._chat = Chat()
    if outcome == 'failure':
        with pytest.raises(m.WMPredictedHTTPError):
            predictor.predicted_write('test', 'write', {}, {})
    else:
        assert predictor.predicted_write('test', 'write', {}, {}) == {'message': 'Predicted by program'}
    assert predictor.n_python_calls > 0
