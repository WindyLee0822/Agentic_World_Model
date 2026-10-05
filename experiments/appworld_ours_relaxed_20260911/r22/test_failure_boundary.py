import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).parent / 'staged_changes/src/tts/env/appworld_ours_failure_boundary.py'
spec = importlib.util.spec_from_file_location('failure_boundary', path)
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


class RenderingWorld:
    """Exercise the execution boundary that turns Python exceptions into text."""
    def __init__(self, error=None):
        self.error = error
        self.calls = 0
        self.requester = SimpleNamespace(request=self.request)

    def request(self):
        self.calls += 1
        if self.error:
            raise self.error
        return 'predicted observation'

    def execute(self, code):
        namespace = {'request': self.requester.request}
        try:
            exec(code, namespace)
            return namespace.get('result', 'cell completed')
        except Exception as exc:
            return 'Execution failed: ' + str(exc)


@pytest.mark.parametrize('code', [
    'result = request()',
    "try:\n    request()\nexcept Exception as e:\n    result = str(e)",
    "try:\n    request()\nexcept Exception:\n    result = request()",
])
@pytest.mark.parametrize('error', [RuntimeError('internal detail'), KeyError('internal key')])
def test_internal_errors_never_return_as_solver_observations(code, error):
    world = RenderingWorld(error)
    boundary.install_failure_boundary(world)
    with pytest.raises(boundary.SimulatorObservationFailure):
        world.execute(code)
    assert world.calls == 1
    with pytest.raises(boundary.SimulatorObservationFailure):
        world.execute('result = request()')
    assert world.calls == 1


def test_predicted_api_rejection_remains_an_observation():
    world = RenderingWorld(Exception('predicted API rejection'))
    boundary.install_failure_boundary(world)
    assert world.execute('request()') == 'Execution failed: predicted API rejection'
    world.error = None
    assert world.execute('result = request()') == 'predicted observation'


def test_normal_solver_python_error_does_not_poison_predictor():
    world = RenderingWorld()
    boundary.install_failure_boundary(world)
    assert 'division by zero' in world.execute('1 / 0')
    assert world.execute('result = request()') == 'predicted observation'


def test_plain_model_exception_is_not_a_predicted_api_rejection():
    world = RenderingWorld(Exception('plain internal model error'))
    predictor = SimpleNamespace(_predict=world.request)
    world.requester.request = lambda: predictor._predict()
    boundary.install_failure_boundary(world, predictor)
    with pytest.raises(boundary.SimulatorObservationFailure):
        world.execute('result = request()')


def test_typed_predicted_rejection_is_not_reclassified():
    class PredictedRejection(Exception):
        pass
    world = RenderingWorld(PredictedRejection('predicted rejection'))
    predictor = SimpleNamespace(_predict=world.request)
    def route():
        try:
            return predictor._predict()
        except PredictedRejection as exc:
            raise Exception(str(exc)) from None
    world.requester.request = route
    boundary.install_failure_boundary(world, predictor, PredictedRejection)
    assert world.execute('request()') == 'Execution failed: predicted rejection'


def test_failed_verifier_keeps_candidate_with_zero_score(monkeypatch):
    from tts.env.appworld_ours_wm import AppWorldOursWMPredictor
    from tts.env.appworld_wm import AppWorldWMPredictor

    predictor = object.__new__(AppWorldOursWMPredictor)
    def broken_program(*args, **kwargs):
        raise RuntimeError('Simulator program did not produce a valid verdict')
    predictor._predict = broken_program
    monkeypatch.setattr(AppWorldWMPredictor, 'predict_success',
                        lambda self: self._predict('verifier', 'verdict'))
    world = RenderingWorld()
    boundary.install_failure_boundary(world, predictor)
    assert world.execute('result = request()') == 'predicted observation'
    assert predictor.predict_success() == 0.0
    assert predictor.verifier_error.startswith('Simulator program did not produce')
    assert world.execute('result = request()') == 'predicted observation'


def test_predicted_rejection_restores_request_scope():
    class PredictedRejection(Exception):
        pass
    world = RenderingWorld()
    error = PredictedRejection('predicted rejection')
    def predict(*args, **kwargs):
        raise error
    predictor = SimpleNamespace(_predict=predict)
    def route():
        try:
            return predictor._predict()
        except PredictedRejection as exc:
            raise Exception(str(exc)) from None
    world.requester.request = route
    boundary.install_failure_boundary(world, predictor, PredictedRejection)
    assert 'predicted rejection' in world.execute('request()')
    error = RuntimeError('verifier error outside observation request')
    with pytest.raises(RuntimeError) as caught:
        predictor._predict()
    assert type(caught.value) is RuntimeError
    assert caught.value is error


def test_nested_prediction_failure_cannot_escape_request_scope():
    world = RenderingWorld()
    level = 0
    def predict():
        nonlocal level
        if level:
            raise Exception('plain internal nested failure')
        level += 1
        return world.requester.request()
    predictor = SimpleNamespace(_predict=predict)
    world.requester.request = lambda: predictor._predict()
    boundary.install_failure_boundary(world, predictor)
    with pytest.raises(boundary.SimulatorObservationFailure):
        world.execute('try:\n    request()\nexcept Exception as e:\n    result = str(e)')
