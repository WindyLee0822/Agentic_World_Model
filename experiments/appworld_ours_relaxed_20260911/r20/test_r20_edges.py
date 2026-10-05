from types import SimpleNamespace
import pytest
from tts.env.appworld_ours_model_driven import AppWorldOursModelDrivenPredictor
from tts.env.appworld_ours_setup import wrap_setup_predictions
from tts.env import appworld_wm

class Chat:
    def complete(self, *a, **k):
        return "PYTHON_FINAL:\nprint_json('Model-generated API metadata error')"

@pytest.mark.parametrize('raises', [True, False])
def test_missing_api_error_is_predicted(monkeypatch, capsys, raises):
    monkeypatch.setenv('TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS','0')
    monkeypatch.setattr(appworld_wm,'_api_doc',lambda *a: {})
    p=AppWorldOursModelDrivenPredictor(Chat());p.program_only=True;p.start_rollout()
    calls=[]
    def previous(*a, **k):
        calls.append(k)
        assert k['show'] is False
        if raises: raise Exception('native metadata error')
        return {'message':'native metadata error'}
    w=SimpleNamespace(requester=SimpleNamespace(request=previous))
    wrap_setup_predictions(w,p)
    if raises:
        with pytest.raises(Exception,match='Model-generated API metadata error'):
            w.requester.request('unknown','missing',show=True)
    else:
        # Model output must match the evidence encoding protocol for non-string input.
        p._chat=type('C',(),{'complete':lambda *a,**k: "PYTHON_FINAL:\nprint_json({'message':'Model-generated API metadata error'})"})()
        assert w.requester.request('unknown','missing',show=True,raise_on_failure=False)=={'message':'Model-generated API metadata error'}
    assert p.n_python_calls>0
    assert p.n_setup_predictions==1
    assert 'native metadata error' not in capsys.readouterr().out


def test_answer_and_completion_not_hardcoded_rejection():
    p=AppWorldOursModelDrivenPredictor(None);p.start_rollout()
    assert p.simulated_task_completion is None
    assert p.requires_operational_rejection() is False
    p.hard_timed_out=True
    assert p.requires_operational_rejection() is True
