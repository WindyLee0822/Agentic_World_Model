import json
from types import SimpleNamespace
from tts.env.appworld_ours_model_driven import AppWorldOursModelDrivenPredictor as Predictor
from tts.env.appworld_ours_wm import AppWorldOursWMPredictor, _DiagnosticChat

def test_full_response_logging_preserves_response_and_redacts_token(tmp_path,monkeypatch):
    p=Predictor(None);p._chat=_DiagnosticChat(None,tmp_path)
    response='eyJabc.def.ghi '+('x'*33000)
    monkeypatch.setattr(AppWorldOursWMPredictor,'_complete',lambda *a,**k:response)
    assert p._complete('prompt',max_tokens=1)==response
    p._chat.stream.close()
    event=json.loads(next(tmp_path.glob('*.jsonl')).read_text())
    assert 'eyJabc' not in event['response']
    assert len(event['response'])<=32000
    assert event['response_chars']==len(response)

def test_program_error_logging_does_not_change_result_or_dump_context(tmp_path,monkeypatch):
    p=Predictor(None);p._chat=_DiagnosticChat(None,tmp_path)
    result=SimpleNamespace(error='specific failure',fatal=False,output='')
    monkeypatch.setattr(AppWorldOursWMPredictor,'_run_python_prediction',lambda *a,**k:result)
    assert p._run_python_prediction('bad program',{'gold':'private-context-value'}) is result
    p._chat.stream.close()
    raw=next(tmp_path.glob('*.jsonl')).read_text()
    assert 'private-context-value' not in raw
    assert json.loads(raw)['error']=='specific failure'

def test_closed_diagnostic_stream_does_not_mask_prediction(tmp_path,monkeypatch):
    p=Predictor(None);p._chat=_DiagnosticChat(None,tmp_path);p._chat.stream.close()
    monkeypatch.setattr(AppWorldOursWMPredictor,'_complete',lambda *a,**k:'unchanged')
    assert p._complete('prompt',max_tokens=1)=='unchanged'
