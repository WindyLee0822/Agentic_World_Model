import pytest
from tts.env.appworld_ours_model_driven import AppWorldOursModelDrivenPredictor as Experimental
from tts.env.appworld_ours_wm import AppWorldOursWMPredictor as Original

@pytest.mark.parametrize('cls,expected_calls', [(Experimental, 2), (Original, 4)])
def test_final_program_can_use_fresh_evidence_without_extra_inspection(monkeypatch, cls, expected_calls):
    monkeypatch.setenv('TTS_APPWORLD_WM_PYTHON_READ_TURNS', '5')
    p = cls(None)
    p.start_rollout()
    final = 'PYTHON_FINAL:\nprint_json({"verdict": "SUCCESS" if data["reads"][0]["result"]["observed"] == 17 else "FAIL"})'
    programs = iter([
        'PYTHON_READ:\nprint_json({"queries":[{"app":"test","api":"read","args":{}}]})',
        final,
        'PYTHON_READ:\nprint_json({"queries":[],"facts":{"observed":data["reads"][0]["result"]["observed"]}})',
        final,
    ])
    calls = []
    def complete(prompt, **kwargs):
        calls.append(prompt)
        return next(programs)
    monkeypatch.setattr(p, '_complete', complete)
    reads = []
    def read(app, api, args):
        reads.append((app, api))
        return {'observed': 17}
    output = p._python_read_loop('Verify', 'VERDICT: SUCCESS or FAIL', context={},
        read_fn=read, max_tokens=2048, temperature=0, budgeted=False,
        python_required=True, output_kind='verdict')
    assert 'SUCCESS' in output
    assert len(calls) == expected_calls
    assert reads == [('test', 'read')]
    assert p.n_python_calls > 0


def test_context_protocol_is_opt_in_and_does_not_expose_values():
    context = {'args': {'secret': 'argument-marker'}, 'facts': {'secret': 'fact-marker'}}
    kwargs = dict(required=True, context_chars=100, output_kind='verdict', allow_reads=True)
    new = Experimental._python_protocol(context, **kwargs)
    old = Original._python_protocol(context, **kwargs)
    assert new.startswith(old)
    assert 'INSPECTION IS OPTIONAL' not in old
    assert "data['facts']['file_exists_result']" in new
    assert 'argument-marker' not in new and 'fact-marker' not in new
