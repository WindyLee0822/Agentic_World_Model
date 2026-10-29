import json
import pytest
from tts.env.appworld_wm import AppWorldWMPredictor
from tts.env.appworld_ours_wm import AppWorldOursWMPredictor
from tts.env.appworld_ours_model_driven import AppWorldOursModelDrivenPredictor


@pytest.mark.parametrize('cls,enabled', [
    (AppWorldWMPredictor, False),
    (AppWorldOursWMPredictor, False),
    (AppWorldOursModelDrivenPredictor, True),
])
def test_final_request_retains_latest_evidence_only_for_opt_in(cls, enabled, monkeypatch):
    monkeypatch.setenv('TTS_APPWORLD_WM_PYTHON_READ_TURNS', '2')
    predictor = cls(None)
    if isinstance(predictor, AppWorldOursWMPredictor):
        predictor.start_rollout()
    prompts = []
    responses = iter([
        'PYTHON_READ:\nprint_json({"queries":[{"app":"file_system","api":"file_exists","args":{}}]})',
        'PYTHON_READ:\nprint_json({"queries":[{"app":"file_system","api":"show_file","args":{}}],"facts":{"last_turn_exists":data["reads"][0]["result"]["exists"]}})',
        'PYTHON_FINAL:\nprint_json({"confirmed":data["facts"]["python_computed_facts_2"]["last_turn_exists"]})',
    ])
    def complete(prompt, **kwargs):
        prompts.append(prompt)
        return next(responses)
    monkeypatch.setattr(predictor, '_complete', complete)
    reads = []
    def read(app, api, args):
        reads.append((app, api))
        return {'exists': True} if api == 'file_exists' else {'uninspected_value': 'not-visible-raw'}
    result = predictor._python_read_loop(
        'Predict a test observation', 'PREDICT: <JSON>', context={}, read_fn=read,
        max_tokens=2048, temperature=0, budgeted=True,
        python_required=True, output_kind='json')
    assert len(reads) == 2
    assert len(prompts) == 3
    final = prompts[-1]
    assert ('last_turn_exists' in final) is enabled
    assert ('file_system.show_file' in final) is enabled
    assert 'not-visible-raw' not in final
    assert 'true' in result.lower()
    if not enabled:
        # Byte-for-byte legacy final request, not just equivalent parsing.
        expected = prompts[0].split('\n\nCOPIED READ RESULTS SO FAR:')[0]
        expected += '\n\nYou are out of read turns. Return the final answer directly or with `PYTHON_FINAL:`; do not request more reads.'
        assert final == expected
