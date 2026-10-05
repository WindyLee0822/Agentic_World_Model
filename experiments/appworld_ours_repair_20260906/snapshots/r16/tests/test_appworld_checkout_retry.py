import pytest
from tts.env.appworld_ours_wm import AppWorldOursWMPredictor
from tts.env.appworld_wm import AppWorldWMPredictor

class NoModel:
    def complete(self, *args, **kwargs):
        raise AssertionError('Fixture bookkeeping should not call the model')

@pytest.mark.parametrize('predictor_type, resolved', [(AppWorldOursWMPredictor, True), (AppWorldWMPredictor, False)])
def test_successful_alternate_card_checkout_can_complete_only_in_ours(monkeypatch, predictor_type, resolved):
    monkeypatch.setenv('TTS_APPWORLD_COMPLETION_GUARD', '1')
    p = predictor_type(NoModel())
    first = {'payment_card_id': 1, 'address_id': 2}
    retry = {'payment_card_id': 3, 'address_id': 2}
    p._record_rejected_write('amazon', 'place_order', first, {'message': 'Expired card'})
    p._record_applied_write('amazon', 'place_order', retry, {'order_id': 9}, p._write_key('amazon', 'place_order', retry))
    p.simulate_task_completion({})
    assert (p.simulated_task_completion is not None) is resolved
    assert p.n_completion_deferrals == (0 if resolved else 1)

def test_checkout_to_another_address_does_not_resolve_original_failure(monkeypatch):
    monkeypatch.setenv('TTS_APPWORLD_COMPLETION_GUARD', '1')
    p = AppWorldOursWMPredictor(NoModel())
    first = {'payment_card_id': 1, 'address_id': 2}
    other = {'payment_card_id': 3, 'address_id': 4}
    p._record_rejected_write('amazon', 'place_order', first, {'message': 'Expired card'})
    p._record_applied_write('amazon', 'place_order', other, {'order_id': 9}, p._write_key('amazon', 'place_order', other))
    p.simulate_task_completion({})
    assert p.simulated_task_completion is None
    assert p.n_completion_deferrals == 1
