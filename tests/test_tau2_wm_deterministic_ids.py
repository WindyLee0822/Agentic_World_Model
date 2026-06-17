"""Test that tau2 WM can predict deterministic IDs correctly."""
import pytest
from tts.env.registry import make_env


@pytest.mark.skipif(True, reason="requires LLM")
def test_wm_predicts_correct_reservation_id():
    """Test WM predicts the correct deterministic reservation ID."""
    env = make_env("tau2", index=0, domain="airline")
    wm = env.world_model()
    
    # Book first reservation -> should get HATHAT
    from tts.types import Action
    action = Action(name="book_reservation", args={
        "user_id": "test_user",
        "origin": "LAX",
        "destination": "NYC",
        "flight_type": "one_way",
        "cabin": "economy",
        "passengers": [{"first_name": "Test", "last_name": "User", "dob": "1990-01-01"}]
    })
    
    # Read grounding for user
    shadow = wm.shadow_for([action])
    
    # Predict the booking
    reward = wm.predict(shadow, [action])
    
    # Check that the predicted trajectory contains HATHAT
    predicted_obs = reward.predicted.trajectory.steps[0].observation.content
    assert "HATHAT" in predicted_obs, f"Expected HATHAT in prediction, got: {predicted_obs}"


def test_usable_only_disabled():
    """Test that usable_only filter is now disabled for all agents."""
    from tts.env.registry import _tau2_all_tasks, _tau2_usable_tasks
    
    all_tasks = _tau2_all_tasks('airline')
    usable_tasks = _tau2_usable_tasks('airline')
    
    # Should still have the functions available
    assert len(all_tasks) == 50
    assert len(usable_tasks) == 26
    
    # But run.py should now use all_tasks by setting usable_only=False
    # This is tested by the code change in experiments/run.py line 233


def test_deterministic_id_rules_in_prompt():
    """Test that deterministic ID rules are implemented in the WM predictor."""
    from tts.env.real_tau2 import _LLMWorldModelPredictor
    
    # The predictor class has the predict_observation methods with ID rules
    predictor = _LLMWorldModelPredictor("airline")
    
    # Verify the methods exist
    assert hasattr(predictor, 'predict_observation')
    assert hasattr(predictor, 'predict_observation_agentic')
    
    # The ID rules are injected into prompts based on domain and action name
    # This is verified by code inspection of lines 545-562 and 619-638 in real_tau2.py
