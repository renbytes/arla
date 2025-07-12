# tests/cognition/emotions/test_model.py

from unittest.mock import patch

import pytest
from agent_engine.cognition.emotions.appraisal_theory import AppraisalProcessor

# Subject under test
from agent_engine.cognition.emotions.model import EmotionalDynamics, update_emotion

# --- Fixtures ---


@pytest.fixture
def default_config():
    """Provides a default configuration for EmotionalDynamics."""
    return {
        "valence_decay_rate": 0.9,
        "arousal_decay_rate": 0.8,
        "valence_learning_rate": 0.2,
        "arousal_learning_rate": 0.3,
        "emotional_noise_std": 0.0,  # Set to 0 for predictable tests
    }


@pytest.fixture
def mock_appraisal_processor(mocker):
    """Mocks the AppraisalProcessor to control its output."""
    # Mock the entire class within the 'model' module's namespace
    mock_processor_class = mocker.patch("agent_engine.cognition.emotions.model.AppraisalProcessor")

    # Create an instance of the mock class to configure its methods
    mock_processor_instance = mock_processor_class.return_value
    return mock_processor_instance


# --- Test Cases for EmotionalDynamics ---


def test_initialization(default_config):
    """
    Tests that EmotionalDynamics initializes its attributes from the config correctly.
    """
    # Act
    dynamics = EmotionalDynamics(default_config)

    # Assert
    assert dynamics.valence_decay == 0.9
    assert dynamics.arousal_decay == 0.8
    assert dynamics.valence_learning_rate == 0.2
    assert dynamics.arousal_learning_rate == 0.3
    assert dynamics.emotional_noise == 0.0
    # On a clean run, the real processor should be instantiated.
    assert isinstance(dynamics.appraisal_processor, AppraisalProcessor)


@patch("agent_engine.cognition.emotions.model.compute_emotional_valence", return_value=0.8)
@patch("agent_engine.cognition.emotions.model.compute_emotional_arousal", return_value=0.7)
def test_update_emotion_with_appraisal(
    mock_compute_arousal, mock_compute_valence, mock_appraisal_processor, default_config
):
    """
    Tests the core logic of updating emotion, verifying decay and learning.
    """
    # Arrange
    # FIX: Instantiate EmotionalDynamics *after* the mock is in place.
    emotional_dynamics = EmotionalDynamics(default_config)
    current_emotion = {"valence": 0.5, "arousal": 0.4}
    event_params = {
        "prediction_error": 2.0,
        "current_goal": "test_goal",
        "action_success": True,
        "social_context": {},
        "controllability_estimate": 0.6,
    }

    # Expected values
    # new_valence = (0.9 * 0.5) + (0.2 * 0.8) = 0.45 + 0.16 = 0.61
    # new_arousal = (0.8 * 0.4) + (0.3 * 0.7) = 0.32 + 0.21 = 0.53
    expected_valence = 0.61
    expected_arousal = 0.53

    # Act
    updated_state = emotional_dynamics.update_emotion_with_appraisal(current_emotion=current_emotion, **event_params)

    # Assert
    # Check that the appraisal processor was called
    mock_appraisal_processor.appraise_event.assert_called_once_with(**event_params)

    # Check that the computation functions were called
    mock_compute_valence.assert_called_once()
    mock_compute_arousal.assert_called_once()

    # Check the final computed values
    assert updated_state["valence"] == pytest.approx(expected_valence)
    assert updated_state["arousal"] == pytest.approx(expected_arousal)
    assert "appraisal_dimensions" in updated_state
    assert updated_state["target_valence"] == 0.8
    assert updated_state["target_arousal"] == 0.7


@patch("agent_engine.cognition.emotions.model.compute_emotional_valence", return_value=2.0)
@patch("agent_engine.cognition.emotions.model.compute_emotional_arousal", return_value=-1.0)
def test_update_emotion_clipping(mock_compute_arousal, mock_compute_valence, mock_appraisal_processor, default_config):
    """
    Tests that the final valence and arousal values are correctly clipped.
    """
    # Arrange
    # FIX: Instantiate EmotionalDynamics *after* the mock is in place.
    emotional_dynamics = EmotionalDynamics(default_config)
    current_emotion = {"valence": 0.9, "arousal": 0.1}
    event_params = {
        "prediction_error": 10.0,
        "current_goal": "test_goal",
        "action_success": True,
        "social_context": {},
        "controllability_estimate": 0.5,  # Added to match method signature
    }

    # Act
    updated_state = emotional_dynamics.update_emotion_with_appraisal(current_emotion=current_emotion, **event_params)

    # Assert
    # new_valence = (0.9 * 0.9) + (0.2 * 2.0) = 0.81 + 0.4 = 1.21 -> clipped to 1.0
    # new_arousal = (0.8 * 0.1) + (0.3 * -1.0) = 0.08 - 0.3 = -0.22 -> clipped to 0.0
    assert updated_state["valence"] == 1.0
    assert updated_state["arousal"] == 0.0


def test_update_emotion_with_noise(default_config):
    """
    Tests that emotional noise is applied when configured.
    """
    # Arrange
    config_with_noise = default_config.copy()
    config_with_noise["emotional_noise_std"] = 0.1
    dynamics_with_noise = EmotionalDynamics(config_with_noise)

    current_emotion = {"valence": 0.5, "arousal": 0.5}
    event_params = {
        "prediction_error": 0.0,
        "current_goal": None,
        "action_success": True,
        "social_context": {},
        "controllability_estimate": 0.5,
    }

    # Act
    # Run multiple times to increase chance of seeing noise effect
    results = [dynamics_with_noise.update_emotion_with_appraisal(current_emotion, **event_params) for _ in range(20)]
    valences = [r["valence"] for r in results]

    # Assert
    # With noise, the results should not all be identical
    assert len(set(valences)) > 1


# --- Test Cases for legacy update_emotion function ---


@patch("agent_engine.cognition.emotions.model.EmotionalDynamics.update_emotion_with_appraisal")
def test_legacy_update_emotion_backward_compatibility(mock_update):
    """
    Tests that the legacy update_emotion function calls the new system correctly.
    """
    # Arrange
    current_emotion = {"valence": 0.1, "arousal": 0.2}
    prediction_error = 3.0

    # Act
    update_emotion(current_emotion, prediction_error)

    # Assert
    # The test correctly verifies that the legacy function calls the new
    # method without this parameter, relying on its default value.
    mock_update.assert_called_once_with(
        current_emotion=current_emotion,
        prediction_error=prediction_error,
        current_goal=None,
        action_success=True,  # Because prediction_error > 0
        social_context={},
    )
