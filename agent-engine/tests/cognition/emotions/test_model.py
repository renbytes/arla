# agent-engine/tests/cognition/emotions/test_model.py

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from agent_engine.cognition.emotions.appraisal_theory import AppraisalProcessor

# Subject under test
from agent_engine.cognition.emotions.model import EmotionalDynamics

# --- Fixtures ---


@pytest.fixture
def default_config():
    """Provides a mock config object with the full nested structure."""
    return SimpleNamespace(
        agent=SimpleNamespace(
            emotional_dynamics=SimpleNamespace(
                temporal={
                    "valence_decay_rate": 0.9,
                    "arousal_decay_rate": 0.8,
                    "valence_learning_rate": 0.2,
                    "arousal_learning_rate": 0.3,
                },
                noise_std=0.0,  # Set to 0 for predictable tests
                appraisal_weights=SimpleNamespace(goal_relevance=0.7, agency=0.5, social_feedback=0.3),
            )
        )
    )


@pytest.fixture
def mock_appraisal_processor(mocker):
    """Mocks the AppraisalProcessor to control its output."""
    mock_processor_class = mocker.patch("agent_engine.cognition.emotions.model.AppraisalProcessor")
    return mock_processor_class.return_value


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
    expected_valence = (0.9 * 0.5) + (0.2 * 0.8)  # 0.45 + 0.16 = 0.61
    expected_arousal = (0.8 * 0.4) + (0.3 * 0.7)  # 0.32 + 0.21 = 0.53

    # Act
    updated_state = emotional_dynamics.update_emotion_with_appraisal(current_emotion=current_emotion, **event_params)

    # Assert
    mock_appraisal_processor.appraise_event.assert_called_once_with(**event_params)
    mock_compute_valence.assert_called_once()
    mock_compute_arousal.assert_called_once()
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
    emotional_dynamics = EmotionalDynamics(default_config)
    current_emotion = {"valence": 0.9, "arousal": 0.1}
    event_params = {
        "prediction_error": 10.0,
        "current_goal": "test_goal",
        "action_success": True,
        "social_context": {},
        "controllability_estimate": 0.5,
    }

    # Act
    updated_state = emotional_dynamics.update_emotion_with_appraisal(current_emotion=current_emotion, **event_params)

    # Assert
    assert updated_state["valence"] == 1.0
    assert updated_state["arousal"] == 0.0


def test_update_emotion_with_noise(default_config):
    """
    Tests that emotional noise is applied when configured.
    """
    # Arrange: Modify the mock config object for this specific test
    default_config.agent.emotional_dynamics.noise_std = 0.1
    dynamics_with_noise = EmotionalDynamics(default_config)

    current_emotion = {"valence": 0.5, "arousal": 0.5}
    event_params = {
        "prediction_error": 0.0,
        "current_goal": None,
        "action_success": True,
        "social_context": {},
        "controllability_estimate": 0.5,
    }

    # Act
    results = [dynamics_with_noise.update_emotion_with_appraisal(current_emotion, **event_params) for _ in range(20)]
    valences = [r["valence"] for r in results]

    # Assert
    # With noise, the results should not all be identical
    assert len(set(valences)) > 1
