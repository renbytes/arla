# tests/cognition/emotions/test_appraisal_theory.py

from types import SimpleNamespace

import pytest

# Subject under test
from agent_engine.cognition.emotions.appraisal_theory import (
    AppraisalDimensions,
    AppraisalProcessor,
    compute_emotional_arousal,
    compute_emotional_valence,
)

# Fixtures


@pytest.fixture
def appraisal_processor():
    """Provides a default AppraisalProcessor instance."""
    # Create a mock config with the nested structure the class now expects.
    mock_config = SimpleNamespace(
        agent=SimpleNamespace(
            emotional_dynamics=SimpleNamespace(
                appraisal_weights=SimpleNamespace(goal_relevance=0.7, agency=0.5, social_feedback=0.3)
            )
        )
    )
    return AppraisalProcessor(mock_config)


@pytest.fixture
def base_appraisal():
    """Provides a neutral AppraisalDimensions instance."""
    return AppraisalDimensions(
        goal_relevance=0.5,
        goal_congruence=0.0,
        agency=0.5,
        controllability=0.5,
        certainty=0.5,
        social_approval=0.0,
    )


# Test Cases for AppraisalProcessor


def test_appraise_event_positive_outcome(appraisal_processor):
    """
    Tests appraisal for a successful, goal-relevant event.
    """
    # Arrange
    event_params = {
        "prediction_error": 5.0,
        "current_goal": "achieve_something",
        "action_success": True,
        "social_context": {},
        "controllability_estimate": 0.8,
    }

    # Act
    appraisal = appraisal_processor.appraise_event(**event_params)

    # Assert
    assert appraisal.goal_relevance == pytest.approx(0.6)  # (5.0 / 10.0) * 1.2
    assert appraisal.goal_congruence == pytest.approx(1.0)  # 5.0 / 5.0, clipped
    assert appraisal.agency == pytest.approx(0.56)  # 0.7 * 0.8
    assert appraisal.controllability > 0
    assert appraisal.certainty == pytest.approx(0.1)  # 1.0 - 5.0 / 5.0, with min
    assert appraisal.social_approval == 0.0


def test_appraise_event_negative_outcome(appraisal_processor):
    """
    Tests appraisal for a failed, goal-relevant event with social context.
    """
    # Arrange
    event_params = {
        "prediction_error": -8.0,
        "current_goal": "avoid_danger",
        "action_success": False,
        "social_context": {"other_agents_present": True, "action_intent": "COMPETE"},
        "controllability_estimate": 0.3,
    }

    # Act
    appraisal = appraisal_processor.appraise_event(**event_params)

    # Assert
    assert appraisal.goal_relevance == pytest.approx(0.88)  # (8.0 / 10.0) * 1.1
    assert appraisal.goal_congruence == pytest.approx(-1.0)  # -8.0 / 5.0, clipped
    assert appraisal.agency == pytest.approx(0.09)  # 0.3 * 0.3
    assert appraisal.controllability == pytest.approx(0.16)  # (1 - 8/10) * 0.8
    assert appraisal.certainty == pytest.approx(0.1)  # 1 - 8/5, with min
    assert appraisal.social_approval == -0.3


def test_appraise_event_no_goal(appraisal_processor):
    """
    Tests appraisal for an event with no active goal.
    """
    # Arrange
    event_params = {
        "prediction_error": 0.5,
        "current_goal": None,
        "action_success": True,
        "social_context": {},
        "controllability_estimate": 0.5,
    }

    # Act
    appraisal = appraisal_processor.appraise_event(**event_params)

    # Assert
    assert appraisal.goal_relevance == 0.3  # Default when no goal


# Test Cases for compute_emotional_valence


def test_compute_valence_positive(base_appraisal):
    """
    Tests that positive goal congruence leads to positive valence.
    """
    base_appraisal.goal_congruence = 0.8
    valence = compute_emotional_valence(base_appraisal)
    assert valence > 0


def test_compute_valence_negative(base_appraisal):
    """
    Tests that negative goal congruence leads to negative valence.
    """
    base_appraisal.goal_congruence = -0.8
    valence = compute_emotional_valence(base_appraisal)
    assert valence < 0


def test_compute_valence_clipping(base_appraisal):
    """
    Tests that valence is correctly clipped to the [-1.0, 1.0] range.
    """
    # Arrange for a very high potential valence
    base_appraisal.goal_relevance = 1.0
    base_appraisal.goal_congruence = 1.0
    base_appraisal.agency = 1.0
    base_appraisal.controllability = 1.0
    base_appraisal.social_approval = 1.0

    # Act
    valence = compute_emotional_valence(base_appraisal)

    # Assert
    assert valence == 1.0

    # Arrange for a very low potential valence by removing positive contributors.
    # This ensures the pre-clipped value is <-1.0, properly testing the clipping.
    base_appraisal.goal_relevance = 1.0
    base_appraisal.goal_congruence = -1.0
    base_appraisal.agency = 0.0  # Set to 0 to prevent positive boost
    base_appraisal.controllability = 0.0  # Set to 0 to prevent positive boost
    base_appraisal.social_approval = -1.0
    valence_low = compute_emotional_valence(base_appraisal)
    assert valence_low == -1.0


# Test Cases for compute_emotional_arousal


def test_compute_arousal_high_error(base_appraisal):
    """
    Tests that a high prediction error leads to high arousal.
    """
    arousal = compute_emotional_arousal(base_appraisal, prediction_error=10.0)
    assert arousal > 0.8


def test_compute_arousal_low_certainty_and_control(base_appraisal):
    """
    Tests that low certainty and controllability increase arousal.
    """
    base_appraisal.certainty = 0.1
    base_appraisal.controllability = 0.2
    arousal = compute_emotional_arousal(base_appraisal, prediction_error=1.0)
    # Base arousal from error is 1/5 = 0.2
    # Uncertainty boost = (1-0.1)*0.5 = 0.45
    # Control stress = (1-0.2)*0.3 = 0.24
    # Expected > 0.2 + 0.45 + 0.24
    assert arousal > 0.8


def test_compute_arousal_clipping(base_appraisal):
    """
    Tests that arousal is correctly clipped to the [0.0, 1.0] range.
    """
    # Arrange for very high potential arousal
    base_appraisal.goal_relevance = 1.0
    base_appraisal.certainty = 0.0
    base_appraisal.controllability = 0.0

    # Act
    arousal = compute_emotional_arousal(base_appraisal, prediction_error=100.0)

    # Assert
    assert arousal == 1.0
