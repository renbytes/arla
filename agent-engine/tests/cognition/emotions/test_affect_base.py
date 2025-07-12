# tests/cognition/emotions/test_affect_base.py

import numpy as np
import pytest

# Subject under test
from agent_engine.cognition.emotions.affect_base import (
    AffectiveExperience,
    init_affect_state,
    update_affect_state,
)

# --- Test Cases for AffectiveExperience ---


@pytest.fixture
def default_experience_params():
    """Provides a default set of parameters for creating an AffectiveExperience instance."""
    return {
        "valence": 0.5,
        "arousal": 0.6,
        "prediction_delta_magnitude": 0.2,
        "predictive_delta_smooth": 0.3,
        "health_norm": 0.8,
        "time_norm": 0.7,
        "res_norm": 0.9,
        "action_type_one_hot": np.array([0, 1, 0], dtype=np.float32),
        "outcome_reward": 5.0,
        "prediction_error": 1.0,
        "is_positive_outcome": True,
    }


def test_affective_experience_initialization(default_experience_params):
    """
    Tests that the AffectiveExperience class initializes all attributes correctly.
    """
    # Act
    exp = AffectiveExperience(**default_experience_params)

    # Assert
    assert exp.valence == 0.5
    assert exp.arousal == 0.6
    assert exp.prediction_delta_magnitude == 0.2
    assert exp.predictive_delta_smooth == 0.3
    assert exp.health_norm == 0.8
    assert exp.time_norm == 0.7
    assert exp.res_norm == 0.9
    np.testing.assert_array_equal(exp.action_type_one_hot, np.array([0, 1, 0]))
    assert exp.outcome_reward == 5.0
    assert exp.prediction_error == 1.0
    assert exp.is_positive_outcome is True


def test_affective_experience_vector_creation(default_experience_params):
    """
    Tests that the concatenated vector is created correctly and has the right shape and type.
    """
    # Act
    exp = AffectiveExperience(**default_experience_params)
    vector = exp.vector

    # Assert
    expected_vector = np.array(
        [0.5, 0.6, 0.2, 0.3, 0.8, 0.7, 0.9, 0, 1, 0, 5.0, 1.0, 1.0],
        dtype=np.float32,
    )
    assert vector.dtype == np.float32
    assert vector.shape == (13,)
    np.testing.assert_allclose(vector, expected_vector, atol=1e-6)


def test_affective_experience_to_dict(default_experience_params):
    """
    Tests that the to_dict method correctly serializes the object's data.
    """
    # Act
    exp = AffectiveExperience(**default_experience_params)
    exp_dict = exp.to_dict()

    # Assert
    assert exp_dict["valence"] == 0.5
    assert exp_dict["arousal"] == 0.6
    assert exp_dict["outcome_reward"] == 5.0
    assert exp_dict["is_positive_outcome"] is True
    # Check that the numpy array is converted to a list
    assert exp_dict["action_type_one_hot"] == [0.0, 1.0, 0.0]


# --- Test Cases for init_affect_state ---


def test_init_affect_state():
    """
    Tests that the initial affect state is created with the correct neutral values.
    """
    # Act
    initial_state = init_affect_state()

    # Assert
    expected_state = {
        "prediction_delta_magnitude": 0.0,
        "prev_reward": 0.0,
        "predictive_delta_smooth": 0.5,
    }
    assert initial_state == expected_state


# --- Test Cases for update_affect_state ---


@pytest.mark.parametrize(
    "prev_reward, prev_smooth, current_reward, expected_delta, expected_smooth, expected_error",
    [
        # Positive prediction error (pleasant surprise)
        (5.0, 0.5, 10.0, 5.0, 0.5 * 0.8 + 5.0 * 0.2, 5.0),
        # Negative prediction error (disappointment)
        (10.0, 0.5, 2.0, 8.0, 0.5 * 0.8 + 8.0 * 0.2, -8.0),
        # No change in reward
        (5.0, 0.5, 5.0, 0.0, 0.5 * 0.8 + 0.0 * 0.2, 0.0),
        # From negative to positive reward
        (-5.0, 0.8, 5.0, 10.0, 0.8 * 0.8 + 10.0 * 0.2, 10.0),
        # Clipping check for smoothed delta (should not exceed 1.0)
        (0.0, 0.9, 100.0, 100.0, 1.0, 100.0),
    ],
)
def test_update_affect_state(
    prev_reward,
    prev_smooth,
    current_reward,
    expected_delta,
    expected_smooth,
    expected_error,
):
    """
    Tests the update_affect_state function with various scenarios.
    """
    # Act
    (
        new_delta,
        new_prev_reward,
        new_smooth,
        prediction_error,
    ) = update_affect_state(prev_reward, prev_smooth, current_reward)

    # Assert
    assert new_delta == pytest.approx(expected_delta)
    assert new_prev_reward == pytest.approx(current_reward)
    # The calculation for expected_smooth can sometimes result in values slightly over 1.0
    # before clipping, so we test against the clipped expectation.
    assert new_smooth == pytest.approx(np.clip(expected_smooth, 0.0, 1.0))
    assert prediction_error == pytest.approx(expected_error)
