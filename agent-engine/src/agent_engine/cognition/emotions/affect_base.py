# cognition/emotions/affect_base.py

from typing import Any, Dict, Tuple

import numpy as np


class AffectiveExperience:
    """Represents a snapshot of an agent's affective and
    physiological state with action context."""

    def __init__(
        self,
        valence: float,
        arousal: float,
        prediction_delta_magnitude: float,
        predictive_delta_smooth: float,
        health_norm: float,
        time_norm: float,
        res_norm: float,
        action_type_one_hot: np.ndarray,
        outcome_reward: float,
        prediction_error: float,
        is_positive_outcome: bool,
    ):
        # Continuous affective states (renamed to be neutral)
        self.valence = valence
        self.arousal = arousal
        self.prediction_delta_magnitude = prediction_delta_magnitude
        # Represents a smoothed signal of predictive delta
        self.predictive_delta_smooth = predictive_delta_smooth

        # Normalized physiological states
        self.health_norm = health_norm
        self.time_norm = time_norm
        self.res_norm = res_norm

        # Contextual information about the action and its outcome
        self.action_type_one_hot = action_type_one_hot
        self.outcome_reward = outcome_reward
        self.prediction_error = prediction_error
        self.is_positive_outcome = is_positive_outcome

        # Combine into a single vector for clustering
        self.vector = np.concatenate(
            [
                np.array(
                    [
                        valence,
                        arousal,
                        prediction_delta_magnitude,
                        predictive_delta_smooth,
                    ]
                ),
                np.array([health_norm, time_norm, res_norm]),
                action_type_one_hot,
                np.array(
                    [
                        outcome_reward,
                        prediction_error,
                        1.0 if is_positive_outcome else 0.0,
                    ]
                ),
            ]
        ).astype(np.float32)

    # FIX: Added the return type annotation -> Dict[str, Any].
    # This resolves the [no-untyped-def] error.
    def to_dict(self) -> Dict[str, Any]:
        """Converts the AffectiveExperience to a dictionary for logging/transfer."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "prediction_delta_magnitude": self.prediction_delta_magnitude,
            "predictive_delta_smooth": self.predictive_delta_smooth,
            "health_norm": self.health_norm,
            "time_norm": self.time_norm,
            "res_norm": self.res_norm,
            "action_type_one_hot": self.action_type_one_hot.tolist(),
            "outcome_reward": self.outcome_reward,
            "prediction_error": self.prediction_error,
            "is_positive_outcome": self.is_positive_outcome,
        }


def init_affect_state() -> Dict[str, float]:
    """Initializes the agent's core affective state with
    neutral, mathematically descriptive terms."""
    # This function should only return attributes relevant to AffectComponent
    return {
        "prediction_delta_magnitude": 0.0,
        "prev_reward": 0.0,
        "predictive_delta_smooth": 0.5,
    }


def update_affect_state(
    prev_reward: float, prev_predictive_delta_smooth: float, current_reward: float
) -> Tuple[float, float, float, float]:
    """
    Updates the agent's core affect metrics
        (prediction error magnitude, prev_reward, smoothed predictive delta)
    based on the received reward.
    Returns (new_prediction_delta_magnitude, new_prev_reward,
            new_predictive_delta_smooth, prediction_error).
    """
    prediction_error = current_reward - prev_reward

    new_prediction_delta_magnitude = abs(prediction_error)

    # Update predictive_delta_smooth using the current and previous smoothed value
    new_predictive_delta_smooth = 0.8 * prev_predictive_delta_smooth + 0.2 * abs(prediction_error)
    new_predictive_delta_smooth = np.clip(new_predictive_delta_smooth, 0.0, 1.0)  # Keep within [0, 1]

    return (
        new_prediction_delta_magnitude,
        current_reward,
        new_predictive_delta_smooth,
        prediction_error,
    )
