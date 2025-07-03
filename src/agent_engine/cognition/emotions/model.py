# emotion/model.py


from typing import Any, Dict, Optional

import numpy as np

from .appraisal_theory import (
    AppraisalDimensions,
    AppraisalProcessor,
    compute_emotional_arousal,
    compute_emotional_valence,
)


class EmotionalDynamics:
    """Psychologically-grounded emotional state updates with proper temporal dynamics"""

    def __init__(self, config: Dict[str, Any]):
        self.valence_decay = config.get("valence_decay_rate", 0.95)
        self.arousal_decay = config.get("arousal_decay_rate", 0.90)
        self.valence_learning_rate = config.get("valence_learning_rate", 0.2)
        self.arousal_learning_rate = config.get("arousal_learning_rate", 0.3)
        self.emotional_noise = config.get("emotional_noise_std", 0.02)

        self.appraisal_processor = AppraisalProcessor(config)

    def update_emotion_with_appraisal(
        self,
        current_emotion: Dict[str, float],
        prediction_error: float,
        current_goal: Optional[str],
        action_success: bool,
        social_context: Dict[str, Any],
        controllability_estimate: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Update emotional state using appraisal theory and proper temporal dynamics

        Implements: v_{t+1} = α_v * v_t + β_v * appraise(δ_t, goal_t, agency_t) + ε_v
                      a_{t+1} = α_a * a_t + β_a * |δ_t| * importance(context_t)
        """

        # Perform cognitive appraisal
        appraisal: AppraisalDimensions = self.appraisal_processor.appraise_event(
            prediction_error=prediction_error,
            current_goal=current_goal,
            action_success=action_success,
            social_context=social_context,
            controllability_estimate=controllability_estimate,
        )

        # Compute target emotional states from appraisal
        target_valence = compute_emotional_valence(appraisal)
        target_arousal = compute_emotional_arousal(appraisal, prediction_error)

        # Apply temporal dynamics with decay
        current_valence = current_emotion.get("valence", 0.0)
        current_arousal = current_emotion.get("arousal", 0.5)

        # Valence update with decay and learning
        new_valence = (
            self.valence_decay * current_valence
            + self.valence_learning_rate * target_valence
            + np.random.normal(0, self.emotional_noise)
        )

        # Arousal update with decay and learning
        new_arousal = (
            self.arousal_decay * current_arousal
            + self.arousal_learning_rate * target_arousal
            + np.random.normal(0, self.emotional_noise)
        )

        # Apply bounds
        new_valence = np.clip(new_valence, -1.0, 1.0)
        new_arousal = np.clip(new_arousal, 0.0, 1.0)

        updated_emotion: Dict[str, Any] = {
            "valence": new_valence,
            "arousal": new_arousal,
            "appraisal_dimensions": appraisal,
            "target_valence": target_valence,
            "target_arousal": target_arousal,
        }
        return updated_emotion


# Backward compatibility function
def update_emotion(emotion: Dict[str, float], prediction_error: float) -> Dict[str, Any]:
    """Legacy function for backward compatibility - use EmotionalDynamics instead"""
    dynamics = EmotionalDynamics({})
    return dynamics.update_emotion_with_appraisal(
        current_emotion=emotion,
        prediction_error=prediction_error,
        current_goal=None,
        action_success=prediction_error > 0,
        social_context={},
    )
