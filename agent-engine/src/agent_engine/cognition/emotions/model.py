# agent-engine/src/agent_engine/cognition/emotions/model.py


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

    def __init__(self, config: Any):
        # Use direct attribute access on the validated Pydantic model
        temporal_config = config.agent.emotional_dynamics.temporal
        self.valence_decay = temporal_config["valence_decay_rate"]
        self.arousal_decay = temporal_config["arousal_decay_rate"]
        self.valence_learning_rate = temporal_config["valence_learning_rate"]
        self.arousal_learning_rate = temporal_config["arousal_learning_rate"]
        self.emotional_noise = config.agent.emotional_dynamics.noise_std

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
