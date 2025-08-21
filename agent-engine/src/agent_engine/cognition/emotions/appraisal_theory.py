# agent-engine/src/agent_engine/cognition/emotions/appraisal_theory.py

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AppraisalDimensions:
    """Core appraisal dimensions from psychological literature"""

    goal_relevance: float
    goal_congruence: float
    agency: float
    controllability: float
    certainty: float
    social_approval: float


class AppraisalProcessor:
    """Implements Lazarus & Folkman appraisal theory for emotional evaluation"""

    def __init__(self, config: Any):
        # Use direct attribute access on the validated Pydantic model
        weights = config.agent.emotional_dynamics.appraisal_weights
        self.goal_weight = weights.goal_relevance
        self.agency_weight = weights.agency
        self.social_weight = weights.social_feedback

    def appraise_event(
        self,
        prediction_error: float,
        current_goal: Optional[str],
        action_success: bool,
        social_context: Dict[str, Any],
        controllability_estimate: float = 0.5,
    ) -> AppraisalDimensions:
        """
        Perform cognitive appraisal of an event to determine emotional response
        """
        goal_relevance = self._assess_goal_relevance(
            prediction_error, current_goal, action_success
        )
        goal_congruence = self._assess_goal_congruence(prediction_error, action_success)
        agency = self._assess_agency(action_success, controllability_estimate)
        controllability = self._assess_controllability(prediction_error, social_context)
        certainty = self._assess_certainty(prediction_error)
        social_approval = self._assess_social_approval(social_context)

        return AppraisalDimensions(
            goal_relevance=goal_relevance,
            goal_congruence=goal_congruence,
            agency=agency,
            controllability=controllability,
            certainty=certainty,
            social_approval=social_approval,
        )

    def _assess_goal_relevance(
        self, prediction_error: float, current_goal: Optional[str], action_success: bool
    ) -> float:
        """Assess how relevant this event is to current goals"""
        if current_goal is None:
            return 0.3

        base_relevance = min(abs(prediction_error) / 10.0, 1.0)
        if action_success and prediction_error > 0:
            base_relevance *= 1.2
        elif not action_success and prediction_error < 0:
            base_relevance *= 1.1

        return float(np.clip(base_relevance, 0.0, 1.0))

    def _assess_goal_congruence(
        self, prediction_error: float, action_success: bool
    ) -> float:
        """Assess whether event helps (+1) or hinders (-1) goal achievement"""
        if action_success and prediction_error > 0:
            return min(prediction_error / 5.0, 1.0)
        elif not action_success and prediction_error < 0:
            return max(prediction_error / 5.0, -1.0)
        else:
            return 0.0

    def _assess_agency(
        self, action_success: bool, controllability_estimate: float
    ) -> float:
        """Assess personal agency in the outcome"""
        base_agency = 0.7 if action_success else 0.3
        return base_agency * controllability_estimate

    def _assess_controllability(
        self, prediction_error: float, social_context: Dict[str, Any]
    ) -> float:
        """Assess future controllability of similar situations"""
        error_factor = max(0.2, 1.0 - abs(prediction_error) / 10.0)
        social_factor = (
            0.8 if social_context.get("other_agents_present", False) else 1.0
        )
        return error_factor * social_factor

    def _assess_certainty(self, prediction_error: float) -> float:
        """Assess predictability of the outcome"""
        return max(0.1, 1.0 - abs(prediction_error) / 5.0)

    def _assess_social_approval(self, social_context: Dict[str, Any]) -> float:
        """Assess likely social approval/disapproval"""
        if not social_context.get("other_agents_present", False):
            return 0.0
        if social_context.get("action_intent") == "COOPERATE":
            return 0.6
        elif social_context.get("action_intent") == "COMPETE":
            return -0.3
        else:
            return 0.0


def compute_emotional_valence(appraisal: AppraisalDimensions) -> float:
    """Compute valence based on appraisal dimensions"""
    primary_valence = appraisal.goal_congruence * appraisal.goal_relevance
    agency_boost = appraisal.agency * 0.3
    control_boost = appraisal.controllability * 0.2
    social_boost = appraisal.social_approval * 0.4
    total_valence = primary_valence + agency_boost + control_boost + social_boost
    return float(np.clip(total_valence, -1.0, 1.0))


def compute_emotional_arousal(
    appraisal: AppraisalDimensions, prediction_error: float
) -> float:
    """Compute arousal based on appraisal dimensions and prediction error magnitude"""
    error_arousal = min(abs(prediction_error) / 5.0, 1.0)
    relevance_multiplier = 1.0 + appraisal.goal_relevance
    uncertainty_boost = (1.0 - appraisal.certainty) * 0.5
    control_stress = (1.0 - appraisal.controllability) * 0.3
    total_arousal = (
        error_arousal * relevance_multiplier + uncertainty_boost + control_stress
    )
    return float(np.clip(total_arousal, 0.0, 1.0))
