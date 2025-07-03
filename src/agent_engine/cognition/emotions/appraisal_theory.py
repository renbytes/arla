# emotion/appraisal_theory.py

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AppraisalDimensions:
    """Core appraisal dimensions from psychological literature"""

    goal_relevance: float  # How relevant is this event to current goals?
    goal_congruence: float  # Does this help or hinder goal achievement?
    agency: float  # How much control do I have over this?
    controllability: float  # Can this be changed/influenced?
    certainty: float  # How predictable/uncertain is this?
    social_approval: float  # What do others think of this?


class AppraisalProcessor:
    """Implements Lazarus & Folkman appraisal theory for emotional evaluation"""

    def __init__(self, config: Dict[str, Any]):
        self.goal_weight = config.get("appraisal_goal_weight", 0.7)
        self.agency_weight = config.get("appraisal_agency_weight", 0.5)
        self.social_weight = config.get("appraisal_social_weight", 0.3)

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

        Based on Lazarus & Folkman (1984) and Scherer (2001) appraisal theories
        """

        # Goal relevance: Is this event important to my current objectives?
        goal_relevance = self._assess_goal_relevance(prediction_error, current_goal, action_success)

        # Goal congruence: Does this help or hinder my goals?
        goal_congruence = self._assess_goal_congruence(prediction_error, action_success)

        # Agency: How much control did I have over this outcome?
        agency = self._assess_agency(action_success, controllability_estimate)

        # Controllability: Can I influence future similar events?
        controllability = self._assess_controllability(prediction_error, social_context)

        # Certainty: How predictable was this outcome?
        certainty = self._assess_certainty(prediction_error)

        # Social approval: What do others think of this action/outcome?
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
            return 0.3  # Moderate relevance when no clear goal

        # Higher relevance for larger prediction errors and goal-related actions
        base_relevance = min(abs(prediction_error) / 10.0, 1.0)

        # Boost relevance if action was explicitly goal-directed
        if action_success and prediction_error > 0:
            base_relevance *= 1.2
        elif not action_success and prediction_error < 0:
            base_relevance *= 1.1

        # FIX: Cast the numpy float to a standard float to match the return type annotation.
        return float(np.clip(base_relevance, 0.0, 1.0))

    def _assess_goal_congruence(self, prediction_error: float, action_success: bool) -> float:
        """Assess whether event helps (+1) or hinders (-1) goal achievement"""
        if action_success and prediction_error > 0:
            return min(prediction_error / 5.0, 1.0)  # Positive congruence
        elif not action_success and prediction_error < 0:
            return max(prediction_error / 5.0, -1.0)  # Negative congruence
        else:
            return 0.0  # Mixed or neutral

    def _assess_agency(self, action_success: bool, controllability_estimate: float) -> float:
        """Assess personal agency in the outcome"""
        base_agency = 0.7 if action_success else 0.3  # Success implies agency
        return base_agency * controllability_estimate

    def _assess_controllability(self, prediction_error: float, social_context: Dict[str, Any]) -> float:
        """Assess future controllability of similar situations"""
        # Large prediction errors suggest low controllability
        error_factor = max(0.2, 1.0 - abs(prediction_error) / 10.0)

        # Social situations are often less controllable
        social_factor = 0.8 if social_context.get("other_agents_present", False) else 1.0

        return error_factor * social_factor

    def _assess_certainty(self, prediction_error: float) -> float:
        """Assess predictability of the outcome"""
        # Larger prediction errors indicate lower certainty
        return max(0.1, 1.0 - abs(prediction_error) / 5.0)

    def _assess_social_approval(self, social_context: Dict[str, Any]) -> float:
        """Assess likely social approval/disapproval"""
        if not social_context.get("other_agents_present", False):
            return 0.0  # No social context

        # Cooperative actions generally receive approval
        if social_context.get("action_intent") == "COOPERATE":
            return 0.6
        elif social_context.get("action_intent") == "COMPETE":
            return -0.3
        else:
            return 0.0


def compute_emotional_valence(appraisal: AppraisalDimensions) -> float:
    """
    Compute valence based on appraisal dimensions
    Based on Scherer's Component Process Model
    """
    # Goal congruence is primary driver of valence
    primary_valence = appraisal.goal_congruence * appraisal.goal_relevance

    # Agency and controllability modulate valence
    agency_boost = appraisal.agency * 0.3  # Feeling in control is positive
    control_boost = appraisal.controllability * 0.2

    # Social approval contributes to valence
    social_boost = appraisal.social_approval * 0.4

    total_valence = primary_valence + agency_boost + control_boost + social_boost
    # FIX: Cast the numpy float to a standard float.
    return float(np.clip(total_valence, -1.0, 1.0))


def compute_emotional_arousal(appraisal: AppraisalDimensions, prediction_error: float) -> float:
    """
    Compute arousal based on appraisal dimensions and prediction error magnitude
    """
    # Base arousal from prediction error magnitude
    error_arousal = min(abs(prediction_error) / 5.0, 1.0)

    # Goal relevance amplifies arousal
    relevance_multiplier = 1.0 + appraisal.goal_relevance

    # Uncertainty increases arousal
    uncertainty_boost = (1.0 - appraisal.certainty) * 0.5

    # Low controllability increases arousal (stress response)
    control_stress = (1.0 - appraisal.controllability) * 0.3

    total_arousal = error_arousal * relevance_multiplier + uncertainty_boost + control_stress
    # FIX: Cast the numpy float to a standard float.
    return float(np.clip(total_arousal, 0.0, 1.0))
